#!/usr/bin/env python3
"""
Training script for Transformer-based language models on Australian tourism corpus.

This script provides comprehensive training for Transformer models with 
multi-head attention, device-aware optimization, and TensorBoard logging.
"""

import os
import sys
import argparse
import yaml
import time
import pickle
from pathlib import Path
from typing import Dict, Any, Tuple, List

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from tqdm import tqdm

# Add the parent directory to the path so we can import our modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import (
    detect_device,
    get_run_logdir,
    save_model_checkpoint,
    count_parameters,
    calculate_perplexity,
    generate_sample_text,
    AustralianTextSampler,
    create_transformer_language_model
)
from data.process_data import Vocabulary, LanguageModelDataset, collate_fn


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def load_data_and_vocab(data_dir: str, language: str = 'en') -> Tuple[DataLoader, DataLoader, DataLoader, Vocabulary]:
    """
    Load processed datasets and vocabulary.
    
    Args:
        data_dir: Directory containing processed data
        language: Language to load ('en' or 'vi')
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader, vocabulary)
    """
    # Load vocabulary
    vocab_path = os.path.join(data_dir, f'{language}_vocabulary.pkl')
    vocabulary = Vocabulary()
    vocabulary.load(vocab_path)
    
    # Load datasets
    with open(os.path.join(data_dir, f'{language}_train_dataset.pkl'), 'rb') as f:
        train_dataset = pickle.load(f)
    with open(os.path.join(data_dir, f'{language}_val_dataset.pkl'), 'rb') as f:
        val_dataset = pickle.load(f)
    with open(os.path.join(data_dir, f'{language}_test_dataset.pkl'), 'rb') as f:
        test_dataset = pickle.load(f)
    
    # Create data loaders with appropriate batch size for Transformer
    batch_size = 16  # Smaller batch size for Transformers due to memory requirements
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    
    print(f"✅ Loaded {language} data:")
    print(f"   Train batches: {len(train_loader)}")
    print(f"   Val batches: {len(val_loader)}")
    print(f"   Test batches: {len(test_loader)}")
    print(f"   Vocabulary size: {len(vocabulary)}")
    print(f"   Batch size: {batch_size} (optimized for Transformer)")
    
    return train_loader, val_loader, test_loader, vocabulary


def create_learning_rate_scheduler(optimizer: optim.Optimizer, warmup_steps: int, d_model: int):
    """
    Create learning rate scheduler with warmup for Transformer training.
    
    Args:
        optimizer: PyTorch optimizer
        warmup_steps: Number of warmup steps
        d_model: Model dimension
        
    Returns:
        Learning rate scheduler function
    """
    def lr_lambda(step):
        if step == 0:
            return 0
        return min(step ** -0.5, step * (warmup_steps ** -1.5)) * (d_model ** -0.5)
    
    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def train_epoch(model: nn.Module, train_loader: DataLoader, optimizer: optim.Optimizer,
                criterion: nn.Module, device: torch.device, clip_grad_norm: float = 1.0) -> Tuple[float, float]:
    """
    Train Transformer model for one epoch.
    
    Args:
        model: Transformer language model
        train_loader: Training data loader
        optimizer: Optimizer
        criterion: Loss function
        device: Device for computation
        clip_grad_norm: Gradient clipping norm
        
    Returns:
        Tuple of (average_loss, perplexity)
    """
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    progress_bar = tqdm(train_loader, desc="Training", leave=False)
    
    for batch in progress_bar:
        input_ids = batch['input_ids'].to(device)
        targets = batch['targets'].to(device)
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass
        logits = model(input_ids)
        
        # Calculate loss (ignore padding tokens)
        loss = criterion(logits.view(-1, logits.size(-1)), targets.view(-1))
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping (important for Transformer stability)
        if clip_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)
        
        # Update parameters
        optimizer.step()
        
        # Accumulate loss
        total_loss += loss.item()
        num_batches += 1
        
        # Update progress bar
        progress_bar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'ppl': f'{calculate_perplexity(loss.item()):.1f}'
        })
    
    avg_loss = total_loss / num_batches
    perplexity = calculate_perplexity(avg_loss)
    
    return avg_loss, perplexity


def evaluate_model(model: nn.Module, data_loader: DataLoader, criterion: nn.Module, 
                   device: torch.device) -> Tuple[float, float]:
    """
    Evaluate Transformer model on validation or test data.
    
    Args:
        model: Transformer language model
        data_loader: Data loader for evaluation
        criterion: Loss function
        device: Device for computation
        
    Returns:
        Tuple of (average_loss, perplexity)
    """
    model.eval()
    total_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Evaluating", leave=False):
            input_ids = batch['input_ids'].to(device)
            targets = batch['targets'].to(device)
            
            # Forward pass
            logits = model(input_ids)
            
            # Calculate loss
            loss = criterion(logits.view(-1, logits.size(-1)), targets.view(-1))
            
            total_loss += loss.item()
            num_batches += 1
    
    avg_loss = total_loss / num_batches
    perplexity = calculate_perplexity(avg_loss)
    
    return avg_loss, perplexity


def generate_text_samples_transformer(model: nn.Module, vocabulary: Vocabulary, sampler: AustralianTextSampler,
                                    device: torch.device, language: str = 'en', num_samples: int = 3) -> List[str]:
    """
    Generate text samples using Transformer model.
    
    Args:
        model: Trained Transformer model
        vocabulary: Vocabulary object
        sampler: Text sampler for prompts
        device: Device for computation
        language: Language for generation
        num_samples: Number of samples to generate
        
    Returns:
        List of generated text samples
    """
    model.eval()
    samples = []
    
    with torch.no_grad():
        for _ in range(num_samples):
            prompt = sampler.get_random_prompt(language)
            try:
                # Use Transformer-specific generation
                input_ids = vocabulary.encode(prompt, language, add_special_tokens=True)
                input_tensor = torch.tensor([input_ids], dtype=torch.long).to(device)
                
                # Generate using model's generate method
                generated_tensor = model.generate(
                    input_tensor, 
                    max_length=25,
                    temperature=0.8,
                    top_k=50,
                    top_p=0.9,
                    device=device
                )
                
                # Decode generated text
                generated_text = vocabulary.decode(generated_tensor[0].tolist(), remove_special_tokens=True)
                samples.append(f"Prompt: '{prompt}' → '{generated_text}'")
                
            except Exception as e:
                samples.append(f"Generation failed for prompt '{prompt}': {str(e)}")
    
    return samples


def train_transformer_model(config: Dict[str, Any], language: str = 'en', model_type: str = 'transformer'):
    """
    Main training function for Transformer language models.
    
    Args:
        config: Configuration dictionary
        language: Language to train on ('en' or 'vi')
        model_type: Type of Transformer model ('transformer', 'gpt')
    """
    print("🤖 Training Transformer Language Model for Australian Tourism")
    print("=" * 65)
    print(f"Language: {language.upper()}")
    print(f"Model type: {model_type.upper()}")
    
    # Detect device
    device, device_info = detect_device()
    print(f"Device: {device} ({device_info})")
    
    # Load data and vocabulary
    data_dir = 'processed_data'  # Assuming data has been processed
    if not os.path.exists(data_dir):
        print("❌ Processed data not found. Please run data/process_data.py first.")
        return
    
    train_loader, val_loader, test_loader, vocabulary = load_data_and_vocab(data_dir, language)
    
    # Create model
    transformer_config = config['transformer_model']
    model = create_transformer_language_model(
        model_type,
        vocab_size=len(vocabulary),
        embed_dim=transformer_config['embed_dim'],
        num_heads=transformer_config['num_heads'],
        num_layers=transformer_config['num_layers'],
        ff_dim=transformer_config['ff_dim'],
        max_seq_len=transformer_config['max_seq_len'],
        dropout=transformer_config['dropout']
    ).to(device)
    
    # Print model information
    model_info = model.get_model_info()
    print(f"\n📊 Model Information:")
    for key, value in model_info.items():
        print(f"   {key}: {value}")
    
    # Setup training
    training_config = config['training']
    
    # Use label smoothing for better Transformer training
    criterion = nn.CrossEntropyLoss(
        ignore_index=vocabulary.word2idx.get(vocabulary.PAD_TOKEN, 0),
        label_smoothing=0.1
    )
    
    # Use Adam optimizer with specific Transformer settings
    optimizer = optim.Adam(
        model.parameters(), 
        lr=training_config['learning_rate'],
        betas=(0.9, 0.98),  # Transformer-specific beta values
        eps=1e-9,
        weight_decay=training_config['weight_decay']
    )
    
    # Learning rate scheduler with warmup
    warmup_steps = training_config.get('warmup_steps', 4000)
    scheduler = create_learning_rate_scheduler(optimizer, warmup_steps, transformer_config['embed_dim'])
    
    # TensorBoard logging
    log_dir = get_run_logdir(f"{model_type}_language_model_{language}")
    writer = SummaryWriter(log_dir)
    
    # Text sampler for generation monitoring
    text_sampler = AustralianTextSampler()
    
    print(f"\n🚀 Starting training...")
    print(f"TensorBoard logs: {log_dir}")
    print(f"Warmup steps: {warmup_steps}")
    
    best_val_loss = float('inf')
    patience_counter = 0
    step = 0
    
    # Training loop
    for epoch in range(training_config['epochs']):
        epoch_start_time = time.time()
        
        # Training
        model.train()
        epoch_loss = 0.0
        num_batches = 0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}", leave=False)
        
        for batch in progress_bar:
            input_ids = batch['input_ids'].to(device)
            targets = batch['targets'].to(device)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            logits = model(input_ids)
            
            # Calculate loss
            loss = criterion(logits.view(-1, logits.size(-1)), targets.view(-1))
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), training_config['clip_grad_norm'])
            
            # Update parameters and learning rate
            optimizer.step()
            scheduler.step()
            
            # Accumulate loss
            epoch_loss += loss.item()
            num_batches += 1
            step += 1
            
            # Update progress bar
            current_lr = optimizer.param_groups[0]['lr']
            progress_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'lr': f'{current_lr:.2e}'
            })
            
            # Log batch-level metrics
            if step % 100 == 0:
                writer.add_scalar('Loss/Train_Step', loss.item(), step)
                writer.add_scalar('Learning_Rate_Step', current_lr, step)
        
        train_loss = epoch_loss / num_batches
        train_perplexity = calculate_perplexity(train_loss)
        
        # Validation
        val_loss, val_perplexity = evaluate_model(model, val_loader, criterion, device)
        
        epoch_time = time.time() - epoch_start_time
        
        # Logging
        print(f"\nEpoch {epoch + 1}/{training_config['epochs']} ({epoch_time:.1f}s)")
        print(f"  Train Loss: {train_loss:.4f}, Perplexity: {train_perplexity:.2f}")
        print(f"  Val Loss: {val_loss:.4f}, Perplexity: {val_perplexity:.2f}")
        print(f"  Learning Rate: {optimizer.param_groups[0]['lr']:.2e}")
        
        # TensorBoard logging
        writer.add_scalar('Loss/Train_Epoch', train_loss, epoch)
        writer.add_scalar('Loss/Validation', val_loss, epoch)
        writer.add_scalar('Perplexity/Train', train_perplexity, epoch)
        writer.add_scalar('Perplexity/Validation', val_perplexity, epoch)
        
        # Generate text samples for monitoring
        if epoch % 3 == 0:
            text_samples = generate_text_samples_transformer(model, vocabulary, text_sampler, device, language)
            print(f"\n📝 Generated samples:")
            for sample in text_samples:
                print(f"    {sample}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            
            # Save checkpoint
            checkpoint_path = os.path.join(log_dir, f'best_model_{model_type}_{language}.pt')
            save_model_checkpoint(
                model, optimizer, epoch, val_loss, checkpoint_path,
                len(vocabulary), config
            )
            print(f"  💾 Best model saved (val_loss: {val_loss:.4f})")
        else:
            patience_counter += 1
        
        # Early stopping (more patient for Transformer)
        if patience_counter >= 8:
            print(f"\n⏰ Early stopping triggered (patience: 8)")
            break
    
    # Final evaluation on test set
    print(f"\n🎯 Final evaluation on test set...")
    test_loss, test_perplexity = evaluate_model(model, test_loader, criterion, device)
    print(f"Test Loss: {test_loss:.4f}, Perplexity: {test_perplexity:.2f}")
    
    # Log final results
    writer.add_scalar('Final/Test_Loss', test_loss, 0)
    writer.add_scalar('Final/Test_Perplexity', test_perplexity, 0)
    
    # Generate final samples with different settings
    print(f"\n🎨 Final generation samples:")
    print("=" * 50)
    
    model.eval()
    with torch.no_grad():
        prompts = text_sampler.get_all_prompts(language)[:3]
        
        for temp in [0.7, 1.0, 1.3]:
            print(f"\nTemperature {temp}:")
            for prompt in prompts:
                try:
                    input_ids = vocabulary.encode(prompt, language, add_special_tokens=True)
                    input_tensor = torch.tensor([input_ids], dtype=torch.long).to(device)
                    
                    generated_tensor = model.generate(
                        input_tensor, 
                        max_length=30,
                        temperature=temp,
                        top_k=50,
                        top_p=0.9,
                        device=device
                    )
                    
                    generated_text = vocabulary.decode(generated_tensor[0].tolist(), remove_special_tokens=True)
                    print(f"  '{prompt}' → '{generated_text}'")
                    
                except Exception as e:
                    print(f"  '{prompt}' → Generation failed: {e}")
    
    writer.close()
    print(f"\n✅ Training completed! Best validation loss: {best_val_loss:.4f}")
    print(f"📊 TensorBoard logs: {log_dir}")


def main():
    """Main function for Transformer training script."""
    parser = argparse.ArgumentParser(description='Train Transformer language models')
    parser.add_argument('--config', type=str, default='config.yaml', help='Configuration file')
    parser.add_argument('--language', type=str, choices=['en', 'vi'], default='en',
                       help='Language to train on')
    parser.add_argument('--model', type=str, choices=['transformer', 'gpt'],
                       default='transformer', help='Transformer model type')
    parser.add_argument('--data_dir', type=str, default='processed_data',
                       help='Directory with processed data')
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Start training
    train_transformer_model(config, args.language, args.model)


if __name__ == '__main__':
    main()