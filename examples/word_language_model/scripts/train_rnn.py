#!/usr/bin/env python3
"""
Training script for RNN-based language models on Australian tourism corpus.

This script provides comprehensive training for LSTM, GRU, and BiLSTM models
with device-aware optimization and TensorBoard logging.
"""

import os
import sys
import argparse
import yaml
import time
import pickle
from pathlib import Path
from typing import Dict, Any, Tuple

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
    create_rnn_language_model
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
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)
    
    print(f"✅ Loaded {language} data:")
    print(f"   Train batches: {len(train_loader)}")
    print(f"   Val batches: {len(val_loader)}")
    print(f"   Test batches: {len(test_loader)}")
    print(f"   Vocabulary size: {len(vocabulary)}")
    
    return train_loader, val_loader, test_loader, vocabulary


def train_epoch(model: nn.Module, train_loader: DataLoader, optimizer: optim.Optimizer,
                criterion: nn.Module, device: torch.device, clip_grad_norm: float = 1.0) -> Tuple[float, float]:
    """
    Train model for one epoch.
    
    Args:
        model: Language model
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
        if hasattr(model, 'init_hidden'):
            # For RNN models
            hidden = model.init_hidden(input_ids.size(0), device)
            logits, _ = model(input_ids, hidden)
        else:
            # For other models
            logits = model(input_ids)
        
        # Calculate loss
        loss = criterion(logits.view(-1, logits.size(-1)), targets.view(-1))
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping
        if clip_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)
        
        # Update parameters
        optimizer.step()
        
        # Accumulate loss
        total_loss += loss.item()
        num_batches += 1
        
        # Update progress bar
        progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    avg_loss = total_loss / num_batches
    perplexity = calculate_perplexity(avg_loss)
    
    return avg_loss, perplexity


def evaluate_model(model: nn.Module, data_loader: DataLoader, criterion: nn.Module, 
                   device: torch.device) -> Tuple[float, float]:
    """
    Evaluate model on validation or test data.
    
    Args:
        model: Language model
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
            if hasattr(model, 'init_hidden'):
                # For RNN models
                hidden = model.init_hidden(input_ids.size(0), device)
                logits, _ = model(input_ids, hidden)
            else:
                # For other models
                logits = model(input_ids)
            
            # Calculate loss
            loss = criterion(logits.view(-1, logits.size(-1)), targets.view(-1))
            
            total_loss += loss.item()
            num_batches += 1
    
    avg_loss = total_loss / num_batches
    perplexity = calculate_perplexity(avg_loss)
    
    return avg_loss, perplexity


def generate_text_samples(model: nn.Module, vocabulary: Vocabulary, sampler: AustralianTextSampler,
                         device: torch.device, language: str = 'en', num_samples: int = 3) -> List[str]:
    """
    Generate text samples for monitoring during training.
    
    Args:
        model: Trained model
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
                generated_text = generate_sample_text(
                    model, vocabulary, prompt, 
                    max_length=30, temperature=0.8, 
                    device=device, language=language
                )
                samples.append(f"Prompt: '{prompt}' → '{generated_text}'")
            except Exception as e:
                samples.append(f"Generation failed for prompt '{prompt}': {str(e)}")
    
    return samples


def train_rnn_model(config: Dict[str, Any], language: str = 'en', model_type: str = 'lstm'):
    """
    Main training function for RNN language models.
    
    Args:
        config: Configuration dictionary
        language: Language to train on ('en' or 'vi')
        model_type: Type of RNN model ('lstm', 'gru', 'bilstm')
    """
    print("🧠 Training RNN Language Model for Australian Tourism")
    print("=" * 60)
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
    rnn_config = config['rnn_model']
    model = create_rnn_language_model(
        model_type,
        vocab_size=len(vocabulary),
        embed_dim=rnn_config['embed_dim'],
        hidden_dim=rnn_config['hidden_dim'],
        num_layers=rnn_config['num_layers'],
        dropout=rnn_config['dropout'],
        tie_weights=rnn_config['tie_weights']
    ).to(device)
    
    # Print model information
    model_info = model.get_model_info()
    print(f"\n📊 Model Information:")
    for key, value in model_info.items():
        print(f"   {key}: {value}")
    
    # Setup training
    training_config = config['training']
    criterion = nn.CrossEntropyLoss(ignore_index=vocabulary.word2idx.get(vocabulary.PAD_TOKEN, 0))
    optimizer = optim.Adam(model.parameters(), lr=training_config['learning_rate'], 
                          weight_decay=training_config['weight_decay'])
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)
    
    # TensorBoard logging
    log_dir = get_run_logdir(f"{model_type}_language_model_{language}")
    writer = SummaryWriter(log_dir)
    
    # Text sampler for generation monitoring
    text_sampler = AustralianTextSampler()
    
    print(f"\n🚀 Starting training...")
    print(f"TensorBoard logs: {log_dir}")
    
    best_val_loss = float('inf')
    patience_counter = 0
    
    # Training loop
    for epoch in range(training_config['epochs']):
        epoch_start_time = time.time()
        
        # Training
        train_loss, train_perplexity = train_epoch(
            model, train_loader, optimizer, criterion, device,
            training_config['clip_grad_norm']
        )
        
        # Validation
        val_loss, val_perplexity = evaluate_model(model, val_loader, criterion, device)
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        
        epoch_time = time.time() - epoch_start_time
        
        # Logging
        print(f"\nEpoch {epoch + 1}/{training_config['epochs']} ({epoch_time:.1f}s)")
        print(f"  Train Loss: {train_loss:.4f}, Perplexity: {train_perplexity:.2f}")
        print(f"  Val Loss: {val_loss:.4f}, Perplexity: {val_perplexity:.2f}")
        print(f"  Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")
        
        # TensorBoard logging
        writer.add_scalar('Loss/Train', train_loss, epoch)
        writer.add_scalar('Loss/Validation', val_loss, epoch)
        writer.add_scalar('Perplexity/Train', train_perplexity, epoch)
        writer.add_scalar('Perplexity/Validation', val_perplexity, epoch)
        writer.add_scalar('Learning_Rate', optimizer.param_groups[0]['lr'], epoch)
        
        # Generate text samples for monitoring
        if epoch % 5 == 0:
            text_samples = generate_text_samples(model, vocabulary, text_sampler, device, language)
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
        
        # Early stopping
        if patience_counter >= 10:
            print(f"\n⏰ Early stopping triggered (patience: 10)")
            break
    
    # Final evaluation on test set
    print(f"\n🎯 Final evaluation on test set...")
    test_loss, test_perplexity = evaluate_model(model, test_loader, criterion, device)
    print(f"Test Loss: {test_loss:.4f}, Perplexity: {test_perplexity:.2f}")
    
    # Log final results
    writer.add_scalar('Final/Test_Loss', test_loss, 0)
    writer.add_scalar('Final/Test_Perplexity', test_perplexity, 0)
    
    # Generate final samples
    print(f"\n🎨 Final generation samples:")
    final_samples = generate_text_samples(model, vocabulary, text_sampler, device, language, 5)
    for sample in final_samples:
        print(f"  {sample}")
    
    writer.close()
    print(f"\n✅ Training completed! Best validation loss: {best_val_loss:.4f}")
    print(f"📊 TensorBoard logs: {log_dir}")


def main():
    """Main function for RNN training script."""
    parser = argparse.ArgumentParser(description='Train RNN language models')
    parser.add_argument('--config', type=str, default='config.yaml', help='Configuration file')
    parser.add_argument('--language', type=str, choices=['en', 'vi'], default='en',
                       help='Language to train on')
    parser.add_argument('--model', type=str, choices=['lstm', 'gru', 'bilstm', 'attention_lstm'],
                       default='lstm', help='RNN model type')
    parser.add_argument('--data_dir', type=str, default='processed_data',
                       help='Directory with processed data')
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Start training
    train_rnn_model(config, args.language, args.model)


if __name__ == '__main__':
    main()