#!/usr/bin/env python3
"""
Text generation utility for trained language models.

This script provides interactive text generation using trained RNN or 
Transformer models with Australian tourism context.
"""

import os
import sys
import argparse
import pickle
import torch
from typing import List, Optional

# Add the parent directory to the path so we can import our modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import (
    detect_device,
    create_rnn_language_model,
    create_transformer_language_model,
    AustralianTextSampler,
    load_model_checkpoint
)
from data.process_data import Vocabulary, LanguageModelDataset


def load_trained_model(checkpoint_path: str, model_type: str, vocab_size: int, device: torch.device):
    """
    Load a trained model from checkpoint.
    
    Args:
        checkpoint_path: Path to model checkpoint
        model_type: Type of model ('lstm', 'gru', 'transformer', etc.)
        vocab_size: Vocabulary size
        device: Device to load model on
        
    Returns:
        Loaded model
    """
    # Create model architecture
    if model_type in ['lstm', 'gru', 'bilstm', 'attention_lstm']:
        model = create_rnn_language_model(
            model_type, vocab_size,
            embed_dim=256, hidden_dim=256, num_layers=2
        )
    else:  # transformer, gpt
        model = create_transformer_language_model(
            model_type, vocab_size,
            embed_dim=256, num_heads=8, num_layers=6
        )
    
    model = model.to(device)
    
    # Load checkpoint
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"✅ Model loaded from {checkpoint_path}")
        print(f"   Trained for {checkpoint['epoch']} epochs")
        print(f"   Best loss: {checkpoint['loss']:.4f}")
    else:
        print(f"⚠️  Checkpoint not found: {checkpoint_path}")
        print("   Using untrained model for demonstration")
    
    return model


def generate_text_interactive(model, vocabulary: Vocabulary, device: torch.device, 
                             language: str = 'en', model_type: str = 'lstm'):
    """
    Interactive text generation session.
    
    Args:
        model: Trained language model
        vocabulary: Vocabulary object
        device: Device for computation
        language: Language for generation
        model_type: Type of model for generation method selection
    """
    print(f"\n🎨 Interactive Text Generation ({language.upper()})")
    print("=" * 50)
    print("Enter prompts to generate Australian tourism text!")
    print("Commands: 'quit' to exit, 'examples' for sample prompts")
    print("Settings: temperature=0.8, max_length=30")
    print("-" * 50)
    
    sampler = AustralianTextSampler()
    
    while True:
        try:
            prompt = input("\nPrompt: ").strip()
            
            if prompt.lower() == 'quit':
                print("👋 Goodbye!")
                break
                
            elif prompt.lower() == 'examples':
                examples = sampler.get_all_prompts(language)
                print("\n📝 Sample prompts:")
                for i, example in enumerate(examples[:5], 1):
                    print(f"  {i}. {example}")
                continue
            
            elif not prompt:
                prompt = sampler.get_random_prompt(language)
                print(f"Using random prompt: '{prompt}'")
            
            # Generate text
            print(f"🔄 Generating...")
            
            if model_type in ['lstm', 'gru', 'bilstm', 'attention_lstm']:
                # RNN generation
                input_tokens = vocabulary.encode(prompt, language, add_special_tokens=True)
                input_tensor = torch.tensor([input_tokens], dtype=torch.long).to(device)
                
                model.eval()
                with torch.no_grad():
                    generated_tensor = model.generate(
                        input_tensor, max_length=25, temperature=0.8, device=device
                    )
                
                generated_text = vocabulary.decode(generated_tensor[0].tolist(), remove_special_tokens=True)
                
            else:
                # Transformer generation
                input_tokens = vocabulary.encode(prompt, language, add_special_tokens=True)
                input_tensor = torch.tensor([input_tokens], dtype=torch.long).to(device)
                
                model.eval()
                with torch.no_grad():
                    generated_tensor = model.generate(
                        input_tensor, max_length=25, temperature=0.8,
                        top_k=50, top_p=0.9, device=device
                    )
                
                generated_text = vocabulary.decode(generated_tensor[0].tolist(), remove_special_tokens=True)
            
            print(f"✨ Generated: '{generated_text}'")
            
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Generation failed: {e}")


def batch_generate_samples(model, vocabulary: Vocabulary, device: torch.device,
                          language: str = 'en', model_type: str = 'lstm', num_samples: int = 5):
    """
    Generate multiple text samples for evaluation.
    
    Args:
        model: Trained language model
        vocabulary: Vocabulary object  
        device: Device for computation
        language: Language for generation
        model_type: Type of model
        num_samples: Number of samples to generate
    """
    print(f"\n🎯 Batch Text Generation ({num_samples} samples)")
    print("=" * 50)
    
    sampler = AustralianTextSampler()
    prompts = sampler.get_all_prompts(language)
    
    # Test different temperature settings
    temperatures = [0.7, 1.0, 1.3]
    
    for temp in temperatures:
        print(f"\n🌡️  Temperature: {temp}")
        print("-" * 30)
        
        for i in range(min(num_samples, len(prompts))):
            prompt = prompts[i]
            
            try:
                if model_type in ['lstm', 'gru', 'bilstm', 'attention_lstm']:
                    # RNN generation
                    input_tokens = vocabulary.encode(prompt, language, add_special_tokens=True)
                    input_tensor = torch.tensor([input_tokens], dtype=torch.long).to(device)
                    
                    model.eval()
                    with torch.no_grad():
                        generated_tensor = model.generate(
                            input_tensor, max_length=20, temperature=temp, device=device
                        )
                    
                    generated_text = vocabulary.decode(generated_tensor[0].tolist(), remove_special_tokens=True)
                    
                else:
                    # Transformer generation
                    input_tokens = vocabulary.encode(prompt, language, add_special_tokens=True)
                    input_tensor = torch.tensor([input_tokens], dtype=torch.long).to(device)
                    
                    model.eval()
                    with torch.no_grad():
                        generated_tensor = model.generate(
                            input_tensor, max_length=20, temperature=temp,
                            top_k=50, top_p=0.9, device=device
                        )
                    
                    generated_text = vocabulary.decode(generated_tensor[0].tolist(), remove_special_tokens=True)
                
                print(f"  '{prompt}' → '{generated_text}'")
                
            except Exception as e:
                print(f"  '{prompt}' → Generation failed: {e}")


def main():
    """Main function for text generation utility."""
    parser = argparse.ArgumentParser(description='Generate text with trained language models')
    parser.add_argument('--model_type', type=str, choices=['lstm', 'gru', 'transformer'], 
                       default='lstm', help='Type of model to use')
    parser.add_argument('--language', type=str, choices=['en', 'vi'], default='en',
                       help='Language for generation')
    parser.add_argument('--checkpoint', type=str, help='Path to model checkpoint')
    parser.add_argument('--mode', type=str, choices=['interactive', 'batch'], default='interactive',
                       help='Generation mode')
    parser.add_argument('--num_samples', type=int, default=5, help='Number of samples for batch mode')
    parser.add_argument('--data_dir', type=str, default='processed_data', help='Data directory')
    
    args = parser.parse_args()
    
    print("🎨 Word-level Language Model Text Generation")
    print("=" * 50)
    print(f"Model: {args.model_type.upper()}")
    print(f"Language: {args.language.upper()}")
    print(f"Mode: {args.mode}")
    
    # Detect device
    device, device_info = detect_device()
    print(f"Device: {device} ({device_info})")
    
    # Load vocabulary
    vocab_path = os.path.join(args.data_dir, f'{args.language}_vocabulary.pkl')
    if not os.path.exists(vocab_path):
        print(f"❌ Vocabulary not found: {vocab_path}")
        print("Please run data/process_data.py first to create vocabulary.")
        return
    
    vocabulary = Vocabulary()
    vocabulary.load(vocab_path)
    print(f"✅ Loaded vocabulary: {len(vocabulary)} words")
    
    # Load or create model
    if args.checkpoint and os.path.exists(args.checkpoint):
        model = load_trained_model(args.checkpoint, args.model_type, len(vocabulary), device)
    else:
        print(f"\n⚠️  No checkpoint provided or found. Creating untrained model for demonstration.")
        
        if args.model_type in ['lstm', 'gru', 'bilstm']:
            model = create_rnn_language_model(
                args.model_type, len(vocabulary),
                embed_dim=128, hidden_dim=128, num_layers=2
            )
        else:
            model = create_transformer_language_model(
                args.model_type, len(vocabulary),
                embed_dim=128, num_heads=4, num_layers=3
            )
        
        model = model.to(device)
        print("   Note: Generated text may not be coherent without training.")
    
    # Run generation
    if args.mode == 'interactive':
        generate_text_interactive(model, vocabulary, device, args.language, args.model_type)
    else:
        batch_generate_samples(model, vocabulary, device, args.language, args.model_type, args.num_samples)


if __name__ == '__main__':
    main()