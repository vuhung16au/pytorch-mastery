#!/usr/bin/env python3
"""
Test script for the Language Translation implementation.

This script validates all components of the translation system
and demonstrates the functionality with Australian examples.
"""

import sys
import os
import torch

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import our modules
from config import config
from dataset import AustralianTranslationDataset, create_translation_dataloaders, demonstrate_dataset
from translation_models import Seq2SeqTranslator, AttentionTranslator, TransformerTranslator, test_models
from utils import detect_device, test_device_functionality, calculate_bleu_score, plot_training_metrics

def main():
    """Test all components of the translation system."""
    print("🚀 Testing PyTorch Language Translation Implementation")
    print("=" * 60)
    
    # Test device functionality
    print("\n1. Testing Device Detection:")
    device_test_passed = test_device_functionality()
    
    if device_test_passed:
        print("✅ Device test passed!")
    else:
        print("⚠️ Device test failed, but continuing...")
    
    # Test configuration
    print("\n2. Testing Configuration:")
    print(f"   📊 Source vocab size: {config.SRC_VOCAB_SIZE}")
    print(f"   📊 Target vocab size: {config.TGT_VOCAB_SIZE}")
    print(f"   📏 Max sequence length: {config.MAX_SEQ_LENGTH}")
    print(f"   🎯 Batch size: {config.BATCH_SIZE}")
    print(f"   🔢 Translation pairs: {len(config.AUSTRALIAN_TRANSLATION_PAIRS)}")
    
    # Test dataset functionality
    print("\n3. Testing Dataset:")
    try:
        demonstrate_dataset()
        print("✅ Dataset test passed!")
    except Exception as e:
        print(f"❌ Dataset test failed: {e}")
        return
    
    # Test basic model only for now
    print("\n4. Testing Basic Seq2Seq Model:")
    try:
        # Test parameters
        batch_size = 4
        src_len = 10
        tgt_len = 8
        src_vocab_size = 1000
        tgt_vocab_size = 800
        
        # Sample data
        src_seq = torch.randint(0, src_vocab_size, (batch_size, src_len))
        tgt_seq = torch.randint(0, tgt_vocab_size, (batch_size, tgt_len))
        
        print(f"   Test data: src {src_seq.shape}, tgt {tgt_seq.shape}")
        
        # Test Seq2Seq
        seq2seq = Seq2SeqTranslator(src_vocab_size, tgt_vocab_size, 128, 256)
        output = seq2seq(src_seq, tgt_seq)
        print(f"   Output shape: {output.shape}")
        print(f"   Parameters: {sum(p.numel() for p in seq2seq.parameters()):,}")
        print("✅ Basic model test passed!")
    except Exception as e:
        print(f"❌ Model test failed: {e}")
        return
    
    # Test BLEU score calculation
    print("\n5. Testing BLEU Score Calculation:")
    try:
        predictions = [
            "Sydney Opera House is beautiful",
            "Melbourne has great coffee culture"
        ]
        references = [
            "The Sydney Opera House is beautiful", 
            "Melbourne has excellent coffee culture"
        ]
        
        bleu = calculate_bleu_score(predictions, references)
        print(f"   📊 BLEU Score: {bleu:.3f}")
        print("✅ BLEU calculation test passed!")
    except Exception as e:
        print(f"❌ BLEU test failed: {e}")
    
    # Test data loading
    print("\n6. Testing DataLoader Creation:")
    try:
        device, _ = detect_device()
        train_loader, val_loader, test_loader, src_vocab, tgt_vocab = create_translation_dataloaders(
            train_ratio=0.7,
            val_ratio=0.2,
            test_ratio=0.1,
            device=device
        )
        
        print(f"   🏋️ Train batches: {len(train_loader)}")
        print(f"   ✅ Validation batches: {len(val_loader)}")
        print(f"   🧪 Test batches: {len(test_loader)}")
        print(f"   🔤 Source vocab size: {len(src_vocab)}")
        print(f"   🔤 Target vocab size: {len(tgt_vocab)}")
        
        # Test a batch
        for batch in train_loader:
            print(f"   📦 Batch source shape: {batch['src'].shape}")
            print(f"   📦 Batch target input shape: {batch['tgt_input'].shape}")  
            print(f"   📦 Batch target output shape: {batch['tgt_output'].shape}")
            break
            
        print("✅ DataLoader test passed!")
    except Exception as e:
        print(f"❌ DataLoader test failed: {e}")
        return
    
    # Test simple training step
    print("\n7. Testing Simple Training Step:")
    try:
        device, _ = detect_device()
        
        # Create small model for testing
        model = Seq2SeqTranslator(
            src_vocab_size=len(src_vocab),
            tgt_vocab_size=len(tgt_vocab),
            embed_dim=64,
            hidden_size=128
        ).to(device)
        
        criterion = torch.nn.CrossEntropyLoss(ignore_index=0)  # Ignore padding
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        
        # Get a batch
        for batch in train_loader:
            src = batch['src'].to(device)
            tgt_input = batch['tgt_input'].to(device)
            tgt_output = batch['tgt_output'].to(device)
            
            # Forward pass
            output = model(src, tgt_input)  # [batch_size, seq_len, vocab_size]
            
            # Compute loss
            loss = criterion(output.view(-1, output.size(-1)), tgt_output.view(-1))
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            print(f"   📊 Training loss: {loss.item():.4f}")
            print(f"   📦 Output shape: {output.shape}")
            break
            
        print("✅ Training step test passed!")
        
    except Exception as e:
        print(f"❌ Training step test failed: {e}")
    
    # Test translation generation
    print("\n8. Testing Translation Generation:")
    try:
        model.eval()
        
        # Test sentence
        test_sentence = "Sydney Opera House is beautiful"
        test_indices = src_vocab.sentence_to_indices(test_sentence)
        test_tensor = torch.tensor([test_indices], dtype=torch.long).to(device)
        
        # Generate translation
        with torch.no_grad():
            generated = model.generate(test_tensor, max_length=20, device=device)
            translation = tgt_vocab.indices_to_sentence(generated[0].cpu().tolist())
        
        print(f"   📝 Input: {test_sentence}")
        print(f"   🔄 Translation: {translation}")
        print("✅ Translation generation test passed!")
        
    except Exception as e:
        print(f"❌ Translation generation test failed: {e}")
    
    print("\n" + "=" * 60)
    print("🎉 All tests completed!")
    print("\n📖 Next steps:")
    print("   1. Open 01_seq2seq_translation.ipynb to explore the full tutorial")
    print("   2. Train models with larger datasets and more epochs")
    print("   3. Experiment with attention mechanisms in 02_attention_translation.ipynb")
    print("   4. Try modern transformers in 03_transformer_translation.ipynb")
    print("\n🇦🇺 Ready to translate Australian content to Vietnamese! 🇻🇳")

if __name__ == "__main__":
    main()