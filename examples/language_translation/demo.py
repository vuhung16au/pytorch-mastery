#!/usr/bin/env python3
"""
Quick Demo: Australian English-Vietnamese Translation with PyTorch

This script demonstrates the language translation system with Australian examples.
Run this to see the translation capabilities in action!
"""

import sys
import os
import torch

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import config
from dataset import AustralianTranslationDataset, create_translation_dataloaders
from translation_models import Seq2SeqTranslator
from utils import detect_device, calculate_bleu_score

def quick_demo():
    """Quick demonstration of the translation system."""
    print("🇦🇺 → 🇻🇳 Australian English-Vietnamese Translation Demo")
    print("=" * 60)
    
    # Setup
    device, device_info = detect_device()
    print(f"🖥️  Device: {device} ({device_info})")
    
    # Load data
    print(f"\n📚 Loading Australian translation data...")
    train_loader, val_loader, test_loader, src_vocab, tgt_vocab = create_translation_dataloaders(
        train_ratio=0.8,
        val_ratio=0.1, 
        test_ratio=0.1,
        device=device
    )
    
    print(f"✅ Data loaded: {len(src_vocab)} English words, {len(tgt_vocab)} Vietnamese words")
    
    # Create model
    print(f"\n🧠 Creating translation model...")
    model = Seq2SeqTranslator(
        src_vocab_size=len(src_vocab),
        tgt_vocab_size=len(tgt_vocab),
        embed_dim=128,
        hidden_size=256
    ).to(device)
    
    print(f"✅ Model created with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Quick training (just a few steps for demo)
    print(f"\n🏋️ Quick training (demo purposes - 5 steps only)...")
    criterion = torch.nn.CrossEntropyLoss(ignore_index=0)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    model.train()
    for step, batch in enumerate(train_loader):
        if step >= 5:  # Just 5 steps for demo
            break
            
        src = batch['src'].to(device)
        tgt_input = batch['tgt_input'].to(device)
        tgt_output = batch['tgt_output'].to(device)
        
        optimizer.zero_grad()
        output = model(src, tgt_input)
        loss = criterion(output.view(-1, output.size(-1)), tgt_output.view(-1))
        loss.backward()
        optimizer.step()
        
        if step == 0:
            print(f"   Step 1: Loss = {loss.item():.4f}")
        elif step == 4:
            print(f"   Step 5: Loss = {loss.item():.4f}")
    
    print(f"✅ Quick training completed!")
    
    # Test translations
    print(f"\n🔮 Testing translations on sample Australian phrases:")
    model.eval()
    
    # Sample Australian phrases
    test_phrases = [
        "Sydney Opera House is beautiful",
        "Melbourne has great coffee",
        "Kangaroos are unique animals",
        "Australian beaches are amazing",
        "The weather in Brisbane is warm"
    ]
    
    predictions = []
    references = []
    
    for phrase in test_phrases:
        try:
            # Get expected translation if it exists
            expected = None
            for en_text, vi_text in config.AUSTRALIAN_TRANSLATION_PAIRS:
                if phrase.lower() in en_text.lower() or en_text.lower() in phrase.lower():
                    expected = vi_text
                    break
            
            # Convert to indices
            indices = src_vocab.sentence_to_indices(phrase)
            if len(indices) == 0:
                continue
                
            input_tensor = torch.tensor([indices], dtype=torch.long).to(device)
            
            # Generate translation
            with torch.no_grad():
                generated = model.generate(input_tensor, max_length=30, device=device)
                translation = tgt_vocab.indices_to_sentence(generated[0].cpu().tolist())
            
            print(f"\n   📝 English: {phrase}")
            print(f"   🔄 Translation: {translation}")
            if expected:
                print(f"   ✅ Expected: {expected}")
                predictions.append(translation)
                references.append(expected)
            else:
                print(f"   💡 (Generated translation - no reference available)")
                
        except Exception as e:
            print(f"   ❌ Translation failed: {e}")
    
    # Calculate BLEU score if we have references
    if predictions and references:
        bleu_score = calculate_bleu_score(predictions, references)
        print(f"\n📊 BLEU Score: {bleu_score:.3f}")
        print(f"   Note: Score is low due to minimal training (this is just a demo!)")
    
    print(f"\n" + "=" * 60)
    print(f"🎉 Demo completed successfully!")
    print(f"\n📖 To explore further:")
    print(f"   • Run the full test: python test_translation.py")
    print(f"   • Open Jupyter notebooks for detailed tutorials") 
    print(f"   • Check README.md for comprehensive documentation")
    print(f"\n💡 For better translations, train longer with:")
    print(f"   • More epochs (50+ instead of 5 steps)")
    print(f"   • Larger models (512+ hidden size)")
    print(f"   • More data (expand AUSTRALIAN_TRANSLATION_PAIRS)")
    print(f"   • Attention mechanisms (see attention_translation.py)")
    
    print(f"\n🇦🇺 G'day mate! Your PyTorch translation system is ready! 🇻🇳")

if __name__ == "__main__":
    try:
        quick_demo()
    except KeyboardInterrupt:
        print(f"\n👋 Demo interrupted by user")
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        print(f"💡 Try running: python test_translation.py")
        sys.exit(1)