"""
Custom PyTorch datasets for English-Vietnamese translation with Australian context.

This module implements dataset classes optimized for the repository's focus on
Australian tourism, culture, and multilingual examples.
"""

import torch
from torch.utils.data import Dataset, DataLoader, random_split
import torch.nn.functional as F
from typing import List, Tuple, Dict, Optional, Union
import random
import re
from collections import Counter
from pathlib import Path

from config import config

class TranslationVocabulary:
    """
    Vocabulary class for translation tasks with special token handling.
    
    Manages word-to-index mappings for both source and target languages,
    following PyTorch best practices for NLP datasets.
    """
    
    def __init__(self, language: str = "en"):
        self.language = language
        self.word2idx = {
            config.PAD_TOKEN: config.PAD_IDX,
            config.UNK_TOKEN: config.UNK_IDX, 
            config.SOS_TOKEN: config.SOS_IDX,
            config.EOS_TOKEN: config.EOS_IDX
        }
        self.idx2word = {v: k for k, v in self.word2idx.items()}
        self.next_idx = 4  # Start after special tokens
        
    def add_sentence(self, sentence: str):
        """Add words from sentence to vocabulary."""
        words = self.tokenize(sentence)
        for word in words:
            self.add_word(word)
    
    def add_word(self, word: str) -> int:
        """Add word to vocabulary and return its index."""
        if word not in self.word2idx:
            self.word2idx[word] = self.next_idx
            self.idx2word[self.next_idx] = word
            self.next_idx += 1
        return self.word2idx[word]
    
    def sentence_to_indices(self, sentence: str, add_eos: bool = False) -> List[int]:
        """Convert sentence to list of word indices."""
        words = self.tokenize(sentence)
        indices = [self.word2idx.get(word, config.UNK_IDX) for word in words]
        
        if add_eos:
            indices.append(config.EOS_IDX)
            
        return indices
    
    def indices_to_sentence(self, indices: List[int]) -> str:
        """Convert list of indices back to sentence."""
        words = []
        for idx in indices:
            if idx == config.EOS_IDX:
                break
            if idx not in [config.PAD_IDX, config.SOS_IDX]:
                words.append(self.idx2word.get(idx, config.UNK_TOKEN))
        return ' '.join(words)
    
    def tokenize(self, sentence: str) -> List[str]:
        """Tokenize sentence with language-specific handling."""
        # Basic preprocessing
        sentence = sentence.lower().strip()
        
        # Vietnamese-specific preprocessing
        if self.language == "vi":
            # Handle Vietnamese diacritics and tone marks
            sentence = self._normalize_vietnamese(sentence)
        
        # Basic tokenization (split on whitespace and punctuation)
        tokens = re.findall(r'\b\w+\b|[^\w\s]', sentence)
        return [token for token in tokens if token.strip()]
    
    def _normalize_vietnamese(self, text: str) -> str:
        """Basic Vietnamese text normalization."""
        # This is a simplified normalization - in practice, you'd use
        # specialized Vietnamese NLP libraries like VnCoreNLP or pyvi
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Basic punctuation normalization
        text = text.replace('"', '"').replace('"', '"')
        text = text.replace(''', "'").replace(''', "'")
        
        return text
    
    def __len__(self) -> int:
        return len(self.word2idx)

class AustralianTranslationDataset(Dataset):
    """
    PyTorch Dataset for English-Vietnamese translation with Australian context.
    
    Designed specifically for the repository's focus on Australian tourism,
    culture, wildlife, and geography examples.
    
    TensorFlow equivalent:
        dataset = tf.data.Dataset.from_tensor_slices((en_texts, vi_texts))
        dataset = dataset.map(preprocess_function)
        dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    
    Args:
        translation_pairs: List of (english, vietnamese) tuples
        src_vocab: Source language vocabulary (English)
        tgt_vocab: Target language vocabulary (Vietnamese)  
        max_length: Maximum sequence length
        transform: Optional data transformation function
    
    Example:
        >>> # Create dataset with Australian tourism translations
        >>> dataset = AustralianTranslationDataset(
        ...     config.AUSTRALIAN_TRANSLATION_PAIRS[:10],
        ...     src_vocab, tgt_vocab, max_length=50
        ... )
        >>> dataloader = DataLoader(dataset, batch_size=8, shuffle=True)
        >>> 
        >>> for batch in dataloader:
        ...     print(f"Source: {batch['src'].shape}")  # [batch_size, seq_len]
        ...     print(f"Target: {batch['tgt'].shape}")  # [batch_size, seq_len]
        ...     break
    """
    
    def __init__(
        self, 
        translation_pairs: List[Tuple[str, str]],
        src_vocab: Optional[TranslationVocabulary] = None,
        tgt_vocab: Optional[TranslationVocabulary] = None,
        max_length: int = config.MAX_SEQ_LENGTH,
        build_vocab: bool = True,
        transform=None
    ):
        self.translation_pairs = translation_pairs
        self.max_length = max_length
        self.transform = transform
        
        # Initialize vocabularies
        if src_vocab is None:
            self.src_vocab = TranslationVocabulary("en")
        else:
            self.src_vocab = src_vocab
            
        if tgt_vocab is None:
            self.tgt_vocab = TranslationVocabulary("vi")  
        else:
            self.tgt_vocab = tgt_vocab
        
        # Build vocabularies from data
        if build_vocab:
            self._build_vocabularies()
        
        print(f"🔤 Dataset created with {len(self.translation_pairs)} translation pairs")
        print(f"   📊 English vocabulary: {len(self.src_vocab)} words")
        print(f"   📊 Vietnamese vocabulary: {len(self.tgt_vocab)} words")
        print(f"   📏 Max sequence length: {self.max_length}")
        
    def _build_vocabularies(self):
        """Build vocabularies from translation pairs."""
        print("🔨 Building vocabularies from Australian translation data...")
        
        for en_text, vi_text in self.translation_pairs:
            self.src_vocab.add_sentence(en_text)
            self.tgt_vocab.add_sentence(vi_text)
    
    def __len__(self) -> int:
        return len(self.translation_pairs)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get translation pair as tensors.
        
        Returns:
            Dictionary containing:
            - src: Source sentence indices [seq_len]
            - tgt: Target sentence indices [seq_len] 
            - src_text: Original source text
            - tgt_text: Original target text
        """
        en_text, vi_text = self.translation_pairs[idx]
        
        # Convert to indices
        src_indices = self.src_vocab.sentence_to_indices(en_text)
        tgt_indices = self.tgt_vocab.sentence_to_indices(vi_text, add_eos=True)
        
        # Add SOS token to target for teacher forcing
        tgt_input = [config.SOS_IDX] + tgt_indices[:-1]  # Input to decoder
        tgt_output = tgt_indices  # Expected output (with EOS)
        
        # Pad sequences to max_length
        src_padded = self._pad_sequence(src_indices, self.max_length)
        tgt_input_padded = self._pad_sequence(tgt_input, self.max_length)  
        tgt_output_padded = self._pad_sequence(tgt_output, self.max_length)
        
        # Convert to tensors
        result = {
            'src': torch.tensor(src_padded, dtype=torch.long),
            'tgt_input': torch.tensor(tgt_input_padded, dtype=torch.long),
            'tgt_output': torch.tensor(tgt_output_padded, dtype=torch.long),
            'src_text': en_text,
            'tgt_text': vi_text,
            'src_length': len(src_indices),
            'tgt_length': len(tgt_indices)
        }
        
        # Apply transform if provided
        if self.transform:
            result = self.transform(result)
            
        return result
    
    def _pad_sequence(self, sequence: List[int], max_length: int) -> List[int]:
        """Pad sequence to max_length with PAD_IDX."""
        if len(sequence) >= max_length:
            return sequence[:max_length]
        else:
            return sequence + [config.PAD_IDX] * (max_length - len(sequence))
    
    def get_sample_batch(self, batch_size: int = 3) -> Dict[str, torch.Tensor]:
        """Get a sample batch for testing/debugging."""
        indices = random.sample(range(len(self)), min(batch_size, len(self)))
        batch = [self[i] for i in indices]
        
        return self.collate_fn(batch)
    
    @staticmethod
    def collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """
        Custom collate function for DataLoader.
        
        TensorFlow equivalent:
            Automatic batching with tf.data.Dataset.batch()
        """
        # Stack tensors
        src = torch.stack([item['src'] for item in batch])
        tgt_input = torch.stack([item['tgt_input'] for item in batch])
        tgt_output = torch.stack([item['tgt_output'] for item in batch])
        
        # Collect metadata
        src_texts = [item['src_text'] for item in batch]
        tgt_texts = [item['tgt_text'] for item in batch]
        src_lengths = torch.tensor([item['src_length'] for item in batch])
        tgt_lengths = torch.tensor([item['tgt_length'] for item in batch])
        
        return {
            'src': src,  # [batch_size, max_seq_len]
            'tgt_input': tgt_input,  # [batch_size, max_seq_len]
            'tgt_output': tgt_output,  # [batch_size, max_seq_len]
            'src_texts': src_texts,
            'tgt_texts': tgt_texts,
            'src_lengths': src_lengths,
            'tgt_lengths': tgt_lengths
        }

def create_translation_dataloaders(
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    batch_size: Optional[int] = None,
    device: Optional[torch.device] = None,
    shuffle: bool = True
) -> Tuple[DataLoader, DataLoader, DataLoader, TranslationVocabulary, TranslationVocabulary]:
    """
    Create train, validation, and test DataLoaders for translation.
    
    Args:
        train_ratio: Proportion of data for training
        val_ratio: Proportion of data for validation  
        test_ratio: Proportion of data for testing
        batch_size: Batch size (device-specific if None)
        device: Target device for optimization
        shuffle: Whether to shuffle training data
    
    Returns:
        train_loader, val_loader, test_loader, src_vocab, tgt_vocab
        
    Example:
        >>> train_loader, val_loader, test_loader, src_vocab, tgt_vocab = create_translation_dataloaders()
        >>> print(f"Train batches: {len(train_loader)}")
        >>> print(f"Validation batches: {len(val_loader)}")  
        >>> print(f"Test batches: {len(test_loader)}")
    """
    # Create dataset with all Australian translation pairs
    full_dataset = AustralianTranslationDataset(
        config.AUSTRALIAN_TRANSLATION_PAIRS,
        max_length=config.MAX_SEQ_LENGTH,
        build_vocab=True
    )
    
    # Split dataset
    total_size = len(full_dataset)
    train_size = int(train_ratio * total_size)
    val_size = int(val_ratio * total_size)  
    test_size = total_size - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = random_split(
        full_dataset, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)  # Reproducible split
    )
    
    # Determine batch size based on device
    if batch_size is None:
        if device is not None:
            device_config = config.get_device_config(device)
            batch_size = device_config['batch_size']
        else:
            batch_size = config.BATCH_SIZE
    
    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=AustralianTranslationDataset.collate_fn,
        num_workers=2,
        pin_memory=torch.cuda.is_available()
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=AustralianTranslationDataset.collate_fn,
        num_workers=2,
        pin_memory=torch.cuda.is_available()
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=AustralianTranslationDataset.collate_fn,
        num_workers=2,
        pin_memory=torch.cuda.is_available()
    )
    
    print(f"📊 Data split complete:")
    print(f"   🏋️ Train: {len(train_dataset)} examples ({len(train_loader)} batches)")
    print(f"   ✅ Validation: {len(val_dataset)} examples ({len(val_loader)} batches)")
    print(f"   🧪 Test: {len(test_dataset)} examples ({len(test_loader)} batches)")
    print(f"   📦 Batch size: {batch_size}")
    
    return train_loader, val_loader, test_loader, full_dataset.src_vocab, full_dataset.tgt_vocab

def demonstrate_dataset():
    """Demonstrate dataset functionality with Australian examples."""
    print("🚀 Demonstrating Australian Translation Dataset")
    print("=" * 60)
    
    # Create sample dataset
    sample_pairs = config.AUSTRALIAN_TRANSLATION_PAIRS[:5]
    dataset = AustralianTranslationDataset(sample_pairs, max_length=50)
    
    print(f"\n📝 Sample translations:")
    for i in range(min(3, len(dataset))):
        example = dataset[i]
        print(f"\n{i+1}. English: {example['src_text']}")
        print(f"   Vietnamese: {example['tgt_text']}")
        print(f"   Source indices: {example['src'][:10].tolist()}... (length: {example['src_length']})")
        print(f"   Target indices: {example['tgt_output'][:10].tolist()}... (length: {example['tgt_length']})")
    
    # Test DataLoader
    dataloader = DataLoader(
        dataset, 
        batch_size=2, 
        shuffle=False,
        collate_fn=AustralianTranslationDataset.collate_fn
    )
    
    print(f"\n🔄 Testing DataLoader with batch size 2:")
    for batch in dataloader:
        print(f"   Source batch shape: {batch['src'].shape}")
        print(f"   Target input shape: {batch['tgt_input'].shape}")
        print(f"   Target output shape: {batch['tgt_output'].shape}")
        print(f"   Batch texts: {len(batch['src_texts'])} source, {len(batch['tgt_texts'])} target")
        break
    
    print(f"\n✅ Dataset demonstration complete!")

if __name__ == "__main__":
    demonstrate_dataset()