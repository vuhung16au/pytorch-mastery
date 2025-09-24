"""
PyTorch Translation Models for English-Vietnamese Translation.

This module implements various neural machine translation architectures
optimized for Australian context examples and following repository standards.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Tuple, Optional, List, Dict
import random

from config import config
from utils import safe_to_device

class Seq2SeqTranslator(nn.Module):
    """
    Basic sequence-to-sequence translator with LSTM encoder-decoder.
    
    Designed for English-Vietnamese translation with Australian tourism context.
    This is the foundational model that demonstrates core seq2seq concepts.
    
    TensorFlow equivalent:
        encoder = tf.keras.Sequential([
            tf.keras.layers.Embedding(src_vocab_size, embed_dim),
            tf.keras.layers.LSTM(hidden_size, return_state=True)
        ])
        decoder = tf.keras.Sequential([
            tf.keras.layers.Embedding(tgt_vocab_size, embed_dim),
            tf.keras.layers.LSTM(hidden_size, return_sequences=True),
            tf.keras.layers.Dense(tgt_vocab_size, activation='softmax')
        ])
    
    Args:
        src_vocab_size: English vocabulary size
        tgt_vocab_size: Vietnamese vocabulary size
        embed_dim: Embedding dimension
        hidden_size: LSTM hidden state size
        num_layers: Number of LSTM layers
        dropout_rate: Dropout probability
        
    Example:
        >>> model = Seq2SeqTranslator(10000, 8000, 256, 512)
        >>> src_seq = torch.randint(0, 10000, (16, 20))  # [batch_size, seq_len]
        >>> tgt_seq = torch.randint(0, 8000, (16, 15))
        >>> output = model(src_seq, tgt_seq)
        >>> print(output.shape)  # [batch_size, tgt_seq_len, tgt_vocab_size]
    """
    
    def __init__(
        self,
        src_vocab_size: int,
        tgt_vocab_size: int, 
        embed_dim: int = config.EMBED_DIM,
        hidden_size: int = config.HIDDEN_SIZE,
        num_layers: int = config.NUM_LAYERS,
        dropout_rate: float = config.DROPOUT_RATE
    ):
        super(Seq2SeqTranslator, self).__init__()
        
        self.src_vocab_size = src_vocab_size
        self.tgt_vocab_size = tgt_vocab_size
        self.embed_dim = embed_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Encoder components
        self.src_embedding = nn.Embedding(src_vocab_size, embed_dim, padding_idx=config.PAD_IDX)
        self.encoder = nn.LSTM(
            embed_dim, hidden_size, num_layers,
            batch_first=True, dropout=dropout_rate if num_layers > 1 else 0
        )
        
        # Decoder components  
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, embed_dim, padding_idx=config.PAD_IDX)
        self.decoder = nn.LSTM(
            embed_dim, hidden_size, num_layers,
            batch_first=True, dropout=dropout_rate if num_layers > 1 else 0
        )
        
        # Output projection
        self.output_projection = nn.Linear(hidden_size, tgt_vocab_size)
        self.dropout = nn.Dropout(dropout_rate)
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        """Initialize model weights using Xavier uniform initialization."""
        for name, param in self.named_parameters():
            if 'weight' in name and len(param.shape) > 1:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)
    
    def forward(self, src_seq: torch.Tensor, tgt_seq: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for training (teacher forcing).
        
        Args:
            src_seq: Source sequences [batch_size, src_len]
            tgt_seq: Target sequences [batch_size, tgt_len]
            
        Returns:
            torch.Tensor: Output logits [batch_size, tgt_len, tgt_vocab_size]
        """
        # Encode source sequence
        src_embedded = self.src_embedding(src_seq)
        encoder_outputs, (hidden, cell) = self.encoder(src_embedded)
        
        # Decode target sequence  
        tgt_embedded = self.tgt_embedding(tgt_seq)
        decoder_outputs, _ = self.decoder(tgt_embedded, (hidden, cell))
        
        # Apply dropout and project to vocabulary
        decoder_outputs = self.dropout(decoder_outputs)
        output_logits = self.output_projection(decoder_outputs)
        
        return output_logits
    
    def generate(
        self, 
        src_seq: torch.Tensor, 
        max_length: int = config.MAX_SEQ_LENGTH,
        device: Optional[torch.device] = None
    ) -> torch.Tensor:
        """
        Generate translation using greedy decoding.
        
        Args:
            src_seq: Source sequence [1, src_len] 
            max_length: Maximum generation length
            device: Target device
            
        Returns:
            torch.Tensor: Generated sequence [1, gen_len]
        """
        self.eval()
        
        if device is None:
            device = next(self.parameters()).device
            
        with torch.no_grad():
            # Encode source
            src_seq = safe_to_device(src_seq, device, "src_seq")
            src_embedded = self.src_embedding(src_seq)
            _, (hidden, cell) = self.encoder(src_embedded)
            
            # Start generation with SOS token
            generated = [config.SOS_IDX]
            decoder_input = torch.tensor([[config.SOS_IDX]], device=device)
            
            for _ in range(max_length):
                # Decode one step
                tgt_embedded = self.tgt_embedding(decoder_input)
                decoder_output, (hidden, cell) = self.decoder(tgt_embedded, (hidden, cell))
                
                # Get next token
                output_logits = self.output_projection(decoder_output)
                next_token_id = torch.argmax(output_logits, dim=-1).item()
                
                generated.append(next_token_id)
                
                # Stop if EOS token
                if next_token_id == config.EOS_IDX:
                    break
                    
                # Prepare next input
                decoder_input = torch.tensor([[next_token_id]], device=device)
            
            return torch.tensor([generated], device=device)

class BahdanauAttention(nn.Module):
    """
    Bahdanau attention mechanism for sequence-to-sequence models.
    
    Implements additive attention with learnable alignment weights,
    allowing the decoder to focus on relevant parts of the source sequence.
    
    Reference: "Neural Machine Translation by Jointly Learning to Align and Translate"
    """
    
    def __init__(self, hidden_size: int, key_size: int = None):
        super(BahdanauAttention, self).__init__()
        
        if key_size is None:
            key_size = hidden_size
            
        self.hidden_size = hidden_size
        self.key_size = key_size
        
        # Attention components
        self.query_projection = nn.Linear(hidden_size, key_size, bias=False)
        self.key_projection = nn.Linear(hidden_size, key_size, bias=False)
        self.energy_projection = nn.Linear(key_size, 1, bias=False)
        
    def forward(
        self, 
        query: torch.Tensor, 
        keys: torch.Tensor, 
        values: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute attention-weighted context vector.
        
        Args:
            query: Decoder hidden state [batch_size, 1, hidden_size]
            keys: Encoder hidden states [batch_size, src_len, hidden_size]  
            values: Encoder hidden states [batch_size, src_len, hidden_size]
            mask: Optional padding mask [batch_size, src_len]
            
        Returns:
            context: Attention-weighted context [batch_size, 1, hidden_size]
            attention_weights: Attention weights [batch_size, 1, src_len]
        """
        batch_size, src_len, _ = keys.size()
        
        # Project query and keys
        projected_query = self.query_projection(query)  # [batch, 1, key_size]
        projected_keys = self.key_projection(keys)      # [batch, src_len, key_size]
        
        # Compute attention energies
        # Broadcast query to match keys: [batch, src_len, key_size]
        expanded_query = projected_query.expand(batch_size, src_len, self.key_size)
        
        # Additive attention: tanh(W_q * query + W_k * keys)
        energies = torch.tanh(expanded_query + projected_keys)
        attention_scores = self.energy_projection(energies).squeeze(-1)  # [batch, src_len]
        
        # Apply mask if provided (set masked positions to large negative value)
        if mask is not None:
            attention_scores.masked_fill_(mask == 0, -1e9)
        
        # Compute attention weights
        attention_weights = F.softmax(attention_scores, dim=-1)  # [batch, src_len]
        attention_weights = attention_weights.unsqueeze(1)       # [batch, 1, src_len]
        
        # Compute context vector
        context = torch.bmm(attention_weights, values)  # [batch, 1, hidden_size]
        
        return context, attention_weights

class AttentionTranslator(nn.Module):
    """
    Sequence-to-sequence translator with Bahdanau attention.
    
    Improves upon basic seq2seq by allowing the decoder to attend to
    different parts of the source sequence at each decoding step.
    
    Key improvements:
    - Context vector from all encoder states
    - Attention weights for interpretability  
    - Better handling of long sequences
    - Reduced information bottleneck
    
    Args:
        src_vocab_size: English vocabulary size
        tgt_vocab_size: Vietnamese vocabulary size
        embed_dim: Embedding dimension
        hidden_size: LSTM hidden state size
        num_layers: Number of LSTM layers
        dropout_rate: Dropout probability
        
    Example:
        >>> model = AttentionTranslator(10000, 8000, 256, 512)
        >>> src_seq = torch.randint(0, 10000, (16, 20))
        >>> tgt_seq = torch.randint(0, 8000, (16, 15))
        >>> output, attention_weights = model(src_seq, tgt_seq)
        >>> print(f"Output: {output.shape}, Attention: {attention_weights.shape}")
    """
    
    def __init__(
        self,
        src_vocab_size: int,
        tgt_vocab_size: int,
        embed_dim: int = config.EMBED_DIM,
        hidden_size: int = config.HIDDEN_SIZE,
        num_layers: int = config.NUM_LAYERS,
        dropout_rate: float = config.DROPOUT_RATE
    ):
        super(AttentionTranslator, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Encoder
        self.src_embedding = nn.Embedding(src_vocab_size, embed_dim, padding_idx=config.PAD_IDX)
        self.encoder = nn.LSTM(
            embed_dim, hidden_size, num_layers,
            batch_first=True, dropout=dropout_rate if num_layers > 1 else 0,
            bidirectional=True  # Bidirectional for better representations
        )
        
        # Decoder  
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, embed_dim, padding_idx=config.PAD_IDX)
        
        # Attention mechanism
        self.attention = BahdanauAttention(hidden_size * 2)  # *2 for bidirectional
        
        # Decoder LSTM (input: embedding + context)
        self.decoder = nn.LSTM(
            embed_dim + hidden_size * 2, hidden_size, num_layers,
            batch_first=True, dropout=dropout_rate if num_layers > 1 else 0
        )
        
        # Output projection (input: decoder output + context)
        self.output_projection = nn.Linear(hidden_size + hidden_size * 2, tgt_vocab_size)
        self.dropout = nn.Dropout(dropout_rate)
        
        # Bridge encoder bidirectional to decoder unidirectional
        self.bridge_hidden = nn.Linear(hidden_size * 2, hidden_size)
        self.bridge_cell = nn.Linear(hidden_size * 2, hidden_size)
        
        self._init_weights()
        
    def _init_weights(self):
        """Initialize model weights."""
        for name, param in self.named_parameters():
            if 'weight' in name and len(param.shape) > 1:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)
    
    def forward(
        self, 
        src_seq: torch.Tensor, 
        tgt_seq: torch.Tensor, 
        src_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with attention.
        
        Args:
            src_seq: Source sequences [batch_size, src_len]
            tgt_seq: Target sequences [batch_size, tgt_len]  
            src_mask: Source padding mask [batch_size, src_len]
            
        Returns:
            output_logits: [batch_size, tgt_len, tgt_vocab_size]
            attention_weights: [batch_size, tgt_len, src_len]
        """
        batch_size, tgt_len = tgt_seq.size()
        
        # Encode source sequence
        src_embedded = self.src_embedding(src_seq)
        encoder_outputs, (hidden, cell) = self.encoder(src_embedded)
        
        # Bridge bidirectional encoder to unidirectional decoder
        # hidden: [num_layers * 2, batch_size, hidden_size] -> [num_layers, batch_size, hidden_size]
        hidden = hidden.view(self.num_layers, 2, batch_size, self.hidden_size)
        hidden = torch.cat([hidden[:, 0, :, :], hidden[:, 1, :, :]], dim=-1)  # Concatenate forward and backward
        cell = cell.view(self.num_layers, 2, batch_size, self.hidden_size) 
        cell = torch.cat([cell[:, 0, :, :], cell[:, 1, :, :]], dim=-1)  # Concatenate forward and backward
        
        # Apply bridge layers to reduce dimension back to hidden_size
        hidden = self.bridge_hidden(hidden.view(-1, self.hidden_size * 2))
        hidden = hidden.view(self.num_layers, batch_size, self.hidden_size)
        cell = self.bridge_cell(cell.view(-1, self.hidden_size * 2))
        cell = cell.view(self.num_layers, batch_size, self.hidden_size)
        
        # Decode with attention
        tgt_embedded = self.tgt_embedding(tgt_seq)
        
        decoder_outputs = []
        attention_weights_list = []
        decoder_hidden = (hidden, cell)
        
        for t in range(tgt_len):
            # Current target embedding
            current_input = tgt_embedded[:, t:t+1, :]  # [batch, 1, embed_dim]
            
            # Compute attention
            query = decoder_hidden[0][-1:].transpose(0, 1)  # [batch, 1, hidden_size]
            context, attention_weights = self.attention(
                query, encoder_outputs, encoder_outputs, src_mask
            )
            
            # Concatenate embedding with context
            decoder_input = torch.cat([current_input, context], dim=-1)
            
            # Decode one step
            decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
            
            # Combine decoder output with context for final projection
            combined_output = torch.cat([decoder_output, context], dim=-1)
            
            decoder_outputs.append(combined_output)
            attention_weights_list.append(attention_weights)
        
        # Stack outputs
        decoder_outputs = torch.cat(decoder_outputs, dim=1)  # [batch, tgt_len, hidden_size + context_size]
        attention_weights = torch.cat(attention_weights_list, dim=1)  # [batch, tgt_len, src_len]
        
        # Final projection
        output_logits = self.output_projection(decoder_outputs)
        
        return output_logits, attention_weights

class PositionalEncoding(nn.Module):
    """Positional encoding for Transformer models."""
    
    def __init__(self, d_model: int, max_len: int = 5000):
        super(PositionalEncoding, self).__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:x.size(0), :]

class TransformerTranslator(nn.Module):
    """
    Modern Transformer-based translator for English-Vietnamese translation.
    
    Implements the "Attention Is All You Need" architecture with:
    - Multi-head self-attention
    - Positional encodings
    - Feed-forward networks
    - Layer normalization and residual connections
    
    Args:
        src_vocab_size: English vocabulary size
        tgt_vocab_size: Vietnamese vocabulary size
        d_model: Model dimension (embedding size)
        nhead: Number of attention heads
        num_encoder_layers: Number of encoder layers
        num_decoder_layers: Number of decoder layers
        dim_feedforward: Feed-forward dimension
        dropout: Dropout rate
        
    Example:
        >>> model = TransformerTranslator(10000, 8000, 512, 8, 6, 6)
        >>> src_seq = torch.randint(0, 10000, (16, 20))
        >>> tgt_seq = torch.randint(0, 8000, (16, 15))
        >>> output = model(src_seq, tgt_seq)
        >>> print(output.shape)  # [batch_size, tgt_len, tgt_vocab_size]
    """
    
    def __init__(
        self,
        src_vocab_size: int,
        tgt_vocab_size: int,
        d_model: int = config.D_MODEL,
        nhead: int = config.N_HEADS,
        num_encoder_layers: int = config.N_ENCODER_LAYERS,
        num_decoder_layers: int = config.N_DECODER_LAYERS,
        dim_feedforward: int = config.DIM_FEEDFORWARD,
        dropout: float = config.DROPOUT_RATE
    ):
        super(TransformerTranslator, self).__init__()
        
        self.d_model = d_model
        self.src_vocab_size = src_vocab_size
        self.tgt_vocab_size = tgt_vocab_size
        
        # Embeddings
        self.src_embedding = nn.Embedding(src_vocab_size, d_model, padding_idx=config.PAD_IDX)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model, padding_idx=config.PAD_IDX)
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(d_model)
        
        # Transformer
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        
        # Output projection
        self.output_projection = nn.Linear(d_model, tgt_vocab_size)
        self.dropout = nn.Dropout(dropout)
        
        self._init_weights()
        
    def _init_weights(self):
        """Initialize model weights."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(
        self, 
        src_seq: torch.Tensor, 
        tgt_seq: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
        tgt_mask: Optional[torch.Tensor] = None,
        src_key_padding_mask: Optional[torch.Tensor] = None,
        tgt_key_padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass through transformer.
        
        Args:
            src_seq: Source sequences [batch_size, src_len]
            tgt_seq: Target sequences [batch_size, tgt_len]
            src_mask: Source attention mask
            tgt_mask: Target attention mask (causal mask)
            src_key_padding_mask: Source padding mask  
            tgt_key_padding_mask: Target padding mask
            
        Returns:
            torch.Tensor: Output logits [batch_size, tgt_len, tgt_vocab_size]
        """
        # Embeddings with positional encoding
        src_embedded = self.src_embedding(src_seq) * math.sqrt(self.d_model)
        tgt_embedded = self.tgt_embedding(tgt_seq) * math.sqrt(self.d_model)
        
        src_embedded = self.pos_encoding(src_embedded.transpose(0, 1)).transpose(0, 1)
        tgt_embedded = self.pos_encoding(tgt_embedded.transpose(0, 1)).transpose(0, 1)
        
        src_embedded = self.dropout(src_embedded)
        tgt_embedded = self.dropout(tgt_embedded)
        
        # Generate causal mask for target if not provided
        if tgt_mask is None:
            tgt_len = tgt_seq.size(1)
            tgt_mask = self.transformer.generate_square_subsequent_mask(tgt_len)
            tgt_mask = tgt_mask.to(src_seq.device)
        
        # Transformer forward pass
        transformer_output = self.transformer(
            src=src_embedded,
            tgt=tgt_embedded,
            src_mask=src_mask,
            tgt_mask=tgt_mask,
            src_key_padding_mask=src_key_padding_mask,
            tgt_key_padding_mask=tgt_key_padding_mask
        )
        
        # Project to vocabulary
        output_logits = self.output_projection(transformer_output)
        
        return output_logits

def create_padding_mask(sequences: torch.Tensor, pad_idx: int = config.PAD_IDX) -> torch.Tensor:
    """
    Create padding mask for sequences.
    
    Args:
        sequences: Input sequences [batch_size, seq_len]
        pad_idx: Padding token index
        
    Returns:
        torch.Tensor: Padding mask [batch_size, seq_len] (1 for valid tokens, 0 for padding)
    """
    return (sequences != pad_idx)

def test_models():
    """Test all translation models with sample data."""
    print("🧪 Testing Translation Models")
    print("=" * 50)
    
    # Test parameters
    batch_size = 4
    src_len = 10
    tgt_len = 8
    src_vocab_size = 1000
    tgt_vocab_size = 800
    
    # Sample data
    src_seq = torch.randint(0, src_vocab_size, (batch_size, src_len))
    tgt_seq = torch.randint(0, tgt_vocab_size, (batch_size, tgt_len))
    
    print(f"Test data: src {src_seq.shape}, tgt {tgt_seq.shape}")
    
    # Test Seq2Seq
    print(f"\n1. Testing Basic Seq2Seq Translator:")
    seq2seq = Seq2SeqTranslator(src_vocab_size, tgt_vocab_size, 128, 256)
    output = seq2seq(src_seq, tgt_seq)
    print(f"   Output shape: {output.shape}")
    print(f"   Parameters: {sum(p.numel() for p in seq2seq.parameters()):,}")
    
    # Test Attention
    print(f"\n2. Testing Attention Translator:")
    attention_model = AttentionTranslator(src_vocab_size, tgt_vocab_size, 128, 256)
    output, attention_weights = attention_model(src_seq, tgt_seq)
    print(f"   Output shape: {output.shape}")
    print(f"   Attention shape: {attention_weights.shape}")
    print(f"   Parameters: {sum(p.numel() for p in attention_model.parameters()):,}")
    
    # Test Transformer
    print(f"\n3. Testing Transformer Translator:")
    transformer = TransformerTranslator(src_vocab_size, tgt_vocab_size, 256, 4, 2, 2)
    output = transformer(src_seq, tgt_seq)
    print(f"   Output shape: {output.shape}")
    print(f"   Parameters: {sum(p.numel() for p in transformer.parameters()):,}")
    
    print(f"\n✅ All models tested successfully!")

if __name__ == "__main__":
    test_models()