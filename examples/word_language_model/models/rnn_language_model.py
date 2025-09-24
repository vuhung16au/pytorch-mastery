"""
RNN-based Language Models for Australian Tourism Corpus.

This module implements various RNN architectures (LSTM, GRU, Bidirectional LSTM)
for word-level language modeling with comprehensive Australian context support.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict, Any
import math


class RNNLanguageModel(nn.Module):
    """
    Base RNN Language Model for Australian tourism text generation.
    
    This implementation uses:
    - Configurable RNN architectures (LSTM, GRU, vanilla RNN)
    - Dropout regularization for better generalization
    - Weight tying between input and output embeddings
    - Support for both English and Vietnamese text
    
    TensorFlow equivalent:
        model = tf.keras.Sequential([
            tf.keras.layers.Embedding(vocab_size, embed_dim),
            tf.keras.layers.LSTM(hidden_dim, return_sequences=True),
            tf.keras.layers.Dense(vocab_size)
        ])
    
    Args:
        vocab_size (int): Size of vocabulary
        embed_dim (int): Embedding dimension (typically 256-512)
        hidden_dim (int): Hidden state dimension (typically 512-1024)
        num_layers (int): Number of RNN layers (typically 2-3)
        rnn_type (str): Type of RNN ('lstm', 'gru', 'rnn')
        dropout (float): Dropout probability for regularization
        tie_weights (bool): Whether to tie input and output embeddings
        bidirectional (bool): Whether to use bidirectional RNN
    """
    
    def __init__(self, vocab_size: int, embed_dim: int = 512, hidden_dim: int = 512,
                 num_layers: int = 2, rnn_type: str = 'lstm', dropout: float = 0.2,
                 tie_weights: bool = True, bidirectional: bool = False):
        super(RNNLanguageModel, self).__init__()
        
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.rnn_type = rnn_type.lower()
        self.dropout_rate = dropout
        self.tie_weights = tie_weights
        self.bidirectional = bidirectional
        
        # Calculate actual hidden dimension for bidirectional RNNs
        self.rnn_hidden_dim = hidden_dim // 2 if bidirectional else hidden_dim
        self.num_directions = 2 if bidirectional else 1
        
        # Australian tourism vocabulary examples
        self.australian_examples = {
            'cities': ["Sydney", "Melbourne", "Brisbane", "Perth", "Adelaide"],
            'landmarks': ["Opera House", "Harbour Bridge", "Uluru", "Great Barrier Reef"],
            'activities': ["surfing", "diving", "bushwalking", "wine tasting"]
        }
        
        # Input embedding layer
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        
        # RNN layer
        if self.rnn_type == 'lstm':
            self.rnn = nn.LSTM(
                embed_dim, self.rnn_hidden_dim, num_layers,
                batch_first=True, dropout=dropout if num_layers > 1 else 0,
                bidirectional=bidirectional
            )
        elif self.rnn_type == 'gru':
            self.rnn = nn.GRU(
                embed_dim, self.rnn_hidden_dim, num_layers,
                batch_first=True, dropout=dropout if num_layers > 1 else 0,
                bidirectional=bidirectional
            )
        elif self.rnn_type == 'rnn':
            self.rnn = nn.RNN(
                embed_dim, self.rnn_hidden_dim, num_layers,
                batch_first=True, dropout=dropout if num_layers > 1 else 0,
                bidirectional=bidirectional, nonlinearity='tanh'
            )
        else:
            raise ValueError(f"Unsupported RNN type: {self.rnn_type}")
        
        # Dropout layer
        self.dropout = nn.Dropout(dropout)
        
        # Output projection layer
        self.output_projection = nn.Linear(hidden_dim, vocab_size)
        
        # Initialize weights
        self._init_weights()
        
        # Tie input and output embeddings if specified
        if tie_weights:
            if embed_dim != hidden_dim:
                print(f"⚠️  Cannot tie weights: embed_dim ({embed_dim}) != hidden_dim ({hidden_dim})")
            else:
                self.output_projection.weight = self.embedding.weight
                print("🔗 Input and output embeddings tied")
    
    def _init_weights(self):
        """Initialize weights using Xavier uniform initialization."""
        # Initialize embedding weights
        nn.init.uniform_(self.embedding.weight, -0.1, 0.1)
        
        # Initialize RNN weights
        for name, param in self.rnn.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)
                # Initialize forget gate bias to 1 for LSTM
                if self.rnn_type == 'lstm' and 'bias_ih' in name:
                    hidden_size = param.size(0) // 4
                    param.data[hidden_size:2*hidden_size].fill_(1.0)
        
        # Initialize output projection
        nn.init.xavier_uniform_(self.output_projection.weight)
        nn.init.zeros_(self.output_projection.bias)
    
    def forward(self, input_ids: torch.Tensor, hidden: Optional[Tuple] = None) -> Tuple[torch.Tensor, Tuple]:
        """
        Forward pass through the RNN language model.
        
        Args:
            input_ids: Input token indices [batch_size, seq_len]
            hidden: Initial hidden state (optional)
            
        Returns:
            Tuple of (logits, hidden_state)
                - logits: [batch_size, seq_len, vocab_size]
                - hidden_state: Tuple of hidden states for next step
        """
        batch_size, seq_len = input_ids.size()
        
        # Embedding lookup
        embedded = self.embedding(input_ids)  # [batch_size, seq_len, embed_dim]
        embedded = self.dropout(embedded)
        
        # RNN forward pass
        rnn_output, hidden_state = self.rnn(embedded, hidden)
        # rnn_output: [batch_size, seq_len, hidden_dim * num_directions]
        
        # Apply dropout to RNN output
        rnn_output = self.dropout(rnn_output)
        
        # Project to vocabulary size
        logits = self.output_projection(rnn_output)  # [batch_size, seq_len, vocab_size]
        
        return logits, hidden_state
    
    def init_hidden(self, batch_size: int, device: torch.device) -> Tuple:
        """
        Initialize hidden state for RNN.
        
        Args:
            batch_size: Batch size
            device: Device to create tensors on
            
        Returns:
            Initial hidden state tuple
        """
        hidden_size = self.num_layers * self.num_directions
        
        if self.rnn_type == 'lstm':
            h_0 = torch.zeros(hidden_size, batch_size, self.rnn_hidden_dim, device=device)
            c_0 = torch.zeros(hidden_size, batch_size, self.rnn_hidden_dim, device=device)
            return (h_0, c_0)
        else:  # GRU or RNN
            h_0 = torch.zeros(hidden_size, batch_size, self.rnn_hidden_dim, device=device)
            return h_0
    
    def generate(self, start_tokens: torch.Tensor, max_length: int = 50,
                temperature: float = 1.0, device: torch.device = torch.device('cpu')) -> torch.Tensor:
        """
        Generate text using the trained model.
        
        Args:
            start_tokens: Starting tokens [1, seq_len]
            max_length: Maximum generation length
            temperature: Sampling temperature
            device: Device for computation
            
        Returns:
            Generated token sequence
        """
        self.eval()
        
        with torch.no_grad():
            batch_size = start_tokens.size(0)
            generated = start_tokens
            hidden = self.init_hidden(batch_size, device)
            
            for _ in range(max_length):
                # Forward pass
                logits, hidden = self.forward(generated, hidden)
                
                # Get logits for last time step and apply temperature
                next_token_logits = logits[:, -1, :] / temperature
                
                # Sample next token
                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                
                # Append to generated sequence
                generated = torch.cat([generated, next_token], dim=1)
        
        return generated
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model architecture information."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'architecture': f"{self.rnn_type.upper()} Language Model",
            'vocab_size': self.vocab_size,
            'embed_dim': self.embed_dim,
            'hidden_dim': self.hidden_dim,
            'num_layers': self.num_layers,
            'bidirectional': self.bidirectional,
            'dropout': self.dropout_rate,
            'tie_weights': self.tie_weights,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'australian_context': True,
            'multilingual_support': True
        }


class LSTMLanguageModel(RNNLanguageModel):
    """
    LSTM-based Language Model optimized for Australian tourism text.
    
    This model excels at:
    - Capturing long-term dependencies in Australian tourism descriptions
    - Learning patterns in location names and cultural references
    - Generating coherent multi-sentence descriptions
    
    Example usage:
        >>> model = LSTMLanguageModel(vocab_size=10000, embed_dim=512, hidden_dim=512)
        >>> # Train on: "Sydney Opera House is a UNESCO World Heritage site..."
        >>> # Generate: "Melbourne coffee culture thrives in hidden laneways..."
    """
    
    def __init__(self, vocab_size: int, embed_dim: int = 512, hidden_dim: int = 512,
                 num_layers: int = 2, dropout: float = 0.2, tie_weights: bool = True):
        super(LSTMLanguageModel, self).__init__(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            rnn_type='lstm',
            dropout=dropout,
            tie_weights=tie_weights,
            bidirectional=False
        )


class GRULanguageModel(RNNLanguageModel):
    """
    GRU-based Language Model for efficient Australian text generation.
    
    GRU advantages:
    - Faster training than LSTM (fewer parameters)
    - Good performance on shorter sequences
    - Efficient memory usage for large vocabularies
    
    Ideal for:
    - Quick prototyping of Australian NLP applications
    - Resource-constrained environments
    - Real-time text generation applications
    """
    
    def __init__(self, vocab_size: int, embed_dim: int = 512, hidden_dim: int = 512,
                 num_layers: int = 2, dropout: float = 0.2, tie_weights: bool = True):
        super(GRULanguageModel, self).__init__(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            rnn_type='gru',
            dropout=dropout,
            tie_weights=tie_weights,
            bidirectional=False
        )


class BiLSTMLanguageModel(RNNLanguageModel):
    """
    Bidirectional LSTM Language Model for enhanced Australian text understanding.
    
    Features:
    - Forward and backward context processing
    - Better understanding of Australian cultural references
    - Enhanced performance on complex sentence structures
    
    Use cases:
    - High-quality text generation
    - Understanding complex Australian tourism descriptions
    - Processing multilingual English-Vietnamese content
    """
    
    def __init__(self, vocab_size: int, embed_dim: int = 512, hidden_dim: int = 512,
                 num_layers: int = 2, dropout: float = 0.2, tie_weights: bool = False):
        # Note: tie_weights disabled by default for bidirectional models
        # because hidden_dim becomes embed_dim * 2 internally
        super(BiLSTMLanguageModel, self).__init__(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            rnn_type='lstm',
            dropout=dropout,
            tie_weights=tie_weights,
            bidirectional=True
        )


class AttentionLSTMLanguageModel(nn.Module):
    """
    LSTM Language Model with Attention Mechanism for Australian Tourism.
    
    This advanced model incorporates:
    - Self-attention over LSTM hidden states
    - Focus on relevant Australian context words
    - Enhanced generation quality for tourism content
    
    TensorFlow equivalent:
        # TensorFlow doesn't have built-in attention for RNNs
        # This would require custom implementation with tf.keras.layers.Attention
    """
    
    def __init__(self, vocab_size: int, embed_dim: int = 512, hidden_dim: int = 512,
                 num_layers: int = 2, dropout: float = 0.2, num_attention_heads: int = 8):
        super(AttentionLSTMLanguageModel, self).__init__()
        
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_attention_heads = num_attention_heads
        
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        
        # LSTM layers
        self.lstm = nn.LSTM(
            embed_dim, hidden_dim, num_layers,
            batch_first=True, dropout=dropout if num_layers > 1 else 0
        )
        
        # Multi-head attention layer
        self.attention = nn.MultiheadAttention(
            hidden_dim, num_attention_heads, dropout=dropout, batch_first=True
        )
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(hidden_dim)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Output projection
        self.output_projection = nn.Linear(hidden_dim, vocab_size)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights for attention-based LSTM."""
        nn.init.uniform_(self.embedding.weight, -0.1, 0.1)
        nn.init.xavier_uniform_(self.output_projection.weight)
        nn.init.zeros_(self.output_projection.bias)
    
    def forward(self, input_ids: torch.Tensor, hidden: Optional[Tuple] = None) -> Tuple[torch.Tensor, Tuple]:
        """
        Forward pass with attention mechanism.
        
        Args:
            input_ids: Input token indices [batch_size, seq_len]
            hidden: Initial hidden state
            
        Returns:
            Tuple of (logits, hidden_state)
        """
        # Embedding
        embedded = self.embedding(input_ids)
        embedded = self.dropout(embedded)
        
        # LSTM forward pass
        lstm_output, hidden_state = self.lstm(embedded, hidden)
        
        # Self-attention over LSTM outputs
        attended_output, _ = self.attention(lstm_output, lstm_output, lstm_output)
        
        # Residual connection and layer normalization
        output = self.layer_norm(lstm_output + attended_output)
        output = self.dropout(output)
        
        # Project to vocabulary
        logits = self.output_projection(output)
        
        return logits, hidden_state
    
    def init_hidden(self, batch_size: int, device: torch.device) -> Tuple:
        """Initialize hidden states for LSTM."""
        h_0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim, device=device)
        c_0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim, device=device)
        return (h_0, c_0)


# Model factory function
def create_rnn_language_model(model_type: str, vocab_size: int, **kwargs) -> nn.Module:
    """
    Factory function to create RNN language models.
    
    Args:
        model_type: Type of model ('lstm', 'gru', 'bilstm', 'attention_lstm')
        vocab_size: Vocabulary size
        **kwargs: Additional model parameters
        
    Returns:
        Instantiated model
    """
    model_classes = {
        'lstm': LSTMLanguageModel,
        'gru': GRULanguageModel,
        'bilstm': BiLSTMLanguageModel,
        'attention_lstm': AttentionLSTMLanguageModel
    }
    
    if model_type not in model_classes:
        raise ValueError(f"Unsupported model type: {model_type}. "
                        f"Available types: {list(model_classes.keys())}")
    
    model_class = model_classes[model_type]
    return model_class(vocab_size, **kwargs)


if __name__ == '__main__':
    # Example usage and testing
    print("🧠 Testing RNN Language Models for Australian Tourism")
    print("=" * 60)
    
    vocab_size = 10000
    batch_size = 4
    seq_len = 32
    
    # Test LSTM model
    lstm_model = LSTMLanguageModel(vocab_size, embed_dim=256, hidden_dim=256)
    print(f"\n📊 LSTM Model Info:")
    for key, value in lstm_model.get_model_info().items():
        print(f"   {key}: {value}")
    
    # Test forward pass
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    hidden = lstm_model.init_hidden(batch_size, input_ids.device)
    
    logits, new_hidden = lstm_model(input_ids, hidden)
    print(f"\n🔄 Forward Pass Test:")
    print(f"   Input shape: {input_ids.shape}")
    print(f"   Output shape: {logits.shape}")
    print(f"   Hidden shape: {[h.shape for h in new_hidden]}")
    
    # Test text generation
    start_tokens = torch.randint(0, vocab_size, (1, 5))
    generated = lstm_model.generate(start_tokens, max_length=10)
    print(f"\n🎯 Generation Test:")
    print(f"   Start tokens: {start_tokens.tolist()}")
    print(f"   Generated: {generated.tolist()}")
    
    print("\n✅ All tests passed! Ready for Australian tourism language modeling!")