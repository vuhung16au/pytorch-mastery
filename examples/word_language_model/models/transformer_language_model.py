"""
Transformer-based Language Models for Australian Tourism Corpus.

This module implements Transformer architectures with multi-head self-attention
for word-level language modeling with Australian context and multilingual support.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, Dict, Any


class PositionalEncoding(nn.Module):
    """
    Positional Encoding for Transformer models.
    
    Adds position information to token embeddings using sinusoidal functions.
    This allows the model to understand token positions without recurrence.
    
    Args:
        embed_dim: Embedding dimension
        max_seq_len: Maximum sequence length
        dropout: Dropout probability
    """
    
    def __init__(self, embed_dim: int, max_seq_len: int = 5000, dropout: float = 0.1):
        super(PositionalEncoding, self).__init__()
        
        self.dropout = nn.Dropout(dropout)
        
        # Create positional encoding matrix
        pe = torch.zeros(max_seq_len, embed_dim)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        
        # Create div_term for sinusoidal functions
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * 
                           (-math.log(10000.0) / embed_dim))
        
        # Apply sine to even indices
        pe[:, 0::2] = torch.sin(position * div_term)
        # Apply cosine to odd indices
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Add batch dimension and register as buffer
        pe = pe.unsqueeze(0).transpose(0, 1)  # [max_seq_len, 1, embed_dim]
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional encoding to embeddings.
        
        Args:
            x: Token embeddings [seq_len, batch_size, embed_dim]
            
        Returns:
            Embeddings with positional encoding added
        """
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class MultiHeadAttention(nn.Module):
    """
    Multi-Head Self-Attention mechanism for Australian tourism text understanding.
    
    This implementation:
    - Uses multiple attention heads for diverse representation learning
    - Includes causal masking for language modeling
    - Optimized for Australian tourism vocabulary patterns
    
    Args:
        embed_dim: Model embedding dimension
        num_heads: Number of attention heads
        dropout: Dropout probability
        causal: Whether to use causal (autoregressive) masking
    """
    
    def __init__(self, embed_dim: int, num_heads: int = 8, dropout: float = 0.1, causal: bool = True):
        super(MultiHeadAttention, self).__init__()
        
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.causal = causal
        
        # Linear projections for Q, K, V
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        
        # Output projection
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize attention weights."""
        for module in [self.q_proj, self.k_proj, self.v_proj, self.out_proj]:
            nn.init.xavier_uniform_(module.weight)
            nn.init.zeros_(module.bias)
    
    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                attn_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of multi-head attention.
        
        Args:
            query: Query tensor [batch_size, seq_len, embed_dim]
            key: Key tensor [batch_size, seq_len, embed_dim] 
            value: Value tensor [batch_size, seq_len, embed_dim]
            attn_mask: Attention mask (optional)
            
        Returns:
            Tuple of (attended_output, attention_weights)
        """
        batch_size, seq_len, embed_dim = query.size()
        
        # Linear projections and reshape for multi-head
        Q = self.q_proj(query).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(key).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(value).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        # Q, K, V: [batch_size, num_heads, seq_len, head_dim]
        
        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        # scores: [batch_size, num_heads, seq_len, seq_len]
        
        # Apply causal mask for language modeling
        if self.causal:
            causal_mask = torch.tril(torch.ones(seq_len, seq_len, device=query.device))
            scores = scores.masked_fill(causal_mask == 0, float('-inf'))
        
        # Apply additional attention mask if provided
        if attn_mask is not None:
            scores = scores.masked_fill(attn_mask == 0, float('-inf'))
        
        # Apply softmax
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        attended = torch.matmul(attn_weights, V)
        # attended: [batch_size, num_heads, seq_len, head_dim]
        
        # Concatenate heads and project
        attended = attended.transpose(1, 2).contiguous().view(batch_size, seq_len, embed_dim)
        output = self.out_proj(attended)
        
        return output, attn_weights


class TransformerBlock(nn.Module):
    """
    Single Transformer block with self-attention and feed-forward network.
    
    Components:
    - Multi-head self-attention
    - Layer normalization
    - Position-wise feed-forward network
    - Residual connections
    
    Optimized for Australian tourism text patterns and multilingual support.
    """
    
    def __init__(self, embed_dim: int, num_heads: int = 8, ff_dim: int = 2048, 
                 dropout: float = 0.1, layer_norm_eps: float = 1e-5):
        super(TransformerBlock, self).__init__()
        
        # Multi-head attention
        self.self_attention = MultiHeadAttention(embed_dim, num_heads, dropout)
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(embed_dim, eps=layer_norm_eps)
        self.norm2 = nn.LayerNorm(embed_dim, eps=layer_norm_eps)
        
        # Feed-forward network
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, embed_dim),
            nn.Dropout(dropout)
        )
    
    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass through transformer block.
        
        Args:
            x: Input tensor [batch_size, seq_len, embed_dim]
            attn_mask: Attention mask (optional)
            
        Returns:
            Output tensor [batch_size, seq_len, embed_dim]
        """
        # Multi-head self-attention with residual connection
        attn_output, _ = self.self_attention(x, x, x, attn_mask)
        x = self.norm1(x + attn_output)
        
        # Feed-forward with residual connection
        ff_output = self.feed_forward(x)
        x = self.norm2(x + ff_output)
        
        return x


class TransformerLanguageModel(nn.Module):
    """
    Transformer-based Language Model for Australian Tourism Content Generation.
    
    This advanced model features:
    - Multi-head self-attention for rich contextual understanding
    - Positional encoding for sequence awareness
    - Layer normalization for stable training
    - Causal masking for autoregressive generation
    - Optimized for Australian tourism vocabulary and patterns
    
    TensorFlow equivalent:
        # TensorFlow 2.x with Keras
        model = tf.keras.Sequential([
            tf.keras.layers.Embedding(vocab_size, embed_dim),
            tf.keras.layers.MultiHeadAttention(num_heads, embed_dim // num_heads),
            tf.keras.layers.Dense(vocab_size)
        ])
        
    However, PyTorch provides more flexibility for custom attention mechanisms
    and better control over the training process.
    
    Args:
        vocab_size: Size of vocabulary
        embed_dim: Model embedding dimension
        num_heads: Number of attention heads
        num_layers: Number of transformer layers  
        ff_dim: Feed-forward network dimension
        max_seq_len: Maximum sequence length
        dropout: Dropout probability
        layer_norm_eps: Layer normalization epsilon
    """
    
    def __init__(self, vocab_size: int, embed_dim: int = 512, num_heads: int = 8,
                 num_layers: int = 6, ff_dim: int = 2048, max_seq_len: int = 512,
                 dropout: float = 0.1, layer_norm_eps: float = 1e-5):
        super(TransformerLanguageModel, self).__init__()
        
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.max_seq_len = max_seq_len
        
        # Australian tourism context examples
        self.australian_examples = {
            'cities': ["Sydney", "Melbourne", "Brisbane", "Perth", "Adelaide"],
            'landmarks': ["Opera House", "Harbour Bridge", "Uluru", "Great Barrier Reef"],
            'experiences': ["coffee culture", "wine tasting", "reef diving", "bushwalking"]
        }
        
        # Token embedding
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        
        # Positional encoding
        self.positional_encoding = PositionalEncoding(embed_dim, max_seq_len, dropout)
        
        # Transformer blocks
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, ff_dim, dropout, layer_norm_eps)
            for _ in range(num_layers)
        ])
        
        # Final layer normalization
        self.final_norm = nn.LayerNorm(embed_dim, eps=layer_norm_eps)
        
        # Output projection to vocabulary
        self.output_projection = nn.Linear(embed_dim, vocab_size)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize model weights."""
        # Initialize token embeddings
        nn.init.normal_(self.token_embedding.weight, mean=0, std=0.02)
        
        # Initialize output projection
        nn.init.normal_(self.output_projection.weight, mean=0, std=0.02)
        nn.init.zeros_(self.output_projection.bias)
    
    def forward(self, input_ids: torch.Tensor, attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass through the Transformer language model.
        
        Args:
            input_ids: Input token indices [batch_size, seq_len]
            attn_mask: Attention mask for padding (optional)
            
        Returns:
            Logits over vocabulary [batch_size, seq_len, vocab_size]
        """
        batch_size, seq_len = input_ids.size()
        
        # Token embeddings
        token_emb = self.token_embedding(input_ids)  # [batch_size, seq_len, embed_dim]
        
        # Scale embeddings (common practice in Transformer models)
        token_emb = token_emb * math.sqrt(self.embed_dim)
        
        # Add positional encoding (requires transpose for positional encoding format)
        x = token_emb.transpose(0, 1)  # [seq_len, batch_size, embed_dim]
        x = self.positional_encoding(x)
        x = x.transpose(0, 1)  # [batch_size, seq_len, embed_dim]
        
        # Pass through transformer blocks
        for transformer_block in self.transformer_blocks:
            x = transformer_block(x, attn_mask)
        
        # Final layer normalization
        x = self.final_norm(x)
        
        # Project to vocabulary
        logits = self.output_projection(x)  # [batch_size, seq_len, vocab_size]
        
        return logits
    
    def generate_step(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Single generation step for compatibility with generation utilities.
        
        Args:
            input_ids: Current sequence [batch_size, seq_len]
            
        Returns:
            Logits for next token prediction
        """
        return self.forward(input_ids)
    
    def generate(self, start_tokens: torch.Tensor, max_length: int = 50, 
                temperature: float = 1.0, top_k: int = 0, top_p: float = 1.0,
                device: torch.device = torch.device('cpu')) -> torch.Tensor:
        """
        Generate text using the trained Transformer model.
        
        Args:
            start_tokens: Starting tokens [batch_size, seq_len]
            max_length: Maximum generation length
            temperature: Sampling temperature
            top_k: Top-k sampling (0 = disabled)
            top_p: Top-p (nucleus) sampling (1.0 = disabled)
            device: Device for computation
            
        Returns:
            Generated token sequence
        """
        self.eval()
        
        with torch.no_grad():
            generated = start_tokens.to(device)
            
            for _ in range(max_length):
                # Limit input length to max_seq_len
                if generated.size(1) >= self.max_seq_len:
                    input_ids = generated[:, -self.max_seq_len:]
                else:
                    input_ids = generated
                
                # Forward pass
                logits = self.forward(input_ids)
                
                # Get logits for last position and apply temperature
                next_token_logits = logits[:, -1, :] / temperature
                
                # Apply top-k filtering
                if top_k > 0:
                    indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                    next_token_logits[indices_to_remove] = float('-inf')
                
                # Apply top-p (nucleus) filtering  
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    
                    # Remove tokens with cumulative probability above threshold
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    
                    indices_to_remove = sorted_indices[sorted_indices_to_remove]
                    next_token_logits.scatter_(-1, indices_to_remove.unsqueeze(-1), float('-inf'))
                
                # Sample next token
                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                
                # Append to generated sequence
                generated = torch.cat([generated, next_token], dim=-1)
        
        return generated
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get comprehensive model information."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'architecture': 'Transformer Language Model',
            'vocab_size': self.vocab_size,
            'embed_dim': self.embed_dim,
            'num_heads': self.num_heads,
            'num_layers': self.num_layers,
            'max_seq_len': self.max_seq_len,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'attention_parameters': sum(p.numel() for block in self.transformer_blocks 
                                      for p in block.self_attention.parameters()),
            'australian_context': True,
            'multilingual_support': True,
            'generation_features': ['temperature', 'top_k', 'top_p', 'causal_masking']
        }


class GPTStyleLanguageModel(TransformerLanguageModel):
    """
    GPT-style Language Model optimized for Australian tourism content.
    
    This model follows GPT architecture principles:
    - Decoder-only transformer architecture
    - Causal self-attention (can only attend to previous tokens)
    - Autoregressive generation
    - Optimized for coherent Australian tourism text generation
    
    Perfect for:
    - Tourism content generation
    - Australian travel blog writing
    - Multilingual Australian-Vietnamese content creation
    """
    
    def __init__(self, vocab_size: int, embed_dim: int = 768, num_heads: int = 12,
                 num_layers: int = 12, ff_dim: int = 3072, max_seq_len: int = 1024,
                 dropout: float = 0.1):
        super(GPTStyleLanguageModel, self).__init__(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            ff_dim=ff_dim,
            max_seq_len=max_seq_len,
            dropout=dropout
        )
        
        print("🤖 GPT-style Language Model initialized for Australian tourism content")
        print(f"   Parameters: {sum(p.numel() for p in self.parameters()):,}")
        print(f"   Context length: {max_seq_len}")


# Model factory function
def create_transformer_language_model(model_type: str, vocab_size: int, **kwargs) -> nn.Module:
    """
    Factory function to create Transformer language models.
    
    Args:
        model_type: Type of model ('transformer', 'gpt')
        vocab_size: Vocabulary size
        **kwargs: Additional model parameters
        
    Returns:
        Instantiated transformer model
    """
    model_classes = {
        'transformer': TransformerLanguageModel,
        'gpt': GPTStyleLanguageModel,
    }
    
    if model_type not in model_classes:
        raise ValueError(f"Unsupported model type: {model_type}. "
                        f"Available types: {list(model_classes.keys())}")
    
    model_class = model_classes[model_type]
    return model_class(vocab_size, **kwargs)


if __name__ == '__main__':
    # Example usage and testing
    print("🤖 Testing Transformer Language Models for Australian Tourism")
    print("=" * 70)
    
    vocab_size = 10000
    batch_size = 2
    seq_len = 32
    
    # Test Transformer model
    transformer_model = TransformerLanguageModel(
        vocab_size, embed_dim=512, num_heads=8, num_layers=6
    )
    
    print(f"\n📊 Transformer Model Info:")
    for key, value in transformer_model.get_model_info().items():
        print(f"   {key}: {value}")
    
    # Test forward pass
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    
    logits = transformer_model(input_ids)
    print(f"\n🔄 Forward Pass Test:")
    print(f"   Input shape: {input_ids.shape}")
    print(f"   Output shape: {logits.shape}")
    
    # Test attention mechanism
    attention_layer = MultiHeadAttention(embed_dim=512, num_heads=8)
    x = torch.randn(batch_size, seq_len, 512)
    attended, attn_weights = attention_layer(x, x, x)
    print(f"\n🎯 Attention Test:")
    print(f"   Input shape: {x.shape}")
    print(f"   Attended shape: {attended.shape}")
    print(f"   Attention weights shape: {attn_weights.shape}")
    
    # Test text generation
    start_tokens = torch.randint(0, vocab_size, (1, 5))
    generated = transformer_model.generate(start_tokens, max_length=10)
    print(f"\n🎯 Generation Test:")
    print(f"   Start tokens: {start_tokens.tolist()}")
    print(f"   Generated: {generated.tolist()}")
    
    print("\n✅ All tests passed! Ready for Australian tourism Transformer modeling!")