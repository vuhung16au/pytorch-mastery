"""
Word-level Language Modeling with RNN and Transformer architectures.

This package provides PyTorch implementations of language models for Australian
tourism corpus with English-Vietnamese multilingual support.
"""

from .rnn_language_model import (
    RNNLanguageModel,
    LSTMLanguageModel, 
    GRULanguageModel,
    BiLSTMLanguageModel,
    AttentionLSTMLanguageModel,
    create_rnn_language_model
)

from .transformer_language_model import (
    TransformerLanguageModel,
    PositionalEncoding,
    MultiHeadAttention,
    TransformerBlock,
    GPTStyleLanguageModel,
    create_transformer_language_model
)

from .utils import (
    detect_device,
    get_run_logdir,
    save_model_checkpoint,
    load_model_checkpoint,
    count_parameters,
    get_device_memory_info,
    initialize_weights,
    calculate_perplexity,
    generate_sample_text,
    AustralianTextSampler
)

__all__ = [
    # RNN Models
    'RNNLanguageModel',
    'LSTMLanguageModel', 
    'GRULanguageModel',
    'BiLSTMLanguageModel',
    'AttentionLSTMLanguageModel',
    'create_rnn_language_model',
    
    # Transformer Models
    'TransformerLanguageModel',
    'PositionalEncoding',
    'MultiHeadAttention',
    'TransformerBlock',
    'GPTStyleLanguageModel',
    'create_transformer_language_model',
    
    # Utilities
    'detect_device',
    'get_run_logdir',
    'save_model_checkpoint',
    'load_model_checkpoint',
    'count_parameters',
    'get_device_memory_info',
    'initialize_weights',
    'calculate_perplexity',
    'generate_sample_text',
    'AustralianTextSampler'
]

__version__ = '1.0.0'
__author__ = 'PyTorch Mastery - Australian Language Modeling'