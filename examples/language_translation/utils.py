"""
Utility functions for PyTorch language translation implementation.

This module provides helper functions for device detection, training utilities,
evaluation metrics, and visualization following repository standards.
"""

import torch
import torch.nn.functional as F
import platform
import time
import os
from typing import List, Dict, Tuple, Optional, Union
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
import tempfile

# Set seaborn style for better notebook aesthetics
sns.set_style("whitegrid")
sns.set_palette("husl")

def detect_device():
    """
    Detect the best available PyTorch device with comprehensive hardware support.
    
    Priority order:
    1. CUDA (NVIDIA GPUs) - Best performance for deep learning
    2. MPS (Apple Silicon) - Optimized for M1/M2/M3 Macs  
    3. CPU (Universal) - Always available fallback
    
    Returns:
        torch.device: The optimal device for PyTorch operations
        str: Human-readable device description for logging
    """
    # Check for CUDA (NVIDIA GPU)
    if torch.cuda.is_available():
        device = torch.device("cuda")
        gpu_name = torch.cuda.get_device_name(0)
        device_info = f"CUDA GPU: {gpu_name}"
        
        # Additional CUDA info for optimization
        cuda_version = torch.version.cuda
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        
        print(f"🚀 Using CUDA acceleration")
        print(f"   GPU: {gpu_name}")
        print(f"   CUDA Version: {cuda_version}")
        print(f"   GPU Memory: {gpu_memory:.1f} GB")
        
        return device, device_info
    
    # Check for MPS (Apple Silicon)
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device("mps")
        device_info = "Apple Silicon MPS"
        
        # Get system info for Apple Silicon
        system_info = platform.uname()
        
        print(f"🍎 Using Apple Silicon MPS acceleration")
        print(f"   System: {system_info.system} {system_info.release}")
        print(f"   Machine: {system_info.machine}")
        print(f"   Processor: {system_info.processor}")
        
        return device, device_info
    
    # Fallback to CPU
    else:
        device = torch.device("cpu")
        device_info = "CPU (No GPU acceleration available)"
        
        # Get CPU info for optimization guidance
        cpu_count = torch.get_num_threads()
        system_info = platform.uname()
        
        print(f"💻 Using CPU (no GPU acceleration detected)")
        print(f"   Processor: {system_info.processor}")
        print(f"   PyTorch Threads: {cpu_count}")
        print(f"   System: {system_info.system} {system_info.release}")
        
        # Provide optimization suggestions for CPU-only setups
        print(f"\n💡 CPU Optimization Tips:")
        print(f"   • Reduce batch size to prevent memory issues")
        print(f"   • Consider using smaller models for faster training")
        print(f"   • Enable PyTorch optimizations: torch.set_num_threads({cpu_count})")
        
        return device, device_info

def get_run_logdir(base_name: str = "translation_training") -> str:
    """
    Generate unique log directory for TensorBoard with timestamp.
    
    Following repository standards for platform-specific log directories.
    
    Args:
        base_name: Base name for the log directory
        
    Returns:
        str: Full path to the log directory
        
    Example:
        >>> log_dir = get_run_logdir("seq2seq_translation")
        >>> print(log_dir)  # './tensorboard_logs/seq2seq_translation_2024_01_15-14_30_22'
    """
    import sys
    
    # Platform-specific log directory setup
    if 'google.colab' in str(get_ipython() if 'get_ipython' in globals() else ''):
        # Google Colab: Save logs to /content/tensorboard_logs
        root_logdir = "/content/tensorboard_logs"
    elif 'kaggle' in os.environ.get('KAGGLE_URL_BASE', ''):
        # Kaggle: Save logs to ./tensorboard_logs/
        root_logdir = "./tensorboard_logs"
    else:
        # Local: Save logs to ./tensorboard_logs/
        root_logdir = "./tensorboard_logs"
    
    # Create timestamp
    timestamp = time.strftime("%Y_%m_%d-%H_%M_%S")
    
    # Create unique directory name
    log_dir = f"{root_logdir}/{base_name}_{timestamp}"
    
    # Ensure directory exists
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    
    return log_dir

def safe_to_device(tensor: torch.Tensor, device: torch.device, tensor_name: str = "tensor") -> torch.Tensor:
    """
    Safely move tensor to device with error handling.
    
    Args:
        tensor: Input tensor
        device: Target device
        tensor_name: Name of tensor for error reporting
        
    Returns:
        torch.Tensor: Tensor moved to device
        
    Raises:
        RuntimeError: If device transfer fails
    """
    try:
        return tensor.to(device)
    except RuntimeError as e:
        if "CUDA out of memory" in str(e):
            print(f"❌ CUDA out of memory for {tensor_name}")
            print(f"   Tensor shape: {tensor.shape}")
            print(f"   Tensor dtype: {tensor.dtype}")
            print(f"💡 Try reducing batch size or using gradient accumulation")
            torch.cuda.empty_cache()
            raise e
        elif "MPS" in str(e):
            print(f"⚠️  MPS error for {tensor_name}, falling back to CPU")
            return tensor.to('cpu')
        else:
            print(f"❌ Device error for {tensor_name}: {e}")
            raise e

def calculate_bleu_score(predicted_sentences: List[str], reference_sentences: List[str]) -> float:
    """
    Calculate BLEU score for translation evaluation.
    
    Simplified implementation for educational purposes.
    In production, use established libraries like sacrebleu or nltk.
    
    Args:
        predicted_sentences: Model-generated translations
        reference_sentences: Ground truth translations
        
    Returns:
        float: BLEU score (0.0 to 1.0, higher is better)
        
    Example:
        >>> predictions = ["The Sydney Opera House is beautiful", "Melbourne has great coffee"]
        >>> references = ["Sydney Opera House is beautiful", "Melbourne has excellent coffee"]
        >>> bleu = calculate_bleu_score(predictions, references)
        >>> print(f"BLEU Score: {bleu:.3f}")
    """
    if len(predicted_sentences) != len(reference_sentences):
        raise ValueError("Number of predictions must match number of references")
    
    total_bleu = 0.0
    
    for pred, ref in zip(predicted_sentences, reference_sentences):
        # Tokenize sentences
        pred_tokens = pred.lower().split()
        ref_tokens = ref.lower().split()
        
        if len(pred_tokens) == 0:
            continue
            
        # Calculate precision for n-grams (n=1,2,3,4)
        precisions = []
        for n in range(1, 5):
            pred_ngrams = get_ngrams(pred_tokens, n)
            ref_ngrams = get_ngrams(ref_tokens, n)
            
            if len(pred_ngrams) == 0:
                precisions.append(0.0)
                continue
                
            matches = sum(min(pred_ngrams.get(ng, 0), ref_ngrams.get(ng, 0)) 
                         for ng in pred_ngrams)
            precision = matches / len(pred_tokens) if len(pred_tokens) > 0 else 0.0
            precisions.append(precision)
        
        # Calculate brevity penalty
        pred_len = len(pred_tokens)
        ref_len = len(ref_tokens)
        
        if pred_len >= ref_len:
            bp = 1.0
        else:
            bp = np.exp(1 - ref_len / pred_len) if pred_len > 0 else 0.0
        
        # Calculate BLEU score for this sentence
        if all(p > 0 for p in precisions):
            bleu = bp * np.exp(np.mean(np.log(precisions)))
        else:
            bleu = 0.0
            
        total_bleu += bleu
    
    return total_bleu / len(predicted_sentences) if predicted_sentences else 0.0

def get_ngrams(tokens: List[str], n: int) -> Dict[Tuple[str, ...], int]:
    """Extract n-grams from tokens."""
    ngrams = {}
    for i in range(len(tokens) - n + 1):
        ngram = tuple(tokens[i:i+n])
        ngrams[ngram] = ngrams.get(ngram, 0) + 1
    return ngrams

def plot_training_metrics(train_losses: List[float], 
                         val_losses: List[float], 
                         bleu_scores: List[float],
                         save_path: Optional[str] = None):
    """
    Plot training metrics with seaborn styling for Australian tourism model.
    
    Args:
        train_losses: Training loss values per epoch
        val_losses: Validation loss values per epoch  
        bleu_scores: BLEU scores per epoch
        save_path: Optional path to save the plot
        
    Example:
        >>> plot_training_metrics([2.1, 1.8, 1.5], [2.3, 1.9, 1.6], [0.15, 0.22, 0.28])
    """
    epochs = range(1, len(train_losses) + 1)
    
    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Loss plot
    ax1.plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2)
    ax1.plot(epochs, val_losses, 'r-', label='Validation Loss', linewidth=2)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('English-Vietnamese Translation - Training Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # BLEU score plot  
    ax2.plot(epochs, bleu_scores, 'g-', label='BLEU Score', linewidth=2, marker='o')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('BLEU Score')
    ax2.set_title('Translation Quality - BLEU Score Progress')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Improve layout
    plt.tight_layout()
    
    # Save plot if path provided
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"📊 Plot saved to {save_path}")
    
    plt.show()

def visualize_attention(attention_weights: torch.Tensor,
                       source_tokens: List[str],
                       target_tokens: List[str],
                       save_path: Optional[str] = None):
    """
    Visualize attention weights as a heatmap.
    
    Args:
        attention_weights: Attention weights tensor [tgt_len, src_len]
        source_tokens: List of source tokens
        target_tokens: List of target tokens
        save_path: Optional path to save the plot
        
    Example:
        >>> # Attention weights from model
        >>> attention = torch.rand(6, 8)  # [target_len, source_len]
        >>> src_tokens = ["Sydney", "Opera", "House", "is", "beautiful", "<eos>"]
        >>> tgt_tokens = ["Nhà", "hát", "Opera", "Sydney", "đẹp", "<eos>"]
        >>> visualize_attention(attention, src_tokens, tgt_tokens)
    """
    # Convert to numpy if tensor
    if torch.is_tensor(attention_weights):
        attention_weights = attention_weights.detach().cpu().numpy()
    
    # Create heatmap
    plt.figure(figsize=(10, 8))
    
    # Use seaborn for better aesthetics
    sns.heatmap(
        attention_weights,
        xticklabels=source_tokens,
        yticklabels=target_tokens,
        cmap='Blues',
        annot=True,
        fmt='.2f',
        cbar_kws={'label': 'Attention Weight'}
    )
    
    plt.title('Translation Attention Visualization\n(English → Vietnamese)')
    plt.xlabel('Source Tokens (English)')
    plt.ylabel('Target Tokens (Vietnamese)')
    
    # Rotate labels for better readability
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    plt.tight_layout()
    
    # Save plot if path provided
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"🎯 Attention plot saved to {save_path}")
    
    plt.show()

def count_parameters(model: torch.nn.Module) -> Dict[str, int]:
    """
    Count total and trainable parameters in a PyTorch model.
    
    Args:
        model: PyTorch model
        
    Returns:
        Dict with parameter counts
        
    Example:
        >>> from translation_models import Seq2SeqTranslator
        >>> model = Seq2SeqTranslator(1000, 800, 256)
        >>> params = count_parameters(model)
        >>> print(f"Total parameters: {params['total']:,}")
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        'total': total_params,
        'trainable': trainable_params,
        'non_trainable': total_params - trainable_params
    }

def print_model_info(model: torch.nn.Module, model_name: str = "Model"):
    """Print comprehensive model information."""
    params = count_parameters(model)
    
    print(f"📋 {model_name} Information:")
    print(f"   🔢 Total parameters: {params['total']:,}")
    print(f"   🎯 Trainable parameters: {params['trainable']:,}")
    print(f"   🔒 Non-trainable parameters: {params['non_trainable']:,}")
    print(f"   💾 Model size: ~{params['total'] * 4 / 1024**2:.1f} MB (fp32)")
    
def save_translation_examples(source_texts: List[str],
                             target_texts: List[str],
                             predicted_texts: List[str],
                             scores: List[float],
                             save_path: str):
    """
    Save translation examples to file for analysis.
    
    Args:
        source_texts: Original English texts
        target_texts: Reference Vietnamese translations  
        predicted_texts: Model predictions
        scores: Individual BLEU scores
        save_path: Path to save results
    """
    with open(save_path, 'w', encoding='utf-8') as f:
        f.write("English-Vietnamese Translation Results\n")
        f.write("=" * 50 + "\n\n")
        
        for i, (src, ref, pred, score) in enumerate(zip(source_texts, target_texts, predicted_texts, scores)):
            f.write(f"Example {i+1} (BLEU: {score:.3f}):\n")
            f.write(f"English:    {src}\n")
            f.write(f"Reference:  {ref}\n") 
            f.write(f"Predicted:  {pred}\n")
            f.write("-" * 40 + "\n\n")
    
    print(f"📝 Translation examples saved to {save_path}")

def format_time(seconds: float) -> str:
    """Format time duration for display."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds // 60
        seconds = seconds % 60
        return f"{int(minutes)}m {seconds:.0f}s"
    else:
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        return f"{int(hours)}h {int(minutes)}m"

def test_device_functionality():
    """Test basic PyTorch operations on the detected device."""
    device, device_info = detect_device()
    
    print(f"🧪 Testing device functionality: {device}")
    
    try:
        # Test basic tensor operations
        x = torch.randn(100, 100).to(device)
        y = torch.randn(100, 100).to(device)
        z = torch.matmul(x, y)
        
        # Test neural network operations
        layer = torch.nn.Linear(100, 50).to(device)
        output = layer(x)
        
        # Test gradient computation
        loss = output.mean()
        loss.backward()
        
        print(f"✅ Device {device} is working correctly!")
        print(f"   Matrix multiplication: {z.shape}")
        print(f"   Neural network forward: {output.shape}")
        print(f"   Gradient computation: ✓")
        
        return True
        
    except Exception as e:
        print(f"❌ Device {device} test failed: {e}")
        print(f"💡 Consider falling back to CPU or checking device compatibility")
        return False

if __name__ == "__main__":
    print("🔧 Testing Translation Utilities")
    print("=" * 40)
    
    # Test device detection
    device_test_passed = test_device_functionality()
    
    # Test BLEU score calculation
    print(f"\n📊 Testing BLEU Score Calculation:")
    predictions = [
        "Sydney Opera House is beautiful",
        "Melbourne has great coffee culture"
    ]
    references = [
        "The Sydney Opera House is beautiful", 
        "Melbourne has excellent coffee culture"
    ]
    
    bleu = calculate_bleu_score(predictions, references)
    print(f"   BLEU Score: {bleu:.3f}")
    
    # Test parameter counting
    print(f"\n🔢 Testing Parameter Counting:")
    test_model = torch.nn.Sequential(
        torch.nn.Linear(100, 50),
        torch.nn.ReLU(),
        torch.nn.Linear(50, 10)
    )
    print_model_info(test_model, "Test Model")
    
    print(f"\n✅ All utility tests completed!")