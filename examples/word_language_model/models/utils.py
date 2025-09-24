"""
Model utilities and helper functions for word-level language modeling.

This module provides device detection, model loading/saving, and other
utility functions following the PyTorch Mastery repository standards.
"""

import torch
import torch.nn as nn
import platform
import os
import time
from typing import Dict, Any, Optional, Tuple
from pathlib import Path


def detect_device() -> Tuple[torch.device, str]:
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


def get_run_logdir(base_name: str = "language_model") -> str:
    """
    Generate unique TensorBoard log directory with timestamp.
    
    Args:
        base_name: Base name for the log directory
        
    Returns:
        Path to log directory
    """
    # Platform-specific TensorBoard log directory setup
    import sys
    
    IS_COLAB = 'google.colab' in str(globals().get('get_ipython', lambda: '')())
    IS_KAGGLE = 'kaggle' in os.environ.get('KAGGLE_URL_BASE', '')
    IS_LOCAL = not (IS_COLAB or IS_KAGGLE)
    
    if IS_COLAB:
        # Google Colab: Save logs to /content/tensorboard_logs
        root_logdir = "/content/tensorboard_logs"
    elif IS_KAGGLE:
        # Kaggle: Save logs to ./tensorboard_logs/
        root_logdir = "./tensorboard_logs"
    else:
        # Local: Save logs to ./tensorboard_logs/
        root_logdir = "./tensorboard_logs"
    
    # Create timestamped directory
    timestamp = time.strftime("%Y_%m_%d-%H_%M_%S")
    run_logdir = os.path.join(root_logdir, f"{base_name}_{timestamp}")
    
    # Ensure directory exists
    os.makedirs(run_logdir, exist_ok=True)
    
    return run_logdir


def save_model_checkpoint(model: nn.Module, optimizer: torch.optim.Optimizer, 
                         epoch: int, loss: float, save_path: str, 
                         vocab_size: int, config: Dict[str, Any]):
    """
    Save model checkpoint with training state and configuration.
    
    Args:
        model: PyTorch model
        optimizer: Optimizer state
        epoch: Current epoch
        loss: Current loss value
        save_path: Path to save checkpoint
        vocab_size: Vocabulary size
        config: Model configuration
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'vocab_size': vocab_size,
        'config': config,
        'model_class': model.__class__.__name__
    }
    
    torch.save(checkpoint, save_path)
    print(f"📁 Model checkpoint saved to {save_path}")


def load_model_checkpoint(model: nn.Module, optimizer: torch.optim.Optimizer,
                         load_path: str, device: torch.device) -> Tuple[int, float]:
    """
    Load model checkpoint and restore training state.
    
    Args:
        model: PyTorch model (architecture should match saved model)
        optimizer: Optimizer to restore state
        load_path: Path to checkpoint file
        device: Device to load model on
        
    Returns:
        Tuple of (start_epoch, best_loss)
    """
    checkpoint = torch.load(load_path, map_location=device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    start_epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    
    print(f"📁 Model checkpoint loaded from {load_path}")
    print(f"   Resuming from epoch {start_epoch}, loss: {loss:.4f}")
    
    return start_epoch, loss


def count_parameters(model: nn.Module) -> Dict[str, int]:
    """
    Count trainable and total parameters in the model.
    
    Args:
        model: PyTorch model
        
    Returns:
        Dictionary with parameter counts
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        'total_parameters': total_params,
        'trainable_parameters': trainable_params,
        'non_trainable_parameters': total_params - trainable_params
    }


def get_device_memory_info(device: torch.device) -> str:
    """Get memory information for the current device."""
    if device.type == 'cuda':
        allocated = torch.cuda.memory_allocated(device) / 1024**3
        reserved = torch.cuda.memory_reserved(device) / 1024**3
        return f"GPU {allocated:.1f}GB/{reserved:.1f}GB"
    elif device.type == 'mps':
        return "MPS (memory monitoring not available)"
    else:
        try:
            import psutil
            memory = psutil.virtual_memory()
            return f"RAM {memory.percent}% used"
        except ImportError:
            return "RAM (monitoring not available)"


def initialize_weights(model: nn.Module, initialization: str = 'xavier_uniform'):
    """
    Initialize model weights using specified initialization scheme.
    
    Args:
        model: PyTorch model
        initialization: Initialization scheme ('xavier_uniform', 'xavier_normal', 'kaiming_uniform', 'kaiming_normal')
    """
    init_func = {
        'xavier_uniform': nn.init.xavier_uniform_,
        'xavier_normal': nn.init.xavier_normal_,
        'kaiming_uniform': nn.init.kaiming_uniform_,
        'kaiming_normal': nn.init.kaiming_normal_
    }.get(initialization, nn.init.xavier_uniform_)
    
    for name, param in model.named_parameters():
        if 'weight' in name and param.dim() > 1:
            init_func(param)
        elif 'bias' in name:
            nn.init.zeros_(param)
    
    print(f"🎯 Model weights initialized using {initialization}")


def calculate_perplexity(loss: float) -> float:
    """
    Calculate perplexity from cross-entropy loss.
    
    Args:
        loss: Cross-entropy loss value
        
    Returns:
        Perplexity value
    """
    return torch.exp(torch.tensor(loss)).item()


def generate_sample_text(model: nn.Module, vocabulary: Any, prompt: str, 
                        max_length: int = 50, temperature: float = 1.0,
                        top_k: int = 0, top_p: float = 1.0, 
                        device: torch.device = torch.device('cpu'),
                        language: str = 'en') -> str:
    """
    Generate text using the trained language model.
    
    Args:
        model: Trained language model
        vocabulary: Vocabulary object
        prompt: Starting text prompt
        max_length: Maximum generation length
        temperature: Sampling temperature (higher = more random)
        top_k: Top-k sampling (0 = disabled)
        top_p: Top-p (nucleus) sampling (1.0 = disabled)
        device: Device to run inference on
        language: Language code ('en' or 'vi')
        
    Returns:
        Generated text string
    """
    model.eval()
    
    # Encode prompt
    input_ids = vocabulary.encode(prompt, language, add_special_tokens=True)
    input_ids = torch.tensor(input_ids, dtype=torch.long).unsqueeze(0).to(device)
    
    generated = input_ids
    
    with torch.no_grad():
        for _ in range(max_length):
            # Forward pass
            if hasattr(model, 'generate_step'):
                # For transformer models with generate_step method
                outputs = model.generate_step(generated)
            else:
                # For RNN models
                outputs = model(generated)
            
            # Get logits for next token prediction
            if isinstance(outputs, tuple):
                logits = outputs[0]  # For models that return (logits, hidden)
            else:
                logits = outputs
            
            # Apply temperature scaling
            logits = logits[:, -1, :] / temperature
            
            # Apply top-k filtering
            if top_k > 0:
                indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
                logits[indices_to_remove] = float('-inf')
            
            # Apply top-p (nucleus) filtering
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                
                # Remove tokens with cumulative probability above the threshold
                sorted_indices_to_remove = cumulative_probs > top_p
                # Keep the first token above the threshold
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                
                indices_to_remove = sorted_indices[sorted_indices_to_remove]
                logits.scatter_(-1, indices_to_remove.unsqueeze(-1), float('-inf'))
            
            # Sample next token
            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            # Check for EOS token
            if next_token.item() == vocabulary.word2idx.get(vocabulary.EOS_TOKEN, -1):
                break
            
            # Append to generated sequence
            generated = torch.cat([generated, next_token], dim=-1)
    
    # Decode generated sequence
    generated_text = vocabulary.decode(generated[0].tolist(), remove_special_tokens=True)
    
    return generated_text


class AustralianTextSampler:
    """
    Utility class for generating Australian tourism-themed text samples.
    """
    
    def __init__(self):
        self.english_prompts = [
            "Sydney Opera House",
            "Melbourne coffee culture",
            "Great Barrier Reef",
            "Uluru at sunset",
            "Brisbane river cruise",
            "Perth beaches",
            "Tasmania wilderness",
            "Adelaide wine regions",
            "Darwin tropical climate",
            "Canberra parliament house"
        ]
        
        self.vietnamese_prompts = [
            "Nhà hát Opera Sydney",
            "Văn hóa cà phê Melbourne",
            "Rạn san hô Great Barrier Reef",
            "Uluru lúc hoàng hôn",
            "Du nuyền sông Brisbane",
            "Bãi biển Perth",
            "Vùng hoang dã Tasmania",
            "Vùng rượu vang Adelaide",
            "Khí hậu nhiệt đới Darwin",
            "Tòa nhà quốc hội Canberra"
        ]
    
    def get_random_prompt(self, language: str = 'en') -> str:
        """Get a random prompt for text generation."""
        import random
        
        if language == 'en':
            return random.choice(self.english_prompts)
        elif language == 'vi':
            return random.choice(self.vietnamese_prompts)
        else:
            return random.choice(self.english_prompts + self.vietnamese_prompts)
    
    def get_all_prompts(self, language: str = 'en') -> list:
        """Get all prompts for a given language."""
        if language == 'en':
            return self.english_prompts.copy()
        elif language == 'vi':
            return self.vietnamese_prompts.copy()
        else:
            return self.english_prompts + self.vietnamese_prompts