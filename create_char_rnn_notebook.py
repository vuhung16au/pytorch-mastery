#!/usr/bin/env python3
"""
Create the Character-Level RNN notebook for name classification
"""

import json
import os

def create_char_rnn_notebook():
    """Create comprehensive character-level RNN notebook."""
    
    # Notebook metadata
    notebook = {
        "cells": [],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "codemirror_mode": {
                    "name": "ipython",
                    "version": 3
                },
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.12.3"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }
    
    # Cell 1: Title and Header
    notebook["cells"].append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "# NLP From Scratch: Classifying Names with a Character-Level RNN 🇦🇺\n",
            "\n",
            "[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/vuhung16au/pytorch-mastery/blob/main/examples/pytorch-nlp/classify-names-character-level-RNN.ipynb)\n",
            "[![View on GitHub](https://img.shields.io/badge/View_on-GitHub-blue?logo=github)](https://github.com/vuhung16au/pytorch-mastery/blob/main/examples/pytorch-nlp/classify-names-character-level-RNN.ipynb)\n",
            "\n",
            "A comprehensive introduction to character-level Recurrent Neural Networks (RNNs) for name classification using PyTorch, featuring Australian names and locations with Vietnamese multilingual support.\n",
            "\n",
            "## Learning Objectives\n",
            "\n",
            "By the end of this notebook, you will:\n",
            "\n",
            "- 🔤 **Master character-level text processing** with PyTorch\n",
            "- 🧠 **Build RNN from scratch** for sequence classification\n",
            "- 🇦🇺 **Classify Australian names and locations** by origin/type\n",
            "- 🌏 **Handle multilingual text** with English-Vietnamese examples\n",
            "- 🔄 **Compare with TensorFlow** approaches for RNN implementation\n",
            "- 📊 **Implement comprehensive logging** with TensorBoard\n",
            "\n",
            "## What You'll Build\n",
            "\n",
            "1. **Australian Name Origin Classifier** - Classify names by ethnic origin (English, Irish, Greek, Vietnamese, etc.)\n",
            "2. **Location Type Classifier** - Distinguish between cities, suburbs, landmarks, and natural features\n",
            "3. **Character-level RNN Architecture** - Build vanilla RNN, LSTM, and GRU variants\n",
            "4. **Multilingual Support** - Handle both English and Vietnamese character sets\n",
            "\n",
            "---"
        ]
    })
    
    # Cell 2: Environment Detection and Setup
    notebook["cells"].append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Environment Detection and Setup\n",
            "import sys\n",
            "import subprocess\n",
            "import os\n",
            "import time\n",
            "\n",
            "# Detect the runtime environment\n",
            "IS_COLAB = \"google.colab\" in sys.modules\n",
            "IS_KAGGLE = \"kaggle_secrets\" in sys.modules or \"kaggle\" in os.environ.get('KAGGLE_URL_BASE', '')\n",
            "IS_LOCAL = not (IS_COLAB or IS_KAGGLE)\n",
            "\n",
            "print(f\"Environment detected:\")\n",
            "print(f\"  - Local: {IS_LOCAL}\")\n",
            "print(f\"  - Google Colab: {IS_COLAB}\")\n",
            "print(f\"  - Kaggle: {IS_KAGGLE}\")\n",
            "\n",
            "# Platform-specific system setup\n",
            "if IS_COLAB:\n",
            "    print(\"\\nSetting up Google Colab environment...\")\n",
            "    !apt update -qq\n",
            "    !apt install -y -qq software-properties-common\n",
            "elif IS_KAGGLE:\n",
            "    print(\"\\nSetting up Kaggle environment...\")\n",
            "    # Kaggle usually has most packages pre-installed\n",
            "else:\n",
            "    print(\"\\nSetting up local environment...\")"
        ]
    })
    
    # Cell 3: Package Installation
    notebook["cells"].append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Install required packages for this notebook\n",
            "required_packages = [\n",
            "    \"torch\",\n",
            "    \"pandas\",\n",
            "    \"seaborn\",\n",
            "    \"matplotlib\",\n",
            "    \"tensorboard\",\n",
            "    \"scikit-learn\",\n",
            "    \"numpy\"\n",
            "]\n",
            "\n",
            "print(\"Installing required packages...\")\n",
            "for package in required_packages:\n",
            "    if IS_COLAB or IS_KAGGLE:\n",
            "        !pip install -q {package}\n",
            "    else:\n",
            "        try:\n",
            "            subprocess.run([sys.executable, \"-m\", \"pip\", \"install\", \"-q\", package],\n",
            "                          capture_output=True, check=True)\n",
            "        except subprocess.CalledProcessError:\n",
            "            print(f\"Note: {package} installation skipped (likely already installed)\")\n",
            "    print(f\"✓ {package}\")\n",
            "\n",
            "print(\"\\n📦 Package installation completed!\")"
        ]
    })
    
    # Cell 4: Import Libraries
    notebook["cells"].append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Import essential libraries\n",
            "import torch\n",
            "import torch.nn as nn\n",
            "import torch.optim as optim\n",
            "import torch.nn.functional as F\n",
            "from torch.utils.data import Dataset, DataLoader\n",
            "from torch.utils.tensorboard import SummaryWriter\n",
            "\n",
            "# Data handling and visualization\n",
            "import numpy as np\n",
            "import pandas as pd\n",
            "import seaborn as sns\n",
            "import matplotlib.pyplot as plt\n",
            "from sklearn.model_selection import train_test_split\n",
            "from sklearn.preprocessing import LabelEncoder\n",
            "from sklearn.metrics import classification_report, confusion_matrix, accuracy_score\n",
            "\n",
            "# Text processing and utilities\n",
            "import re\n",
            "import string\n",
            "import unicodedata\n",
            "import random\n",
            "from collections import defaultdict, Counter\n",
            "import time\n",
            "from datetime import datetime\n",
            "import platform\n",
            "\n",
            "# Set style for better notebook aesthetics\n",
            "sns.set_style(\"whitegrid\")\n",
            "sns.set_palette(\"husl\")\n",
            "plt.rcParams['figure.figsize'] = (12, 8)\n",
            "\n",
            "# Set random seeds for reproducibility\n",
            "torch.manual_seed(42)\n",
            "np.random.seed(42)\n",
            "random.seed(42)\n",
            "\n",
            "print(f\"✅ PyTorch {torch.__version__} ready!\")\n",
            "print(f\"📊 Libraries imported successfully!\")"
        ]
    })
    
    # Cell 5: Device Detection
    notebook["cells"].append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "def detect_device():\n",
            "    \"\"\"\n",
            "    Detect the best available PyTorch device with comprehensive hardware support.\n",
            "    \n",
            "    Priority order:\n",
            "    1. CUDA (NVIDIA GPUs) - Best performance for deep learning\n",
            "    2. MPS (Apple Silicon) - Optimized for M1/M2/M3 Macs  \n",
            "    3. CPU (Universal) - Always available fallback\n",
            "    \n",
            "    Returns:\n",
            "        torch.device: The optimal device for PyTorch operations\n",
            "        str: Human-readable device description for logging\n",
            "    \"\"\"\n",
            "    # Check for CUDA (NVIDIA GPU)\n",
            "    if torch.cuda.is_available():\n",
            "        device = torch.device(\"cuda\")\n",
            "        gpu_name = torch.cuda.get_device_name(0)\n",
            "        device_info = f\"CUDA GPU: {gpu_name}\"\n",
            "        \n",
            "        # Additional CUDA info for optimization\n",
            "        cuda_version = torch.version.cuda\n",
            "        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3\n",
            "        \n",
            "        print(f\"🚀 Using CUDA acceleration\")\n",
            "        print(f\"   GPU: {gpu_name}\")\n",
            "        print(f\"   CUDA Version: {cuda_version}\")\n",
            "        print(f\"   GPU Memory: {gpu_memory:.1f} GB\")\n",
            "        \n",
            "        return device, device_info\n",
            "    \n",
            "    # Check for MPS (Apple Silicon)\n",
            "    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():\n",
            "        device = torch.device(\"mps\")\n",
            "        device_info = \"Apple Silicon MPS\"\n",
            "        \n",
            "        # Get system info for Apple Silicon\n",
            "        system_info = platform.uname()\n",
            "        \n",
            "        print(f\"🍎 Using Apple Silicon MPS acceleration\")\n",
            "        print(f\"   System: {system_info.system} {system_info.release}\")\n",
            "        print(f\"   Machine: {system_info.machine}\")\n",
            "        print(f\"   Processor: {system_info.processor}\")\n",
            "        \n",
            "        return device, device_info\n",
            "    \n",
            "    # Fallback to CPU\n",
            "    else:\n",
            "        device = torch.device(\"cpu\")\n",
            "        device_info = \"CPU (No GPU acceleration available)\"\n",
            "        \n",
            "        # Get CPU info for optimization guidance\n",
            "        cpu_count = torch.get_num_threads()\n",
            "        system_info = platform.uname()\n",
            "        \n",
            "        print(f\"💻 Using CPU (no GPU acceleration detected)\")\n",
            "        print(f\"   Processor: {system_info.processor}\")\n",
            "        print(f\"   PyTorch Threads: {cpu_count}\")\n",
            "        print(f\"   System: {system_info.system} {system_info.release}\")\n",
            "        \n",
            "        # Provide optimization suggestions for CPU-only setups\n",
            "        print(f\"\\n💡 CPU Optimization Tips:\")\n",
            "        print(f\"   • Reduce batch size to prevent memory issues\")\n",
            "        print(f\"   • Consider using smaller models for faster training\")\n",
            "        print(f\"   • Enable PyTorch optimizations: torch.set_num_threads({cpu_count})\")\n",
            "        \n",
            "        return device, device_info\n",
            "\n",
            "# Usage in all PyTorch notebooks\n",
            "device, device_info = detect_device()\n",
            "print(f\"\\n✅ PyTorch device selected: {device}\")\n",
            "print(f\"📊 Device info: {device_info}\")\n",
            "\n",
            "# Set global device for the notebook\n",
            "DEVICE = device"
        ]
    })
    
    # Save the notebook
    output_path = "examples/pytorch-nlp/classify-names-character-level-RNN.ipynb"
    with open(output_path, "w") as f:
        json.dump(notebook, f, indent=2)
    
    print(f"✅ Character-Level RNN notebook created: {output_path}")
    return output_path

if __name__ == "__main__":
    create_char_rnn_notebook()