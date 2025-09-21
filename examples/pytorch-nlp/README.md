# PyTorch + NLP Examples

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/vuhung16au/pytorch-mastery/blob/main/examples/pytorch-nlp/)
[![View on GitHub](https://img.shields.io/badge/View_on-GitHub-blue?logo=github)](https://github.com/vuhung16au/pytorch-mastery/blob/main/examples/pytorch-nlp/)

This directory contains comprehensive PyTorch Natural Language Processing (NLP) examples designed for learners transitioning from TensorFlow to PyTorch, with a focus on Australian context examples and English-Vietnamese multilingual support.

## Learning Objectives

The goal is to learn basic NLP with PyTorch through hands-on examples that demonstrate:

- **Foundation Concepts**: Deep learning for NLP with PyTorch fundamentals
- **Word Embeddings**: Encoding lexical semantics for Australian tourism and Vietnamese text
- **Sequence Models**: LSTM, GRU, and RNN models for Australian text classification
- **Advanced Techniques**: Bi-LSTM CRF for complex NLP tasks with multilingual support
- **TensorFlow Transition**: Clear comparisons to help TensorFlow users understand PyTorch patterns

## Notebooks Overview

### 1. 📚 Deep Learning NLP Foundations
**File**: `01_deep_learning_nlp.ipynb`

Introduction to deep learning concepts for NLP using PyTorch with Australian examples.

**Key Topics**:
- PyTorch tensor operations for text processing
- Neural network basics for NLP with Australian tourism data
- Text preprocessing pipelines with English-Vietnamese examples
- Basic classification models for Australian city sentiment analysis

**Learning Outcomes**:
- Understand PyTorch fundamentals for NLP applications
- Build simple neural networks for text classification
- Process Australian tourism reviews and Vietnamese translations
- Compare PyTorch vs TensorFlow approaches for NLP

### 2. 🔤 Word Embeddings
**File**: `02_word_embeddings_nllp.ipynb`

Comprehensive guide to word embeddings with Australian context and multilingual support.

**Key Topics**:
- Word2Vec, GloVe, and FastText embeddings
- Training custom embeddings on Australian tourism corpus
- English-Vietnamese word alignment and cross-lingual embeddings
- Visualization of Australian city and landmark embeddings

**Learning Outcomes**:
- Master word embedding techniques in PyTorch
- Create domain-specific embeddings for Australian NLP tasks
- Handle multilingual embeddings for English-Vietnamese text
- Visualize semantic relationships in Australian tourism vocabulary

### 3. 🔄 Sequence Models
**File**: `03_sequence_models_nlp.ipynb`

Advanced sequence modeling with LSTM and GRU networks for Australian NLP applications.

**Key Topics**:
- RNN, LSTM, and GRU architectures in PyTorch
- Sentiment analysis of Australian restaurant reviews
- Part-of-speech tagging for Australian English and Vietnamese text
- Sequence-to-sequence models for English-Vietnamese translation

**Learning Outcomes**:
- Build and train sequence models from scratch
- Apply LSTM/GRU to real Australian NLP datasets
- Handle variable-length sequences in multilingual contexts
- Implement attention mechanisms for better performance

### 4. 🚀 Advanced NLP Techniques
**File**: `04_advanced_nlp.ipynb`

State-of-the-art NLP techniques including Bi-LSTM CRF for complex sequence labeling.

**Key Topics**:
- Bi-directional LSTM with CRF layers
- Named Entity Recognition (NER) for Australian locations and organizations
- Advanced optimization techniques and dynamic computation graphs
- Integration with Hugging Face transformers for Australian NLP tasks

**Learning Outcomes**:
- Implement advanced neural architectures for NLP
- Master dynamic decision making in sequence models
- Apply Bi-LSTM CRF to real-world Australian NER tasks
- Bridge PyTorch fundamentals to modern transformer architectures

## 🇦🇺 Australian Context Examples

All notebooks feature practical examples using Australian data and scenarios:

### Text Classification Examples
```python
# Australian tourism sentiment analysis
tourism_reviews = [
    "The Sydney Opera House is absolutely breathtaking! Best experience ever.",
    "Melbourne's coffee scene is overrated and expensive.",
    "Uluru at sunset was a spiritual experience I'll never forget.",
    "Perth beaches are perfect for families with young children."
]

# Vietnamese translations for multilingual learning
vietnamese_reviews = [
    "Nhà hát Opera Sydney thật ngoạn mục! Trải nghiệm tuyệt vời nhất.",
    "Cảnh cà phê Melbourne bị đánh giá quá cao và đắt đỏ.",
    "Uluru lúc hoàng hôn là trải nghiệm tâm linh tôi sẽ không bao giờ quên.",
    "Bãi biển Perth hoàn hảo cho các gia đình có con nhỏ."
]
```

### Named Entity Recognition Examples
```python
# Australian locations and organizations
australian_entities = [
    "Sydney Opera House",        # LANDMARK
    "Great Barrier Reef",        # NATURAL_FEATURE  
    "Commonwealth Bank",         # ORGANIZATION
    "New South Wales",           # STATE
    "Qantas Airways",           # ORGANIZATION
    "Bondi Beach",              # LOCATION
]
```

### Classification Categories
```python
# Australian-specific categories
australian_cities = ["Sydney", "Melbourne", "Brisbane", "Perth", "Adelaide", "Darwin", "Hobart", "Canberra"]
aussie_animals = ["kangaroo", "koala", "wombat", "echidna", "platypus", "dingo", "crocodile", "kookaburra"]
tourism_categories = ["beaches", "restaurants", "attractions", "accommodation", "transport", "activities"]
```

## 🌏 Multilingual Support (English-Vietnamese)

### Translation Pairs
```python
# English-Vietnamese translation examples
translation_pairs = [
    ("Welcome to Australia", "Chào mừng đến Australia"),
    ("Sydney harbour is beautiful", "Cảng Sydney rất đẹp"),
    ("I love Australian coffee", "Tôi yêu cà phê Úc"),
    ("The weather in Melbourne is unpredictable", "Thời tiết ở Melbourne không thể đoán trước")
]
```

### Cross-lingual Tasks
```python
# Sentiment analysis across languages
multilingual_sentiment = {
    'en': ["Sydney beaches are amazing!", "Brisbane weather is too hot"],
    'vi': ["Bãi biển Sydney tuyệt vời!", "Thời tiết Brisbane quá nóng"]
}
```

## 🛠️ Technical Requirements

### Environment Setup

**Prerequisites**:
```bash
# Python 3.8+ required
python --version

# Install PyTorch and core dependencies (3-5 minutes)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Install Hugging Face ecosystem (2-3 minutes)
pip install transformers datasets tokenizers

# Install visualization and ML libraries (2-3 minutes)  
pip install numpy pandas seaborn matplotlib scikit-learn jupyter tensorboard
```

**Verification**:
```bash
python -c "
import torch, transformers, datasets
print(f'PyTorch: {torch.__version__}')
print(f'Transformers: {transformers.__version__}')
print('✅ Environment ready for PyTorch NLP!')
"
```

### Device Support

All notebooks include intelligent device detection:
- **CUDA**: NVIDIA GPU acceleration (if available)
- **MPS**: Apple Silicon optimization (M1/M2/M3 Macs)
- **CPU**: Universal fallback with optimization

### Cross-Platform Compatibility

- **Local Development**: Full feature support with device detection
- **Google Colab**: Optimized for cloud GPU usage
- **Kaggle**: Compatible with Kaggle notebook environment
- **Jupyter Lab/Notebook**: Complete interactive experience

## 🎯 Learning Path Progression

### Phase 1: PyTorch NLP Foundations (Week 1)
- Complete `01_deep_learning_nlp.ipynb`
- Understand tensor operations for text
- Build first neural network for Australian text classification
- Compare with TensorFlow approaches

### Phase 2: Embeddings and Representation (Week 2)  
- Work through `02_word_embeddings_nllp.ipynb`
- Train custom embeddings on Australian corpus
- Explore English-Vietnamese embedding alignment
- Visualize semantic relationships

### Phase 3: Sequential Modeling (Week 3)
- Master `03_sequence_models_nlp.ipynb`
- Implement LSTM/GRU from scratch
- Apply to Australian sentiment analysis
- Handle multilingual sequence tasks

### Phase 4: Advanced Techniques (Week 4)
- Complete `04_advanced_nlp.ipynb`
- Implement Bi-LSTM CRF architecture
- Tackle complex NER tasks
- Bridge to Hugging Face transformers

## 📊 TensorBoard Integration

All training examples include comprehensive TensorBoard logging:

```python
from torch.utils.tensorboard import SummaryWriter

# Automatic log directory creation
writer = SummaryWriter('runs/australian_sentiment_analysis')

# Training metrics logging
writer.add_scalar('Loss/Train', train_loss, epoch)
writer.add_scalar('Accuracy/Validation', val_acc, epoch)
writer.add_histogram('Embeddings/Word_Vectors', embeddings, epoch)
```

**Viewing Results**:
```bash
# Start TensorBoard
tensorboard --logdir runs/

# Open browser to http://localhost:6006
```

## 🔄 TensorFlow vs PyTorch Comparison

Each notebook includes clear comparisons to help TensorFlow users transition:

| Concept | TensorFlow | PyTorch |
|---------|------------|---------|
| **Model Definition** | `tf.keras.Sequential` | `nn.Module` subclass |
| **Training Loop** | `model.fit()` | Manual with `loss.backward()` |
| **Dynamic Graphs** | `tf.GradientTape` | Native support |
| **Device Placement** | `with tf.device()` | `.to(device)` |

## 🌟 Key Features

- **📖 Educational Focus**: Step-by-step explanations with detailed comments
- **🇦🇺 Australian Context**: Real-world examples using Australian data
- **🌏 Multilingual**: English-Vietnamese support throughout
- **🔧 Cross-Platform**: Works on Local, Colab, and Kaggle
- **📊 Comprehensive Logging**: TensorBoard integration for all training
- **🚀 Performance**: Device optimization (CUDA/MPS/CPU)
- **🔄 Transition-Friendly**: Clear TensorFlow comparisons

## 🚀 Quick Start

1. **Clone and Navigate**:
   ```bash
   git clone https://github.com/vuhung16au/pytorch-mastery.git
   cd pytorch-mastery/examples/pytorch-nlp/
   ```

2. **Install Dependencies**:
   ```bash
   pip install torch transformers datasets seaborn tensorboard
   ```

3. **Start Jupyter**:
   ```bash
   jupyter lab
   ```

4. **Open Notebooks**:
   - Start with `01_deep_learning_nlp.ipynb` for foundations
   - Progress through `02_word_embeddings_nllp.ipynb`
   - Continue with `03_sequence_models_nlp.ipynb`
   - Complete with `04_advanced_nlp.ipynb`

## 🎓 Expected Learning Outcomes

After completing these examples, you will:

- ✅ Master PyTorch fundamentals for NLP applications
- ✅ Understand word embeddings and semantic representations
- ✅ Build and train sequence models (LSTM, GRU, Bi-LSTM)
- ✅ Implement advanced architectures like Bi-LSTM CRF
- ✅ Handle multilingual NLP tasks effectively
- ✅ Transition confidently from TensorFlow to PyTorch
- ✅ Apply techniques to real Australian NLP challenges
- ✅ Integrate with modern transformer architectures

## 📚 Additional Resources

- [PyTorch NLP Tutorials](https://pytorch.org/tutorials/beginner/nlp/)
- [Hugging Face Transformers](https://huggingface.co/transformers/)
- [TensorBoard for PyTorch](https://pytorch.org/docs/stable/tensorboard.html)
- [Australian Text Corpora](https://github.com/Australian-Text-Analytics-Platform)

## 🤝 Contributing

These examples are part of the PyTorch Mastery learning repository. Contributions that maintain the Australian context and educational focus are welcome!

---

**🎉 Ready to master PyTorch NLP with Australian flair? Let's get started! 🇦🇺**