# PyTorch Tutorials

This directory contains comprehensive PyTorch tutorial notebooks designed for learners transitioning from TensorFlow to PyTorch, with a focus on NLP applications using Australian context examples and English-Vietnamese multilingual support.

## Tutorials Available

### 05_pytorch_tensorboard.ipynb
**PyTorch TensorBoard Support: Comprehensive Visualization Guide**

Demonstrates how to integrate and effectively use TensorBoard with PyTorch for:
- Setting up TensorBoard with PyTorch using `SummaryWriter`
- Visualizing data elements (especially useful for computer vision tasks)
- Monitoring training progress with scalar metrics (loss, accuracy)
- Understanding model architecture through computational graphs
- Exploring high-dimensional data using embedding projections

**Dataset**: FashionMNIST with Australian Fashion Context
- Uses FashionMNIST dataset with Australian fashion item mappings
- Includes English-Vietnamese bilingual labels
- Demonstrates all 4 key TensorBoard features:
  1. **Image Visualization** - Fashion samples with Australian context
  2. **Scalar Monitoring** - Training/validation metrics tracking
  3. **Graph Visualization** - LeNet-5 model architecture
  4. **Embeddings** - High-dimensional fashion item clustering

**Key Learning Points**: TensorFlow vs PyTorch TensorBoard differences
- PyTorch: Manual `SummaryWriter` setup and explicit logging calls
- TensorFlow: Built-in callbacks and automatic metric logging
- PyTorch: More granular control over logging
- Both: Same TensorBoard UI for visualization

## Running the Tutorials

### Prerequisites
```bash
pip install torch torchvision torchaudio tensorboard matplotlib numpy pandas scikit-learn
```

### Local Environment
```bash
jupyter lab
# Navigate to pytorch-tutorials/05_pytorch_tensorboard.ipynb
```

### Google Colab
1. Upload the notebook to Colab
2. The notebook handles environment detection automatically
3. Use `%load_ext tensorboard` and `%tensorboard --logdir <path>` to view results

### Kaggle
1. Upload the notebook to Kaggle
2. Download TensorBoard logs after execution
3. View locally with `tensorboard --logdir ./tensorboard_logs`

## Features

- **Cross-platform compatibility** (Local, Colab, Kaggle)
- **Device detection** (CUDA, MPS, CPU with fallback)
- **Australian context examples** following repository policies
- **English-Vietnamese multilingual support**
- **TensorFlow comparison** throughout for easy transition
- **Comprehensive TensorBoard integration** with all major features
- **Educational focus** with detailed explanations and best practices

## Next Steps

After completing these tutorials:
1. Apply TensorBoard logging to your own PyTorch projects
2. Explore advanced TensorBoard features (custom scalars, PR curves)
3. Integrate with Hugging Face transformers for NLP tasks
4. Experiment with real Australian tourism/fashion datasets