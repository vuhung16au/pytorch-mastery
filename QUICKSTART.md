# Quickstart

This guide shows how to prepare a local Python 3.9/3.10 environment named `.venv`, install the dependencies that power the notebooks, and run the interactive examples inside `examples/`. A short section covers the equally easy Colab/Kaggle workflows.

## 1. Prepare a local `.venv`

### Shared steps
1. Clone the repo (if you have not already) and switch to its root:
   ```bash
   git clone https://github.com/vuhung16au/pytorch-mastery.git
   cd pytorch-mastery
   ```
2. Create the virtual environment with Python 3.9 or 3.10:
   ```bash
   python3.10 -m venv .venv
   source .venv/bin/activate
   python -m pip install --upgrade pip wheel
   ```
3. Install notebook tooling used across the examples:
   ```bash
   pip install jupyterlab tensorboard matplotlib numpy pandas scikit-learn
   ```

### macOS (Apple Silicon)
1. Install the PyTorch packages that include the native macOS builds:
   ```bash
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
   ```
2. Confirm the environment knows it is running without CUDA:
   ```bash
   python -c "import torch; print('CUDA:', torch.cuda.is_available())"
   ```
   *A macOS machine will report `False`, but the notebooks fall back to CPU seamlessly.*

### Ubuntu (CUDA-enabled)
1. Check that the NVIDIA driver, CUDA toolkit, and `nvidia-smi` are installed (Legion Pro 7 ships with these). Run:
   ```bash
   nvidia-smi
   ```
2. Install CUDA-aware PyTorch wheels (adjust `cu121` if you have a different CUDA version):
   ```bash
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
   ```
3. Verify CUDA is detected by the Python runtime:
   ```bash
   python -c "import torch; print('CUDA:', torch.cuda.is_available())"
   ```

### Optional extras
- Some notebooks in `examples/word_language_model` rely on additional libraries. Install them after the core stack with:
  ```bash
  pip install -r examples/word_language_model/requirements.txt
  ```
- `validate_notebook.py` targets `examples/hello-pytorch-cifar.ipynb`; run it to sanity-check structure+execution:
  ```bash
  python validate_notebook.py
  ```

## 2. Run the notebooks
1. From the repo root (with `.venv` activated), launch Jupyter Lab:
   ```bash
   cd examples
   jupyter lab
   ```
2. Use the browser interface to open any `examples/*.ipynb` notebook, for example:
   - `examples/pytorch-tutorials/01_pytorch_introduction.ipynb`
   - `examples/pytorch-nlp/02_word_embeddings_nllp.ipynb`
   - `examples/language_translation/04_huggingface_translation.ipynb`
3. Execute the cells sequentially. Each notebook logs TensorBoard data to a `tensorboard_logs/` folder; start TensorBoard if you need the visualizations:
   ```bash
   tensorboard --logdir tensorboard_logs/
   ```

## 3. Cloud notebooks (Colab / Kaggle)
1. **Colab**: Open https://colab.research.google.com/, click “GitHub,” paste `https://github.com/vuhung16au/pytorch-mastery`, and select the notebook you want. Colab autostarts the kernel, so just run the cells.
2. **Kaggle**: Go to https://www.kaggle.com/, create a new notebook, and use “Add data” → “URL” to link this repo. Upload the specific `.ipynb`, run its cells, and download any TensorBoard logs if needed.

Both platforms already provide the Python runtime, so no local `.venv` is needed there—just ensure chosen runtime matches Python 3.9/3.10 if that matters for compatibility.
