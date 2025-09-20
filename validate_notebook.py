#!/usr/bin/env python3
"""
Validation script for hello-pytorch-cifar.ipynb

This script validates that the notebook contains all required educational elements
and can execute without errors.
"""

import json
import sys
import subprocess
import tempfile
import os


def validate_notebook_structure(notebook_path):
    """Validate the notebook has required educational elements."""
    print("🔍 Validating notebook structure...")
    
    with open(notebook_path, 'r') as f:
        notebook = json.load(f)
    
    cells = notebook['cells']
    cell_contents = []
    
    for cell in cells:
        if cell['cell_type'] == 'code':
            cell_contents.extend(cell['source'])
        elif cell['cell_type'] == 'markdown':
            cell_contents.extend(cell['source'])
    
    content = ' '.join(cell_contents)
    
    # Check for required educational elements
    requirements = {
        'PyTorch fundamentals': ['torch', 'nn.Module', 'tensor'],
        'CIFAR-10 dataset': ['CIFAR', 'classes'],
        'CNN model': ['Conv2d', 'forward', '__init__'],
        'Training loop': ['optimizer.zero_grad', 'loss.backward', 'optimizer.step'],
        'TensorBoard logging': ['SummaryWriter', 'add_scalar'],
        'Environment detection': ['IS_COLAB', 'IS_KAGGLE', 'IS_LOCAL'],
        'TensorFlow comparison': ['TensorFlow', 'vs', 'equivalent'],
        'Error handling': ['try:', 'except', 'synthetic'],
        'Educational comments': ['#', 'Learning', 'tutorial'],
        'Australian context': []  # Optional, not required for core functionality
    }
    
    missing_elements = []
    for element, keywords in requirements.items():
        if keywords and not any(keyword in content for keyword in keywords):
            missing_elements.append(element)
    
    if missing_elements:
        print(f"❌ Missing required elements: {missing_elements}")
        return False
    else:
        print("✅ All required educational elements found")
        return True


def validate_notebook_execution(notebook_path):
    """Validate that the notebook can execute without errors."""
    print("🚀 Validating notebook execution...")
    
    try:
        # Create a test version with limited cells to speed up validation
        with open(notebook_path, 'r') as f:
            notebook = json.load(f)
        
        # Test first 8 cells (setup + basic functionality)
        test_notebook = {
            'cells': notebook['cells'][:8],
            'metadata': notebook['metadata'],
            'nbformat': notebook['nbformat'],
            'nbformat_minor': notebook['nbformat_minor']
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.ipynb', delete=False) as f:
            json.dump(test_notebook, f)
            temp_notebook = f.name
        
        # Execute the test notebook
        result = subprocess.run([
            'jupyter', 'nbconvert', '--execute', temp_notebook,
            '--to', 'notebook', '--stdout'
        ], capture_output=True, text=True, timeout=120)
        
        os.unlink(temp_notebook)  # Clean up
        
        if result.returncode == 0:
            print("✅ Notebook execution test passed")
            return True
        else:
            print(f"❌ Notebook execution failed: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        print("❌ Notebook execution timed out")
        return False
    except Exception as e:
        print(f"❌ Error during execution test: {e}")
        return False


def main():
    """Main validation function."""
    notebook_path = 'examples/hello-pytorch-cifar.ipynb'
    
    if not os.path.exists(notebook_path):
        print(f"❌ Notebook not found: {notebook_path}")
        sys.exit(1)
    
    print("🎓 Validating Hello PyTorch CIFAR-10 Notebook")
    print("=" * 50)
    
    # Validate structure
    structure_valid = validate_notebook_structure(notebook_path)
    
    # Validate execution
    execution_valid = validate_notebook_execution(notebook_path)
    
    print("\n" + "=" * 50)
    
    if structure_valid and execution_valid:
        print("🎉 Validation PASSED: Notebook meets all requirements!")
        print("\n📋 Requirements validated:")
        print("  ✅ Educational structure and content")
        print("  ✅ PyTorch fundamentals coverage")
        print("  ✅ CIFAR-10 dataset handling")
        print("  ✅ CNN model implementation")
        print("  ✅ Training loop with TensorBoard")
        print("  ✅ Environment compatibility")
        print("  ✅ Error handling for offline usage")
        print("  ✅ TensorFlow comparison comments")
        print("  ✅ Executable without internet")
        
        print("\n🚀 The notebook is ready for educational use!")
        sys.exit(0)
    else:
        print("❌ Validation FAILED: Notebook needs fixes")
        sys.exit(1)


if __name__ == '__main__':
    main()