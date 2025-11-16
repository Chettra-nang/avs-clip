#!/bin/bash
# Prepare code for Kaggle GPU training

echo "=========================================="
echo "Preparing Code for Kaggle GPU Training"
echo "=========================================="

# Create kaggle_upload directory
echo "Creating kaggle_upload directory..."
mkdir -p kaggle_upload

# Copy essential files
echo "Copying code files..."
cp -r models kaggle_upload/
cp -r training kaggle_upload/
cp -r configs kaggle_upload/
cp -r common kaggle_upload/
cp -r envs kaggle_upload/
cp -r custom_envs kaggle_upload/

# Copy requirements
echo "Copying requirements..."
cat > kaggle_upload/requirements.txt << 'EOF'
gymnasium>=0.29.0
highway-env>=1.8.2
torch>=2.0.0
torchvision>=0.15.0
transformers>=4.30.0
pillow>=9.5.0
numpy>=1.24.0
pyyaml>=6.0
tqdm>=4.65.0
tensorboard>=2.13.0
opencv-python>=4.8.0
EOF

# Create Kaggle training script
echo "Creating Kaggle training script..."
cat > kaggle_upload/kaggle_train.py << 'EOFPY'
"""
Complete training script for Kaggle GPU.
Run this in a Kaggle notebook with GPU enabled.
"""
import os
import sys
import torch
import gymnasium as gym
import highway_env
import custom_envs
from pathlib import Path

# Add current directory to path
sys.path.insert(0, '/kaggle/working')

def setup_environment():
    """Setup Kaggle environment."""
    print("=" * 80)
    print("KAGGLE GPU TRAINING SETUP")
    print("=" * 80)
    
    # Check GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n✓ Device: {device}")
    if torch.cuda.is_available():
        print(f"✓ GPU: {torch.cuda.get_device_name(0)}")
        print(f"✓ GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    # Create directories
    os.makedirs('/kaggle/working/checkpoints', exist_ok=True)
    os.makedirs('/kaggle/working/logs', exist_ok=True)
    print("\n✓ Directories created")
    
    return device

def train_clip_on_kaggle(device='cuda'):
    """Fine-tune CLIP on Kaggle."""
    from training.train_clip import train_clip
    
    print("\n" + "=" * 80)
    print("STEP 1: CLIP FINE-TUNING")
    print("=" * 80)
    
    # Note: You need to upload your data as a Kaggle dataset
    # For now, we'll skip this and use pre-trained CLIP
    print("\n⚠️  Skipping CLIP fine-tuning (no data uploaded)")
    print("Using pre-trained CLIP instead")
    
    return None

def train_mappo_on_kaggle(device='cuda', total_steps=20000):
    """Train MAPPO on Kaggle."""
    from training.train_mappo import main as train_mappo_main
    import argparse
    
    print("\n" + "=" * 80)
    print("STEP 2: MAPPO TRAINING")
    print("=" * 80)
    
    # Override sys.argv for argparse
    sys.argv = [
        'kaggle_train.py',
        '--scenario', 'highway_heterogeneous',
        '--steps', str(total_steps),
        '--horizon', '256',
        '--lr', '3e-4',
        '--device', device,
        '--logdir', '/kaggle/working/logs'
    ]
    
    print(f"\n✓ Training for {total_steps} steps...")
    print(f"✓ Scenario: highway_heterogeneous")
    print(f"✓ Device: {device}")
    
    # Run training
    train_mappo_main()
    
    print("\n✓ Training complete!")

def save_results():
    """Archive results for download."""
    import shutil
    
    print("\n" + "=" * 80)
    print("SAVING RESULTS")
    print("=" * 80)
    
    # Create archives
    if os.path.exists('/kaggle/working/checkpoints'):
        shutil.make_archive('/kaggle/working/training_checkpoints', 'zip', 
                          '/kaggle/working/checkpoints')
        print("✓ Checkpoints archived: training_checkpoints.zip")
    
    if os.path.exists('/kaggle/working/logs'):
        shutil.make_archive('/kaggle/working/training_logs', 'zip',
                          '/kaggle/working/logs')
        print("✓ Logs archived: training_logs.zip")
    
    print("\n✓ Download these files from Kaggle Output tab")

def main():
    """Main training pipeline."""
    # Setup
    device = setup_environment()
    
    # Train CLIP (optional, skip if no data)
    train_clip_on_kaggle(device)
    
    # Train MAPPO
    train_mappo_on_kaggle(device, total_steps=20000)
    
    # Save results
    save_results()
    
    print("\n" + "=" * 80)
    print("TRAINING COMPLETE!")
    print("=" * 80)
    print("\nNext steps:")
    print("1. Download training_checkpoints.zip from Output tab")
    print("2. Download training_logs.zip from Output tab")
    print("3. Extract locally and evaluate model")

if __name__ == '__main__':
    main()
EOFPY

# Create Kaggle notebook template
echo "Creating Kaggle notebook template..."
cat > kaggle_upload/kaggle_notebook.ipynb << 'EOFJSON'
{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Ambulance Priority MAPPO Training on Kaggle GPU\n",
    "\n",
    "This notebook trains the MAPPO model with CLIP vision encoder on Kaggle's free GPU.\n",
    "\n",
    "**Setup:**\n",
    "1. Enable GPU: Settings → Accelerator → GPU T4 x2\n",
    "2. Enable Internet: Settings → Internet → ON\n",
    "3. Add your code dataset: Add Data → Search for your uploaded code\n",
    "\n",
    "**Expected time:** 2-3 hours for 20k steps"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# CELL 1: Install Dependencies\n",
    "!pip install --quiet gymnasium highway-env transformers torch torchvision\n",
    "!pip install --quiet pillow numpy pyyaml tqdm tensorboard opencv-python\n",
    "\n",
    "print(\"✓ Dependencies installed\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# CELL 2: Extract Code (if uploaded as dataset)\n",
    "import os\n",
    "import sys\n",
    "import tarfile\n",
    "\n",
    "# If you uploaded code as tar.gz\n",
    "code_path = '/kaggle/input/ambulance-priority-code/kaggle_package.tar.gz'\n",
    "if os.path.exists(code_path):\n",
    "    with tarfile.open(code_path, 'r:gz') as tar:\n",
    "        tar.extractall('/kaggle/working/')\n",
    "    print(\"✓ Code extracted from dataset\")\n",
    "else:\n",
    "    # Or clone from GitHub\n",
    "    !git clone https://github.com/YOUR_USERNAME/YOUR_REPO.git /kaggle/working/code\n",
    "    print(\"✓ Code cloned from GitHub\")\n",
    "\n",
    "# Add to path\n",
    "sys.path.insert(0, '/kaggle/working')\n",
    "print(\"✓ Code ready\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# CELL 3: Verify GPU\n",
    "import torch\n",
    "\n",
    "print(f\"GPU Available: {torch.cuda.is_available()}\")\n",
    "if torch.cuda.is_available():\n",
    "    print(f\"GPU Name: {torch.cuda.get_device_name(0)}\")\n",
    "    print(f\"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB\")\n",
    "else:\n",
    "    print(\"⚠️  No GPU detected! Enable GPU in Settings.\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# CELL 4: Run Training\n",
    "from kaggle_train import main\n",
    "\n",
    "# This will train for 20k steps (~2-3 hours)\n",
    "main()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# CELL 5: Download Results\n",
    "print(\"Training complete!\")\n",
    "print(\"\\nDownload these files from the Output tab:\")\n",
    "print(\"  - training_checkpoints.zip\")\n",
    "print(\"  - training_logs.zip\")\n",
    "print(\"\\nOr upload to Google Drive:\")\n",
    "\n",
    "# Optional: Upload to Google Drive\n",
    "# from google.colab import drive\n",
    "# drive.mount('/content/drive')\n",
    "# import shutil\n",
    "# shutil.copytree('/kaggle/working/checkpoints', '/content/drive/MyDrive/ambulance_training/')"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "name": "python",
   "version": "3.10.0"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}
EOFJSON

# Create archive
echo "Creating archive..."
cd kaggle_upload
tar -czf ../kaggle_package.tar.gz .
cd ..

echo ""
echo "=========================================="
echo "✓ Kaggle package created!"
echo "=========================================="
echo ""
echo "File created: kaggle_package.tar.gz"
echo "Size: $(du -h kaggle_package.tar.gz | cut -f1)"
echo ""
echo "Next steps:"
echo "1. Upload kaggle_package.tar.gz to Kaggle as a dataset"
echo "2. Create a new Kaggle notebook"
echo "3. Copy cells from kaggle_upload/kaggle_notebook.ipynb"
echo "4. Enable GPU (Settings → GPU T4 x2)"
echo "5. Run the notebook!"
echo ""
echo "Or use GitHub:"
echo "1. Push your code to GitHub"
echo "2. In Kaggle notebook: !git clone YOUR_REPO_URL"
echo "3. Run training"
echo ""
