#!/bin/bash
set -e

# VLM-MARL Highway - Automated Setup Script for Ubuntu + RTX 4090
# This script sets up the complete environment for training

echo "╔══════════════════════════════════════════════════════════════════════╗"
echo "║     VLM-MARL Highway - Automated Setup for Ubuntu + RTX 4090        ║"
echo "╚══════════════════════════════════════════════════════════════════════╝"
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if running on Ubuntu
if [ ! -f /etc/os-release ]; then
    echo -e "${RED}Error: Cannot detect OS. This script is for Ubuntu Linux.${NC}"
    exit 1
fi

source /etc/os-release
if [[ "$ID" != "ubuntu" ]]; then
    echo -e "${YELLOW}Warning: This script is designed for Ubuntu. Detected: $ID${NC}"
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

echo -e "${GREEN}✓ OS Check: Ubuntu $VERSION_ID${NC}"

# Check for NVIDIA GPU
if ! command -v nvidia-smi &> /dev/null; then
    echo -e "${RED}Error: nvidia-smi not found. Please install NVIDIA drivers first.${NC}"
    echo "Run: sudo apt install nvidia-driver-535"
    exit 1
fi

GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
echo -e "${GREEN}✓ GPU Detected: $GPU_NAME${NC}"

# Check Python version
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}Error: Python 3 not found. Installing...${NC}"
    sudo apt update
    sudo apt install -y python3.10 python3.10-venv python3-pip
fi

PYTHON_VERSION=$(python3 --version | cut -d' ' -f2)
echo -e "${GREEN}✓ Python Version: $PYTHON_VERSION${NC}"

# Create virtual environment
echo ""
echo "Creating virtual environment..."
if [ -d "venv" ]; then
    echo -e "${YELLOW}Warning: venv directory already exists. Removing...${NC}"
    rm -rf venv
fi

python3 -m venv venv
source venv/bin/activate

echo -e "${GREEN}✓ Virtual environment created${NC}"

# Upgrade pip
echo ""
echo "Upgrading pip..."
pip install --upgrade pip setuptools wheel > /dev/null 2>&1
echo -e "${GREEN}✓ Pip upgraded${NC}"

# Install PyTorch with CUDA
echo ""
echo "Installing PyTorch with CUDA 11.8 (this may take a few minutes)..."
pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu118 > /dev/null 2>&1
echo -e "${GREEN}✓ PyTorch installed${NC}"

# Verify CUDA
echo ""
echo "Verifying CUDA availability..."
python -c "import torch; assert torch.cuda.is_available(), 'CUDA not available'; print(f'CUDA Version: {torch.version.cuda}'); print(f'GPU: {torch.cuda.get_device_name(0)}')"
echo -e "${GREEN}✓ CUDA verified${NC}"

# Install other dependencies
echo ""
echo "Installing project dependencies..."
pip install -r requirements.txt > /dev/null 2>&1
echo -e "${GREEN}✓ Dependencies installed${NC}"

# Create necessary directories
echo ""
echo "Creating project directories..."
mkdir -p data models runs logs
echo -e "${GREEN}✓ Directories created${NC}"

# Set PYTHONPATH
echo ""
echo "Configuring environment..."
echo 'export PYTHONPATH=.' >> venv/bin/activate
echo -e "${GREEN}✓ PYTHONPATH configured${NC}"

# Create GPU optimization script
cat > gpu_config.sh << 'EOF'
#!/bin/bash
# GPU optimization for RTX 4090
export NVIDIA_TF32_OVERRIDE=1
export TF_FORCE_GPU_ALLOW_GROWTH=true
export CUDA_LAUNCH_BLOCKING=0
export TORCH_CUDA_ARCH_LIST="8.9"
export CUDNN_BENCHMARK=1
echo "GPU optimizations enabled for RTX 4090"
EOF
chmod +x gpu_config.sh
echo -e "${GREEN}✓ GPU optimization script created${NC}"

# Run quick test
echo ""
echo "Running quick test..."
python data_collection/collect_data.py \
    --scenario highway_heterogeneous \
    --episodes 1 \
    --max-steps 10 \
    --output-dir test_output \
    --seed 42 > /dev/null 2>&1

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Quick test passed${NC}"
    rm -rf test_output
else
    echo -e "${RED}✗ Quick test failed${NC}"
    echo "Please check the error messages above"
    exit 1
fi

# Print summary
echo ""
echo "╔══════════════════════════════════════════════════════════════════════╗"
echo "║                    SETUP COMPLETE ✓                                  ║"
echo "╚══════════════════════════════════════════════════════════════════════╝"
echo ""
echo "System Information:"
echo "  OS: Ubuntu $VERSION_ID"
echo "  GPU: $GPU_NAME"
echo "  Python: $PYTHON_VERSION"
echo "  PyTorch: $(python -c 'import torch; print(torch.__version__)')"
echo "  CUDA: $(python -c 'import torch; print(torch.version.cuda)')"
echo ""
echo "Next Steps:"
echo ""
echo "1. Activate environment:"
echo "   source venv/bin/activate"
echo "   source gpu_config.sh"
echo ""
echo "2. Run full pipeline:"
echo "   ./run.sh highway_heterogeneous --heterogeneous"
echo ""
echo "3. Or run step-by-step:"
echo "   # Data collection (45 min)"
echo "   python data_collection/collect_data.py --scenario highway_heterogeneous --episodes 70 --output-dir data/highway_heterogeneous_normal"
echo "   python data_collection/collect_data.py --scenario highway_heterogeneous_dense --episodes 30 --output-dir data/highway_heterogeneous_dense"
echo ""
echo "   # CLIP training (1-2 hours)"
echo "   python training/train_clip.py --data_dirs data/highway_heterogeneous --epochs 5"
echo ""
echo "   # MAPPO training (3-4 hours)"
echo "   python training/train_mappo.py --scenario highway_heterogeneous --steps 100000"
echo ""
echo "4. Monitor training:"
echo "   tensorboard --logdir runs --port 6006"
echo ""
echo "For detailed instructions, see SETUP_GUIDE.md"
echo ""
echo -e "${GREEN}Happy training! 🚀${NC}"
