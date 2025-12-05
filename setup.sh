#!/bin/bash
# Setup script for macOS and Linux

set -e

echo "=========================================="
echo "Intelli-ODM Environment Setup"
echo "Platform: macOS/Linux"
echo "=========================================="

# Check Python version
echo -e "\n[1/5] Checking Python version..."
if ! command -v python3 &> /dev/null; then
    echo "ERROR: Python 3 is not installed. Please install Python 3.10 or higher."
    exit 1
fi

PYTHON_VERSION=$(python3 --version | cut -d' ' -f2)
echo "Found Python $PYTHON_VERSION"

# Create virtual environment
echo -e "\n[2/5] Creating virtual environment..."
python3 -m venv .venv
echo "Virtual environment created at .venv/"

# Activate virtual environment
echo -e "\n[3/5] Activating virtual environment..."
source .venv/bin/activate

# Upgrade pip
echo -e "\n[4/5] Upgrading pip, setuptools, and wheel..."
python -m pip install --upgrade pip setuptools wheel

# Install dependencies
echo -e "\n[5/5] Installing dependencies from requirements.txt..."
pip install -r requirements.txt

echo -e "\n=========================================="
echo "✅ Setup complete!"
echo "=========================================="
echo ""
echo "To activate the virtual environment, run:"
echo "  source .venv/bin/activate"
echo ""
echo "To deactivate, run:"
echo "  deactivate"
echo ""
echo "Next steps:"
echo "  1. Install Ollama: https://ollama.com/install"
echo "  2. Pull Llama3 model: ollama pull llama3:8b"
echo "  3. Run the system: python orchestrator.py"
echo ""

