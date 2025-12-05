# Installation Guide - Intelli-ODM

This guide covers installation on **Windows**, **macOS**, and **Linux**.

---

## Table of Contents
- [Prerequisites](#prerequisites)
- [Windows Installation](#windows-installation)
- [macOS Installation](#macos-installation)
- [Linux Installation](#linux-installation)
- [Verify Installation](#verify-installation)
- [Troubleshooting](#troubleshooting)

---

## Prerequisites

### All Platforms
- **Python 3.10 or higher** (Python 3.11+ recommended)
- **Git** (for cloning the repository)
- **Ollama** (for LLM functionality)
- **At least 8GB RAM** (16GB recommended for larger datasets)

---

## Windows Installation

### Method 1: Using PowerShell (Recommended)

1. **Clone the repository**
```powershell
git clone https://github.com/your-org/intelli-odm.git
cd intelli-odm
```

2. **Run the setup script**
```powershell
# If you get execution policy errors, run:
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser

# Run setup
.\setup.ps1
```

### Method 2: Using Command Prompt

1. **Clone the repository**
```cmd
git clone https://github.com/your-org/intelli-odm.git
cd intelli-odm
```

2. **Run the setup script**
```cmd
setup.bat
```

### Method 3: Manual Installation (Windows)

```cmd
# Create virtual environment
python -m venv .venv

# Activate virtual environment
.venv\Scripts\activate

# Upgrade pip
python -m pip install --upgrade pip setuptools wheel

# Install dependencies
pip install -r requirements.txt
```

### Installing Ollama (Windows)

1. Download from: https://ollama.com/download/windows
2. Install the executable
3. Pull the Llama3 model:
```cmd
ollama pull llama3:8b
```

### Known Issues on Windows

**Prophet Installation Issues:**
- Prophet requires C++ build tools on Windows
- **Solution 1**: Use Conda instead
```cmd
conda create -n intelli-odm python=3.10
conda activate intelli-odm
conda install -c conda-forge prophet
pip install -r requirements.txt
```

- **Solution 2**: Install Visual C++ Build Tools
  - Download from: https://visualstudio.microsoft.com/downloads/
  - Select "Desktop development with C++" workload

---

## macOS Installation

### Method 1: Using Setup Script (Recommended)

1. **Clone the repository**
```bash
git clone https://github.com/your-org/intelli-odm.git
cd intelli-odm
```

2. **Make setup script executable and run**
```bash
chmod +x setup.sh
./setup.sh
```

### Method 2: Manual Installation

```bash
# Create virtual environment
python3 -m venv .venv

# Activate virtual environment
source .venv/bin/activate

# Upgrade pip
python -m pip install --upgrade pip setuptools wheel

# Install dependencies
pip install -r requirements.txt
```

### Installing Ollama (macOS)

```bash
# Using Homebrew
brew install ollama

# Or download from: https://ollama.com/download/mac

# Pull the Llama3 model
ollama pull llama3:8b
```

### Apple Silicon (M1/M2/M3) Notes

Most packages have native ARM builds. If you encounter issues:

```bash
# Use Rosetta 2 for x86_64 packages
arch -x86_64 /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Or use Conda with ARM support
conda create -n intelli-odm python=3.10
conda activate intelli-odm
conda install -c conda-forge prophet pandas numpy scikit-learn
pip install -r requirements.txt
```

---

## Linux Installation

### Ubuntu/Debian

1. **Install prerequisites**
```bash
sudo apt-get update
sudo apt-get install -y python3 python3-venv python3-pip build-essential
```

2. **Clone the repository**
```bash
git clone https://github.com/your-org/intelli-odm.git
cd intelli-odm
```

3. **Run setup script**
```bash
chmod +x setup.sh
./setup.sh
```

### Fedora/RHEL/CentOS

1. **Install prerequisites**
```bash
sudo dnf install -y python3 python3-pip gcc gcc-c++ make
```

2. **Clone and setup**
```bash
git clone https://github.com/your-org/intelli-odm.git
cd intelli-odm
chmod +x setup.sh
./setup.sh
```

### Arch Linux

1. **Install prerequisites**
```bash
sudo pacman -S python python-pip base-devel
```

2. **Clone and setup**
```bash
git clone https://github.com/your-org/intelli-odm.git
cd intelli-odm
chmod +x setup.sh
./setup.sh
```

### Installing Ollama (Linux)

```bash
# One-line install
curl -fsSL https://ollama.com/install.sh | sh

# Pull the Llama3 model
ollama pull llama3:8b
```

---

## Verify Installation

After installation, verify everything is working:

```bash
# Activate virtual environment
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate

# Check Python version
python --version

# Check installed packages
pip list

# Verify Ollama
ollama --version
ollama list

# Run a simple test
python -c "import pandas, numpy, sklearn, chromadb; print('✅ All core packages imported successfully!')"
```

---

## Troubleshooting

### Issue: Virtual environment activation not working

**Windows PowerShell:**
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

**Windows CMD:**
- Use `.venv\Scripts\activate.bat` instead of `.venv\Scripts\activate`

### Issue: Prophet installation fails

**Solution:**
```bash
# Use Conda
conda install -c conda-forge prophet

# Or use prophet-lite (no Stan backend)
pip install prophet --no-binary prophet
```

### Issue: ChromaDB installation fails

**Solution:**
```bash
# Install system dependencies first
# Ubuntu/Debian:
sudo apt-get install -y libsqlite3-dev

# macOS:
brew install sqlite

# Then reinstall
pip install chromadb --force-reinstall
```

### Issue: CVXPY solver errors

**Solution:**
```bash
# Install additional solvers
pip install cvxopt
pip install scs

# Or use conda
conda install -c conda-forge cvxpy
```

### Issue: Permission denied on Linux/macOS

**Solution:**
```bash
# Make scripts executable
chmod +x setup.sh

# Or run with bash explicitly
bash setup.sh
```

### Issue: Python version too old

**Solution:**

**Ubuntu/Debian:**
```bash
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt-get update
sudo apt-get install python3.11
```

**macOS:**
```bash
brew install python@3.11
```

**Windows:**
- Download from: https://www.python.org/downloads/

### Issue: Out of memory errors

**Solution:**
- Ensure you have at least 8GB RAM
- Close other applications
- For large datasets, process in batches
- Consider using a smaller LLM model

---

## Using Conda (Alternative Method - All Platforms)

Conda often provides better cross-platform compatibility:

```bash
# Create environment
conda create -n intelli-odm python=3.10

# Activate environment
conda activate intelli-odm

# Install conda packages first
conda install -c conda-forge prophet pandas numpy scikit-learn

# Install remaining packages via pip
pip install -r requirements.txt
```

---

## Next Steps

After successful installation:

1. **Configure the system**: Create a `.env` file with your settings
2. **Prepare sample data**: Add CSV files to `examples/sample_input/`
3. **Run the orchestrator**: `python orchestrator.py`
4. **Read the documentation**: Check `Readme.md` for usage examples

---

## Getting Help

If you encounter issues not covered here:

1. Check the [README.md](Readme.md) for more details
2. Review the [GitHub Issues](https://github.com/your-org/intelli-odm/issues)
3. Create a new issue with:
   - Your OS and version
   - Python version
   - Full error message
   - Steps to reproduce

---

**Last Updated:** December 2025

