#!/bin/bash

# setup_ollama.sh
# Script to set up Ollama for the Intelli-ODM project

set -e  # Exit on error

echo "========================================="
echo "Setting up Ollama for Intelli-ODM"
echo "========================================="
echo ""

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Step 1: Check if Ollama is installed
echo -e "${YELLOW}Step 1: Checking Ollama installation...${NC}"
if command -v ollama &> /dev/null; then
    echo -e "${GREEN}✓ Ollama is installed${NC}"
    ollama --version
else
    echo -e "${RED}✗ Ollama is not installed${NC}"
    echo ""
    echo "Please install Ollama first:"
    echo "  macOS/Linux: curl -fsSL https://ollama.com/install.sh | sh"
    echo "  Or visit: https://ollama.com/download"
    exit 1
fi

echo ""

# Step 2: Check if Ollama service is running
echo -e "${YELLOW}Step 2: Checking Ollama service...${NC}"
if curl -s http://localhost:11434 > /dev/null 2>&1; then
    echo -e "${GREEN}✓ Ollama service is running${NC}"
else
    echo -e "${YELLOW}⚠ Ollama service is not running. Starting it...${NC}"
    # Try to start Ollama (it usually runs as a service)
    ollama serve > /dev/null 2>&1 &
    sleep 2
    if curl -s http://localhost:11434 > /dev/null 2>&1; then
        echo -e "${GREEN}✓ Ollama service started${NC}"
    else
        echo -e "${RED}✗ Could not start Ollama service${NC}"
        echo "Please start Ollama manually: ollama serve"
        exit 1
    fi
fi

echo ""

# Step 3: Check if llama3:8b model is available
echo -e "${YELLOW}Step 3: Checking for llama3:8b model...${NC}"
if ollama list | grep -q "llama3:8b"; then
    echo -e "${GREEN}✓ llama3:8b model is already installed${NC}"
else
    echo -e "${YELLOW}⚠ llama3:8b model not found. Pulling it...${NC}"
    echo "This may take several minutes depending on your internet connection..."
    ollama pull llama3:8b
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓ llama3:8b model pulled successfully${NC}"
    else
        echo -e "${RED}✗ Failed to pull llama3:8b model${NC}"
        exit 1
    fi
fi

echo ""

# Step 4: Install Python dependencies
echo -e "${YELLOW}Step 4: Installing Python dependencies...${NC}"
if [ -d "venv" ]; then
    echo "Virtual environment found. Activating it..."
    source venv/bin/activate
else
    echo "Creating virtual environment..."
    python3 -m venv venv
    source venv/bin/activate
fi

pip install --upgrade pip
pip install -r requirements.txt

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Python dependencies installed successfully${NC}"
else
    echo -e "${RED}✗ Failed to install Python dependencies${NC}"
    exit 1
fi

echo ""
echo -e "${GREEN}=========================================${NC}"
echo -e "${GREEN}Setup completed successfully!${NC}"
echo -e "${GREEN}=========================================${NC}"
echo ""
echo "You can now use Ollama in your project:"
echo "  python example_usage.py"
echo ""
echo "Or test the connection:"
echo "  python verify_ollama.py"

