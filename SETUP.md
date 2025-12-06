# Ollama Setup Guide for Intelli-ODM

This guide will help you set up Ollama for the Intelli-ODM project.

## Quick Setup

### Option 1: Automated Setup (Recommended)

Run the setup script:

```bash
./setup_ollama.sh
```

This script will:
- Check if Ollama is installed
- Verify Ollama service is running
- Pull the llama3:8b model if not available
- Set up Python virtual environment
- Install all Python dependencies

### Option 2: Manual Setup

#### Step 1: Install Ollama

**macOS/Linux:**
```bash
curl -fsSL https://ollama.com/install.sh | sh
```

**Windows:**
Download the installer from [https://ollama.com/download](https://ollama.com/download)

#### Step 2: Verify Ollama Installation

```bash
ollama --version
```

#### Step 3: Start Ollama Service

Ollama usually runs automatically as a service. If not, start it:

```bash
ollama serve
```

Verify it's running:
```bash
curl http://localhost:11434
```

You should see: "Ollama is running"

#### Step 4: Pull the llama3:8b Model

```bash
ollama pull llama3:8b
```

This will download the model (approximately 4.7GB). The first time may take several minutes.

#### Step 5: Set Up Python Environment

```bash
# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

#### Step 6: Verify Setup

Run the verification script:

```bash
python verify_ollama.py
```

This will check:
- Ollama service is running
- llama3:8b model is available
- LLM generation works correctly

## Troubleshooting

### Ollama service not running

If you get connection errors:

1. Check if Ollama is installed: `ollama --version`
2. Start the service: `ollama serve`
3. Check if port 11434 is available: `lsof -i :11434`

### Model not found

If the model is not available:

1. Pull it manually: `ollama pull llama3:8b`
2. List available models: `ollama list`
3. Check available disk space (model is ~4.7GB)

### Python import errors

If you get import errors:

1. Make sure virtual environment is activated
2. Reinstall dependencies: `pip install -r requirements.txt`
3. Check Python version: `python3 --version` (should be 3.8+)

### Connection refused errors

If you see connection refused:

1. Ensure Ollama service is running: `ollama serve`
2. Check firewall settings
3. Verify Ollama is listening on port 11434

## Testing the Setup

After setup, test with a simple example:

```python
from ollama_client import create_llama_client

# Create client
client = create_llama_client()

# Generate a response
response = client.generate(
    prompt="Hello! Can you confirm you're working?",
    system="You are a helpful assistant."
)

print(response)
```

Or run the example script:

```bash
python example_usage.py
```

## Next Steps

Once setup is complete, you can:

1. Run the full workflow: `python example_usage.py`
2. Use the orchestrator in your own code
3. Customize agents for your specific use case

For more information, see the main [README.md](Readme.md).

