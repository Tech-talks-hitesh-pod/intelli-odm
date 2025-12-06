# Intelli-ODM

This project leverages Python alongside the powerful Ollama llama3:8b LLM model to deliver advanced AI capabilities.

## Tech Stack

- **Python**  
  Used for core application logic and orchestration.

- **Ollama: llama3:8b**  
  Integrated as the foundational large language model. See [Ollama Documentation](https://ollama.com/) for details.

## Getting Started

### Prerequisites

- Python 3.8+
- [Ollama](https://ollama.com/) installed locally

### Quick Installation

**Option 1: Automated Setup (Recommended)**

Run the setup script:
```bash
./setup_ollama.sh
```

**Option 2: Manual Installation**

See [SETUP.md](SETUP.md) for detailed manual installation instructions.

### Quick Start

1. **Run the setup script** (if not done already):
   ```bash
   ./setup_ollama.sh
   ```

2. **Verify the installation**:
   ```bash
   python verify_ollama.py
   ```

3. **Test with example**:
   ```bash
   python example_usage.py
   ```

For detailed setup instructions, troubleshooting, and more information, see [SETUP.md](SETUP.md).

## Usage

### Basic Usage

The project includes an `ollama_client.py` module that provides a convenient wrapper for interacting with Ollama:

```python
from ollama_client import create_llama_client

# Create a client (automatically pulls model if not available)
llama_client = create_llama_client(model_name="llama3:8b", auto_pull=True)

# Generate a response
response = llama_client.generate(
    prompt="What are the key attributes of wireless headphones?",
    system="You are a product analysis expert."
)
print(response)
```

### Using with Orchestrator

```python
from ollama_client import create_llama_client
from orchestrator import OrchestratorAgent

# Initialize Ollama client
llama_client = create_llama_client()

# Create orchestrator with constraints
constraint_params = {
    "budget": 100000,
    "min_order_quantity": 10,
    "capacity": 5000
}

orchestrator = OrchestratorAgent(
    llama_client=llama_client,
    constraint_params=constraint_params
)

# Run workflow
input_data = {
    "files": {
        "sales": "path/to/sales.csv",
        "inventory": "path/to/inventory.csv",
        "price": "path/to/price.csv"
    },
    "product_description": "High-end wireless headphones with noise cancellation"
}

price_options = [99.99, 129.99, 149.99]
final_recommendation = orchestrator.run_workflow(input_data, price_options)
```

See `example_usage.py` for a complete example.

### OllamaClient API

The `OllamaClient` class provides the following methods:

- `generate(prompt, system=None, **kwargs)`: Generate a response from the LLM
- `stream(prompt, system=None, **kwargs)`: Stream responses in real-time
- `check_model_available()`: Check if the model is available locally
- `pull_model()`: Download the model if not available

## License

See [LICENSE](./LICENSE) for details.
