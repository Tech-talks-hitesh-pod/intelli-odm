# example_usage.py
# Example script demonstrating how to use Ollama with the Intelli-ODM project

from ollama_client import create_llama_client
from orchestrator import OrchestratorAgent

def main():
    """
    Example usage of the Intelli-ODM system with Ollama.
    """
    # Initialize Ollama client
    print("Initializing Ollama client...")
    llama_client = create_llama_client(model_name="llama3:8b", auto_pull=True)
    
    # Example constraint parameters for procurement agent
    constraint_params = {
        "budget": 100000,
        "min_order_quantity": 10,
        "capacity": 5000
    }
    
    # Initialize orchestrator with Ollama client
    orchestrator = OrchestratorAgent(
        llama_client=llama_client,
        constraint_params=constraint_params
    )
    
    # Example input data
    input_data = {
        "files": {
            "sales": "path/to/sales.csv",
            "inventory": "path/to/inventory.csv",
            "price": "path/to/price.csv"
        },
        "product_description": "High-end wireless headphones with noise cancellation"
    }
    
    price_options = [99.99, 129.99, 149.99]
    
    # Run the workflow
    print("Running workflow...")
    final_recommendation = orchestrator.run_workflow(input_data, price_options)
    
    print("Final Recommendation:", final_recommendation)
    
    # Example: Direct LLM usage
    print("\n--- Direct LLM Example ---")
    response = llama_client.generate(
        prompt="What are the key attributes of wireless headphones?",
        system="You are a product analysis expert."
    )
    print("LLM Response:", response)


if __name__ == "__main__":
    main()

