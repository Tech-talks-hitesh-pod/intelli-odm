# ollama_client.py
# Utility module for setting up and using Ollama LLM client

import ollama
from typing import Optional, Dict, Any

class OllamaClient:
    """
    Wrapper class for Ollama client to interact with llama3:8b model.
    Provides a consistent interface for LLM interactions across agents.
    """
    
    def __init__(self, model_name: str = "llama3:8b", base_url: Optional[str] = None):
        """
        Initialize Ollama client.
        
        Args:
            model_name: Name of the Ollama model to use (default: llama3:8b)
            base_url: Optional custom base URL for Ollama API (default: http://localhost:11434)
        """
        self.model_name = model_name
        self.client = ollama.Client(host=base_url) if base_url else ollama.Client()
        
    def generate(self, prompt: str, system: Optional[str] = None, **kwargs) -> str:
        """
        Generate a response from the LLM.
        
        Args:
            prompt: The user prompt/question
            system: Optional system message to set context
            **kwargs: Additional parameters (temperature, top_p, etc.)
            
        Returns:
            Generated text response
        """
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        
        response = self.client.chat(
            model=self.model_name,
            messages=messages,
            **kwargs
        )
        
        return response['message']['content']
    
    def stream(self, prompt: str, system: Optional[str] = None, **kwargs):
        """
        Stream responses from the LLM (for real-time output).
        
        Args:
            prompt: The user prompt/question
            system: Optional system message to set context
            **kwargs: Additional parameters
            
        Yields:
            Chunks of generated text
        """
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        
        stream = self.client.chat(
            model=self.model_name,
            messages=messages,
            stream=True,
            **kwargs
        )
        
        for chunk in stream:
            if 'message' in chunk and 'content' in chunk['message']:
                yield chunk['message']['content']
    
    def check_model_available(self) -> bool:
        """
        Check if the specified model is available locally.
        
        Returns:
            True if model is available, False otherwise
        """
        try:
            models = self.client.list()
            # Handle both dict and object responses
            if hasattr(models, 'models'):
                # Object response
                model_names = [model.model for model in models.models]
            elif isinstance(models, dict) and 'models' in models:
                # Dict response
                model_list = models['models']
                if model_list and len(model_list) > 0:
                    # Check if it's a list of dicts or objects
                    if isinstance(model_list[0], dict):
                        model_names = [model.get('name', model.get('model', '')) for model in model_list]
                    else:
                        model_names = [getattr(model, 'model', getattr(model, 'name', '')) for model in model_list]
                else:
                    model_names = []
            else:
                model_names = []
            return self.model_name in model_names
        except Exception as e:
            print(f"Error checking model availability: {e}")
            return False
    
    def pull_model(self):
        """
        Pull/download the model if not available locally.
        """
        try:
            print(f"Pulling model {self.model_name}...")
            self.client.pull(self.model_name)
            print(f"Model {self.model_name} pulled successfully!")
        except Exception as e:
            print(f"Error pulling model: {e}")
            raise


def create_llama_client(model_name: str = "llama3:8b", auto_pull: bool = True) -> OllamaClient:
    """
    Factory function to create and configure Ollama client.
    
    Args:
        model_name: Name of the Ollama model to use
        auto_pull: If True, automatically pull the model if not available
        
    Returns:
        Configured OllamaClient instance
    """
    client = OllamaClient(model_name=model_name)
    
    if auto_pull and not client.check_model_available():
        print(f"Model {model_name} not found locally.")
        client.pull_model()
    
    return client

