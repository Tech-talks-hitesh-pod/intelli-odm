"""LLM client factory for Ollama and OpenAI integration with LangSmith tracking."""

import logging
import os
from typing import Dict, Any, Optional, Union
from abc import ABC, abstractmethod
import json
import time

logger = logging.getLogger(__name__)

# LangSmith/LangChain tracking setup
_langsmith_initialized = False
_langsmith_callbacks = None

def setup_langsmith_tracking(api_key: Optional[str] = None, 
                            project: str = "intelli-odm",
                            endpoint: str = "https://api.smith.langchain.com",
                            enabled: bool = False):
    """
    Setup LangSmith tracking for agent call monitoring using LangChain callbacks.
    
    Args:
        api_key: LangSmith API key
        project: Project name for tracking
        endpoint: LangSmith API endpoint
        enabled: Whether to enable tracking
    """
    global _langsmith_initialized, _langsmith_callbacks
    
    if not enabled or not api_key:
        os.environ.pop("LANGCHAIN_TRACING_V2", None)
        os.environ.pop("LANGCHAIN_API_KEY", None)
        os.environ.pop("LANGCHAIN_PROJECT", None)
        os.environ.pop("LANGCHAIN_ENDPOINT", None)
        _langsmith_initialized = False
        _langsmith_callbacks = None
        return None
    
    try:
        # Set environment variables for LangSmith
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        os.environ["LANGCHAIN_API_KEY"] = api_key
        os.environ["LANGCHAIN_PROJECT"] = project
        os.environ["LANGCHAIN_ENDPOINT"] = endpoint
        
        # Try to import and setup LangChain callbacks
        try:
            from langchain.callbacks import LangChainTracer
            from langsmith import Client
            
            # Create LangSmith client
            client = Client(api_key=api_key, api_url=endpoint)
            
            # Create callback handler
            _langsmith_callbacks = LangChainTracer(project_name=project, client=client)
            _langsmith_initialized = True
            
            logger.info(f"LangSmith tracking enabled for project: {project}")
            return _langsmith_callbacks
            
        except ImportError:
            logger.warning("LangChain not installed. LangSmith tracking may not work. Install with: pip install langchain langsmith")
            _langsmith_initialized = False
            _langsmith_callbacks = None
            return None
        except Exception as e:
            logger.warning(f"Failed to setup LangChain callbacks: {e}")
            # Still set environment variables for basic tracking
            _langsmith_initialized = True
            return None
            
    except Exception as e:
        logger.error(f"Failed to setup LangSmith tracking: {e}")
        _langsmith_initialized = False
        _langsmith_callbacks = None
        return None

def get_langsmith_callbacks():
    """Get LangSmith callbacks if initialized."""
    return _langsmith_callbacks

class LLMError(Exception):
    """Base exception for LLM operations."""
    pass

class LLMClient(ABC):
    """Abstract base class for LLM clients."""
    
    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """Generate response from LLM."""
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """Check if LLM service is available."""
        pass

class OllamaClient(LLMClient):
    """Ollama client implementation."""
    
    def __init__(self, base_url: str = "http://localhost:11434", 
                 model: str = "llama3:8b", timeout: int = 300):
        """Initialize Ollama client."""
        self.base_url = base_url.rstrip('/')
        self.model = model
        self.timeout = timeout
        
        # Try to import ollama
        try:
            import ollama
            self.client = ollama.Client(host=base_url)
            logger.info(f"Initialized Ollama client with model {model}")
        except ImportError:
            raise LLMError("Ollama package not installed. Run: pip install ollama")
        except Exception as e:
            raise LLMError(f"Failed to initialize Ollama client: {e}")
    
    def generate(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """Generate response using Ollama with LangSmith tracking."""
        try:
            # Extract parameters
            temperature = kwargs.get('temperature', 0.1)
            max_tokens = kwargs.get('max_tokens', 500)
            
            # Get LangSmith callbacks if available
            callbacks = get_langsmith_callbacks()
            
            # Use LangChain wrapper if callbacks are available
            if callbacks:
                try:
                    from langchain_community.llms import Ollama
                    from langchain.callbacks.manager import CallbackManager
                    
                    llm = Ollama(
                        model=self.model,
                        base_url=self.base_url,
                        temperature=temperature,
                        num_predict=max_tokens,
                        callback_manager=CallbackManager([callbacks])
                    )
                    
                    response_text = llm.invoke(prompt)
                    
                    return {
                        'response': response_text.strip(),
                        'model': self.model,
                        'finish_reason': 'completed'
                    }
                except Exception as e:
                    logger.warning(f"LangChain Ollama wrapper failed, using direct call: {e}")
            
            # Fallback to direct Ollama API call
            response = self.client.generate(
                model=self.model,
                prompt=prompt,
                options={
                    'temperature': temperature,
                    'num_predict': max_tokens
                }
            )
            
            return {
                'response': response['response'].strip(),
                'model': self.model,
                'finish_reason': 'completed'
            }
            
        except Exception as e:
            logger.error(f"Ollama generation failed: {e}")
            raise LLMError(f"Ollama generation failed: {e}")
    
    def is_available(self) -> bool:
        """Check if Ollama service is available."""
        try:
            # Try to list models to test connection
            models = self.client.list()
            return any(model['name'] == self.model for model in models.get('models', []))
        except Exception:
            return False

class OpenAIClient(LLMClient):
    """OpenAI client implementation."""
    
    def __init__(self, api_key: str, model: str = "gpt-4o-mini", temperature: float = 0.1):
        """Initialize OpenAI client."""
        self.model = model
        self.temperature = temperature
        
        # Try to import openai
        try:
            import openai
            self.client = openai.Client(api_key=api_key)
            logger.info(f"Initialized OpenAI client with model {model}")
        except ImportError:
            raise LLMError("OpenAI package not installed. Run: pip install openai")
        except Exception as e:
            raise LLMError(f"Failed to initialize OpenAI client: {e}")
    
    def generate(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """Generate response using OpenAI with LangSmith tracking."""
        try:
            # Extract parameters
            temperature = kwargs.get('temperature', self.temperature)
            max_tokens = kwargs.get('max_tokens', 500)
            
            # Get LangSmith callbacks if available
            callbacks = get_langsmith_callbacks()
            
            # Use LangChain wrapper if callbacks are available
            if callbacks:
                try:
                    from langchain_openai import ChatOpenAI
                    from langchain.callbacks.manager import CallbackManager
                    from langchain.schema import HumanMessage
                    
                    llm = ChatOpenAI(
                        model=self.model,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        openai_api_key=self.client.api_key,
                        callback_manager=CallbackManager([callbacks])
                    )
                    
                    messages = [HumanMessage(content=prompt)]
                    response = llm.invoke(messages)
                    
                    return {
                        'response': response.content.strip(),
                        'model': self.model,
                        'finish_reason': 'completed'
                    }
                except Exception as e:
                    logger.warning(f"LangChain OpenAI wrapper failed, using direct call: {e}")
            
            # Fallback to direct OpenAI API call
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                temperature=temperature,
                max_tokens=max_tokens
            )
            
            return {
                'response': response.choices[0].message.content.strip(),
                'model': self.model,
                'finish_reason': response.choices[0].finish_reason,
                'usage': {
                    'prompt_tokens': response.usage.prompt_tokens,
                    'completion_tokens': response.usage.completion_tokens,
                    'total_tokens': response.usage.total_tokens
                }
            }
            
        except Exception as e:
            logger.error(f"OpenAI generation failed: {e}")
            raise LLMError(f"OpenAI generation failed: {e}")
    
    def is_available(self) -> bool:
        """Check if OpenAI service is available."""
        try:
            # Try a simple completion to test connection
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": "test"}],
                max_tokens=1
            )
            return True
        except Exception:
            return False

class LLMClientFactory:
    """Factory for creating LLM clients."""
    
    @staticmethod
    def create_client(config: Dict[str, Any], langsmith_config: Optional[Dict[str, Any]] = None) -> LLMClient:
        """
        Create LLM client based on configuration.
        
        Args:
            config: LLM configuration dictionary
            langsmith_config: Optional LangSmith configuration for tracking
                {
                    'enabled': bool,
                    'api_key': str,
                    'project': str,
                    'endpoint': str
                }
            
        Returns:
            LLM client instance
            
        Raises:
            LLMError: If client creation fails
        """
        # Setup LangSmith tracking if configured
        if langsmith_config:
            setup_langsmith_tracking(
                api_key=langsmith_config.get('api_key'),
                project=langsmith_config.get('project', 'intelli-odm'),
                endpoint=langsmith_config.get('endpoint', 'https://api.smith.langchain.com'),
                enabled=langsmith_config.get('enabled', False)
            )
        
        provider = config.get('provider', 'ollama')
        
        if provider == 'ollama':
            return OllamaClient(
                base_url=config.get('base_url', 'http://localhost:11434'),
                model=config.get('model', 'llama3:8b'),
                timeout=config.get('timeout', 300)
            )
        
        elif provider == 'openai':
            api_key = config.get('api_key')
            if not api_key:
                raise LLMError("OpenAI API key is required")
            
            return OpenAIClient(
                api_key=api_key,
                model=config.get('model', 'gpt-4o-mini'),
                temperature=config.get('temperature', 0.1)
            )
        
        else:
            raise LLMError(f"Unsupported LLM provider: {provider}")

def parse_json_response(response: str, fallback: Optional[Dict] = None) -> Dict[str, Any]:
    """
    Parse JSON response from LLM, with fallback handling.
    
    Args:
        response: LLM response string
        fallback: Fallback dict if parsing fails
        
    Returns:
        Parsed JSON dictionary
    """
    # Try to find JSON in the response
    response = response.strip()
    
    # Look for JSON block markers
    if '```json' in response:
        start = response.find('```json') + 7
        end = response.find('```', start)
        if end != -1:
            json_str = response[start:end].strip()
        else:
            json_str = response[start:].strip()
    elif response.startswith('{') and response.endswith('}'):
        json_str = response
    else:
        # Try to extract JSON from the response
        import re
        json_pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
        matches = re.findall(json_pattern, response)
        if matches:
            json_str = matches[0]
        else:
            logger.warning(f"No JSON found in response: {response[:100]}...")
            return fallback or {}
    
    try:
        return json.loads(json_str)
    except json.JSONDecodeError as e:
        logger.warning(f"Failed to parse JSON: {e}. Response: {json_str[:100]}...")
        return fallback or {}

def retry_llm_call(client: LLMClient, prompt: str, max_retries: int = 3, 
                   **kwargs) -> Dict[str, Any]:
    """
    Retry LLM call with exponential backoff.
    
    Args:
        client: LLM client instance
        prompt: Prompt to send
        max_retries: Maximum number of retries
        **kwargs: Additional arguments for generate()
        
    Returns:
        LLM response
        
    Raises:
        LLMError: If all retries fail
    """
    last_error = None
    
    for attempt in range(max_retries):
        try:
            return client.generate(prompt, **kwargs)
        except Exception as e:
            last_error = e
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt  # Exponential backoff
                logger.warning(f"LLM call failed (attempt {attempt + 1}), retrying in {wait_time}s: {e}")
                time.sleep(wait_time)
            else:
                logger.error(f"LLM call failed after {max_retries} attempts: {e}")
    
    raise LLMError(f"LLM call failed after {max_retries} attempts: {last_error}")