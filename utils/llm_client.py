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
                            project: Optional[str] = None,
                            endpoint: Optional[str] = None,
                            enabled: Optional[bool] = None):
    """
    Setup LangSmith tracking for agent call monitoring using LangChain callbacks.
    Reads from .env file via settings if parameters are not provided.
    
    Args:
        api_key: LangSmith API key (if None, reads from settings)
        project: Project name for tracking (if None, reads from settings)
        endpoint: LangSmith API endpoint (if None, reads from settings)
        enabled: Whether to enable tracking (if None, reads from settings)
    """
    global _langsmith_initialized, _langsmith_callbacks
    
    # Try to import settings to read from .env
    try:
        from config.settings import settings
        
        # Use settings values if parameters not provided
        if enabled is None:
            enabled = settings.langchain_tracing_v2
        if api_key is None:
            api_key = settings.langchain_api_key
        if project is None:
            project = settings.langchain_project
        if endpoint is None:
            endpoint = settings.langchain_endpoint
    except ImportError:
        # If settings not available, use defaults
        if enabled is None:
            enabled = os.getenv("LANGCHAIN_TRACING_V2", "false").lower() == "true"
        if api_key is None:
            # Support both LANGCHAIN_API_KEY and LANGSMITH_API_KEY
            api_key = os.getenv("LANGCHAIN_API_KEY") or os.getenv("LANGSMITH_API_KEY")
        if project is None:
            project = os.getenv("LANGCHAIN_PROJECT", "intelli-odm")
        if endpoint is None:
            endpoint = os.getenv("LANGCHAIN_ENDPOINT", "https://api.smith.langchain.com")
    
    if not enabled:
        logger.info("LangSmith tracing is disabled (LANGCHAIN_TRACING_V2=false or not set)")
        os.environ.pop("LANGCHAIN_TRACING_V2", None)
        os.environ.pop("LANGCHAIN_API_KEY", None)
        os.environ.pop("LANGCHAIN_PROJECT", None)
        os.environ.pop("LANGCHAIN_ENDPOINT", None)
        _langsmith_initialized = False
        _langsmith_callbacks = None
        return None
    
    if not api_key:
        logger.warning("LangSmith API key not found. Set LANGCHAIN_API_KEY or LANGSMITH_API_KEY in .env file")
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
            # Import LangChainTracer from langchain_core.tracers (correct path for langchain >= 0.1.0)
            from langchain_core.tracers import LangChainTracer
            from langsmith import Client
            
            # Create LangSmith client
            client = Client(api_key=api_key, api_url=endpoint)
            
            # Create callback handler
            _langsmith_callbacks = LangChainTracer(project_name=project, client=client)
            _langsmith_initialized = True
            
            logger.info(f"✅ LangSmith tracking enabled for project: {project}")
            logger.info(f"   API Key: {api_key[:10]}...{api_key[-4:] if len(api_key) > 14 else '***'}")
            logger.info(f"   Endpoint: {endpoint}")
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
                    
                    llm = Ollama(
                        model=self.model,
                        base_url=self.base_url,
                        temperature=temperature,
                        num_predict=max_tokens
                    )
                    
                    # Invoke with callbacks
                    response_text = llm.invoke(prompt, config={"callbacks": [callbacks]})
                    
                    return {
                        'response': response_text.strip(),
                        'model': self.model,
                        'finish_reason': 'completed'
                    }
                except Exception as e:
                    logger.warning(f"LangChain Ollama wrapper failed, using direct call: {e}")
                    logger.info("Note: Direct Ollama calls may not be tracked in LangSmith. Ensure LANGCHAIN_TRACING_V2=true and LANGCHAIN_API_KEY are set for SDK-level tracking.")
            
            # Fallback to direct Ollama API call
            # Note: Direct API calls will be tracked by LangSmith SDK if environment variables are set
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
            response = self.client.list()
            # Ollama returns a ListResponse object with a 'models' attribute
            # Each model is a Model object with a 'model' attribute (not 'name')
            if hasattr(response, 'models'):
                available_models = [model.model for model in response.models]
            else:
                # Fallback for dictionary-like response
                available_models = [model.get('model', model.get('name', '')) for model in response.get('models', [])]
            
            # Check if exact model name exists
            if self.model in available_models:
                return True
            
            # Also check if base model name matches (e.g., "llama3" in "llama3:latest")
            model_base = self.model.split(':')[0]
            if any(model_base in model for model in available_models):
                logger.info(f"Model {self.model} not found, but similar model available. Using available model.")
                return True
            return False
        except Exception as e:
            logger.debug(f"Ollama availability check failed: {e}")
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
                    from langchain_core.messages import HumanMessage
                    
                    llm = ChatOpenAI(
                        model=self.model,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        openai_api_key=self.client.api_key
                    )
                    
                    messages = [HumanMessage(content=prompt)]
                    # Invoke with callbacks
                    response = llm.invoke(messages, config={"callbacks": [callbacks]})
                    
                    return {
                        'response': response.content.strip(),
                        'model': self.model,
                        'finish_reason': 'completed'
                    }
                except Exception as e:
                    logger.warning(f"LangChain OpenAI wrapper failed, using direct call: {e}")
                    logger.info("Note: Direct OpenAI calls may not be tracked in LangSmith. Ensure LANGCHAIN_TRACING_V2=true and LANGCHAIN_API_KEY are set for SDK-level tracking.")
            
            # Fallback to direct OpenAI API call
            # Note: Direct API calls will be tracked by LangSmith SDK if environment variables are set
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
            langsmith_config: Optional LangSmith configuration for tracking.
                If None, will read from .env file via settings.
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
        # Setup LangSmith tracking - will read from .env if langsmith_config is None
        if langsmith_config:
            setup_langsmith_tracking(
                api_key=langsmith_config.get('api_key'),
                project=langsmith_config.get('project'),
                endpoint=langsmith_config.get('endpoint'),
                enabled=langsmith_config.get('enabled')
            )
        else:
            # Read from .env file via settings
            setup_langsmith_tracking()
        
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