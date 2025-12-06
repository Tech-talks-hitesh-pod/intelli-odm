#!/usr/bin/env python3
"""
Test script to verify LangSmith tracking is working.
"""

import os
import sys
import logging

# IMPORTANT: Load environment variables from .env file BEFORE checking them
try:
    from dotenv import load_dotenv
    load_dotenv()
    print("✅ Loaded .env file")
except ImportError:
    print("⚠️  python-dotenv not installed. Install with: pip install python-dotenv")
    print("   Trying to load from config.settings instead...")
    try:
        from config.settings import settings
        # Set environment variables from settings
        if settings.langchain_tracing_v2:
            os.environ["LANGCHAIN_TRACING_V2"] = "true"
        if settings.langchain_api_key:
            os.environ["LANGCHAIN_API_KEY"] = settings.langchain_api_key
        if settings.langchain_project:
            os.environ["LANGCHAIN_PROJECT"] = settings.langchain_project
        if settings.langchain_endpoint:
            os.environ["LANGCHAIN_ENDPOINT"] = settings.langchain_endpoint
        print("✅ Loaded environment variables from config.settings")
    except Exception as e:
        print(f"⚠️  Could not load from settings: {e}")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_langsmith_setup():
    """Test LangSmith setup and configuration."""
    print("="*80)
    print("LANGSMITH TRACKING TEST")
    print("="*80)
    
    # Load from settings (which reads .env automatically)
    try:
        from config.settings import settings
        print("\n✅ Loaded settings from config.settings (reads .env automatically)")
        
        # Set environment variables from settings for compatibility
        if settings.langchain_tracing_v2:
            os.environ["LANGCHAIN_TRACING_V2"] = "true"
        if settings.langchain_api_key:
            os.environ["LANGCHAIN_API_KEY"] = settings.langchain_api_key
        if settings.langchain_project:
            os.environ["LANGCHAIN_PROJECT"] = settings.langchain_project
        if settings.langchain_endpoint:
            os.environ["LANGCHAIN_ENDPOINT"] = settings.langchain_endpoint
    except Exception as e:
        print(f"⚠️  Could not load from settings: {e}")
    
    # Check environment variables (now set from settings)
    print("\n1. Checking Environment Variables:")
    print("-" * 60)
    tracing_v2 = os.getenv("LANGCHAIN_TRACING_V2")
    api_key = os.getenv("LANGCHAIN_API_KEY") or os.getenv("LANGSMITH_API_KEY")
    project = os.getenv("LANGCHAIN_PROJECT")
    endpoint = os.getenv("LANGCHAIN_ENDPOINT")
    
    print(f"   LANGCHAIN_TRACING_V2: {tracing_v2}")
    print(f"   LANGCHAIN_API_KEY: {'SET' if api_key else 'NOT SET'}")
    if api_key:
        if api_key == "your_langsmith_api_key_here":
            print(f"   ⚠️  API Key is still the placeholder - replace with your actual key!")
        else:
            print(f"   API Key (first 10 chars): {api_key[:10]}...")
    print(f"   LANGCHAIN_PROJECT: {project}")
    print(f"   LANGCHAIN_ENDPOINT: {endpoint}")
    
    if not tracing_v2 or tracing_v2.lower() != "true":
        print("\n❌ LANGCHAIN_TRACING_V2 is not set to 'true'")
        print("   Check your .env file has: LANGCHAIN_TRACING_V2=true")
        return False
    
    if not api_key or api_key == "your_langsmith_api_key_here":
        print("\n❌ LANGCHAIN_API_KEY is not set or is still a placeholder")
        print("   Update your .env file with your actual LangSmith API key")
        print("   Get it from: https://smith.langchain.com → Settings → API Keys")
        return False
    
    print("\n✅ Environment variables are set correctly")
    
    # Test LangSmith setup
    print("\n2. Testing LangSmith Setup:")
    print("-" * 60)
    try:
        from utils.llm_client import setup_langsmith_tracking, get_langsmith_callbacks
        
        callbacks = setup_langsmith_tracking()
        if callbacks:
            print("✅ LangSmith callbacks initialized")
        else:
            print("⚠️  LangSmith callbacks not available (may still work with SDK-level tracking)")
        
        # Check if callbacks are available
        callbacks_check = get_langsmith_callbacks()
        print(f"   Callbacks available: {callbacks_check is not None}")
        
    except Exception as e:
        print(f"❌ Failed to setup LangSmith: {e}")
        return False
    
    # Test traceable decorator
    print("\n3. Testing LangSmith Traceable Decorator:")
    print("-" * 60)
    try:
        from langsmith import traceable
        
        # Use valid run_type: "chain", "llm", "tool", "retriever", "embedding", "prompt", "parser"
        @traceable(name="test_trace", run_type="chain", tags=["test"])
        def test_function():
            return "test result"
        
        result = test_function()
        print(f"✅ Traceable decorator works: {result}")
        
    except ImportError:
        print("❌ langsmith package not installed. Install with: pip install langsmith")
        return False
    except Exception as e:
        print(f"❌ Traceable decorator failed: {e}")
        return False
    
    # Test actual LLM call (if available)
    print("\n4. Testing LLM Call with Tracking:")
    print("-" * 60)
    try:
        from utils.llm_client import LLMClientFactory
        from config.settings import settings
        
        llm_config = {
            'provider': settings.llm_provider,
            'openai_api_key': settings.openai_api_key if settings.llm_provider == 'openai' else None,
            'openai_model': settings.openai_model if settings.llm_provider == 'openai' else 'gpt-4o-mini',
            'openai_temperature': 0.1,
            'ollama_base_url': settings.ollama_base_url if settings.llm_provider == 'ollama' else 'http://localhost:11434',
            'ollama_model': settings.ollama_model if settings.llm_provider == 'ollama' else 'llama3:8b'
        }
        
        llm_client = LLMClientFactory.create_client(llm_config)
        print(f"✅ LLM client created: {type(llm_client).__name__}")
        
        # Make a test call
        test_prompt = "Say 'Hello, LangSmith!' in one sentence."
        print(f"   Making test LLM call...")
        result = llm_client.generate(test_prompt, temperature=0.1, max_tokens=50)
        print(f"✅ LLM call completed")
        print(f"   Response: {result.get('response', 'N/A')[:100]}")
        print("\n✅ Check LangSmith dashboard at https://smith.langchain.com")
        print(f"   Project: {project or 'intelli-odm'}")
        
    except Exception as e:
        print(f"⚠️  LLM test failed: {e}")
        print("   This is OK if LLM is not configured, but tracking should still work")
    
    print("\n" + "="*80)
    print("TEST COMPLETE")
    print("="*80)
    print("\nNext steps:")
    print("1. Check your .env file has:")
    print("   LANGCHAIN_TRACING_V2=true")
    print("   LANGCHAIN_API_KEY=your_key_here")
    print("   LANGCHAIN_PROJECT=intelli-odm")
    print("\n2. Visit https://smith.langchain.com to see traces")
    print("\n3. Make a prediction in the Streamlit app and check for new traces")
    
    return True

if __name__ == "__main__":
    success = test_langsmith_setup()
    sys.exit(0 if success else 1)

