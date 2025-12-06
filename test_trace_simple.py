#!/usr/bin/env python3
"""Simple test to verify LangSmith traceable works."""

import os
import sys

# Load from .env if available
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Try to load from settings
try:
    from config.settings import settings
    if settings.langchain_tracing_v2:
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
    if settings.langchain_api_key:
        os.environ["LANGCHAIN_API_KEY"] = settings.langchain_api_key
    if settings.langchain_project:
        os.environ["LANGCHAIN_PROJECT"] = settings.langchain_project
except Exception as e:
    print(f"Could not load settings: {e}")

print("Environment check:")
print(f"LANGCHAIN_TRACING_V2: {os.getenv('LANGCHAIN_TRACING_V2')}")
print(f"LANGCHAIN_API_KEY: {'SET' if os.getenv('LANGCHAIN_API_KEY') else 'NOT SET'}")

try:
    from langsmith import traceable
    
    @traceable(name="test_function", run_type="chain")
    def test_func(x: int, y: str) -> str:
        return f"Result: {x} {y}"
    
    result = test_func(42, "hello")
    print(f"✅ Traceable works! Result: {result}")
    print("Check LangSmith dashboard for the trace")
    
except ImportError as e:
    print(f"❌ LangSmith not installed: {e}")
    sys.exit(1)
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

