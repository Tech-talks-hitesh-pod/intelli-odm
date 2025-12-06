#!/usr/bin/env python3
"""
verify_ollama.py
Script to verify Ollama installation and connection
"""

import sys
import requests
from ollama_client import create_llama_client, OllamaClient

def check_ollama_service():
    """Check if Ollama service is running"""
    try:
        response = requests.get("http://localhost:11434", timeout=5)
        if response.status_code == 200:
            print("✓ Ollama service is running on http://localhost:11434")
            return True
        else:
            print(f"✗ Ollama service returned status code: {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print("✗ Cannot connect to Ollama service at http://localhost:11434")
        print("  Make sure Ollama is running: ollama serve")
        return False
    except Exception as e:
        print(f"✗ Error checking Ollama service: {e}")
        return False

def check_model_availability():
    """Check if llama3:8b model is available"""
    try:
        client = OllamaClient(model_name="llama3:8b")
        if client.check_model_available():
            print("✓ llama3:8b model is available")
            return True
        else:
            print("✗ llama3:8b model is not available")
            print("  Pull it with: ollama pull llama3:8b")
            return False
    except Exception as e:
        print(f"✗ Error checking model availability: {e}")
        return False

def test_llm_generation():
    """Test LLM generation with a simple prompt"""
    try:
        print("\nTesting LLM generation...")
        client = create_llama_client(model_name="llama3:8b", auto_pull=False)
        
        response = client.generate(
            prompt="Say 'Hello, Ollama is working!' in one sentence.",
            system="You are a helpful assistant."
        )
        
        print("✓ LLM generation successful")
        print(f"  Response: {response[:100]}..." if len(response) > 100 else f"  Response: {response}")
        return True
    except Exception as e:
        print(f"✗ Error testing LLM generation: {e}")
        return False

def main():
    """Run all verification checks"""
    print("=" * 50)
    print("Ollama Verification Script")
    print("=" * 50)
    print()
    
    checks = []
    
    # Check 1: Ollama service
    print("1. Checking Ollama service...")
    checks.append(check_ollama_service())
    print()
    
    # Check 2: Model availability
    print("2. Checking model availability...")
    checks.append(check_model_availability())
    print()
    
    # Check 3: LLM generation (only if previous checks passed)
    if all(checks[:2]):
        checks.append(test_llm_generation())
    else:
        print("3. Skipping LLM generation test (previous checks failed)")
        checks.append(False)
    
    print()
    print("=" * 50)
    if all(checks):
        print("✓ All checks passed! Ollama is ready to use.")
        return 0
    else:
        print("✗ Some checks failed. Please fix the issues above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())

