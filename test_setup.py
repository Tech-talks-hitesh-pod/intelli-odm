#!/usr/bin/env python3
"""
Quick test script to verify thedemo setup.
"""

import os
import sys
import requests

def test_python_packages():
    """Test that all required Python packages are installed."""
    print("🐍 Testing Python packages...")
    
    try:
        import streamlit
        import pandas
        import plotly
        import numpy
        import chromadb
        import sentence_transformers
        print("✅ Core packages installed")
        
        import ollama
        print("✅ Ollama package available")
        
        return True
    except ImportError as e:
        print(f"❌ Missing package: {e}")
        return False

def test_ollama_service():
    """Test Ollama service connectivity."""
    print("🤖 Testing Ollama service...")
    
    try:
        response = requests.get("http://localhost:11434/api/version", timeout=5)
        if response.status_code == 200:
            print("✅ Ollama service running")
            return True
        else:
            print(f"⚠️ Ollama service issue: Status {response.status_code}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"❌ Ollama service not accessible: {e}")
        return False

def test_ollama_model():
    """Test that Llama3 model is available."""
    print("📦 Testing Ollama model...")
    
    try:
        import ollama
        client = ollama.Client()
        models = client.list()
        
        model_names = [model['name'] for model in models.get('models', [])]
        if 'llama3:8b' in model_names:
            print("✅ Llama3 model available")
            return True
        else:
            print(f"⚠️ Llama3 model not found. Available: {model_names}")
            return False
    except Exception as e:
        print(f"❌ Model check failed: {e}")
        return False

def test_knowledge_base():
    """Test knowledge base initialization."""
    print("🧠 Testing knowledge base...")
    
    try:
        from shared_knowledge_base import SharedKnowledgeBase
        kb = SharedKnowledgeBase()
        
        size = kb.get_collection_size()
        print(f"✅ Knowledge base ready ({size} products)")
        return True
    except Exception as e:
        print(f"❌ Knowledge base issue: {e}")
        return False

def test_demo_scenarios():
    """Test demo scenarios."""
    print("🎬 Testing demo scenarios...")
    
    try:
        from demo_scenarios import DemoScenarios
        
        scenarios = DemoScenarios.get_available_scenarios()
        print(f"✅ {len(scenarios)} demo scenarios available")
        
        # Test one scenario
        scenario_data = DemoScenarios.get_scenario_data("seasonal_demand")
        print(f"✅ Scenario data structure valid")
        return True
    except Exception as e:
        print(f"❌ Demo scenarios issue: {e}")
        return False

def test_file_structure():
    """Test that required files exist."""
    print("📁 Testing file structure...")
    
    required_files = [
        "streamlit_app.py",
        "demo_scenarios.py",
        "shared_knowledge_base.py",
        "launch_demo.sh",
        ".env"
    ]
    
    missing_files = []
    for file in required_files:
        if not os.path.exists(file):
            missing_files.append(file)
    
    if missing_files:
        print(f"❌ Missing files: {missing_files}")
        return False
    else:
        print("✅ All required files present")
        return True

def main():
    """Run all tests."""
    print("🎯 Intelli-ODMDemo System Test")
    print("=" * 50)
    
    tests = [
        test_file_structure,
        test_python_packages,
        test_ollama_service,
        test_ollama_model,
        test_knowledge_base,
        test_demo_scenarios
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
            print()
        except Exception as e:
            print(f"❌ Test failed with exception: {e}")
            results.append(False)
            print()
    
    # Summary
    passed = sum(results)
    total = len(results)
    
    print("=" * 50)
    print(f"📊 Test Results: {passed}/{total} passed")
    
    if passed == total:
        print("🎉 All tests passed!demo is ready!")
        print()
        print("🚀 To start demo:")
        print("   ./launch_demo.sh")
        print()
        print("🌐 Demo will be available at:")
        print("   http://localhost:8501")
        return 0
    else:
        print("⚠️ Some tests failed. Please check the issues above.")
        print()
        print("💡 Try running setup again:")
        print("   ./setup.sh")
        return 1

if __name__ == "__main__":
    sys.exit(main())