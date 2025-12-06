#!/usr/bin/env python3
"""
Simple script to check LangSmith environment variables from .env file.
"""

import os
import sys

def check_env_file():
    """Check .env file directly."""
    print("="*80)
    print("LANGSMITH ENVIRONMENT CHECK")
    print("="*80)
    
    env_file = ".env"
    if not os.path.exists(env_file):
        print(f"\n❌ {env_file} file not found!")
        print("   Create it with:")
        print("   LANGCHAIN_TRACING_V2=true")
        print("   LANGCHAIN_API_KEY=your_key_here")
        print("   LANGCHAIN_PROJECT=intelli-odm")
        return False
    
    print(f"\n✅ Found {env_file} file")
    print("\nReading .env file:")
    print("-" * 60)
    
    env_vars = {}
    with open(env_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#') and '=' in line:
                key, value = line.split('=', 1)
                key = key.strip()
                value = value.strip().strip('"').strip("'")
                env_vars[key] = value
                print(f"   {key}: {value}")
    
    # Check required variables
    print("\n" + "="*80)
    print("STATUS CHECK:")
    print("="*80)
    
    issues = []
    
    if 'LANGCHAIN_TRACING_V2' not in env_vars:
        issues.append("❌ LANGCHAIN_TRACING_V2 not found in .env")
    elif env_vars['LANGCHAIN_TRACING_V2'].lower() != 'true':
        issues.append(f"❌ LANGCHAIN_TRACING_V2 is '{env_vars['LANGCHAIN_TRACING_V2']}' (should be 'true')")
    else:
        print("✅ LANGCHAIN_TRACING_V2 is set correctly")
    
    if 'LANGCHAIN_API_KEY' not in env_vars:
        issues.append("❌ LANGCHAIN_API_KEY not found in .env")
    elif env_vars['LANGCHAIN_API_KEY'] in ['your_langsmith_api_key_here', '', 'your_key_here']:
        issues.append("❌ LANGCHAIN_API_KEY is still a placeholder - replace with your actual API key!")
        issues.append("   Get your key from: https://smith.langchain.com → Settings → API Keys")
    else:
        print(f"✅ LANGCHAIN_API_KEY is set (first 10 chars: {env_vars['LANGCHAIN_API_KEY'][:10]}...)")
    
    if 'LANGCHAIN_PROJECT' not in env_vars:
        print("⚠️  LANGCHAIN_PROJECT not set (will default to 'intelli-odm')")
    else:
        print(f"✅ LANGCHAIN_PROJECT: {env_vars['LANGCHAIN_PROJECT']}")
    
    if issues:
        print("\n" + "\n".join(issues))
        print("\n" + "="*80)
        print("NEXT STEPS:")
        print("="*80)
        print("1. Edit your .env file")
        print("2. Replace 'your_langsmith_api_key_here' with your actual LangSmith API key")
        print("3. Get your API key from: https://smith.langchain.com")
        print("4. Restart your Streamlit app after updating .env")
        return False
    
    print("\n✅ All environment variables are configured correctly!")
    print("\n" + "="*80)
    print("NOTE:")
    print("="*80)
    print("Environment variables from .env are loaded by:")
    print("  - config.settings (using pydantic-settings)")
    print("  - utils.llm_client.setup_langsmith_tracking()")
    print("\nAfter updating .env, restart your Streamlit app to load the new values.")
    
    return True

if __name__ == "__main__":
    success = check_env_file()
    sys.exit(0 if success else 1)

