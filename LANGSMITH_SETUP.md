# LangSmith Tracking Setup Guide

## Problem
No traces appearing in LangSmith dashboard.

## Root Cause
The `.env` file is empty - LangSmith environment variables are not configured.

## Solution

### Step 1: Add LangSmith Configuration to `.env`

Add these lines to your `.env` file:

```bash
# LangSmith / LangChain Observability
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=your_langsmith_api_key_here
LANGCHAIN_PROJECT=intelli-odm
LANGCHAIN_ENDPOINT=https://api.smith.langchain.com
```

**To get your LangSmith API key:**
1. Go to https://smith.langchain.com
2. Sign in or create an account
3. Go to Settings → API Keys
4. Create a new API key
5. Copy the key and paste it in `.env` as `LANGCHAIN_API_KEY`

### Step 2: Verify Installation

Make sure `langsmith` is installed:
```bash
pip install langsmith
```

### Step 3: Test the Setup

Run the test script:
```bash
python3 test_langsmith.py
```

This will verify:
- Environment variables are set correctly
- LangSmith SDK is installed
- Tracking is working

### Step 4: Restart Your Application

After updating `.env`:
1. **Restart your Streamlit app** (the .env file is read at startup)
2. Make a prediction in the app
3. Check LangSmith dashboard at https://smith.langchain.com
4. Look for traces under the "intelli-odm" project

## What Was Fixed

1. **Early Initialization**: LangSmith is now initialized at module import time, before any LLM calls
2. **Multiple Tracking Methods**:
   - LangChain callbacks (when LangChain is installed)
   - LangSmith `@traceable` decorators for direct API calls
   - Environment variable-based SDK tracking
3. **Better Error Handling**: Clear warnings if LangSmith is not configured
4. **Comprehensive Logging**: Logs show when tracking is enabled/disabled

## Verification

After setup, you should see in the logs:
```
✅ LangSmith tracking enabled for project: intelli-odm
   API Key: lsv2_pt_...
   Endpoint: https://api.smith.langchain.com
   Environment variables set for SDK-level tracking
```

## Troubleshooting

### Still no traces?

1. **Check .env file is being read**:
   ```bash
   python3 -c "from config.settings import settings; print(settings.langchain_tracing_v2)"
   ```

2. **Verify environment variables**:
   ```bash
   python3 test_langsmith.py
   ```

3. **Check LangSmith dashboard**:
   - Make sure you're looking at the correct project: "intelli-odm"
   - Check if there are any filters applied
   - Try refreshing the page

4. **Check logs**:
   - Look for "✅ LangSmith tracking enabled" messages
   - Look for "✅ LLM call wrapped with LangSmith traceable" messages

5. **Test with a simple trace**:
   ```python
   from langsmith import traceable
   
   @traceable(name="test")
   def test():
       return "hello"
   
   test()
   ```
   Then check LangSmith dashboard immediately.

## Expected Behavior

Once configured correctly:
- Every LLM call should create a trace in LangSmith
- Traces will show:
  - Input prompts
  - Model responses
  - Metadata (product descriptions, etc.)
  - Tags (odm, sales-prediction, procurement)
  - Timing information

