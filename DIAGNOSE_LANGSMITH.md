# LangSmith Tracing Diagnosis

## Current Status
- Environment variables are NOT set in the base Python environment
- This means traces won't appear until you:
  1. Set up your `.env` file with LangSmith API key
  2. Restart your Streamlit app (it reads .env at startup)

## Quick Fix

1. **Check your .env file:**
   ```bash
   cat .env | grep LANGCHAIN
   ```

2. **If missing, add to .env:**
   ```bash
   LANGCHAIN_TRACING_V2=true
   LANGCHAIN_API_KEY=your_actual_api_key_here
   LANGCHAIN_PROJECT=intelli-odm
   LANGCHAIN_ENDPOINT=https://api.smith.langchain.com
   ```

3. **Get your API key:**
   - Go to https://smith.langchain.com
   - Sign in
   - Settings → API Keys → Create new key
   - Copy the key (starts with `lsv2_pt_...`)

4. **Restart Streamlit:**
   - Stop your current Streamlit app (Ctrl+C)
   - Start it again: `streamlit run odm_app.py`
   - The .env file is read at startup

## Verification

After restarting, check the logs for:
```
✅ LangSmith tracking enabled for project: intelli-odm
```

If you see this, tracing is working. Make a prediction and check LangSmith dashboard.

## Expected Traces

Once working, you should see:
- `Predict_Product_Sales` (main trace)
- `Vector_DB_Search` (child trace)
- `Analyze_Product_Viability` (child trace)
- `Generate_Store_Predictions` (child trace)
- `Ollama-{model}` or `OpenAI-{model}` (LLM call, child trace)

