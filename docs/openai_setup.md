# Switching from Ollama to OpenAI

This guide explains how to configure the Intelli-ODM system to use OpenAI instead of local Ollama.

## Quick Setup

### 1. Update your `.env` file

Edit the `.env` file in the project root and make these changes:

```bash
# Change LLM provider from ollama to openai
LLM_PROVIDER=openai

# Add your OpenAI API key (required)
OPENAI_API_KEY=your_openai_api_key_here

# Optional: Change the model (default is gpt-4o-mini)
OPENAI_MODEL=gpt-4o-mini
# Or use other models like:
# OPENAI_MODEL=gpt-4
# OPENAI_MODEL=gpt-3.5-turbo

# Optional: Adjust temperature (default is 0.1)
OPENAI_TEMPERATURE=0.1
```

### 2. Example `.env` file for OpenAI

```bash
# LLM Configuration
LLM_PROVIDER=openai
OPENAI_API_KEY=sk-your-actual-api-key-here
OPENAI_MODEL=gpt-4o-mini
OPENAI_TEMPERATURE=0.1

# LangSmith / LangChain Observability (optional but recommended)
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=your_langsmith_api_key_here
LANGCHAIN_PROJECT=intelli-odm
LANGCHAIN_ENDPOINT=https://api.smith.langchain.com

# Database Configuration
CHROMA_PERSIST_DIRECTORY=data/chroma_db

# Logging
LOG_LEVEL=INFO
```

### 3. Restart the Application

After updating `.env`, restart your Streamlit app:

```bash
# Stop the current app (Ctrl+C)
# Then restart:
streamlit run streamlit_app.py
```

## Verification

The system will automatically:
- ✅ Detect `LLM_PROVIDER=openai` from `.env`
- ✅ Use your `OPENAI_API_KEY` for authentication
- ✅ Initialize the OpenAI client
- ✅ Show an error if the API key is missing

## Available OpenAI Models

You can use any of these models by setting `OPENAI_MODEL`:

- `gpt-4o-mini` (default, cost-effective)
- `gpt-4o` (more capable)
- `gpt-4-turbo` (high performance)
- `gpt-4` (original GPT-4)
- `gpt-3.5-turbo` (faster, cheaper)

## Switching Back to Ollama

To switch back to Ollama, simply change:

```bash
LLM_PROVIDER=ollama
```

And make sure Ollama is running locally on `http://localhost:11434`.

## Troubleshooting

### Error: "OpenAI API key not found"
- Make sure `OPENAI_API_KEY` is set in your `.env` file
- Check that the API key starts with `sk-`
- Restart the Streamlit app after updating `.env`

### Error: "Invalid API key"
- Verify your OpenAI API key is correct
- Check your OpenAI account has available credits
- Ensure the API key hasn't been revoked

### No tracking in LangSmith
- Make sure `LANGCHAIN_TRACING_V2=true` in `.env`
- Set `LANGCHAIN_API_KEY` with your LangSmith API key
- Restart the app after updating

## Cost Considerations

OpenAI API usage is charged per token:
- **gpt-4o-mini**: ~$0.15 per 1M input tokens, ~$0.60 per 1M output tokens
- **gpt-4o**: ~$2.50 per 1M input tokens, ~$10 per 1M output tokens
- **gpt-3.5-turbo**: ~$0.50 per 1M input tokens, ~$1.50 per 1M output tokens

Monitor your usage at: https://platform.openai.com/usage

