#!/bin/bash
# Docker entrypoint script for Intelli-ODM

set -e

echo "=========================================="
echo "Intelli-ODM Docker Container"
echo "=========================================="

# Check if .env exists, if not copy from example
if [ ! -f /app/.env ]; then
    echo "No .env file found, copying from config.example..."
    cp /app/config.example /app/.env
fi

# Wait for Ollama to be ready
echo "Checking Ollama connection..."
OLLAMA_URL=${OLLAMA_URL:-http://host.docker.internal:11434}
MAX_RETRIES=30
RETRY_COUNT=0

while [ $RETRY_COUNT -lt $MAX_RETRIES ]; do
    if curl -s -f "${OLLAMA_URL}/api/tags" > /dev/null 2>&1; then
        echo "✅ Ollama is ready at ${OLLAMA_URL}"
        break
    fi
    
    RETRY_COUNT=$((RETRY_COUNT + 1))
    echo "Waiting for Ollama... (${RETRY_COUNT}/${MAX_RETRIES})"
    sleep 2
done

if [ $RETRY_COUNT -eq $MAX_RETRIES ]; then
    echo "⚠️  Warning: Could not connect to Ollama at ${OLLAMA_URL}"
    echo "    Make sure Ollama is running on your host machine"
    echo "    You can start it with: ollama serve"
fi

# Verify Python packages
echo "Verifying Python packages..."
python -c "import pandas, numpy, sklearn, pulp, cvxpy, ollama; print('✅ All core packages available')" || {
    echo "❌ Error: Some Python packages are missing"
    exit 1
}

# Create necessary directories
mkdir -p /app/data/input /app/data/output /app/logs /app/chroma_db /app/models

echo "=========================================="
echo "Starting application..."
echo "=========================================="

# Execute the main command
exec "$@"

