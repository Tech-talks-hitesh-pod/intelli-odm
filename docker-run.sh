#!/bin/bash
# Cross-platform Docker run script
# Automatically detects OS and uses appropriate docker-compose configuration

set -e

# Detect OS
detect_os() {
    case "$(uname -s)" in
        Linux*)     echo "linux";;
        Darwin*)    echo "mac";;
        CYGWIN*|MINGW*|MSYS*)    echo "windows";;
        *)          echo "unknown";;
    esac
}

OS=$(detect_os)

echo "=========================================="
echo "Intelli-ODM Docker Launcher"
echo "Detected OS: $OS"
echo "=========================================="

# Set compose files based on OS
COMPOSE_FILES="-f docker-compose.yml"

case "$OS" in
    linux)
        echo "Using Linux-optimized configuration..."
        COMPOSE_FILES="$COMPOSE_FILES -f docker-compose.linux.yml"
        ;;
    mac)
        echo "Using macOS-optimized configuration..."
        COMPOSE_FILES="$COMPOSE_FILES -f docker-compose.mac.yml"
        ;;
    windows)
        echo "Using Windows-optimized configuration..."
        COMPOSE_FILES="$COMPOSE_FILES -f docker-compose.windows.yml"
        ;;
    *)
        echo "Unknown OS, using base configuration..."
        ;;
esac

# Check if docker-compose exists
if ! command -v docker-compose &> /dev/null; then
    echo "ERROR: docker-compose is not installed"
    exit 1
fi

# Check if Ollama is running (on non-Linux systems)
if [ "$OS" != "linux" ]; then
    echo ""
    echo "Checking if Ollama is running on host..."
    if curl -s -f http://localhost:11434/api/tags > /dev/null 2>&1; then
        echo "✅ Ollama is running on host"
    else
        echo "⚠️  Warning: Ollama doesn't seem to be running"
        echo "   Start it with: ollama serve"
        echo "   Then pull model: ollama pull llama3:8b"
    fi
fi

echo ""
echo "Starting containers with command:"
echo "docker-compose $COMPOSE_FILES $@"
echo ""

# Run docker-compose with detected configuration
exec docker-compose $COMPOSE_FILES "$@"

