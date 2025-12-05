# Docker Setup Guide - Intelli-ODM

This guide explains how to run Intelli-ODM in Docker while keeping Ollama running on your Windows host machine.

---

## 🐳 Why Docker?

**Benefits:**
- ✅ **No compiler issues** - All dependencies (including numpy) install cleanly
- ✅ **Consistent environment** - Works the same on Windows, macOS, Linux
- ✅ **Easy setup** - No need for Visual Studio C++ Build Tools
- ✅ **Isolated** - Doesn't interfere with your system Python
- ✅ **Reproducible** - Same environment for dev and production

---

## 📋 Prerequisites

### 1. Install Docker Desktop

**Windows:**
- Download from: https://www.docker.com/products/docker-desktop/
- Install and restart your computer
- Make sure WSL 2 is enabled (Docker will prompt you)

**Verify installation:**
```cmd
docker --version
docker-compose --version
```

### 2. Install Ollama on Host (Windows)

Ollama will run on your Windows machine, not in Docker:

1. Download from: https://ollama.com/download/windows
2. Install and start Ollama
3. Pull the model:

```cmd
ollama pull llama3:8b
```

4. Verify Ollama is running:

```cmd
ollama list
```

---

## 🚀 Quick Start

### Method 1: Using Platform-Optimized Scripts (Recommended)

**Windows (PowerShell):**
```powershell
.\docker-run.ps1 up --build
```

**Windows (CMD):**
```cmd
docker-run.bat up --build
```

**macOS / Linux:**
```bash
chmod +x docker-run.sh
./docker-run.sh up --build
```

These scripts automatically detect your OS and use optimized configurations!

### Method 2: Using Docker Compose Directly

**All Platforms (Base Configuration):**
```bash
# 1. Clone and navigate to project
cd intelli-odm

# 2. Create .env file (optional - will use defaults)
cp config.example .env  # or 'copy' on Windows

# 3. Build and start the container
docker-compose up --build

# To run in background:
docker-compose up -d --build
```

**Platform-Specific Optimizations:**

```bash
# Linux (uses host network for better performance)
docker-compose -f docker-compose.yml -f docker-compose.linux.yml up --build

# macOS (uses delegated volumes for better performance)
docker-compose -f docker-compose.yml -f docker-compose.mac.yml up --build

# Windows (uses named volumes for better performance)
docker-compose -f docker-compose.yml -f docker-compose.windows.yml up --build
```

### Method 2: Using Docker CLI

```cmd
# Build the image
docker build -t intelli-odm .

# Run the container
docker run -it --rm ^
  -v "%cd%\data:/app/data" ^
  -v "%cd%\logs:/app/logs" ^
  -v "%cd%\.env:/app/.env:ro" ^
  -e OLLAMA_URL=http://host.docker.internal:11434 ^
  --add-host host.docker.internal:host-gateway ^
  intelli-odm
```

---

## 📁 Project Structure for Docker

```
intelli-odm/
├── Dockerfile                 # Docker image definition
├── docker-compose.yml         # Docker Compose configuration
├── .dockerignore             # Files to exclude from image
├── docker-entrypoint.sh      # Container startup script
├── .env                      # Environment variables (create this)
├── agents/                   # Application code (mounted as volume)
├── data/                     # Data files (mounted as volume)
│   ├── input/
│   └── output/
├── logs/                     # Log files (mounted as volume)
├── chroma_db/               # Vector DB storage (mounted as volume)
└── models/                   # Model cache (mounted as volume)
```

---

## 🔧 Configuration

### Environment Variables

Edit `.env` file or set in `docker-compose.yml`:

```bash
# Ollama Configuration
OLLAMA_URL=http://host.docker.internal:11434
OLLAMA_MODEL=llama3:8b

# Knowledge Base
KB_TYPE=chroma
KB_PATH=./chroma_db

# Logging
LOG_LEVEL=INFO
LOG_FILE=./logs/intelli_odm.log
```

### Accessing Ollama from Container

**Windows/Mac Docker Desktop:**
- Use `host.docker.internal:11434` to access host machine
- Already configured in `docker-compose.yml`

**Linux:**
- Uncomment `network_mode: "host"` in `docker-compose.yml`
- OR use host's IP address instead of `host.docker.internal`

---

## 🛠️ Common Commands

### Container Management

```bash
# Start containers
docker-compose up

# Start in background
docker-compose up -d

# Stop containers
docker-compose down

# View logs
docker-compose logs -f

# Restart containers
docker-compose restart

# Rebuild after code changes
docker-compose up --build
```

### Development

```bash
# Execute commands in running container
docker-compose exec intelli-odm bash

# Run Python script in container
docker-compose exec intelli-odm python orchestrator.py

# Install additional packages
docker-compose exec intelli-odm pip install package-name

# Run tests
docker-compose exec intelli-odm pytest

# Format code
docker-compose exec intelli-odm black .
```

### Debugging

```bash
# Check container status
docker-compose ps

# View container logs
docker-compose logs intelli-odm

# Check resource usage
docker stats intelli-odm

# Inspect container
docker inspect intelli-odm

# Shell into running container
docker-compose exec intelli-odm /bin/bash
```

---

## 📊 Volume Mounts

The `docker-compose.yml` mounts these directories:

| Host Path | Container Path | Purpose |
|-----------|---------------|---------|
| `./agents` | `/app/agents` | Application code (live reload) |
| `./data` | `/app/data` | Input/output data files |
| `./logs` | `/app/logs` | Application logs |
| `./chroma_db` | `/app/chroma_db` | Vector database storage |
| `./models` | `/app/models` | Model cache |
| `./.env` | `/app/.env` | Environment configuration |

**Benefits:**
- Code changes reflect immediately (no rebuild needed)
- Data persists between container restarts
- Logs accessible from host machine

---

## 🔍 Troubleshooting

### Issue: Can't connect to Ollama

**Symptoms:**
```
Error: Could not connect to Ollama at http://host.docker.internal:11434
```

**Solutions:**

1. **Check Ollama is running on host:**
```cmd
ollama list
```

2. **Test Ollama API from host:**
```cmd
curl http://localhost:11434/api/tags
```

3. **Test from container:**
```bash
docker-compose exec intelli-odm curl http://host.docker.internal:11434/api/tags
```

4. **Windows Firewall:**
   - Allow Docker through Windows Firewall
   - Allow Ollama through Windows Firewall

### Issue: Numpy import error

**This shouldn't happen in Docker, but if it does:**

```bash
# Rebuild the image from scratch
docker-compose build --no-cache
docker-compose up
```

### Issue: Permission denied on volumes

**Windows:**
```cmd
# Give Docker access to the drive in Docker Desktop settings
# Settings → Resources → File Sharing → Add the drive
```

### Issue: Container exits immediately

```bash
# Check logs
docker-compose logs intelli-odm

# Run with interactive shell
docker-compose run --rm intelli-odm /bin/bash
```

### Issue: Out of memory

Edit `docker-compose.yml`:

```yaml
deploy:
  resources:
    limits:
      memory: 16G  # Increase this
```

### Issue: Slow build on Windows

**Use WSL 2:**
- Docker Desktop → Settings → General → Use WSL 2 based engine
- Store project files in WSL filesystem for better performance

---

## 🚦 Health Checks

The container includes a health check:

```bash
# Check container health
docker inspect --format='{{.State.Health.Status}}' intelli-odm

# View health check logs
docker inspect --format='{{range .State.Health.Log}}{{.Output}}{{end}}' intelli-odm
```

---

## 📦 Optional: Installing Prophet

Prophet is optional. To include it:

### Method 1: Modify Dockerfile

Add to Dockerfile before the `RUN pip install` line:

```dockerfile
# Install Prophet dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Then install requirements (which can now include prophet)
```

Uncomment prophet in `requirements.txt` and rebuild:

```bash
docker-compose build --no-cache
```

### Method 2: Install in running container

```bash
docker-compose exec intelli-odm pip install prophet
```

---

## 🎯 Production Deployment

### Build optimized image

```bash
# Build production image
docker build -t intelli-odm:latest .

# Tag for registry
docker tag intelli-odm:latest your-registry/intelli-odm:latest

# Push to registry
docker push your-registry/intelli-odm:latest
```

### Run in production

```bash
docker run -d \
  --name intelli-odm-prod \
  --restart unless-stopped \
  -v /path/to/data:/app/data \
  -v /path/to/logs:/app/logs \
  -e OLLAMA_URL=http://ollama-host:11434 \
  your-registry/intelli-odm:latest
```

---

## 🆚 Docker vs Local Installation

| Feature | Docker | Local Install |
|---------|--------|---------------|
| **Setup Time** | 5 minutes | 15-30 minutes |
| **Dependencies** | Automatic | Manual |
| **Compiler Needed** | ❌ No | ✅ Yes (Windows) |
| **Isolation** | ✅ Yes | ❌ No |
| **Resource Overhead** | ~100MB | ~0MB |
| **Development** | Good (volumes) | Best |
| **Production** | Best | Good |

---

## 📝 Next Steps

After Docker setup:

1. **Configure environment:**
   ```cmd
   copy config.example .env
   notepad .env
   ```

2. **Start the system:**
   ```cmd
   docker-compose up
   ```

3. **Run your first workflow:**
   ```cmd
   docker-compose exec intelli-odm python orchestrator.py
   ```

4. **Check logs:**
   ```cmd
   docker-compose logs -f
   ```

---

## 🔗 Resources

- Docker Desktop: https://www.docker.com/products/docker-desktop/
- Ollama: https://ollama.com/
- Docker Compose Docs: https://docs.docker.com/compose/
- WSL 2: https://docs.microsoft.com/en-us/windows/wsl/install

---

## 💡 Tips

1. **Use volumes for data persistence** - Data in volumes survives container restarts
2. **Mount source code** - Edit code on host, runs immediately in container
3. **Keep Ollama on host** - Better GPU access, easier management
4. **Use Docker Compose** - Simpler than raw Docker commands
5. **Enable WSL 2** - Much better performance on Windows

---

**Last Updated:** December 2025

