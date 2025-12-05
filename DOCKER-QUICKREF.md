# Docker Quick Reference - Intelli-ODM

## 🚀 Quick Start (Cross-Platform)

### Windows (PowerShell)
```powershell
# Auto-detect and run with optimizations
.\docker-run.ps1 up --build
```

### Windows (CMD)
```cmd
# Auto-detect and run with optimizations
docker-run.bat up --build
```

### macOS / Linux
```bash
# Auto-detect and run with optimizations
chmod +x docker-run.sh
./docker-run.sh up --build
```

### Or Use Docker Compose Directly
```bash
# Works on all platforms
docker-compose up --build
```

---

## 📋 Most Used Commands

### Start/Stop

```bash
# Start (foreground)
docker-compose up

# Start (background)
docker-compose up -d

# Stop
docker-compose down

# Restart
docker-compose restart
```

### Build & Update

```bash
# Build/rebuild
docker-compose build

# Rebuild from scratch
docker-compose build --no-cache

# Update and restart
docker-compose up --build -d
```

### Logs & Debugging

```bash
# View logs
docker-compose logs

# Follow logs (live)
docker-compose logs -f

# Logs for specific service
docker-compose logs intelli-odm

# Last 100 lines
docker-compose logs --tail=100 intelli-odm
```

### Execute Commands

```bash
# Open bash shell
docker-compose exec intelli-odm bash

# Run Python script
docker-compose exec intelli-odm python orchestrator.py

# Run tests
docker-compose exec intelli-odm pytest

# Format code
docker-compose exec intelli-odm black .

# Check Python packages
docker-compose exec intelli-odm pip list
```

### Maintenance

```bash
# View running containers
docker-compose ps

# View resource usage
docker stats intelli-odm

# Remove everything (clean start)
docker-compose down -v

# Remove unused images
docker image prune -a

# View disk usage
docker system df
```

---

## 🔧 Troubleshooting One-Liners

```bash
# Can't connect to Ollama?
curl http://localhost:11434/api/tags

# Rebuild without cache
docker-compose build --no-cache && docker-compose up

# Fresh start
docker-compose down -v && docker-compose up --build

# Shell into container for debugging
docker-compose run --rm intelli-odm bash

# Check if numpy works
docker-compose exec intelli-odm python -c "import numpy; print(numpy.__version__)"

# Verify all packages
docker-compose exec intelli-odm python -c "import pandas, numpy, sklearn, chromadb, pulp, cvxpy, ollama; print('OK')"
```

---

## 📁 Important Paths

| Purpose | Host Path | Container Path |
|---------|-----------|----------------|
| Source code | `./agents/` | `/app/agents/` |
| Data input | `./data/input/` | `/app/data/input/` |
| Data output | `./data/output/` | `/app/data/output/` |
| Logs | `./logs/` | `/app/logs/` |
| Vector DB | `./chroma_db/` | `/app/chroma_db/` |
| Config | `./.env` | `/app/.env` |

---

## 🐛 Common Issues

### "Cannot connect to Ollama"
```bash
# Check Ollama is running on host
ollama list

# Test Ollama endpoint
curl http://localhost:11434/api/tags

# Check from container
docker-compose exec intelli-odm curl http://host.docker.internal:11434/api/tags
```

### "Container exits immediately"
```bash
# Check logs
docker-compose logs intelli-odm

# Run interactively
docker-compose run --rm intelli-odm bash
```

### "Permission denied"
```bash
# Windows: Share drive in Docker Desktop settings
# Settings → Resources → File Sharing

# Linux: Fix permissions
sudo chown -R $USER:$USER ./data ./logs ./chroma_db
```

### "Out of memory"
Edit `docker-compose.yml` → increase `memory` limit

### "Slow on Windows"
- Use WSL 2 backend (Docker Desktop → Settings → General)
- Store files in WSL filesystem for better performance

---

## 🎯 Development Workflow

```bash
# 1. Start containers
docker-compose up -d

# 2. Edit code on host (changes reflect immediately)
code agents/data_ingestion_agent.py

# 3. Run/test in container
docker-compose exec intelli-odm python orchestrator.py

# 4. View logs
docker-compose logs -f

# 5. Stop when done
docker-compose down
```

---

## 📊 Monitoring

```bash
# Container status
docker-compose ps

# Resource usage (CPU, RAM, Network)
docker stats intelli-odm

# Health check status
docker inspect --format='{{.State.Health.Status}}' intelli-odm

# Container uptime
docker inspect --format='{{.State.StartedAt}}' intelli-odm
```

---

## 🔗 URLs

- **Ollama (host):** http://localhost:11434
- **Ollama (from container):** http://host.docker.internal:11434
- **Future API:** http://localhost:8000

---

## 💡 Pro Tips

1. **Use `.env` file** for configuration instead of editing docker-compose.yml
2. **Mount source code as volume** for instant code updates
3. **Keep data in volumes** so it persists between rebuilds
4. **Use `docker-compose logs -f`** to debug startup issues
5. **Run Ollama on host** for better GPU access
6. **Use WSL 2** on Windows for much better performance

---

## 🆘 Emergency Reset

```bash
# Nuclear option - clean everything and start fresh
docker-compose down -v
docker system prune -a -f
docker volume prune -f
docker-compose up --build
```

---

For detailed documentation, see `DOCKER.md`

