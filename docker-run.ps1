# Cross-platform Docker run script for Windows PowerShell
# Automatically uses Windows-optimized configuration

Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "Intelli-ODM Docker Launcher" -ForegroundColor Cyan
Write-Host "Platform: Windows (PowerShell)" -ForegroundColor Cyan
Write-Host "==========================================" -ForegroundColor Cyan

# Check if docker-compose exists
if (-not (Get-Command docker-compose -ErrorAction SilentlyContinue)) {
    Write-Host "ERROR: docker-compose is not installed" -ForegroundColor Red
    Write-Host "Please install Docker Desktop for Windows" -ForegroundColor Red
    exit 1
}

# Check if Ollama is running
Write-Host "`nChecking if Ollama is running on host..." -ForegroundColor Yellow
try {
    $response = Invoke-WebRequest -Uri "http://localhost:11434/api/tags" -UseBasicParsing -TimeoutSec 2 -ErrorAction Stop
    Write-Host "✅ Ollama is running on host" -ForegroundColor Green
} catch {
    Write-Host "⚠️  Warning: Ollama doesn't seem to be running" -ForegroundColor Yellow
    Write-Host "   Start it with: ollama serve" -ForegroundColor Gray
    Write-Host "   Then pull model: ollama pull llama3:8b" -ForegroundColor Gray
}

Write-Host "`nStarting containers with Windows-optimized configuration..." -ForegroundColor Yellow
Write-Host "Command: docker-compose -f docker-compose.yml -f docker-compose.windows.yml $args" -ForegroundColor Gray
Write-Host ""

# Run docker-compose with Windows configuration
& docker-compose -f docker-compose.yml -f docker-compose.windows.yml $args

