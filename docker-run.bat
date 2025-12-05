@echo off
REM Cross-platform Docker run script for Windows
REM Automatically uses Windows-optimized configuration

echo ==========================================
echo Intelli-ODM Docker Launcher
echo Platform: Windows
echo ==========================================

REM Check if docker-compose exists
where docker-compose >nul 2>nul
if errorlevel 1 (
    echo ERROR: docker-compose is not installed
    echo Please install Docker Desktop for Windows
    exit /b 1
)

REM Check if Ollama is running
echo.
echo Checking if Ollama is running on host...
curl -s -f http://localhost:11434/api/tags >nul 2>nul
if errorlevel 1 (
    echo WARNING: Ollama doesn't seem to be running
    echo    Start it with: ollama serve
    echo    Then pull model: ollama pull llama3:8b
) else (
    echo OK: Ollama is running on host
)

echo.
echo Starting containers with Windows-optimized configuration...
echo Command: docker-compose -f docker-compose.yml -f docker-compose.windows.yml %*
echo.

REM Run docker-compose with Windows configuration
docker-compose -f docker-compose.yml -f docker-compose.windows.yml %*

