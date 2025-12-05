# Dockerfile for Intelli-ODM
# Multi-stage build for smaller final image

FROM python:3.11-slim as builder

# Set working directory
WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --user -r requirements.txt

# Final stage
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Copy Python dependencies from builder
COPY --from=builder /root/.local /root/.local

# Copy application code
COPY agents/ ./agents/
COPY orchestrator.py .
COPY shared_knowledge_base.py .
COPY config.example .
COPY docker-entrypoint.sh /usr/local/bin/

# Make entrypoint script executable
RUN chmod +x /usr/local/bin/docker-entrypoint.sh

# Make sure scripts are in PATH
ENV PATH=/root/.local/bin:$PATH

# Set entrypoint
ENTRYPOINT ["docker-entrypoint.sh"]

# Set Python environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Create necessary directories
RUN mkdir -p /app/data/input /app/data/output /app/logs /app/chroma_db /app/models

# Expose port if needed (for future API/web interface)
EXPOSE 8000

# Default command
CMD ["python", "orchestrator.py"]

