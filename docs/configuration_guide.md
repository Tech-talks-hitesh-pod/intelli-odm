# Configuration Guide

## Overview
This guide covers the complete setup and configuration of the Intelli-ODM system, including environment setup, dependency installation, and system configuration.

## System Requirements

### Hardware Requirements
- **CPU**: 8+ cores recommended for optimal performance
- **RAM**: 16GB minimum, 32GB recommended for large datasets
- **Storage**: 50GB+ available space for models and data
- **GPU**: Optional, but recommended for large-scale embeddings (NVIDIA GPU with CUDA support)

### Software Requirements
- **Operating System**: Linux (Ubuntu 20.04+), macOS (10.15+), or Windows 10/11
- **Python**: 3.9+ (3.11 recommended)
- **Docker**: Optional, for containerized deployment
- **Git**: For repository management

## Installation Guide

### 1. Clone the Repository
```bash
git clone https://github.com/your-org/intelli-odm.git
cd intelli-odm
```

### 2. Python Environment Setup

#### Using Virtual Environment
```bash
# Create virtual environment
python -m venv .venv

# Activate virtual environment
# On Linux/macOS:
source .venv/bin/activate
# On Windows:
.venv\Scripts\activate

# Upgrade pip
pip install --upgrade pip
```

#### Using Conda (Alternative)
```bash
# Create conda environment
conda create -n intelli-odm python=3.11
conda activate intelli-odm
```

### 3. Install Dependencies
```bash
# Install all dependencies
pip install -r requirements.txt

# Verify installation
python -c "import pandas, numpy, sklearn; print('Core dependencies installed successfully')"
```

### 4. Install Ollama

#### Linux/macOS
```bash
# Install Ollama
curl -fsSL https://ollama.com/install.sh | sh

# Start Ollama service
ollama serve &

# Pull Llama3 model (this may take 10-15 minutes)
ollama pull llama3:8b

# Verify installation
ollama list
```

#### Windows
```bash
# Download and install from https://ollama.com/download
# Then in PowerShell/Command Prompt:
ollama pull llama3:8b
ollama list
```

### 5. Setup Vector Database

#### Option A: ChromaDB (Recommended for Development)
ChromaDB is included in requirements.txt and requires no additional setup for local development.

```python
# Test ChromaDB installation
python -c "import chromadb; print('ChromaDB installed successfully')"
```

#### Option B: FAISS (For High Performance)
```bash
# FAISS is included in requirements.txt
# Test installation
python -c "import faiss; print('FAISS installed successfully')"
```

#### Option C: Production Vector Databases
For production deployments, consider:
- **Weaviate**: Scalable vector database with GraphQL API
- **Pinecone**: Managed vector database service
- **pgvector**: PostgreSQL extension for vector storage

## Configuration Files

### 1. Environment Variables
Create `.env` file in the project root:

```bash
# Copy template
cp .env.template .env

# Edit configuration
nano .env
```

**.env File Configuration**:
```bash
# Ollama Configuration
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3:8b
OLLAMA_TIMEOUT=300

# Vector Database Configuration
VECTOR_DB_TYPE=chromadb
CHROMADB_PERSIST_DIR=./data/chromadb
CHROMADB_HOST=localhost
CHROMADB_PORT=8000

# Alternative: FAISS Configuration
# VECTOR_DB_TYPE=faiss
# FAISS_INDEX_PATH=./data/faiss_index

# Alternative: Weaviate Configuration
# VECTOR_DB_TYPE=weaviate
# WEAVIATE_URL=http://localhost:8080
# WEAVIATE_API_KEY=your_api_key

# Business Constraints (Default Values)
DEFAULT_BUDGET=1000000
DEFAULT_MOQ=200
DEFAULT_PACK_SIZE=20
DEFAULT_LEAD_TIME_DAYS=30
DEFAULT_SAFETY_STOCK_FACTOR=1.2

# Performance Configuration
MAX_WORKERS=4
EMBEDDING_BATCH_SIZE=32
CACHE_TTL_SECONDS=3600

# Logging Configuration
LOG_LEVEL=INFO
LOG_FILE=logs/intelli_odm.log
LOG_FORMAT=%(asctime)s - %(name)s - %(levelname)s - %(message)s

# Development Settings
DEBUG_MODE=false
ENABLE_PROFILING=false
```

### 2. Application Settings
**config/settings.py**:
```python
import os
from typing import Dict, Any
from pydantic import BaseSettings

class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    # Ollama Configuration
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3:8b"
    ollama_timeout: int = 300
    
    # Vector Database
    vector_db_type: str = "chromadb"
    chromadb_persist_dir: str = "./data/chromadb"
    chromadb_host: str = "localhost"
    chromadb_port: int = 8000
    
    # Business Constraints
    default_budget: int = 1000000
    default_moq: int = 200
    default_pack_size: int = 20
    default_lead_time_days: int = 30
    default_safety_stock_factor: float = 1.2
    
    # Performance
    max_workers: int = 4
    embedding_batch_size: int = 32
    cache_ttl_seconds: int = 3600
    
    # Logging
    log_level: str = "INFO"
    log_file: str = "logs/intelli_odm.log"
    
    class Config:
        env_file = ".env"

# Global settings instance
settings = Settings()
```

### 3. Agent-Specific Configurations
**config/agent_configs.py**:
```python
# Data Ingestion Agent Configuration
DATA_INGESTION_CONFIG = {
    "validation": {
        "max_missing_percentage": 10.0,
        "required_columns": {
            "sales": ["date", "store_id", "sku", "units_sold"],
            "inventory": ["store_id", "sku", "on_hand"],
            "pricing": ["store_id", "sku", "price"],
            "products": ["product_id", "description"]
        },
        "date_formats": ["%Y-%m-%d", "%d/%m/%Y", "%m/%d/%Y"],
        "outlier_detection": {
            "method": "iqr",
            "threshold": 3.0
        }
    },
    "feature_engineering": {
        "velocity_window_days": 30,
        "seasonality_periods": [7, 30, 365],
        "store_tier_thresholds": [0.7, 0.4]  # High, Medium, Low
    }
}

# Attribute & Analogy Agent Configuration
ATTRIBUTE_AGENT_CONFIG = {
    "llm": {
        "max_retries": 3,
        "temperature": 0.1,
        "max_tokens": 500,
        "prompt_template": "extract_attributes_v1.jinja2"
    },
    "similarity": {
        "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
        "similarity_threshold": 0.6,
        "max_comparables": 10
    },
    "trend_analysis": {
        "time_windows": ["1M", "3M", "6M"],
        "trend_confidence_threshold": 0.7
    }
}

# Demand Forecasting Agent Configuration
DEMAND_FORECASTING_CONFIG = {
    "method_selection": {
        "analogy_threshold": {
            "min_comparables": 3,
            "min_similarity": 0.7
        },
        "timeseries_threshold": {
            "min_history_months": 12,
            "seasonality_detection": True
        },
        "ml_threshold": {
            "min_stores": 10,
            "min_features": 5
        }
    },
    "forecasting": {
        "forecast_horizon_days": 60,
        "confidence_levels": [0.1, 0.9],  # 10th and 90th percentiles
        "seasonality_periods": [7, 30.44, 365.25]
    },
    "price_sensitivity": {
        "elasticity_method": "log_linear",
        "price_range_factor": 0.3  # +/-30% of base price
    }
}

# Procurement & Allocation Agent Configuration
PROCUREMENT_CONFIG = {
    "optimization": {
        "solver": "pulp",  # Options: "pulp", "cvxpy", "gurobi"
        "solver_timeout_seconds": 300,
        "mip_gap_tolerance": 0.01,
        "objectives": {
            "revenue": {"weight": 0.6},
            "margin": {"weight": 0.3},
            "risk": {"weight": 0.1}
        }
    },
    "constraints": {
        "enforce_pack_size": True,
        "allow_partial_allocation": False,
        "min_store_allocation": 10
    }
}
```

### 4. Model-Specific Configurations
**config/model_configs.py**:
```python
# Forecasting Model Configurations
FORECASTING_MODELS = {
    "prophet": {
        "seasonality_mode": "multiplicative",
        "yearly_seasonality": True,
        "weekly_seasonality": True,
        "daily_seasonality": False,
        "holidays": "IN",  # India holidays
        "changepoint_prior_scale": 0.05,
        "seasonality_prior_scale": 10.0
    },
    "arima": {
        "seasonal_order": (1, 1, 1, 7),  # Weekly seasonality
        "auto_arima": True,
        "max_p": 5,
        "max_q": 5,
        "seasonal": True
    },
    "gradient_boosting": {
        "n_estimators": 100,
        "learning_rate": 0.1,
        "max_depth": 6,
        "random_state": 42,
        "validation_fraction": 0.2
    }
}

# LLM Prompt Templates
PROMPT_TEMPLATES = {
    "attribute_extraction": {
        "template": """
Extract structured attributes from this fashion product description.
Return a JSON object with these fields: material, color, pattern, sleeve, neckline, fit, category, style, target_gender.

Product Description: "{description}"

Guidelines:
- Use standardized values (e.g., "Cotton" not "cotton blend")
- If information is unclear, use "Unknown"
- Be consistent with fashion industry terminology

JSON Response:
""",
        "temperature": 0.1,
        "max_tokens": 400
    },
    "trend_analysis": {
        "template": """
Analyze market trends for this product category based on the provided data.

Product Attributes: {attributes}
Historical Performance: {performance_data}
Time Period: {time_period}

Provide insights on:
1. Category performance trend
2. Seasonal patterns
3. Popular attributes/styles
4. Market opportunities

Response:
""",
        "temperature": 0.3,
        "max_tokens": 600
    },
    "forecast_method_selection": {
        "template": """
Recommend the best forecasting method for this new product based on available data.

Available Data:
- Comparable Products: {comparable_count} (avg similarity: {avg_similarity})
- Historical Data: {history_months} months
- Store Count: {store_count}
- Feature Richness: {feature_score}

Choose from: analogy, timeseries, ml_regression
Provide rationale for your choice.

Recommendation:
""",
        "temperature": 0.2,
        "max_tokens": 300
    }
}
```

## Directory Setup

### 1. Create Required Directories
```bash
# Create directory structure
mkdir -p data/chromadb
mkdir -p data/uploads
mkdir -p data/exports
mkdir -p logs
mkdir -p temp
mkdir -p cache

# Set permissions
chmod 755 data logs temp cache
```

### 2. Directory Structure
```
intelli-odm/
├── data/
│   ├── chromadb/           # Vector database storage
│   ├── uploads/            # Uploaded CSV files
│   ├── exports/            # Generated reports
│   └── sample/             # Sample data files
├── logs/                   # Application logs
├── temp/                   # Temporary files
├── cache/                  # Cached results
└── config/                 # Configuration files
```

## Testing the Setup

### 1. Basic System Test
```bash
# Test Python environment
python -c "
import sys
print(f'Python version: {sys.version}')

# Test core dependencies
try:
    import pandas as pd
    import numpy as np
    import sklearn
    import chromadb
    print('✓ Core dependencies installed')
except ImportError as e:
    print(f'✗ Missing dependency: {e}')

# Test Ollama connection
try:
    import ollama
    client = ollama.Client()
    models = client.list()
    print(f'✓ Ollama connected, models: {[m[\"name\"] for m in models[\"models\"]]}')
except Exception as e:
    print(f'✗ Ollama connection failed: {e}')
"
```

### 2. Agent Initialization Test
```bash
# Test agent initialization
python -c "
from orchestrator import OrchestratorAgent
from config.settings import settings
import ollama

try:
    # Test Ollama client
    client = ollama.Client(host=settings.ollama_base_url)
    
    # Test orchestrator initialization
    constraints = {
        'budget': settings.default_budget,
        'MOQ': settings.default_moq,
        'pack_size': settings.default_pack_size
    }
    
    orchestrator = OrchestratorAgent(client, constraints)
    print('✓ Orchestrator initialized successfully')
    
except Exception as e:
    print(f'✗ Orchestrator initialization failed: {e}')
"
```

### 3. Vector Database Test
```bash
# Test vector database
python -c "
from shared_knowledge_base import SharedKnowledgeBase
import numpy as np

try:
    kb = SharedKnowledgeBase()
    
    # Test embedding storage and retrieval
    test_embedding = np.random.rand(384).astype('float32')
    kb.store_product(
        product_id='TEST-001',
        attributes={'category': 'T-Shirt', 'color': 'Blue'},
        embeddings=test_embedding,
        metadata={'test': True}
    )
    
    # Test similarity search
    similar = kb.find_similar_products(test_embedding, top_k=1)
    print(f'✓ Vector database working, found {len(similar)} similar products')
    
except Exception as e:
    print(f'✗ Vector database test failed: {e}')
"
```

## Performance Optimization

### 1. Memory Management
```python
# config/performance.py
MEMORY_CONFIG = {
    "pandas": {
        "chunksize": 10000,  # Process large CSV files in chunks
        "low_memory": False,
        "engine": "c"  # Use C engine for better performance
    },
    "numpy": {
        "memory_map": True,  # Memory-map large arrays
        "copy": False  # Avoid unnecessary copying
    },
    "chromadb": {
        "persist_on_shutdown": True,
        "anonymized_telemetry": False
    }
}
```

### 2. Caching Configuration
```python
# config/cache.py
CACHE_CONFIG = {
    "embeddings": {
        "enabled": True,
        "ttl_seconds": 86400,  # 24 hours
        "max_size_mb": 500
    },
    "forecasts": {
        "enabled": True,
        "ttl_seconds": 3600,   # 1 hour
        "max_size_mb": 100
    },
    "llm_responses": {
        "enabled": True,
        "ttl_seconds": 7200,   # 2 hours
        "max_size_mb": 50
    }
}
```

## Security Configuration

### 1. Environment Security
```bash
# Secure file permissions
chmod 600 .env
chmod 600 config/*.py

# Git ignore sensitive files
echo ".env" >> .gitignore
echo "*.key" >> .gitignore
echo "credentials.json" >> .gitignore
```

### 2. API Security (if exposing via API)
```python
# config/security.py
SECURITY_CONFIG = {
    "api_key_required": True,
    "rate_limiting": {
        "requests_per_minute": 60,
        "requests_per_hour": 1000
    },
    "cors": {
        "allow_origins": ["http://localhost:3000"],
        "allow_methods": ["GET", "POST"],
        "allow_headers": ["Content-Type", "Authorization"]
    },
    "encryption": {
        "algorithm": "AES-256",
        "key_rotation_days": 30
    }
}
```

## Troubleshooting

### Common Issues

#### 1. Ollama Connection Issues
```bash
# Check if Ollama is running
ps aux | grep ollama

# Restart Ollama service
ollama serve &

# Check model availability
ollama list

# Test model interaction
ollama run llama3:8b "Hello, how are you?"
```

#### 2. Vector Database Issues
```bash
# Clear ChromaDB cache
rm -rf ./data/chromadb/*

# Reinstall ChromaDB
pip uninstall chromadb
pip install chromadb
```

#### 3. Memory Issues
```python
# Monitor memory usage
import psutil
print(f"Memory usage: {psutil.virtual_memory().percent}%")

# Optimize pandas operations
import pandas as pd
pd.set_option('mode.chained_assignment', None)
pd.set_option('compute.use_bottleneck', True)
pd.set_option('compute.use_numexpr', True)
```

#### 4. Performance Issues
```bash
# Enable profiling
export ENABLE_PROFILING=true

# Monitor system resources
htop  # or top on macOS

# Check disk space
df -h
```

### Health Check Script
Create `scripts/health_check.py`:
```python
#!/usr/bin/env python3
"""System health check script for Intelli-ODM."""

import sys
import subprocess
import importlib
from config.settings import settings

def check_dependencies():
    """Check if all required dependencies are installed."""
    required_packages = [
        'pandas', 'numpy', 'sklearn', 'chromadb', 
        'sentence_transformers', 'pulp', 'ollama'
    ]
    
    missing = []
    for package in required_packages:
        try:
            importlib.import_module(package)
        except ImportError:
            missing.append(package)
    
    if missing:
        print(f"✗ Missing packages: {', '.join(missing)}")
        return False
    else:
        print("✓ All required packages installed")
        return True

def check_ollama():
    """Check Ollama service and model availability."""
    try:
        result = subprocess.run(['ollama', 'list'], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            if 'llama3:8b' in result.stdout:
                print("✓ Ollama service running with llama3:8b model")
                return True
            else:
                print("✗ llama3:8b model not found")
                return False
        else:
            print("✗ Ollama service not running")
            return False
    except FileNotFoundError:
        print("✗ Ollama not installed")
        return False

def check_directories():
    """Check if required directories exist."""
    import os
    required_dirs = ['data', 'logs', 'temp', 'cache']
    
    for dir_name in required_dirs:
        if not os.path.exists(dir_name):
            print(f"✗ Missing directory: {dir_name}")
            return False
    
    print("✓ All required directories exist")
    return True

def main():
    """Run complete health check."""
    print("Intelli-ODM Health Check")
    print("=" * 30)
    
    checks = [
        check_dependencies,
        check_ollama,
        check_directories
    ]
    
    passed = sum(check() for check in checks)
    total = len(checks)
    
    print("=" * 30)
    print(f"Health Check Results: {passed}/{total} checks passed")
    
    if passed == total:
        print("✓ System is ready for use")
        return 0
    else:
        print("✗ System configuration issues detected")
        return 1

if __name__ == "__main__":
    sys.exit(main())
```

Run health check:
```bash
python scripts/health_check.py
```

## Production Deployment

### Docker Configuration
**Dockerfile**:
```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Ollama
RUN curl -fsSL https://ollama.com/install.sh | sh

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create required directories
RUN mkdir -p data logs temp cache

# Expose ports
EXPOSE 8000 11434

# Start script
CMD ["python", "orchestrator.py"]
```

**docker-compose.yml**:
```yaml
version: '3.8'

services:
  intelli-odm:
    build: .
    ports:
      - "8000:8000"
    volumes:
      - ./data:/app/data
      - ./logs:/app/logs
    environment:
      - OLLAMA_BASE_URL=http://ollama:11434
    depends_on:
      - ollama
      
  ollama:
    image: ollama/ollama
    ports:
      - "11434:11434"
    volumes:
      - ollama_data:/root/.ollama
      
volumes:
  ollama_data:
```

This configuration guide provides everything needed to set up and configure the Intelli-ODM system for both development and production use.