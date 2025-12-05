# Development Guide

## Overview
This guide provides comprehensive information for developers working on the Intelli-ODM system, including development setup, coding standards, testing procedures, and contribution guidelines.

## Table of Contents
1. [Development Environment Setup](#development-environment-setup)
2. [Code Architecture](#code-architecture)
3. [Coding Standards](#coding-standards)
4. [Testing Strategy](#testing-strategy)
5. [Performance Guidelines](#performance-guidelines)
6. [Contributing Guidelines](#contributing-guidelines)
7. [Debugging and Profiling](#debugging-and-profiling)
8. [Deployment](#deployment)

## Development Environment Setup

### Prerequisites
- Python 3.9+ (3.11 recommended)
- Git
- Docker (optional, for containerized development)
- IDE with Python support (VS Code, PyCharm recommended)

### Setup Steps

#### 1. Clone and Setup Repository
```bash
# Clone repository
git clone https://github.com/your-org/intelli-odm.git
cd intelli-odm

# Create development branch
git checkout -b feature/your-feature-name

# Setup virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install development dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt  # Additional dev tools
```

#### 2. Pre-commit Hooks Setup
```bash
# Install pre-commit
pip install pre-commit

# Setup pre-commit hooks
pre-commit install

# Test pre-commit hooks
pre-commit run --all-files
```

#### 3. IDE Configuration

**VS Code Settings** (`.vscode/settings.json`):
```json
{
    "python.defaultInterpreterPath": ".venv/bin/python",
    "python.linting.enabled": true,
    "python.linting.flake8Enabled": true,
    "python.linting.mypyEnabled": true,
    "python.formatting.provider": "black",
    "python.formatting.blackArgs": ["--line-length=88"],
    "python.testing.pytestEnabled": true,
    "python.testing.pytestArgs": ["tests/"],
    "files.exclude": {
        "**/__pycache__": true,
        "**/*.pyc": true,
        ".pytest_cache": true
    }
}
```

**PyCharm Configuration**:
- Enable Black formatter
- Configure Flake8 linter
- Setup pytest as test runner
- Enable type checking with mypy

## Code Architecture

### Design Patterns

#### 1. Agent Pattern
Each agent follows a consistent interface pattern:

```python
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from dataclasses import dataclass

class BaseAgent(ABC):
    """Base class for all agents in the system."""
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self._validate_config()
    
    @abstractmethod
    def _validate_config(self) -> None:
        """Validate agent configuration."""
        pass
    
    @abstractmethod
    def run(self, *args, **kwargs) -> Any:
        """Main agent execution method."""
        pass
    
    def _preprocess(self, data: Any) -> Any:
        """Preprocess input data."""
        return data
    
    def _postprocess(self, result: Any) -> Any:
        """Postprocess output data."""
        return result

# Example implementation
class DataIngestionAgent(BaseAgent):
    def _validate_config(self) -> None:
        required_keys = ['validation_thresholds', 'feature_config']
        for key in required_keys:
            if key not in self.config:
                raise ValueError(f"Missing required config key: {key}")
    
    def run(self, files: Dict[str, str]) -> Dict[str, Any]:
        """Execute data ingestion pipeline."""
        validated_data = self.validate(files)
        structured_data = self.structure(validated_data)
        enriched_data = self.feature_engineering(structured_data)
        return self._postprocess(enriched_data)
```

#### 2. Strategy Pattern for Forecasting
```python
from abc import ABC, abstractmethod

class ForecastingStrategy(ABC):
    """Abstract base class for forecasting strategies."""
    
    @abstractmethod
    def forecast(self, data: Dict) -> Dict:
        pass
    
    @abstractmethod
    def get_confidence_score(self, data: Dict) -> float:
        pass

class AnalogyForecaster(ForecastingStrategy):
    def forecast(self, data: Dict) -> Dict:
        # Analogy-based forecasting implementation
        pass
    
    def get_confidence_score(self, data: Dict) -> float:
        # Calculate confidence based on similarity scores
        pass

class TimeSeriesForecaster(ForecastingStrategy):
    def forecast(self, data: Dict) -> Dict:
        # Time-series forecasting implementation
        pass
    
    def get_confidence_score(self, data: Dict) -> float:
        # Calculate confidence based on model fit
        pass
```

#### 3. Factory Pattern for Model Creation
```python
class ForecastingModelFactory:
    """Factory for creating forecasting models."""
    
    _strategies = {
        'analogy': AnalogyForecaster,
        'timeseries': TimeSeriesForecaster,
        'ml_regression': MLForecaster
    }
    
    @classmethod
    def create_forecaster(cls, method: str, config: Dict) -> ForecastingStrategy:
        if method not in cls._strategies:
            raise ValueError(f"Unknown forecasting method: {method}")
        
        return cls._strategies[method](config)
    
    @classmethod
    def register_strategy(cls, name: str, strategy_class: type) -> None:
        """Register a new forecasting strategy."""
        cls._strategies[name] = strategy_class
```

### Data Models

#### 1. Using Pydantic for Data Validation
```python
from pydantic import BaseModel, validator
from typing import List, Optional, Dict
from datetime import datetime
from enum import Enum

class ProductCategory(str, Enum):
    TSHIRT = "tshirt"
    POLO = "polo"
    DRESS = "dress"
    JEANS = "jeans"
    SHIRT = "shirt"

class ProductAttributes(BaseModel):
    """Product attributes model with validation."""
    material: str
    color: str
    pattern: str
    sleeve: Optional[str] = None
    category: ProductCategory
    confidence: float
    
    @validator('confidence')
    def confidence_must_be_valid(cls, v):
        if not 0 <= v <= 1:
            raise ValueError('Confidence must be between 0 and 1')
        return v
    
    @validator('material')
    def material_must_be_standardized(cls, v):
        # Standardize material names
        material_mapping = {
            'cotton': 'Cotton',
            'polyester': 'Polyester',
            'blend': 'Cotton Blend'
        }
        return material_mapping.get(v.lower(), v.title())

class ForecastResult(BaseModel):
    """Forecast result model."""
    store_id: str
    mean_demand: float
    low_ci: float
    high_ci: float
    confidence: float
    method_used: str
    forecast_date: datetime
    
    @validator('mean_demand', 'low_ci', 'high_ci')
    def demand_must_be_positive(cls, v):
        if v < 0:
            raise ValueError('Demand values must be non-negative')
        return v
    
    class Config:
        schema_extra = {
            "example": {
                "store_id": "store_001",
                "mean_demand": 156.7,
                "low_ci": 134.2,
                "high_ci": 179.3,
                "confidence": 0.82,
                "method_used": "analogy",
                "forecast_date": "2024-12-05T10:30:00"
            }
        }
```

#### 2. Error Handling with Custom Exceptions
```python
class IntelliODMException(Exception):
    """Base exception for Intelli-ODM system."""
    def __init__(self, message: str, error_code: Optional[str] = None):
        self.message = message
        self.error_code = error_code
        super().__init__(self.message)

class DataValidationError(IntelliODMException):
    """Raised when input data validation fails."""
    pass

class AttributeExtractionError(IntelliODMException):
    """Raised when LLM attribute extraction fails."""
    pass

class ForecastingError(IntelliODMException):
    """Raised when forecasting methods fail."""
    pass

# Usage example
try:
    attributes = agent.extract_attributes(description)
except AttributeExtractionError as e:
    logger.error(f"Attribute extraction failed: {e.message}")
    # Implement fallback logic
    attributes = get_default_attributes()
```

### Logging and Monitoring

#### 1. Structured Logging
```python
import logging
import json
from datetime import datetime
from typing import Dict, Any

class StructuredLogger:
    """Structured logger for Intelli-ODM system."""
    
    def __init__(self, name: str):
        self.logger = logging.getLogger(name)
        self._setup_logger()
    
    def _setup_logger(self):
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)
    
    def log_agent_execution(self, agent_name: str, input_data: Dict, 
                           result: Dict, execution_time: float):
        """Log agent execution details."""
        log_data = {
            'timestamp': datetime.now().isoformat(),
            'agent': agent_name,
            'execution_time_seconds': execution_time,
            'input_size': len(str(input_data)),
            'output_size': len(str(result)),
            'status': 'success'
        }
        self.logger.info(f"Agent execution: {json.dumps(log_data)}")
    
    def log_error(self, error: Exception, context: Dict[str, Any] = None):
        """Log error with context."""
        log_data = {
            'timestamp': datetime.now().isoformat(),
            'error_type': type(error).__name__,
            'error_message': str(error),
            'context': context or {}
        }
        self.logger.error(f"Error occurred: {json.dumps(log_data)}")

# Usage
logger = StructuredLogger(__name__)
```

#### 2. Performance Monitoring
```python
import time
import psutil
from functools import wraps
from typing import Callable

def monitor_performance(func: Callable) -> Callable:
    """Decorator to monitor function performance."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        start_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        
        try:
            result = func(*args, **kwargs)
            status = 'success'
        except Exception as e:
            result = None
            status = 'error'
            raise
        finally:
            end_time = time.time()
            end_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
            
            performance_data = {
                'function': func.__name__,
                'execution_time': end_time - start_time,
                'memory_used_mb': end_memory - start_memory,
                'status': status
            }
            
            logger.info(f"Performance: {json.dumps(performance_data)}")
        
        return result
    return wrapper

# Usage
@monitor_performance
def expensive_operation(data):
    # Your code here
    pass
```

## Coding Standards

### 1. PEP 8 Compliance
- Line length: 88 characters (Black standard)
- Use type hints for all functions
- Docstrings for all classes and functions
- Consistent naming conventions

### 2. Code Formatting
```python
# Use Black for automatic formatting
black --line-length 88 --target-version py39 .

# Use isort for import sorting
isort --profile black .

# Configuration in pyproject.toml
[tool.black]
line-length = 88
target-version = ['py39']
include = '\.pyi?$'

[tool.isort]
profile = "black"
multi_line_output = 3
```

### 3. Type Hints
```python
from typing import Dict, List, Optional, Union, Tuple, Any
import pandas as pd

def process_sales_data(
    sales_df: pd.DataFrame,
    date_range: Tuple[str, str],
    store_filter: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Process sales data for analysis.
    
    Args:
        sales_df: Sales DataFrame with required columns
        date_range: Start and end date tuple (YYYY-MM-DD format)
        store_filter: Optional list of store IDs to include
        
    Returns:
        Dictionary containing processed sales metrics
        
    Raises:
        DataValidationError: If required columns are missing
        ValueError: If date range is invalid
    """
    pass
```

### 4. Documentation Standards
```python
class DataIngestionAgent:
    """
    Agent responsible for data ingestion, validation, and preprocessing.
    
    This agent handles the ingestion of raw CSV files containing product,
    sales, inventory, and pricing data. It validates data quality,
    standardizes formats, and engineers features for downstream analysis.
    
    Attributes:
        config: Configuration dictionary for validation and processing parameters
        logger: Structured logger instance
        
    Example:
        >>> agent = DataIngestionAgent(config={'max_missing_pct': 10})
        >>> result = agent.run(file_paths)
        >>> print(f"Processed {len(result['sales'])} sales records")
    """
    
    def validate(self, files: Dict[str, str]) -> Dict[str, pd.DataFrame]:
        """
        Validate input data files for completeness and quality.
        
        Performs comprehensive validation including:
        - File existence and readability
        - Required column presence
        - Data type consistency
        - Missing value analysis
        - Outlier detection
        
        Args:
            files: Dictionary mapping file types to file paths
                  Expected keys: 'products', 'sales', 'inventory', 'pricing'
                  
        Returns:
            Dictionary of validated DataFrames
            
        Raises:
            DataValidationError: If validation fails on critical checks
            FileNotFoundError: If required files are missing
            
        Note:
            Warnings for non-critical issues are logged but don't raise exceptions
        """
        pass
```

## Testing Strategy

### 1. Test Structure
```
tests/
├── unit/                    # Unit tests for individual components
│   ├── test_agents/
│   │   ├── test_data_ingestion.py
│   │   ├── test_attribute_analogy.py
│   │   ├── test_demand_forecasting.py
│   │   └── test_procurement_allocation.py
│   ├── test_models/
│   └── test_utils/
├── integration/             # Integration tests
│   ├── test_workflow.py
│   ├── test_knowledge_base.py
│   └── test_orchestrator.py
├── fixtures/               # Test data and fixtures
│   ├── sample_data/
│   └── mock_responses/
└── conftest.py            # pytest configuration
```

### 2. Unit Testing with pytest
```python
import pytest
import pandas as pd
from unittest.mock import Mock, patch
from agents.data_ingestion_agent import DataIngestionAgent, DataValidationError

class TestDataIngestionAgent:
    """Test suite for DataIngestionAgent."""
    
    @pytest.fixture
    def agent(self):
        """Create agent instance for testing."""
        config = {
            'max_missing_percentage': 10.0,
            'outlier_threshold': 3.0
        }
        return DataIngestionAgent(config)
    
    @pytest.fixture
    def sample_sales_data(self):
        """Create sample sales data for testing."""
        return pd.DataFrame({
            'date': ['2024-01-01', '2024-01-02', '2024-01-03'],
            'store_id': ['store_001', 'store_001', 'store_002'],
            'sku': ['P001', 'P002', 'P001'],
            'units_sold': [10, 15, 8],
            'revenue': [299, 349, 239]
        })
    
    def test_validate_valid_data(self, agent, sample_sales_data):
        """Test validation with valid data."""
        # Create temporary CSV files
        with patch('pandas.read_csv') as mock_read_csv:
            mock_read_csv.return_value = sample_sales_data
            
            files = {
                'sales': 'test_sales.csv',
                'inventory': 'test_inventory.csv',
                'pricing': 'test_pricing.csv',
                'products': 'test_products.csv'
            }
            
            result = agent.validate(files)
            assert 'sales' in result
            assert len(result['sales']) == 3
    
    def test_validate_missing_columns(self, agent):
        """Test validation with missing required columns."""
        invalid_data = pd.DataFrame({
            'date': ['2024-01-01'],
            'store_id': ['store_001']
            # Missing 'sku', 'units_sold', 'revenue'
        })
        
        with patch('pandas.read_csv') as mock_read_csv:
            mock_read_csv.return_value = invalid_data
            
            files = {'sales': 'test_sales.csv'}
            
            with pytest.raises(DataValidationError) as exc_info:
                agent.validate(files)
            
            assert "missing columns" in str(exc_info.value).lower()
    
    @pytest.mark.parametrize("missing_pct,should_raise", [
        (5.0, False),   # Below threshold
        (15.0, True),   # Above threshold
    ])
    def test_validate_missing_data_threshold(self, agent, missing_pct, should_raise):
        """Test validation with different levels of missing data."""
        # Create data with specified missing percentage
        data = pd.DataFrame({
            'date': ['2024-01-01'] * 100,
            'store_id': ['store_001'] * 100,
            'sku': ['P001'] * 100,
            'units_sold': [10] * 100,
            'revenue': [None] * int(missing_pct) + [299] * (100 - int(missing_pct))
        })
        
        with patch('pandas.read_csv') as mock_read_csv:
            mock_read_csv.return_value = data
            
            files = {'sales': 'test_sales.csv'}
            
            if should_raise:
                with pytest.raises(DataValidationError):
                    agent.validate(files)
            else:
                result = agent.validate(files)
                assert 'sales' in result
```

### 3. Integration Testing
```python
class TestWorkflowIntegration:
    """Test complete workflow integration."""
    
    @pytest.fixture
    def orchestrator(self):
        """Create orchestrator for integration testing."""
        # Mock LLM client for testing
        mock_client = Mock()
        mock_client.generate.return_value = {
            'response': '{"material": "Cotton", "color": "White"}'
        }
        
        constraints = {
            'budget': 100000,
            'MOQ': 50,
            'pack_size': 10
        }
        
        return OrchestratorAgent(mock_client, constraints)
    
    def test_end_to_end_workflow(self, orchestrator, sample_data_files):
        """Test complete end-to-end workflow."""
        input_data = {
            'files': sample_data_files,
            'product_description': 'White cotton t-shirt'
        }
        
        price_options = [299, 349, 399]
        
        # Mock knowledge base responses
        with patch('shared_knowledge_base.SharedKnowledgeBase') as mock_kb:
            mock_kb.return_value.find_similar_products.return_value = [
                {'sku': 'P001', 'similarity_score': 0.85}
            ]
            
            result = orchestrator.run_workflow(input_data, price_options)
            
            # Verify result structure
            assert 'allocation_plan' in result
            assert 'demand_forecast' in result
            assert 'business_recommendation' in result
            
            # Verify business logic
            assert result['allocation_plan']['total_procurement_qty'] > 0
            assert result['allocation_plan']['expected_revenue'] > 0
```

### 4. Mocking External Dependencies
```python
@pytest.fixture
def mock_ollama_client():
    """Mock Ollama client for testing."""
    client = Mock()
    
    # Mock attribute extraction response
    client.generate.side_effect = [
        {
            'response': json.dumps({
                'material': 'Cotton',
                'color': 'White',
                'category': 'T-Shirt',
                'confidence': 0.85
            })
        },
        {
            'response': 'Based on the comparable products, this category shows strong growth...'
        }
    ]
    
    return client

@patch('chromadb.Client')
def test_knowledge_base_operations(mock_chromadb):
    """Test knowledge base operations with mocked ChromaDB."""
    # Setup mock
    mock_collection = Mock()
    mock_chromadb.return_value.get_or_create_collection.return_value = mock_collection
    
    kb = SharedKnowledgeBase()
    
    # Test store operation
    kb.store_product('P001', {'category': 'T-Shirt'}, np.array([0.1, 0.2]), {})
    mock_collection.add.assert_called_once()
    
    # Test query operation
    mock_collection.query.return_value = {
        'ids': [['P002']],
        'distances': [[0.15]],
        'metadatas': [[{'category': 'T-Shirt'}]]
    }
    
    results = kb.find_similar_products(np.array([0.1, 0.2]), top_k=1)
    assert len(results) == 1
    assert results[0]['similarity_score'] > 0.8
```

### 5. Performance Testing
```python
import time
import pytest

class TestPerformance:
    """Performance tests for system components."""
    
    @pytest.mark.performance
    def test_data_ingestion_performance(self, large_dataset):
        """Test data ingestion performance with large dataset."""
        agent = DataIngestionAgent()
        
        start_time = time.time()
        result = agent.run(large_dataset)
        execution_time = time.time() - start_time
        
        # Performance assertions
        assert execution_time < 30.0  # Should complete within 30 seconds
        assert len(result['sales']) > 100000  # Should handle large datasets
    
    @pytest.mark.performance
    def test_forecasting_performance(self, orchestrator, sample_data):
        """Test forecasting performance."""
        start_time = time.time()
        
        result = orchestrator.run_workflow(sample_data, [299, 349, 399])
        
        execution_time = time.time() - start_time
        assert execution_time < 60.0  # Should complete within 1 minute
```

### 6. Test Configuration
**conftest.py**:
```python
import pytest
import pandas as pd
import numpy as np
import tempfile
import os

@pytest.fixture(scope="session")
def sample_data_files():
    """Create sample data files for testing."""
    # Create temporary directory
    temp_dir = tempfile.mkdtemp()
    
    # Create sample data
    products_df = pd.DataFrame({
        'product_id': ['P001', 'P002', 'P003'],
        'description': ['White cotton t-shirt', 'Blue polo shirt', 'Black jeans'],
        'category': ['TSHIRT', 'POLO', 'JEANS']
    })
    
    sales_df = pd.DataFrame({
        'date': pd.date_range('2024-01-01', '2024-03-31', freq='D').repeat(3),
        'store_id': ['store_001', 'store_002', 'store_003'] * 90,
        'sku': ['P001'] * 270,
        'units_sold': np.random.poisson(10, 270),
        'revenue': np.random.normal(3000, 500, 270)
    })
    
    # Save files
    products_file = os.path.join(temp_dir, 'products.csv')
    sales_file = os.path.join(temp_dir, 'sales.csv')
    
    products_df.to_csv(products_file, index=False)
    sales_df.to_csv(sales_file, index=False)
    
    yield {
        'products': products_file,
        'sales': sales_file,
        'inventory': sales_file,  # Reuse for simplicity
        'pricing': sales_file
    }
    
    # Cleanup
    import shutil
    shutil.rmtree(temp_dir)

# Configure pytest markers
def pytest_configure(config):
    config.addinivalue_line(
        "markers", "performance: mark test as performance test"
    )
    config.addinivalue_line(
        "markers", "integration: mark test as integration test"
    )
```

### 7. Continuous Integration
**.github/workflows/test.yml**:
```yaml
name: Test Suite

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: [3.9, 3.10, 3.11]
    
    steps:
    - uses: actions/checkout@v2
    
    - name: Set up Python
      uses: actions/setup-python@v2
      with:
        python-version: ${{ matrix.python-version }}
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt
        pip install -r requirements-dev.txt
    
    - name: Lint with flake8
      run: |
        flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics
        flake8 . --count --exit-zero --max-complexity=10 --max-line-length=88
    
    - name: Type check with mypy
      run: mypy agents/ orchestrator.py
    
    - name: Test with pytest
      run: |
        pytest tests/unit/ -v --cov=agents --cov=orchestrator
        pytest tests/integration/ -v -m "not performance"
    
    - name: Performance tests
      run: pytest tests/ -v -m performance
      continue-on-error: true  # Allow performance tests to fail without failing build
```

This development guide provides comprehensive guidance for maintaining code quality, implementing robust testing, and following best practices in the Intelli-ODM system development.