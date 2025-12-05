# Intelli-ODM Project Structure

## Directory Structure

```
intelli-odm/
├── agents/                           # Core agent implementations
│   ├── __init__.py                   # Agent package initialization
│   ├── data_ingestion_agent.py       # Data validation and cleaning
│   ├── attribute_analogy_agent.py    # LLM-powered attribute extraction
│   ├── demand_forecasting_agent.py   # Multi-method demand forecasting
│   └── procurement_allocation_agent.py # Optimization-based allocation
│
├── docs/                             # Comprehensive documentation
│   ├── architecture.md               # System architecture overview
│   ├── project_structure.md          # This file - project organization
│   ├── agent_specifications.md       # Detailed agent specifications
│   ├── api_reference.md              # Complete API documentation
│   ├── configuration_guide.md        # Setup and configuration
│   ├── user_manual.md               # End-to-end usage guide
│   └── development_guide.md          # Development and contribution guide
│
├── examples/                         # Example data and notebooks
│   ├── sample_data/                  # Sample CSV files for testing
│   │   ├── products.csv              # Product catalog
│   │   ├── sales.csv                # Historical sales data
│   │   ├── inventory.csv            # Current inventory levels
│   │   └── pricing.csv              # Current pricing data
│   ├── notebooks/                    # Jupyter notebooks
│   │   ├── end_to_end_example.ipynb # Complete workflow example
│   │   ├── agent_testing.ipynb      # Individual agent testing
│   │   └── data_exploration.ipynb   # Sample data analysis
│   └── outputs/                      # Example outputs
│       ├── sample_recommendation.json
│       └── forecast_visualization.png
│
├── config/                           # Configuration files
│   ├── __init__.py
│   ├── settings.py                   # Application settings
│   ├── agent_configs.py             # Agent-specific configurations
│   └── model_configs.py             # Model and LLM configurations
│
├── tests/                            # Test suite
│   ├── __init__.py
│   ├── test_agents/                  # Agent-specific tests
│   │   ├── test_data_ingestion.py
│   │   ├── test_attribute_analogy.py
│   │   ├── test_demand_forecasting.py
│   │   └── test_procurement_allocation.py
│   ├── test_integration.py           # Integration tests
│   ├── test_orchestrator.py          # Orchestrator tests
│   └── test_knowledge_base.py        # Knowledge base tests
│
├── utils/                            # Utility modules
│   ├── __init__.py
│   ├── data_validation.py            # Data validation utilities
│   ├── logging_config.py            # Logging configuration
│   ├── model_utils.py               # Model loading and utilities
│   └── optimization_utils.py        # Optimization helpers
│
├── orchestrator.py                   # Main orchestrator agent
├── shared_knowledge_base.py          # Vector database interface
├── requirements.txt                  # Python dependencies
├── .env.template                     # Environment variables template
├── .gitignore                        # Git ignore patterns
├── setup.py                          # Package installation script
└── README.md                         # Main project documentation
```

## File Responsibilities

### Core System Files

#### `orchestrator.py`
**Purpose**: Central coordination and workflow management

**Key Classes**:
- `OrchestratorAgent`: Main workflow coordinator

**Responsibilities**:
- Initialize and coordinate all agents
- Manage data flow between agents
- Handle errors and implement retry logic
- Synthesize final recommendations
- Provide user-friendly output formatting

#### `shared_knowledge_base.py`
**Purpose**: Persistent storage for product data and embeddings

**Key Classes**:
- `SharedKnowledgeBase`: Vector database interface
- `ProductEmbedding`: Product embedding model
- `PerformanceMetrics`: Historical performance data model

**Responsibilities**:
- Store and retrieve product embeddings
- Enable semantic similarity search
- Manage historical performance data
- Support incremental learning

### Agent Implementations

#### `agents/data_ingestion_agent.py`
**Purpose**: Data validation, cleaning, and feature engineering

**Key Classes**:
- `DataIngestionAgent`: Main data processing class
- `DataValidator`: Input validation utilities
- `FeatureEngineer`: Feature computation logic

**Input Formats**:
- Product CSV: `product_id, vendor_sku, description, category, color, material`
- Sales CSV: `date, store_id, sku, units_sold, revenue`
- Inventory CSV: `store_id, sku, on_hand, in_transit`
- Pricing CSV: `store_id, sku, price, markdown_flag`

**Output**:
- Cleaned DataFrames with standardized columns
- Computed features (velocity, price statistics, store tiers)
- Data quality reports

#### `agents/attribute_analogy_agent.py`
**Purpose**: LLM-powered attribute extraction and product similarity

**Key Classes**:
- `AttributeAnalogyAgent`: Main attribute processing class
- `AttributeExtractor`: LLM-based attribute parsing
- `SimilaritySearcher`: Product similarity computation
- `TrendAnalyzer`: Market trend analysis

**LLM Integration**:
- Uses Ollama client for Llama3 interactions
- Custom prompt templates for attribute extraction
- Structured output parsing and validation

**Output**:
- Structured product attributes (JSON format)
- List of comparable products with similarity scores
- Trend analysis and market insights

#### `agents/demand_forecasting_agent.py`
**Purpose**: Multi-method demand forecasting with price sensitivity

**Key Classes**:
- `DemandForecastingAgent`: Main forecasting orchestrator
- `AnalogyForecaster`: Analogy-based forecasting
- `TimeSeriesForecaster`: Prophet/ARIMA forecasting
- `MLForecaster`: Machine learning-based forecasting
- `PriceSensitivityAnalyzer`: Price elasticity analysis

**Forecasting Methods**:
1. **Analogy-based**: Scale comparable product sales by similarity
2. **Time-series**: Use Prophet for seasonal patterns
3. **ML Regression**: Store-specific feature-based models

**Output**:
- Store-level demand forecasts (mean, confidence intervals)
- Price sensitivity curves
- Model selection rationale

#### `agents/procurement_allocation_agent.py`
**Purpose**: Optimization-based procurement and allocation planning

**Key Classes**:
- `ProcurementAllocationAgent`: Main optimization coordinator
- `ConstraintValidator`: Business constraint checking
- `AllocationOptimizer`: Mathematical optimization solver
- `RecommendationGenerator`: Human-readable output generation

**Optimization Features**:
- Linear and mixed-integer programming
- Multiple objective functions (revenue, margin, sell-through)
- Constraint handling (MOQ, capacity, budget)
- Sensitivity analysis

**Output**:
- Optimal procurement quantity
- Store-level allocation plan
- Business recommendation with rationale

### Configuration and Setup

#### `config/settings.py`
**Purpose**: Application-wide configuration management

**Configuration Sections**:
- Database connections (ChromaDB, FAISS)
- Ollama/LLM settings
- Business constraint defaults
- Logging configuration
- Performance tuning parameters

#### `config/agent_configs.py`
**Purpose**: Agent-specific configuration parameters

**Agent Configurations**:
- Data validation thresholds
- Feature engineering parameters
- Forecasting model hyperparameters
- Optimization solver settings

#### `.env.template`
**Purpose**: Environment variable template

**Variables**:
```bash
# Ollama Configuration
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3:8b

# Knowledge Base
CHROMADB_PERSIST_DIR=./chromadb
CHROMADB_HOST=localhost
CHROMADB_PORT=8000

# Business Constraints
DEFAULT_BUDGET=1000000
DEFAULT_MOQ=200
DEFAULT_PACK_SIZE=20

# Logging
LOG_LEVEL=INFO
LOG_FILE=intelli_odm.log
```

### Documentation Structure

#### `docs/architecture.md`
High-level system architecture, design patterns, and technology stack

#### `docs/agent_specifications.md`
Detailed specifications for each agent including inputs, outputs, and algorithms

#### `docs/api_reference.md`
Complete API documentation with method signatures and examples

#### `docs/configuration_guide.md`
Setup instructions, configuration options, and troubleshooting

#### `docs/user_manual.md`
End-to-end usage guide with examples and best practices

#### `docs/development_guide.md`
Development setup, coding standards, and contribution guidelines

### Testing Structure

#### `tests/test_agents/`
Unit tests for each individual agent:
- Input/output validation
- Algorithm correctness
- Error handling scenarios

#### `tests/test_integration.py`
Integration tests for multi-agent workflows:
- End-to-end workflow testing
- Data flow validation
- Performance benchmarking

#### `tests/test_orchestrator.py`
Orchestrator-specific tests:
- Workflow coordination
- Error handling and recovery
- Result synthesis

### Example and Demo Structure

#### `examples/sample_data/`
Realistic sample data for testing and demonstrations:
- Product catalogs with various categories
- Historical sales data with seasonal patterns
- Inventory and pricing data

#### `examples/notebooks/`
Jupyter notebooks for different use cases:
- Complete workflow demonstration
- Individual agent testing and debugging
- Data exploration and analysis

## Data Models and Schemas

### Product Attributes Schema
```json
{
    "material": "string",
    "color": "string",
    "pattern": "string",
    "sleeve": "string",
    "neckline": "string",
    "fit": "string",
    "gsm": "number",
    "category": "string",
    "subcategory": "string",
    "season": "string",
    "tags": ["string"]
}
```

### Forecast Output Schema
```json
{
    "product_id": "string",
    "forecast_period": "60_days",
    "store_forecasts": {
        "store_001": {
            "mean": 150,
            "low_ci": 120,
            "high_ci": 180,
            "confidence": 0.85
        }
    },
    "price_sensitivity": {
        "299": {"units": 1200, "revenue": 358800},
        "349": {"units": 1100, "revenue": 383900}
    }
}
```

### Allocation Output Schema
```json
{
    "total_procurement_qty": 2500,
    "allocation": {
        "store_001": 150,
        "store_002": 130,
        "store_003": 120
    },
    "expected_performance": {
        "total_revenue": 875000,
        "expected_sellthrough": 0.78,
        "margin_percent": 0.45
    },
    "recommendation": "Proceed with procurement. Strong comparables and favorable trends."
}
```

This structure ensures maintainable, scalable code with clear separation of concerns and comprehensive documentation for all stakeholders.