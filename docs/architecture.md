# Intelli-ODM Architecture Documentation

## Overview
Intelli-ODM is a multi-agent system that leverages Large Language Models (LLM) and traditional data science techniques to solve retail demand forecasting and procurement optimization problems.

## System Architecture

### High-Level Design Pattern
The system follows a **Hierarchical Multi-Agent Architecture** where specialized agents work together under the coordination of an Orchestrator Agent.

```
┌─────────────────────────────────────────────────────────────────┐
│                    ORCHESTRATOR AGENT                          │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │            Workflow Coordination                        │   │
│  │  • Data Flow Management                                 │   │
│  │  • Error Handling & Recovery                           │   │
│  │  • Result Synthesis                                    │   │
│  └─────────────────────────────────────────────────────────┘   │
└─────────────────────────┬───────────────────────────────────────┘
                          │
          ┌───────────────┼───────────────┐
          │               │               │
          ▼               ▼               ▼
┌─────────────────┐ ┌─────────────┐ ┌──────────────┐
│ DATA INGESTION  │ │ ATTRIBUTE & │ │   DEMAND     │
│     AGENT       │ │   ANALOGY   │ │ FORECASTING  │
│                 │ │    AGENT    │ │    AGENT     │
│ • Validation    │ │ • LLM Parse │ │ • Model      │
│ • Cleaning      │ │ • Similarity│ │   Selection  │
│ • Features      │ │ • Trends    │ │ • Forecast   │
└─────────────────┘ └─────────────┘ └──────────────┘
          │               │               │
          └───────────────┼───────────────┘
                          ▼
                ┌──────────────────────┐
                │  PROCUREMENT &       │
                │  ALLOCATION AGENT    │
                │                      │
                │ • Constraint Check   │
                │ • Optimization       │
                │ • Final Allocation   │
                └──────────────────────┘
```

## Core Components

### 1. Orchestrator Agent (`orchestrator.py`)
**Role**: Central coordinator and workflow manager

**Responsibilities**:
- Coordinate execution flow across all agents
- Manage data passing between agents
- Handle errors and recovery scenarios
- Synthesize final recommendations
- Maintain execution context and logging

**Key Methods**:
```python
run_workflow(input_data, price_options) -> final_recommendation
```

### 2. Data Ingestion Agent (`agents/data_ingestion_agent.py`)
**Role**: Data validation, cleaning, and feature engineering

**Responsibilities**:
- Validate input files (products, sales, inventory, pricing)
- Standardize data formats and column names
- Handle missing data and outliers
- Engineer features (velocity, price bands, store tiers)
- Output clean, structured DataFrames

**Key Methods**:
```python
validate(files) -> validation_report
structure(raw_data) -> {sales_df, inventory_df, price_df, products_df}
feature_engineering(df) -> engineered_features
run(files) -> cleaned_structured_features
```

### 3. Attribute & Analogy Agent (`agents/attribute_analogy_agent.py`)
**Role**: LLM-powered attribute extraction and comparable product discovery

**Responsibilities**:
- Parse natural language product descriptions using Llama3
- Extract structured attributes (material, color, pattern, etc.)
- Generate semantic embeddings for products
- Query SharedKnowledgeBase for similar products
- Analyze market trends and seasonal patterns

**Key Methods**:
```python
extract_attributes(description) -> structured_attributes
find_comparables(attributes, top_n=5) -> comparable_products
analyze_trends(attributes) -> trend_analysis
run(product_description) -> (attributes, comparables, trends)
```

### 4. Demand Forecasting Agent (`agents/demand_forecasting_agent.py`)
**Role**: Store-level demand prediction with multiple forecasting approaches

**Responsibilities**:
- Select optimal forecasting model based on data availability
- Execute forecasting using multiple methods:
  - Analogy-based scaling
  - Time-series models (Prophet, ARIMA)
  - Machine learning regression
- Generate confidence intervals and scenarios
- Perform price sensitivity analysis

**Key Methods**:
```python
select_forecast_method(comparables_data) -> method_choice
forecast_by_analogy(comparables, store_data) -> forecast
forecast_by_timeseries(historical_data) -> forecast
price_sensitivity_analysis(forecast, price_points) -> elasticity
run(comparables, sales_df, inventory_df, price_df, price_options) -> (forecast, sensitivity)
```

### 5. Procurement & Allocation Agent (`agents/procurement_allocation_agent.py`)
**Role**: Optimization-based procurement and store allocation

**Responsibilities**:
- Validate business constraints (MOQ, budget, capacity)
- Formulate optimization problem (linear/mixed-integer programming)
- Solve using optimization libraries (PuLP, CVXPY)
- Generate actionable recommendations with explanations

**Key Methods**:
```python
validate_constraints(data) -> constraint_report
optimize_allocation(demand_forecast, constraints) -> allocation_plan
generate_recommendation(allocation_plan) -> human_readable_recommendation
run(forecast, inventory, price_data) -> final_recommendation
```

### 6. Shared Knowledge Base (`shared_knowledge_base.py`)
**Role**: Persistent storage for product embeddings, metadata, and historical performance

**Responsibilities**:
- Store product embeddings and metadata
- Enable semantic similarity search
- Maintain historical sales and performance data
- Support incremental learning and updates

**Key Methods**:
```python
store_product(product_id, attributes, embeddings, metadata)
find_similar_products(query_embedding, top_k) -> similar_products
store_performance_data(product_id, performance_metrics)
query_trends(category, timeframe) -> trend_data
```

## Data Flow Architecture

### Input Data Pipeline
```
Raw Files (CSV/JSON)
    │
    ▼ [Data Ingestion Agent]
Cleaned DataFrames
    │
    ▼ [Attribute Agent]
Product Attributes + Comparables
    │
    ▼ [Demand Agent]
Store-level Forecasts + Price Sensitivity
    │
    ▼ [Procurement Agent]
Final Allocation Plan + Recommendations
```

### Knowledge Base Integration
```
Product Description
    │
    ▼ [LLM Processing]
Structured Attributes
    │
    ▼ [Embedding Generation]
Vector Embeddings
    │
    ▼ [Similarity Search]
Comparable Products + Historical Performance
```

## Technology Stack

### LLM Integration
- **Ollama**: Local LLM hosting
- **Llama3-8B**: Core reasoning model
- **LangChain**: LLM workflow management

### Data Processing
- **Pandas**: Data manipulation
- **NumPy**: Numerical computing
- **SciPy**: Statistical functions

### Machine Learning
- **Scikit-learn**: ML algorithms
- **Prophet**: Time-series forecasting
- **Statsmodels**: Statistical modeling

### Optimization
- **PuLP**: Linear programming
- **CVXPY**: Convex optimization

### Vector Storage
- **ChromaDB**: Vector database
- **Sentence-Transformers**: Embedding generation
- **FAISS**: Similarity search

## Design Principles

### 1. Separation of Concerns
Each agent has a single, well-defined responsibility, enabling independent development and testing.

### 2. LLM for Qualitative Tasks
LLMs handle reasoning, attribute extraction, and explanations. Quantitative tasks use specialized libraries for reliability and interpretability.

### 3. Modular Architecture
Components can be replaced or upgraded independently (e.g., switching from analogy-based to ML forecasting).

### 4. Data Privacy
All processing happens locally with Ollama, ensuring sensitive retail data never leaves the organization.

### 5. Extensibility
The agent-based architecture allows easy addition of new capabilities (e.g., dynamic pricing, assortment optimization).

## Error Handling Strategy

### Agent-Level Error Handling
- Each agent validates inputs and provides meaningful error messages
- Graceful degradation when optional components fail
- Fallback mechanisms for critical operations

### Orchestrator-Level Recovery
- Retry logic for transient failures
- Alternative workflow paths when agents fail
- Comprehensive logging for debugging

## Performance Considerations

### Scalability
- Agents can be parallelized for processing multiple products
- Knowledge base supports horizontal scaling
- Optimization problems can be decomposed for large store networks

### Caching Strategy
- Product embeddings cached in knowledge base
- Forecast results cached for similar products
- Model artifacts cached for repeated use

## Security Considerations

### Data Protection
- Local processing ensures data privacy
- No external API calls for sensitive operations
- Configurable data retention policies

### Access Control
- Role-based access to different system components
- Audit logging for all operations
- Secure configuration management