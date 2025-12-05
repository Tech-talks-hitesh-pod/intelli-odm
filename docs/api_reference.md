# API Reference

## Overview
This document provides the complete API reference for the Intelli-ODM system, including all public methods, parameters, return types, and usage examples.

## Core Classes

### OrchestratorAgent

#### `__init__(llm_client, constraint_params)`
Initialize the orchestrator with LLM client and business constraints.

**Parameters**:
- `llm_client`: Ollama client instance for LLM communication
- `constraint_params` (Dict): Business constraint parameters
  ```python
  {
      "budget": 1000000,           # Total budget in INR
      "MOQ": 200,                  # Minimum order quantity
      "pack_size": 20,             # Package size for orders
      "lead_time_days": 30,        # Vendor lead time
      "safety_stock_factor": 1.2,  # Safety stock multiplier
      "max_store_capacity": 500    # Maximum units per store
  }
  ```

**Example**:
```python
from orchestrator import OrchestratorAgent
import ollama

client = ollama.Client(host='http://localhost:11434')
constraints = {
    "budget": 1000000,
    "MOQ": 200,
    "pack_size": 20
}
orchestrator = OrchestratorAgent(client, constraints)
```

#### `run_workflow(input_data, price_options)`
Execute the complete multi-agent workflow for demand forecasting and allocation.

**Parameters**:
- `input_data` (Dict): Input data package
  ```python
  {
      "files": {
          "products": "path/to/products.csv",
          "sales": "path/to/sales.csv",
          "inventory": "path/to/inventory.csv",
          "pricing": "path/to/pricing.csv"
      },
      "product_description": "White cotton t-shirt, short sleeves, chest print"
  }
  ```
- `price_options` (List[float]): Price points to analyze (e.g., [299, 349, 399])

**Returns**: Complete recommendation package
```python
{
    "status": "success",
    "product_info": {
        "description": "White cotton t-shirt, short sleeves, chest print",
        "extracted_attributes": ProductAttributes,
        "comparable_products": List[ComparableProduct]
    },
    "demand_forecast": Dict[str, ForecastResult],
    "price_sensitivity": PriceSensitivityAnalysis,
    "allocation_plan": AllocationPlan,
    "business_recommendation": str,
    "execution_summary": ExecutionSummary
}
```

**Example**:
```python
input_data = {
    "files": {
        "products": "data/products.csv",
        "sales": "data/sales.csv",
        "inventory": "data/inventory.csv",
        "pricing": "data/pricing.csv"
    },
    "product_description": "Navy blue cotton polo shirt with embroidered logo"
}

price_options = [399, 449, 499]
result = orchestrator.run_workflow(input_data, price_options)

print(f"Recommended procurement: {result['allocation_plan']['total_procurement_qty']} units")
print(f"Expected revenue: ₹{result['allocation_plan']['expected_revenue']:,.0f}")
```

---

## DataIngestionAgent

### `validate(files)`
Validate input data files for completeness and quality.

**Parameters**:
- `files` (Dict[str, str]): File paths dictionary

**Returns**: `ValidationReport`
```python
{
    "is_valid": bool,
    "errors": List[str],           # Critical errors preventing processing
    "warnings": List[str],         # Non-critical issues
    "file_stats": {
        "products": {
            "rows": 1500,
            "columns": 8,
            "missing_data_pct": 2.3
        }
    },
    "data_quality_score": 0.87     # Overall quality score (0-1)
}
```

**Example**:
```python
from agents.data_ingestion_agent import DataIngestionAgent

agent = DataIngestionAgent()
validation = agent.validate({
    "products": "data/products.csv",
    "sales": "data/sales.csv",
    "inventory": "data/inventory.csv",
    "pricing": "data/pricing.csv"
})

if validation["is_valid"]:
    print("Data validation passed")
else:
    print("Validation errors:", validation["errors"])
```

### `structure(raw_data)`
Standardize and clean raw data into consistent format.

**Parameters**:
- `raw_data` (Dict[str, pd.DataFrame]): Raw DataFrames from CSV files

**Returns**: Dictionary of cleaned DataFrames
```python
{
    "sales": pd.DataFrame,      # Columns: date, store_id, sku, units_sold, revenue
    "inventory": pd.DataFrame,  # Columns: store_id, sku, on_hand, in_transit
    "pricing": pd.DataFrame,    # Columns: store_id, sku, price, markdown_flag
    "products": pd.DataFrame    # Columns: product_id, description, category, attributes
}
```

### `feature_engineering(data)`
Compute derived features and metrics.

**Parameters**:
- `data` (Dict[str, pd.DataFrame]): Structured DataFrames

**Returns**: Enhanced data with computed features
```python
{
    "sales": pd.DataFrame,           # + velocity, seasonality_index
    "inventory": pd.DataFrame,       # + days_of_stock, stock_ratio
    "pricing": pd.DataFrame,         # + price_position, discount_depth
    "products": pd.DataFrame,        # + category_performance
    "store_tiers": pd.DataFrame,     # Store performance classification
    "features": {
        "velocity_stats": Dict,
        "seasonality_patterns": Dict,
        "price_elasticity_hints": Dict
    }
}
```

**Example**:
```python
# Load and process data
agent = DataIngestionAgent()
raw_data = agent.load_csv_files(file_paths)
clean_data = agent.structure(raw_data)
enriched_data = agent.feature_engineering(clean_data)

# Access computed features
velocity_stats = enriched_data["features"]["velocity_stats"]
store_tiers = enriched_data["store_tiers"]
```

---

## AttributeAnalogyAgent

### `__init__(llm_client, knowledge_base)`
Initialize with LLM client and knowledge base connection.

**Parameters**:
- `llm_client`: Ollama client for LLM interactions
- `knowledge_base`: SharedKnowledgeBase instance

### `extract_attributes(description)`
Parse product description into structured attributes using LLM.

**Parameters**:
- `description` (str): Natural language product description

**Returns**: `ProductAttributes` object
```python
{
    "material": "Cotton",
    "color": "White",
    "pattern": "Solid", 
    "sleeve": "Short",
    "neckline": "Crew",
    "fit": "Regular",
    "gsm": 180,
    "category": "T-Shirt",
    "subcategory": "Basic Tee",
    "season": "All Season",
    "style": "Casual",
    "target_gender": "Unisex",
    "tags": ["basic", "everyday", "cotton"],
    "confidence": 0.92
}
```

**Example**:
```python
from agents.attribute_analogy_agent import AttributeAnalogyAgent
import ollama

client = ollama.Client()
kb = SharedKnowledgeBase()
agent = AttributeAnalogyAgent(client, kb)

description = "Premium cotton dress shirt with French cuffs and spread collar"
attributes = agent.extract_attributes(description)

print(f"Material: {attributes['material']}")
print(f"Style: {attributes['style']}")
print(f"Confidence: {attributes['confidence']}")
```

### `find_comparable_products(attributes, top_n=5)`
Find similar products from knowledge base using attribute similarity.

**Parameters**:
- `attributes` (Dict): Structured product attributes
- `top_n` (int): Number of similar products to return

**Returns**: List of comparable products
```python
[
    {
        "sku": "TS-114",
        "similarity_score": 0.88,
        "attributes": ProductAttributes,
        "historical_performance": {
            "avg_monthly_sales": 450,
            "sell_through_rate": 0.78,
            "avg_price": 349,
            "seasonality_pattern": "spring_summer_peak"
        },
        "match_reasons": ["material_match", "category_match", "target_demographic"]
    }
]
```

**Example**:
```python
attributes = {
    "material": "Cotton",
    "category": "T-Shirt",
    "color": "White"
}

comparables = agent.find_comparable_products(attributes, top_n=3)

for comp in comparables:
    print(f"SKU: {comp['sku']}, Similarity: {comp['similarity_score']:.2f}")
    print(f"Avg Sales: {comp['historical_performance']['avg_monthly_sales']}")
```

### `analyze_trends(attributes, time_window="3M")`
Analyze market trends for the product category.

**Parameters**:
- `attributes` (Dict): Product attributes
- `time_window` (str): Analysis period ("1M", "3M", "6M", "1Y")

**Returns**: Trend analysis results
```python
{
    "category_trend": "Increasing",      # Overall category performance
    "trend_strength": 0.65,              # Trend strength (0-1)
    "seasonal_pattern": {
        "peak_months": ["March", "April", "May"],
        "low_months": ["November", "December"],
        "seasonal_factor": 1.3
    },
    "color_trends": {
        "White": {"popularity": 0.85, "trend": "Stable"},
        "Black": {"popularity": 0.78, "trend": "Increasing"}
    },
    "style_preferences": {
        "Casual": 0.72,
        "Formal": 0.28
    },
    "market_insights": "Cotton t-shirts showing strong growth in premium segment...",
    "confidence": 0.81
}
```

---

## DemandForecastingAgent

### `__init__(llm_client)`
Initialize forecasting agent with LLM client for method selection guidance.

### `select_forecast_method(comparables, store_data)`
Choose optimal forecasting approach based on data availability and quality.

**Parameters**:
- `comparables` (List[Dict]): Comparable products data
- `store_data` (Dict): Store-level historical data

**Returns**: Selected method name
```python
"analogy"          # Analogy-based forecasting
"timeseries"       # Time-series forecasting  
"ml_regression"    # ML regression forecasting
```

**Selection Logic**:
- **Analogy**: High-similarity comparables available (similarity > 0.7, count >= 3)
- **Time-series**: Sufficient historical data (>= 12 months) with clear patterns
- **ML Regression**: Large store network (>= 10 stores) with rich feature data

**Example**:
```python
from agents.demand_forecasting_agent import DemandForecastingAgent

agent = DemandForecastingAgent(llm_client)
method = agent.select_forecast_method(comparables, historical_data)
print(f"Selected forecasting method: {method}")
```

### `forecast_demand(method, **kwargs)`
Generate store-level demand forecast using specified method.

**Parameters**:
- `method` (str): Forecasting method ("analogy", "timeseries", "ml_regression")
- `**kwargs`: Method-specific parameters

**Method-specific Parameters**:

**Analogy Method**:
```python
forecast_demand(
    method="analogy",
    comparables=comparable_products,
    target_attributes=product_attributes,
    store_performance=store_data,
    forecast_days=60
)
```

**Time-series Method**:
```python
forecast_demand(
    method="timeseries",
    historical_sales=sales_df,
    seasonality=True,
    holidays=True,
    forecast_days=60
)
```

**ML Regression Method**:
```python
forecast_demand(
    method="ml_regression",
    store_features=store_df,
    product_features=product_df,
    historical_data=sales_df,
    forecast_days=60
)
```

**Returns**: Store-level forecast results
```python
{
    "store_001": {
        "mean_demand": 156.7,         # Expected units over forecast period
        "low_ci": 134.2,              # 10th percentile confidence interval
        "high_ci": 179.3,             # 90th percentile confidence interval
        "confidence": 0.82,           # Model confidence score
        "method_used": "analogy",     # Forecasting method applied
        "rationale": "Based on 3 highly similar products with avg similarity 0.84..."
    }
}
```

### `price_sensitivity_analysis(base_forecast, price_points)`
Analyze demand response to different pricing scenarios.

**Parameters**:
- `base_forecast` (Dict): Base demand forecast at current pricing
- `price_points` (List[float]): List of prices to analyze

**Returns**: Price sensitivity analysis
```python
{
    "price_elasticity": -1.2,           # Overall price elasticity coefficient
    "optimal_price": 379,               # Revenue-maximizing price point
    "price_scenarios": {
        "299": {
            "demand_multiplier": 1.15,   # Demand change vs. base price
            "expected_units": 1380,      # Total expected units across stores
            "expected_revenue": 412620,  # Total expected revenue
            "margin_impact": -0.15,      # Margin change vs. base
            "risk_assessment": "High volume, lower margin"
        },
        "349": {
            "demand_multiplier": 1.0,    # Base scenario
            "expected_units": 1200,
            "expected_revenue": 418800,
            "margin_impact": 0.0,
            "risk_assessment": "Balanced risk-return"
        },
        "399": {
            "demand_multiplier": 0.82,
            "expected_units": 984,
            "expected_revenue": 392616,
            "margin_impact": 0.22,
            "risk_assessment": "Lower volume, higher margin"
        }
    },
    "recommendations": [
        "Price point ₹349 maximizes revenue while maintaining healthy volume",
        "Avoid pricing above ₹399 due to steep demand decline"
    ]
}
```

**Example**:
```python
# Generate base forecast
forecast = agent.forecast_demand("analogy", comparables=comparables, ...)

# Analyze price sensitivity
price_points = [299, 329, 349, 379, 399, 429]
sensitivity = agent.price_sensitivity_analysis(forecast, price_points)

# Find optimal price point
optimal_price = sensitivity["optimal_price"]
optimal_revenue = sensitivity["price_scenarios"][str(optimal_price)]["expected_revenue"]
print(f"Optimal price: ₹{optimal_price}, Expected revenue: ₹{optimal_revenue:,.0f}")
```

---

## ProcurementAllocationAgent

### `__init__(constraint_params)`
Initialize with business constraint parameters.

**Parameters**:
- `constraint_params` (Dict): Business constraints configuration

### `validate_constraints(demand_forecast, constraints)`
Check if demand forecast is feasible given business constraints.

**Parameters**:
- `demand_forecast` (Dict): Store-level demand predictions
- `constraints` (Dict): Business constraint parameters

**Returns**: Constraint validation report
```python
{
    "is_feasible": True,
    "violated_constraints": [],          # List of constraint violations
    "warnings": [                        # Non-critical warnings
        "Store_005 demand exceeds typical capacity by 15%"
    ],
    "adjustments_needed": {              # Suggested adjustments
        "reduce_allocation_store_005": 50,
        "increase_safety_stock": 0.1
    },
    "risk_assessment": "Medium risk due to capacity constraints in 2 stores"
}
```

**Example**:
```python
from agents.procurement_allocation_agent import ProcurementAllocationAgent

constraints = {
    "budget": 500000,
    "MOQ": 200,
    "pack_size": 20,
    "max_store_capacity": 300
}

agent = ProcurementAllocationAgent(constraints)
validation = agent.validate_constraints(demand_forecast, constraints)

if not validation["is_feasible"]:
    print("Constraint violations:", validation["violated_constraints"])
```

### `optimize_allocation(demand_forecast, constraints, objective="revenue")`
Solve optimization problem for procurement quantity and store allocation.

**Parameters**:
- `demand_forecast` (Dict): Store-level demand forecasts
- `constraints` (Dict): Business constraints
- `objective` (str): Optimization objective

**Objective Options**:
- `"revenue"`: Maximize total revenue
- `"margin"`: Maximize total margin
- `"sellthrough"`: Maximize sell-through rate
- `"risk_adjusted"`: Maximize risk-adjusted return

**Returns**: Optimization results
```python
{
    "status": "optimal",                 # Solver status
    "total_procurement_qty": 2400,       # Total units to procure
    "store_allocations": {               # Units allocated per store
        "store_001": 180,
        "store_002": 160,
        "store_003": 140,
        # ...
    },
    "expected_performance": {
        "total_revenue": 836400,         # Expected total revenue
        "total_margin": 376380,          # Expected total margin
        "expected_sellthrough": 0.78,    # Expected sell-through rate
        "inventory_risk": 0.23           # Risk of excess inventory
    },
    "sensitivity_analysis": {
        "budget_sensitivity": 0.85,      # Revenue change per budget unit
        "capacity_constraints": [        # Active capacity constraints
            "store_001", "store_003"
        ],
        "moq_impact": 15                 # Units above optimal due to MOQ
    },
    "solver_details": {
        "optimization_time": 0.127,      # Solver execution time (seconds)
        "iterations": 23,                # Solver iterations
        "objective_value": 836400        # Final objective function value
    }
}
```

**Example**:
```python
# Run optimization
result = agent.optimize_allocation(
    demand_forecast=forecast_results,
    constraints=business_constraints,
    objective="revenue"
)

# Extract key results
total_qty = result["total_procurement_qty"]
total_revenue = result["expected_performance"]["total_revenue"]
sellthrough = result["expected_performance"]["expected_sellthrough"]

print(f"Procure {total_qty} units")
print(f"Expected revenue: ₹{total_revenue:,.0f}")
print(f"Expected sell-through: {sellthrough:.1%}")

# Check store allocations
allocations = result["store_allocations"]
for store, qty in allocations.items():
    print(f"{store}: {qty} units")
```

### `generate_recommendation(optimization_result, forecast)`
Create human-readable business recommendation.

**Parameters**:
- `optimization_result` (Dict): Output from optimization
- `forecast` (Dict): Original demand forecast

**Returns**: Formatted recommendation string
```python
"""
PROCUREMENT RECOMMENDATION FOR PRODUCT: Navy Cotton Polo Shirt

EXECUTIVE SUMMARY
✓ Recommendation: PROCEED with procurement
✓ Procurement Quantity: 2,400 units
✓ Investment Required: ₹432,000
✓ Expected Revenue: ₹836,400
✓ Expected Margin: ₹376,380 (45%)
✓ Expected Sell-through: 78% (60 days)

KEY INSIGHTS
• Strong comparable products indicate healthy demand (avg similarity: 0.84)
• Seasonal timing favorable (spring/summer peak approaching)
• Optimal price point identified at ₹349
• 15 stores can absorb 80% of inventory in first 30 days

ALLOCATION STRATEGY
• Prioritize Tier 1 stores (40% of allocation)
• Balanced distribution to Tier 2 stores (45% allocation)
• Conservative allocation to Tier 3/Outlet stores (15%)

RISK ASSESSMENT
• LOW-MEDIUM risk profile
• Main risks: Seasonal demand shift, competitive pricing pressure
• Mitigation: Monitor sell-through in first 2 weeks, adjust pricing if needed

IMPLEMENTATION
• Place order with vendor (MOQ: 200 units satisfied)
• Coordinate delivery to distribution center
• Implement staged rollout: Tier 1 stores first, then Tier 2/3
"""
```

---

## SharedKnowledgeBase

### `__init__(config=None)`
Initialize vector database connection.

**Parameters**:
- `config` (Dict, optional): Database configuration

### `store_product(product_id, attributes, embeddings, metadata)`
Store product information in the knowledge base.

**Parameters**:
- `product_id` (str): Unique product identifier
- `attributes` (Dict): Structured product attributes
- `embeddings` (np.ndarray): Vector representation
- `metadata` (Dict): Additional metadata

**Example**:
```python
kb = SharedKnowledgeBase()

# Store new product
kb.store_product(
    product_id="POLO-001",
    attributes=extracted_attributes,
    embeddings=product_embedding,
    metadata={
        "date_added": "2024-01-15",
        "category_performance": 0.78,
        "seasonal_pattern": "spring_summer"
    }
)
```

### `find_similar_products(query_embedding, top_k=5)`
Find similar products using vector similarity search.

**Parameters**:
- `query_embedding` (np.ndarray): Query vector
- `top_k` (int): Number of results to return

**Returns**: List of similar products with similarity scores
```python
[
    {
        "product_id": "TS-114",
        "similarity_score": 0.88,
        "attributes": Dict,
        "metadata": Dict
    }
]
```

### `query_performance_data(product_ids, metrics, time_period)`
Retrieve historical performance data for products.

**Parameters**:
- `product_ids` (List[str]): Product identifiers
- `metrics` (List[str]): Performance metrics to retrieve
- `time_period` (str): Time period for analysis

**Returns**: Performance data dictionary

---

## Data Models

### ProductAttributes
```python
@dataclass
class ProductAttributes:
    material: str
    color: str
    pattern: str
    sleeve: str
    neckline: str
    fit: str
    gsm: Optional[int]
    category: str
    subcategory: str
    season: str
    style: str
    target_gender: str
    tags: List[str]
    confidence: float
```

### ForecastResult
```python
@dataclass
class ForecastResult:
    store_id: str
    mean_demand: float
    low_ci: float
    high_ci: float
    confidence: float
    method_used: str
    rationale: str
    forecast_date: datetime
```

### AllocationPlan
```python
@dataclass
class AllocationPlan:
    total_procurement_qty: int
    store_allocations: Dict[str, int]
    expected_revenue: float
    expected_margin: float
    expected_sellthrough: float
    risk_score: float
    solver_status: str
    optimization_time: float
```

## Exception Classes

```python
class IntelliODMException(Exception):
    """Base exception for Intelli-ODM system"""

class DataValidationError(IntelliODMException):
    """Raised when input data validation fails"""

class AttributeExtractionError(IntelliODMException):
    """Raised when LLM attribute extraction fails"""

class ForecastingError(IntelliODMException):
    """Raised when forecasting methods fail"""

class OptimizationError(IntelliODMException):
    """Raised when optimization solver fails"""

class KnowledgeBaseError(IntelliODMException):
    """Raised when knowledge base operations fail"""
```

## Usage Examples

### Complete Workflow Example
```python
import ollama
from orchestrator import OrchestratorAgent

# Initialize system
client = ollama.Client(host='http://localhost:11434')
constraints = {
    "budget": 750000,
    "MOQ": 200,
    "pack_size": 20,
    "lead_time_days": 30
}

orchestrator = OrchestratorAgent(client, constraints)

# Prepare input data
input_data = {
    "files": {
        "products": "data/products.csv",
        "sales": "data/sales_history.csv", 
        "inventory": "data/current_inventory.csv",
        "pricing": "data/current_pricing.csv"
    },
    "product_description": "Premium cotton dress shirt, white, spread collar, French cuffs"
}

price_options = [799, 899, 999, 1099]

# Execute workflow
try:
    result = orchestrator.run_workflow(input_data, price_options)
    
    # Extract key results
    procurement_qty = result["allocation_plan"]["total_procurement_qty"]
    expected_revenue = result["allocation_plan"]["expected_revenue"]
    recommendation = result["business_recommendation"]
    
    print(f"Recommendation: Procure {procurement_qty} units")
    print(f"Expected Revenue: ₹{expected_revenue:,.0f}")
    print(f"\nDetailed Recommendation:\n{recommendation}")
    
except Exception as e:
    print(f"Workflow failed: {e}")
```

This API reference provides comprehensive documentation for integrating and using the Intelli-ODM system in production environments.