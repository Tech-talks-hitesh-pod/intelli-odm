# Agent Specifications

## Overview
This document provides detailed specifications for each agent in the Intelli-ODM system, including their interfaces, algorithms, and implementation details.

## 1. Data Ingestion Agent

### Responsibility
Transform raw input files into clean, structured data ready for downstream analysis.

### Class: `DataIngestionAgent`

#### Public Methods

##### `validate(files: Dict[str, str]) -> ValidationReport`
**Purpose**: Validate input file formats, completeness, and data quality.

**Parameters**:
- `files`: Dictionary mapping file types to file paths
  - `"products"`: Product catalog CSV path
  - `"sales"`: Historical sales CSV path  
  - `"inventory"`: Current inventory CSV path
  - `"pricing"`: Current pricing CSV path

**Returns**: `ValidationReport` object containing:
```python
{
    "is_valid": bool,
    "errors": List[str],
    "warnings": List[str],
    "file_stats": Dict[str, Dict],
    "data_quality_score": float
}
```

**Validation Rules**:
- File existence and readability
- Required columns presence
- Data type consistency
- Date format validation
- Missing value percentage < 10%
- Duplicate record detection

##### `structure(raw_data: Dict) -> Dict[str, pd.DataFrame]`
**Purpose**: Standardize column names, data types, and formats.

**Parameters**:
- `raw_data`: Dictionary of raw DataFrames from CSV files

**Returns**: Dictionary of cleaned DataFrames:
```python
{
    "sales": pd.DataFrame,      # Standardized sales data
    "inventory": pd.DataFrame,  # Standardized inventory data
    "pricing": pd.DataFrame,    # Standardized pricing data
    "products": pd.DataFrame    # Standardized product data
}
```

**Standardization Rules**:
- Column name normalization (lowercase, underscore-separated)
- Date parsing and standardization
- Price formatting (numeric)
- SKU/ID standardization

##### `feature_engineering(data: Dict[str, pd.DataFrame]) -> Dict`
**Purpose**: Compute derived features for downstream analysis.

**Parameters**:
- `data`: Dictionary of structured DataFrames

**Returns**: Dictionary containing original DataFrames plus engineered features:
```python
{
    "sales": pd.DataFrame,       # Enhanced with velocity, seasonality
    "inventory": pd.DataFrame,   # Enhanced with stock metrics
    "pricing": pd.DataFrame,     # Enhanced with price bands
    "products": pd.DataFrame,    # Enhanced with category groupings
    "store_tiers": pd.DataFrame, # Store classification
    "features": Dict             # Computed feature metadata
}
```

**Engineered Features**:
- Sales velocity (units per day/week)
- Seasonal patterns (weekly/monthly averages)
- Price positioning (percentiles within category)
- Store performance tiers (A/B/C classification)
- Days of stock coverage
- Historical sell-through rates

##### `run(files: Dict[str, str]) -> Dict`
**Purpose**: Execute complete data ingestion pipeline.

**Parameters**:
- `files`: File paths dictionary

**Returns**: Complete processed data package

#### Internal Methods

##### `_clean_sales_data(df: pd.DataFrame) -> pd.DataFrame`
- Remove zero/negative sales
- Handle date gaps
- Detect and flag outliers

##### `_classify_stores(sales_df: pd.DataFrame) -> pd.DataFrame`
- Compute store performance metrics
- Assign tier classifications
- Calculate store-specific features

##### `_compute_velocity_metrics(sales_df: pd.DataFrame) -> pd.DataFrame`
- Weekly/monthly sales rates
- Seasonal adjustment factors
- Trend indicators

---

## 2. Attribute & Analogy Agent

### Responsibility
Extract structured attributes from product descriptions and find comparable products using LLM reasoning.

### Class: `AttributeAnalogyAgent`

#### Initialization
```python
def __init__(self, llm_client, knowledge_base):
    self.llm = llm_client
    self.kb = knowledge_base
    self.attribute_extractor = AttributeExtractor(llm_client)
    self.similarity_searcher = SimilaritySearcher(knowledge_base)
```

#### Public Methods

##### `extract_attributes(description: str) -> Dict`
**Purpose**: Parse natural language product description into structured attributes.

**Parameters**:
- `description`: Free-text product description

**Returns**: Structured attributes dictionary:
```python
{
    "material": str,           # e.g., "Cotton", "Polyester"
    "color": str,             # e.g., "White", "Navy Blue"
    "pattern": str,           # e.g., "Solid", "Stripes", "Printed"
    "sleeve": str,            # e.g., "Short", "Long", "Sleeveless"
    "neckline": str,          # e.g., "Crew", "V-neck", "Scoop"
    "fit": str,               # e.g., "Regular", "Slim", "Oversized"
    "gsm": int,               # Fabric weight (if applicable)
    "category": str,          # e.g., "T-Shirt", "Dress", "Jeans"
    "subcategory": str,       # More specific classification
    "season": str,            # e.g., "Spring/Summer", "Fall/Winter"
    "style": str,             # e.g., "Casual", "Formal", "Athletic"
    "target_gender": str,     # e.g., "Men", "Women", "Unisex"
    "tags": List[str],        # Additional descriptive tags
    "confidence": float       # Extraction confidence score
}
```

**LLM Prompt Template**:
```
Extract structured attributes from this product description. 
Respond in JSON format with the following fields: {field_list}

Description: "{product_description}"

Guidelines:
- Use standardized terms for each attribute
- If information is missing, use null
- Provide confidence score between 0-1
```

##### `find_comparable_products(attributes: Dict, top_n: int = 5) -> List[Dict]`
**Purpose**: Find similar products from the knowledge base.

**Parameters**:
- `attributes`: Structured product attributes
- `top_n`: Number of similar products to return

**Returns**: List of comparable products:
```python
[
    {
        "sku": str,
        "similarity_score": float,
        "attributes": Dict,
        "historical_performance": Dict,
        "match_reasons": List[str]
    }
]
```

**Similarity Algorithm**:
1. Generate embedding for input attributes
2. Query vector database for semantic similarity
3. Apply attribute-specific weights
4. Filter by category relevance
5. Rank by composite similarity score

##### `analyze_trends(attributes: Dict, time_window: str = "3M") -> Dict`
**Purpose**: Analyze market trends for the product category.

**Parameters**:
- `attributes`: Product attributes
- `time_window`: Analysis time window ("1M", "3M", "6M", "1Y")

**Returns**: Trend analysis:
```python
{
    "category_trend": str,        # "Increasing", "Stable", "Declining"
    "seasonal_pattern": Dict,     # Monthly performance patterns
    "color_trends": Dict,         # Popular colors in category
    "style_preferences": Dict,    # Style trend analysis
    "market_insights": str,       # LLM-generated insights
    "confidence": float
}
```

##### `run(product_description: str) -> Tuple[Dict, List[Dict], Dict]`
**Purpose**: Execute complete attribute analysis workflow.

**Returns**: Tuple of (attributes, comparables, trends)

#### Internal Methods

##### `_validate_attributes(attributes: Dict) -> Dict`
- Check attribute value consistency
- Apply business rules validation
- Flag potential errors

##### `_generate_embedding(attributes: Dict) -> np.ndarray`
- Create vector representation of attributes
- Use sentence transformer for text attributes
- Combine with categorical encodings

##### `_rank_by_business_rules(comparables: List, attributes: Dict) -> List`
- Apply business-specific similarity rules
- Weight by category importance
- Consider seasonal relevance

---

## 3. Demand Forecasting Agent

### Responsibility
Generate store-level demand forecasts using multiple forecasting methodologies.

### Class: `DemandForecastingAgent`

#### Initialization
```python
def __init__(self, llm_client):
    self.llm = llm_client
    self.analogy_forecaster = AnalogyForecaster()
    self.timeseries_forecaster = TimeSeriesForecaster()
    self.ml_forecaster = MLForecaster()
```

#### Public Methods

##### `select_forecast_method(comparables: List[Dict], store_data: Dict) -> str`
**Purpose**: Choose optimal forecasting method based on data availability and quality.

**Parameters**:
- `comparables`: List of comparable products
- `store_data`: Historical store performance data

**Returns**: Selected method ("analogy", "timeseries", "ml_regression")

**Selection Logic**:
```python
if len(comparables) >= 3 and avg_similarity > 0.7:
    return "analogy"
elif historical_data_months >= 12 and seasonality_detected:
    return "timeseries"
elif store_count >= 10 and feature_richness > 0.6:
    return "ml_regression"
else:
    return "analogy"  # fallback
```

##### `forecast_demand(method: str, **kwargs) -> Dict[str, Dict]`
**Purpose**: Generate store-level demand forecast using specified method.

**Parameters**:
- `method`: Forecasting method to use
- `**kwargs`: Method-specific parameters

**Returns**: Store-level forecasts:
```python
{
    "store_001": {
        "mean_demand": float,      # Expected units (60-day)
        "low_ci": float,           # 10th percentile
        "high_ci": float,          # 90th percentile
        "confidence": float,       # Model confidence
        "method_used": str,        # Forecasting method
        "rationale": str          # LLM-generated explanation
    }
}
```

##### `price_sensitivity_analysis(base_forecast: Dict, price_points: List[float]) -> Dict`
**Purpose**: Analyze demand sensitivity to different price points.

**Parameters**:
- `base_forecast`: Base demand forecast
- `price_points`: List of prices to analyze

**Returns**: Price sensitivity analysis:
```python
{
    "price_elasticity": float,           # Overall elasticity coefficient
    "optimal_price": float,              # Revenue-maximizing price
    "price_scenarios": {
        "299": {
            "demand_multiplier": 1.15,
            "expected_units": 1380,
            "expected_revenue": 412200,
            "margin_impact": 0.42
        }
    }
}
```

##### `run(comparables, sales_df, inventory_df, price_df, price_options) -> Tuple[Dict, Dict]`
**Purpose**: Execute complete demand forecasting workflow.

**Returns**: Tuple of (forecast_results, sensitivity_analysis)

#### Forecasting Methods

##### Analogy-Based Forecasting
```python
def forecast_by_analogy(comparables, target_attributes, store_data):
    """
    Scale comparable product performance by:
    1. Attribute similarity weights
    2. Store-specific adjustment factors
    3. Seasonal adjustment
    4. Market trend adjustment
    """
    base_performance = weighted_average(comparables, similarity_weights)
    store_adjustments = compute_store_factors(store_data)
    seasonal_adjustments = compute_seasonal_factors(target_attributes)
    return base_performance * store_adjustments * seasonal_adjustments
```

##### Time-Series Forecasting
```python
def forecast_by_timeseries(historical_data, forecast_horizon=60):
    """
    Use Prophet model for trend and seasonality:
    1. Fit Prophet model to historical sales
    2. Account for holidays and events
    3. Generate probabilistic forecasts
    4. Adjust for product lifecycle stage
    """
    model = Prophet(seasonality_mode='multiplicative')
    model.add_country_holidays('IN')  # India holidays
    model.fit(historical_data)
    return model.predict(future_dates)
```

##### ML Regression Forecasting
```python
def forecast_by_ml(store_features, product_features, target_variable):
    """
    Feature-based regression model:
    1. Store features (size, tier, demographics)
    2. Product features (category, price, attributes)
    3. Temporal features (season, trends)
    4. Cross-product interactions
    """
    features = combine_features(store_features, product_features)
    model = GradientBoostingRegressor(n_estimators=100)
    model.fit(features, target_variable)
    return model.predict(new_product_features)
```

---

## 4. Procurement & Allocation Agent

### Responsibility
Optimize procurement quantity and store allocation subject to business constraints.

### Class: `ProcurementAllocationAgent`

#### Initialization
```python
def __init__(self, constraint_params):
    self.constraints = constraint_params
    self.optimizer = AllocationOptimizer()
    self.validator = ConstraintValidator()
```

#### Public Methods

##### `validate_constraints(demand_forecast: Dict, constraints: Dict) -> Dict`
**Purpose**: Check business constraints against forecast data.

**Parameters**:
- `demand_forecast`: Store-level demand predictions
- `constraints`: Business constraint parameters

**Returns**: Constraint validation report:
```python
{
    "is_feasible": bool,
    "violated_constraints": List[str],
    "warnings": List[str],
    "adjustments_needed": Dict,
    "risk_assessment": str
}
```

**Constraint Types**:
- Minimum Order Quantity (MOQ)
- Budget limitations
- Store capacity limits
- Pack size requirements
- Lead time constraints
- Vendor minimum quantities

##### `optimize_allocation(demand_forecast: Dict, constraints: Dict, objective: str = "revenue") -> Dict`
**Purpose**: Solve optimization problem for procurement and allocation.

**Parameters**:
- `demand_forecast`: Store-level demand forecasts
- `constraints`: Business constraints
- `objective`: Optimization objective ("revenue", "margin", "sellthrough")

**Returns**: Optimization results:
```python
{
    "total_procurement_qty": int,
    "store_allocations": Dict[str, int],
    "expected_performance": {
        "total_revenue": float,
        "total_margin": float,
        "expected_sellthrough": float,
        "risk_metrics": Dict
    },
    "sensitivity_analysis": Dict,
    "solver_status": str,
    "optimization_time": float
}
```

**Optimization Formulation**:
```
Maximize: Σ(store_allocation[i] * price[i] * sellthrough_prob[i])
Subject to:
  - Σ(store_allocation[i]) >= MOQ
  - store_allocation[i] <= store_capacity[i] ∀i
  - store_allocation[i] <= demand_forecast[i] * safety_factor ∀i
  - Σ(store_allocation[i] * cost) <= budget
  - store_allocation[i] % pack_size == 0 ∀i
```

##### `generate_recommendation(optimization_result: Dict, forecast: Dict) -> str`
**Purpose**: Create human-readable business recommendation.

**Parameters**:
- `optimization_result`: Output from optimization
- `forecast`: Original demand forecast

**Returns**: Formatted recommendation text with:
- Executive summary
- Key metrics and performance expectations
- Risk assessment
- Implementation guidance

##### `run(forecast, inventory_data, price_data) -> Dict`
**Purpose**: Execute complete procurement optimization workflow.

#### Optimization Algorithms

##### Linear Programming Solver
```python
def solve_linear_allocation(demand, constraints):
    """
    Standard LP formulation for allocation:
    - Linear objective function
    - Linear constraints
    - Continuous variables (rounded post-solution)
    """
    prob = pulp.LpProblem("Allocation", pulp.LpMaximize)
    # Add variables, objective, constraints
    prob.solve()
    return extract_solution(prob)
```

##### Mixed-Integer Programming
```python
def solve_integer_allocation(demand, constraints):
    """
    MIP formulation for discrete allocations:
    - Integer allocation variables
    - Binary selection variables
    - Pack size constraints
    """
    prob = pulp.LpProblem("AllocationMIP", pulp.LpMaximize)
    # Add integer variables and constraints
    prob.solve()
    return extract_solution(prob)
```

##### Multi-Objective Optimization
```python
def solve_multiobjective(demand, constraints, weights):
    """
    Weighted combination of objectives:
    - Revenue maximization
    - Risk minimization
    - Sell-through optimization
    """
    objectives = {
        'revenue': compute_revenue_objective,
        'risk': compute_risk_objective,
        'sellthrough': compute_sellthrough_objective
    }
    combined_objective = weighted_sum(objectives, weights)
    return optimize(combined_objective, constraints)
```

---

## Shared Data Models

### ValidationReport
```python
@dataclass
class ValidationReport:
    is_valid: bool
    errors: List[str]
    warnings: List[str]
    file_stats: Dict[str, Dict]
    data_quality_score: float
    timestamp: datetime
```

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

## Error Handling

### Common Exceptions
```python
class IntelliODMException(Exception):
    """Base exception for Intelli-ODM system"""

class DataValidationError(IntelliODMException):
    """Raised when data validation fails"""

class AttributeExtractionError(IntelliODMException):
    """Raised when LLM attribute extraction fails"""

class ForecastingError(IntelliODMException):
    """Raised when forecasting methods fail"""

class OptimizationError(IntelliODMException):
    """Raised when optimization solver fails"""
```

### Error Recovery Strategies
- Graceful degradation to simpler methods
- Fallback to default parameters
- User notification with actionable guidance
- Detailed logging for debugging