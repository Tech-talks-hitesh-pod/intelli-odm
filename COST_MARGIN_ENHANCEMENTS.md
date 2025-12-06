# Cost, Margin Target, and Top Banner Enhancements

## Overview

Enhanced the Demand Forecasting Agent to maximize ROS, sell-through, and minimize margin erosion by:
1. Checking cost of procuring articles
2. Analyzing historical MRP, selling price, discount, and margin
3. Adding margin target input from UI
4. Implementing per-store maximum quantity constraint (500)
5. Displaying pricing metrics with consistency checks
6. Adding top banner summary with key metrics

---

## 1. Cost Data Input

### Input Data Structure
```python
cost_data = pd.DataFrame({
    'sku': ['TS-114', 'TS-203', 'TS-301'],
    'cost': [209.4, 276.15, 318.6]  # Cost of procuring each article
})
```

### Integration
- Added `cost_data` parameter to `run()` method
- Cost is checked for each article before forecasting
- Used to calculate actual margin: `margin = (price - cost) / price`
- Ensures margin >= target margin before recommending purchase

---

## 2. Historical Data Analysis

### Historical Pricing Analysis
The agent now analyzes:
- **MRP**: Maximum Retail Price (should be same across all stores)
- **Selling Price**: Average selling price (can vary by store)
- **Discount**: Average discount percentage (can vary by store)
- **Margin**: Historical margin percentage (must be >= target margin)

### Method: `_analyze_historical_pricing()`
```python
historical_pricing = {
    'TS-114': {
        'mrp': 499.0,
        'mrp_consistent': True,  # Same across all stores
        'avg_selling_price': 349.0,
        'price_consistency': 'variable',  # Can vary by store
        'avg_discount': 30.1,
        'discount_consistency': 'variable',
        'avg_margin': 40.0
    }
}
```

---

## 3. Margin Target from UI

### UI Input
```html
<label for="margin-target">Margin Target (%)</label>
<input 
    type="number" 
    id="margin-target" 
    name="margin-target" 
    min="0" 
    max="100" 
    step="0.1" 
    value="30"
    placeholder="Enter target margin (0-100%)"
/>
```

### API Integration
```python
# User inputs margin target from UI (e.g., 30%)
margin_target = 0.30  # 30%

# Run forecasting with margin target
results, sensitivity = agent.run(
    comparables, sales_data, inventory_data, price_data,
    price_options, product_attributes, 
    forecast_horizon_days=60,
    cost_data=cost_data,  # DataFrame with sku and cost columns
    margin_target=margin_target,  # From UI input
    max_quantity_per_store=500
)
```

### Behavior
- **Margin >= Target**: Article recommended for purchase
- **Margin < Target**: Article skipped with `skip_margin_risk` recommendation
- Margin is automatically adjusted to meet target if possible

---

## 4. Per-Store Maximum Quantity Constraint

### Constraint
- **Maximum Quantity per Store**: 500 units
- Applied before optimization
- If forecast exceeds 500, quantity is capped at 500

### Implementation
```python
# Apply per-store maximum quantity constraint (500)
if forecast.get('forecast_quantity', 0) > max_quantity_per_store:
    forecast['forecast_quantity'] = max_quantity_per_store
    forecast['reasoning'] += f" | Capped at {max_quantity_per_store} units per store"
```

---

## 5. Output: Pricing Metrics

### Article-Level Metrics
Each article now includes:

```python
{
    'article': 'TS-114',
    'mrp': 499.0,  # Same across all stores
    'mrp_consistent': True,  # Indicates if MRP is same across stores
    'average_selling_price': 349.0,  # Can change or remain same at store level
    'price_consistency': 'variable',  # 'consistent' or 'variable'
    'average_discount': 30.1,  # Can change or remain same at store level
    'discount_consistency': 'variable',  # 'consistent' or 'variable'
    'margin_pct': 0.40,  # Can change or remain same at store level
    'margin_consistency': 'consistent',  # 'consistent' or 'variable'
    'margin_meets_target': True,  # Must be >= target margin
    'target_margin': 0.30,  # Target margin from UI
    'margin_adjusted': False  # Whether margin was adjusted to meet target
}
```

### Consistency Checks
- **MRP**: Should be same across all stores (`mrp_consistent: True`)
- **Selling Price**: Can vary by store (indicated by `price_consistency`)
- **Discount**: Can vary by store (indicated by `discount_consistency`)
- **Margin**: Must be >= target margin, can vary by store (indicated by `margin_consistency`)

---

## 6. Top Banner Summary

### Display as Top Banner
The top banner shows:

```python
{
    'total_unique_skus': 3,  # Total unique SKUs bought
    'total_quantity_bought': 12400,  # Total quantity bought at SKU level
    'total_stores': 25,  # Total stores for which these have been bought
    'total_buy_cost': 2598000.0,  # Total buy cost (quantity X cost for each SKU)
    'total_sales_value': 3245700.0,  # Total sales value for all SKUs
    'average_margin_achieved': 0.40,  # Average margin achieved
    'target_margin': 0.30,  # Target margin from UI
    'margin_vs_target': {
        'achieved': 0.40,
        'target': 0.30,
        'difference': 0.10,
        'meets_target': True,
        'achieved_pct': 40.0,
        'target_pct': 30.0,
        'difference_pct': 10.0
    }
}
```

### Calculation Details

1. **Total Unique SKUs**: Count of distinct articles recommended
2. **Total Quantity Bought**: Sum of quantities across all SKUs
3. **Total Stores**: Unique stores where articles are allocated
4. **Total Buy Cost**: `Σ(quantity × cost)` for each SKU
   - Example: `(2200 × 209.4) + (1800 × 276.15) + (1200 × 318.6) = ₹2,598,000`
5. **Total Sales Value**: Sum of `net_sales_value` for all articles
6. **Average Margin Achieved**: Average of all article margins
7. **Margin vs Target**: Comparison showing if target is met

---

## Updated Method Signatures

### `run()` Method
```python
def run(self, comparables: List[Dict], sales_data: pd.DataFrame, 
       inventory_data: pd.DataFrame, price_data: pd.DataFrame, 
       price_options: List[float], product_attributes: Optional[Dict] = None,
       forecast_horizon_days: int = 60, variance_threshold: Optional[float] = None,
       cost_data: Optional[pd.DataFrame] = None, 
       margin_target: Optional[float] = None,
       max_quantity_per_store: int = 500) -> Tuple[Dict[str, Any], Dict[str, Any]]:
```

### `forecast_store_level()` Method
```python
def forecast_store_level(self, sales_data: pd.DataFrame,
                       inventory_data: pd.DataFrame,
                       price_data: pd.DataFrame,
                       product_attributes: Dict[str, Any],
                       comparables: List[Dict],
                       forecast_horizon_days: int = 60,
                       cost_data: Optional[pd.DataFrame] = None,
                       margin_target: Optional[float] = None,
                       max_quantity_per_store: int = 500) -> Dict[str, Any]:
```

---

## Complete Output Structure

### Recommendations Output
```python
{
    'articles_to_buy': ['TS-114', 'TS-203', 'TS-301'],
    'store_allocations': {...},
    'article_level_metrics': {
        'TS-114': {
            'article': 'TS-114',
            'mrp': 499.0,
            'mrp_consistent': True,
            'average_selling_price': 349.0,
            'price_consistency': 'variable',
            'average_discount': 30.1,
            'discount_consistency': 'variable',
            'margin_pct': 0.40,
            'margin_consistency': 'consistent',
            'margin_meets_target': True,
            'target_margin': 0.30,
            # ... other metrics
        }
    },
    'top_banner': {
        'total_unique_skus': 3,
        'total_quantity_bought': 12400,
        'total_stores': 25,
        'total_buy_cost': 2598000.0,
        'total_sales_value': 3245700.0,
        'average_margin_achieved': 0.40,
        'target_margin': 0.30,
        'margin_vs_target': {...}
    },
    'pricing_analysis': {...}  # Historical pricing analysis
}
```

---

## Usage Example

```python
from agents.demand_forecasting_agent import DemandForecastingAgent
import pandas as pd

# Initialize agent
agent = DemandForecastingAgent(
    llama_client=llama_client,
    enable_hitl=True
)

# Prepare cost data
cost_data = pd.DataFrame({
    'sku': ['TS-114', 'TS-203', 'TS-301'],
    'cost': [209.4, 276.15, 318.6]
})

# User inputs from UI
margin_target = 0.30  # 30% from UI
max_quantity_per_store = 500  # Constraint

# Run forecasting
results, sensitivity = agent.run(
    comparables, sales_data, inventory_data, price_data,
    price_options, product_attributes,
    forecast_horizon_days=60,
    cost_data=cost_data,
    margin_target=margin_target,
    max_quantity_per_store=max_quantity_per_store
)

# Access top banner
top_banner = results['recommendations']['top_banner']
print(f"Total SKUs: {top_banner['total_unique_skus']}")
print(f"Total Quantity: {top_banner['total_quantity_bought']}")
print(f"Total Stores: {top_banner['total_stores']}")
print(f"Total Buy Cost: ₹{top_banner['total_buy_cost']:,.0f}")
print(f"Total Sales Value: ₹{top_banner['total_sales_value']:,.0f}")
print(f"Avg Margin: {top_banner['average_margin_achieved']*100:.1f}%")
print(f"Target Margin: {top_banner['target_margin']*100:.1f}%")
print(f"Meets Target: {top_banner['margin_vs_target']['meets_target']}")

# Access article-level pricing
for article, metrics in results['recommendations']['article_level_metrics'].items():
    print(f"\n{article}:")
    print(f"  MRP: ₹{metrics['mrp']:.2f} (consistent: {metrics['mrp_consistent']})")
    print(f"  Avg Selling Price: ₹{metrics['average_selling_price']:.2f} ({metrics['price_consistency']})")
    print(f"  Avg Discount: {metrics['average_discount']:.1f}% ({metrics['discount_consistency']})")
    print(f"  Margin: {metrics['margin_pct']*100:.1f}% ({metrics['margin_consistency']}, >= target: {metrics['margin_meets_target']})")
```

---

## Validation Rules

1. **Cost Data**: Must have 'sku' and 'cost' columns
2. **Margin Target**: Must be between 0 and 1 (0-100%)
3. **Per-Store Quantity**: Automatically capped at 500
4. **Margin Requirement**: All recommendations must have margin >= target margin
5. **MRP Consistency**: MRP should be same across all stores (flagged if not)

---

## All Capabilities Intact

✅ Model Selection  
✅ Factor Analysis  
✅ Store-Level Forecasting  
✅ Optimization (ROS, STR, Margin)  
✅ Fallback Mechanisms  
✅ Pydantic Validation  
✅ HITL Workflow  
✅ Configurable Variance Threshold  
✅ **NEW**: Cost Data Integration  
✅ **NEW**: Margin Target from UI  
✅ **NEW**: Per-Store Quantity Constraint (500)  
✅ **NEW**: Historical Pricing Analysis  
✅ **NEW**: Pricing Metrics with Consistency Checks  
✅ **NEW**: Top Banner Summary  

---

## UI Integration Checklist

- [ ] **Cost Data Input**: File upload or manual entry for cost per SKU
- [ ] **Margin Target Input**: Number input field (0-100%)
- [ ] **Top Banner Display**: Prominent banner showing all 6 metrics
- [ ] **Article-Level Pricing Display**: Table showing MRP, selling price, discount, margin with consistency indicators
- [ ] **Per-Store Quantity Display**: Show capped quantities if constraint applied

---

## Notes

- Cost data is optional but recommended for accurate margin calculation
- If cost data is not provided, margin is estimated from default margin percentage
- Margin target defaults to `min_margin_pct` if not provided from UI
- Per-store quantity constraint is configurable (default: 500)
- All pricing metrics include consistency indicators (consistent/variable)
- Top banner is automatically generated and included in recommendations

