# Variance Threshold Configuration

## Overview

The variance threshold (previously fixed at 5%) is now configurable via UI input before demand forecasting is triggered. This allows users to customize the auto-approval threshold based on their business requirements.

## Changes Made

### 1. Updated `run()` Method

The `run()` method now accepts an optional `variance_threshold` parameter:

```python
def run(self, comparables: List[Dict], sales_data: pd.DataFrame, 
       inventory_data: pd.DataFrame, price_data: pd.DataFrame, 
       price_options: List[float], product_attributes: Optional[Dict] = None,
       forecast_horizon_days: int = 60, 
       variance_threshold: Optional[float] = None) -> Tuple[Dict[str, Any], Dict[str, Any]]:
```

**Parameters:**
- `variance_threshold`: Optional float (0.05 = 5%, 0.10 = 10%, etc.)
  - If `None`, uses the value set during initialization (default: 0.05)
  - Must be between 0 and 1 (0-100%)
  - Automatically updates the HITL workflow when provided

### 2. New Methods for Variance Threshold Management

#### `set_variance_threshold(variance_threshold: float)`

Set or update the variance threshold before or after initialization.

```python
# Example usage
agent.set_variance_threshold(0.10)  # Set to 10%
```

**Returns:**
```python
{
    'variance_threshold': 0.10,
    'variance_threshold_pct': 10.0,
    'message': 'Variance threshold updated to 10.0%',
    'updated_at': '2025-12-05T12:00:00'
}
```

#### `get_variance_threshold()`

Get the current variance threshold setting.

```python
# Example usage
threshold_info = agent.get_variance_threshold()
```

**Returns:**
```python
{
    'variance_threshold': 0.05,
    'variance_threshold_pct': 5.0,
    'description': 'Auto-approve if variance < 5.0%, flag for approval if >= 5.0%'
}
```

### 3. Automatic HITL Workflow Update

When the variance threshold is set (either via `run()` or `set_variance_threshold()`), the HITL workflow is automatically updated to use the new threshold.

## Usage Examples

### Example 1: Set Variance Threshold Before Running Forecast

```python
from agents.demand_forecasting_agent import DemandForecastingAgent

# Initialize agent
agent = DemandForecastingAgent(
    llama_client=llama_client,
    enable_hitl=True,
    variance_threshold=0.05  # Default 5%
)

# User sets variance threshold from UI (e.g., 10%)
user_variance_threshold = 0.10  # 10%

# Run forecasting with UI-provided variance threshold
results, sensitivity = agent.run(
    comparables, sales_data, inventory_data, price_data,
    price_options, product_attributes, 
    forecast_horizon_days=60,
    variance_threshold=user_variance_threshold  # From UI input
)
```

### Example 2: Set Variance Threshold Using Dedicated Method

```python
# Set variance threshold before running forecast
agent.set_variance_threshold(0.08)  # 8%

# Run forecasting (will use the set threshold)
results, sensitivity = agent.run(
    comparables, sales_data, inventory_data, price_data,
    price_options, product_attributes, 
    forecast_horizon_days=60
)
```

### Example 3: Get Current Threshold

```python
# Get current threshold
threshold_info = agent.get_variance_threshold()
print(f"Current threshold: {threshold_info['variance_threshold_pct']}%")
# Output: Current threshold: 5.0%
```

## UI Integration

### Required UI Component

Add a variance threshold input field in the UI before triggering demand forecasting:

```html
<!-- Example UI Input -->
<label for="variance-threshold">Variance Threshold (%)</label>
<input 
    type="number" 
    id="variance-threshold" 
    name="variance-threshold" 
    min="0" 
    max="100" 
    step="0.1" 
    value="5"
    placeholder="Enter variance threshold (0-100%)"
/>
```

### API Call from UI

```javascript
// Example JavaScript/TypeScript
const varianceThreshold = parseFloat(document.getElementById('variance-threshold').value) / 100;

// Call forecasting API with variance threshold
const response = await fetch('/api/forecast', {
    method: 'POST',
    body: JSON.stringify({
        comparables: comparables,
        sales_data: sales_data,
        inventory_data: inventory_data,
        price_data: price_data,
        price_options: price_options,
        product_attributes: product_attributes,
        forecast_horizon_days: 60,
        variance_threshold: varianceThreshold  // From UI input
    })
});
```

## Behavior

### Auto-Approval Rules

- **Variance < Threshold**: Auto-approve → Re-run allocation algorithm automatically
- **Variance >= Threshold**: Flag for approval → Wait for category head approval

### Examples

| User Input | Threshold | Edit Variance | Status |
|------------|-----------|---------------|--------|
| 5% | 0.05 | 3% | ✅ Auto-Approved |
| 5% | 0.05 | 7% | ⚠️ Needs Approval |
| 10% | 0.10 | 8% | ✅ Auto-Approved |
| 10% | 0.10 | 12% | ⚠️ Needs Approval |
| 3% | 0.03 | 2% | ✅ Auto-Approved |
| 3% | 0.03 | 4% | ⚠️ Needs Approval |

## Validation

The variance threshold is validated to ensure:
- Value is between 0 and 1 (0-100%)
- Raises `ValueError` if invalid

```python
# Invalid examples (will raise ValueError)
agent.set_variance_threshold(-0.05)  # Negative
agent.set_variance_threshold(1.5)    # > 100%
```

## Metadata

The variance threshold is included in the HITL metadata returned by the `run()` method:

```python
{
    'hitl_metadata': {
        'enabled': True,
        'available_stores': [...],
        'store_mappings': {...},
        'variance_threshold': 0.10,
        'variance_threshold_pct': 10.0
    }
}
```

## Backward Compatibility

- If `variance_threshold` is not provided in `run()`, the system uses the default value (0.05 = 5%)
- Existing code without the parameter will continue to work
- The threshold can still be set during initialization

## All Capabilities Intact

✅ Model Selection  
✅ Factor Analysis  
✅ Store-Level Forecasting  
✅ Optimization (ROS, STR, Margin)  
✅ Fallback Mechanisms  
✅ Pydantic Validation  
✅ HITL Workflow  
✅ **NEW**: Configurable Variance Threshold from UI  

