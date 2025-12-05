# User Manual

## Overview
This manual provides end-to-end guidance for using the Intelli-ODM system to make data-driven procurement and allocation decisions for new fashion products.

## Table of Contents
1. [Getting Started](#getting-started)
2. [Data Preparation](#data-preparation)
3. [Running Analysis](#running-analysis)
4. [Interpreting Results](#interpreting-results)
5. [Best Practices](#best-practices)
6. [Troubleshooting](#troubleshooting)
7. [Advanced Usage](#advanced-usage)

## Getting Started

### System Overview
Intelli-ODM helps answer four critical questions for new products:
1. **Should we procure this product?**
2. **How much should we buy?**
3. **Which stores should receive how many units?**
4. **What price will optimize performance?**

### User Workflow
```
Input Data → Product Analysis → Demand Forecast → Allocation Plan → Business Decision
```

### Prerequisites
Before using the system, ensure:
- ✅ System is properly installed and configured
- ✅ Ollama service is running with Llama3 model
- ✅ Historical data files are available
- ✅ Product description is ready

## Data Preparation

### Required Data Files
The system requires four CSV files with historical data:

#### 1. Products File (`products.csv`)
Contains your existing product catalog.

**Required Columns**:
```csv
product_id,vendor_sku,description,category,color,material,size_set
P001,V-TSH-001,"White cotton t-shirt basic crew neck short sleeve",TSHIRT,White,Cotton,"S,M,L,XL"
P002,V-TSH-002,"Navy blue polo shirt with collar short sleeve",POLO,Navy,Cotton,"S,M,L,XL,XXL"
P003,V-DRS-001,"Floral print summer dress sleeveless midi length",DRESS,Floral,Polyester,"XS,S,M,L"
```

**Optional Columns**: `subcategory`, `brand`, `season`, `launch_date`, `price_range`

#### 2. Sales File (`sales.csv`)
Historical sales data by store and product.

**Required Columns**:
```csv
date,store_id,sku,units_sold,revenue
2024-01-01,store_001,P001,15,4485
2024-01-01,store_002,P001,12,3588
2024-01-01,store_003,P002,8,2792
```

**Data Quality Requirements**:
- Date format: YYYY-MM-DD
- At least 6 months of sales history recommended
- Daily or weekly granularity preferred

#### 3. Inventory File (`inventory.csv`)
Current inventory levels by store.

**Required Columns**:
```csv
store_id,sku,on_hand,in_transit
store_001,P001,45,0
store_002,P001,32,20
store_003,P002,67,0
```

#### 4. Pricing File (`pricing.csv`)
Current pricing by store and product.

**Required Columns**:
```csv
store_id,sku,price,markdown_flag
store_001,P001,299,FALSE
store_002,P001,299,FALSE
store_003,P002,349,FALSE
```

### Data Quality Checklist
Before uploading data, verify:
- [ ] No missing required columns
- [ ] Consistent date formats
- [ ] No duplicate records (same date/store/sku)
- [ ] Reasonable value ranges (no negative sales/prices)
- [ ] Store IDs consistent across all files
- [ ] Product IDs consistent between products and sales files

### Sample Data Preparation

#### Excel to CSV Conversion
If your data is in Excel format:

1. **Open Excel file**
2. **Save As → CSV (Comma delimited)**
3. **Ensure UTF-8 encoding** for special characters
4. **Verify column headers match required format**

#### Data Validation Script
Use this script to validate your data before analysis:

```python
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def validate_data_files(file_paths):
    """Validate input data files for common issues."""
    
    errors = []
    warnings = []
    
    # Load files
    try:
        products = pd.read_csv(file_paths['products'])
        sales = pd.read_csv(file_paths['sales'])
        inventory = pd.read_csv(file_paths['inventory'])
        pricing = pd.read_csv(file_paths['pricing'])
    except Exception as e:
        errors.append(f"Error loading files: {e}")
        return {"errors": errors, "warnings": warnings}
    
    # Check required columns
    required_cols = {
        'products': ['product_id', 'description', 'category'],
        'sales': ['date', 'store_id', 'sku', 'units_sold', 'revenue'],
        'inventory': ['store_id', 'sku', 'on_hand'],
        'pricing': ['store_id', 'sku', 'price']
    }
    
    for file_name, df in [('products', products), ('sales', sales), 
                          ('inventory', inventory), ('pricing', pricing)]:
        missing_cols = set(required_cols[file_name]) - set(df.columns)
        if missing_cols:
            errors.append(f"{file_name}.csv missing columns: {missing_cols}")
    
    # Check data quality
    if not errors:  # Only if files loaded successfully
        
        # Date format check
        try:
            sales['date'] = pd.to_datetime(sales['date'])
            min_date = sales['date'].min()
            max_date = sales['date'].max()
            
            if (datetime.now().date() - max_date.date()).days > 30:
                warnings.append("Sales data appears outdated (last sale > 30 days ago)")
                
            if (max_date - min_date).days < 180:
                warnings.append("Limited sales history (< 6 months)")
                
        except Exception:
            errors.append("Invalid date format in sales.csv")
        
        # Check for missing data
        for name, df in [('products', products), ('sales', sales), 
                        ('inventory', inventory), ('pricing', pricing)]:
            missing_pct = df.isnull().sum().sum() / (df.shape[0] * df.shape[1]) * 100
            if missing_pct > 10:
                warnings.append(f"{name}.csv has {missing_pct:.1f}% missing data")
        
        # Check for negative values
        if (sales['units_sold'] < 0).any() or (sales['revenue'] < 0).any():
            warnings.append("Negative sales values found")
            
        if (pricing['price'] <= 0).any():
            errors.append("Invalid prices (zero or negative) found")
        
        # Check consistency
        product_ids = set(products['product_id'])
        sales_skus = set(sales['sku'])
        common_products = product_ids.intersection(sales_skus)
        
        if len(common_products) == 0:
            errors.append("No matching products between products.csv and sales.csv")
        elif len(common_products) / len(product_ids) < 0.5:
            warnings.append("Less than 50% of products have sales history")
    
    return {
        "errors": errors,
        "warnings": warnings,
        "summary": {
            "products_count": len(products) if 'products' in locals() else 0,
            "sales_records": len(sales) if 'sales' in locals() else 0,
            "date_range": f"{min_date.date()} to {max_date.date()}" if 'min_date' in locals() else "Unknown",
            "stores_count": len(sales['store_id'].unique()) if 'sales' in locals() else 0
        }
    }

# Usage example
file_paths = {
    'products': 'data/products.csv',
    'sales': 'data/sales.csv',
    'inventory': 'data/inventory.csv',
    'pricing': 'data/pricing.csv'
}

validation_result = validate_data_files(file_paths)
print("Validation Results:")
print(f"Errors: {validation_result['errors']}")
print(f"Warnings: {validation_result['warnings']}")
print(f"Summary: {validation_result['summary']}")
```

## Running Analysis

### Basic Usage

#### 1. Simple Analysis
```python
from orchestrator import OrchestratorAgent
import ollama

# Initialize system
client = ollama.Client(host='http://localhost:11434')
constraints = {
    "budget": 500000,        # Budget in INR
    "MOQ": 200,             # Minimum order quantity
    "pack_size": 20,        # Pack size for orders
    "lead_time_days": 30    # Vendor lead time
}

orchestrator = OrchestratorAgent(client, constraints)

# Prepare input
input_data = {
    "files": {
        "products": "data/products.csv",
        "sales": "data/sales.csv",
        "inventory": "data/inventory.csv",
        "pricing": "data/pricing.csv"
    },
    "product_description": "White cotton t-shirt with round neck and short sleeves, 180 GSM"
}

# Price points to analyze
price_options = [299, 349, 399]

# Run analysis
result = orchestrator.run_workflow(input_data, price_options)

# Print key results
print("PROCUREMENT RECOMMENDATION")
print("=" * 40)
print(f"Total Quantity: {result['allocation_plan']['total_procurement_qty']} units")
print(f"Expected Revenue: ₹{result['allocation_plan']['expected_revenue']:,.0f}")
print(f"Expected Margin: ₹{result['allocation_plan']['expected_margin']:,.0f}")
print(f"Sell-through Rate: {result['allocation_plan']['expected_sellthrough']:.1%}")

print("\nSTORE ALLOCATION:")
allocations = result['allocation_plan']['store_allocations']
for store, qty in allocations.items():
    print(f"  {store}: {qty} units")

print(f"\nRECOMMENDATION:")
print(result['business_recommendation'])
```

#### 2. Batch Analysis
For analyzing multiple products:

```python
import pandas as pd

# Product list for analysis
products_to_analyze = [
    {
        "description": "Navy blue cotton polo shirt with collar",
        "price_options": [399, 449, 499]
    },
    {
        "description": "Black denim jeans straight fit mid rise",
        "price_options": [999, 1199, 1399]
    },
    {
        "description": "Floral print summer dress sleeveless",
        "price_options": [799, 899, 999]
    }
]

results = []
for i, product in enumerate(products_to_analyze):
    print(f"Analyzing product {i+1}: {product['description']}")
    
    input_data = {
        "files": file_paths,  # Same data files
        "product_description": product["description"]
    }
    
    try:
        result = orchestrator.run_workflow(input_data, product["price_options"])
        
        results.append({
            "product": product["description"],
            "procurement_qty": result['allocation_plan']['total_procurement_qty'],
            "expected_revenue": result['allocation_plan']['expected_revenue'],
            "expected_margin": result['allocation_plan']['expected_margin'],
            "sellthrough_rate": result['allocation_plan']['expected_sellthrough'],
            "status": "Success"
        })
        
    except Exception as e:
        results.append({
            "product": product["description"],
            "status": f"Error: {e}"
        })

# Create summary report
summary_df = pd.DataFrame(results)
print("\nBATCH ANALYSIS SUMMARY:")
print(summary_df.to_string(index=False))
```

### Advanced Configuration

#### 1. Custom Business Constraints
```python
# Advanced constraint configuration
advanced_constraints = {
    "budget": 1000000,
    "MOQ": 200,
    "pack_size": 20,
    "lead_time_days": 30,
    "max_store_capacity": 500,
    "min_store_allocation": 10,
    "safety_stock_factor": 1.2,
    "seasonal_adjustment": True,
    "store_tier_weights": {
        "tier_1": 1.0,
        "tier_2": 0.8,
        "tier_3": 0.6
    }
}

orchestrator = OrchestratorAgent(client, advanced_constraints)
```

#### 2. Seasonal Analysis
```python
# Analyze product for different seasons
seasons = ["Spring", "Summer", "Monsoon", "Winter"]
seasonal_results = {}

for season in seasons:
    # Modify product description to include seasonal context
    seasonal_description = f"{base_description} - {season} collection"
    
    input_data = {
        "files": file_paths,
        "product_description": seasonal_description,
        "season": season  # Additional context
    }
    
    result = orchestrator.run_workflow(input_data, price_options)
    seasonal_results[season] = result['allocation_plan']['expected_revenue']

# Compare seasonal performance
best_season = max(seasonal_results, key=seasonal_results.get)
print(f"Best performing season: {best_season}")
```

#### 3. Price Optimization
```python
# Find optimal price point
price_range = range(299, 500, 25)  # Test prices from 299 to 499 in steps of 25
price_analysis = {}

for price in price_range:
    result = orchestrator.run_workflow(input_data, [price])
    
    price_analysis[price] = {
        "revenue": result['allocation_plan']['expected_revenue'],
        "margin": result['allocation_plan']['expected_margin'],
        "volume": result['allocation_plan']['total_procurement_qty']
    }

# Find revenue-maximizing price
optimal_price = max(price_analysis, key=lambda p: price_analysis[p]['revenue'])
print(f"Optimal price for revenue: ₹{optimal_price}")
```

## Interpreting Results

### Understanding the Output

The system provides comprehensive results in several sections:

#### 1. Product Analysis
```python
product_info = result['product_info']
```

**Key Fields**:
- `extracted_attributes`: Structured product attributes extracted by LLM
- `comparable_products`: Similar products found in knowledge base
- `trend_analysis`: Market trends and seasonal patterns

**Interpretation**:
- **High similarity scores (>0.8)**: Strong comparable products available, higher forecast confidence
- **Low similarity scores (<0.6)**: Limited comparables, higher uncertainty
- **Trend strength**: Positive trends indicate growing category demand

#### 2. Demand Forecast
```python
demand_forecast = result['demand_forecast']
```

**Store-Level Forecast Fields**:
- `mean_demand`: Expected units to sell in forecast period
- `low_ci` / `high_ci`: 10th and 90th percentile confidence bounds
- `confidence`: Model confidence score (0-1)
- `method_used`: Forecasting method applied

**Interpretation Guidelines**:

| Confidence Level | Interpretation | Action |
|-----------------|---------------|---------|
| > 0.8 | High confidence | Proceed with recommended allocation |
| 0.6 - 0.8 | Medium confidence | Consider conservative allocation |
| < 0.6 | Low confidence | Require additional validation |

#### 3. Price Sensitivity Analysis
```python
price_sensitivity = result['price_sensitivity']
```

**Key Metrics**:
- `price_elasticity`: How demand responds to price changes
- `optimal_price`: Revenue-maximizing price point
- `price_scenarios`: Performance at different price points

**Elasticity Interpretation**:
- **Elastic (< -1.0)**: Price-sensitive product, lower prices increase revenue
- **Unit Elastic (≈ -1.0)**: Revenue relatively stable across price range
- **Inelastic (> -1.0)**: Price increases can boost revenue

#### 4. Allocation Plan
```python
allocation_plan = result['allocation_plan']
```

**Key Outputs**:
- `total_procurement_qty`: Total units to procure
- `store_allocations`: Units per store
- `expected_performance`: Revenue, margin, sell-through projections

### Performance Metrics

#### Revenue Projections
- **Conservative**: Use low_ci forecast values
- **Optimistic**: Use high_ci forecast values
- **Expected**: Use mean forecast values

#### Risk Assessment
```python
def assess_risk_level(result):
    """Assess overall risk level of the recommendation."""
    
    confidence = result['demand_forecast']['avg_confidence']
    sellthrough = result['allocation_plan']['expected_sellthrough']
    comparable_similarity = result['product_info']['avg_comparable_similarity']
    
    risk_score = 0
    
    # Confidence risk
    if confidence < 0.6:
        risk_score += 3
    elif confidence < 0.8:
        risk_score += 1
    
    # Sell-through risk
    if sellthrough < 0.6:
        risk_score += 3
    elif sellthrough < 0.75:
        risk_score += 1
    
    # Comparable similarity risk
    if comparable_similarity < 0.6:
        risk_score += 2
    elif comparable_similarity < 0.8:
        risk_score += 1
    
    if risk_score <= 2:
        return "LOW"
    elif risk_score <= 4:
        return "MEDIUM"
    else:
        return "HIGH"

risk_level = assess_risk_level(result)
print(f"Overall Risk Assessment: {risk_level}")
```

### Decision Framework

#### Go/No-Go Decision Matrix

| Factor | Go Conditions | No-Go Conditions |
|--------|---------------|------------------|
| **Comparable Products** | Similarity > 0.7, Count ≥ 3 | Similarity < 0.5, Count < 2 |
| **Forecast Confidence** | > 0.75 | < 0.6 |
| **Expected Sell-through** | > 0.7 | < 0.5 |
| **ROI** | > 25% | < 15% |
| **Trend Direction** | Positive/Stable | Declining |

#### Allocation Strategy Guidelines

**Conservative Allocation** (High Risk):
- Allocate to top-performing stores only
- Reduce quantities by 20-30%
- Monitor performance closely in first 2 weeks

**Aggressive Allocation** (Low Risk):
- Full recommended allocation
- Include tier 2/3 stores
- Higher initial inventory investment

**Balanced Allocation** (Medium Risk):
- 80% of recommended allocation
- Focus on tier 1 and select tier 2 stores
- Phased rollout approach

## Best Practices

### 1. Data Quality Management

#### Regular Data Updates
- Update sales data weekly minimum
- Refresh inventory data daily
- Review pricing data for accuracy monthly

#### Data Quality Monitoring
```python
# Create automated data quality checks
def monitor_data_quality(file_paths):
    """Monitor data quality over time."""
    
    quality_metrics = {}
    
    # Check data freshness
    sales_df = pd.read_csv(file_paths['sales'])
    sales_df['date'] = pd.to_datetime(sales_df['date'])
    days_since_last_sale = (datetime.now() - sales_df['date'].max()).days
    
    quality_metrics['data_freshness_days'] = days_since_last_sale
    quality_metrics['data_freshness_status'] = 'Good' if days_since_last_sale <= 7 else 'Stale'
    
    # Check completeness
    missing_pct = sales_df.isnull().sum().sum() / (sales_df.shape[0] * sales_df.shape[1]) * 100
    quality_metrics['completeness_pct'] = 100 - missing_pct
    
    # Check consistency
    zero_sales_pct = (sales_df['units_sold'] == 0).mean() * 100
    quality_metrics['zero_sales_pct'] = zero_sales_pct
    
    return quality_metrics

# Run quality check
quality = monitor_data_quality(file_paths)
print("Data Quality Report:")
for metric, value in quality.items():
    print(f"  {metric}: {value}")
```

### 2. Model Performance Monitoring

#### Forecast Accuracy Tracking
```python
def track_forecast_accuracy(predictions, actuals):
    """Track forecast accuracy over time."""
    
    # Calculate accuracy metrics
    mape = np.mean(np.abs((actuals - predictions) / actuals)) * 100
    mse = np.mean((actuals - predictions) ** 2)
    
    accuracy_score = max(0, 100 - mape)
    
    return {
        'mape': mape,
        'mse': mse,
        'accuracy_score': accuracy_score
    }
```

#### Model Retraining Schedule
- **Weekly**: Update comparable product weights
- **Monthly**: Retrain demand forecasting models
- **Quarterly**: Review and update business constraints
- **Annually**: Comprehensive model evaluation and updates

### 3. Business Integration

#### Integration with Buying Workflow
1. **Pre-Season Planning**: Use for new product evaluation
2. **Mid-Season Adjustments**: Validate reorder decisions
3. **Post-Season Analysis**: Evaluate forecast accuracy

#### Stakeholder Communication
```python
def generate_executive_summary(result):
    """Generate executive summary for stakeholders."""
    
    summary = f"""
    EXECUTIVE SUMMARY - {result['product_info']['description'][:50]}...
    
    RECOMMENDATION: {'PROCEED' if result['allocation_plan']['expected_sellthrough'] > 0.7 else 'REVIEW'}
    
    KEY METRICS:
    • Procurement Quantity: {result['allocation_plan']['total_procurement_qty']:,} units
    • Investment Required: ₹{result['allocation_plan']['total_procurement_qty'] * 200:,}  # Estimated cost
    • Expected Revenue: ₹{result['allocation_plan']['expected_revenue']:,.0f}
    • Expected Margin: ₹{result['allocation_plan']['expected_margin']:,.0f}
    • ROI: {(result['allocation_plan']['expected_margin'] / (result['allocation_plan']['total_procurement_qty'] * 200)) * 100:.1f}%
    • Sell-through: {result['allocation_plan']['expected_sellthrough']:.1%}
    
    RISK LEVEL: {assess_risk_level(result)}
    
    TOP PERFORMING STORES:
    """
    
    # Add top 5 stores by allocation
    allocations = result['allocation_plan']['store_allocations']
    top_stores = sorted(allocations.items(), key=lambda x: x[1], reverse=True)[:5]
    
    for store, qty in top_stores:
        summary += f"    • {store}: {qty} units\n"
    
    return summary

# Generate summary
exec_summary = generate_executive_summary(result)
print(exec_summary)
```

## Troubleshooting

### Common Issues and Solutions

#### 1. Low Forecast Confidence
**Symptoms**: Confidence scores < 0.6, wide confidence intervals

**Causes**:
- Limited comparable products
- Poor data quality
- High variability in historical sales

**Solutions**:
```python
# Check comparable product quality
comparables = result['product_info']['comparable_products']
if len(comparables) < 3 or max(c['similarity_score'] for c in comparables) < 0.7:
    print("Action: Expand product description or adjust similarity thresholds")
    
# Check data quality
validation = validate_data_files(file_paths)
if len(validation['warnings']) > 0:
    print("Action: Address data quality issues:", validation['warnings'])
```

#### 2. Unrealistic Allocations
**Symptoms**: Very high or very low allocation quantities

**Causes**:
- Incorrect business constraints
- Outliers in historical data
- Seasonal misalignment

**Solutions**:
```python
# Review constraints
if result['allocation_plan']['total_procurement_qty'] > constraints['budget'] / 200:  # Estimated unit cost
    print("Action: Increase budget constraint or reduce price expectations")

# Check for outliers
sales_df = pd.read_csv(file_paths['sales'])
outliers = sales_df[sales_df['units_sold'] > sales_df['units_sold'].quantile(0.99)]
if len(outliers) > 0:
    print(f"Action: Review {len(outliers)} outlier sales records")
```

#### 3. Price Sensitivity Issues
**Symptoms**: Unrealistic price elasticity, flat demand curves

**Causes**:
- Limited price variation in historical data
- Category-specific pricing patterns not captured

**Solutions**:
- Include more diverse price points in analysis
- Use category-specific elasticity benchmarks
- Validate with A/B testing data if available

### Performance Issues

#### Slow Processing
- **Check system resources**: Monitor CPU, memory usage
- **Optimize data size**: Filter to relevant products/timeframes
- **Use parallel processing**: For batch analyses

#### Memory Issues
```python
# Monitor memory usage during processing
import psutil

def monitor_memory():
    memory = psutil.virtual_memory()
    print(f"Memory usage: {memory.percent}%")
    if memory.percent > 80:
        print("Warning: High memory usage")

# Call during processing
monitor_memory()
```

### Getting Help

#### Debug Mode
```python
# Enable detailed logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Run with debug information
result = orchestrator.run_workflow(input_data, price_options, debug=True)
```

#### Support Information
When reporting issues, include:
1. System configuration (OS, Python version)
2. Input data sample (anonymized)
3. Error messages or unexpected outputs
4. Steps to reproduce the issue

## Advanced Usage

### Custom Forecasting Models

#### Adding Custom Model
```python
class CustomForecastingModel:
    def forecast(self, historical_data, product_attributes):
        """Custom forecasting logic."""
        # Your custom implementation
        pass
        
# Register with forecasting agent
forecasting_agent.register_custom_model('custom', CustomForecastingModel())
```

### API Integration

#### REST API Wrapper
```python
from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/analyze', methods=['POST'])
def analyze_product():
    data = request.json
    
    try:
        result = orchestrator.run_workflow(
            data['input_data'], 
            data['price_options']
        )
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
```

### Scheduled Analysis

#### Automated Daily Reports
```python
import schedule
import time

def daily_analysis():
    """Run daily analysis for new products."""
    # Load new products from queue
    new_products = get_new_products_queue()
    
    for product in new_products:
        try:
            result = orchestrator.run_workflow(
                product['input_data'], 
                product['price_options']
            )
            
            # Send results to stakeholders
            send_analysis_report(result)
            
        except Exception as e:
            log_error(f"Failed to analyze {product['description']}: {e}")

# Schedule daily at 9 AM
schedule.every().day.at("09:00").do(daily_analysis)

while True:
    schedule.run_pending()
    time.sleep(60)
```

This user manual provides comprehensive guidance for effectively using the Intelli-ODM system to make informed procurement and allocation decisions.