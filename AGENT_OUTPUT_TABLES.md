# Demand Forecasting Agent - Output in Tabular Form

This document presents the complete output structure of the Demand Forecasting Agent in tabular format, keeping all capabilities intact.

## 1. Model Selection Results

| Field | Description | Example Value | Data Type |
|-------|-------------|---------------|-----------|
| `selected_model` | Selected forecasting model | `'hybrid'` | String |
| `model_params.model_type` | Type of model | `'hybrid'` | String |
| `model_params.components` | Model components used | `['time_series', 'decay', 'analogy']` | List |
| `model_scores.time_series.score` | Time series model score | `0.85` | Float (0-1) |
| `model_scores.decay_model.score` | Decay model score | `0.72` | Float (0-1) |
| `model_scores.analogy_based.score` | Analogy model score | `0.68` | Float (0-1) |
| `model_scores.hybrid.score` | Hybrid model score | `0.88` | Float (0-1) |
| `data_characteristics.has_time_series` | Has time series data | `True` | Boolean |
| `data_characteristics.time_series_length` | Time series length (days) | `90` | Integer |
| `data_characteristics.has_comparables` | Has comparable products | `True` | Boolean |
| `data_characteristics.num_comparables` | Number of comparables | `5` | Integer |
| `data_characteristics.decay_pattern` | Decay pattern detected | `'moderate_decay'` | String |
| `data_characteristics.seasonality_detected` | Seasonality detected | `False` | Boolean |
| `data_characteristics.store_count` | Number of stores | `25` | Integer |
| `data_characteristics.sku_count` | Number of SKUs | `150` | Integer |
| `fallback_used` | Fallback method used | `False` | Boolean |

---

## 2. Factor Analysis Results

### 2.1 Correlation Analysis

| Field | Description | Example Value | Data Type |
|-------|-------------|---------------|-----------|
| `correlation_matrix.price` | Correlation with price | `0.45` | Float (-1 to 1) |
| `correlation_matrix.price_bucket_encoded` | Correlation with price bucket | `0.38` | Float (-1 to 1) |
| `top_correlations.price` | Top correlation: price | `0.45` | Float |
| `top_correlations.store_avg_sales` | Top correlation: store avg sales | `0.32` | Float |
| `strong_correlations` | Correlations > 0.5 | `{'price': 0.52}` | Dict |

### 2.2 PCA Results

| Field | Description | Example Value | Data Type |
|-------|-------------|---------------|-----------|
| `pca_results.n_components` | Number of PCA components | `5` | Integer |
| `pca_results.explained_variance_ratio[0]` | First component variance | `0.35` | Float |
| `pca_results.explained_variance_ratio[1]` | Second component variance | `0.22` | Float |
| `pca_results.cumulative_variance[0]` | Cumulative variance (1st) | `0.35` | Float |
| `pca_results.cumulative_variance[4]` | Cumulative variance (5th) | `0.95` | Float |

### 2.3 Price Bucket Analysis

| Price Bucket | Total Units | Avg Units | Transaction Count | Total Revenue | Avg Price | Price Per Unit |
|--------------|-------------|-----------|-------------------|---------------|-----------|----------------|
| `0-200` | `500` | `8.5` | `59` | `₹75,000` | `₹150` | `₹17.65` |
| `200-300` | `1,500` | `12.5` | `120` | `₹375,000` | `₹250` | `₹20.00` |
| `300-400` | `2,200` | `15.2` | `145` | `₹770,000` | `₹350` | `₹23.03` |
| `400-500` | `1,800` | `13.8` | `130` | `₹810,000` | `₹450` | `₹32.61` |
| `500-1000` | `900` | `10.5` | `86` | `₹675,000` | `₹750` | `₹71.43` |
| `1000+` | `300` | `7.2` | `42` | `₹450,000` | `₹1,500` | `₹208.33` |

### 2.4 Factor Importance

| Rank | Factor | Importance Score | Percentage |
|------|--------|------------------|------------|
| 1 | `price` | `0.25` | 25.0% |
| 2 | `store_avg_sales` | `0.18` | 18.0% |
| 3 | `price_bucket_encoded` | `0.15` | 15.0% |
| 4 | `day_of_week` | `0.12` | 12.0% |
| 5 | `month` | `0.10` | 10.0% |

---

## 3. Store-Level Forecasts

| Store ID | Article | Forecast Qty | Confidence Low | Confidence High | Recommendation | Expected ST | Expected ROS | Margin % | Optimization Score | Method Used |
|----------|---------|--------------|----------------|-----------------|----------------|-------------|--------------|----------|---------------------|------------|
| `store_001` | `TS-114` | `220` | `165` | `275` | `buy` | `75.2%` | `3.67/day` | `40.0%` | `0.82` | `time_series` |
| `store_001` | `TS-203` | `190` | `142` | `238` | `buy` | `73.5%` | `3.15/day` | `38.5%` | `0.78` | `hybrid` |
| `store_002` | `TS-114` | `200` | `150` | `250` | `buy` | `74.8%` | `3.33/day` | `40.0%` | `0.80` | `time_series` |
| `store_002` | `TS-203` | `180` | `135` | `225` | `buy_cautious` | `72.0%` | `3.00/day` | `37.0%` | `0.72` | `decay_model` |
| `store_003` | `TS-114` | `150` | `112` | `188` | `buy` | `76.0%` | `2.50/day` | `41.0%` | `0.75` | `analogy_based` |

**Legend:**
- **Recommendation**: `buy`, `buy_cautious`, `skip`, `skip_margin_risk`
- **Expected ST**: Expected Sell-Through Rate (percentage)
- **Expected ROS**: Expected Rate of Sale (units per day)
- **Method Used**: `time_series`, `decay_model`, `analogy_based`, `hybrid`, `fallback_lr_kmeans`

---

## 4. Aggregated Forecast Summary

| Metric | Value | Unit |
|--------|-------|------|
| `total_quantity` | `12,400` | units |
| `num_articles` | `3` | articles |
| `stores_with_recommendations` | `25` | stores |
| `total_stores` | `25` | stores |
| `total_expected_units_sold` | `9,300` | units |
| `avg_sell_through_rate` | `75.0%` | percentage |
| `avg_margin_pct` | `40.0%` | percentage |
| `total_revenue` | `₹3,245,700` | currency |
| `total_margin_value` | `₹1,298,280` | currency |
| `avg_optimization_score` | `0.78` | score (0-1) |

---

## 5. Store Universe Validation

| Field | Value | Status |
|-------|-------|--------|
| `total_stores` | `25` | - |
| `universe_of_stores` | `30` | - |
| `validation.valid` | `True` | ✅ Valid |
| `validation.message` | `Total stores (25) is within universe (30).` | - |
| `validation.excess_stores` | `0` | - |

**Validation Rules:**
- ✅ **Valid**: `total_stores ≤ universe_of_stores`
- ⚠️ **Warning**: `total_stores > universe_of_stores` (excess stores identified)

---

## 6. Article-Level Metrics (Detailed)

| Article | Style Code | Color | Segment | Family | Class | Brick | Store Exposure | MRP | Avg Selling Price | Avg Discount | Margin % | ROS | STR | Net Sales Value | Total Qty | Expected Units Sold |
|---------|------------|-------|---------|--------|-------|-------|----------------|-----|-------------------|--------------|----------|-----|-----|-----------------|-----------|---------------------|
| `TS-114` | `TSH-001` | `White` | `Casual` | `T-Shirts` | `Basic` | `Men's Apparel` | `15` | `₹499` | `₹349` | `30.1%` | `40.0%` | `3.67/day` | `75.2%` | `₹576,300` | `2,200` | `1,654` |
| `TS-203` | `TSH-002` | `Black` | `Casual` | `T-Shirts` | `Premium` | `Men's Apparel` | `12` | `₹599` | `₹449` | `25.0%` | `38.5%` | `3.15/day` | `73.5%` | `₹485,200` | `1,800` | `1,323` |
| `TS-301` | `TSH-003` | `Blue` | `Sports` | `T-Shirts` | `Active` | `Men's Apparel` | `10` | `₹699` | `₹549` | `21.5%` | `42.0%` | `2.50/day` | `78.0%` | `₹428,220` | `1,200` | `936` |

**Column Descriptions:**
- **Article**: SKU/Article identifier
- **Style Code**: Product style code
- **Color**: Product color
- **Segment**: Product segment (Casual, Sports, Formal, etc.)
- **Family**: Product family (T-Shirts, Jeans, etc.)
- **Class**: Product class (Basic, Premium, Active, etc.)
- **Brick**: Product category/brick
- **Store Exposure**: Number of stores where article is allocated
- **MRP**: Maximum Retail Price
- **Avg Selling Price**: Average selling price across stores
- **Avg Discount**: Average discount percentage
- **Margin %**: Average margin percentage
- **ROS**: Rate of Sale (units per day) = Total Units Sold / Total Days
- **STR**: Sell-Through Rate (percentage) = Expected Units Sold / Total Inventory
- **Net Sales Value**: Expected Units Sold × Average Selling Price
- **Total Qty**: Total quantity to procure
- **Expected Units Sold**: Expected units to be sold

---

## 7. Priority Stores

| Rank | Store ID | Average Optimization Score | Total Quantity | Status |
|------|----------|---------------------------|----------------|--------|
| 1 | `store_001` | `0.85` | `410` | ⭐ Top Priority |
| 2 | `store_005` | `0.82` | `380` | ⭐ High Priority |
| 3 | `store_012` | `0.80` | `350` | ⭐ High Priority |
| 4 | `store_003` | `0.78` | `330` | ✅ Priority |
| 5 | `store_008` | `0.75` | `310` | ✅ Priority |
| 6 | `store_015` | `0.72` | `290` | ✅ Priority |
| 7 | `store_020` | `0.70` | `270` | ✅ Priority |
| 8 | `store_007` | `0.68` | `250` | ✅ Priority |
| 9 | `store_011` | `0.65` | `230` | ✅ Priority |
| 10 | `store_018` | `0.62` | `210` | ✅ Priority |

---

## 8. Risk Assessment

| Risk Category | Articles Affected | Count | Action Required |
|---------------|-------------------|-------|-----------------|
| **Margin Risk** | `TS-501`, `TS-402` | `2` | ⚠️ Consider price adjustments or skip procurement |
| **Low Sell-Through Risk** | `TS-305` | `1` | ⚠️ Consider reducing quantities |
| **High Confidence** | `TS-114`, `TS-203`, `TS-301` | `3` | ✅ Safe to procure |

---

## 9. Expected Metrics Summary

| Metric | Value | Target | Status |
|--------|-------|--------|--------|
| **Total Expected Units Sold** | `9,300` | - | - |
| **Average Sell-Through Rate** | `75.0%` | `75.0%` | ✅ Met Target |
| **Average Margin %** | `40.0%` | `≥25.0%` | ✅ Above Minimum |
| **Total Revenue** | `₹3,245,700` | - | - |
| **Total Margin Value** | `₹1,298,280` | - | - |
| **Average Optimization Score** | `0.78` | `≥0.70` | ✅ Good Performance |

---

## 10. Store Allocations (Detailed)

| Article | Store ID | Quantity | Expected ST | Expected ROS | Margin % | Optimization Score | Recommendation |
|---------|----------|----------|--------------|--------------|----------|---------------------|----------------|
| `TS-114` | `store_001` | `220` | `75.2%` | `3.67/day` | `40.0%` | `0.82` | `buy` |
| `TS-114` | `store_002` | `200` | `74.8%` | `3.33/day` | `40.0%` | `0.80` | `buy` |
| `TS-114` | `store_003` | `150` | `76.0%` | `2.50/day` | `41.0%` | `0.75` | `buy` |
| `TS-203` | `store_001` | `190` | `73.5%` | `3.15/day` | `38.5%` | `0.78` | `buy` |
| `TS-203` | `store_002` | `180` | `72.0%` | `3.00/day` | `37.0%` | `0.72` | `buy_cautious` |
| `TS-301` | `store_001` | `140` | `78.0%` | `2.33/day` | `42.0%` | `0.76` | `buy` |

---

## 11. Optimization Summary (Text Output)

```
Optimized recommendation: Procure 3 articles with total quantity of 12,400 units across 25 stores (universe: 30).

Expected performance:
  - Sell-through rate: 75.0% (target: 75.0%)
  - Average margin: 40.0% (min: 25.0%)
  - Expected revenue: ₹3,245,700
  - Expected margin value: ₹1,298,280
  - Optimization score: 0.78/1.0

Article-Level Metrics:

  Article: TS-114
    Style Code: TSH-001, Color: White, Segment: Casual
    Family: T-Shirts, Class: Basic, Brick: Men's Apparel
    MRP: ₹499.00, Avg Selling Price: ₹349.00, Avg Discount: 30.1%
    Margin: 40.0%, Store Exposure: 15, ROS: 3.67 units/day, STR: 75.2%
    Net Sales Value: ₹576,300

  Article: TS-203
    Style Code: TSH-002, Color: Black, Segment: Casual
    Family: T-Shirts, Class: Premium, Brick: Men's Apparel
    MRP: ₹599.00, Avg Selling Price: ₹449.00, Avg Discount: 25.0%
    Margin: 38.5%, Store Exposure: 12, ROS: 3.15 units/day, STR: 73.5%
    Net Sales Value: ₹485,200

  Article: TS-301
    Style Code: TSH-003, Color: Blue, Segment: Sports
    Family: T-Shirts, Class: Active, Brick: Men's Apparel
    MRP: ₹699.00, Avg Selling Price: ₹549.00, Avg Discount: 21.5%
    Margin: 42.0%, Store Exposure: 10, ROS: 2.50 units/day, STR: 78.0%
    Net Sales Value: ₹428,220

⚠️  Margin risk detected for 1 articles. Consider price adjustments or skip procurement.
```

---

## 12. Sensitivity Analysis Results

| Price Threshold | Below Threshold Avg Sales | Above Threshold Avg Sales | Sensitivity Ratio | Interpretation |
|-----------------|---------------------------|---------------------------|-------------------|-----------------|
| `₹250` | `15.2` units | `10.8` units | `1.41` | High sensitivity |
| `₹350` | `13.5` units | `11.2` units | `1.21` | Moderate sensitivity |
| `₹450` | `12.0` units | `10.5` units | `1.14` | Low sensitivity |

---

## Output Structure Summary

The agent returns a tuple: `(final_results, sensitivity_analysis)`

### Final Results Structure:
```
final_results = {
    'model_selection': {...},           # Table 1
    'factor_analysis': {...},           # Table 2
    'forecast_results': {
        'store_level_forecasts': {...}, # Table 3
        'aggregated_forecast': {...},  # Table 4
        'recommendations': {
            'total_stores': 25,         # Table 5
            'universe_of_stores': 30,   # Table 5
            'store_universe_validation': {...}, # Table 5
            'article_level_metrics': {...},     # Table 6
            'priority_stores': [...],           # Table 7
            'risk_assessment': {...},          # Table 8
            'expected_metrics': {...},         # Table 9
            'store_allocations': {...},        # Table 10
            'optimization_summary': '...'      # Table 11
        }
    },
    'validation_messages': [...],
    'fallback_used': False
}
```

### Sensitivity Analysis Structure:
```
sensitivity_analysis = {
    'price_sensitivity': {...},        # Table 12
    'attr_material_sensitivity': {...},
    'attr_color_sensitivity': {...}
}
```

---

## Key Formulas

| Metric | Formula | Example |
|--------|---------|---------|
| **ROS (Rate of Sale)** | `Total Units Sold / Total Days in Period` | `1,100 units / 30 days = 36.67 units/day` |
| **STR (Sell-Through Rate)** | `Expected Units Sold / Total Inventory` | `1,654 / 2,200 = 75.2%` |
| **Average Discount** | `((MRP - Avg Selling Price) / MRP) × 100` | `((499 - 349) / 499) × 100 = 30.1%` |
| **Net Sales Value** | `Expected Units Sold × Average Selling Price` | `1,654 × ₹349 = ₹576,300` |
| **Margin %** | `(Selling Price - Cost) / Selling Price` | `(349 - 209.4) / 349 = 40.0%` |

---

## Notes

1. **All numeric values** are floats (quantities, percentages, scores)
2. **Percentages** are represented as decimals in code (0.75 = 75%), but displayed as percentages in output
3. **Recommendations** can be: `'buy'`, `'buy_cautious'`, `'skip'`, `'skip_margin_risk'`
4. **Optimization scores** range from 0.0 to 1.0
5. **Store universe validation** ensures `total_stores ≤ universe_of_stores`
6. **Article details** (style_code, color, segment, family, class, brick) come from `attribute_analogy_agent` and `data_ingestion_agent`
7. **All capabilities remain intact** with enhanced reporting and validation

