# Demand Forecasting & Allocation Engine - Web UI

## Overview

A modern web interface for uploading historical sales data and new articles data to run the demand forecasting and allocation engine.

## Features

### 1. Data Upload
- **Historical Sales Data** (Required): Upload CSV/Excel with store_id, sku, units_sold, date
- **Inventory Data** (Optional): Upload with store_id, sku, on_hand
- **Price Data** (Optional): Upload with store_id, sku, price, mrp
- **Cost Data** (Optional): Upload with sku, cost (procurement cost)
- **New Articles Data** (Required): Upload with sku, style_code, color, segment, family, class, brick

### 2. Parameter Configuration
- **Margin Target**: Target margin percentage (0-100%)
- **Variance Threshold**: Auto-approval threshold (0-100%)
- **Forecast Horizon**: Number of days to forecast (default: 60)
- **Max Quantity per Store**: Maximum allocation per store (default: 500)
- **Universe of Stores**: Total stores in organization

### 3. Store Mapping (HITL)
- Add new store IDs with reference to existing stores
- View all store mappings
- Remove mappings

### 4. Results Display

#### Top Banner Summary
- Total unique SKUs bought
- Total quantity bought at SKU level
- Total stores
- Total buy cost
- Total sales value
- Average Margin achieved vs Target Margin

#### Tabbed Results
- **Recommendations**: Summary and optimization details
- **Article Metrics**: MRP, selling price, discount, margin, ROS, STR, net sales value
- **Store Allocations**: Store-level allocations with editable quantities
- **Factor Analysis**: Correlation, PCA, factor importance
- **Audit Trail**: Complete action history

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run the application:
```bash
python app.py
```

3. Open browser:
```
http://localhost:5000
```

## File Formats

### Historical Sales Data (CSV)
```csv
store_id,sku,units_sold,date,revenue
store_001,TS-114,25,2024-01-15,8750
store_002,TS-114,30,2024-01-15,10500
```

### Inventory Data (CSV)
```csv
store_id,sku,on_hand,in_transit
store_001,TS-114,150,50
store_002,TS-114,200,0
```

### Price Data (CSV)
```csv
store_id,sku,price,mrp,markdown_flag
store_001,TS-114,349,499,False
store_002,TS-114,349,499,False
```

### Cost Data (CSV)
```csv
sku,cost
TS-114,209.4
TS-203,276.15
TS-301,318.6
```

### New Articles Data (CSV)
```csv
sku,style_code,color,segment,family,class,brick
TS-114,TSH-001,White,Casual,T-Shirts,Basic,Men's Apparel
TS-203,TSH-002,Black,Casual,T-Shirts,Premium,Men's Apparel
```

## Usage Workflow

1. **Upload Data**: Upload required and optional data files
2. **Configure Parameters**: Set margin target, variance threshold, and other parameters
3. **Store Mapping** (Optional): Add new store mappings if needed
4. **Run Forecast**: Click "Run Demand Forecast" button
5. **Review Results**: 
   - Check top banner summary
   - Review article-level metrics
   - Examine store-level allocations
   - Analyze factor analysis results
6. **Edit Allocations** (HITL): Edit quantities at store level if needed
7. **Approve/Export**: Approve recommendations and export results

## API Endpoints

### POST /api/upload
Upload data files

**Request**: Multipart form data with file fields

**Response**:
```json
{
    "success": true,
    "message": "Files uploaded successfully",
    "files": {
        "sales_data": "uploads/sales_20241205_120000.csv",
        ...
    }
}
```

### POST /api/forecast
Run demand forecasting

**Request**:
```json
{
    "file_paths": {
        "sales_data": "uploads/sales_20241205_120000.csv",
        "inventory_data": "uploads/inventory_20241205_120000.csv",
        "price_data": "uploads/price_20241205_120000.csv",
        "cost_data": "uploads/cost_20241205_120000.csv",
        "new_articles_data": "uploads/new_articles_20241205_120000.csv"
    },
    "margin_target": 30,
    "variance_threshold": 5,
    "forecast_horizon_days": 60,
    "max_quantity_per_store": 500,
    "universe_of_stores": 30,
    "price_options": [200, 300, 400, 500, 600, 700, 800, 900, 1000],
    "store_mappings": [
        {
            "new_store_id": "store_026",
            "reference_store_id": "store_001"
        }
    ]
}
```

**Response**:
```json
{
    "success": true,
    "results": {
        "model_selection": {...},
        "factor_analysis": {...},
        "recommendations": {...},
        "top_banner": {...},
        "article_level_metrics": {...},
        "store_allocations": {...}
    }
}
```

## UI Components

### File Upload
- Drag-and-drop or click to upload
- File validation
- Progress indicators

### Parameter Inputs
- Number inputs with validation
- Tooltips and help text
- Real-time validation

### Results Display
- Responsive tables
- Interactive charts (future enhancement)
- Export functionality (future enhancement)

### HITL Workflow
- Store mapping interface
- Quantity editing
- Approval workflow
- Audit trail viewer

## Future Enhancements

- [ ] Real-time data validation feedback
- [ ] Interactive charts and visualizations
- [ ] Export results to Excel/PDF
- [ ] Save/load forecast configurations
- [ ] Comparison of multiple forecast scenarios
- [ ] User authentication and role-based access
- [ ] Email notifications for approvals
- [ ] Integration with procurement systems

## Troubleshooting

### File Upload Issues
- Ensure files are in CSV or Excel format
- Check file size (max 16MB)
- Verify required columns are present

### Forecast Errors
- Check data validation messages
- Ensure required files are uploaded
- Verify parameter ranges (0-100% for percentages)

### Display Issues
- Clear browser cache
- Check browser console for errors
- Ensure JavaScript is enabled

## Notes

- Files are stored in `uploads/` directory
- LLM client needs to be configured in `app.py`
- For production, use proper authentication and security measures
- Consider using a production WSGI server (e.g., Gunicorn)

