"""
FastAPI Backend for Demand Forecasting & Allocation Engine
"""

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, BackgroundTasks, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse, Response, StreamingResponse
from typing import Optional, List, Dict, Any
import pandas as pd
import numpy as np
import os
import json
from datetime import datetime, timedelta
from pathlib import Path
import uvicorn
import io
import asyncio
import random

# Import agents and utilities
import sys
from pathlib import Path

# Add parent directory to path
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))

from utils.ollama_client import OllamaClient
from utils.audit_logger import AuditLogger, LogStatus
from agents.data_ingestion_agent import DataIngestionAgent
from agents.attribute_analogy_agent import AttributeAnalogyAgent
from agents.demand_forecasting_agent import DemandForecastingAgent
from shared_knowledge_base import SharedKnowledgeBase

app = FastAPI(title="Demand Forecasting & Allocation Engine", version="1.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173"],  # React dev servers
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)
BACKUP_DIR = Path("backups")
BACKUP_DIR.mkdir(exist_ok=True)
FORECAST_RESULTS_DIR = Path("forecast-results")
FORECAST_RESULTS_DIR.mkdir(exist_ok=True)
ALLOWED_EXTENSIONS = {'.csv', '.xlsx', '.xls'}

# Initialize components
ollama_client = OllamaClient(base_url="http://localhost:11434", model="llama3:8b")
audit_logger = AuditLogger(log_dir="audit-logs")
kb = SharedKnowledgeBase()
data_handler = DataIngestionAgent(ollama_client=ollama_client, audit_logger=audit_logger)

# Global state
current_run_id: Optional[str] = None

def convert_numpy_types(obj):
    """Convert numpy types to native Python types for JSON serialization"""
    import numpy as np
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif pd.isna(obj):
        return None
    return obj

def read_data_file(filepath: str) -> pd.DataFrame:
    """Read CSV or Excel file"""
    if filepath.endswith('.csv'):
        return pd.read_csv(filepath)
    elif filepath.endswith(('.xlsx', '.xls')):
        return pd.read_excel(filepath)
    else:
        raise ValueError(f"Unsupported file format: {filepath}")

@app.on_event("startup")
async def startup_event():
    """Initialize on startup"""
    # Check Ollama connection
    if not ollama_client.check_connection():
        print("⚠️  Warning: Ollama is not running or not accessible at http://localhost:11434")
        print("   Please ensure Ollama is running with llama3:8b model")
    else:
        print("✅ Ollama connection successful")
        models = ollama_client.get_available_models()
        print(f"   Available models: {models}")

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Demand Forecasting & Allocation Engine API",
        "version": "1.0.0",
        "status": "running"
    }

@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    ollama_status = ollama_client.check_connection()
    return {
        "status": "healthy",
        "ollama_connected": ollama_status,
        "ollama_models": ollama_client.get_available_models() if ollama_status else []
    }

@app.post("/api/upload")
async def upload_files(
    sales_data: Optional[UploadFile] = File(None),
    inventory_data: Optional[UploadFile] = File(None),
    price_data: Optional[UploadFile] = File(None),
    cost_data: Optional[UploadFile] = File(None),
    new_articles_data: Optional[UploadFile] = File(None)
):
    """Handle file uploads"""
    try:
        uploaded_files = {}
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        files_to_process = [
            ('sales_data', sales_data),
            ('inventory_data', inventory_data),
            ('price_data', price_data),
            ('cost_data', cost_data),
            ('new_articles_data', new_articles_data)
        ]
        
        for field_name, file in files_to_process:
            if file and file.filename:
                # Check file extension
                ext = Path(file.filename).suffix.lower()
                if ext not in ALLOWED_EXTENSIONS:
                    raise HTTPException(
                        status_code=400,
                        detail=f"File {file.filename} has unsupported extension. Allowed: {ALLOWED_EXTENSIONS}"
                    )
                
                # Save file
                filename = f"{field_name}_{timestamp}{ext}"
                filepath = UPLOAD_DIR / filename
                
                with open(filepath, "wb") as f:
                    content = await file.read()
                    f.write(content)
                
                uploaded_files[field_name] = str(filepath)
        
        # Log upload
        audit_logger.log_agent_operation(
            agent_name="DataIngestionAgent",
            description=f"Files uploaded: {list(uploaded_files.keys())}",
            status=LogStatus.SUCCESS,
            inputs={"files": list(uploaded_files.keys())},
            outputs={"file_paths": uploaded_files}
        )
        
        return {
            "success": True,
            "message": "Files uploaded successfully",
            "files": uploaded_files
        }
    
    except Exception as e:
        audit_logger.log_agent_operation(
            agent_name="DataIngestionAgent",
            description="File upload failed",
            status=LogStatus.FAIL,
            error=str(e)
        )
        raise HTTPException(status_code=500, detail=str(e))

# Template and Sample Data Endpoints
SAMPLE_DATA_DIR = Path("examples/sample_input")

# Template CSV headers for each data type
TEMPLATE_HEADERS = {
    "sales_data": "date,store_id,sku,units_sold,revenue,city,avgdiscount,pricebucket",
    "inventory_data": "store_id,sku,on_hand,in_transit",
    "price_data": "store_id,sku,price,markdown_flag",
    "cost_data": "sku,cost,currency",
    "new_articles_data": "product_id,vendor_sku,description,category,color,material,size_set,brick,class,segment,family,brand"
}

# Map frontend keys to sample file names
SAMPLE_FILE_MAP = {
    "sales_data": "sales.csv",
    "inventory_data": "inventory.csv",
    "price_data": "price.csv",
    "cost_data": "cost.csv",  # May not exist, handle gracefully
    "new_articles_data": "products.csv"
}

@app.get("/api/templates/{data_type}")
async def download_template(data_type: str):
    """Download CSV template for a specific data type"""
    if data_type not in TEMPLATE_HEADERS:
        raise HTTPException(status_code=404, detail=f"Template not found for {data_type}")
    
    headers = TEMPLATE_HEADERS[data_type]
    csv_content = headers + "\n"  # Just headers, no data
    
    return Response(
        content=csv_content,
        media_type="text/csv",
        headers={
            "Content-Disposition": f"attachment; filename={data_type}_template.csv"
        }
    )

@app.get("/api/sample/{data_type}")
async def get_sample_data(data_type: str):
    """Get sample CSV data for a specific data type"""
    if data_type not in SAMPLE_FILE_MAP:
        raise HTTPException(status_code=404, detail=f"Sample data not found for {data_type}")
    
    sample_file = SAMPLE_DATA_DIR / SAMPLE_FILE_MAP[data_type]
    
    if not sample_file.exists():
        # If sample file doesn't exist, return template with headers
        if data_type in TEMPLATE_HEADERS:
            headers = TEMPLATE_HEADERS[data_type]
            csv_content = headers + "\n"
            return Response(
                content=csv_content,
                media_type="text/csv",
                headers={
                    "Content-Disposition": f"attachment; filename={data_type}_sample.csv"
                }
            )
        raise HTTPException(status_code=404, detail=f"Sample file not found: {sample_file}")
    
    return FileResponse(
        path=str(sample_file),
        media_type="text/csv",
        filename=f"{data_type}_sample.csv"
    )

async def run_forecast_internal(
    file_paths: Optional[str],
    sales_data_file: Optional[UploadFile],
    inventory_data_file: Optional[UploadFile],
    price_data_file: Optional[UploadFile],
    cost_data_file: Optional[UploadFile],
    new_articles_data_file: Optional[UploadFile],
    margin_target: float,
    variance_threshold: float,
    forecast_horizon_days: int,
    max_quantity_per_store: int,
    universe_of_stores: Optional[int],
    store_mappings: Optional[str],
    log_queue: Optional[asyncio.Queue] = None
):
    """Run demand forecasting and allocation"""
    global current_run_id
    
    try:
        # Start new run if not already started
        if not current_run_id:
            current_run_id = audit_logger.start_run()
        
        # Load data from files - prioritize direct uploads over file paths
        sales_data = None
        inventory_data = None
        price_data = None
        cost_data = None
        new_articles_data = None
        
        # Helper function to read from UploadFile or FileWrapper (offload blocking I/O to thread pool)
        async def read_upload_file(upload_file) -> Optional[pd.DataFrame]:
            if upload_file:
                # Handle both UploadFile and FileWrapper
                if hasattr(upload_file, 'read') and callable(upload_file.read):
                    # It's an UploadFile or FileWrapper with async read
                    if hasattr(upload_file, 'filename') and upload_file.filename:
                        content = await upload_file.read() if asyncio.iscoroutinefunction(upload_file.read) else upload_file.read()
                        filename = upload_file.filename
                    else:
                        return None
                elif hasattr(upload_file, 'content') and hasattr(upload_file, 'filename'):
                    # It's a FileWrapper with direct content access
                    content = upload_file.content
                    filename = upload_file.filename
                else:
                    return None
                
                # Determine file extension
                ext = Path(filename).suffix.lower()
                
                # Offload blocking pandas operations to thread pool
                loop = asyncio.get_event_loop()
                if ext == '.csv':
                    return await loop.run_in_executor(None, pd.read_csv, io.BytesIO(content))
                elif ext in ['.xlsx', '.xls']:
                    return await loop.run_in_executor(None, pd.read_excel, io.BytesIO(content))
            return None
        
        # Try direct file uploads first
        if sales_data_file:
            sales_data = await read_upload_file(sales_data_file)
        if inventory_data_file:
            inventory_data = await read_upload_file(inventory_data_file)
        if price_data_file:
            price_data = await read_upload_file(price_data_file)
        if cost_data_file:
            cost_data = await read_upload_file(cost_data_file)
        if new_articles_data_file:
            new_articles_data = await read_upload_file(new_articles_data_file)
        
        # Fallback to file paths if no direct uploads
        if file_paths:
            try:
                file_paths_dict = json.loads(file_paths)
                
                # Offload blocking file reads to thread pool
                loop = asyncio.get_event_loop()
                if not sales_data and 'sales_data' in file_paths_dict:
                    sales_data = await loop.run_in_executor(None, read_data_file, file_paths_dict['sales_data'])
                
                if not inventory_data and 'inventory_data' in file_paths_dict:
                    inventory_data = await loop.run_in_executor(None, read_data_file, file_paths_dict['inventory_data'])
                
                if not price_data and 'price_data' in file_paths_dict:
                    price_data = await loop.run_in_executor(None, read_data_file, file_paths_dict['price_data'])
                
                if not cost_data and 'cost_data' in file_paths_dict:
                    cost_data = await loop.run_in_executor(None, read_data_file, file_paths_dict['cost_data'])
                
                if not new_articles_data and 'new_articles_data' in file_paths_dict:
                    new_articles_data = await loop.run_in_executor(None, read_data_file, file_paths_dict['new_articles_data'])
            except (json.JSONDecodeError, KeyError) as e:
                # If file_paths is invalid JSON or missing keys, continue without it
                print(f"Warning: Could not parse file_paths: {e}")
        
        # If still no data, try loading from sample files (offload blocking file I/O to thread pool)
        loop = asyncio.get_event_loop()
        if not sales_data and (SAMPLE_DATA_DIR / "sales.csv").exists():
            sales_data = await loop.run_in_executor(None, read_data_file, str(SAMPLE_DATA_DIR / "sales.csv"))
        if not inventory_data and (SAMPLE_DATA_DIR / "inventory.csv").exists():
            inventory_data = await loop.run_in_executor(None, read_data_file, str(SAMPLE_DATA_DIR / "inventory.csv"))
        if not price_data and (SAMPLE_DATA_DIR / "price.csv").exists():
            price_data = await loop.run_in_executor(None, read_data_file, str(SAMPLE_DATA_DIR / "price.csv"))
        if not new_articles_data and (SAMPLE_DATA_DIR / "products.csv").exists():
            new_articles_data = await loop.run_in_executor(None, read_data_file, str(SAMPLE_DATA_DIR / "products.csv"))
        
        # Validate that new_articles_data is provided (required)
        if new_articles_data is None or new_articles_data.empty:
            raise HTTPException(
                status_code=400,
                detail="New Articles Data is required. Please upload a file or use sample data."
            )
        
        # Convert percentages to decimals
        margin_target_decimal = margin_target / 100
        variance_threshold_decimal = variance_threshold / 100
        
        # Extract product attributes from new articles data
        # Use vendor_sku as the key (new articles use vendor_sku, existing data uses sku)
        product_attributes = {}
        if new_articles_data is not None and not new_articles_data.empty:
            for _, row in new_articles_data.iterrows():
                # Try vendor_sku first (for new articles), then sku as fallback
                sku = row.get('vendor_sku', '') or row.get('sku', '')
                if sku:
                    product_attributes[sku] = {
                        'style_code': row.get('style_code', row.get('product_id', 'N/A')),
                        'color': row.get('color', 'N/A'),
                        'segment': row.get('segment', 'N/A'),
                        'family': row.get('family', 'N/A'),
                        'class': row.get('class', 'N/A'),
                        'brick': row.get('brick', 'N/A'),
                        'category': row.get('category', 'N/A'),
                        'material': row.get('material', 'N/A'),
                        'brand': row.get('brand', 'N/A')
                    }
        
        print(f"\n[WORKFLOW] Extracted {len(product_attributes)} product attributes from new articles")
        
        # Initialize agents
        attribute_agent = AttributeAnalogyAgent(ollama_client, kb, audit_logger)
        demand_agent = DemandForecastingAgent(
            ollama_client=ollama_client,
            default_margin_pct=0.40,
            target_sell_through_pct=0.75,
            min_margin_pct=0.25,
            use_llm=True,
            universe_of_stores=universe_of_stores,
            enable_hitl=True,
            variance_threshold=variance_threshold_decimal,
            audit_logger=audit_logger
        )
        
        # Get comparables from attribute analogy agent (offload blocking operations)
        comparables = []
        if new_articles_data is not None and not new_articles_data.empty:
            loop = asyncio.get_event_loop()
            print(f"\n[WORKFLOW] Running Attribute Analogy Agent for {len(new_articles_data)} new articles...")
            # For each new article, find comparables
            for idx, (_, row) in enumerate(new_articles_data.iterrows()):
                # Build product description from attributes
                desc_parts = []
                for attr in ['description', 'category', 'color', 'segment', 'family', 'class', 'brick', 'material', 'brand']:
                    if attr in row and pd.notna(row[attr]):
                        desc_parts.append(f"{attr}: {row[attr]}")
                
                # Use vendor_sku or sku as fallback
                sku = row.get('vendor_sku', '') or row.get('sku', '') or row.get('product_id', '')
                product_description = ", ".join(desc_parts) if desc_parts else sku
                
                print(f"[WORKFLOW] Processing article {idx+1}/{len(new_articles_data)}: {sku}")
                
                # Find comparables - offload blocking agent.run() to thread pool
                # Capture product_description in closure properly
                def run_attribute_agent(desc=product_description):
                    try:
                        comps, _ = attribute_agent.run(desc, sales_data)
                        print(f"[WORKFLOW] Found {len(comps)} comparables for: {desc[:50]}...")
                        return comps
                    except Exception as e:
                        print(f"[WORKFLOW] Error in attribute agent for {desc[:50]}: {e}")
                        import traceback
                        traceback.print_exc()
                        return []
                
                comps = await loop.run_in_executor(None, run_attribute_agent, product_description)
                comparables.extend(comps)
            
            print(f"[WORKFLOW] Attribute Analogy Agent complete. Total comparables: {len(comparables)}")
        
        # Extract new article SKUs (vendor_sku) for forecasting
        new_article_skus = []
        if new_articles_data is not None and not new_articles_data.empty:
            # Get vendor_sku or sku from new articles
            if 'vendor_sku' in new_articles_data.columns:
                new_article_skus = new_articles_data['vendor_sku'].dropna().unique().tolist()
            elif 'sku' in new_articles_data.columns:
                new_article_skus = new_articles_data['sku'].dropna().unique().tolist()
            elif 'product_id' in new_articles_data.columns:
                new_article_skus = new_articles_data['product_id'].dropna().unique().tolist()
        
        print(f"\n[WORKFLOW] New articles to forecast: {new_article_skus}")
        
        # Run demand forecasting - offload blocking agent.run() to thread pool
        print(f"\n[WORKFLOW] Starting Demand Forecasting Agent...")
        print(f"[WORKFLOW] Inputs: {len(comparables)} comparables, {len(product_attributes)} product attributes")
        print(f"[WORKFLOW] New articles (vendor_sku): {len(new_article_skus)}")
        print(f"[WORKFLOW] Sales data: {len(sales_data) if sales_data is not None and not sales_data.empty else 0} rows")
        print(f"[WORKFLOW] Inventory data: {len(inventory_data) if inventory_data is not None and not inventory_data.empty else 0} rows")
        print(f"[WORKFLOW] Price data: {len(price_data) if price_data is not None and not price_data.empty else 0} rows")
        
        # Add new articles to product_attributes if not already there (for forecasting)
        # The demand agent needs to know which articles to forecast for
        # We'll pass the new article SKUs through product_attributes
        if new_article_skus:
            for sku in new_article_skus:
                if sku not in product_attributes:
                    # Find the row for this SKU
                    matching_row = new_articles_data[
                        (new_articles_data.get('vendor_sku', '') == sku) |
                        (new_articles_data.get('sku', '') == sku) |
                        (new_articles_data.get('product_id', '') == sku)
                    ]
                    if not matching_row.empty:
                        row = matching_row.iloc[0]
                        product_attributes[sku] = {
                            'style_code': row.get('style_code', row.get('product_id', 'N/A')),
                            'color': row.get('color', 'N/A'),
                            'segment': row.get('segment', 'N/A'),
                            'family': row.get('family', 'N/A'),
                            'class': row.get('class', 'N/A'),
                            'brick': row.get('brick', 'N/A'),
                            'category': row.get('category', 'N/A'),
                            'material': row.get('material', 'N/A'),
                            'brand': row.get('brand', 'N/A')
                        }
        
        loop = asyncio.get_event_loop()
        def run_demand_agent():
            try:
                print("[WORKFLOW] Demand Forecasting Agent.run() called")
                print(f"[WORKFLOW] Product attributes keys (articles to forecast): {list(product_attributes.keys())}")
                result = demand_agent.run(
                    comparables=comparables,
                    sales_data=sales_data,
                    inventory_data=inventory_data,
                    price_data=price_data,
                    price_options=[200, 300, 400, 500, 600, 700, 800, 900, 1000],
                    product_attributes=product_attributes,  # This should include new article SKUs
                    forecast_horizon_days=forecast_horizon_days,
                    variance_threshold=variance_threshold_decimal,
                    cost_data=cost_data,
                    margin_target=margin_target_decimal,
                    max_quantity_per_store=max_quantity_per_store
                )
                print(f"[WORKFLOW] Demand Forecasting Agent completed. Result type: {type(result)}")
                if isinstance(result, tuple):
                    print(f"[WORKFLOW] Result tuple length: {len(result)}")
                return result
            except Exception as e:
                print(f"[WORKFLOW] ERROR in Demand Forecasting Agent: {e}")
                import traceback
                traceback.print_exc()
                raise
        
        results, sensitivity = await loop.run_in_executor(None, run_demand_agent)
        print(f"[WORKFLOW] Demand Forecasting Agent execution complete")
        
        # Debug: Log results structure
        print(f"\n[DEBUG] Results keys: {results.keys() if isinstance(results, dict) else 'Not a dict'}")
        if isinstance(results, dict):
            if 'recommendations' in results:
                recs = results.get('recommendations', {})
                print(f"[DEBUG] Recommendations keys: {recs.keys() if isinstance(recs, dict) else 'Not a dict'}")
                print(f"[DEBUG] Articles to buy: {recs.get('articles_to_buy', [])}")
                print(f"[DEBUG] Total quantity: {recs.get('total_procurement_quantity', 0)}")
            if 'forecast_results' in results:
                fr = results.get('forecast_results', {})
                print(f"[DEBUG] Forecast results keys: {fr.keys() if isinstance(fr, dict) else 'Not a dict'}")
                if 'store_level_forecasts' in fr:
                    store_forecasts = fr.get('store_level_forecasts', {})
                    print(f"[DEBUG] Number of stores with forecasts: {len(store_forecasts)}")
                    total_articles = sum(len(articles) for articles in store_forecasts.values())
                    print(f"[DEBUG] Total article forecasts: {total_articles}")
                    # Count recommendations
                    buy_count = 0
                    skip_count = 0
                    for store_id, articles in store_forecasts.items():
                        for article, forecast in articles.items():
                            rec = forecast.get('recommendation', 'skip')
                            if rec in ['buy', 'buy_cautious']:
                                buy_count += 1
                            else:
                                skip_count += 1
                    print(f"[DEBUG] Buy recommendations: {buy_count}, Skip recommendations: {skip_count}")
        
        # Format results for UI
        formatted_results = format_results_for_ui(results, sensitivity)
        formatted_results['run_id'] = current_run_id
        
        # Debug: Log formatted results
        print(f"\n[DEBUG] Formatted recommendations: {formatted_results.get('recommendations', {})}")
        print(f"[DEBUG] Formatted top_banner: {formatted_results.get('top_banner', {})}")
        
        # Store forecast results with timestamp
        result_file = FORECAST_RESULTS_DIR / f"{current_run_id}_results.json"
        result_data = {
            "run_id": current_run_id,
            "timestamp": datetime.now().isoformat(),
            "parameters": {
                "margin_target": margin_target_decimal,
                "variance_threshold": variance_threshold_decimal,
                "forecast_horizon_days": forecast_horizon_days,
                "max_quantity_per_store": max_quantity_per_store,
                "universe_of_stores": universe_of_stores
            },
            "results": formatted_results
        }
        with open(result_file, 'w') as f:
            json.dump(result_data, f, indent=2, default=str)
        
        # Log successful completion
        audit_logger.log_agent_operation(
            agent_name="DemandForecastingAgent",
            description="Forecast completed successfully",
            status=LogStatus.SUCCESS,
            outputs={"run_id": current_run_id, "articles_count": len(formatted_results.get('recommendations', {}).get('articles_to_buy', []))}
        )
        
        return {
            "success": True,
            "results": formatted_results,
            "run_id": current_run_id
        }
    
    except Exception as e:
        import traceback
        error_msg = str(e)
        traceback_str = traceback.format_exc()
        
        audit_logger.log_agent_operation(
            agent_name="DemandForecastingAgent",
            description="Forecast failed",
            status=LogStatus.FAIL,
            error=error_msg
        )
        
        raise HTTPException(
            status_code=500,
            detail={
                "error": error_msg,
                "traceback": traceback_str
            }
        )

@app.post("/api/forecast")
async def run_forecast(
    background_tasks: BackgroundTasks,
    file_paths: Optional[str] = Form(None),
    sales_data_file: Optional[UploadFile] = File(None),
    inventory_data_file: Optional[UploadFile] = File(None),
    price_data_file: Optional[UploadFile] = File(None),
    cost_data_file: Optional[UploadFile] = File(None),
    new_articles_data_file: Optional[UploadFile] = File(None),
    margin_target: float = Form(30.0),
    variance_threshold: float = Form(5.0),
    forecast_horizon_days: int = Form(60),
    max_quantity_per_store: int = Form(500),
    universe_of_stores: Optional[int] = Form(None),
    store_mappings: Optional[str] = Form(None)
):
    """Start forecast and return run_id immediately, then run forecast in background"""
    global current_run_id
    
    # Start run immediately to get run_id (before forecast starts)
    current_run_id = audit_logger.start_run()
    run_id = current_run_id
    
    # Store file content before background task (since UploadFile can only be read once)
    file_contents = {}
    
    async def read_file_content(upload_file: Optional[UploadFile], key: str):
        if upload_file and upload_file.filename:
            content = await upload_file.read()
            file_contents[key] = {
                'content': content,
                'filename': upload_file.filename
            }
    
    # Read all upload files before background task
    await read_file_content(sales_data_file, 'sales')
    await read_file_content(inventory_data_file, 'inventory')
    await read_file_content(price_data_file, 'price')
    await read_file_content(cost_data_file, 'cost')
    await read_file_content(new_articles_data_file, 'new_articles')
    
    # Create FileWrapper class for background execution
    class FileWrapper:
        def __init__(self, content, filename):
            self.content = content
            self.filename = filename
            self._file = io.BytesIO(content)
        
        async def read(self):
            return self.content
        
        @property
        def file(self):
            return self._file
    
    # Create file wrapper objects
    sales_file_obj = FileWrapper(file_contents['sales']['content'], file_contents['sales']['filename']) if 'sales' in file_contents else None
    inventory_file_obj = FileWrapper(file_contents['inventory']['content'], file_contents['inventory']['filename']) if 'inventory' in file_contents else None
    price_file_obj = FileWrapper(file_contents['price']['content'], file_contents['price']['filename']) if 'price' in file_contents else None
    cost_file_obj = FileWrapper(file_contents['cost']['content'], file_contents['cost']['filename']) if 'cost' in file_contents else None
    new_articles_file_obj = FileWrapper(file_contents['new_articles']['content'], file_contents['new_articles']['filename']) if 'new_articles' in file_contents else None
    
    # Run forecast in background task (truly async, non-blocking)
    async def run_forecast_background():
        """Run forecast in background - this is async and won't block other requests"""
        try:
            # This runs asynchronously and won't block the main event loop
            await run_forecast_internal(
                file_paths, sales_file_obj, inventory_file_obj, price_file_obj,
                cost_file_obj, new_articles_file_obj, margin_target, variance_threshold,
                forecast_horizon_days, max_quantity_per_store, universe_of_stores, store_mappings
            )
        except Exception as e:
            print(f"Error in background forecast: {e}")
            import traceback
            traceback.print_exc()
            # Log error to audit logger
            audit_logger.log_agent_operation(
                agent_name="DemandForecastingAgent",
                description="Forecast failed in background task",
                status=LogStatus.FAIL,
                error=str(e)
            )
    
    # Add to background tasks - this is non-blocking and returns immediately
    background_tasks.add_task(run_forecast_background)
    
    # Return run_id immediately so frontend can start streaming
    # The forecast will continue running in the background
    return {
        "success": True,
        "run_id": run_id,
        "message": "Forecast started in background. Use SSE endpoint to stream logs in real-time."
    }


def format_results_for_ui(results: Dict[str, Any], sensitivity: Dict[str, Any]) -> Dict[str, Any]:
    """Format results for UI display"""
    # Handle both direct recommendations and nested in forecast_results
    recommendations = results.get('recommendations', {})
    if not recommendations and 'forecast_results' in results:
        recommendations = results.get('forecast_results', {}).get('recommendations', {})
    
    # If still empty, try to extract from store_level_forecasts
    if not recommendations and 'forecast_results' in results:
        forecast_results = results.get('forecast_results', {})
        store_forecasts = forecast_results.get('store_level_forecasts', {})
        
        # If we have store forecasts but no recommendations, create basic structure
        if store_forecasts:
            # Extract articles with 'buy' or 'buy_cautious' recommendations
            articles_to_buy = []
            store_allocations = {}
            total_quantity = 0
            
            for store_id, articles_dict in store_forecasts.items():
                for article, forecast in articles_dict.items():
                    recommendation = forecast.get('recommendation', 'skip')
                    if recommendation in ['buy', 'buy_cautious']:
                        qty = forecast.get('forecast_quantity', 0) or forecast.get('optimized_quantity', 0)
                        if qty > 0:
                            if article not in articles_to_buy:
                                articles_to_buy.append(article)
                                store_allocations[article] = {}
                            store_allocations[article][store_id] = {
                                'quantity': qty,
                                'expected_sell_through': forecast.get('expected_sell_through_rate', 0),
                                'expected_rate_of_sale': forecast.get('expected_rate_of_sale', 0),
                                'margin_pct': forecast.get('margin_pct', 0),
                                'recommendation': recommendation
                            }
                            total_quantity += qty
            
            # Create recommendations structure
            if articles_to_buy:
                recommendations = {
                    'articles_to_buy': articles_to_buy,
                    'store_allocations': store_allocations,
                    'total_procurement_quantity': total_quantity,
                    'total_stores': len(set(
                        store_id 
                        for stores in store_allocations.values() 
                        for store_id in stores.keys()
                    )),
                    'top_banner': {
                        'total_unique_skus': len(articles_to_buy),
                        'total_quantity_bought': total_quantity,
                        'total_stores': len(set(
                            store_id 
                            for stores in store_allocations.values() 
                            for store_id in stores.keys()
                        ))
                    },
                    'article_level_metrics': {},
                    'store_allocations': store_allocations
                }
    
    formatted = {
        'model_selection': results.get('model_selection', {}),
        'factor_analysis': results.get('factor_analysis', {}),
        'recommendations': recommendations,
        'top_banner': recommendations.get('top_banner', {}) if recommendations else {},
        'article_level_metrics': recommendations.get('article_level_metrics', {}) if recommendations else {},
        'store_allocations': recommendations.get('store_allocations', {}) if recommendations else {},
        'sensitivity_analysis': sensitivity,
        'validation_messages': results.get('validation_messages', []),
        'fallback_used': results.get('fallback_used', False),
        'hitl_metadata': results.get('hitl_metadata', {}),
        'forecast_results': results.get('forecast_results', {})  # Include full forecast results for debugging
    }
    
    return formatted

@app.get("/api/audit-logs/{run_id}")
async def get_audit_logs(run_id: str):
    """Get audit logs for a specific run"""
    # Offload file I/O to thread pool to avoid blocking
    loop = asyncio.get_event_loop()
    logs = await loop.run_in_executor(None, audit_logger.get_run_logs, run_id)
    return {
        "success": True,
        "run_id": run_id,
        "logs": logs,
        "count": len(logs)
    }

@app.get("/api/audit-logs")
async def get_all_runs():
    """Get list of all runs"""
    runs = audit_logger.get_all_runs()
    return {
        "success": True,
        "runs": runs,
        "count": len(runs)
    }

@app.get("/api/audit-logs/agent/{agent_name}")
async def get_agent_logs(agent_name: str, run_id: Optional[str] = None):
    """Get logs for a specific agent"""
    logs = audit_logger.get_agent_logs(agent_name, run_id)
    return {
        "success": True,
        "agent_name": agent_name,
        "logs": logs,
        "count": len(logs)
    }

@app.get("/api/forecast-runs")
async def get_forecast_runs():
    """Get list of all forecast runs with metadata"""
    runs = []
    run_ids_seen = set()
    
    # First, get runs from result files
    for result_file in sorted(FORECAST_RESULTS_DIR.glob("*_results.json"), reverse=True):
        try:
            with open(result_file, 'r') as f:
                result_data = json.load(f)
                run_id = result_data.get('run_id', result_file.stem.replace('_results', ''))
                timestamp = result_data.get('timestamp', '')
                
                # Get log count
                logs = audit_logger.get_run_logs(run_id)
                
                runs.append({
                    "run_id": run_id,
                    "timestamp": timestamp,
                    "log_count": len(logs),
                    "parameters": result_data.get('parameters', {}),
                    "summary": {
                        "total_skus": result_data.get('results', {}).get('top_banner', {}).get('total_unique_skus', 0),
                        "total_quantity": result_data.get('results', {}).get('top_banner', {}).get('total_quantity_bought', 0),
                        "total_stores": result_data.get('results', {}).get('top_banner', {}).get('total_stores', 0)
                    }
                })
                run_ids_seen.add(run_id)
        except Exception as e:
            print(f"Error reading result file {result_file}: {e}")
            continue
    
    # Also include runs that have logs but no results file (incomplete runs)
    all_run_ids = audit_logger.get_all_runs()
    for run_id in all_run_ids:
        if run_id not in run_ids_seen:
            logs = audit_logger.get_run_logs(run_id)
            if logs:
                # Get timestamp from first log
                timestamp = logs[0].get('date_time', '') if logs else ''
                runs.append({
                    "run_id": run_id,
                    "timestamp": timestamp,
                    "log_count": len(logs),
                    "parameters": {},
                    "summary": {
                        "total_skus": 0,
                        "total_quantity": 0,
                        "total_stores": 0
                    }
                })
    
    # Sort by timestamp (most recent first)
    runs.sort(key=lambda x: x.get('timestamp', ''), reverse=True)
    
    return {
        "success": True,
        "runs": runs,
        "count": len(runs)
    }

@app.get("/api/forecast-runs/{run_id}")
async def get_forecast_run(run_id: str):
    """Get forecast results and logs for a specific run"""
    # Offload file I/O to thread pool to avoid blocking
    loop = asyncio.get_event_loop()
    
    def load_run_data():
        result_file = FORECAST_RESULTS_DIR / f"{run_id}_results.json"
        
        # Get audit logs for this run (always available)
        logs = audit_logger.get_run_logs(run_id)
        
        # If no logs exist, the run doesn't exist
        if not logs:
            return None
        
        # Try to get results if file exists
        results = {}
        parameters = {}
        timestamp = ""
        
        if result_file.exists():
            try:
                with open(result_file, 'r') as f:
                    result_data = json.load(f)
                    results = result_data.get('results', {})
                    parameters = result_data.get('parameters', {})
                    timestamp = result_data.get('timestamp', '')
            except Exception as e:
                print(f"Error reading result file {result_file}: {e}")
                # Continue without results
        
        # If no timestamp from results, get it from first log
        if not timestamp and logs:
            timestamp = logs[0].get('date_time', '')
        
        return {
            "success": True,
            "run_id": run_id,
            "timestamp": timestamp,
            "parameters": parameters,
            "results": results,
            "logs": logs,
            "log_count": len(logs)
        }
    
    # Run in thread pool to avoid blocking
    result = await loop.run_in_executor(None, load_run_data)
    
    if result is None:
        raise HTTPException(status_code=404, detail=f"Forecast run not found: {run_id}")
    
    return result

@app.get("/api/audit-logs/{run_id}/stream")
async def stream_audit_logs(run_id: str):
    """Stream audit logs for a run using Server-Sent Events"""
    async def event_generator():
        last_count = 0
        max_iterations = 1200  # 10 minutes max (1200 * 0.5s)
        iteration = 0
        completed = False
        
        # Send initial connection message
        yield f"data: {json.dumps({'type': 'connected', 'run_id': run_id})}\n\n"
        
        while iteration < max_iterations and not completed:
            try:
                logs = audit_logger.get_run_logs(run_id)
                
                # If we have new logs, send them immediately
                if len(logs) > last_count:
                    # Send new logs one by one
                    for log in logs[last_count:]:
                        log_json = json.dumps({'type': 'log', 'data': log}, default=str)
                        yield f"data: {log_json}\n\n"
                    last_count = len(logs)
                
                # Check if run is complete
                if logs:
                    last_log = logs[-1]
                    desc = last_log.get('description', '').lower()
                    status = last_log.get('status', '')
                    
                    # Check for completion indicators
                    if ('forecast completed' in desc or 'forecast failed' in desc or
                        'completed successfully' in desc or 
                        (status in ['Success', 'Fail'] and 'forecast' in desc)):
                        completed = True
                        yield f"data: {json.dumps({'type': 'complete', 'run_id': run_id})}\n\n"
                        break
                
                await asyncio.sleep(0.3)  # Poll every 300ms for faster updates
                iteration += 1
            except Exception as e:
                import traceback
                error_msg = f"Error in SSE stream: {str(e)}\n{traceback.format_exc()}"
                yield f"data: {json.dumps({'type': 'error', 'error': error_msg})}\n\n"
                break
        
        # Send final batch if any remaining
        if not completed:
            try:
                final_logs = audit_logger.get_run_logs(run_id)
                for log in final_logs[last_count:]:
                    log_json = json.dumps({'type': 'log', 'data': log}, default=str)
                    yield f"data: {log_json}\n\n"
            except:
                pass
    
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )

# Data Management Endpoints
@app.get("/api/data/{data_type}")
async def get_data(data_type: str):
    """Get current sample data as JSON"""
    if data_type not in SAMPLE_FILE_MAP:
        raise HTTPException(status_code=404, detail=f"Data type not found: {data_type}")
    
    sample_file = SAMPLE_DATA_DIR / SAMPLE_FILE_MAP[data_type]
    
    if not sample_file.exists():
        raise HTTPException(status_code=404, detail=f"Sample file not found: {sample_file}")
    
    try:
        df = read_data_file(str(sample_file))
        # Convert to records for JSON response
        data_records = df.to_dict('records')
        # Convert numpy types to native Python types
        data_records = convert_numpy_types(data_records)
        
        return {
            "success": True,
            "data_type": data_type,
            "data": data_records,
            "columns": list(df.columns),
            "row_count": len(df)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading data: {str(e)}")

@app.post("/api/data/{data_type}/upload")
async def upload_and_validate_data(
    data_type: str,
    file: UploadFile = File(...)
):
    """Upload, validate, and replace sample data with backup"""
    if data_type not in SAMPLE_FILE_MAP:
        raise HTTPException(status_code=400, detail=f"Invalid data type: {data_type}")
    
    # Check file extension
    ext = Path(file.filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file format. Allowed: {ALLOWED_EXTENSIONS}"
        )
    
    try:
        # Save uploaded file temporarily
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        temp_file = UPLOAD_DIR / f"{data_type}_{timestamp}{ext}"
        
        with open(temp_file, "wb") as f:
            content = await file.read()
            f.write(content)
        
        # Initialize data ingestion agent with LLM
        data_agent = DataIngestionAgent(ollama_client=ollama_client, audit_logger=audit_logger)
        
        # Validate data
        validation_result = data_agent.validate(str(temp_file), data_type)
        
        if not validation_result['valid']:
            # Clean up temp file
            temp_file.unlink()
            return {
                "success": False,
                "validation": validation_result,
                "message": "Data validation failed"
            }
        
        # Backup current sample data
        sample_file = SAMPLE_DATA_DIR / SAMPLE_FILE_MAP[data_type]
        backup_version = None
        if sample_file.exists():
            backup_version = datetime.now().strftime('%Y%m%d_%H%M%S')
            backup_file = BACKUP_DIR / f"{data_type}_v{backup_version}{sample_file.suffix}"
            backup_file.parent.mkdir(exist_ok=True)
            
            # Copy current file to backup
            import shutil
            shutil.copy2(sample_file, backup_file)
            
            # Save backup metadata
            backup_metadata = {
                "data_type": data_type,
                "version": backup_version,
                "timestamp": datetime.now().isoformat(),
                "original_filename": sample_file.name,
                "backup_filename": backup_file.name,
                "row_count": len(read_data_file(str(sample_file)))
            }
            
            metadata_file = BACKUP_DIR / f"{data_type}_v{backup_version}_metadata.json"
            with open(metadata_file, 'w') as f:
                json.dump(backup_metadata, f, indent=2)
        
        # Replace sample data with new data
        import shutil
        shutil.copy2(temp_file, sample_file)
        
        # Clean up temp file
        temp_file.unlink()
        
        # Get new data for response
        df = read_data_file(str(sample_file))
        features = data_agent.feature_engineering(df, data_type)
        
        # Convert DataFrame to records and convert numpy types
        data_records = df.to_dict('records')
        data_records = convert_numpy_types(data_records)
        features = convert_numpy_types(features)
        
        # Log operation
        if audit_logger:
            audit_logger.log_agent_operation(
                agent_name="DataIngestionAgent",
                description=f"Data uploaded and validated for {data_type}",
                status=LogStatus.SUCCESS,
                inputs={"data_type": data_type, "filename": file.filename},
                outputs={"validation": validation_result, "features": features}
            )
        
        return {
            "success": True,
            "validation": validation_result,
            "features": features,
            "data": data_records,
            "columns": list(df.columns),
            "row_count": len(df),
            "backup_version": backup_version,
            "message": "Data uploaded, validated, and sample data updated successfully"
        }
        
    except Exception as e:
        import traceback
        error_msg = str(e)
        traceback_str = traceback.format_exc()
        
        if audit_logger:
            audit_logger.log_agent_operation(
                agent_name="DataIngestionAgent",
                description=f"Data upload failed for {data_type}",
                status=LogStatus.FAIL,
                error=error_msg
            )
        
        raise HTTPException(
            status_code=500,
            detail={
                "error": error_msg,
                "traceback": traceback_str
            }
        )

@app.get("/api/data/{data_type}/history")
async def get_data_history(data_type: str):
    """Get backup history for a data type"""
    if data_type not in SAMPLE_FILE_MAP:
        raise HTTPException(status_code=404, detail=f"Data type not found: {data_type}")
    
    history = []
    
    # Find all backup files for this data type
    backup_files = list(BACKUP_DIR.glob(f"{data_type}_v*_metadata.json"))
    
    for metadata_file in sorted(backup_files, reverse=True):  # Most recent first
        try:
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
                # Check if backup file still exists
                backup_file = BACKUP_DIR / metadata['backup_filename']
                if backup_file.exists():
                    metadata['backup_exists'] = True
                    metadata['backup_size'] = backup_file.stat().st_size
                else:
                    metadata['backup_exists'] = False
                history.append(metadata)
        except Exception as e:
            continue
    
    return {
        "success": True,
        "data_type": data_type,
        "history": history,
        "count": len(history)
    }

@app.post("/api/data/{data_type}/generate")
async def generate_sample_data(
    data_type: str,
    num_rows: Optional[int] = Query(None, description="Number of rows to generate (only for new_articles_data)")
):
    """Generate sample data for a specific data type with SKU matching
    
    Args:
        data_type: Type of data to generate
        num_rows: Number of rows to generate (only used for new_articles_data, default: 10)
    """
    # For new_articles_data, use the provided num_rows or default to 10
    if data_type == 'new_articles_data':
        num_rows = num_rows if num_rows is not None else 10
    
    try:
        # Read existing new articles to get SKUs for matching
        products_file = SAMPLE_DATA_DIR / "products.csv"
        existing_skus = []
        if products_file.exists():
            try:
                df_products = read_data_file(str(products_file))
                # Try both vendor_sku and sku columns
                if 'vendor_sku' in df_products.columns:
                    existing_skus = df_products['vendor_sku'].dropna().unique().tolist()
                elif 'sku' in df_products.columns:
                    existing_skus = df_products['sku'].dropna().unique().tolist()
            except Exception as e:
                print(f"Warning: Could not read products file: {e}")
        
        generated_data = []
        
        if data_type == 'sales_data':
            # Generate sales data with 60% matching SKUs
            stores = ['store_001', 'store_002', 'store_003', 'store_004', 'store_005']
            cities = ['Delhi', 'Mumbai', 'Bangalore', 'Chennai', 'Kolkata']
            price_buckets = ['699-999', '1299-1799', '1999-2499', '2999-3999']
            
            # Use 60% of existing SKUs, add 40% new ones
            matching_skus = existing_skus[:int(len(existing_skus) * 0.6)] if existing_skus else []
            new_skus = [f'SKU-{i:03d}' for i in range(1, 11)]
            all_skus = matching_skus + new_skus[:10 - len(matching_skus)]
            
            start_date = datetime.now() - timedelta(days=90)
            for i in range(50):  # Generate 50 rows
                date = start_date + timedelta(days=random.randint(0, 89))
                generated_data.append({
                    'date': date.strftime('%Y-%m-%d'),
                    'store_id': random.choice(stores),
                    'sku': random.choice(all_skus),
                    'units_sold': random.randint(1, 10),
                    'revenue': random.randint(300, 5000),
                    'city': random.choice(cities),
                    'avgdiscount': f"{random.randint(0, 20)}%",
                    'pricebucket': random.choice(price_buckets)
                })
        
        elif data_type == 'inventory_data':
            stores = ['store_001', 'store_002', 'store_003', 'store_004', 'store_005']
            matching_skus = existing_skus[:int(len(existing_skus) * 0.6)] if existing_skus else []
            new_skus = [f'SKU-{i:03d}' for i in range(1, 21)]
            all_skus = matching_skus + new_skus[:max(0, 20 - len(matching_skus))]
            
            for store in stores:
                for sku in all_skus[:15]:  # Limit to 15 SKUs per store
                    generated_data.append({
                        'store_id': store,
                        'sku': sku,
                        'on_hand': random.randint(10, 100),
                        'in_transit': random.randint(0, 20)
                    })
        
        elif data_type == 'price_data':
            stores = ['store_001', 'store_002', 'store_003', 'store_004', 'store_005']
            matching_skus = existing_skus[:int(len(existing_skus) * 0.6)] if existing_skus else []
            new_skus = [f'SKU-{i:03d}' for i in range(1, 21)]
            all_skus = matching_skus + new_skus[:max(0, 20 - len(matching_skus))]
            
            base_prices = [299, 349, 399, 499, 599, 699, 799, 899, 999, 1299, 1499, 1699, 1999]
            for store in stores:
                for sku in all_skus[:15]:
                    generated_data.append({
                        'store_id': store,
                        'sku': sku,
                        'price': random.choice(base_prices),
                        'markdown_flag': random.choice([True, False])
                    })
        
        elif data_type == 'cost_data':
            matching_skus = existing_skus[:int(len(existing_skus) * 0.6)] if existing_skus else []
            new_skus = [f'SKU-{i:03d}' for i in range(1, 21)]
            all_skus = matching_skus + new_skus[:max(0, 20 - len(matching_skus))]
            
            for sku in all_skus[:20]:  # Limit to 20 SKUs
                generated_data.append({
                    'sku': sku,
                    'cost': random.randint(100, 800),
                    'currency': 'INR'
                })
        
        elif data_type == 'new_articles_data':
            # Get existing SKUs from sales/inventory/price data for 60% matching
            existing_skus_for_matching = []
            if (SAMPLE_DATA_DIR / "sales.csv").exists():
                try:
                    df_sales = read_data_file(str(SAMPLE_DATA_DIR / "sales.csv"))
                    if 'sku' in df_sales.columns:
                        existing_skus_for_matching.extend(df_sales['sku'].unique().tolist())
                except:
                    pass
            if (SAMPLE_DATA_DIR / "inventory.csv").exists():
                try:
                    df_inv = read_data_file(str(SAMPLE_DATA_DIR / "inventory.csv"))
                    if 'sku' in df_inv.columns:
                        existing_skus_for_matching.extend(df_inv['sku'].unique().tolist())
                except:
                    pass
            if (SAMPLE_DATA_DIR / "price.csv").exists():
                try:
                    df_price = read_data_file(str(SAMPLE_DATA_DIR / "price.csv"))
                    if 'sku' in df_price.columns:
                        existing_skus_for_matching.extend(df_price['sku'].unique().tolist())
                except:
                    pass
            
            # Get unique SKUs and also check products for vendor_sku
            unique_existing_skus = list(set(existing_skus_for_matching))
            if products_file.exists():
                try:
                    df_products = read_data_file(str(products_file))
                    if 'vendor_sku' in df_products.columns:
                        product_skus = df_products['vendor_sku'].dropna().unique().tolist()
                        unique_existing_skus.extend(product_skus)
                    unique_existing_skus = list(set(unique_existing_skus))
                except:
                    pass
            
            # Use num_rows parameter for new_articles_data (default to 10 if not provided)
            rows_to_generate = num_rows if num_rows else 10
            matching_count = min(int(rows_to_generate * 0.6), len(unique_existing_skus))  # 60% matching
            matching_skus = unique_existing_skus[:matching_count]
            new_skus = [f'VS-{str(i).zfill(3)}' for i in range(100, 100 + (rows_to_generate - matching_count))]
            
            # Generate new articles with 60% matching SKUs
            categories = ['TSHIRT', 'POLO', 'JEANS', 'HOODIE', 'PANTS', 'BLAZER', 'SWEATPANTS', 'JACKET', 'CHINOS', 'SWEATER']
            colors = ['White', 'Black', 'Blue', 'Red', 'Green', 'Gray', 'Navy Blue', 'Beige', 'Purple', 'Yellow']
            materials = ['Cotton', 'Denim', 'Polyester', 'Cotton Blend', 'Wool Blend', 'Nylon']
            size_sets = ['S,M,L,XL', 'M,L,XL', '30,32,34,36', '38,40,42,44']
            classes = ['Top Wear', 'Bottom Wear']
            segments = ['Mens wear', 'Womens wear']
            families = ['Active Wear', 'Casual Wear', 'Formal Wear', 'Classic Wear', 'Denim Wear', 'Outerwear']
            brands = ['GAP', 'BENETTON', 'LEVIS', 'NIKE', 'ADIDAS', 'ZARA', 'PUMA', 'COLUMBIA', 'H&M', 'TOMMY HILFIGER']
            
            all_skus = matching_skus + new_skus
            for i in range(rows_to_generate):
                category = random.choice(categories)
                sku = all_skus[i] if i < len(all_skus) else f'VS-{str(100 + i).zfill(3)}'
                generated_data.append({
                    'product_id': f'P{100 + i}',
                    'vendor_sku': sku,
                    'description': f'{random.choice(colors)} {category.lower()}, sample description {i+1}',
                    'category': category,
                    'color': random.choice(colors),
                    'material': random.choice(materials),
                    'size_set': random.choice(size_sets),
                    'brick': category,
                    'class': random.choice(classes),
                    'segment': random.choice(segments),
                    'family': random.choice(families),
                    'brand': random.choice(brands)
                })
        
        # Convert to DataFrame and then to CSV
        if generated_data:
            df = pd.DataFrame(generated_data)
            csv_buffer = io.StringIO()
            df.to_csv(csv_buffer, index=False)
            csv_content = csv_buffer.getvalue()
            
            return {
                "success": True,
                "data": generated_data,
                "columns": list(df.columns),
                "row_count": len(generated_data),
                "csv": csv_content
            }
        else:
            raise HTTPException(status_code=400, detail=f"Cannot generate data for {data_type}")
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating sample data: {str(e)}")

@app.get("/api/data/{data_type}/restore/{version}")
async def restore_data_version(data_type: str, version: str):
    """Restore a specific version from backup"""
    if data_type not in SAMPLE_FILE_MAP:
        raise HTTPException(status_code=404, detail=f"Data type not found: {data_type}")
    
    metadata_file = BACKUP_DIR / f"{data_type}_v{version}_metadata.json"
    if not metadata_file.exists():
        raise HTTPException(status_code=404, detail=f"Backup version not found: {version}")
    
    try:
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
        
        backup_file = BACKUP_DIR / metadata['backup_filename']
        if not backup_file.exists():
            raise HTTPException(status_code=404, detail=f"Backup file not found: {backup_file}")
        
        sample_file = SAMPLE_DATA_DIR / SAMPLE_FILE_MAP[data_type]
        
        # Backup current sample before restoring
        if sample_file.exists():
            current_backup_version = datetime.now().strftime('%Y%m%d_%H%M%S')
            current_backup_file = BACKUP_DIR / f"{data_type}_v{current_backup_version}{sample_file.suffix}"
            import shutil
            shutil.copy2(sample_file, current_backup_file)
        
        # Restore from backup
        import shutil
        shutil.copy2(backup_file, sample_file)
        
        # Get restored data
        df = read_data_file(str(sample_file))
        data_records = df.to_dict('records')
        # Convert numpy types to native Python types
        data_records = convert_numpy_types(data_records)
        
        if audit_logger:
            audit_logger.log_agent_operation(
                agent_name="DataIngestionAgent",
                description=f"Data restored to version {version} for {data_type}",
                status=LogStatus.SUCCESS
            )
        
        return {
            "success": True,
            "message": f"Data restored to version {version}",
            "data": data_records,
            "columns": list(df.columns),
            "row_count": len(df)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error restoring data: {str(e)}")

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)

