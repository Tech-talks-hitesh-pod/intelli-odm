"""
FastAPI Backend for Demand Forecasting & Allocation Engine
"""

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from typing import Optional, List, Dict, Any
import pandas as pd
import numpy as np
import os
import json
from datetime import datetime
from pathlib import Path
import uvicorn

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
ALLOWED_EXTENSIONS = {'.csv', '.xlsx', '.xls'}

# Initialize components
ollama_client = OllamaClient(base_url="http://localhost:11434", model="llama3:8b")
audit_logger = AuditLogger(log_dir="audit-logs")
kb = SharedKnowledgeBase()
data_handler = DataIngestionAgent()

# Global state
current_run_id: Optional[str] = None

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

@app.post("/api/forecast")
async def run_forecast(
    background_tasks: BackgroundTasks,
    file_paths: str = Form(...),  # JSON string of file paths
    margin_target: float = Form(30.0),
    variance_threshold: float = Form(5.0),
    forecast_horizon_days: int = Form(60),
    max_quantity_per_store: int = Form(500),
    universe_of_stores: Optional[int] = Form(None),
    store_mappings: Optional[str] = Form(None)  # JSON string
):
    """Run demand forecasting and allocation"""
    global current_run_id
    
    try:
        # Start new run
        current_run_id = audit_logger.start_run()
        
        # Parse file paths
        file_paths_dict = json.loads(file_paths)
        
        # Load data from files
        sales_data = None
        inventory_data = None
        price_data = None
        cost_data = None
        new_articles_data = None
        
        if 'sales_data' in file_paths_dict:
            sales_data = read_data_file(file_paths_dict['sales_data'])
        
        if 'inventory_data' in file_paths_dict:
            inventory_data = read_data_file(file_paths_dict['inventory_data'])
        
        if 'price_data' in file_paths_dict:
            price_data = read_data_file(file_paths_dict['price_data'])
        
        if 'cost_data' in file_paths_dict:
            cost_data = read_data_file(file_paths_dict['cost_data'])
        
        if 'new_articles_data' in file_paths_dict:
            new_articles_data = read_data_file(file_paths_dict['new_articles_data'])
        
        # Convert percentages to decimals
        margin_target_decimal = margin_target / 100
        variance_threshold_decimal = variance_threshold / 100
        
        # Extract product attributes from new articles data
        product_attributes = {}
        if new_articles_data is not None and not new_articles_data.empty:
            for _, row in new_articles_data.iterrows():
                sku = row.get('sku', '')
                if sku:
                    product_attributes[sku] = {
                        'style_code': row.get('style_code', 'N/A'),
                        'color': row.get('color', 'N/A'),
                        'segment': row.get('segment', 'N/A'),
                        'family': row.get('family', 'N/A'),
                        'class': row.get('class', 'N/A'),
                        'brick': row.get('brick', 'N/A')
                    }
        
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
        
        # Get comparables from attribute analogy agent
        comparables = []
        if new_articles_data is not None and not new_articles_data.empty:
            # For each new article, find comparables
            for _, row in new_articles_data.iterrows():
                # Build product description from attributes
                desc_parts = []
                for attr in ['style_code', 'color', 'segment', 'family', 'class', 'brick']:
                    if attr in row and pd.notna(row[attr]):
                        desc_parts.append(f"{attr}: {row[attr]}")
                
                product_description = ", ".join(desc_parts) if desc_parts else row.get('sku', '')
                
                # Find comparables
                comps, trends = attribute_agent.run(product_description, sales_data)
                comparables.extend(comps)
        
        # Run demand forecasting
        results, sensitivity = demand_agent.run(
            comparables=comparables,
            sales_data=sales_data,
            inventory_data=inventory_data,
            price_data=price_data,
            price_options=[200, 300, 400, 500, 600, 700, 800, 900, 1000],
            product_attributes=product_attributes,
            forecast_horizon_days=forecast_horizon_days,
            variance_threshold=variance_threshold_decimal,
            cost_data=cost_data,
            margin_target=margin_target_decimal,
            max_quantity_per_store=max_quantity_per_store
        )
        
        # Format results for UI
        formatted_results = format_results_for_ui(results, sensitivity)
        formatted_results['run_id'] = current_run_id
        
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

def format_results_for_ui(results: Dict[str, Any], sensitivity: Dict[str, Any]) -> Dict[str, Any]:
    """Format results for UI display"""
    recommendations = results.get('recommendations', {})
    
    formatted = {
        'model_selection': results.get('model_selection', {}),
        'factor_analysis': results.get('factor_analysis', {}),
        'recommendations': recommendations,
        'top_banner': recommendations.get('top_banner', {}),
        'article_level_metrics': recommendations.get('article_level_metrics', {}),
        'store_allocations': recommendations.get('store_allocations', {}),
        'sensitivity_analysis': sensitivity,
        'validation_messages': results.get('validation_messages', []),
        'fallback_used': results.get('fallback_used', False),
        'hitl_metadata': results.get('hitl_metadata', {})
    }
    
    return formatted

@app.get("/api/audit-logs/{run_id}")
async def get_audit_logs(run_id: str):
    """Get audit logs for a specific run"""
    logs = audit_logger.get_run_logs(run_id)
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

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)

