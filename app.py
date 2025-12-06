# app.py
"""
Flask Web Application for Demand Forecasting and Allocation Engine
"""

from flask import Flask, render_template, request, jsonify, send_file
import pandas as pd
import numpy as np
import os
from werkzeug.utils import secure_filename
from typing import Dict, List, Any, Optional
import json
from datetime import datetime

# Import agents
from agents.data_ingestion_agent import DataIngestionAgent
from agents.attribute_analogy_agent import AttributeAnalogyAgent
from agents.demand_forecasting_agent import DemandForecastingAgent
from shared_knowledge_base import SharedKnowledgeBase

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['SECRET_KEY'] = 'your-secret-key-here'

# Create uploads directory if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Allowed file extensions
ALLOWED_EXTENSIONS = {'csv', 'xlsx', 'xls'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Initialize agents (will be initialized on first use)
data_handler = None
attribute_agent = None
demand_agent = None
kb = None
llama_client = None  # Placeholder - replace with actual LLM client

def initialize_agents():
    """Initialize agents on first use"""
    global data_handler, attribute_agent, demand_agent, kb, llama_client
    
    if kb is None:
        kb = SharedKnowledgeBase()
    
    if data_handler is None:
        data_handler = DataIngestionAgent()
    
    if attribute_agent is None:
        # Initialize with placeholder LLM client
        # Replace with actual LLM client initialization
        # For now, use None - the agent will handle it gracefully
        try:
            attribute_agent = AttributeAnalogyAgent(llama_client, kb)
        except:
            attribute_agent = None
    
    if demand_agent is None:
        # DemandForecastingAgent can work without LLM client (uses fallback)
        demand_agent = DemandForecastingAgent(
            llama_client=llama_client,
            enable_hitl=True,
            variance_threshold=0.05
        )

@app.route('/')
def index():
    """Main page"""
    return render_template('index.html')

@app.route('/api/upload', methods=['POST'])
def upload_files():
    """Handle file uploads"""
    try:
        uploaded_files = {}
        
        # Sales data
        if 'sales_data' in request.files:
            file = request.files['sales_data']
            if file and allowed_file(file.filename):
                filename = secure_filename(f"sales_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(filepath)
                uploaded_files['sales_data'] = filepath
        
        # Inventory data
        if 'inventory_data' in request.files:
            file = request.files['inventory_data']
            if file and allowed_file(file.filename):
                filename = secure_filename(f"inventory_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(filepath)
                uploaded_files['inventory_data'] = filepath
        
        # Price data
        if 'price_data' in request.files:
            file = request.files['price_data']
            if file and allowed_file(file.filename):
                filename = secure_filename(f"price_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(filepath)
                uploaded_files['price_data'] = filepath
        
        # Cost data
        if 'cost_data' in request.files:
            file = request.files['cost_data']
            if file and allowed_file(file.filename):
                filename = secure_filename(f"cost_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(filepath)
                uploaded_files['cost_data'] = filepath
        
        # New articles data
        if 'new_articles_data' in request.files:
            file = request.files['new_articles_data']
            if file and allowed_file(file.filename):
                filename = secure_filename(f"new_articles_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(filepath)
                uploaded_files['new_articles_data'] = filepath
        
        return jsonify({
            'success': True,
            'message': 'Files uploaded successfully',
            'files': uploaded_files
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400

@app.route('/api/forecast', methods=['POST'])
def run_forecast():
    """Run demand forecasting and allocation"""
    try:
        initialize_agents()
        
        # Get uploaded file paths
        data = request.json
        file_paths = data.get('file_paths', {})
        
        # Load data from files
        sales_data = None
        inventory_data = None
        price_data = None
        cost_data = None
        new_articles_data = None
        
        def read_data_file(filepath):
            """Read CSV or Excel file"""
            if filepath.endswith('.csv'):
                return pd.read_csv(filepath)
            elif filepath.endswith(('.xlsx', '.xls')):
                return pd.read_excel(filepath)
            else:
                raise ValueError(f"Unsupported file format: {filepath}")
        
        if 'sales_data' in file_paths:
            sales_data = read_data_file(file_paths['sales_data'])
        
        if 'inventory_data' in file_paths:
            inventory_data = read_data_file(file_paths['inventory_data'])
        
        if 'price_data' in file_paths:
            price_data = read_data_file(file_paths['price_data'])
        
        if 'cost_data' in file_paths:
            cost_data = read_data_file(file_paths['cost_data'])
        
        if 'new_articles_data' in file_paths:
            new_articles_data = read_data_file(file_paths['new_articles_data'])
        
        # Get parameters from UI
        margin_target = data.get('margin_target', 30) / 100  # Convert percentage to decimal
        variance_threshold = data.get('variance_threshold', 5) / 100
        forecast_horizon_days = data.get('forecast_horizon_days', 60)
        max_quantity_per_store = data.get('max_quantity_per_store', 500)
        universe_of_stores = data.get('universe_of_stores', None)
        price_options = data.get('price_options', [200, 300, 400, 500, 600, 700, 800, 900, 1000])
        
        # Extract product attributes from new articles data
        product_attributes = {}
        if new_articles_data is not None and not new_articles_data.empty:
            # Assuming new_articles_data has columns: sku, style_code, color, segment, family, class, brick
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
        
        # Get comparables from attribute analogy agent
        # For now, use empty list - in production, call attribute_analogy_agent
        comparables = []
        if new_articles_data is not None and not new_articles_data.empty:
            # Extract product descriptions from new articles
            product_descriptions = []
            for _, row in new_articles_data.iterrows():
                desc = {
                    'sku': row.get('sku', ''),
                    'attributes': {
                        'style_code': row.get('style_code', ''),
                        'color': row.get('color', ''),
                        'segment': row.get('segment', ''),
                        'family': row.get('family', ''),
                        'class': row.get('class', ''),
                        'brick': row.get('brick', '')
                    }
                }
                product_descriptions.append(desc)
            
            # Get comparables (simplified - in production, call attribute_analogy_agent)
            comparables = product_descriptions
        
        # Update demand agent parameters
        demand_agent.margin_target = margin_target
        demand_agent.universe_of_stores = universe_of_stores
        
        # Run demand forecasting
        results, sensitivity = demand_agent.run(
            comparables=comparables,
            sales_data=sales_data,
            inventory_data=inventory_data,
            price_data=price_data,
            price_options=price_options,
            product_attributes=product_attributes,
            forecast_horizon_days=forecast_horizon_days,
            variance_threshold=variance_threshold,
            cost_data=cost_data,
            margin_target=margin_target,
            max_quantity_per_store=max_quantity_per_store
        )
        
        # Format results for UI
        formatted_results = format_results_for_ui(results, sensitivity)
        
        return jsonify({
            'success': True,
            'results': formatted_results
        })
    
    except Exception as e:
        import traceback
        return jsonify({
            'success': False,
            'error': str(e),
            'traceback': traceback.format_exc()
        }), 500

def format_results_for_ui(results: Dict[str, Any], sensitivity: Dict[str, Any]) -> Dict[str, Any]:
    """Format results for UI display"""
    
    # Extract recommendations from results
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

@app.route('/api/stores', methods=['GET'])
def get_available_stores():
    """Get available stores for mapping"""
    try:
        initialize_agents()
        
        # This would require sales_data to be loaded
        # For now, return empty list
        return jsonify({
            'success': True,
            'stores': []
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)

