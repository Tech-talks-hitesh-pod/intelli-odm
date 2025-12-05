#!/usr/bin/env python3
"""
Debug API for Vector Database Inspection
"""

import sys
import os
from flask import Flask, jsonify, request
from flask_cors import CORS
import json
import logging

# Add project root to path
sys.path.append('.')

try:
    from shared_knowledge_base import SharedKnowledgeBase
    from agents.data_ingestion_agent import DataIngestionAgent
    from demo_scenarios import DemoScenarios
except ImportError as e:
    print(f"Import error: {e}")
    sys.exit(1)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)  # Enable CORS for web access

# Initialize knowledge base
kb = SharedKnowledgeBase()

@app.route('/api/debug/health', methods=['GET'])
def health_check():
    """Basic health check endpoint."""
    return jsonify({
        "status": "healthy",
        "message": "Debug API is running",
        "endpoints": [
            "/api/debug/health",
            "/api/debug/vector-db/status", 
            "/api/debug/vector-db/products",
            "/api/debug/vector-db/count",
            "/api/debug/vector-db/search",
            "/api/debug/demo/load-scenario",
            "/api/debug/demo/scenarios"
        ]
    })

@app.route('/api/debug/vector-db/status', methods=['GET'])
def vector_db_status():
    """Get vector database status and basic info."""
    try:
        # Get collection info
        collection_count = kb.get_collection_size()
        
        # Get collection stats
        stats = kb.get_collection_stats()
        
        return jsonify({
            "status": "success",
            "vector_db": {
                "collection_size": collection_count,
                "collection_stats": stats,
                "is_empty": collection_count == 0
            }
        })
        
    except Exception as e:
        return jsonify({
            "status": "error",
            "error": str(e)
        }), 500

@app.route('/api/debug/vector-db/products', methods=['GET'])
def get_all_products():
    """Get all products from vector database."""
    try:
        limit = request.args.get('limit', 10, type=int)
        limit = min(limit, 100)  # Cap at 100
        
        # Get all products with performance data
        all_products = kb.get_all_products_with_performance()
        
        # Limit results
        products = all_products[:limit] if all_products else []
        
        return jsonify({
            "status": "success", 
            "total_products": len(all_products) if all_products else 0,
            "returned_products": len(products),
            "products": products
        })
        
    except Exception as e:
        return jsonify({
            "status": "error",
            "error": str(e)
        }), 500

@app.route('/api/debug/vector-db/count', methods=['GET'])
def get_product_count():
    """Get simple product count from vector database."""
    try:
        count = kb.get_collection_size()
        
        return jsonify({
            "status": "success",
            "product_count": count,
            "message": f"Vector database contains {count} products"
        })
        
    except Exception as e:
        return jsonify({
            "status": "error", 
            "error": str(e)
        }), 500

@app.route('/api/debug/vector-db/search', methods=['GET'])
def search_similar_products():
    """Search for similar products using vector similarity."""
    try:
        query = request.args.get('q', 'women denim shorts')
        top_k = request.args.get('top_k', 5, type=int)
        top_k = min(top_k, 20)  # Cap at 20
        
        # Search for similar products
        similar_products = kb.find_similar_products(
            query_attributes={},
            query_description=query,
            top_k=top_k
        )
        
        return jsonify({
            "status": "success",
            "query": query,
            "found_products": len(similar_products),
            "similar_products": similar_products
        })
        
    except Exception as e:
        return jsonify({
            "status": "error",
            "error": str(e)
        }), 500

@app.route('/api/debug/demo/scenarios', methods=['GET'])
def get_demo_scenarios():
    """Get available demo scenarios."""
    try:
        scenarios = DemoScenarios.get_available_scenarios()
        
        scenario_info = {}
        for key, scenario in scenarios.items():
            scenario_info[key] = {
                "name": scenario["name"],
                "description": scenario["description"],
                "difficulty": scenario.get("difficulty", "medium")
            }
        
        return jsonify({
            "status": "success",
            "scenarios": scenario_info
        })
        
    except Exception as e:
        return jsonify({
            "status": "error",
            "error": str(e)
        }), 500

@app.route('/api/debug/demo/load-scenario', methods=['POST'])
def load_demo_scenario():
    """Load a specific demo scenario."""
    try:
        data = request.get_json()
        scenario_key = data.get('scenario', 'seasonal_demand')
        
        # Initialize data ingestion agent
        agent = DataIngestionAgent(knowledge_base=kb)
        
        # Load demo data
        result = agent.load_and_process_demo_data('data/sample')
        
        # Get updated counts
        product_count = kb.get_collection_size()
        
        return jsonify({
            "status": "success",
            "message": f"Loaded scenario: {scenario_key}",
            "products_loaded": product_count,
            "processing_result": {
                "products": len(result.get('products', [])) if result.get('products') is not None else 0,
                "sales": len(result.get('sales', [])) if result.get('sales') is not None else 0,
                "inventory": len(result.get('inventory', [])) if result.get('inventory') is not None else 0,
                "pricing": len(result.get('pricing', [])) if result.get('pricing') is not None else 0
            }
        })
        
    except Exception as e:
        return jsonify({
            "status": "error",
            "error": str(e)
        }), 500

@app.route('/api/debug/vector-db/raw', methods=['GET'])
def get_raw_collection():
    """Get raw collection data directly from ChromaDB."""
    try:
        # Access the collection directly
        results = kb._products_collection.get(
            limit=10,
            include=["metadatas", "documents"]
        )
        
        return jsonify({
            "status": "success",
            "raw_results": {
                "ids": results.get("ids", []),
                "metadatas": results.get("metadatas", []),
                "documents": results.get("documents", []),
                "count": len(results.get("ids", []))
            }
        })
        
    except Exception as e:
        return jsonify({
            "status": "error",
            "error": str(e)
        }), 500

if __name__ == '__main__':
    print("🔧 Debug API Server Starting...")
    print("📊 Available endpoints:")
    print("   GET  /api/debug/health")
    print("   GET  /api/debug/vector-db/status")
    print("   GET  /api/debug/vector-db/products?limit=10")
    print("   GET  /api/debug/vector-db/count") 
    print("   GET  /api/debug/vector-db/search?q=women+denim+shorts&top_k=5")
    print("   GET  /api/debug/vector-db/raw")
    print("   GET  /api/debug/demo/scenarios")
    print("   POST /api/debug/demo/load-scenario")
    print()
    print("🚀 Server will run at: http://localhost:5001")
    print("🌐 Test in browser or use curl:")
    print("   curl http://localhost:5001/api/debug/health")
    print("   curl http://localhost:5001/api/debug/vector-db/count")
    print()
    
    app.run(host='0.0.0.0', port=5001, debug=True)