"""Agent-specific configuration parameters."""

from typing import Dict, Any, List

# Data Ingestion Agent Configuration
DATA_INGESTION_CONFIG: Dict[str, Any] = {
    "validation": {
        "max_missing_percentage": 10.0,
        "required_columns": {
            "sales": ["date", "store_id", "sku", "units_sold"],
            "inventory": ["store_id", "sku", "on_hand"],
            "pricing": ["store_id", "sku", "price"],
            "products": ["product_id", "description"]
        },
        "date_formats": ["%Y-%m-%d", "%d/%m/%Y", "%m/%d/%Y"],
        "outlier_detection": {
            "method": "iqr",
            "threshold": 3.0
        }
    },
    "feature_engineering": {
        "velocity_window_days": 30,
        "seasonality_periods": [7, 30, 365],
        "store_tier_thresholds": [0.7, 0.4]  # High, Medium, Low
    }
}

# Attribute & Analogy Agent Configuration
ATTRIBUTE_AGENT_CONFIG: Dict[str, Any] = {
    "llm": {
        "max_retries": 3,
        "temperature": 0.1,
        "max_tokens": 500,
        "timeout": 60
    },
    "similarity": {
        "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
        "similarity_threshold": 0.6,
        "max_comparables": 10
    },
    "trend_analysis": {
        "time_windows": ["1M", "3M", "6M"],
        "trend_confidence_threshold": 0.7
    },
    "prompt_templates": {
        "attribute_extraction": """
Extract structured attributes from this fashion product description.
Return ONLY a valid JSON object with these fields: material, color, pattern, sleeve, neckline, fit, category, style, target_gender.

Product Description: "{description}"

Guidelines:
- Use standardized terms (e.g., "Cotton" not "cotton blend")
- If information is unclear, use "Unknown"
- Be consistent with fashion industry terminology

JSON Response:""",
        "trend_analysis": """
Analyze market trends for this product category based on the provided data.

Product Attributes: {attributes}
Time Period: {time_period}

Provide insights in 2-3 sentences about:
1. Category performance trend
2. Seasonal patterns if any
3. Key market opportunities

Response:"""
    }
}

# Demand Forecasting Agent Configuration
DEMAND_FORECASTING_CONFIG: Dict[str, Any] = {
    "method_selection": {
        "analogy_threshold": {
            "min_comparables": 3,
            "min_similarity": 0.7
        },
        "timeseries_threshold": {
            "min_history_months": 12,
            "seasonality_detection": True
        },
        "ml_threshold": {
            "min_stores": 10,
            "min_features": 5
        }
    },
    "forecasting": {
        "forecast_horizon_days": 60,
        "confidence_levels": [0.1, 0.9],  # 10th and 90th percentiles
        "seasonality_periods": [7, 30.44, 365.25]
    },
    "price_sensitivity": {
        "elasticity_method": "log_linear",
        "price_range_factor": 0.3  # +/-30% of base price
    },
    "models": {
        "prophet": {
            "seasonality_mode": "multiplicative",
            "yearly_seasonality": True,
            "weekly_seasonality": True,
            "daily_seasonality": False,
            "changepoint_prior_scale": 0.05
        },
        "gradient_boosting": {
            "n_estimators": 100,
            "learning_rate": 0.1,
            "max_depth": 6,
            "random_state": 42
        }
    }
}

# Procurement & Allocation Agent Configuration
PROCUREMENT_CONFIG: Dict[str, Any] = {
    "optimization": {
        "solver": "pulp",  # Options: "pulp", "cvxpy"
        "solver_timeout_seconds": 300,
        "mip_gap_tolerance": 0.01,
        "objectives": {
            "revenue": {"weight": 0.6},
            "margin": {"weight": 0.3},
            "risk": {"weight": 0.1}
        }
    },
    "constraints": {
        "enforce_pack_size": True,
        "allow_partial_allocation": False,
        "min_store_allocation": 10,
        "max_capacity_utilization": 0.9
    },
    "risk_assessment": {
        "confidence_weight": 0.4,
        "sellthrough_weight": 0.4,
        "similarity_weight": 0.2
    }
}

# Alias for backward compatibility
PROCUREMENT_AGENT_CONFIG = PROCUREMENT_CONFIG

# Orchestrator Agent Configuration
ORCHESTRATOR_AGENT_CONFIG: Dict[str, Any] = {
    "execution": {
        "max_workers": 3,
        "timeout_seconds": 600,
        "retry_attempts": 2,
        "parallel_processing": True
    },
    "workflow": {
        "enable_phase_skipping": False,
        "continue_on_error": True,
        "min_confidence_threshold": 0.5
    },
    "prompt_templates": {
        "executive_summary": """
Generate a concise executive summary for the Intelli-ODM analysis:

Analysis Results:
- Products processed: {total_products}
- Successful analyses: {successful_analyses}  
- Demand forecasts: {forecasts_generated}
- Procurement recommendations: {procurement_recommendations}
- Total recommended investment: ${total_investment:,.0f}
- Overall success rate: {success_rate:.1%}

Provide a brief 2-3 sentence summary highlighting key findings and recommendations.
"""
    },
    "logging": {
        "log_phase_results": True,
        "log_errors": True,
        "log_metrics": True
    }
}

# Shared Knowledge Base Configuration
KNOWLEDGE_BASE_CONFIG: Dict[str, Any] = {
    "chromadb": {
        "collection_name": "intelli_odm_products",
        "distance_metric": "cosine",
        "embedding_function": "sentence-transformers"
    },
    "embeddings": {
        "model_name": "sentence-transformers/all-MiniLM-L6-v2",
        "batch_size": 32,
        "cache_embeddings": True
    },
    "similarity_search": {
        "default_top_k": 5,
        "similarity_threshold": 0.6,
        "include_metadata": True
    }
}