#!/usr/bin/env python3
"""
Simplified ODM Intelligence Application
Focus: Load ODM data, show summary, predict sales for new products based on similar attributes
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import logging
import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('odm_app.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# IMPORTANT: Initialize LangSmith tracking BEFORE importing other modules
# This ensures environment variables are set before any traceable decorators are used
try:
    from utils.llm_client import setup_langsmith_tracking
    langsmith_callbacks = setup_langsmith_tracking()
    if langsmith_callbacks or os.getenv("LANGCHAIN_TRACING_V2") == "true":
        logger.info("✅ LangSmith tracking initialized at module level")
    else:
        logger.warning("⚠️ LangSmith tracking not configured - set LANGCHAIN_API_KEY in .env")
except Exception as e:
    logger.warning(f"⚠️ LangSmith setup failed: {e}")

# Import our modules
try:
    from shared_knowledge_base import SharedKnowledgeBase
    from agents.data_ingestion_agent import DataIngestionAgent
    from utils.llm_client import LLMClientFactory, LLMClient
    from config.settings import settings
    logger.info("✅ Successfully imported all required modules")
except ImportError as e:
    st.error(f"❌ Import error: {e}")
    logger.error(f"Import error: {e}")
    st.stop()

# Page config
st.set_page_config(
    page_title="ODM Intelligence",
    page_icon="🏪",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_resource
def initialize_system():
    """Initialize the ODM system with knowledge base and agents."""
    logger.info("🔄 Initializing ODM system...")
    
    try:
        # LangSmith should already be initialized at module level, but verify
        langchain_tracing = os.getenv("LANGCHAIN_TRACING_V2")
        langchain_api_key = os.getenv("LANGCHAIN_API_KEY") or os.getenv("LANGSMITH_API_KEY")
        
        if langchain_tracing == "true" and langchain_api_key:
            logger.info(f"✅ LangSmith tracking enabled (Project: {os.getenv('LANGCHAIN_PROJECT', 'intelli-odm')})")
        else:
            logger.warning("⚠️ LangSmith tracking not configured. Add to .env:")
            logger.warning("   LANGCHAIN_TRACING_V2=true")
            logger.warning("   LANGCHAIN_API_KEY=your_api_key_here")
            logger.warning("   LANGCHAIN_PROJECT=intelli-odm")
        
        # Initialize knowledge base
        kb = SharedKnowledgeBase()
        logger.info("✅ Knowledge base initialized")
        
        # Initialize LLM client using factory
        llm_config = {
            'provider': settings.llm_provider,
            'api_key': settings.openai_api_key if settings.llm_provider == 'openai' else None,
            'model': settings.openai_model if settings.llm_provider == 'openai' else settings.ollama_model,
            'temperature': settings.openai_temperature if settings.llm_provider == 'openai' else 0.1,
            'base_url': settings.ollama_base_url if settings.llm_provider == 'ollama' else None,
            'timeout': settings.ollama_timeout if settings.llm_provider == 'ollama' else 300
        }
        
        try:
            llm_client = LLMClientFactory.create_client(llm_config)
            if llm_client is None:
                logger.info("🔄 Demo mode - continuing with statistical analysis only (no AI insights)")
            else:
                logger.info("✅ LLM client initialized")
        except Exception as llm_error:
            logger.warning(f"⚠️ LLM client initialization failed: {llm_error}")
            logger.info("🔄 Continuing with statistical analysis only (no AI insights)")
            llm_client = None
        
        # Initialize data ingestion agent
        data_agent = DataIngestionAgent(llm_client, kb)
        logger.info("✅ Data ingestion agent initialized")
        
        return kb, llm_client, data_agent
        
    except Exception as e:
        logger.error(f"❌ System initialization failed: {e}")
        st.error(f"System initialization failed: {e}")
        st.stop()

@st.cache_data
def load_odm_data(_kb, _data_agent):
    """Load and process ODM data on startup."""
    logger.info("🔄 Loading ODM data...")
    
    try:
        # Paths to ODM files
        dirty_input_file = "data/sample/dirty_odm_input.csv"
        historical_file = "data/sample/odm_historical_dataset_5000.csv"
        
        # Load the data
        result = _data_agent.load_dirty_odm_data(dirty_input_file, historical_file)
        logger.info(f"✅ ODM data loaded: {result.get('summary', {})}")
        
        return result
        
    except Exception as e:
        logger.error(f"❌ Failed to load ODM data: {e}")
        st.error(f"Failed to load ODM data: {e}")
        return None

@st.cache_data
def load_store_data():
    """Load store information from historical dataset."""
    # Wrap with LangSmith tracing
    try:
        from langsmith import traceable
        
        @traceable(name="Load_Store_Data", run_type="retriever", tags=["data-loading", "stores"])
        def _load_stores():
            return _load_store_data_impl()
        
        return _load_stores()
    except ImportError:
        return _load_store_data_impl()
    except Exception as e:
        logger.warning(f"LangSmith tracing failed for store data loading: {e}")
        return _load_store_data_impl()

def _load_store_data_impl():
    """Internal implementation of store data loading."""
    try:
        historical_file = "data/sample/odm_historical_dataset_5000.csv"
        df = pd.read_csv(historical_file)
        
        # Get unique stores with their details
        stores = df[['StoreID', 'StoreName', 'City', 'ClimateTag']].drop_duplicates()
        stores = stores.sort_values('StoreID')
        
        logger.info(f"✅ Loaded {len(stores)} unique stores")
        return stores.to_dict('records')
        
    except Exception as e:
        logger.error(f"❌ Failed to load store data: {e}")
        return []

def get_weather_factor(climate_tag: str, product_description: str) -> float:
    """Calculate weather factor based on climate and product type."""
    # Don't trace individual weather factor calls - they're called many times per store
    # The parent Generate_Store_Predictions trace will show the aggregate results
    return _get_weather_factor_impl(climate_tag, product_description)

def _get_weather_factor_impl(climate_tag: str, product_description: str) -> float:
    """Internal implementation of weather factor calculation."""
    description_lower = product_description.lower()
    climate_lower = str(climate_tag).lower()
    
    # Winter products
    winter_keywords = ['winter', 'wool', 'sweater', 'jacket', 'coat', 'blazer', 'warm']
    # Summer products
    summer_keywords = ['summer', 'cotton', 'linen', 't-shirt', 'tshirt', 'shorts', 'light']
    # Monsoon products
    monsoon_keywords = ['rain', 'waterproof', 'monsoon', 'windcheater']
    
    # Determine product type
    is_winter_product = any(kw in description_lower for kw in winter_keywords)
    is_summer_product = any(kw in description_lower for kw in summer_keywords)
    is_monsoon_product = any(kw in description_lower for kw in monsoon_keywords)
    
    # Climate-based factors
    if 'winter' in climate_lower or 'cold' in climate_lower:
        if is_winter_product:
            return 1.5  # High demand
        elif is_summer_product:
            return 0.6  # Low demand
        else:
            return 1.0
    elif 'hot' in climate_lower or 'humid' in climate_lower:
        if is_summer_product:
            return 1.5  # High demand
        elif is_winter_product:
            return 0.6  # Low demand
        else:
            return 1.0
    elif 'mild' in climate_lower or 'pleasant' in climate_lower:
        return 1.2  # Good for all products
    elif 'monsoon' in climate_lower or 'rain' in climate_lower:
        if is_monsoon_product:
            return 1.4
        else:
            return 0.8
    elif 'festive' in climate_lower:
        return 1.3  # Festive season boost
    else:
        return 1.0  # Neutral

def generate_store_predictions(base_prediction: int, product_description: str, stores: List[Dict]) -> List[Dict]:
    """Generate store-wise predictions based on base prediction and store characteristics."""
    # Wrap with LangSmith tracing
    try:
        from langsmith import traceable
        
        # Use wrapper function to ensure proper trace nesting
        @traceable(
            name="Generate_Store_Predictions", 
            run_type="chain", 
            tags=["store-predictions", "allocation"],
            inputs={"base_prediction": base_prediction, "product_description": product_description, "num_stores": len(stores)}
        )
        def _generate_wrapper(base_pred: int, prod_desc: str, store_list: List[Dict]):
            return _generate_store_predictions_impl(base_pred, prod_desc, store_list)
        
        return _generate_wrapper(base_prediction, product_description, stores)
        
    except ImportError:
        return _generate_store_predictions_impl(base_prediction, product_description, stores)
    except Exception as e:
        logger.warning(f"LangSmith tracing failed for store predictions: {e}")
        return _generate_store_predictions_impl(base_prediction, product_description, stores)

def generate_ai_store_predictions(llm_client, product_description: str, stores: List[Dict], similar_products: List[Dict]) -> List[Dict]:
    """Generate AI-powered store-wise predictions using LLM analysis of historical data."""
    if not llm_client:
        # Fallback to simple weather-based predictions if no LLM
        return generate_store_predictions(100, product_description, stores)
    
    try:
        from langsmith import traceable
        
        @traceable(
            name="AI_Store_Predictions",
            run_type="chain",
            tags=["store-predictions", "llm-powered"],
            inputs={
                "product_description": product_description,
                "num_stores": len(stores),
                "similar_products_count": len(similar_products)
            }
        )
        def _ai_store_prediction_wrapper():
            return _generate_ai_store_predictions_impl(llm_client, product_description, stores, similar_products)
        
        return _ai_store_prediction_wrapper()
    except (ImportError, Exception) as e:
        if not isinstance(e, ImportError):
            logger.warning(f"LangSmith tracing failed for AI store predictions: {e}")
        return _generate_ai_store_predictions_impl(llm_client, product_description, stores, similar_products)

def _generate_ai_store_predictions_impl(llm_client, product_description: str, stores: List[Dict], similar_products: List[Dict]) -> List[Dict]:
    """Internal implementation of AI-powered store predictions."""
    store_predictions = []
    
    # Prepare historical data context for LLM
    store_historical_context = _prepare_store_historical_context(stores, similar_products)
    
    # Create LLM prompt with store-specific historical data
    llm_prompt = f"""You are an expert retail analyst predicting sales for "{product_description}" across different stores in India.

HISTORICAL SALES CONTEXT:
{store_historical_context}

TASK: Predict monthly sales quantity for "{product_description}" in each store below, considering:
1. Historical performance of similar products in each city/store
2. Local climate and weather patterns
3. Demographics and income levels
4. Seasonal factors

STORES TO ANALYZE:
"""
    
    # Add store details to prompt
    for store in stores:
        store_info = f"""
Store: {store.get('StoreName', 'Unknown')} ({store.get('StoreID', 'Unknown')})
City: {store.get('City', 'Unknown')}
Climate: {store.get('Climate', 'Unknown')}
Weather Pattern: {store.get('WeatherPattern', 'Unknown')}
Demographics: {store.get('Demographics', 'Unknown')}
Income Level: {store.get('IncomeLevel', 'Unknown')}
Population Density: {store.get('PopulationDensity', 'Unknown')}
"""
        llm_prompt += store_info
    
    llm_prompt += """
Respond in JSON format:
{
  "store_predictions": [
    {
      "store_id": "ST001",
      "store_name": "Store Name",
      "city": "City",
      "predicted_monthly_sales": 150,
      "confidence_score": 0.75,
      "reasoning": "Brief explanation of prediction rationale",
      "recommendation": "BUY|CAUTIOUS|AVOID"
    }
  ],
  "summary": {
    "total_predicted_sales": 2500,
    "average_per_store": 100,
    "high_potential_stores": 5,
    "overall_recommendation": "BUY"
  }
}"""

    try:
        # Get LLM analysis
        llm_result = llm_client.generate(llm_prompt, temperature=0.1, max_tokens=2000)
        llm_response = llm_result.get('response', llm_result.get('content', str(llm_result)))
        
        # Parse JSON response
        import json
        try:
            llm_data = json.loads(llm_response)
            ai_predictions = llm_data.get('store_predictions', [])
            
            # Convert to expected format
            for ai_pred in ai_predictions:
                store_predictions.append({
                    'store_id': ai_pred.get('store_id', ''),
                    'store_name': ai_pred.get('store_name', ''),
                    'city': ai_pred.get('city', ''),
                    'predicted_demand': ai_pred.get('predicted_monthly_sales', 0),
                    'confidence': ai_pred.get('confidence_score', 0.5),
                    'reasoning': ai_pred.get('reasoning', ''),
                    'recommendation': ai_pred.get('recommendation', 'CAUTIOUS'),
                    'climate': _get_store_climate(stores, ai_pred.get('store_id', '')),
                    'weather_factor': 1.0,  # LLM already considered weather
                    'ai_generated': True
                })
            
            logger.info(f"✅ Generated AI predictions for {len(store_predictions)} stores")
            
        except json.JSONDecodeError:
            logger.warning("Failed to parse LLM JSON response, falling back to simple predictions")
            return generate_store_predictions(100, product_description, stores)
            
    except Exception as e:
        logger.warning(f"AI store prediction failed: {e}, falling back to simple predictions")
        return generate_store_predictions(100, product_description, stores)
    
    # Sort by predicted demand (highest first)
    store_predictions.sort(key=lambda x: x['predicted_demand'], reverse=True)
    return store_predictions

def _prepare_store_historical_context(stores: List[Dict], similar_products: List[Dict]) -> str:
    """Prepare historical sales context for LLM analysis."""
    context = "HISTORICAL SALES DATA FOR SIMILAR PRODUCTS:\n\n"
    
    # Group similar products by city/store if available
    city_sales = {}
    
    for product in similar_products[:5]:  # Top 5 similar products
        sales_data = product.get('sales', {})
        if sales_data.get('performance_by_store'):
            for store_perf in sales_data['performance_by_store'][:10]:  # Top 10 stores
                city = store_perf.get('city', 'Unknown')
                if city not in city_sales:
                    city_sales[city] = []
                city_sales[city].append({
                    'product': product.get('name', 'Unknown'),
                    'units_sold': store_perf.get('units_sold', 0),
                    'climate': store_perf.get('climate', 'Unknown')
                })
    
    # Add city-wise sales context
    for city, sales_list in city_sales.items():
        total_units = sum(s['units_sold'] for s in sales_list)
        avg_units = total_units / len(sales_list) if sales_list else 0
        context += f"{city}: {total_units} total units, {avg_units:.0f} avg per product\n"
        for sale in sales_list[:3]:  # Top 3 products per city
            context += f"  - {sale['product']}: {sale['units_sold']} units ({sale['climate']} climate)\n"
        context += "\n"
    
    return context

def _get_store_climate(stores: List[Dict], store_id: str) -> str:
    """Get climate for a specific store."""
    for store in stores:
        if store.get('StoreID') == store_id:
            return store.get('Climate', 'Unknown')
    return 'Unknown'

def _generate_store_predictions_impl(base_prediction: int, product_description: str, stores: List[Dict]) -> List[Dict]:
    """Internal implementation of store predictions."""
    predictions = []
    weather_factors_summary = {}  # Track weather factors for summary
    
    for store in stores:
        store_id = store.get('StoreID', '')
        store_name = store.get('StoreName', '')
        city = store.get('City', '')
        climate = store.get('ClimateTag', '')
        
        # Calculate weather factor (not traced individually to reduce noise)
        weather_factor = get_weather_factor(climate, product_description)
        
        # Track weather factors by climate for summary
        if climate not in weather_factors_summary:
            weather_factors_summary[climate] = []
        weather_factors_summary[climate].append(weather_factor)
        
        # Calculate predicted demand for this store
        predicted_demand = int(base_prediction * weather_factor)
        
        # Determine recommendation
        if predicted_demand >= 1000:
            recommendation = 'HIGH_PRIORITY_BUY'
        elif predicted_demand >= 500:
            recommendation = 'BUY'
        elif predicted_demand >= 200:
            recommendation = 'MODERATE_BUY'
        else:
            recommendation = 'LOW_QUANTITY'
        
        predictions.append({
            'store_id': store_id,
            'store_name': store_name,
            'city': city,
            'climate': climate,
            'predicted_demand': predicted_demand,
            'weather_factor': weather_factor,
            'recommendation': recommendation
        })
    
    # Sort by predicted demand (descending)
    predictions.sort(key=lambda x: x['predicted_demand'], reverse=True)
    
    # Log summary for LangSmith trace visibility
    logger.info(f"Generated predictions for {len(predictions)} stores")
    logger.info(f"Weather factors by climate: {weather_factors_summary}")
    
    return predictions

def show_data_summary(data_summary: Dict[str, Any]):
    """Display ODM data summary on landing screen."""
    st.header("📊 ODM Data Summary")
    
    summary = data_summary.get('summary', {})
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "New Input Products", 
            summary.get('dirty_input_count', 0),
            help="New products from dirty_odm_input.csv"
        )
    
    with col2:
        st.metric(
            "Historical Products", 
            summary.get('historical_products_count', 0),
            help="Unique products from historical dataset"
        )
    
    with col3:
        st.metric(
            "Total Sales Records", 
            summary.get('historical_records_count', 0),
            help="Individual sales transactions"
        )
    
    with col4:
        cleaning_status = "✅ Applied" if summary.get('cleaning_applied', False) else "❌ Raw Data"
        st.metric(
            "Data Cleaning", 
            cleaning_status,
            help="Whether data cleaning was applied"
        )
    
    # Historical products breakdown
    if 'historical_products' in data_summary:
        st.subheader("📈 Historical Products Analysis")
        
        historical_products = data_summary['historical_products']
        if historical_products:
            # Convert to DataFrame for analysis
            hist_df = pd.DataFrame(historical_products)
            
            # Charts
            col1, col2 = st.columns(2)
            
            with col1:
                if 'segment' in hist_df.columns:
                    segment_counts = hist_df['segment'].value_counts()
                    fig_segment = px.pie(
                        values=segment_counts.values, 
                        names=segment_counts.index,
                        title="Products by Segment"
                    )
                    st.plotly_chart(fig_segment, use_container_width=True)
            
            with col2:
                if 'total_quantity_sold' in hist_df.columns:
                    top_products = hist_df.nlargest(10, 'total_quantity_sold')[['style_name', 'total_quantity_sold']]
                    fig_sales = px.bar(
                        x=top_products['total_quantity_sold'], 
                        y=top_products['style_name'],
                        orientation='h',
                        title="Top 10 Products by Sales Volume"
                    )
                    st.plotly_chart(fig_sales, use_container_width=True)
    
    # New input products
    if 'dirty_input_products' in data_summary:
        st.subheader("🆕 New Input Products")
        
        dirty_products = data_summary['dirty_input_products']
        if dirty_products:
            # Show sample of new products
            dirty_df = pd.DataFrame(dirty_products)
            st.dataframe(
                dirty_df[['style_name', 'style_code', 'description']].head(10),
                use_container_width=True
            )

def search_indexed_products(kb: SharedKnowledgeBase, query: str) -> List[Dict[str, Any]]:
    """Search for products in the vector database."""
    if not query.strip():
        return []
    
    logger.info(f"🔍 Searching for: {query}")
    
    try:
        # Search using vector similarity
        results = kb.find_similar_products(
            query_attributes={}, 
            query_description=query,
            top_k=10
        )
        
        logger.info(f"✅ Found {len(results)} similar products")
        return results
        
    except Exception as e:
        logger.error(f"❌ Search failed: {e}")
        return []

def get_status_indicator(recommendation: str, confidence: float, predicted_sales: int, total_store_demand: int = 0) -> Dict[str, str]:
    """Get Red/Yellow/Green status indicator based on recommendation, confidence, and sales predictions."""
    
    # Use total store demand if available, otherwise use predicted_sales
    effective_sales = total_store_demand if total_store_demand > 0 else predicted_sales
    
    if recommendation == 'AVOID' or predicted_sales == 0:
        return {
            'color': 'RED',
            'message': '🔴 RED - DO NOT PROCURE',
            'description': 'High risk, avoid procurement'
        }
    # GREEN: High sales potential (either high monthly avg OR high total store demand)
    elif (recommendation == 'BUY' and confidence >= 0.6 and (predicted_sales >= 100 or effective_sales >= 1000)) or \
         (predicted_sales >= 200 and confidence >= 0.5) or \
         (effective_sales >= 2000):
        return {
            'color': 'GREEN', 
            'message': '🟢 GREEN - RECOMMENDED',
            'description': 'High confidence, good sales potential'
        }
    # YELLOW: Moderate potential
    elif recommendation == 'BUY' and confidence >= 0.5 and (predicted_sales >= 50 or effective_sales >= 500):
        return {
            'color': 'YELLOW',
            'message': '🟡 YELLOW - PROCEED WITH CAUTION',
            'description': 'Moderate confidence, reasonable sales potential'
        }
    elif recommendation == 'CAUTIOUS' or confidence < 0.5:
        return {
            'color': 'YELLOW',
            'message': '🟡 YELLOW - PROCEED WITH CAUTION', 
            'description': 'Moderate risk, requires careful consideration'
        }
    else:
        return {
            'color': 'YELLOW',
            'message': '🟡 YELLOW - PROCEED WITH CAUTION',
            'description': 'Uncertain outcome, moderate risk'
        }

def are_colors_similar(color1: str, color2: str) -> bool:
    """Check if two colors are similar (e.g., red and light red, but not red and blue)."""
    color1_lower = color1.lower().strip()
    color2_lower = color2.lower().strip()
    
    # Exact match
    if color1_lower == color2_lower:
        return True
    
    # Extract base colors (remove modifiers like "light", "dark", "bright", etc.)
    base_colors = {
        'red': ['red', 'light red', 'dark red', 'bright red', 'deep red', 'crimson', 'maroon', 'burgundy', 'scarlet'],
        'blue': ['blue', 'light blue', 'dark blue', 'bright blue', 'navy', 'sky blue', 'royal blue', 'indigo'],
        'green': ['green', 'light green', 'dark green', 'bright green', 'olive', 'emerald', 'forest green'],
        'yellow': ['yellow', 'light yellow', 'dark yellow', 'bright yellow', 'gold', 'mustard'],
        'orange': ['orange', 'light orange', 'dark orange', 'bright orange', 'peach', 'coral'],
        'purple': ['purple', 'light purple', 'dark purple', 'bright purple', 'violet', 'lavender'],
        'pink': ['pink', 'light pink', 'dark pink', 'bright pink', 'hot pink', 'rose', 'fuchsia'],
        'black': ['black', 'charcoal', 'ebony'],
        'white': ['white', 'ivory', 'cream', 'off-white', 'beige'],
        'brown': ['brown', 'light brown', 'dark brown', 'tan', 'khaki', 'camel'],
        'gray': ['gray', 'grey', 'light gray', 'dark gray', 'silver', 'charcoal'],
    }
    
    # Find base color for each
    base1 = None
    base2 = None
    
    for base, variants in base_colors.items():
        if any(variant in color1_lower for variant in variants):
            base1 = base
        if any(variant in color2_lower for variant in variants):
            base2 = base
    
    # If both have the same base color, they're similar
    if base1 and base2 and base1 == base2:
        return True
    
    # If one color contains the other (e.g., "red" in "light red")
    if base1 and base1 in color2_lower:
        return True
    if base2 and base2 in color1_lower:
        return True
    
    return False

def extract_product_attributes(description: str, similar_products: List[Dict]) -> Dict[str, Any]:
    """Extract product attributes from description and match with sales data fields."""
    description_lower = description.lower()
    
    # Extract color
    color_keywords = ['red', 'blue', 'black', 'white', 'green', 'yellow', 'purple', 'pink', 'brown', 'gray', 'grey', 
                     'orange', 'navy', 'beige', 'maroon', 'teal', 'crimson', 'burgundy', 'olive', 'emerald', 
                     'gold', 'mustard', 'peach', 'coral', 'violet', 'lavender', 'rose', 'fuchsia', 'charcoal', 
                     'ivory', 'cream', 'tan', 'khaki', 'camel', 'silver', 'indigo', 'sky', 'royal', 'forest']
    desc_color = None
    for color in color_keywords:
        if color in description_lower:
            desc_color = color
            break
    
    # Extract category (Brick)
    brick_keywords = ['jeans', 'pants', 'shirt', 'dress', 't-shirt', 'tshirt', 'top', 'jacket', 'kurta', 'shorts', 
                     'jogger', 'cargo', 'hoodie', 'thermal', 'leggings', 'dungaree', 'pyjama', 'lehenga', 
                     'tracksuit', 'suit', 'blazer', 'coat', 'sweater', 'trousers']
    desc_brick = None
    for brick in brick_keywords:
        if brick in description_lower:
            desc_brick = brick
            break
    
    # Extract segment
    desc_segment = None
    if 'men' in description_lower or 'mens' in description_lower:
        desc_segment = 'Mens'
    elif 'women' in description_lower or 'womens' in description_lower:
        desc_segment = 'Womens'
    elif 'kid' in description_lower or 'boys' in description_lower:
        desc_segment = 'Kids'
    elif 'girl' in description_lower:
        desc_segment = 'Girls'
    
    return {
        'color': desc_color,
        'brick': desc_brick,
        'segment': desc_segment
    }

def validate_product_relevance(llm_client: LLMClient, query_product: str, vector_results: List[Dict]) -> Dict[str, Any]:
    """Validate if vector search results are actually relevant for the customer query."""
    try:
        from langsmith import traceable
        
        @traceable(
            name="Product_Relevance_Validation",
            run_type="chain", 
            tags=["relevance", "product-validation"]
        )
        def _validate_wrapper():
            return _validate_relevance_impl(llm_client, query_product, vector_results)
        
        return _validate_wrapper()
    except ImportError:
        return _validate_relevance_impl(llm_client, query_product, vector_results)
    except Exception as e:
        logger.warning(f"LangSmith tracing failed for relevance validation: {e}")
        return _validate_relevance_impl(llm_client, query_product, vector_results)

def _validate_relevance_impl(llm_client: LLMClient, query_product: str, vector_results: List[Dict]) -> Dict[str, Any]:
    """Internal implementation of product relevance validation."""
    import json
    import re
    
    if not vector_results or not llm_client:
        return {
            'is_relevant': len(vector_results) > 0,
            'relevant_products': vector_results,
            'reason': 'No LLM available for validation' if not llm_client else 'No products to validate'
        }
    
    # Prepare product summary for validation
    products_summary = []
    for i, product in enumerate(vector_results[:10]):  # Check top 10 products
        attrs = product.get('attributes', {})
        metadata = product.get('metadata', {})
        
        # Get product details
        name = product.get('name', 'Unknown')
        
        # Get category/brick
        category = (attrs.get('brick') or attrs.get('Brick') or 
                   metadata.get('brick') or metadata.get('Brick') or 
                   attrs.get('category') or 'Unknown')
        
        # Get color
        color = (attrs.get('colour') or attrs.get('Colour') or attrs.get('color') or
                metadata.get('colour') or metadata.get('Colour') or metadata.get('color') or 'Unknown')
        
        # Get segment
        segment = (attrs.get('segment') or attrs.get('Segment') or
                  metadata.get('segment') or metadata.get('Segment') or 'Unknown')
        
        products_summary.append(f"{i+1}. {name} - Category: {category}, Color: {color}, Segment: {segment}")
    
    validation_prompt = f"""You are a product categorization expert. A customer is asking for predictions about "{query_product}".

The vector search returned these similar products:
{chr(10).join(products_summary)}

Your task: Determine if these products are relevant enough to make sales predictions for "{query_product}".

STRICT VALIDATION RULES:
1. Products must be in the SAME CATEGORY (e.g., don't use jeans data to predict dress sales)
2. Different colors within same category are acceptable (e.g., use black dress data for red dress prediction)
3. Different segments within same category are acceptable (e.g., use women's dress for girl's dress)
4. At least 3 relevant products should be available for reliable prediction

Respond in this exact JSON format:
{{
    "is_relevant": true/false,
    "reason": "detailed explanation of your decision",
    "relevant_product_indices": [list of relevant product numbers from 1-{len(products_summary)}],
    "category_match": "exact category that matches the query",
    "confidence": 0.0-1.0
}}

Be strict - it's better to reject and ask for manual input than give wrong predictions based on irrelevant products."""
    
    try:
        response = llm_client.generate(validation_prompt)
        logger.info(f"🤖 LLM validation response: {response[:200]}...")
        
        # Parse JSON response - extract JSON from response if wrapped in text
        
        # Try to extract JSON from the response
        json_match = re.search(r'\{.*\}', response.strip(), re.DOTALL)
        if json_match:
            json_str = json_match.group()
            logger.info(f"📄 Extracted JSON: {json_str}")
            validation_result = json.loads(json_str)
        else:
            # Fallback to trying the whole response
            validation_result = json.loads(response.strip())
        logger.info(f"📝 Parsed validation result: {validation_result}")
        
        # Extract relevant products based on indices
        relevant_indices = validation_result.get('relevant_product_indices', [])
        relevant_products = []
        
        for idx in relevant_indices:
            if 1 <= idx <= len(vector_results):
                relevant_products.append(vector_results[idx - 1])
        
        logger.info(f"✅ Relevance validation: {validation_result['is_relevant']} - {validation_result['reason']}")
        logger.info(f"   Selected {len(relevant_products)} relevant products from {len(vector_results)} total")
        
        return {
            'is_relevant': validation_result['is_relevant'] and len(relevant_products) >= 2,  # Reduced from 3 to 2
            'relevant_products': relevant_products,
            'reason': validation_result['reason'],
            'confidence': validation_result.get('confidence', 0.0),
            'category_match': validation_result.get('category_match', 'Unknown')
        }
        
    except json.JSONDecodeError as e:
        logger.warning(f"Failed to parse relevance validation JSON response: {e}")
        logger.warning(f"Raw LLM response was: {response}")
        # Fallback to simple category matching
        return _simple_relevance_check(query_product, vector_results)
    except Exception as e:
        logger.warning(f"Relevance validation failed: {e}")
        return _simple_relevance_check(query_product, vector_results)

def _simple_relevance_check(query_product: str, vector_results: List[Dict]) -> Dict[str, Any]:
    """Simple fallback relevance check without LLM."""
    query_lower = query_product.lower()
    
    # Extract expected category from query
    expected_category = None
    if 'dress' in query_lower:
        expected_category = 'dress'
    elif 'shirt' in query_lower:
        expected_category = 'shirt'
    elif 'jeans' in query_lower or 'pant' in query_lower:
        expected_category = 'jeans'
    elif 't-shirt' in query_lower or 'tee' in query_lower:
        expected_category = 't-shirt'
    
    if not expected_category:
        # If we can't determine category, allow all products
        return {
            'is_relevant': True,
            'relevant_products': vector_results,
            'reason': 'Could not determine specific category from query'
        }
    
    # Filter products by category
    relevant_products = []
    for product in vector_results:
        attrs = product.get('attributes', {})
        metadata = product.get('metadata', {})
        
        category = (attrs.get('brick', '') or attrs.get('Brick', '') or 
                   metadata.get('brick', '') or metadata.get('Brick', '') or
                   attrs.get('category', '') or metadata.get('category', '')).lower()
        
        # More flexible category matching
        category_match = False
        if expected_category == 'jeans':
            category_match = any(term in category for term in ['jeans', 'denim', 'pant', 'trouser', 'jogger'])
        elif expected_category == 'dress':
            category_match = any(term in category for term in ['dress', 'frock', 'gown'])
        elif expected_category == 'shirt':
            category_match = any(term in category for term in ['shirt', 'blouse'])
        elif expected_category == 't-shirt':
            category_match = any(term in category for term in ['t-shirt', 'tee', 'top'])
        else:
            category_match = expected_category in category
        
        if category_match:
            relevant_products.append(product)
    
    return {
        'is_relevant': len(relevant_products) >= 1,  # At least 1 product needed for simple check
        'relevant_products': relevant_products,
        'reason': f"Found {len(relevant_products)} products matching '{expected_category}' category"
    }

def analyze_product_viability(product_description: str, similar_products: List[Dict]) -> Dict[str, Any]:
    """Analyze if a product combination is viable based on actual sales data."""
    # Wrap with LangSmith tracing
    try:
        from langsmith import traceable
        
        # Use wrapper function to ensure proper trace nesting
        @traceable(
            name="Analyze_Product_Viability", 
            run_type="chain", 
            tags=["viability", "product-analysis"]
        )
        def _analyze_wrapper():
            return _analyze_viability_impl(product_description, similar_products)
        
        return _analyze_wrapper()
    except ImportError:
        # If langsmith not available, use direct call
        return _analyze_viability_impl(product_description, similar_products)
    except Exception as e:
        logger.warning(f"LangSmith tracing failed for viability analysis: {e}")
        return _analyze_viability_impl(product_description, similar_products)

def _analyze_viability_impl(product_description: str, similar_products: List[Dict]) -> Dict[str, Any]:
    """Internal implementation of viability analysis."""
    description_lower = product_description.lower()
    
    # Extract attributes from description
    desc_attrs = extract_product_attributes(product_description, similar_products)
    desc_color = desc_attrs.get('color')
    desc_brick = desc_attrs.get('brick')
    desc_segment = desc_attrs.get('segment')
    
    # Check sales data for exact or similar matches
    exact_matches = 0  # Exact color-category matches
    similar_color_matches = 0  # Similar color (e.g., red vs light red) matches
    category_only_matches = 0  # Same category, different color
    products_with_sales = 0  # Products that actually have sales data
    
    for product in similar_products:
        attrs = product.get('attributes', {})
        # Try multiple field name variations (case insensitive) - check lowercase 'colour' first as that's how data is stored
        product_color = str(attrs.get('colour', attrs.get('Colour', attrs.get('color', attrs.get('Color', ''))))).lower().strip()
        product_brick = str(attrs.get('brick', attrs.get('Brick', attrs.get('category', attrs.get('Category', ''))))).lower().strip()
        product_segment = str(attrs.get('segment', attrs.get('Segment', ''))).lower().strip()
        
        # Also check metadata for these fields if not found in attributes
        metadata = product.get('metadata', {})
        if not product_color and metadata:
            product_color = str(metadata.get('colour', metadata.get('Colour', metadata.get('color', metadata.get('Color', ''))))).lower().strip()
        if not product_brick and metadata:
            product_brick = str(metadata.get('brick', metadata.get('Brick', metadata.get('category', metadata.get('Category', ''))))).lower().strip()
        if not product_segment and metadata:
            product_segment = str(metadata.get('segment', metadata.get('Segment', ''))).lower().strip()
        
        # Also check if attributes are stored directly in product (not nested)
        if not product_color:
            product_color = str(product.get('colour', product.get('Colour', product.get('color', '')))).lower().strip()
        if not product_brick:
            product_brick = str(product.get('brick', product.get('Brick', product.get('category', '')))).lower().strip()
        if not product_segment:
            product_segment = str(product.get('segment', product.get('Segment', ''))).lower().strip()
        
        # Debug logging for first few products
        if len([p for p in similar_products if similar_products.index(p) < 3]) > 0 and similar_products.index(product) < 3:
            logger.debug(f"Product {similar_products.index(product)}: color='{product_color}', brick='{product_brick}', segment='{product_segment}'")
        
        # Check if product has sales data
        sales_data = product.get('sales', {})
        has_sales = sales_data.get('total_units', 0) > 0
        if has_sales:
            products_with_sales += 1
        
        # Match category (Brick) - be more flexible
        brick_match = False
        if desc_brick:
            if product_brick:
                # Direct match or one contains the other
                if desc_brick in product_brick or product_brick in desc_brick:
                    brick_match = True
            # Also check if description contains the brick keyword (for cases where Brick field might be empty)
            if not brick_match and desc_brick in description_lower:
                # If we're looking for jeans and product description/name contains jeans, consider it a match
                product_name = str(product.get('name', '')).lower()
                product_desc = str(product.get('description', '')).lower()
                if desc_brick in product_name or desc_brick in product_desc:
                    brick_match = True
        else:
            # No specific brick mentioned, so any product is a potential match
            brick_match = True
        
        # Match segment if specified - be more lenient
        segment_match = True  # Default to True if not specified
        if desc_segment and product_segment:
            # Check if segments match (case insensitive)
            desc_seg_lower = desc_segment.lower()
            prod_seg_lower = product_segment.lower()
            # Allow partial matches (e.g., "Mens" matches "Men")
            if desc_seg_lower in prod_seg_lower or prod_seg_lower in desc_seg_lower:
                segment_match = True
            else:
                segment_match = False
        
        if not brick_match:
            continue
        if desc_segment and not segment_match:
            continue
        
        category_only_matches += 1
        
        # Match color
        if desc_color and product_color:
            # Exact color match (case insensitive substring match)
            if desc_color in product_color or product_color in desc_color:
                exact_matches += 1
                logger.debug(f"Exact color match: '{desc_color}' in '{product_color}' for {product_brick}")
            # Similar color match (e.g., red ≈ light red, but red ≠ blue)
            elif are_colors_similar(desc_color, product_color):
                similar_color_matches += 1
                logger.debug(f"Similar color match: '{desc_color}' similar to '{product_color}' for {product_brick}")
        elif not desc_color:
            # No color specified, so any product with matching category is considered a match
            exact_matches += 1
    
    # Log matching results for debugging
    logger.info(f"Viability check for '{product_description}': color={desc_color}, brick={desc_brick}, segment={desc_segment}")
    logger.info(f"Matches: exact={exact_matches}, similar={similar_color_matches}, category_only={category_only_matches}, with_sales={products_with_sales}, total_products={len(similar_products)}")
    
    # Decision logic based on sales data
    if desc_color and desc_brick:
        # Specific color and category mentioned
        if exact_matches > 0:
            # Found exact color-category matches in sales data
            logger.info(f"✅ Found {exact_matches} exact matches for {desc_color} {desc_brick}")
            return {
                'viable': True,
                'reason': f'Found {exact_matches} exact {desc_color.title()} {desc_brick} products in sales data',
                'risk_level': 'LOW',
                'market_acceptance': 'PROVEN'
            }
        elif similar_color_matches > 0:
            # Found similar color matches (e.g., light red when searching for red)
            logger.info(f"✅ Found {similar_color_matches} similar color matches for {desc_color} {desc_brick}")
            return {
                'viable': True,
                'reason': f'Found {similar_color_matches} similar color {desc_brick} products in sales data (e.g., {desc_color} variants)',
                'risk_level': 'LOW',
                'market_acceptance': 'ACCEPTABLE'
            }
        elif category_only_matches > 0:
            # Found category matches but different colors (e.g., blue jeans when searching for red jeans)
            logger.warning(f"❌ No {desc_color} {desc_brick} found. Found {category_only_matches} {desc_brick} in different colors")
            return {
                'viable': False,
                'reason': f'No {desc_color.title()} {desc_brick} found in sales data. Found {category_only_matches} {desc_brick} products but all in different colors.',
                'risk_level': 'HIGH',
                'market_acceptance': 'UNPROVEN'
            }
        else:
            # No category matches at all
            logger.warning(f"❌ No {desc_brick} products found in sales data")
            return {
                'viable': False,
                'reason': f'No {desc_brick} products found in sales data',
                'risk_level': 'HIGH',
                'market_acceptance': 'UNPROVEN'
            }
    elif desc_brick:
        # Only category mentioned, no specific color
        if category_only_matches > 0:
            return {
                'viable': True,
                'reason': f'Found {category_only_matches} {desc_brick} products in sales data',
                'risk_level': 'LOW' if products_with_sales > 0 else 'MEDIUM',
                'market_acceptance': 'ACCEPTABLE'
            }
        else:
            return {
                'viable': False,
                'reason': f'No {desc_brick} products found in sales data',
                'risk_level': 'HIGH',
                'market_acceptance': 'UNPROVEN'
            }
    else:
        # Generic product, check if we have any similar products
        if len(similar_products) >= 5:
            return {
                'viable': True,
                'reason': f'Found {len(similar_products)} similar products in sales data',
                'risk_level': 'MEDIUM',
                'market_acceptance': 'ACCEPTABLE'
            }
        else:
            return {
                'viable': False,
                'reason': 'Very limited historical evidence of similar products',
                'risk_level': 'HIGH',
                'market_acceptance': 'UNPROVEN'
            }

def predict_product_sales_comprehensive(kb: SharedKnowledgeBase, llm_client: LLMClient, product_description: str, stores: List[Dict]) -> Dict[str, Any]:
    """Comprehensive prediction with single LLM call including store-wise analysis."""
    try:
        from langsmith import traceable
        
        @traceable(
            name="Comprehensive_Sales_Prediction",
            run_type="chain",
            tags=["sales-prediction", "comprehensive", "store-wise"],
            inputs={
                "product_description": product_description,
                "num_stores": len(stores) if stores else 0
            }
        )
        def _comprehensive_prediction_wrapper():
            return _predict_comprehensive_impl(kb, llm_client, product_description, stores)
        
        return _comprehensive_prediction_wrapper()
    except (ImportError, Exception) as e:
        if not isinstance(e, ImportError):
            logger.warning(f"LangSmith tracing failed: {e}")
        return _predict_comprehensive_impl(kb, llm_client, product_description, stores)

def _predict_comprehensive_impl(kb: SharedKnowledgeBase, llm_client: LLMClient, product_description: str, stores: List[Dict]) -> Dict[str, Any]:
    """Single comprehensive prediction implementation."""
    logger.info(f"🔮 Comprehensive prediction for: {product_description}")
    
    try:
        # Vector search for similar products
        try:
            from langsmith import traceable
            
            @traceable(
                name="Vector_Search",
                run_type="retriever", 
                tags=["vector-db", "similarity-search"]
            )
            def _vector_search():
                return kb.find_similar_products(
                    query_attributes={},
                    query_description=product_description,
                    top_k=15
                )
            
            similar_products = _vector_search()
        except (ImportError, Exception) as e:
            if not isinstance(e, ImportError):
                logger.warning(f"Vector search tracing failed: {e}")
            similar_products = kb.find_similar_products(
                query_attributes={},
                query_description=product_description,
                top_k=15
            )
        
        if not similar_products:
            return _create_fallback_prediction(product_description)
        
        # Validate product relevance before proceeding
        relevance_validation = validate_product_relevance(llm_client, product_description, similar_products)
        logger.info(f"🔍 Relevance validation result: is_relevant={relevance_validation['is_relevant']}, products={len(relevance_validation.get('relevant_products', []))}")
        if not relevance_validation['is_relevant']:
            logger.warning(f"⚠️ Vector search results not relevant for '{product_description}': {relevance_validation['reason']}")
            return _create_fallback_prediction(product_description, relevance_validation['reason'])
        
        # Use only the validated relevant products
        validated_products = relevance_validation['relevant_products']
        logger.info(f"✅ Using {len(validated_products)} validated relevant products out of {len(similar_products)} found")
        
        # Analyze viability with validated products
        viability = analyze_product_viability(product_description, validated_products)
        if not viability['viable']:
            return _create_avoid_prediction(product_description, viability, validated_products)
        
        # Prepare comprehensive LLM prompt with everything
        products_with_sales = [p for p in validated_products if p.get('sales', {}).get('total_units', 0) > 0]
        
        if not llm_client:
            # Fallback to statistical analysis
            return _create_statistical_prediction(product_description, validated_products, products_with_sales, stores)
        
        # Create comprehensive LLM prompt
        comprehensive_prompt = _create_comprehensive_llm_prompt(
            product_description, validated_products, products_with_sales, stores, viability
        )
        
        # Single LLM call with comprehensive analysis
        try:
            from langsmith import traceable
            
            @traceable(
                name="Comprehensive_LLM_Analysis",
                run_type="llm",
                tags=["openai", "comprehensive-analysis"],
                inputs={
                    "product_description": product_description,
                    "similar_products_count": len(similar_products),
                    "stores_count": len(stores) if stores else 0
                }
            )
            def _comprehensive_llm_call():
                return llm_client.generate(comprehensive_prompt, temperature=0.1, max_tokens=3000)
            
            llm_result = _comprehensive_llm_call()
        except (ImportError, Exception) as e:
            if not isinstance(e, ImportError):
                logger.warning(f"LLM analysis tracing failed: {e}")
            llm_result = llm_client.generate(comprehensive_prompt, temperature=0.1, max_tokens=3000)
        
        # Parse comprehensive response
        llm_response = llm_result.get('response', llm_result.get('content', str(llm_result)))
        return _parse_comprehensive_response(llm_response, product_description, validated_products, stores)
        
    except Exception as e:
        logger.error(f"❌ Comprehensive prediction failed: {e}")
        return _create_fallback_prediction(product_description, error=str(e))

def _load_historical_sales_by_city():
    """Load historical sales performance by city from ODM dataset."""
    try:
        import pandas as pd
        
        # Load the ODM historical dataset
        historical_file = "data/sample/odm_historical_dataset_5000.csv"
        df = pd.read_csv(historical_file)
        
        # Group by city and calculate performance metrics
        city_stats = df.groupby(['City', 'ClimateTag']).agg({
            'QuantitySold': 'sum',
            'TotalSales': 'sum',
            'StyleCode': 'nunique'
        }).reset_index()
        
        # Create city performance dictionary
        city_performance = {}
        for _, row in city_stats.iterrows():
            city = row['City']
            city_performance[city] = {
                'total_units': int(row['QuantitySold']),
                'total_revenue': float(row['TotalSales']),
                'product_count': int(row['StyleCode']),
                'climate': row['ClimateTag']
            }
        
        logger.info(f"✅ Loaded historical performance for {len(city_performance)} cities")
        return city_performance
        
    except Exception as e:
        logger.warning(f"Failed to load city performance data: {e}")
        return {}


def _load_city_counts_by_similar_products(product_specific_city_data: list):
    """
    Take product_specific_city_data list, extract product_ids,
    match directly with StyleCode from CSV,
    return only city-wise counts.
    """
    try:
        import pandas as pd
        print(f"Similar data to search city wise {product_specific_city_data}")
        
        historical_file = "data/sample/odm_historical_dataset_5000.csv"
        df = pd.read_csv(historical_file)

        print(f"1")
        # Extract product IDs
        similar_ids = [
            p["attributes"]["style_code"]
            for p in product_specific_city_data
            if p.get("attributes") and p["attributes"].get("style_code")]        
        print(f"2")
        print(f"similar_ids {similar_ids}")

        if not similar_ids:
            return {}

        # Match product IDs against StyleCode
        matched_df = df[df["StyleCode"].astype(str).isin(similar_ids)]

        print(f"Similar data {matched_df}")

        if matched_df.empty:
            print(f"Similar data empty")
            return {}
        
        

        # Group by city, just count purchases
        city_stats = matched_df.groupby("City").agg({
            "QuantitySold": "sum",
            "StyleCode": "nunique"
        }).reset_index()

        city_counts = {
            row["City"]: {
                "similar_average_units": int(row["QuantitySold"]/row["StyleCode"]),
                "unique_products_sold": int(row["StyleCode"])
            }
            for _, row in city_stats.iterrows()
        }

        return city_counts

    except Exception as e:
        print(f"Error computing city counts: {e}")
        return {}


def _load_product_specific_city_performance(product_description: str):
    """Load city performance data specifically for the requested product type."""
    try:
        import pandas as pd
        
        # Load ODM dataset
        historical_file = "data/sample/odm_historical_dataset_5000.csv"
        df = pd.read_csv(historical_file)
        
        # Extract product attributes from description
        desc_lower = product_description.lower()
        
        # First, try to find exact matches with all attributes
        product_filters = []
        matched_attributes = []
        
        # Color matching - map common colors to available dataset colors
        color_mapping = {
            'red': ['maroon', 'coral'],
            'blue': ['navy', 'light blue', 'sky blue', 'denim blue', 'baby blue', 'dark blue'],
            'green': ['forest green', 'deep green', 'bottle green', 'neon green', 'olive', 'emerald'],
            'black': ['black', 'charcoal'],
            'white': ['white', 'ivory'],
            'yellow': ['yellow', 'mustard'],
            'pink': ['peach', 'fuchsia'],
            'grey': ['grey', 'charcoal', 'stone'],
            'brown': ['khaki', 'beige']
        }
        
        matched_colors = []
        for search_color, dataset_colors in color_mapping.items():
            if search_color in desc_lower:
                matched_colors.extend(dataset_colors)
                break
        
        if matched_colors:
            color_filter = df['Colour'].str.lower().str.contains('|'.join(matched_colors), na=False)
            product_filters.append(color_filter)
            matched_attributes.append(f"color ({', '.join(matched_colors)})")
        
        # Category matching
        category_filter = None
        if 'dress' in desc_lower:
            category_filter = df['Brick'].str.lower().str.contains('dress|frock|gown', na=False)
            matched_attributes.append("dress category")
        elif 'shirt' in desc_lower:
            category_filter = df['Brick'].str.lower().str.contains('shirt|top', na=False)
            matched_attributes.append("shirt category")
        elif 'jeans' in desc_lower or 'pant' in desc_lower:
            category_filter = df['Brick'].str.lower().str.contains('jeans|pant|trouser', na=False)
            matched_attributes.append("pants/jeans category")
        elif 't-shirt' in desc_lower or 'tee' in desc_lower:
            category_filter = df['Brick'].str.lower().str.contains('t-shirt|tee|top', na=False)
            matched_attributes.append("t-shirt category")
        
        if category_filter is not None:
            product_filters.append(category_filter)
        
        # Material matching - be more flexible
        if 'cotton' in desc_lower:
            # Include cotton blends
            material_filter = df['Fabric'].str.lower().str.contains('cotton', na=False)
            product_filters.append(material_filter)
            matched_attributes.append("cotton fabric")
        
        # Try different levels of filtering
        filtered_df = None
        filter_description = ""
        
        if len(product_filters) >= 2:
            # Try with all filters first
            combined_filter = product_filters[0]
            for filter_condition in product_filters[1:]:
                combined_filter = combined_filter & filter_condition
            filtered_df = df[combined_filter]
            filter_description = f"all attributes ({', '.join(matched_attributes)})"
            
            # If no results, try with just category
            if len(filtered_df) == 0 and category_filter is not None:
                filtered_df = df[category_filter]
                filter_description = f"category only ({matched_attributes[-1]})"
                
        elif len(product_filters) == 1:
            filtered_df = df[product_filters[0]]
            filter_description = matched_attributes[0] if matched_attributes else "single filter"
        
        if filtered_df is not None and len(filtered_df) > 0:
            # Group by city and calculate performance
            city_stats = filtered_df.groupby('City').agg({
                'QuantitySold': 'sum',
                'TotalSales': 'sum',
                'StyleCode': 'nunique',
                'StyleName': lambda x: list(x.unique())[:3]  # Sample product names
            }).reset_index()
            
            city_performance = {}
            sample_products = []
            
            for _, row in city_stats.iterrows():
                city = row['City']
                total_units = int(row['QuantitySold'])
                product_count = int(row['StyleCode'])
                
                city_performance[city] = {
                    'total_units': total_units,
                    'total_revenue': float(row['TotalSales']),
                    'product_count': product_count,
                    'avg_units_per_product': total_units / product_count if product_count > 0 else 0,
                    'sample_products': row['StyleName']
                }
                
                # Collect sample products for logging
                sample_products.extend(row['StyleName'])
            
            # Remove duplicates from sample products
            sample_products = list(set(sample_products))[:5]
            
            logger.info(f"✅ Found product-specific data for {len(city_performance)} cities, {len(filtered_df)} matching products")
            logger.info(f"   Filter used: {filter_description}")
            logger.info(f"   Sample products: {', '.join(sample_products)}")
            return city_performance
        
        logger.info(f"❌ No specific product data found for: {product_description}")
        logger.info(f"   Tried matching: {', '.join(matched_attributes) if matched_attributes else 'no attributes matched'}")
        return {}
        
    except Exception as e:
        logger.warning(f"Failed to load product-specific city data: {e}")
        return {}

def _create_comprehensive_llm_prompt(product_description: str, similar_products: List[Dict], products_with_sales: List[Dict], stores: List[Dict], viability: Dict) -> str:
    """Create comprehensive LLM prompt with all data."""
    
    # Load historical sales data directly from ODM dataset
    city_performance = _load_historical_sales_by_city()
    #product_specific_city_data = _load_product_specific_city_performance(product_description)
    #print(similar_products)
    #product_specific_city_data = similar_products
    product_specific_city_data = _load_city_counts_by_similar_products(similar_products)


    
    # Prepare historical context with product-specific performance
    historical_context = ""

    
    
    # First show similar products found in vector DB
    for product in products_with_sales[:5]:
        sales = product.get('sales', {})
        historical_context += f"Product: {product.get('name', 'Unknown')}\n"
        historical_context += f"  Total Units: {sales.get('total_units', 0)}\n"
        historical_context += f"  Avg Monthly: {sales.get('avg_monthly_units', 0):.1f} units\n"
        historical_context += "\n"
    
    # Add product-specific city performance if available
    # if product_specific_city_data:
    #     historical_context += f"PRODUCT-SPECIFIC HISTORICAL SALES BY CITY ({product_description}):\n"
    #     for city, data in product_specific_city_data:
    #         climate = city_performance.get(city, {}).get('climate', 'Unknown')
    #         historical_context += f"{city} ({climate} climate):\n"
    #         historical_context += f"  - {data['total_units']} units sold across {data['product_count']} similar products\n"
    #         #historical_context += f"  - Average {data['avg_units_per_product']:.1f} units per similar product\n"
    #         historical_context += f"  - Revenue: ₹{data['total_revenue']:,.0f}\n\n"


    if product_specific_city_data:
        for city, stats in product_specific_city_data.items():
            historical_context += (
                f"{city}:\n"
                f"  - {stats['similar_average_units']} units sold in last 3 months\n"
                f"  - {stats['unique_products_sold']} similar SKUs bought\n\n"
            )


        historical_context += "\n"
    
    

    
    # Add overall city-wise performance for reference
    if city_performance:
        historical_context += "OVERALL CITY PERFORMANCE  in last 3 months (ALL PRODUCT CATEGORIES):\n"
        for city, perf_data in city_performance.items():
            total_units = perf_data.get('total_units', 0)
            product_count = perf_data.get('product_count', 0)
            climate = perf_data.get('climate', 'Unknown')
            
            avg_units_per_product = total_units / product_count if product_count > 0 else 0
            
            historical_context += f"{city} ({climate}): {total_units:,} total units, avg {avg_units_per_product:.0f} per product\n"
        historical_context += "\n"
    
    # Prepare store information with enhanced context
    store_context = "TARGET STORES FOR ANALYSIS:\n"
    for store in stores:
        city = store.get('City', 'Unknown')
        climate = store.get('ClimateTag', 'Unknown')
        store_id = store.get('StoreID', 'Unknown')
        store_name = store.get('StoreName', 'Unknown')
        
        # Add historical performance for this city if available
        city_history = ""
        if city in city_performance:
            perf = city_performance[city]
            avg_units = perf['total_units'] / perf['product_count'] if perf['product_count'] > 0 else 0
            city_history = f" | Historical: {perf['total_units']} units across {perf['product_count']} products (avg: {avg_units:.1f} per product)"
        
        store_context += f"""
Store: {store_name} ({store_id})
City: {city} | Climate: {climate}{city_history}
"""

    # Enhanced viability section
    viability_details = f"""
PRODUCT VIABILITY ASSESSMENT:
- Viable: {viability['viable']}
- Risk Level: {viability['risk_level']}  
- Market Acceptance: {viability['market_acceptance']}
- Analysis: {viability['reason']}
- Exact Matches Found: {viability.get('exact_matches', 0)} (same color + category)
- Category Matches: {viability.get('category_only_matches', 0)} (same category, different colors)
- Products with Sales Data: {viability.get('products_with_sales', 0)}
"""

    prompt = f"""You are an expert retail procurement analyst. Analyze and predict sales for "{product_description}" across {len(stores)} stores in India.

HISTORICAL SALES DATA FOR SIMILAR PRODUCTS:
{historical_context}
{viability_details}

{store_context}

ANALYSIS GUIDELINES:
- Use historical city performance data shown above to inform predictions
- Consider climate factors (e.g., cotton shirts perform better in hot climates)
- Account for city size and market potential
- Make realistic monthly sales predictions based on actual historical data
- Total across all stores should align with overall market assessment

TASK: Provide comprehensive analysis with:
1. Overall monthly sales prediction (aggregate across all stores)
2. Confidence assessment based on data quality
3. Store-wise monthly predictions for each of the {len(stores)} stores listed
4. Overall procurement recommendation

Respond in this JSON format:
{{
  "overall_analysis": {{
    "predicted_monthly_sales": <SUM_OF_ALL_STORE_PREDICTIONS>,
    "confidence": <AVERAGE_CONFIDENCE_OF_ALL_STORES>,
    "procurement_recommendation": "BUY",
    "reasoning": "Your detailed analysis based on historical data and store performance...",
    "status_color": "GREEN",
    "status_message": "🟢 GREEN - STRONG BUY SIGNAL"
  }},
  "store_predictions": [
    {{
      "store_id": "ST01",
      "store_name": "Store 1",
      "city": "Delhi", 
      "predicted_monthly_sales": <YOUR_CALCULATED_VALUE>,
      "confidence": <0.0_TO_1.0_BASED_ON_DATA_QUALITY>,
      "reasoning": "Your analysis for this specific store...",
      "recommendation": "BUY"
    }}
  ],
  "summary": {{
    "total_monthly_sales": <SAME_AS_OVERALL_PREDICTED_MONTHLY_SALES>,
    "high_potential_stores": <COUNT_OF_STORES_WITH_PREDICTION_>_50>,
    "buy_recommendation_stores": <COUNT_OF_STORES_WITH_BUY_RECOMMENDATION>,
    "avoid_stores": <COUNT_OF_STORES_WITH_AVOID_RECOMMENDATION>
  }}
}}

CRITICAL CALCULATION REQUIREMENTS:
1. overall_analysis.predicted_monthly_sales = SUM of all store_predictions[].predicted_monthly_sales
2. overall_analysis.confidence = AVERAGE of all store_predictions[].confidence  
3. summary.total_monthly_sales = SAME as overall_analysis.predicted_monthly_sales
4. summary counts must match actual store predictions provided
5. Provide predictions for ALL {len(stores)} stores listed above
6. Use actual city names and store IDs from the store data provided
7. Base your predictions on the historical data and climate factors provided
8. DO NOT use hardcoded example values - calculate everything from your analysis"""
    
    return prompt

def _clean_llm_json_response(json_text: str) -> str:
    """Clean and fix common JSON formatting issues from LLM responses."""
    import re
    
    # Remove any leading/trailing whitespace
    json_text = json_text.strip()
    
    # Remove any text before the first {
    start_brace = json_text.find('{')
    if start_brace > 0:
        json_text = json_text[start_brace:]
    
    # Remove any text after the last }
    end_brace = json_text.rfind('}')
    if end_brace != -1:
        json_text = json_text[:end_brace + 1]
    
    # Fix common formatting issues
    # 1. Replace single quotes with double quotes (but be careful with apostrophes in text)
    json_text = re.sub(r"'([^']*)'(\s*:)", r'"\1"\2', json_text)  # Keys
    json_text = re.sub(r":\s*'([^']*)'", r': "\1"', json_text)    # String values
    
    # 2. Fix missing quotes around property names (more comprehensive patterns)
    # Handle property names at start of object or after comma
    json_text = re.sub(r'([{,]\s*)([a-zA-Z_][a-zA-Z0-9_-]*)(\s*:)', r'\1"\2"\3', json_text)
    # Handle any remaining unquoted property names
    json_text = re.sub(r'(\s)([a-zA-Z_][a-zA-Z0-9_-]*)(\s*:)', r'\1"\2"\3', json_text)
    
    # 3. Fix property names that might already be partially quoted
    json_text = re.sub(r'([{,]\s*)"?([a-zA-Z_][a-zA-Z0-9_-]*)"?(\s*:)', r'\1"\2"\3', json_text)
    
    # 4. Remove trailing commas
    json_text = re.sub(r',(\s*[}\]])', r'\1', json_text)
    
    # 5. Fix placeholder values that might not be valid JSON
    json_text = re.sub(r'<[^>]*>', r'0', json_text)  # Replace <PLACEHOLDER> with 0
    
    # 6. Ensure boolean values are lowercase
    json_text = re.sub(r'\bTrue\b', 'true', json_text)
    json_text = re.sub(r'\bFalse\b', 'false', json_text)
    json_text = re.sub(r'\bNone\b', 'null', json_text)
    
    # 7. Handle edge case - fix double quotes around already quoted strings
    json_text = re.sub(r'""([^"]*?)""', r'"\1"', json_text)
    
    return json_text

def _parse_json_with_fallbacks(json_text: str) -> dict:
    """Try parsing JSON with progressive cleaning approaches."""
    import json
    import re
    
    # Attempt 1: Try as-is
    try:
        return json.loads(json_text)
    except json.JSONDecodeError as e:
        logger.warning(f"First JSON parse failed at pos {e.pos}: {e}")
        
    # Attempt 2: More aggressive property name fixing
    try:
        # Remove all existing quotes around property names and re-add them
        attempt2 = re.sub(r'"([a-zA-Z_][a-zA-Z0-9_-]*)"(\s*:)', r'\1\2', json_text)
        attempt2 = re.sub(r'([{,]\s*)([a-zA-Z_][a-zA-Z0-9_-]*)(\s*:)', r'\1"\2"\3', attempt2)
        return json.loads(attempt2)
    except json.JSONDecodeError as e:
        logger.warning(f"Second JSON parse failed at pos {e.pos}: {e}")
        
    # Attempt 3: Try to fix specific character position
    try:
        # If we know the position, try to fix around it
        if hasattr(e, 'pos') and e.pos:
            pos = e.pos
            # Look around the error position for common issues
            before = json_text[max(0, pos-10):pos]
            after = json_text[pos:min(len(json_text), pos+10)]
            logger.warning(f"Error context: ...{before}[ERROR HERE]{after}...")
            
            # Try replacing common issues around that position
            attempt3 = json_text[:pos] + json_text[pos:].replace("'", '"', 1)
            return json.loads(attempt3)
    except (json.JSONDecodeError, AttributeError):
        pass
        
    # Attempt 4: Last resort - try to extract just the basic structure
    try:
        # Find overall_analysis section
        overall_match = re.search(r'"overall_analysis"\s*:\s*\{[^}]*"predicted_monthly_sales"\s*:\s*(\d+)[^}]*\}', json_text)
        if overall_match:
            predicted_sales = int(overall_match.group(1))
            return {
                "overall_analysis": {
                    "predicted_monthly_sales": predicted_sales,
                    "confidence": 0.5,
                    "procurement_recommendation": "CAUTIOUS",
                    "reasoning": "Parsed from malformed JSON",
                    "status_color": "YELLOW",
                    "status_message": "🟡 YELLOW - PROCEED WITH CAUTION"
                },
                "store_predictions": [],
                "summary": {"total_monthly_sales": predicted_sales}
            }
    except:
        pass
        
    # If all attempts fail, raise the original error
    raise json.JSONDecodeError("All JSON parsing attempts failed", json_text, 0)

def _parse_comprehensive_response(llm_response: str, product_description: str, similar_products: List[Dict], stores: List[Dict]) -> Dict[str, Any]:
    """Parse comprehensive LLM response into expected format."""
    import json
    
    try:
        # Try to parse JSON response
        if '```json' in llm_response:
            json_start = llm_response.find('```json') + 7
            json_end = llm_response.find('```', json_start)
            json_text = llm_response[json_start:json_end].strip()
        else:
            json_text = llm_response.strip()
        
        # Clean and fix common JSON formatting issues
        json_text = _clean_llm_json_response(json_text)
        logger.info(f"🧹 Cleaned JSON (first 300 chars): {json_text[:300]}...")
        
        # Try parsing with progressive cleaning if needed
        llm_data = _parse_json_with_fallbacks(json_text)
        
        overall = llm_data.get('overall_analysis', {})
        store_preds = llm_data.get('store_predictions', [])
        summary = llm_data.get('summary', {})
        
        # Convert store predictions to expected format
        store_predictions = []
        for pred in store_preds:
            store_predictions.append({
                'store_id': pred.get('store_id', ''),
                'store_name': pred.get('store_name', ''),
                'city': pred.get('city', ''),
                'predicted_demand': pred.get('predicted_monthly_sales', 0),
                'confidence': pred.get('confidence', 0.5),
                'reasoning': pred.get('reasoning', ''),
                'recommendation': pred.get('recommendation', 'CAUTIOUS'),
                'climate': _get_store_climate(stores, pred.get('store_id', '')),
                'weather_factor': 1.0,  # LLM already considered
                'ai_generated': True
            })
        
        # Sort by predicted demand
        store_predictions.sort(key=lambda x: x['predicted_demand'], reverse=True)
        
        # Validate and recalculate overall metrics from store predictions
        if store_predictions:
            calculated_total = sum(p['predicted_demand'] for p in store_predictions)
            calculated_confidence = sum(p['confidence'] for p in store_predictions) / len(store_predictions)
            
            # Get LLM provided values
            llm_total = overall.get('predicted_monthly_sales', 0)
            llm_confidence = overall.get('confidence', 0.5)
            
            # Check if LLM calculations are reasonable (within 10% tolerance)
            total_diff_pct = abs(calculated_total - llm_total) / max(calculated_total, 1) if calculated_total > 0 else 0
            confidence_diff = abs(calculated_confidence - llm_confidence)
            
            if total_diff_pct > 0.1:  # More than 10% difference
                logger.warning(f"⚠️ LLM total ({llm_total}) differs significantly from calculated total ({calculated_total:.0f}). Using calculated value.")
                overall['predicted_monthly_sales'] = calculated_total
                
            if confidence_diff > 0.2:  # More than 0.2 difference 
                logger.warning(f"⚠️ LLM confidence ({llm_confidence:.2f}) differs from calculated confidence ({calculated_confidence:.2f}). Using calculated value.")
                overall['confidence'] = calculated_confidence
                
            # Always ensure summary matches the corrected overall values
            summary['total_monthly_sales'] = overall.get('predicted_monthly_sales', calculated_total)
            
            logger.info(f"✅ Validation: Total={overall.get('predicted_monthly_sales', 0):.0f}, Confidence={overall.get('confidence', 0):.2f}, Stores={len(store_predictions)}")
        
        # Create comprehensive result
        result = {
            'predicted_monthly_sales': overall.get('predicted_monthly_sales', 0),
            'confidence': overall.get('confidence', 0.5),
            'procurement_recommendation': overall.get('procurement_recommendation', 'CAUTIOUS'),
            'status_color': overall.get('status_color', 'YELLOW'),
            'status_message': overall.get('status_message', '🟡 YELLOW - PROCEED WITH CAUTION'),
            'status_description': f"AI-powered comprehensive analysis",
            'similar_products_count': len(similar_products),
            'llm_analysis': f"STATUS: {overall.get('status_message', 'UNKNOWN')}\n\n{overall.get('reasoning', 'No detailed reasoning provided')}",
            'similar_products': similar_products,
            'store_predictions': store_predictions,
            'total_store_demand': summary.get('total_monthly_sales', sum(p['predicted_demand'] for p in store_predictions)),
            'comprehensive_analysis': True,
            'ai_summary': {
                'high_potential_stores': summary.get('high_potential_stores', 0),
                'buy_stores': summary.get('buy_recommendation_stores', 0),
                'total_predicted': summary.get('total_monthly_sales', 0)
            }
        }
        
        logger.info(f"✅ Comprehensive analysis completed: {result['predicted_monthly_sales']} units/month, {len(store_predictions)} stores")
        return result
        
    except json.JSONDecodeError as e:
        logger.error(f"❌ Failed to parse LLM JSON response: {e}")
        logger.error(f"   Character position: {e.pos if hasattr(e, 'pos') else 'unknown'}")
        logger.error(f"   Cleaned JSON text (first 500 chars): {json_text[:500]}...")
        logger.error(f"   Raw LLM response (first 300 chars): {llm_response[:300]}...")
        return _create_fallback_prediction(product_description, error=f"JSON parsing failed: {str(e)}")
    except Exception as e:
        logger.error(f"❌ Error parsing comprehensive response: {e}")
        logger.error(f"   LLM response length: {len(llm_response)} characters")
        return _create_fallback_prediction(product_description, error=str(e))

def _create_fallback_prediction(product_description: str, error: str = None) -> Dict[str, Any]:
    """Create fallback prediction when LLM fails."""
    return {
        'predicted_monthly_sales': 0,
        'confidence': 0.1,
        'procurement_recommendation': 'AVOID',
        'status_color': 'RED',
        'status_message': '🔴 RED - ANALYSIS FAILED',
        'status_description': 'Unable to analyze due to technical issues',
        'similar_products_count': 0,
        'llm_analysis': f'ANALYSIS FAILED: {error or "Technical error occurred"}',
        'similar_products': [],
        'store_predictions': [],
        'error': error or "Unknown error"
    }

def _create_avoid_prediction(product_description: str, viability: Dict, similar_products: List[Dict]) -> Dict[str, Any]:
    """Create avoid recommendation for non-viable products."""
    return {
        'predicted_monthly_sales': 0,
        'confidence': 0.05,
        'procurement_recommendation': 'AVOID',
        'status_color': 'RED',
        'status_message': '🔴 RED - DO NOT PROCURE',
        'status_description': 'Product deemed non-viable',
        'similar_products_count': len(similar_products),
        'llm_analysis': f"STATUS: 🔴 RED - DO NOT PROCURE\n\nREASONING: {viability['reason']}",
        'similar_products': similar_products[:5],
        'store_predictions': [],
        'viability_analysis': viability
    }

def _create_statistical_prediction(product_description: str, similar_products: List[Dict], products_with_sales: List[Dict], stores: List[Dict]) -> Dict[str, Any]:
    """Create statistical prediction when LLM not available.""" 
    if not products_with_sales:
        return _create_fallback_prediction(product_description, "No historical sales data")
    
    # Simple statistical calculation
    total_monthly = sum(p.get('sales', {}).get('avg_monthly_units', 0) for p in products_with_sales)
    avg_monthly = total_monthly / len(products_with_sales) if products_with_sales else 0
    
    return {
        'predicted_monthly_sales': int(avg_monthly),
        'confidence': 0.6,
        'procurement_recommendation': 'CAUTIOUS',
        'status_color': 'YELLOW',
        'status_message': '🟡 YELLOW - STATISTICAL ANALYSIS',
        'status_description': 'Based on statistical analysis only',
        'similar_products_count': len(similar_products),
        'llm_analysis': f"STATISTICAL ANALYSIS\n\nPredicted: {int(avg_monthly)} units/month based on {len(products_with_sales)} similar products",
        'similar_products': similar_products,
        'store_predictions': [],
        'statistical_only': True
    }

def predict_product_sales(kb: SharedKnowledgeBase, llm_client: LLMClient, product_description: str) -> Dict[str, Any]:
    """Predict sales for a new product based on similar historical products."""
    # Wrap entire prediction workflow with LangSmith tracing
    try:
        from langsmith import traceable
        
        # Use wrapper function to ensure proper trace nesting
        @traceable(
            name="Predict_Product_Sales", 
            run_type="chain", 
            tags=["sales-prediction", "procurement", "odm"],
            inputs={"product_description": product_description}
        )
        def _predict_wrapper(prod_desc: str):
            return _predict_sales_impl(kb, llm_client, prod_desc)
        
        return _predict_wrapper(product_description)
    except ImportError:
        # If langsmith not available, use direct call
        return _predict_sales_impl(kb, llm_client, product_description)
    except Exception as e:
        logger.warning(f"LangSmith tracing failed for sales prediction: {e}")
        return _predict_sales_impl(kb, llm_client, product_description)

def _predict_sales_impl(kb: SharedKnowledgeBase, llm_client: LLMClient, product_description: str) -> Dict[str, Any]:
    """Internal implementation of sales prediction."""
    logger.info(f"🔮 Predicting sales for: {product_description}")
    
    try:
        # Find similar products with LangSmith tracing
        try:
            from langsmith import traceable
            
            @traceable(
                name="Vector_Search",
                run_type="retriever",
                tags=["vector-db", "similarity-search"]
            )
            def _vector_search(description: str, top_k: int):
                return kb.find_similar_products(
                    query_attributes={},
                    query_description=description,
                    top_k=top_k
                )
            
            similar_products = _vector_search(product_description, 15)
        except (ImportError, Exception) as e:
            if not isinstance(e, ImportError):
                logger.warning(f"LangSmith tracing failed for vector search: {e}")
            similar_products = kb.find_similar_products(
                query_attributes={},
                query_description=product_description,
                top_k=15
            )
        
        if not similar_products:
            logger.warning("No similar products found for prediction")
            return {
                'prediction': 'Unable to predict - no similar products found',
                'confidence': 0.0,
                'similar_products': []
            }
        
        # Analyze product viability first
        viability = analyze_product_viability(product_description, similar_products)
        logger.info(f"Product viability analysis: {viability['reason']}")
        
        # If product is deemed not viable, return AVOID recommendation immediately
        if not viability['viable']:
            return {
                'predicted_monthly_sales': 0,
                'confidence': 0.05,  # Very low confidence
                'procurement_recommendation': 'AVOID',
                'status_color': 'RED',  # Red status for avoid
                'status_message': '🔴 RED - DO NOT PROCURE',
                'similar_products_count': len(similar_products),
                'llm_analysis': f"PREDICTION: 0 units per month\\nCONFIDENCE: 5%\\nPROCUREMENT_RECOMMENDATION: AVOID\\nSTATUS: 🔴 RED - DO NOT PROCURE\\nREASONING: {viability['reason']}. Market research shows very low acceptance for this color-category combination.\\nRISK_FACTORS: {viability['risk_level']} risk - potential for zero sales and inventory loss\\nMARKET_OPPORTUNITY: {viability['market_acceptance']} - not recommended for procurement",
                'similar_products': similar_products[:5],  # Show top 5 for reference
                'viability_analysis': viability,
                'statistical_prediction': {
                    'method': 'viability_rejection',
                    'total_similar_products': len(similar_products),
                    'products_with_sales': 0,
                    'rejection_reason': viability['reason']
                }
            }
        
        # Filter for products with actual sales data (historical products)
        products_with_sales = [p for p in similar_products if p.get('sales', {}).get('total_units', 0) > 0]
        
        # Also filter for products that match the color-category (even if no sales data)
        # This helps when we have exact matches but they're new products without sales history
        exact_color_category_matches = []
        for p in similar_products:
            attrs = p.get('attributes', {})
            p_color = str(attrs.get('colour', attrs.get('Colour', attrs.get('color', '')))).lower().strip()
            p_brick = str(attrs.get('brick', attrs.get('Brick', attrs.get('category', '')))).lower().strip()
            
            desc_attrs = extract_product_attributes(product_description, similar_products)
            desc_color = desc_attrs.get('color')
            desc_brick = desc_attrs.get('brick')
            
            if desc_color and desc_brick:
                # Check if this is an exact or similar color-category match
                color_match = False
                if desc_color in p_color or p_color in desc_color:
                    color_match = True
                elif are_colors_similar(desc_color, p_color):
                    color_match = True
                
                if color_match and desc_brick in p_brick:
                    exact_color_category_matches.append(p)
        
        if not products_with_sales:
            # If no products have sales, but we have exact color-category matches, use those
            if exact_color_category_matches:
                logger.info(f"Using {len(exact_color_category_matches)} exact color-category matches for estimation")
                top_similar = exact_color_category_matches[:3]
            else:
                # Use the best similar products and estimate
                logger.info("No similar products with sales data, using estimation based on similarity")
                top_similar = similar_products[:3]
            
            # Calculate estimated sales based on similarity
            avg_similarity = sum(p.get('similarity_score', 0) for p in top_similar) / len(top_similar) if top_similar else 0.5
            # Base prediction: higher if we have exact matches, lower if just similar
            base_prediction = 100 if exact_color_category_matches else 50
            estimated_monthly_sales = int(base_prediction + (avg_similarity * 150))
            confidence = min(0.6, 0.3 + (avg_similarity * 0.3))  # Higher confidence if we have exact matches
            
            return {
                'predicted_monthly_sales': estimated_monthly_sales,
                'confidence': confidence,
                'procurement_recommendation': 'CAUTIOUS',
                'similar_products_count': len(similar_products),
                'llm_analysis': f"PREDICTION: {estimated_monthly_sales} units per month\\nCONFIDENCE: {confidence*100:.1f}%\\nPROCUREMENT_RECOMMENDATION: CAUTIOUS\\nREASONING: No similar products with historical sales data found. Estimation based on {len(similar_products)} similar products with average similarity of {sum(p.get('similarity_score', 0) for p in similar_products)/len(similar_products):.2f}\\nRISK_FACTORS: No historical sales data available, untested market\\nMARKET_OPPORTUNITY: Unknown - requires market research",
                'similar_products': similar_products,
                'statistical_prediction': {
                    'method': 'similarity_estimation',
                    'total_similar_products': len(similar_products),
                    'products_with_sales': 0
                }
            }
        
        # Use products with sales data for prediction
        logger.info(f"Using {len(products_with_sales)} products with sales data out of {len(similar_products)} similar products")
        
        # Calculate prediction based on products with sales (use monthly averages)
        total_monthly_sales = sum(p.get('sales', {}).get('avg_monthly_units', 0) for p in products_with_sales)
        avg_monthly_sales = total_monthly_sales / len(products_with_sales) if products_with_sales else 0
        
        # Weight by similarity scores (only for products with sales)
        weighted_monthly_sales = sum(
            p.get('sales', {}).get('avg_monthly_units', 0) * p.get('similarity_score', 0) 
            for p in products_with_sales
        )
        total_similarity = sum(p.get('similarity_score', 0) for p in products_with_sales)
        
        if total_similarity > 0:
            predicted_sales = int(weighted_monthly_sales / total_similarity)
        else:
            predicted_sales = int(avg_monthly_sales)
        
        # Calculate confidence based on similarity scores and number of matches
        avg_similarity = total_similarity / len(products_with_sales) if products_with_sales else 0
        
        # Base confidence from similarity and number of matches
        base_confidence = min(avg_similarity * (len(products_with_sales) / 5), 1.0)
        
        # Boost confidence if we have exact color-category matches (from viability analysis)
        if viability.get('market_acceptance') == 'PROVEN':
            # Proven products get higher confidence
            confidence = min(0.8, base_confidence + 0.2)
        elif viability.get('market_acceptance') == 'ACCEPTABLE':
            # Acceptable products get moderate boost
            confidence = min(0.7, base_confidence + 0.15)
        else:
            confidence = max(0.3, base_confidence)  # Ensure minimum 30% if we have any matches
        
        # Ensure confidence is never 0 if we have products with sales
        if products_with_sales and confidence == 0:
            confidence = 0.3  # Minimum confidence when we have sales data
        
        # Use LLM for detailed analysis (use products with sales data)
        similar_products_text = "\n".join([
            f"- {p.get('name', 'Unknown')}: {p.get('sales', {}).get('avg_monthly_units', 0)} units/month avg (similarity: {p.get('similarity_score', 0):.2f})"
            for p in products_with_sales[:3]
        ])
        
        # Include viability analysis in the LLM prompt
        llm_prompt = f"""You are an ODM procurement analyst. Analyze whether to procure this new product: "{product_description}"

Historical Sales Data for Similar Products:
{similar_products_text}

Viability Pre-Analysis:
- Product Viability: {viability['viable']}
- Risk Level: {viability['risk_level']}
- Market Acceptance: {viability['market_acceptance']}
- Analysis: {viability['reason']}

CRITICAL PROCUREMENT GUIDELINES:
1. NEVER recommend procurement of color-category combinations with no market demand (e.g., pink jeans, purple formal pants)
2. AVOID products where color is inappropriate for the category (bright colors for formal wear, unconventional colors for pants/jeans)
3. Consider cultural and fashion norms - some color combinations are simply not commercially viable
4. If historical data shows no successful products in this color-category combination, recommend AVOID

Consider these factors:
1. MARKET DEMAND: Is there actual demand for this specific combination of attributes?
2. COLOR APPROPRIATENESS: Is the color suitable and commercially viable for this product type?
3. HISTORICAL EVIDENCE: Do we have proof that similar color-category combinations have sold well?
4. FASHION TRENDS: Is this combination aligned with current and future fashion trends?
5. TARGET MARKET: Who would actually buy this product combination?
6. RISK ASSESSMENT: What's the procurement risk vs potential reward?

Provide your analysis in this format:
PREDICTION: [number] units per month
CONFIDENCE: [0-100]%
PROCUREMENT_RECOMMENDATION: [BUY/AVOID/CAUTIOUS] 
REASONING: [detailed explanation including why to buy or avoid, specifically addressing color-category viability]
RISK_FACTORS: [specific risks with this product]
MARKET_OPPORTUNITY: [market potential assessment]"""

        # Generate LLM analysis if available
        if llm_client is not None:
            try:
                # Add metadata for LangSmith tracking
                llm_prompt_with_metadata = f"""[ODM_SALES_PREDICTION] {product_description}

{llm_prompt}"""
                
                # Wrap LLM call with explicit tracing to capture inputs
                try:
                    from langsmith import traceable
                    
                    @traceable(
                        name="LLM_Analysis",
                        run_type="llm",
                        tags=["procurement-analysis", "openai"],
                        inputs={
                            "product_description": product_description,
                            "similar_products_count": len(similar_products),
                            "products_with_sales": len(products_with_sales),
                            "predicted_sales": predicted_sales,
                            "confidence": confidence
                        }
                    )
                    def _llm_analysis(prompt: str):
                        return llm_client.generate(prompt, temperature=0.2, max_tokens=800)
                    
                    llm_result = _llm_analysis(llm_prompt_with_metadata)
                except (ImportError, Exception) as e:
                    if not isinstance(e, ImportError):
                        logger.warning(f"LangSmith tracing failed for LLM analysis: {e}")
                    # Fall back to direct call if LangSmith fails
                    llm_result = llm_client.generate(llm_prompt_with_metadata, temperature=0.2, max_tokens=800)
                
                logger.info("✅ LLM prediction generated with LangSmith tracking")
                
                # Extract the response text from the result
                llm_response = llm_result.get('response', llm_result.get('content', str(llm_result)))
                logger.info("✅ LLM prediction generated successfully")
            except Exception as e:
                logger.warning(f"LLM prediction failed: {e}, using statistical prediction")
                llm_response = f"PREDICTION: {predicted_sales} units per month\nCONFIDENCE: {confidence*100:.1f}%\nPROCUREMENT_RECOMMENDATION: CAUTIOUS\nREASONING: Based on statistical analysis of {len(similar_products)} similar products\nRISK_FACTORS: Limited AI analysis due to technical issues\nMARKET_OPPORTUNITY: Moderate based on similar product performance"
        else:
            # No LLM available - use statistical analysis only
            logger.info("Using statistical analysis only (no LLM available)")
            llm_response = f"PREDICTION: {predicted_sales} units per month\nCONFIDENCE: {confidence*100:.1f}%\nPROCUREMENT_RECOMMENDATION: CAUTIOUS\nREASONING: Based on statistical analysis of {len(products_with_sales)} similar products with sales data\nRISK_FACTORS: No AI analysis available - based on historical data only\nMARKET_OPPORTUNITY: Moderate based on similar product performance"
        
        # Extract procurement recommendation
        procurement_recommendation = "CAUTIOUS"  # Default
        if "PROCUREMENT_RECOMMENDATION:" in llm_response:
            try:
                rec_line = [line for line in llm_response.split('\n') if 'PROCUREMENT_RECOMMENDATION:' in line][0]
                procurement_recommendation = rec_line.split(':')[1].strip()
            except:
                procurement_recommendation = "CAUTIOUS"
        
        # Get status indicator (Red/Yellow/Green) - will update with total store demand later if available
        status = get_status_indicator(procurement_recommendation, confidence, predicted_sales, total_store_demand=0)
        
        result = {
            'predicted_monthly_sales': predicted_sales,
            'confidence': confidence,
            'procurement_recommendation': procurement_recommendation,
            'status_color': status['color'],
            'status_message': status['message'], 
            'status_description': status['description'],
            'similar_products_count': len(similar_products),
            'llm_analysis': f"STATUS: {status['message']}\\n\\n{llm_response}",
            'similar_products': similar_products,  # Show all similar products
            'products_with_sales': products_with_sales,  # Show which ones have sales data
            'viability_analysis': viability,  # Include viability analysis
            'statistical_prediction': {
                'avg_monthly_sales': avg_monthly_sales,
                'weighted_monthly_sales': predicted_sales,
                'total_similar_products': len(similar_products),
                'products_with_sales_data': len(products_with_sales)
            }
        }
        
        logger.info(f"✅ Prediction complete: {predicted_sales} units/month (confidence: {confidence:.2f})")
        return result
        
    except Exception as e:
        logger.error(f"❌ Prediction failed: {e}")
        return {
            'predicted_monthly_sales': 0,
            'confidence': 0.0,
            'procurement_recommendation': 'AVOID',
            'status_color': 'RED',
            'status_message': '🔴 RED - ERROR IN ANALYSIS',
            'status_description': 'Technical error occurred during prediction',
            'similar_products_count': 0,
            'llm_analysis': f'PREDICTION: 0 units per month\\nCONFIDENCE: 0%\\nPROCUREMENT_RECOMMENDATION: AVOID\\nSTATUS: 🔴 RED - ERROR IN ANALYSIS\\nREASONING: Technical error occurred: {e}\\nRISK_FACTORS: System error - avoid procurement until resolved\\nMARKET_OPPORTUNITY: Cannot assess due to technical issues',
            'similar_products': [],
            'error': str(e)
        }

def main():
    """Main application."""
    st.title("🏪 ODM Intelligence Platform")
    st.markdown("*AI-powered product sales prediction based on historical ODM data*")
    
    # Initialize system
    kb, llm_client, data_agent = initialize_system()
    
    # Load ODM data
    with st.spinner("Loading ODM data..."):
        odm_data = load_odm_data(kb, data_agent)
    
    if not odm_data:
        st.error("Failed to load ODM data. Please check the data files and try again.")
        return
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Choose a page:",
        ["📊 Data Summary", "🔍 Search Products", "🔮 Predict Sales", "🔄 Reindex Database"]
    )
    
    if page == "📊 Data Summary":
        show_data_summary(odm_data)
        
        # Show vector database stats
        st.subheader("🗄️ Vector Database Status")
        stats = kb.get_collection_stats()
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Products in Vector DB", stats.get('products_count', 0))
        with col2:
            st.metric("Database Status", stats.get('status', 'unknown'))
    
    elif page == "🔍 Search Products":
        st.header("🔍 Search Indexed Products")
        st.markdown("Search through the vector database to see what products are indexed.")
        
        # Search input
        search_query = st.text_input(
            "Enter search query:",
            placeholder="e.g., red shirt, denim jacket, cotton top",
            help="Search by product name, color, material, or description"
        )
        
        if search_query:
            with st.spinner("Searching..."):
                search_results = search_indexed_products(kb, search_query)
            
            if search_results:
                st.success(f"Found {len(search_results)} similar products")
                
                # Display results
                for i, product in enumerate(search_results, 1):
                    with st.expander(f"{i}. {product.get('name', 'Unknown Product')} (Similarity: {product.get('similarity_score', 0):.3f})"):
                        col1, col2 = st.columns([2, 1])
                        
                        with col1:
                            st.write("**Description:**", product.get('description', 'N/A'))
                            st.write("**Attributes:**", product.get('attributes', {}))
                        
                        with col2:
                            sales_data = product.get('sales', {})
                            st.metric("Total Units Sold", sales_data.get('total_units', 0))
                            st.metric("Total Revenue", f"₹{sales_data.get('total_revenue', 0):,.0f}")
                            st.metric("Avg Monthly Sales", f"{sales_data.get('avg_monthly_units', 0):.1f}")
            else:
                st.warning("No similar products found. Try different search terms.")
    
    elif page == "🔮 Predict Sales":
        st.header("🔮 Sales Prediction for New Products")
        st.markdown("Enter a product description to predict sales based on similar historical products.")
        
        # Product input
        new_product = st.text_area(
            "Describe the new product:",
            placeholder="e.g., Red cotton t-shirt with short sleeves for men",
            help="Describe the product including color, material, style, and target audience"
        )
        
        if st.button("🚀 Predict Sales", disabled=not new_product.strip()):
            with st.spinner("🤖 Analyzing products and generating comprehensive store-wise predictions..."):
                # Load store data for comprehensive analysis
                stores = load_store_data()
                prediction = predict_product_sales_comprehensive(kb, llm_client, new_product, stores)
            
            if prediction.get('predicted_monthly_sales') is not None:
                # Show prediction results
                st.success("✅ Analysis Complete!")
                
                # Status Indicator (Red/Yellow/Green)
                status_color = prediction.get('status_color', 'YELLOW')
                status_message = prediction.get('status_message', '🟡 YELLOW - PROCEED WITH CAUTION')
                status_description = prediction.get('status_description', 'Moderate risk')
                
                if status_color == 'RED':
                    st.error(f"**{status_message}**")
                    st.error(f"**{status_description}**")
                elif status_color == 'GREEN':
                    st.success(f"**{status_message}**")
                    st.success(f"**{status_description}**")
                else:  # YELLOW
                    st.warning(f"**{status_message}**")
                    st.warning(f"**{status_description}**")
                
                # Detailed Recommendation
                recommendation = prediction.get('procurement_recommendation', 'CAUTIOUS')
                st.write(f"**Detailed Recommendation:** {recommendation}")
                
                # Store predictions are included in comprehensive analysis
                total_store_demand = prediction.get('total_store_demand', 0)
                store_predictions = prediction.get('store_predictions', [])
                
                # Key metrics
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    # Show total predicted sales (sum across all stores)
                    if total_store_demand > 0:
                        predicted_total = total_store_demand
                        help_text = f"Total predicted monthly sales across all {len(stores)} stores based on weather and climate factors"
                    else:
                        # Estimate total if store predictions not available yet
                        num_stores = len(stores) if stores else 10  # Default to 10 stores
                        predicted_total = prediction['predicted_monthly_sales'] * num_stores
                        help_text = f"Estimated total monthly sales across {num_stores} stores (base: {prediction['predicted_monthly_sales']} units/store avg)"
                    
                    st.metric(
                        "Predicted Total Monthly Sales", 
                        f"{predicted_total:,} units",
                        help=help_text
                    )
                with col2:
                    confidence_pct = prediction['confidence'] * 100
                    st.metric(
                        "Confidence Level", 
                        f"{confidence_pct:.1f}%"
                    )
                with col3:
                    st.metric(
                        "Similar Products Found", 
                        prediction['similar_products_count']
                    )
                with col4:
                    products_with_sales = len(prediction.get('products_with_sales', []))
                    st.metric(
                        "With Historical Sales", 
                        f"{products_with_sales} products"
                    )
                
                # LLM Analysis
                if prediction.get('llm_analysis'):
                    st.subheader("🤖 AI Analysis")
                    st.text_area("Detailed Analysis:", prediction['llm_analysis'], height=200)
                
                # Similar products used for prediction
                if prediction.get('similar_products'):
                    st.subheader("📊 Similar Products Used for Prediction")
                    
                    for i, similar in enumerate(prediction['similar_products'][:5], 1):
                        with st.expander(f"{i}. {similar.get('name', 'Unknown')} (Similarity: {similar.get('similarity_score', 0):.3f})"):
                            col1, col2 = st.columns(2)
                            with col1:
                                st.write("**Description:**", similar.get('description', 'N/A'))
                                st.write("**Attributes:**", similar.get('attributes', {}))
                            with col2:
                                sales = similar.get('sales', {})
                                st.write(f"**Total Sales:** {sales.get('total_units', 0)} units")
                                st.write(f"**Revenue:** ₹{sales.get('total_revenue', 0):,.0f}")
                                st.write(f"**Monthly Avg:** {sales.get('avg_monthly_units', 0):.1f} units")
                
                # Store-wise predictions
                st.subheader("🏪 AI-Powered Store-wise Sales Predictions")
                st.markdown("*LLM-generated predictions for each store based on historical sales data and local factors*")
                
                # Store predictions are now included in comprehensive analysis
                store_predictions = prediction.get('store_predictions', [])
                total_store_demand = prediction.get('total_store_demand', 0)
                
                if store_predictions:
                        # Summary metrics (use already calculated total_store_demand or recalculate)
                        if total_store_demand == 0:
                            total_store_demand = sum(pred['predicted_demand'] for pred in store_predictions)
                        green_stores = len([p for p in store_predictions if p['predicted_demand'] >= 1000])
                        high_priority = len([p for p in store_predictions if p['recommendation'] == 'HIGH_PRIORITY_BUY'])
                        
                        # Recalculate status with total store demand for better accuracy
                        updated_status = get_status_indicator(
                            prediction.get('procurement_recommendation', 'CAUTIOUS'),
                            prediction.get('confidence', 0.5),
                            prediction.get('predicted_monthly_sales', 0),
                            total_store_demand=total_store_demand
                        )
                        
                        # Update prediction with new status if it improved (especially if it changed to GREEN)
                        if updated_status['color'] == 'GREEN' or \
                           (updated_status['color'] == 'GREEN' and prediction.get('status_color') != 'GREEN'):
                            prediction['status_color'] = updated_status['color']
                            prediction['status_message'] = updated_status['message']
                            prediction['status_description'] = updated_status['description']
                            
                            # Show updated status banner
                            st.info(f"📊 **Status Updated:** {updated_status['message']} - Based on total store demand of {total_store_demand:,} units across {len(store_predictions)} stores")
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("🏪 Total Store Demand", f"{total_store_demand:,} units")
                        with col2:
                            st.metric("🟢 HIGH Priority Stores (≥1000 units)", high_priority)
                        with col3:
                            st.metric("✅ BUY Recommendation Stores", len([p for p in store_predictions if p['recommendation'] in ['BUY', 'HIGH_PRIORITY_BUY']]))
                        
                        # Top 10 stores
                        st.write("**Top 10 Performing Stores:**")
                        top_10_stores = store_predictions[:10]
                        
                        store_data = []
                        for store in top_10_stores:
                            store_data.append({
                                'Store ID': store['store_id'],
                                'Store Name': store['store_name'],
                                'City': store['city'],
                                'Climate': store['climate'],
                                'Predicted Demand': f"{store['predicted_demand']:,} units",
                                'Weather Factor': f"{store['weather_factor']:.2f}x",
                                'Recommendation': store['recommendation']
                            })
                        
                        # Display as dataframe
                        df = pd.DataFrame(store_data)
                        st.dataframe(df, use_container_width=True)
                        
                        # Climate breakdown
                        with st.expander("🌍 Climate-wise Performance Breakdown"):
                            climate_summary = {}
                            for pred in store_predictions:
                                climate = pred['climate']
                                if climate not in climate_summary:
                                    climate_summary[climate] = {'stores': 0, 'total_demand': 0}
                                climate_summary[climate]['stores'] += 1
                                climate_summary[climate]['total_demand'] += pred['predicted_demand']
                            
                            for climate, data in climate_summary.items():
                                avg_demand = data['total_demand'] / data['stores']
                                col1, col2, col3 = st.columns(3)
                                with col1:
                                    st.write(f"**{climate}**")
                                with col2:
                                    st.write(f"{data['total_demand']:,} total units")
                                with col3:
                                    st.write(f"{avg_demand:.0f} avg per store")
                        
                        # All stores table
                        with st.expander("📋 View All Store Predictions"):
                            all_store_data = []
                            for store in store_predictions:
                                all_store_data.append({
                                    'Store ID': store['store_id'],
                                    'Store Name': store['store_name'],
                                    'City': store['city'],
                                    'Climate': store['climate'],
                                    'Predicted Demand': store['predicted_demand'],
                                    'Weather Factor': f"{store['weather_factor']:.2f}x",
                                    'Recommendation': store['recommendation']
                                })
                            
                            all_df = pd.DataFrame(all_store_data)
                            st.dataframe(all_df, use_container_width=True)
                else:
                    st.warning("⚠️ Could not load store data. Store-wise predictions unavailable.")
            else:
                st.error("❌ Prediction failed. Please try with a different product description.")
    
    elif page == "🔄 Reindex Database":
        st.header("🔄 Reindex Vector Database")
        st.markdown("Clear and reindex all products and sales data in the vector database.")
        
        st.warning("⚠️ **Warning**: This will delete all existing data and reindex from CSV files.")
        
        # Get current stats
        current_stats = kb.get_collection_stats()
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Current Products", current_stats.get('products_count', 0))
        with col2:
            st.metric("Database Status", current_stats.get('status', 'unknown'))
        
        if st.button("🔄 Reindex Database", type="primary"):
            with st.spinner("Reindexing vector database... This may take a few minutes."):
                try:
                    # Clear existing data
                    kb.reset_collections()
                    kb._products_collection = kb._get_or_create_collection("products")
                    kb._performance_collection = kb._get_or_create_collection("performance")
                    st.success("✅ Cleared existing data")
                    
                    # Clear cache to force reload
                    load_odm_data.clear()
                    
                    # Reload and index data
                    st.info("📦 Loading and indexing products...")
                    odm_data = load_odm_data(kb, data_agent)
                    
                    if odm_data:
                        # Get new stats
                        new_stats = kb.get_collection_stats()
                        st.success(f"✅ Reindexing complete!")
                        st.balloons()
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Products Indexed", new_stats.get('products_count', 0))
                        with col2:
                            st.metric("Input Products", odm_data.get('summary', {}).get('dirty_input_count', 0))
                        with col3:
                            st.metric("Historical Products", odm_data.get('summary', {}).get('historical_products_count', 0))
                        
                        st.info("🔄 Please refresh the page (F5) to see updated data in other pages.")
                    else:
                        st.error("❌ Failed to reindex data. Check logs for details.")
                        
                except Exception as e:
                    st.error(f"❌ Reindexing failed: {e}")
                    logger.error(f"Reindexing error: {e}", exc_info=True)
        
        st.markdown("---")
        st.markdown("### Alternative: Use Command Line")
        st.code("""
# Run from terminal:
python3 reindex_vector_db.py
        """, language="bash")
        
        st.markdown("---")
        st.markdown("### Data Files")
        st.info(f"""
**Input Products:** `data/sample/dirty_odm_input.csv`  
**Historical Data:** `data/sample/odm_historical_dataset_5000.csv`  
**Vector DB Location:** `{settings.chromadb_persist_dir}`
        """)
    
if __name__ == "__main__":
    main()