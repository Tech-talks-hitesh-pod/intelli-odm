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
        # Initialize knowledge base
        kb = SharedKnowledgeBase()
        logger.info("✅ Knowledge base initialized")
        
        # Initialize LLM client using factory
        llm_config = {
            'provider': settings.llm_provider,
            'openai_api_key': settings.openai_api_key if settings.llm_provider == 'openai' else None,
            'openai_model': settings.openai_model if settings.llm_provider == 'openai' else 'gpt-4o-mini',
            'openai_temperature': settings.openai_temperature if settings.llm_provider == 'openai' else 0.1,
            'ollama_base_url': settings.ollama_base_url if settings.llm_provider == 'ollama' else 'http://localhost:11434',
            'ollama_model': settings.ollama_model if settings.llm_provider == 'ollama' else 'llama3:8b'
        }
        
        try:
            llm_client = LLMClientFactory.create_client(llm_config)
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
    predictions = []
    
    for store in stores:
        store_id = store.get('StoreID', '')
        store_name = store.get('StoreName', '')
        city = store.get('City', '')
        climate = store.get('ClimateTag', '')
        
        # Calculate weather factor
        weather_factor = get_weather_factor(climate, product_description)
        
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

def analyze_product_viability(product_description: str, similar_products: List[Dict]) -> Dict[str, Any]:
    """Analyze if a product combination is viable based on actual sales data."""
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

def predict_product_sales(kb: SharedKnowledgeBase, llm_client: LLMClient, product_description: str) -> Dict[str, Any]:
    """Predict sales for a new product based on similar historical products."""
    logger.info(f"🔮 Predicting sales for: {product_description}")
    
    try:
        # Find similar products
        similar_products = kb.find_similar_products(
            query_attributes={},
            query_description=product_description,
            top_k=15  # Get more products to better analyze viability
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
                
                llm_result = llm_client.generate(llm_prompt_with_metadata, temperature=0.2, max_tokens=800)
                # Extract the response text from the result
                llm_response = llm_result.get('response', llm_result.get('content', str(llm_result)))
                logger.info("✅ LLM prediction generated with LangSmith tracking")
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
        ["📊 Data Summary", "🔍 Search Products", "🔮 Predict Sales"]
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
            with st.spinner("Analyzing similar products and generating prediction..."):
                prediction = predict_product_sales(kb, llm_client, new_product)
            
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
                
                # Calculate total store demand early for display
                stores = load_store_data()
                total_store_demand = 0
                store_predictions = []
                if stores:
                    base_prediction = prediction.get('predicted_monthly_sales', 100)
                    store_predictions = generate_store_predictions(base_prediction, new_product, stores)
                    if store_predictions:
                        total_store_demand = sum(pred['predicted_demand'] for pred in store_predictions)
                
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
                st.subheader("🏪 Store-wise Sales Predictions")
                st.markdown("*Predictions for each store based on weather patterns and climate factors*")
                
                # Use already calculated store predictions if available, otherwise calculate
                if stores and not store_predictions:
                    base_prediction = prediction.get('predicted_monthly_sales', 100)
                    store_predictions = generate_store_predictions(base_prediction, new_product, stores)
                
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

if __name__ == "__main__":
    main()