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

def get_status_indicator(recommendation: str, confidence: float, predicted_sales: int) -> Dict[str, str]:
    """Get Red/Yellow/Green status indicator based on recommendation and confidence."""
    
    if recommendation == 'AVOID' or predicted_sales == 0:
        return {
            'color': 'RED',
            'message': '🔴 RED - DO NOT PROCURE',
            'description': 'High risk, avoid procurement'
        }
    elif recommendation == 'BUY' and confidence >= 0.7 and predicted_sales >= 100:
        return {
            'color': 'GREEN', 
            'message': '🟢 GREEN - RECOMMENDED',
            'description': 'High confidence, good sales potential'
        }
    elif recommendation == 'BUY' and confidence >= 0.5 and predicted_sales >= 50:
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

def analyze_product_viability(product_description: str, similar_products: List[Dict]) -> Dict[str, Any]:
    """Analyze if a product combination is viable based on market logic."""
    description_lower = product_description.lower()
    
    # Define problematic color-category combinations
    problematic_combinations = {
        'jeans': ['pink', 'bright pink', 'hot pink', 'neon', 'purple', 'yellow', 'orange'],
        'pants': ['pink', 'bright pink', 'hot pink', 'neon', 'purple'],
        'trousers': ['pink', 'bright pink', 'hot pink', 'neon', 'purple', 'yellow'],
        'formal pants': ['pink', 'bright pink', 'hot pink', 'neon', 'purple', 'yellow'],
        'shorts': ['pink', 'bright pink', 'hot pink'] if 'mens' in description_lower else []
    }
    
    # Check for problematic combinations
    for category, problematic_colors in problematic_combinations.items():
        if category in description_lower:
            for color in problematic_colors:
                if color in description_lower:
                    return {
                        'viable': False,
                        'reason': f'{color.title()} color is not commercially viable for {category}',
                        'risk_level': 'HIGH',
                        'market_acceptance': 'VERY_LOW'
                    }
    
    # Check if we found any actual historical data for this color-category combination
    matching_products = 0
    category_matches = 0
    color_matches = 0
    
    for product in similar_products:
        attrs = product.get('attributes', {})
        product_color = str(attrs.get('Colour', '')).lower()
        product_category = str(attrs.get('Brick', '')).lower()
        
        # Check category matches (more lenient)
        description_words = description_lower.split()
        category_keywords = ['jeans', 'pants', 'shirt', 'dress', 't-shirt', 'tshirt', 'top', 'jacket']
        
        for word in description_words:
            if word in category_keywords and (word in product_category or any(cat in word for cat in product_category.split())):
                category_matches += 1
                break
        
        # Check color matches
        color_keywords = ['red', 'blue', 'black', 'white', 'green', 'yellow', 'purple', 'pink', 'brown', 'gray', 'grey']
        for color in color_keywords:
            if color in description_lower and color in product_color:
                color_matches += 1
                break
        
        # Count exact color-category combinations
        if any(color in product_color for color in description_words) and \
           any(cat in product_category for cat in description_words):
            matching_products += 1
    
    # Be more lenient - if we have category matches or reasonable similar products, allow it
    if category_matches >= 3 or len(similar_products) >= 10:
        return {
            'viable': True,
            'reason': f'Found {category_matches} category matches and {len(similar_products)} similar products',
            'risk_level': 'MEDIUM' if matching_products == 0 else 'LOW',
            'market_acceptance': 'ACCEPTABLE'
        }
    
    if matching_products == 0 and category_matches < 2:
        # Very limited historical evidence
        return {
            'viable': False,
            'reason': 'Very limited historical evidence of similar products being successful',
            'risk_level': 'HIGH',
            'market_acceptance': 'UNPROVEN'
        }
    
    return {
        'viable': True,
        'reason': 'Product combination appears viable',
        'risk_level': 'LOW',
        'market_acceptance': 'ACCEPTABLE'
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
        
        if not products_with_sales:
            # If no products have sales, use the best similar products and estimate
            logger.info("No similar products with sales data, using estimation based on similarity")
            # Use top similar products and estimate based on market averages
            top_similar = similar_products[:3]
            estimated_monthly_sales = int(50 + (sum(p.get('similarity_score', 0) for p in top_similar) / len(top_similar)) * 100)
            confidence = 0.4  # Lower confidence for estimation
            
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
        confidence = min(avg_similarity * (len(products_with_sales) / 5), 1.0)  # Max confidence when 5+ similar products with sales
        
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
        
        # Get status indicator (Red/Yellow/Green)
        status = get_status_indicator(procurement_recommendation, confidence, predicted_sales)
        
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
                
                # Key metrics
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric(
                        "Predicted Monthly Sales", 
                        f"{prediction['predicted_monthly_sales']} units"
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
            else:
                st.error("❌ Prediction failed. Please try with a different product description.")

if __name__ == "__main__":
    main()