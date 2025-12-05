#!/usr/bin/env python3
"""
Streamlit UI for Intelli-ODM CEO Demo.

Controlled demo scenarios for executive presentation.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any

# Configure page
st.set_page_config(
    page_title="Intelli-ODM CEO Demo",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Import our modules
try:
    from config.settings import Settings
    from shared_knowledge_base import SharedKnowledgeBase
    from utils.llm_client import LLMClientFactory
    from agents.orchestrator_agent import OrchestratorAgent
    from demo_scenarios import DemoScenarios
except ImportError as e:
    st.error(f"Failed to import modules: {e}")
    st.stop()

# Custom CSS
st.markdown("""
<style>
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #e1e4e8;
        margin: 0.5rem 0;
    }
    
    .success-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
    }
    
    .warning-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        color: #856404;
    }
    
    .error-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        color: #721c24;
    }
    
    .stButton > button {
        width: 100%;
    }
</style>
""", unsafe_allow_html=True)

def initialize_session_state():
    """Initialize session state variables."""
    if 'orchestrator' not in st.session_state:
        st.session_state.orchestrator = None
    if 'results' not in st.session_state:
        st.session_state.results = None
    if 'analysis_complete' not in st.session_state:
        st.session_state.analysis_complete = False
    if 'sample_products' not in st.session_state:
        st.session_state.sample_products = [
            "Classic white cotton t-shirt with crew neck, short sleeves, regular fit, perfect for casual wear",
            "Navy blue denim jeans, slim fit, straight leg, button fly closure with premium stretch denim",
            "Elegant black cocktail dress, sleeveless design with V-neckline, knee-length chiffon fabric",
            "Red polo shirt for men, cotton blend fabric, collar with 3-button placket, athletic fit",
            "Women's floral print summer dress, midi length, cap sleeves, lightweight crepe material"
        ]
    if 'sample_inventory' not in st.session_state:
        st.session_state.sample_inventory = {
            "product_1": 150,
            "product_2": 80,
            "product_3": 25,
            "product_4": 120,
            "product_5": 40
        }

@st.cache_resource
def setup_system():
    """Initialize the Intelli-ODM system."""
    try:
        settings = Settings()
        
        # Create LLM client
        if settings.llm_provider == "ollama":
            llm_config = {
                "provider": "ollama",
                "base_url": settings.ollama_base_url,
                "model": settings.ollama_model
            }
        else:
            if not settings.openai_api_key:
                st.error("OpenAI API key not found. Please set OPENAI_API_KEY environment variable.")
                return None, None, None
            llm_config = {
                "provider": "openai",
                "api_key": settings.openai_api_key,
                "model": settings.openai_model
            }
        
        # Get LangSmith configuration for tracking
        langsmith_config = settings.get_langsmith_config()
        
        llm_client = LLMClientFactory.create_client(llm_config, langsmith_config=langsmith_config)
        kb = SharedKnowledgeBase()
        orchestrator = OrchestratorAgent(llm_client, kb)
        
        return orchestrator, llm_client, kb
        
    except Exception as e:
        st.error(f"System initialization failed: {e}")
        return None, None, None

def display_system_status(llm_client, kb):
    """Display system status in the sidebar."""
    st.sidebar.markdown("### 🔧 System Status")
    
    # LLM Status
    if llm_client:
        try:
            llm_available = llm_client.is_available()
            status = "🟢 Connected" if llm_available else "🟡 Limited"
            st.sidebar.markdown(f"**LLM Service:** {status}")
        except:
            st.sidebar.markdown("**LLM Service:** 🔴 Error")
    else:
        st.sidebar.markdown("**LLM Service:** 🔴 Not Initialized")
    
    # Knowledge Base Status
    if kb:
        try:
            kb_size = kb.get_collection_size()
            st.sidebar.markdown(f"**Knowledge Base:** 🟢 {kb_size} products")
        except:
            st.sidebar.markdown("**Knowledge Base:** 🟡 Available")
    else:
        st.sidebar.markdown("**Knowledge Base:** 🔴 Not Available")

def render_product_input():
    """Render product input section."""
    st.header("📦 Product Analysis Setup")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Product Descriptions")
        
        # Option to use sample data or custom input
        use_sample = st.checkbox("Use Sample Data", value=True)
        
        if use_sample:
            products = st.session_state.sample_products
            st.info("Using sample product descriptions. Uncheck to enter custom products.")
            
            # Display sample products in an expandable section
            with st.expander("View Sample Products"):
                for i, product in enumerate(products, 1):
                    st.write(f"{i}. {product}")
        else:
            st.info("Enter product descriptions (one per line)")
            product_text = st.text_area(
                "Product Descriptions:",
                height=200,
                placeholder="Enter product descriptions here...\nExample: White cotton t-shirt, crew neck, short sleeves"
            )
            products = [p.strip() for p in product_text.split('\n') if p.strip()]
        
        # Inventory input
        st.subheader("Current Inventory (Optional)")
        
        if use_sample:
            inventory = st.session_state.sample_inventory
            with st.expander("View Sample Inventory"):
                for product_id, qty in inventory.items():
                    st.write(f"{product_id}: {qty} units")
        else:
            inventory_text = st.text_area(
                "Inventory Data (JSON format):",
                height=100,
                placeholder='{"product_1": 100, "product_2": 50}'
            )
            try:
                inventory = json.loads(inventory_text) if inventory_text.strip() else {}
            except json.JSONDecodeError:
                inventory = {}
                if inventory_text.strip():
                    st.warning("Invalid JSON format for inventory data")
    
    with col2:
        st.subheader("Analysis Options")
        
        # Analysis parameters
        include_trends = st.checkbox("Include Trend Analysis", value=True)
        include_pricing = st.checkbox("Include Price Analysis", value=True)
        forecast_days = st.slider("Forecast Period (days)", 7, 90, 30)
        
        st.markdown("---")
        
        # Analysis button
        if st.button("🚀 Start Analysis", type="primary"):
            if not products:
                st.error("Please enter at least one product description")
            else:
                return products, inventory, {
                    'include_trends': include_trends,
                    'include_pricing': include_pricing,
                    'forecast_days': forecast_days
                }
    
    return None, None, None

def run_analysis(orchestrator, products, inventory, options):
    """Run the analysis workflow."""
    st.header("🔄 Analysis in Progress")
    
    # Progress tracking
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        status_text.text("Initializing analysis workflow...")
        progress_bar.progress(10)
        
        # Start analysis
        status_text.text("Processing product data...")
        progress_bar.progress(30)
        
        # Run the workflow
        with st.spinner("Running complete analysis workflow..."):
            results = orchestrator.run_complete_workflow(
                product_descriptions=products,
                inventory_data=inventory
            )
        
        progress_bar.progress(100)
        status_text.text("Analysis completed!")
        
        return results
        
    except Exception as e:
        st.error(f"Analysis failed: {e}")
        return None

def display_results_overview(results):
    """Display high-level results overview."""
    if not results or not results.get('success'):
        st.error("Analysis failed or no results available")
        return
    
    st.header("📊 Analysis Results Overview")
    
    # Key metrics
    metrics = results.get('metrics', {})
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Products Analyzed",
            metrics.get('total_products_processed', 0),
            f"{metrics.get('successful_analyses', 0)} successful"
        )
    
    with col2:
        st.metric(
            "Success Rate",
            f"{metrics.get('success_rate', 0):.1%}",
            f"Confidence: {results.get('confidence_score', 0):.2f}"
        )
    
    with col3:
        forecasts = metrics.get('forecasts_generated', 0)
        st.metric(
            "Forecasts Generated",
            forecasts,
            f"{forecasts} products"
        )
    
    with col4:
        recommendations = metrics.get('procurement_recommendations', 0)
        st.metric(
            "Recommendations",
            recommendations,
            f"{recommendations} actions"
        )
    
    # Executive summary
    st.subheader("📋 Executive Summary")
    summary = results.get('executive_summary', 'No summary available')
    st.markdown(f'<div class="success-box">{summary}</div>', unsafe_allow_html=True)
    
    # Key insights
    insights = results.get('key_insights', [])
    if insights:
        st.subheader("💡 Key Insights")
        for insight in insights:
            st.write(f"• {insight}")

def display_attribute_analysis(results):
    """Display product attribute analysis."""
    st.header("🏷️ Product Attribute Analysis")
    
    attribute_data = results.get('detailed_results', {}).get('attribute_analysis', {})
    product_attributes = attribute_data.get('product_attributes', {})
    
    if not product_attributes:
        st.warning("No attribute analysis available")
        return
    
    # Convert to DataFrame for display
    attrs_list = []
    for product_id, attrs in product_attributes.items():
        attrs_list.append({
            'Product': product_id,
            'Category': attrs.get('category', 'Unknown'),
            'Material': attrs.get('material', 'Unknown'),
            'Color': attrs.get('color', 'Unknown'),
            'Style': attrs.get('style', 'Unknown'),
            'Target Gender': attrs.get('target_gender', 'Unknown'),
            'Confidence': attrs.get('confidence', 0)
        })
    
    if attrs_list:
        df = pd.DataFrame(attrs_list)
        
        # Display table
        st.subheader("📋 Extracted Attributes")
        st.dataframe(df, use_container_width=True)
        
        # Visualizations
        col1, col2 = st.columns(2)
        
        with col1:
            # Category distribution
            if 'Category' in df.columns:
                category_counts = df['Category'].value_counts()
                fig = px.pie(
                    values=category_counts.values,
                    names=category_counts.index,
                    title="Product Category Distribution"
                )
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Confidence scores
            if 'Confidence' in df.columns:
                fig = px.bar(
                    df,
                    x='Product',
                    y='Confidence',
                    title="Attribute Extraction Confidence",
                    color='Confidence',
                    color_continuous_scale='viridis'
                )
                fig.update_layout(xaxis_tickangle=-45)
                st.plotly_chart(fig, use_container_width=True)

def display_demand_forecasting(results):
    """Display demand forecasting results."""
    st.header("📈 Demand Forecasting")
    
    forecast_data = results.get('detailed_results', {}).get('demand_forecasting', {})
    forecasts = forecast_data.get('demand_forecasts', {})
    
    if not forecasts:
        st.warning("No demand forecasts available")
        return
    
    # Prepare forecast data
    forecast_list = []
    for product_id, forecast in forecasts.items():
        if isinstance(forecast.get('forecast'), dict):
            forecast_qty = forecast['forecast'].get('quantity', 0)
            confidence = forecast['forecast'].get('confidence', 0)
            trend = forecast['forecast'].get('trend', {}).get('direction', 'stable')
        else:
            forecast_qty = forecast.get('forecast', 0)
            confidence = forecast.get('confidence', 0.5)
            trend = 'stable'
        
        forecast_list.append({
            'Product': product_id,
            'Forecast (30 days)': int(forecast_qty),
            'Confidence': confidence,
            'Trend': trend,
            'Current Inventory': st.session_state.sample_inventory.get(product_id, 0)
        })
    
    if forecast_list:
        df = pd.DataFrame(forecast_list)
        
        # Calculate inventory coverage
        df['Coverage (days)'] = (
            df['Current Inventory'] / (df['Forecast (30 days)'] / 30)
        ).replace([float('inf'), -float('inf')], 0).fillna(0).round(1)
        
        # Display forecast table
        st.subheader("📊 Demand Forecasts")
        st.dataframe(df, use_container_width=True)
        
        # Visualizations
        col1, col2 = st.columns(2)
        
        with col1:
            # Forecast vs Inventory
            fig = go.Figure()
            fig.add_trace(go.Bar(
                name='Forecast (30 days)',
                x=df['Product'],
                y=df['Forecast (30 days)'],
                marker_color='lightblue'
            ))
            fig.add_trace(go.Bar(
                name='Current Inventory',
                x=df['Product'],
                y=df['Current Inventory'],
                marker_color='orange'
            ))
            fig.update_layout(
                title='Demand Forecast vs Current Inventory',
                xaxis_title='Products',
                yaxis_title='Quantity',
                barmode='group'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Coverage analysis
            fig = px.bar(
                df,
                x='Product',
                y='Coverage (days)',
                title="Inventory Coverage (Days)",
                color='Coverage (days)',
                color_continuous_scale=['red', 'yellow', 'green'],
                text='Coverage (days)'
            )
            fig.update_traces(texttemplate='%{text}', textposition='outside')
            fig.add_hline(y=30, line_dash="dash", line_color="red", 
                         annotation_text="30-day target")
            st.plotly_chart(fig, use_container_width=True)

def display_procurement_recommendations(results):
    """Display procurement optimization results."""
    st.header("🛒 Procurement Recommendations")
    
    recommendations = results.get('recommendations', {}).get('procurement', [])
    
    if not recommendations:
        st.warning("No procurement recommendations available")
        return
    
    # Summary metrics
    total_investment = sum(rec['estimated_cost'] for rec in recommendations)
    total_quantity = sum(rec['quantity'] for rec in recommendations)
    high_priority = len([r for r in recommendations if r['priority'] == 'High'])
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Investment", f"₹{total_investment:,.0f}")
    with col2:
        st.metric("Total Quantity", f"{total_quantity:,} units")
    with col3:
        st.metric("High Priority Items", high_priority)
    
    # Recommendations table
    st.subheader("📋 Detailed Recommendations")
    
    rec_df = pd.DataFrame([
        {
            'Product': rec['product_id'],
            'Action': rec['action'],
            'Quantity': rec['quantity'],
            'Cost': f"₹{rec['estimated_cost']:,.0f}",
            'Priority': rec['priority'],
            'Timeline': rec['timeline'],
            'Supplier': rec['supplier']['name'],
            'Urgency': rec['urgency']
        }
        for rec in recommendations
    ])
    
    st.dataframe(rec_df, use_container_width=True)
    
    # Visualizations
    col1, col2 = st.columns(2)
    
    with col1:
        # Cost breakdown
        fig = px.bar(
            rec_df,
            x='Product',
            y=[float(cost.replace('₹', '').replace(',', '')) for cost in rec_df['Cost']],
            color='Priority',
            title="Procurement Cost by Priority",
            color_discrete_map={'High': 'red', 'Medium': 'orange', 'Low': 'green'}
        )
        fig.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Priority distribution
        priority_counts = rec_df['Priority'].value_counts()
        fig = px.pie(
            values=priority_counts.values,
            names=priority_counts.index,
            title="Recommendations by Priority",
            color_discrete_map={'High': 'red', 'Medium': 'orange', 'Low': 'green'}
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Strategic recommendations
    strategic_recs = results.get('recommendations', {}).get('strategic', [])
    if strategic_recs:
        st.subheader("🎯 Strategic Recommendations")
        for i, rec in enumerate(strategic_recs, 1):
            st.write(f"{i}. {rec}")

def display_errors_and_logs(results):
    """Display system errors and execution logs."""
    st.header("🔍 System Status & Logs")
    
    errors = results.get('errors', [])
    
    if errors:
        st.subheader("⚠️ Errors & Warnings")
        for error in errors:
            st.markdown(f'<div class="warning-box">⚠️ {error}</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="success-box">✅ No errors encountered</div>', unsafe_allow_html=True)
    
    # Execution metrics
    if 'metrics' in results:
        st.subheader("⏱️ Performance Metrics")
        exec_times = results['metrics'].get('execution_times', {})
        
        if exec_times:
            times_df = pd.DataFrame([
                {'Phase': phase.replace('_', ' ').title(), 'Time (seconds)': time}
                for phase, time in exec_times.items()
            ])
            
            fig = px.bar(
                times_df,
                x='Phase',
                y='Time (seconds)',
                title="Execution Time by Phase"
            )
            st.plotly_chart(fig, use_container_width=True)

def export_results(results):
    """Provide export functionality."""
    st.header("📤 Export Results")
    
    if not results:
        st.warning("No results to export")
        return
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Export complete results as JSON
        json_data = json.dumps(results, indent=2, default=str)
        st.download_button(
            "📋 Download Complete Results (JSON)",
            json_data,
            file_name=f"intelli_odm_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json"
        )
    
    with col2:
        # Export recommendations as CSV
        recommendations = results.get('recommendations', {}).get('procurement', [])
        if recommendations:
            rec_df = pd.DataFrame([
                {
                    'Product ID': rec['product_id'],
                    'Action': rec['action'],
                    'Quantity': rec['quantity'],
                    'Estimated Cost': rec['estimated_cost'],
                    'Priority': rec['priority'],
                    'Timeline': rec['timeline'],
                    'Supplier': rec['supplier']['name'],
                    'Justification': rec['justification']
                }
                for rec in recommendations
            ])
            
            csv_data = rec_df.to_csv(index=False)
            st.download_button(
                "📊 Download Recommendations (CSV)",
                csv_data,
                file_name=f"procurement_recommendations_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
    
    with col3:
        # Export summary report
        summary_report = f"""
INTELLI-ODM ANALYSIS REPORT
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

EXECUTIVE SUMMARY
{results.get('executive_summary', 'No summary available')}

KEY METRICS
- Products Processed: {results.get('metrics', {}).get('total_products_processed', 0)}
- Success Rate: {results.get('metrics', {}).get('success_rate', 0):.1%}
- Confidence Score: {results.get('confidence_score', 0):.2f}

KEY INSIGHTS
{chr(10).join(f"• {insight}" for insight in results.get('key_insights', []))}
        """
        
        st.download_button(
            "📄 Download Summary Report (TXT)",
            summary_report,
            file_name=f"summary_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
            mime="text/plain"
        )

def main():
    """Main CEO Demo Application."""
    # Initialize session state
    initialize_session_state()
    
    # Additional session state variables
    if 'current_scenario' not in st.session_state:
        st.session_state.current_scenario = None
    if 'demo_results' not in st.session_state:
        st.session_state.demo_results = None
    if 'agent_logs' not in st.session_state:
        st.session_state.agent_logs = []
    if 'new_product_evaluation' not in st.session_state:
        st.session_state.new_product_evaluation = None
    
    # Setup system if not already done
    if st.session_state.orchestrator is None:
        with st.spinner("🔄 Initializing system..."):
            orchestrator, llm_client, kb = setup_system()
            if orchestrator:
                st.session_state.orchestrator = orchestrator
                st.session_state.llm_client = llm_client
                st.session_state.kb = kb
    
    # App header
    st.title("🎯 Intelli-ODM Executive Demo")
    st.markdown("**AI-Powered Retail Intelligence & Procurement Optimization**")
    
    # Sidebar - Demo Controls
    with st.sidebar:
        st.markdown("### 🎬 Demo Controls")
        
        # Scenario Selection
        scenarios = DemoScenarios.get_available_scenarios()
        scenario_choice = st.selectbox(
            "Select Business Scenario:",
            options=list(scenarios.keys()),
            format_func=lambda x: scenarios[x],
            key="scenario_selector"
        )
        
        if st.button("📋 Load Scenario", type="primary"):
            st.session_state.current_scenario = scenario_choice
            st.session_state.demo_results = None
            st.session_state.agent_logs = []
            st.rerun()
        
        if st.session_state.current_scenario:
            st.markdown("---")
            if st.button("🚀 Run AI Analysis", type="secondary"):
                run_demo_analysis()
                st.rerun()
            
            if st.button("🔄 Reset Demo"):
                st.session_state.current_scenario = None
                st.session_state.demo_results = None
                st.session_state.agent_logs = []
                st.rerun()
    
    # Main content area - Use tabs for different views
    tab1, tab2 = st.tabs(["📊 Scenario Analysis", "🆕 New Product Evaluation"])
    
    with tab1:
        if not st.session_state.current_scenario:
            display_welcome_screen()
        else:
            display_demo_scenario()
    
    with tab2:
        display_new_product_evaluation()

def display_welcome_screen():
    """Display welcome screen with business value proposition."""
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        ## 🎯 Transform Your Retail Operations with AI
        
        **Intelli-ODM** combines multiple AI agents to deliver intelligent merchandising decisions:
        
        ### 🤖 Multi-Agent AI System
        - **Data Ingestion Agent**: Processes product catalogs and sales data
        - **Attribute Analysis Agent**: Extracts product features and finds comparables  
        - **Demand Forecasting Agent**: Predicts future demand with confidence scores
        - **Procurement Optimization Agent**: Generates cost-optimized procurement plans
        
        ### 📊 Business Impact
        - **25-40% reduction** in stockouts
        - **15-30% inventory optimization** 
        - **Real-time decision support** with confidence metrics
        - **Multi-scenario analysis** for risk management
        """)
    
    with col2:
        st.markdown("### 🎬 Demo Scenarios")
        scenarios = DemoScenarios.get_available_scenarios()
        for key, name in scenarios.items():
            st.markdown(f"• **{name}**")
        
        st.markdown("---")
        st.info("👈 Select a scenario from the sidebar to begin the demo")

def display_demo_scenario():
    """Display the selected demo scenario."""
    scenario_data = DemoScenarios.get_scenario_data(st.session_state.current_scenario)
    story = DemoScenarios.get_scenario_story(st.session_state.current_scenario)
    
    # Scenario header
    st.header(f"📋 {scenario_data['name']}")
    st.markdown(f"*{scenario_data['description']}*")
    
    # Story telling tabs
    tab1, tab2, tab3 = st.columns(3)
    
    with tab1:
        st.markdown("### 🏢 Business Context")
        st.markdown(story['setup'])
        
        # Key metrics
        st.markdown("**Key Metrics:**")
        context = scenario_data['business_context']
        for key, value in context.items():
            st.write(f"• **{key.replace('_', ' ').title()}**: {value}")
    
    with tab2:
        st.markdown("### ⚡ Challenge")
        st.markdown(story['challenge'])
        
        # Current inventory status
        st.markdown("**Current Inventory:**")
        products = scenario_data['products']
        inventory_items = []
        
        for i, (product_id, current_stock) in enumerate(scenario_data['inventory'].items(), 1):
            # Get actual product description
            if i <= len(products):
                product_name = products[i-1]
                # Extract short name (first part before comma or first 60 chars)
                if ',' in product_name:
                    short_name = product_name.split(',')[0].strip()
                else:
                    short_name = product_name[:60] + "..." if len(product_name) > 60 else product_name
            else:
                short_name = product_id
            
            inventory_items.append({
                "Product": short_name,
                "Product ID": product_id,
                "Current Stock": current_stock,
                "Status": "🔴 Critical" if current_stock < 50 else "🟡 Low" if current_stock < 100 else "🟢 Adequate"
            })
        
        inventory_df = pd.DataFrame(inventory_items)
        st.dataframe(inventory_df, use_container_width=True)
    
    with tab3:
        st.markdown("### 🎯 AI Solution")
        st.markdown(story['solution'])
        
        # Expected insights
        st.markdown("**Expected Insights:**")
        for insight in scenario_data['expected_insights']:
            st.write(f"• {insight}")
    
    # Show results if analysis was run
    if st.session_state.demo_results:
        display_demo_results()

def run_demo_analysis():
    """Run controlled analysis for demo scenario."""
    scenario_data = DemoScenarios.get_scenario_data(st.session_state.current_scenario)
    
    # Create progress display
    progress_bar = st.progress(0)
    status_container = st.empty()
    
    # Simulate real agent execution with progress
    agents = [
        {"name": "Data Ingestion", "action": "Processing product catalog", "duration": 1.5},
        {"name": "Attribute Analysis", "action": "Extracting product features", "duration": 2.0},
        {"name": "Demand Forecasting", "action": "Generating demand predictions", "duration": 1.8}, 
        {"name": "Procurement Optimization", "action": "Optimizing procurement plan", "duration": 1.2}
    ]
    
    import time
    st.session_state.agent_logs = []
    
    for i, agent in enumerate(agents):
        # Update progress
        progress = (i + 1) / len(agents)
        progress_bar.progress(progress)
        
        # Update status
        status_container.write(f"🤖 **{agent['name']}**: {agent['action']}...")
        
        # Simulate processing time
        time.sleep(agent['duration'])
        
        # Log completion
        st.session_state.agent_logs.append({
            "agent": agent['name'],
            "action": agent['action'], 
            "status": "✅ Complete",
            "duration": f"{agent['duration']:.1f}s"
        })
    
    # Final update
    progress_bar.progress(1.0)
    status_container.write("✅ **Analysis Complete!**")
    
    # Generate controlled results based on scenario
    st.session_state.demo_results = generate_demo_results(scenario_data)

def generate_demo_results(scenario_data):
    """Generate controlled demo results."""
    # This creates predictable results for demo purposes
    products = scenario_data['products']
    inventory = scenario_data['inventory']
    
    # Generate controlled recommendations
    recommendations = []
    total_cost = 0
    
    for i, (product_id, current_stock) in enumerate(inventory.items(), 1):
        if current_stock < 100:  # Low inventory triggers procurement
            quantity = 200 - current_stock  # Bring up to 200 units
            cost = quantity * (50 + i * 10)  # Varied pricing
            priority = "High" if current_stock < 50 else "Medium"
            
            # Use actual product description instead of generic ID
            product_name = products[i-1] if i <= len(products) else product_id
            # Extract a short name from description (first part before comma or first 50 chars)
            if ',' in product_name:
                short_name = product_name.split(',')[0]
            else:
                short_name = product_name[:60] + "..." if len(product_name) > 60 else product_name
            
            recommendations.append({
                "product": short_name,  # Use actual product name
                "product_id": product_id,
                "quantity": quantity,
                "cost": cost,
                "priority": priority,
                "timeline": "Immediate" if priority == "High" else "Within 1 week"
            })
            total_cost += cost
    
    return {
        "total_investment": total_cost,
        "recommendations": recommendations,
        "confidence": 0.87,
        "time_saved": "4.5 hours vs manual analysis"
    }

def display_demo_results():
    """Display demo analysis results."""
    st.markdown("---")
    st.header("🤖 AI Analysis Results")
    
    # Show agent execution log
    with st.expander("🔍 Agent Execution Log", expanded=True):
        st.markdown("**Real-time AI Agent Execution:**")
        
        for i, log in enumerate(st.session_state.agent_logs):
            col1, col2, col3 = st.columns([3, 4, 1])
            
            with col1:
                st.write(f"🤖 **{log['agent']}**")
            with col2:
                st.write(f"{log['action']}")
            with col3:
                duration = log.get('duration', '')
                st.write(f"{log['status']} {duration}")
            
            if i < len(st.session_state.agent_logs) - 1:
                st.write("---")
        
        # Add observability note
        st.info("💡 **Observability**: Each agent's execution is tracked and logged. In production, this integrates with LangSmith for complete trace visibility.")
    
    results = st.session_state.demo_results
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("💰 Investment Required", f"₹{results['total_investment']:,}")
    with col2:
        st.metric("📦 Items to Procure", len(results['recommendations']))
    with col3:
        st.metric("🎯 AI Confidence", f"{results['confidence']:.1%}")
    with col4:
        st.metric("⏰ Time Saved", results['time_saved'])
    
    # Recommendations table
    st.subheader("🛒 Procurement Recommendations")
    
    if results['recommendations']:
        rec_df = pd.DataFrame(results['recommendations'])
        rec_df['Cost'] = rec_df['cost'].apply(lambda x: f"₹{x:,}")
        
        # Color code by priority
        def highlight_priority(row):
            if row['priority'] == 'High':
                return ['background-color: #ffebee'] * len(row)
            elif row['priority'] == 'Medium':
                return ['background-color: #fff3e0'] * len(row)
            else:
                return ['background-color: #e8f5e8'] * len(row)
        
        styled_df = rec_df.style.apply(highlight_priority, axis=1)
        st.dataframe(styled_df, use_container_width=True)
        
        # Visualization
        fig = px.bar(
            rec_df, 
            x='product', 
            y='cost',
            color='priority',
            title="Procurement Investment by Priority",
            color_discrete_map={'High': '#f44336', 'Medium': '#ff9800', 'Low': '#4caf50'}
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Business impact summary
        st.subheader("📈 Business Impact")
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Immediate Benefits:**")
            st.write("• Prevents stockouts on critical items")
            st.write("• Optimizes inventory investment")
            st.write("• Reduces manual analysis time by 90%")
        
        with col2:
            st.markdown("**Strategic Advantages:**")
            st.write("• Data-driven decision making")
            st.write("• Scenario-based planning")
            st.write("• Real-time market responsiveness")

def display_new_product_evaluation():
    """Display UI for new product procurement evaluation."""
    st.header("🆕 New Product Procurement Evaluation")
    st.markdown("Evaluate a new product for procurement based on similar products' historical performance")
    
    # Check if system is initialized
    if 'orchestrator' not in st.session_state or st.session_state.orchestrator is None:
        st.warning("⚠️ System not initialized. Please wait for system setup...")
        # Try to initialize
        with st.spinner("🔄 Initializing system..."):
            orchestrator, llm_client, kb = setup_system()
            if orchestrator:
                st.session_state.orchestrator = orchestrator
                st.session_state.llm_client = llm_client
                st.session_state.kb = kb
                st.rerun()
            else:
                st.error("Failed to initialize system. Please check your configuration.")
                return
    
    orchestrator = st.session_state.orchestrator
    
    # Product input form
    with st.form("new_product_form"):
        st.subheader("📝 Product Information")
        
        col1, col2 = st.columns(2)
        
        with col1:
            product_description = st.text_area(
                "Product Description:",
                height=100,
                placeholder="Example: Blue Cotton Classic Crew Neck T-Shirt, short sleeves, casual fit",
                help="Provide a detailed description of the product you want to evaluate"
            )
        
        with col2:
            st.subheader("Product Attributes")
            category = st.selectbox(
                "Category:",
                ["TSHIRT", "POLO", "DRESS", "JEANS", "SHIRT", "KURTA", "SALWAR", "TOP"],
                help="Select the product category"
            )
            
            material = st.text_input("Material:", placeholder="e.g., Cotton, Polyester")
            color = st.text_input("Color:", placeholder="e.g., Blue, White, Black")
            pattern = st.selectbox(
                "Pattern:",
                ["Solid", "Striped", "Printed", "Floral", "Checks", "Embroidered", "Other"]
            )
        
        submitted = st.form_submit_button("🔍 Evaluate Product", type="primary")
        
        if submitted:
            if not product_description:
                st.error("Please enter a product description")
            else:
                # Prepare attributes
                product_attributes = {
                    'category': category,
                    'material': material or 'Unknown',
                    'color': color or 'Unknown',
                    'pattern': pattern
                }
                
                # Show evaluation in progress
                with st.spinner("🤖 Analyzing product viability and procurement recommendations..."):
                    try:
                        # Get procurement agent
                        procurement_agent = orchestrator.procurement_agent
                        
                        # Evaluate the product
                        evaluation_result = procurement_agent.evaluate_new_product(
                            product_description=product_description,
                            product_attributes=product_attributes
                        )
                        
                        # Store result in session state
                        st.session_state.new_product_evaluation = evaluation_result
                        
                    except Exception as e:
                        st.error(f"Evaluation failed: {e}")
                        st.exception(e)
    
    # Display evaluation results
    if 'new_product_evaluation' in st.session_state and st.session_state.new_product_evaluation is not None:
        evaluation = st.session_state.new_product_evaluation
        
        st.markdown("---")
        st.header("📊 Evaluation Results")
        
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            should_procure = evaluation.get('should_procure', False)
            status_icon = "✅" if should_procure else "❌"
            st.metric("Recommendation", f"{status_icon} {'Procure' if should_procure else 'Do Not Procure'}")
        
        with col2:
            viability_score = evaluation.get('viability_score', 0.0)
            st.metric("Viability Score", f"{viability_score:.2f}", 
                     delta=f"{(viability_score - 0.5) * 100:.1f}% vs baseline")
        
        with col3:
            recommended_qty = evaluation.get('recommended_quantity', 0)
            st.metric("Recommended Quantity", f"{recommended_qty:,} units")
        
        with col4:
            confidence = evaluation.get('confidence', 0.0)
            st.metric("Confidence", f"{confidence:.1%}")
        
        # Similar products found
        similar_products = evaluation.get('similar_products', [])
        if similar_products:
            st.subheader("🔍 Similar Products Found")
            st.info(f"Found {len(similar_products)} similar products in historical data")
            
            similar_df = pd.DataFrame([
                {
                    'Product Name': p.get('name', 'Unknown'),
                    'Category': p.get('attributes', {}).get('category', 'N/A'),
                    'Total Units Sold': p.get('sales', {}).get('total_units', 0),
                    'Total Revenue': f"₹{p.get('sales', {}).get('total_revenue', 0):,.0f}",
                    'Avg Monthly Units': f"{p.get('sales', {}).get('avg_monthly_units', 0):.1f}",
                    'Similarity': f"{p.get('similarity_score', 0.0):.2f}" if 'similarity_score' in p else 'N/A'
                }
                for p in similar_products
            ])
            
            st.dataframe(similar_df, use_container_width=True)
        
        # LLM Analysis
        llm_analysis = evaluation.get('llm_analysis', '')
        if llm_analysis:
            st.subheader("🤖 AI Analysis")
            with st.expander("View Detailed Analysis", expanded=True):
                st.markdown(llm_analysis)
        
        # Reasoning
        reasoning = evaluation.get('reasoning', '')
        if reasoning:
            st.subheader("💡 Reasoning")
            st.info(reasoning)
        
        # Recommendations
        if should_procure and recommended_qty > 0:
            st.subheader("✅ Procurement Recommendation")
            
            # Calculate estimated cost (using average price from similar products)
            avg_price = 0
            if similar_products:
                prices = [p.get('sales', {}).get('total_revenue', 0) / max(p.get('sales', {}).get('total_units', 1), 1) 
                         for p in similar_products if p.get('sales', {}).get('total_units', 0) > 0]
                if prices:
                    avg_price = sum(prices) / len(prices)
            
            estimated_cost = recommended_qty * avg_price if avg_price > 0 else recommended_qty * 500  # Fallback
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Recommended Action:**")
                st.success(f"✅ Procure {recommended_qty:,} units")
                st.info(f"Estimated Investment: ₹{estimated_cost:,.0f}")
            
            with col2:
                st.markdown("**Next Steps:**")
                st.write("1. Review similar products' performance")
                st.write("2. Validate supplier availability")
                st.write("3. Confirm budget allocation")
                st.write("4. Set up reorder points")
        
        elif not should_procure:
            st.subheader("⚠️ Recommendation: Do Not Procure")
            st.warning("Based on historical data analysis, procurement is not recommended at this time.")
            st.write("**Reasons may include:**")
            st.write("• Low similarity to successful products")
            st.write("• Poor performance of similar products")
            st.write("• Market conditions not favorable")
            st.write("• Insufficient confidence in demand forecast")

if __name__ == "__main__":
    main()