"""
LangGraph-based Orchestrator for Demand Forecasting Module
Uses LangChain and LangGraph to orchestrate the multi-agent workflow
"""

from typing import Dict, List, Any, Optional, TypedDict, Annotated
import pandas as pd
import concurrent.futures
try:
    from langgraph.graph import StateGraph, END
    from langchain_core.messages import HumanMessage, SystemMessage
    from langchain_core.prompts import ChatPromptTemplate
    LANGGRAPH_AVAILABLE = True
except ImportError:
    LANGGRAPH_AVAILABLE = False
    StateGraph = None
    END = None

from agents.attribute_analogy_agent import AttributeAnalogyAgent
from agents.demand_forecasting_agent import DemandForecastingAgent
from agents.data_ingestion_agent import DataIngestionAgent
from utils.ollama_client import OllamaClient
from utils.audit_logger import AuditLogger, LogStatus
from shared_knowledge_base import SharedKnowledgeBase


class ForecastState(TypedDict, total=False):
    """State for the demand forecasting workflow"""
    # Input data
    sales_data: pd.DataFrame
    inventory_data: pd.DataFrame
    price_data: pd.DataFrame
    cost_data: Optional[pd.DataFrame]
    new_articles_data: pd.DataFrame
    
    # Parameters
    forecast_horizon_days: int
    variance_threshold: float
    margin_target: float
    max_quantity_per_store: int
    universe_of_stores: Optional[int]
    price_options: List[float]
    
    # Intermediate results
    product_attributes: Dict[str, Any]
    comparables: List[Dict]
    model_selection: Optional[Dict[str, Any]]
    factor_analysis: Optional[Dict[str, Any]]
    demand_agent: Any  # DemandForecastingAgent instance (reused across nodes)
    
    # Final results
    forecast_results: Optional[Dict[str, Any]]
    recommendations: Optional[Dict[str, Any]]
    sensitivity_analysis: Optional[Dict[str, Any]]
    
    # Metadata
    run_id: str
    audit_logger: AuditLogger
    errors: List[str]
    status: str  # "in_progress", "completed", "failed"


class DemandForecastingOrchestrator:
    """
    LangGraph-based orchestrator for demand forecasting workflow
    
    Workflow:
    1. Extract Product Attributes (parallel for all articles)
    2. Find Comparables (parallel for all articles)
    3. Select Forecasting Model
    4. Run Sensitivity Analysis
    5. Generate Store-Level Forecasts
    6. Generate Recommendations
    """
    
    def __init__(
        self,
        ollama_client: OllamaClient,
        audit_logger: AuditLogger,
        knowledge_base: Optional[SharedKnowledgeBase] = None
    ):
        """
        Initialize the orchestrator
        
        Args:
            ollama_client: Ollama client for LLM interactions
            audit_logger: Audit logger for tracking operations
            knowledge_base: Shared knowledge base (optional)
        """
        self.ollama_client = ollama_client
        self.audit_logger = audit_logger
        self.kb = knowledge_base or SharedKnowledgeBase()
        
        # Initialize agents (reused across workflow)
        self.data_ingestion_agent = DataIngestionAgent(
            ollama_client=ollama_client,
            audit_logger=audit_logger
        )
        self.attribute_agent = AttributeAnalogyAgent(
            ollama_client=ollama_client,
            knowledge_base=self.kb,
            audit_logger=audit_logger
        )
        # Demand agent will be created per workflow run with specific parameters
        
        # Build the LangGraph workflow
        self.workflow = self._build_workflow()
    
    def _build_workflow(self):
        """Build the LangGraph workflow"""
        if not LANGGRAPH_AVAILABLE:
            raise ImportError("LangGraph is not installed. Please install it with: pip install langgraph langchain")
        
        workflow = StateGraph(ForecastState)
        
        # Add nodes
        workflow.add_node("extract_attributes", self._extract_product_attributes)
        workflow.add_node("find_comparables", self._find_comparables)
        workflow.add_node("select_model", self._select_forecasting_model)
        workflow.add_node("sensitivity_analysis", self._run_sensitivity_analysis)
        workflow.add_node("store_forecasting", self._run_store_forecasting)
        workflow.add_node("generate_recommendations", self._generate_recommendations)
        
        # Define the flow
        workflow.set_entry_point("extract_attributes")
        workflow.add_edge("extract_attributes", "find_comparables")
        workflow.add_edge("find_comparables", "select_model")
        workflow.add_edge("select_model", "sensitivity_analysis")
        workflow.add_edge("sensitivity_analysis", "store_forecasting")
        workflow.add_edge("store_forecasting", "generate_recommendations")
        workflow.add_edge("generate_recommendations", END)
        
        return workflow.compile()
    
    def _extract_product_attributes(self, state: ForecastState) -> ForecastState:
        """Extract product attributes from new articles data (parallel)"""
        try:
            if self.audit_logger:
                self.audit_logger.log_agent_operation(
                    agent_name="Orchestrator",
                    description="Extracting product attributes from new articles",
                    status=LogStatus.IN_PROGRESS,
                    inputs={"num_articles": len(state["new_articles_data"])}
                )
            
            product_attributes = {}
            new_articles_data = state["new_articles_data"]
            
            if new_articles_data is not None and not new_articles_data.empty:
                for _, row in new_articles_data.iterrows():
                    # Use vendor_sku or sku as fallback
                    sku = row.get('vendor_sku', '') or row.get('sku', '') or row.get('product_id', '')
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
            
            state["product_attributes"] = product_attributes
            
            if self.audit_logger:
                self.audit_logger.log_agent_operation(
                    agent_name="Orchestrator",
                    description="Product attributes extracted",
                    status=LogStatus.SUCCESS,
                    outputs={"num_articles": len(product_attributes)}
                )
            
        except Exception as e:
            state["errors"].append(f"Error extracting attributes: {str(e)}")
            state["status"] = "failed"
            if self.audit_logger:
                self.audit_logger.log_agent_operation(
                    agent_name="Orchestrator",
                    description="Failed to extract product attributes",
                    status=LogStatus.FAIL,
                    error=str(e)
                )
        
        return state
    
    def _find_comparables(self, state: ForecastState) -> ForecastState:
        """Find comparables for all new articles (parallel execution)"""
        try:
            if self.audit_logger:
                self.audit_logger.log_agent_operation(
                    agent_name="Orchestrator",
                    description="Finding comparables for new articles (parallel)",
                    status=LogStatus.IN_PROGRESS
                )
            
            comparables = []
            new_articles_data = state["new_articles_data"]
            sales_data = state["sales_data"]
            
            if new_articles_data is not None and not new_articles_data.empty:
                # Prepare all product descriptions
                article_descriptions = []
                for _, row in new_articles_data.iterrows():
                    desc_parts = []
                    for attr in ['description', 'category', 'color', 'segment', 'family', 'class', 'brick', 'material', 'brand']:
                        if attr in row and pd.notna(row[attr]):
                            desc_parts.append(f"{attr}: {row[attr]}")
                    
                    sku = row.get('vendor_sku', '') or row.get('sku', '') or row.get('product_id', '')
                    product_description = ", ".join(desc_parts) if desc_parts else sku
                    article_descriptions.append((product_description, sku))
                
                # Run attribute analogy agent in parallel using thread pool
                import concurrent.futures
                
                def run_agent(description, article_sku):
                    try:
                        comps, _ = self.attribute_agent.run(description, sales_data)
                        return comps
                    except Exception as e:
                        print(f"Error finding comparables for {article_sku}: {e}")
                        return []
                
                # Use ThreadPoolExecutor for parallel execution
                with concurrent.futures.ThreadPoolExecutor(max_workers=min(len(article_descriptions), 10)) as executor:
                    futures = [
                        executor.submit(run_agent, desc, sku)
                        for desc, sku in article_descriptions
                    ]
                    
                    for future in concurrent.futures.as_completed(futures):
                        try:
                            result = future.result()
                            comparables.extend(result)
                        except Exception as e:
                            print(f"Error in parallel comparable search: {e}")
            
            state["comparables"] = comparables
            
            if self.audit_logger:
                self.audit_logger.log_agent_operation(
                    agent_name="Orchestrator",
                    description="Comparables found (parallel execution complete)",
                    status=LogStatus.SUCCESS,
                    outputs={"total_comparables": len(comparables)}
                )
            
        except Exception as e:
            state["errors"].append(f"Error finding comparables: {str(e)}")
            state["status"] = "failed"
            if self.audit_logger:
                self.audit_logger.log_agent_operation(
                    agent_name="Orchestrator",
                    description="Failed to find comparables",
                    status=LogStatus.FAIL,
                    error=str(e)
                )
        
        return state
    
    def _select_forecasting_model(self, state: ForecastState) -> ForecastState:
        """Select the best forecasting model"""
        try:
            if self.audit_logger:
                self.audit_logger.log_agent_operation(
                    agent_name="Orchestrator",
                    description="Selecting forecasting model",
                    status=LogStatus.IN_PROGRESS
                )
            
            # Initialize demand forecasting agent (create once, reuse in later nodes)
            if "demand_agent" not in state:
                state["demand_agent"] = DemandForecastingAgent(
                    ollama_client=self.ollama_client,
                    default_margin_pct=0.40,
                    target_sell_through_pct=0.75,
                    min_margin_pct=0.25,
                    use_llm=True,
                    universe_of_stores=state["universe_of_stores"],
                    enable_hitl=True,
                    variance_threshold=state["variance_threshold"],
                    audit_logger=state["audit_logger"]
                )
            
            demand_agent = state["demand_agent"]
            
            # Select model
            model_selection = demand_agent.select_model(
                state["comparables"],
                state["sales_data"],
                state["inventory_data"],
                state["price_data"]
            )
            
            state["model_selection"] = model_selection
            
            if self.audit_logger:
                self.audit_logger.log_agent_operation(
                    agent_name="Orchestrator",
                    description=f"Model selected: {model_selection.get('selected_model', 'unknown')}",
                    status=LogStatus.SUCCESS,
                    outputs={"model_selection": model_selection}
                )
            
        except Exception as e:
            state["errors"].append(f"Error selecting model: {str(e)}")
            state["status"] = "failed"
            if self.audit_logger:
                self.audit_logger.log_agent_operation(
                    agent_name="Orchestrator",
                    description="Failed to select model",
                    status=LogStatus.FAIL,
                    error=str(e)
                )
        
        return state
    
    def _run_sensitivity_analysis(self, state: ForecastState) -> ForecastState:
        """Run sensitivity and factor analysis"""
        try:
            if self.audit_logger:
                self.audit_logger.log_agent_operation(
                    agent_name="Orchestrator",
                    description="Running sensitivity and factor analysis",
                    status=LogStatus.IN_PROGRESS
                )
            
            # Reuse demand agent from previous step
            demand_agent = state.get("demand_agent")
            if not demand_agent:
                # Create if not exists (shouldn't happen, but safety check)
                demand_agent = DemandForecastingAgent(
                    ollama_client=self.ollama_client,
                    default_margin_pct=0.40,
                    target_sell_through_pct=0.75,
                    min_margin_pct=0.25,
                    use_llm=True,
                    universe_of_stores=state["universe_of_stores"],
                    enable_hitl=True,
                    variance_threshold=state["variance_threshold"],
                    audit_logger=state["audit_logger"]
                )
                state["demand_agent"] = demand_agent
            
            # Run sensitivity analysis
            factor_analysis = demand_agent.sensitivity_and_factor_analysis(
                state["sales_data"],
                state["price_data"],
                state["product_attributes"],
                state["comparables"]
            )
            
            state["factor_analysis"] = factor_analysis
            state["sensitivity_analysis"] = factor_analysis.get("sensitivity_results", {})
            
            if self.audit_logger:
                self.audit_logger.log_agent_operation(
                    agent_name="Orchestrator",
                    description="Sensitivity analysis completed",
                    status="Success"
                )
            
        except Exception as e:
            state["errors"].append(f"Error in sensitivity analysis: {str(e)}")
            state["status"] = "failed"
            if self.audit_logger:
                self.audit_logger.log_agent_operation(
                    agent_name="Orchestrator",
                    description="Failed sensitivity analysis",
                    status=LogStatus.FAIL,
                    error=str(e)
                )
        
        return state
    
    def _run_store_forecasting(self, state: ForecastState) -> ForecastState:
        """Run store-level forecasting"""
        try:
            if self.audit_logger:
                self.audit_logger.log_agent_operation(
                    agent_name="Orchestrator",
                    description="Running store-level forecasting",
                    status=LogStatus.IN_PROGRESS
                )
            
            # Reuse demand agent from previous steps
            demand_agent = state.get("demand_agent")
            if not demand_agent:
                # Create if not exists (shouldn't happen, but safety check)
                demand_agent = DemandForecastingAgent(
                    ollama_client=self.ollama_client,
                    default_margin_pct=0.40,
                    target_sell_through_pct=0.75,
                    min_margin_pct=0.25,
                    use_llm=True,
                    universe_of_stores=state["universe_of_stores"],
                    enable_hitl=True,
                    variance_threshold=state["variance_threshold"],
                    audit_logger=state["audit_logger"]
                )
                state["demand_agent"] = demand_agent
            
            # Set the selected model from previous step
            if state.get("model_selection"):
                demand_agent.selected_model = state["model_selection"].get("selected_model", "hybrid")
                demand_agent.model_params = state["model_selection"].get("model_params", {})
            
            # Run store-level forecasting
            forecast_results = demand_agent.forecast_store_level(
                state["sales_data"],
                state["inventory_data"],
                state["price_data"],
                state["product_attributes"],
                state["comparables"],
                state["forecast_horizon_days"],
                state["cost_data"],
                state["margin_target"],
                state["max_quantity_per_store"]
            )
            
            state["forecast_results"] = forecast_results
            
            if self.audit_logger:
                self.audit_logger.log_agent_operation(
                    agent_name="Orchestrator",
                    description="Store-level forecasting completed",
                    status=LogStatus.SUCCESS,
                    outputs={
                        "stores_forecasted": len(forecast_results.get("store_level_forecasts", {})),
                        "recommendations_count": len(forecast_results.get("recommendations", {}).get("articles_to_buy", []))
                    }
                )
            
        except Exception as e:
            state["errors"].append(f"Error in store forecasting: {str(e)}")
            state["status"] = "failed"
            if self.audit_logger:
                self.audit_logger.log_agent_operation(
                    agent_name="Orchestrator",
                    description="Failed store-level forecasting",
                    status=LogStatus.FAIL,
                    error=str(e)
                )
        
        return state
    
    def _generate_recommendations(self, state: ForecastState) -> ForecastState:
        """Generate final recommendations (already done in forecast_store_level, but extract here)"""
        try:
            if state.get("forecast_results"):
                recommendations = state["forecast_results"].get("recommendations", {})
                state["recommendations"] = recommendations
                state["status"] = "completed"
                
                if self.audit_logger:
                    self.audit_logger.log_agent_operation(
                        agent_name="Orchestrator",
                        description="Demand forecasting workflow completed",
                        status=LogStatus.SUCCESS,
                        outputs={
                            "articles_recommended": len(recommendations.get("articles_to_buy", [])),
                            "total_quantity": recommendations.get("total_procurement_quantity", 0),
                            "total_stores": recommendations.get("total_stores", 0)
                        }
                    )
            else:
                state["status"] = "failed"
                state["errors"].append("No forecast results available")
        
        except Exception as e:
            state["errors"].append(f"Error generating recommendations: {str(e)}")
            state["status"] = "failed"
        
        return state
    
    def run(
        self,
        sales_data: pd.DataFrame,
        inventory_data: pd.DataFrame,
        price_data: pd.DataFrame,
        new_articles_data: pd.DataFrame,
        cost_data: Optional[pd.DataFrame] = None,
        forecast_horizon_days: int = 60,
        variance_threshold: float = 0.05,
        margin_target: float = 0.30,
        max_quantity_per_store: int = 500,
        universe_of_stores: Optional[int] = None,
        price_options: Optional[List[float]] = None,
        run_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Run the complete demand forecasting workflow using LangGraph
        
        Returns:
            Dictionary with forecast_results, recommendations, and sensitivity_analysis
        """
        if price_options is None:
            price_options = [200, 300, 400, 500, 600, 700, 800, 900, 1000]
        
        # Initialize state
        initial_state: ForecastState = {
            "sales_data": sales_data,
            "inventory_data": inventory_data,
            "price_data": price_data,
            "cost_data": cost_data,
            "new_articles_data": new_articles_data,
            "forecast_horizon_days": forecast_horizon_days,
            "variance_threshold": variance_threshold,
            "margin_target": margin_target,
            "max_quantity_per_store": max_quantity_per_store,
            "universe_of_stores": universe_of_stores,
            "price_options": price_options,
            "product_attributes": {},
            "comparables": [],
            "model_selection": None,
            "factor_analysis": None,
            "forecast_results": None,
            "recommendations": None,
            "sensitivity_analysis": None,
            "run_id": run_id or "unknown",
            "audit_logger": self.audit_logger,
            "errors": [],
            "status": "in_progress"
        }
        
        # Run the workflow
        try:
            final_state = self.workflow.invoke(initial_state)
            
            # Return results in expected format (tuple like standard workflow)
            results = {
                "model_selection": final_state.get("model_selection", {}),
                "factor_analysis": final_state.get("factor_analysis", {}),
                "forecast_results": final_state.get("forecast_results", {}),
                "recommendations": final_state.get("recommendations", {}),
                "validation_messages": final_state.get("errors", []),
                "fallback_used": False
            }
            
            # Extract sensitivity analysis from factor_analysis
            sensitivity = final_state.get("factor_analysis", {}).get("sensitivity_results", {})
            
            # Return as tuple to match standard workflow format
            return results, sensitivity
        except Exception as e:
            if self.audit_logger:
                self.audit_logger.log_agent_operation(
                    agent_name="Orchestrator",
                    description="Workflow execution failed",
                    status=LogStatus.FAIL,
                    error=str(e)
                )
            raise
