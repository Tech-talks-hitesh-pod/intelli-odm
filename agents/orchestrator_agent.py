"""Orchestrator Agent for coordinating all agents in the Intelli-ODM system."""

import logging
import json
import asyncio
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
import traceback

from config.agent_configs import ORCHESTRATOR_AGENT_CONFIG
from shared_knowledge_base import SharedKnowledgeBase
from utils.llm_client import LLMClient, LLMError, retry_llm_call

# Import all agents
from agents.data_ingestion_agent import DataIngestionAgent
from agents.attribute_analogy_agent import AttributeAnalogyAgent  
from agents.demand_forecasting_agent import DemandForecastingAgent
from agents.procurement_allocation_agent import ProcurementAllocationAgent

logger = logging.getLogger(__name__)

class OrchestratorError(Exception):
    """Raised when orchestration fails."""
    pass

class OrchestratorAgent:
    """
    Master orchestrator agent that coordinates all other agents.
    
    Manages the complete workflow from data ingestion through procurement
    recommendations, with comprehensive error handling and result synthesis.
    """
    
    def __init__(self, llm_client: LLMClient, knowledge_base: SharedKnowledgeBase,
                 config: Optional[Dict] = None):
        """
        Initialize orchestrator with all sub-agents.
        
        Args:
            llm_client: LLM client instance
            knowledge_base: Shared knowledge base instance
            config: Orchestrator configuration
        """
        self.llm_client = llm_client
        self.knowledge_base = knowledge_base
        self.config = config or ORCHESTRATOR_AGENT_CONFIG
        
        # Initialize all agents
        try:
            self.attribute_agent = AttributeAnalogyAgent(llm_client, knowledge_base)
            # Data ingestion agent needs attribute agent for attribute extraction
            self.data_agent = DataIngestionAgent(llm_client, knowledge_base, self.attribute_agent)
            self.demand_agent = DemandForecastingAgent(llm_client, knowledge_base)
            self.procurement_agent = ProcurementAllocationAgent(llm_client, knowledge_base)
            
            # Processed data cache
            self.processed_data = None
            
            logger.info("Orchestrator initialized successfully with all agents")
            
        except Exception as e:
            logger.error(f"Failed to initialize orchestrator: {e}")
            raise OrchestratorError(f"Orchestrator initialization failed: {e}")
        
        # Execution tracking
        self.execution_history = []
        self.current_session = None
        self.processed_data = None  # Cache for processed demo data
    
    def load_demo_data(self, data_dir: str = "data/sample") -> Dict[str, Any]:
        """
        Load and process demo data using DataIngestionAgent.
        This runs on demo load and prepares all data for forecasting/procurement.
        
        Args:
            data_dir: Directory containing demo data CSV files
            
        Returns:
            Processed data dictionary with validation summary
        """
        logger.info(f"Loading and processing demo data from {data_dir}")
        
        try:
            # Use data ingestion agent to load and process
            processed_data = self.data_agent.load_and_process_demo_data(data_dir)
            
            # Cache processed data
            self.processed_data = processed_data
            
            # Update agents with processed data
            self.procurement_agent.processed_data = processed_data
            
            logger.info("Demo data loaded and processed successfully")
            logger.info(f"  - Products: {len(processed_data.get('products', []))}")
            logger.info(f"  - Sales records: {len(processed_data.get('sales', []))}")
            logger.info(f"  - Inventory records: {len(processed_data.get('inventory', []))}")
            logger.info(f"  - Stores: {len(processed_data.get('stores', []))}")
            logger.info(f"  - Product attributes extracted: {len(processed_data.get('product_attributes', {}))}")
            
            return processed_data
            
        except Exception as e:
            logger.error(f"Failed to load demo data: {e}")
            raise OrchestratorError(f"Demo data loading failed: {e}")
    
    def load_dirty_odm_data(self, dirty_input_file: str, historical_file: str) -> Dict[str, Any]:
        """
        Load dirty ODM data directly without cleaning/validation.
        Stores raw data in vector DB for new product evaluation.
        
        Args:
            dirty_input_file: Path to dirty_odm_input.csv (new products to evaluate)
            historical_file: Path to odm_historical_dataset_5000.csv (historical data)
            
        Returns:
            Dictionary with loaded dirty data
        """
        logger.info(f"Loading dirty ODM data from {dirty_input_file} and {historical_file}")
        
        try:
            # Use data ingestion agent to load dirty ODM data
            processed_data = self.data_agent.load_dirty_odm_data(dirty_input_file, historical_file)
            
            # Cache processed data
            self.processed_data = processed_data
            
            # Update agents with processed data
            self.procurement_agent.processed_data = processed_data
            
            logger.info("✅ Dirty ODM data loaded and stored in vector DB")
            logger.info(f"  - Dirty Input Products: {processed_data.get('summary', {}).get('dirty_input_count', 0)}")
            logger.info(f"  - Historical Products: {processed_data.get('summary', {}).get('historical_products_count', 0)}")
            logger.info(f"  - Historical Records: {processed_data.get('summary', {}).get('historical_records_count', 0)}")
            logger.info(f"  - Cleaning Applied: {processed_data.get('summary', {}).get('cleaning_applied', False)}")
            
            return processed_data
            
        except Exception as e:
            logger.error(f"Failed to load dirty ODM data: {e}")
            raise OrchestratorError(f"Dirty ODM data loading failed: {e}")
    
    def load_odm_data(self, odm_file_path: str) -> Dict[str, Any]:
        """
        Load and process ODM data using DataIngestionAgent.
        This specifically handles dirty ODM input data cleaning and normalization.
        
        Args:
            odm_file_path: Path to dirty_odm_input.csv file
            
        Returns:
            Processed ODM data dictionary with cleaned products and forecasts
        """
        logger.info(f"Loading and processing ODM data from {odm_file_path}")
        
        try:
            # Use data ingestion agent to load and process ODM data
            processed_data = self.data_agent.load_and_process_odm_data(odm_file_path)
            
            # Cache processed data
            self.processed_data = processed_data
            
            # Update agents with processed data
            self.procurement_agent.processed_data = processed_data
            
            logger.info("ODM data loaded and processed successfully")
            logger.info(f"  - ODM Products: {len(processed_data.get('products', []))}")
            logger.info(f"  - Generated sales records: {len(processed_data.get('sales', []))}")
            logger.info(f"  - Inventory records: {len(processed_data.get('inventory', []))}")
            logger.info(f"  - Processing summary: {processed_data.get('processing_summary', {})}")
            
            return processed_data
            
        except Exception as e:
            logger.error(f"Failed to load ODM data: {e}")
            raise OrchestratorError(f"ODM data loading failed: {e}")
    
    def start_session(self, session_id: Optional[str] = None) -> str:
        """
        Start a new orchestration session.
        
        Args:
            session_id: Optional session identifier
            
        Returns:
            Session ID
        """
        if session_id is None:
            session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        self.current_session = {
            'id': session_id,
            'start_time': datetime.now(),
            'status': 'active',
            'results': {},
            'errors': [],
            'metrics': {
                'total_execution_time': 0,
                'agent_execution_times': {},
                'success_rate': 0
            }
        }
        
        logger.info(f"Started orchestration session: {session_id}")
        return session_id
    
    def run_complete_workflow(self, product_descriptions: List[str],
                            inventory_data: Optional[Dict] = None,
                            sales_history: Optional[List[Dict]] = None) -> Dict[str, Any]:
        """
        Execute the complete Intelli-ODM workflow.
        
        Args:
            product_descriptions: List of product descriptions to analyze
            inventory_data: Current inventory levels
            sales_history: Historical sales data
            
        Returns:
            Complete workflow results
        """
        session_id = self.start_session()
        
        try:
            logger.info(f"Starting complete workflow for {len(product_descriptions)} products")
            
            # Phase 1: Data Ingestion and Processing
            phase1_results = self._execute_data_phase(
                product_descriptions, inventory_data, sales_history
            )
            
            # Phase 2: Attribute Analysis and Product Matching
            phase2_results = self._execute_analysis_phase(
                product_descriptions, phase1_results
            )
            
            # Phase 3: Demand Forecasting
            phase3_results = self._execute_forecasting_phase(
                phase1_results, phase2_results
            )
            
            # Phase 4: Procurement Optimization
            phase4_results = self._execute_procurement_phase(
                phase1_results, phase3_results
            )
            
            # Phase 5: Result Synthesis
            final_results = self._synthesize_results(
                phase1_results, phase2_results, phase3_results, phase4_results
            )
            
            # Complete session
            self._complete_session(final_results)
            
            logger.info("Complete workflow executed successfully")
            return final_results
            
        except Exception as e:
            error_msg = f"Workflow execution failed: {e}"
            logger.error(error_msg)
            logger.error(traceback.format_exc())
            
            # Record error and return partial results
            self._handle_workflow_error(e)
            return {
                'success': False,
                'error': error_msg,
                'session_id': session_id,
                'partial_results': self.current_session.get('results', {}),
                'timestamp': datetime.now().isoformat()
            }
    
    def _execute_data_phase(self, product_descriptions: List[str],
                           inventory_data: Optional[Dict],
                           sales_history: Optional[List[Dict]]) -> Dict[str, Any]:
        """Execute Phase 1: Data ingestion and processing."""
        logger.info("Executing Phase 1: Data Ingestion")
        
        phase_start = datetime.now()
        phase_results = {
            'processed_products': {},
            'inventory_analysis': {},
            'data_quality_score': 0.0,
            'errors': []
        }
        
        try:
            # Process each product description
            for i, description in enumerate(product_descriptions):
                try:
                    logger.info(f"Processing product {i+1}/{len(product_descriptions)}")
                    
                    # Ingest and process product data
                    result = self.data_agent.run(description)
                    
                    product_id = f"product_{i+1}"
                    phase_results['processed_products'][product_id] = result
                    
                except Exception as e:
                    error_msg = f"Failed to process product {i+1}: {e}"
                    logger.error(error_msg)
                    phase_results['errors'].append(error_msg)
            
            # Process inventory data if provided
            if inventory_data:
                try:
                    inventory_result = self.data_agent.analyze_inventory(inventory_data)
                    phase_results['inventory_analysis'] = inventory_result
                except Exception as e:
                    error_msg = f"Failed to process inventory: {e}"
                    logger.error(error_msg)
                    phase_results['errors'].append(error_msg)
            
            # Process sales history if provided
            if sales_history:
                try:
                    sales_result = self.data_agent.analyze_sales_history(sales_history)
                    phase_results['sales_analysis'] = sales_result
                except Exception as e:
                    error_msg = f"Failed to process sales history: {e}"
                    logger.error(error_msg)
                    phase_results['errors'].append(error_msg)
            
            # Calculate data quality score
            successful_products = len([p for p in phase_results['processed_products'].values() 
                                     if not p.get('error')])
            total_products = len(product_descriptions)
            phase_results['data_quality_score'] = successful_products / total_products if total_products > 0 else 0
            
            execution_time = (datetime.now() - phase_start).total_seconds()
            self.current_session['metrics']['agent_execution_times']['data_ingestion'] = execution_time
            self.current_session['results']['phase1'] = phase_results
            
            logger.info(f"Phase 1 completed in {execution_time:.2f}s with quality score {phase_results['data_quality_score']:.2f}")
            return phase_results
            
        except Exception as e:
            logger.error(f"Phase 1 failed: {e}")
            phase_results['errors'].append(str(e))
            return phase_results
    
    def _execute_analysis_phase(self, product_descriptions: List[str],
                               phase1_results: Dict[str, Any]) -> Dict[str, Any]:
        """Execute Phase 2: Attribute analysis and product matching."""
        logger.info("Executing Phase 2: Attribute Analysis")
        
        phase_start = datetime.now()
        phase_results = {
            'product_attributes': {},
            'comparable_products': {},
            'trend_analysis': {},
            'errors': []
        }
        
        try:
            # Use ThreadPoolExecutor for parallel processing
            with ThreadPoolExecutor(max_workers=3) as executor:
                future_to_product = {}
                
                # Submit attribute extraction tasks
                for i, description in enumerate(product_descriptions):
                    product_id = f"product_{i+1}"
                    
                    # Skip if product processing failed in phase 1
                    if phase1_results['processed_products'].get(product_id, {}).get('error'):
                        continue
                    
                    future = executor.submit(self._analyze_single_product, description, product_id)
                    future_to_product[future] = product_id
                
                # Collect results
                for future in as_completed(future_to_product):
                    product_id = future_to_product[future]
                    try:
                        attributes, comparables, trends = future.result()
                        
                        phase_results['product_attributes'][product_id] = attributes
                        phase_results['comparable_products'][product_id] = comparables
                        phase_results['trend_analysis'][product_id] = trends
                        
                    except Exception as e:
                        error_msg = f"Analysis failed for {product_id}: {e}"
                        logger.error(error_msg)
                        phase_results['errors'].append(error_msg)
            
            execution_time = (datetime.now() - phase_start).total_seconds()
            self.current_session['metrics']['agent_execution_times']['attribute_analysis'] = execution_time
            self.current_session['results']['phase2'] = phase_results
            
            successful_analyses = len(phase_results['product_attributes'])
            logger.info(f"Phase 2 completed in {execution_time:.2f}s with {successful_analyses} successful analyses")
            return phase_results
            
        except Exception as e:
            logger.error(f"Phase 2 failed: {e}")
            phase_results['errors'].append(str(e))
            return phase_results
    
    def _analyze_single_product(self, description: str, product_id: str) -> Tuple[Dict, List, Dict]:
        """Analyze a single product with attribute agent."""
        try:
            return self.attribute_agent.run(description)
        except Exception as e:
            logger.error(f"Failed to analyze {product_id}: {e}")
            return {}, [], {}
    
    def _execute_forecasting_phase(self, phase1_results: Dict[str, Any],
                                  phase2_results: Dict[str, Any]) -> Dict[str, Any]:
        """Execute Phase 3: Demand forecasting."""
        logger.info("Executing Phase 3: Demand Forecasting")
        
        phase_start = datetime.now()
        phase_results = {
            'demand_forecasts': {},
            'price_analysis': {},
            'market_insights': {},
            'errors': []
        }
        
        try:
            # Prepare forecast data
            for product_id in phase2_results.get('product_attributes', {}):
                try:
                    attributes = phase2_results['product_attributes'][product_id]
                    comparables = phase2_results['comparable_products'].get(product_id, [])
                    
                    # Prepare historical data (using mock data if not available)
                    historical_data = self._prepare_historical_data(product_id, attributes)
                    
                    # Run demand forecasting
                    forecast_result = self.demand_agent.run(
                        attributes=attributes,
                        historical_sales=historical_data.get('sales', []),
                        comparable_products=comparables,
                        market_conditions=historical_data.get('market', {}),
                        time_horizon=30  # 30-day forecast
                    )
                    
                    phase_results['demand_forecasts'][product_id] = forecast_result
                    
                except Exception as e:
                    error_msg = f"Forecasting failed for {product_id}: {e}"
                    logger.error(error_msg)
                    phase_results['errors'].append(error_msg)
            
            # Aggregate market insights
            phase_results['market_insights'] = self._aggregate_market_insights(
                phase_results['demand_forecasts']
            )
            
            execution_time = (datetime.now() - phase_start).total_seconds()
            self.current_session['metrics']['agent_execution_times']['demand_forecasting'] = execution_time
            self.current_session['results']['phase3'] = phase_results
            
            successful_forecasts = len(phase_results['demand_forecasts'])
            logger.info(f"Phase 3 completed in {execution_time:.2f}s with {successful_forecasts} forecasts")
            return phase_results
            
        except Exception as e:
            logger.error(f"Phase 3 failed: {e}")
            phase_results['errors'].append(str(e))
            return phase_results
    
    def _execute_procurement_phase(self, phase1_results: Dict[str, Any],
                                  phase3_results: Dict[str, Any]) -> Dict[str, Any]:
        """Execute Phase 4: Procurement optimization."""
        logger.info("Executing Phase 4: Procurement Optimization")
        
        phase_start = datetime.now()
        phase_results = {
            'procurement_plan': {},
            'optimization_summary': {},
            'recommendations': [],
            'errors': []
        }
        
        try:
            # Prepare procurement data
            demand_data = {}
            inventory_data = {}
            price_data = {}
            
            # Aggregate demand forecasts
            for product_id, forecast in phase3_results.get('demand_forecasts', {}).items():
                if forecast.get('forecast'):
                    demand_data[product_id] = forecast['forecast']
                
                # Use inventory from phase 1 or default
                inventory_data[product_id] = phase1_results.get('inventory_analysis', {}).get(product_id, 100)
                
                # Use price analysis or default
                price_data[product_id] = forecast.get('pricing', {}).get('current_price', 50.0)
            
            if demand_data:
                # Run procurement optimization
                procurement_result = self.procurement_agent.run(
                    demand_forecast=demand_data,
                    inventory=inventory_data,
                    price=price_data
                )
                
                phase_results['procurement_plan'] = procurement_result
                phase_results['recommendations'] = procurement_result.get('recommendations', [])
                phase_results['optimization_summary'] = procurement_result.get('summary', {})
            else:
                phase_results['errors'].append("No valid demand forecasts available for procurement")
            
            execution_time = (datetime.now() - phase_start).total_seconds()
            self.current_session['metrics']['agent_execution_times']['procurement'] = execution_time
            self.current_session['results']['phase4'] = phase_results
            
            recommendations_count = len(phase_results['recommendations'])
            logger.info(f"Phase 4 completed in {execution_time:.2f}s with {recommendations_count} recommendations")
            return phase_results
            
        except Exception as e:
            logger.error(f"Phase 4 failed: {e}")
            phase_results['errors'].append(str(e))
            return phase_results
    
    def _synthesize_results(self, phase1: Dict, phase2: Dict, 
                           phase3: Dict, phase4: Dict) -> Dict[str, Any]:
        """Synthesize results from all phases into final recommendations."""
        logger.info("Synthesizing final results")
        
        try:
            # Calculate overall success metrics
            total_products = len(phase1.get('processed_products', {}))
            successful_analyses = len(phase2.get('product_attributes', {}))
            successful_forecasts = len(phase3.get('demand_forecasts', {}))
            total_recommendations = len(phase4.get('recommendations', []))
            
            success_rate = (successful_analyses / total_products) if total_products > 0 else 0
            
            # Generate executive summary
            executive_summary = self._generate_executive_summary(
                phase1, phase2, phase3, phase4, success_rate
            )
            
            # Compile all errors
            all_errors = []
            for phase in [phase1, phase2, phase3, phase4]:
                all_errors.extend(phase.get('errors', []))
            
            # Create final results structure
            final_results = {
                'success': True,
                'session_id': self.current_session['id'],
                'executive_summary': executive_summary,
                'detailed_results': {
                    'data_processing': phase1,
                    'attribute_analysis': phase2,
                    'demand_forecasting': phase3,
                    'procurement_optimization': phase4
                },
                'key_insights': self._extract_key_insights(phase2, phase3, phase4),
                'recommendations': {
                    'procurement': phase4.get('recommendations', []),
                    'strategic': self._generate_strategic_recommendations(phase2, phase3, phase4)
                },
                'metrics': {
                    'total_products_processed': total_products,
                    'successful_analyses': successful_analyses,
                    'forecasts_generated': successful_forecasts,
                    'procurement_recommendations': total_recommendations,
                    'success_rate': success_rate,
                    'execution_times': self.current_session['metrics']['agent_execution_times']
                },
                'errors': all_errors,
                'confidence_score': self._calculate_confidence_score(phase1, phase2, phase3, phase4),
                'timestamp': datetime.now().isoformat()
            }
            
            return final_results
            
        except Exception as e:
            logger.error(f"Result synthesis failed: {e}")
            return {
                'success': False,
                'error': f"Synthesis failed: {e}",
                'session_id': self.current_session['id'],
                'timestamp': datetime.now().isoformat()
            }
    
    def _prepare_historical_data(self, product_id: str, attributes: Dict) -> Dict[str, Any]:
        """Prepare historical data for forecasting."""
        # This is a simplified version - in real implementation,
        # this would fetch actual historical data
        import random
        from datetime import datetime, timedelta
        
        # Generate mock historical sales data
        sales_data = []
        base_date = datetime.now() - timedelta(days=90)
        
        for i in range(90):
            date = base_date + timedelta(days=i)
            # Generate sales with some seasonality
            seasonal_factor = 1 + 0.3 * np.sin(2 * np.pi * i / 30)  # Monthly cycle
            sales = max(0, int(random.gauss(100, 20) * seasonal_factor))
            
            sales_data.append({
                'date': date.isoformat(),
                'quantity': sales,
                'revenue': sales * 50  # Assume $50 per unit
            })
        
        return {
            'sales': sales_data,
            'market': {
                'season': 'spring',
                'economic_indicator': 1.02,
                'competitor_activity': 'moderate'
            }
        }
    
    def _aggregate_market_insights(self, forecasts: Dict) -> Dict[str, Any]:
        """Aggregate market insights from individual forecasts."""
        insights = {
            'overall_demand_trend': 'stable',
            'high_demand_categories': [],
            'seasonal_patterns': {},
            'price_sensitivity_summary': {}
        }
        
        if not forecasts:
            return insights
        
        # Analyze demand trends across products
        demand_changes = []
        for forecast in forecasts.values():
            if isinstance(forecast.get('forecast'), dict):
                growth = forecast['forecast'].get('trend', {}).get('growth_rate', 0)
                demand_changes.append(growth)
        
        if demand_changes:
            avg_growth = sum(demand_changes) / len(demand_changes)
            if avg_growth > 0.05:
                insights['overall_demand_trend'] = 'increasing'
            elif avg_growth < -0.05:
                insights['overall_demand_trend'] = 'decreasing'
        
        return insights
    
    def _extract_key_insights(self, phase2: Dict, phase3: Dict, phase4: Dict) -> List[str]:
        """Extract key insights from analysis results."""
        insights = []
        
        # Attribute insights
        if phase2.get('product_attributes'):
            categories = [attrs.get('category', 'Unknown') 
                         for attrs in phase2['product_attributes'].values()]
            popular_category = max(set(categories), key=categories.count) if categories else 'Unknown'
            insights.append(f"Most common product category: {popular_category}")
        
        # Demand insights
        if phase3.get('demand_forecasts'):
            total_forecast_demand = sum(
                forecast.get('forecast', {}).get('quantity', 0) 
                for forecast in phase3['demand_forecasts'].values()
            )
            insights.append(f"Total forecasted demand: {total_forecast_demand} units")
        
        # Procurement insights
        if phase4.get('optimization_summary'):
            total_investment = phase4['optimization_summary'].get('total_cost', 0)
            insights.append(f"Recommended procurement investment: ${total_investment:,.0f}")
        
        return insights
    
    def _generate_strategic_recommendations(self, phase2: Dict, phase3: Dict, phase4: Dict) -> List[str]:
        """Generate high-level strategic recommendations."""
        recommendations = []
        
        # Analyze trends and patterns
        if phase2.get('trend_analysis'):
            recommendations.append("Monitor seasonal trends for optimal inventory planning")
        
        if phase3.get('demand_forecasts'):
            recommendations.append("Implement dynamic pricing based on demand forecasts")
        
        if phase4.get('procurement_plan'):
            recommendations.append("Diversify supplier base to reduce procurement risks")
        
        recommendations.append("Establish regular forecasting cycles for continuous optimization")
        
        return recommendations
    
    def _calculate_confidence_score(self, phase1: Dict, phase2: Dict, 
                                   phase3: Dict, phase4: Dict) -> float:
        """Calculate overall confidence score for the analysis."""
        scores = []
        
        # Data quality score
        scores.append(phase1.get('data_quality_score', 0.5))
        
        # Analysis success rate
        total_products = len(phase1.get('processed_products', {}))
        successful_analyses = len(phase2.get('product_attributes', {}))
        analysis_rate = successful_analyses / total_products if total_products > 0 else 0
        scores.append(analysis_rate)
        
        # Forecasting success rate
        successful_forecasts = len(phase3.get('demand_forecasts', {}))
        forecast_rate = successful_forecasts / total_products if total_products > 0 else 0
        scores.append(forecast_rate)
        
        # Procurement optimization success
        has_recommendations = len(phase4.get('recommendations', [])) > 0
        scores.append(1.0 if has_recommendations else 0.3)
        
        return sum(scores) / len(scores)
    
    def _generate_executive_summary(self, phase1: Dict, phase2: Dict, 
                                   phase3: Dict, phase4: Dict, success_rate: float) -> str:
        """Generate executive summary using LLM if available."""
        try:
            if not self.llm_client.is_available():
                return self._generate_simple_summary(phase1, phase2, phase3, phase4, success_rate)
            
            # Prepare data for LLM
            summary_data = {
                'total_products': len(phase1.get('processed_products', {})),
                'successful_analyses': len(phase2.get('product_attributes', {})),
                'forecasts_generated': len(phase3.get('demand_forecasts', {})),
                'procurement_recommendations': len(phase4.get('recommendations', [])),
                'total_investment': phase4.get('optimization_summary', {}).get('total_cost', 0),
                'success_rate': success_rate
            }
            
            prompt_template = self.config.get('prompt_templates', {}).get('executive_summary', """
Generate a concise executive summary for the Intelli-ODM analysis:

Analysis Results:
- Products processed: {total_products}
- Successful analyses: {successful_analyses}  
- Demand forecasts: {forecasts_generated}
- Procurement recommendations: {procurement_recommendations}
- Total recommended investment: ${total_investment:,.0f}
- Overall success rate: {success_rate:.1%}

Provide a brief 2-3 sentence summary highlighting key findings and recommendations.
""")
            
            prompt = prompt_template.format(**summary_data)
            
            response = retry_llm_call(
                self.llm_client,
                prompt,
                max_retries=2,
                temperature=0.3,
                max_tokens=300
            )
            
            return response['response']
            
        except Exception as e:
            logger.warning(f"Failed to generate LLM summary: {e}")
            return self._generate_simple_summary(phase1, phase2, phase3, phase4, success_rate)
    
    def _generate_simple_summary(self, phase1: Dict, phase2: Dict, 
                                phase3: Dict, phase4: Dict, success_rate: float) -> str:
        """Generate simple summary without LLM."""
        total_products = len(phase1.get('processed_products', {}))
        recommendations = len(phase4.get('recommendations', []))
        investment = phase4.get('optimization_summary', {}).get('total_cost', 0)
        
        return (f"Analyzed {total_products} products with {success_rate:.1%} success rate. "
                f"Generated {recommendations} procurement recommendations "
                f"totaling ${investment:,.0f} investment.")
    
    def _complete_session(self, final_results: Dict[str, Any]) -> None:
        """Complete the current session."""
        if self.current_session:
            self.current_session['status'] = 'completed'
            self.current_session['end_time'] = datetime.now()
            self.current_session['metrics']['total_execution_time'] = (
                self.current_session['end_time'] - self.current_session['start_time']
            ).total_seconds()
            self.current_session['metrics']['success_rate'] = final_results.get('metrics', {}).get('success_rate', 0)
            
            self.execution_history.append(self.current_session.copy())
            
            logger.info(f"Session {self.current_session['id']} completed successfully")
    
    def _handle_workflow_error(self, error: Exception) -> None:
        """Handle workflow errors and update session."""
        if self.current_session:
            self.current_session['status'] = 'failed'
            self.current_session['errors'].append({
                'error': str(error),
                'timestamp': datetime.now().isoformat(),
                'traceback': traceback.format_exc()
            })
            
            logger.error(f"Session {self.current_session['id']} failed with error: {error}")
    
    def get_session_status(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get status of a specific session."""
        if self.current_session and self.current_session['id'] == session_id:
            return self.current_session.copy()
        
        for session in self.execution_history:
            if session['id'] == session_id:
                return session.copy()
        
        return None
    
    def get_execution_history(self) -> List[Dict[str, Any]]:
        """Get complete execution history."""
        return self.execution_history.copy()