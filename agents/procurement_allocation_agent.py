"""Procurement Allocation Agent for optimized procurement planning."""

import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from config.agent_configs import PROCUREMENT_AGENT_CONFIG
from shared_knowledge_base import SharedKnowledgeBase
from utils.llm_client import LLMClient, LLMError, retry_llm_call
from utils.prompt_builder import PromptBuilder
from utils.data_summarizer import DataSummarizer

try:
    import pulp
except ImportError:
    pulp = None

logger = logging.getLogger(__name__)

class ProcurementOptimizationError(Exception):
    """Raised when procurement optimization fails."""
    pass

class ProcurementAllocationAgent:
    """
    Agent for optimized procurement allocation and planning.
    
    Uses linear programming optimization to determine optimal procurement
    quantities considering budget, capacity, MOQ, and demand constraints.
    """
    
    def __init__(self, llm_client: LLMClient, knowledge_base: SharedKnowledgeBase, 
                 config: Optional[Dict] = None):
        """
        Initialize with LLM client and knowledge base.
        
        Args:
            llm_client: LLM client instance
            knowledge_base: Shared knowledge base instance
            config: Agent configuration
        """
        self.llm_client = llm_client
        self.knowledge_base = knowledge_base
        self.config = config or PROCUREMENT_AGENT_CONFIG
        self.prompt_builder = PromptBuilder()
        self.data_summarizer = DataSummarizer()
        
        if pulp is None:
            logger.warning("PuLP not available - optimization features limited")
        
        # Default constraints
        self.default_constraints = {
            'budget': 1000000,  # $1M budget
            'storage_capacity': 10000,  # 10k units
            'min_order_quantity': 100,  # MOQ per product
            'supplier_capacity': 5000,  # Max units per supplier
            'lead_time_days': 30,  # Standard lead time
            'safety_stock_ratio': 0.2,  # 20% safety stock
            'max_procurement_horizon': 90  # 90 days
        }
    
    def check_constraints(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Check procurement constraints against demand and inventory data.
        
        Args:
            data: Dictionary containing demand, inventory, and price data
            
        Returns:
            Constraint validation results
        """
        logger.info("Checking procurement constraints")
        
        try:
            demand_forecast = data.get('demand', {})
            current_inventory = data.get('inventory', {})
            pricing_data = data.get('price', {})
            
            constraints = self.config.get('constraints', self.default_constraints)
            
            validation_results = {
                'budget_feasible': True,
                'capacity_feasible': True,
                'moq_feasible': True,
                'lead_time_feasible': True,
                'constraint_violations': [],
                'warnings': [],
                'recommendations': []
            }
            
            # Calculate total procurement needs
            total_procurement_cost = 0
            total_procurement_volume = 0
            
            for product_id, forecast_data in demand_forecast.items():
                if isinstance(forecast_data, dict):
                    forecast_quantity = forecast_data.get('forecast', 0)
                else:
                    forecast_quantity = forecast_data
                
                current_stock = current_inventory.get(product_id, 0)
                safety_stock = forecast_quantity * constraints['safety_stock_ratio']
                procurement_needed = max(0, forecast_quantity + safety_stock - current_stock)
                
                # Get price data
                if isinstance(pricing_data.get(product_id), dict):
                    unit_price = pricing_data[product_id].get('unit_price', 50)
                else:
                    unit_price = pricing_data.get(product_id, 50)
                
                total_procurement_cost += procurement_needed * unit_price
                total_procurement_volume += procurement_needed
                
                # Check MOQ constraints
                if procurement_needed > 0 and procurement_needed < constraints['min_order_quantity']:
                    validation_results['warnings'].append(
                        f"Product {product_id}: Procurement quantity {procurement_needed} below MOQ {constraints['min_order_quantity']}"
                    )
            
            # Budget constraint check
            if total_procurement_cost > constraints['budget']:
                validation_results['budget_feasible'] = False
                validation_results['constraint_violations'].append(
                    f"Budget constraint violated: Required ${total_procurement_cost:,.0f} > Available ${constraints['budget']:,.0f}"
                )
            
            # Capacity constraint check
            if total_procurement_volume > constraints['storage_capacity']:
                validation_results['capacity_feasible'] = False
                validation_results['constraint_violations'].append(
                    f"Storage capacity violated: Required {total_procurement_volume} > Available {constraints['storage_capacity']}"
                )
            
            validation_results['total_cost'] = total_procurement_cost
            validation_results['total_volume'] = total_procurement_volume
            validation_results['budget_utilization'] = total_procurement_cost / constraints['budget']
            validation_results['capacity_utilization'] = total_procurement_volume / constraints['storage_capacity']
            
            logger.info(f"Constraint check completed - Feasible: {validation_results['budget_feasible'] and validation_results['capacity_feasible']}")
            return validation_results
            
        except Exception as e:
            logger.error(f"Constraint checking failed: {e}")
            return {
                'budget_feasible': False,
                'capacity_feasible': False,
                'error': str(e)
            }

    def optimize(self, demand_forecast: Dict[str, Any], inventory: Dict[str, int], 
                price: Dict[str, Any]) -> Dict[str, Any]:
        """
        Optimize procurement allocation using linear programming.
        
        Args:
            demand_forecast: Forecasted demand per product
            inventory: Current inventory levels
            price: Pricing data per product
            
        Returns:
            Optimization results with procurement recommendations
        """
        logger.info("Starting procurement optimization")
        
        if pulp is None:
            return self._fallback_optimization(demand_forecast, inventory, price)
        
        try:
            constraints = self.config.get('constraints', self.default_constraints)
            
            # Create products list
            products = list(demand_forecast.keys())
            
            # Create the optimization problem
            prob = pulp.LpProblem("Procurement_Optimization", pulp.LpMinimize)
            
            # Decision variables: quantity to procure for each product
            procure_qty = {}
            for product in products:
                procure_qty[product] = pulp.LpVariable(
                    f"procure_{product}", 
                    lowBound=0, 
                    cat='Integer'
                )
            
            # Objective: Minimize total procurement cost + holding cost
            total_cost = 0
            for product in products:
                # Get unit price
                if isinstance(price.get(product), dict):
                    unit_price = price[product].get('unit_price', 50)
                else:
                    unit_price = price.get(product, 50)
                
                # Calculate holding cost (simplified)
                holding_cost_rate = 0.02  # 2% of unit price per period
                
                total_cost += procure_qty[product] * (unit_price + unit_price * holding_cost_rate)
            
            prob += total_cost
            
            # Constraints
            
            # 1. Budget constraint
            budget_constraint = 0
            for product in products:
                if isinstance(price.get(product), dict):
                    unit_price = price[product].get('unit_price', 50)
                else:
                    unit_price = price.get(product, 50)
                budget_constraint += procure_qty[product] * unit_price
            
            prob += budget_constraint <= constraints['budget'], "Budget_Constraint"
            
            # 2. Storage capacity constraint
            storage_constraint = 0
            for product in products:
                storage_constraint += procure_qty[product]
            
            prob += storage_constraint <= constraints['storage_capacity'], "Storage_Constraint"
            
            # 3. Demand fulfillment constraints
            for product in products:
                # Calculate required procurement
                if isinstance(demand_forecast[product], dict):
                    forecast_qty = demand_forecast[product].get('forecast', 0)
                else:
                    forecast_qty = demand_forecast[product]
                
                current_stock = inventory.get(product, 0)
                safety_stock = forecast_qty * constraints['safety_stock_ratio']
                min_procurement = max(0, forecast_qty + safety_stock - current_stock)
                
                # Must procure at least the minimum needed
                if min_procurement > 0:
                    prob += procure_qty[product] >= min_procurement, f"Min_Procurement_{product}"
                
                # MOQ constraint
                # Create binary variable for whether we order this product
                order_binary = pulp.LpVariable(f"order_{product}", cat='Binary')
                
                # If we order, must be at least MOQ
                prob += procure_qty[product] >= constraints['min_order_quantity'] * order_binary, f"MOQ_Lower_{product}"
                prob += procure_qty[product] <= constraints['supplier_capacity'] * order_binary, f"MOQ_Upper_{product}"
            
            # Solve the problem
            prob.solve(pulp.PULP_CBC_CMD(msg=0))
            
            # Extract results
            optimization_results = {
                'status': pulp.LpStatus[prob.status],
                'objective_value': pulp.value(prob.objective),
                'procurement_plan': {},
                'summary': {
                    'total_cost': 0,
                    'total_volume': 0,
                    'products_to_order': 0
                }
            }
            
            if prob.status == pulp.LpStatusOptimal:
                for product in products:
                    qty = int(procure_qty[product].varValue or 0)
                    if qty > 0:
                        if isinstance(price.get(product), dict):
                            unit_price = price[product].get('unit_price', 50)
                        else:
                            unit_price = price.get(product, 50)
                        
                        optimization_results['procurement_plan'][product] = {
                            'quantity': qty,
                            'unit_price': unit_price,
                            'total_cost': qty * unit_price,
                            'urgency': self._calculate_urgency(product, demand_forecast, inventory),
                            'supplier_recommendation': self._recommend_supplier(product)
                        }
                        
                        optimization_results['summary']['total_cost'] += qty * unit_price
                        optimization_results['summary']['total_volume'] += qty
                        optimization_results['summary']['products_to_order'] += 1
                
                logger.info(f"Optimization completed successfully - {optimization_results['summary']['products_to_order']} products to order")
            else:
                logger.warning(f"Optimization failed with status: {prob.status}")
                return self._fallback_optimization(demand_forecast, inventory, price)
            
            return optimization_results
            
        except Exception as e:
            logger.error(f"Optimization failed: {e}")
            return self._fallback_optimization(demand_forecast, inventory, price)
    
    def _fallback_optimization(self, demand_forecast: Dict, inventory: Dict, price: Dict) -> Dict[str, Any]:
        """Fallback optimization when PuLP is not available."""
        logger.info("Using fallback optimization")
        
        constraints = self.config.get('constraints', self.default_constraints)
        
        optimization_results = {
            'status': 'Fallback',
            'procurement_plan': {},
            'summary': {
                'total_cost': 0,
                'total_volume': 0,
                'products_to_order': 0
            }
        }
        
        # Simple heuristic: order based on priority
        products_priority = []
        
        for product_id, forecast_data in demand_forecast.items():
            if isinstance(forecast_data, dict):
                forecast_qty = forecast_data.get('forecast', 0)
            else:
                forecast_qty = forecast_data
            
            current_stock = inventory.get(product_id, 0)
            safety_stock = forecast_qty * constraints['safety_stock_ratio']
            procurement_needed = max(0, forecast_qty + safety_stock - current_stock)
            
            if procurement_needed > 0:
                urgency = self._calculate_urgency(product_id, demand_forecast, inventory)
                products_priority.append((product_id, procurement_needed, urgency))
        
        # Sort by urgency (higher is more urgent)
        products_priority.sort(key=lambda x: x[2], reverse=True)
        
        # Allocate within budget and capacity
        remaining_budget = constraints['budget']
        remaining_capacity = constraints['storage_capacity']
        
        for product_id, procurement_needed, urgency in products_priority:
            if isinstance(price.get(product_id), dict):
                unit_price = price[product_id].get('unit_price', 50)
            else:
                unit_price = price.get(product_id, 50)
            
            # Check MOQ
            procurement_qty = max(procurement_needed, constraints['min_order_quantity'])
            total_cost = procurement_qty * unit_price
            
            if total_cost <= remaining_budget and procurement_qty <= remaining_capacity:
                optimization_results['procurement_plan'][product_id] = {
                    'quantity': procurement_qty,
                    'unit_price': unit_price,
                    'total_cost': total_cost,
                    'urgency': urgency,
                    'supplier_recommendation': self._recommend_supplier(product_id)
                }
                
                remaining_budget -= total_cost
                remaining_capacity -= procurement_qty
                optimization_results['summary']['total_cost'] += total_cost
                optimization_results['summary']['total_volume'] += procurement_qty
                optimization_results['summary']['products_to_order'] += 1
        
        return optimization_results
    
    def _calculate_urgency(self, product_id: str, demand_forecast: Dict, inventory: Dict) -> float:
        """Calculate procurement urgency score."""
        if isinstance(demand_forecast.get(product_id), dict):
            forecast_qty = demand_forecast[product_id].get('forecast', 0)
        else:
            forecast_qty = demand_forecast.get(product_id, 0)
        
        current_stock = inventory.get(product_id, 0)
        
        if forecast_qty == 0:
            return 0.0
        
        # Stock-out risk calculation
        stock_coverage = current_stock / forecast_qty if forecast_qty > 0 else float('inf')
        
        if stock_coverage < 0.5:  # Less than half month coverage
            return 1.0
        elif stock_coverage < 1.0:  # Less than one month coverage
            return 0.7
        else:
            return 0.3
    
    def _recommend_supplier(self, product_id: str) -> Dict[str, Any]:
        """Recommend supplier for product."""
        # Simplified supplier recommendation
        suppliers = [
            {'name': 'Supplier A', 'reliability': 0.9, 'lead_time': 20, 'cost_factor': 1.0},
            {'name': 'Supplier B', 'reliability': 0.8, 'lead_time': 15, 'cost_factor': 1.1},
            {'name': 'Supplier C', 'reliability': 0.85, 'lead_time': 25, 'cost_factor': 0.95}
        ]
        
        # For now, recommend based on reliability
        best_supplier = max(suppliers, key=lambda s: s['reliability'])
        return best_supplier

    def generate_recommendation(self, optimization_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert optimization output to actionable procurement recommendations.
        
        Args:
            optimization_result: Results from optimization
            
        Returns:
            Actionable procurement recommendations
        """
        logger.info("Generating procurement recommendations")
        
        try:
            if not optimization_result.get('procurement_plan'):
                return {
                    'recommendations': [],
                    'summary': 'No procurement needed at this time',
                    'total_investment': 0,
                    'confidence': 0.5
                }
            
            recommendations = []
            
            for product_id, plan in optimization_result['procurement_plan'].items():
                recommendation = {
                    'product_id': product_id,
                    'action': 'PROCURE',
                    'quantity': plan['quantity'],
                    'estimated_cost': plan['total_cost'],
                    'urgency': plan['urgency'],
                    'priority': 'High' if plan['urgency'] > 0.7 else 'Medium' if plan['urgency'] > 0.4 else 'Low',
                    'supplier': plan['supplier_recommendation'],
                    'justification': self._generate_justification(product_id, plan),
                    'timeline': self._generate_timeline(plan['urgency'])
                }
                recommendations.append(recommendation)
            
            # Sort by urgency and cost
            recommendations.sort(key=lambda x: (x['urgency'], x['estimated_cost']), reverse=True)
            
            # Generate LLM insights
            llm_insights = self._get_llm_procurement_insights(optimization_result)
            
            final_recommendations = {
                'recommendations': recommendations,
                'summary': optimization_result['summary'],
                'optimization_status': optimization_result.get('status', 'Unknown'),
                'total_investment': optimization_result['summary']['total_cost'],
                'products_count': optimization_result['summary']['products_to_order'],
                'insights': llm_insights,
                'confidence': 0.85 if optimization_result.get('status') == 'Optimal' else 0.7,
                'generated_at': datetime.now().isoformat()
            }
            
            logger.info(f"Generated {len(recommendations)} procurement recommendations")
            return final_recommendations
            
        except Exception as e:
            logger.error(f"Failed to generate recommendations: {e}")
            return {
                'recommendations': [],
                'error': str(e),
                'confidence': 0.0
            }
    
    def _generate_justification(self, product_id: str, plan: Dict) -> str:
        """Generate justification for procurement recommendation."""
        urgency_desc = "high" if plan['urgency'] > 0.7 else "medium" if plan['urgency'] > 0.4 else "low"
        return f"Procurement needed due to {urgency_desc} urgency. Recommended quantity: {plan['quantity']} units."
    
    def _generate_timeline(self, urgency: float) -> str:
        """Generate timeline based on urgency."""
        if urgency > 0.7:
            return "Order immediately - within 1-2 days"
        elif urgency > 0.4:
            return "Order within 1 week"
        else:
            return "Order within 2 weeks"
    
    def _get_llm_procurement_insights(self, optimization_result: Dict) -> str:
        """Get LLM-generated procurement insights."""
        try:
            if not self.llm_client.is_available():
                return "Standard procurement recommendations based on demand forecast and inventory levels."
            
            prompt_template = self.config.get('prompt_templates', {}).get('procurement_insights', """
Analyze the following procurement optimization results and provide strategic insights:

Optimization Results:
{results}

Provide insights on:
1. Overall procurement strategy effectiveness
2. Budget allocation efficiency  
3. Risk factors and mitigation strategies
4. Seasonal considerations
5. Supplier diversification opportunities

Keep response concise and actionable.
""")
            
            prompt = prompt_template.format(results=str(optimization_result))
            
            response = retry_llm_call(
                self.llm_client,
                prompt,
                max_retries=2,
                temperature=0.3,
                max_tokens=500
            )
            
            return response['response']
            
        except Exception as e:
            logger.warning(f"Failed to get LLM procurement insights: {e}")
            return f"Optimization completed successfully with {optimization_result['summary']['products_to_order']} products recommended for procurement."

    def evaluate_new_product(self, product_description: str, 
                            product_attributes: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluate a new product for procurement using LLM with historical data context.
        
        Args:
            product_description: Description of the new product
            product_attributes: Attributes of the new product
            
        Returns:
            Procurement evaluation with viability score and recommendations
        """
        logger.info(f"Evaluating new product: {product_description}")
        
        try:
            # Find similar products from knowledge base
            similar_products = self.knowledge_base.find_similar_products(
                query_attributes=product_attributes,
                query_description=product_description,
                top_k=5
            )
            
            # Get sales data for similar products
            similar_products_data = []
            for similar in similar_products:
                product_id = similar['product_id']
                perf_data = self.knowledge_base.get_product_performance(product_id)
                
                product_data = {
                    'product_id': product_id,
                    'name': similar.get('description', 'Unknown'),
                    'attributes': similar.get('attributes', {}),
                    'similarity_score': similar.get('similarity_score', 0.0)
                }
                
                # Extract sales data from performance records
                if perf_data:
                    latest_perf = perf_data[-1].get('data', {})
                    product_data['sales'] = {
                        'total_units': latest_perf.get('total_units_sold', 0),
                        'total_revenue': latest_perf.get('total_revenue', 0),
                        'avg_monthly_units': latest_perf.get('avg_monthly_sales', 0)
                    }
                
                similar_products_data.append(product_data)
            
            # Also get from data summarizer
            data_similar = self.data_summarizer.get_similar_products_data(
                attributes=product_attributes,
                top_k=3
            )
            
            # Merge and deduplicate
            existing_ids = {p['product_id'] for p in similar_products_data}
            for data_product in data_similar:
                if data_product['product_id'] not in existing_ids:
                    similar_products_data.append(data_product)
            
            # Build prompt with data context
            prompt = self.prompt_builder.build_procurement_prompt(
                product_description=product_description,
                product_attributes=product_attributes,
                similar_products_data=similar_products_data
            )
            
            # Get LLM recommendation
            response = retry_llm_call(
                self.llm_client,
                prompt,
                max_retries=3,
                temperature=0.2,
                max_tokens=1000
            )
            
            # Parse response (try to extract structured data)
            llm_response = response.get('response', '')
            
            # Try to extract viability score
            viability_score = 0.5  # Default
            if 'viability' in llm_response.lower() or 'score' in llm_response.lower():
                import re
                score_match = re.search(r'(?:viability|score)[:\s]+([0-9.]+)', llm_response, re.IGNORECASE)
                if score_match:
                    try:
                        viability_score = float(score_match.group(1))
                        if viability_score > 1.0:
                            viability_score = viability_score / 100.0  # Convert percentage
                    except:
                        pass
            
            # Try to extract recommended quantity
            recommended_quantity = 0
            if 'procure' in llm_response.lower() or 'quantity' in llm_response.lower():
                qty_match = re.search(r'(?:procure|quantity|units?)[:\s]+([0-9,]+)', llm_response, re.IGNORECASE)
                if qty_match:
                    try:
                        recommended_quantity = int(qty_match.group(1).replace(',', ''))
                    except:
                        pass
            
            # Determine recommendation
            should_procure = viability_score >= 0.6 and recommended_quantity > 0
            
            evaluation_result = {
                'should_procure': should_procure,
                'viability_score': viability_score,
                'recommended_quantity': recommended_quantity,
                'similar_products_found': len(similar_products_data),
                'similar_products': similar_products_data[:3],  # Top 3
                'llm_analysis': llm_response,
                'reasoning': self._extract_reasoning(llm_response),
                'confidence': min(1.0, len(similar_products_data) / 5.0)  # More similar products = higher confidence
            }
            
            logger.info(f"Product evaluation completed - Viability: {viability_score:.2f}, Should Procure: {should_procure}")
            return evaluation_result
            
        except Exception as e:
            logger.error(f"Failed to evaluate new product: {e}")
            return {
                'should_procure': False,
                'viability_score': 0.0,
                'recommended_quantity': 0,
                'error': str(e),
                'confidence': 0.0
            }
    
    def _extract_reasoning(self, llm_response: str) -> str:
        """Extract reasoning from LLM response."""
        # Try to find reasoning section
        reasoning_keywords = ['reasoning', 'analysis', 'because', 'due to', 'considering']
        lines = llm_response.split('\n')
        reasoning_lines = []
        
        for i, line in enumerate(lines):
            if any(keyword in line.lower() for keyword in reasoning_keywords):
                # Take next few lines as reasoning
                reasoning_lines.extend(lines[i:min(i+5, len(lines))])
                break
        
        return '\n'.join(reasoning_lines) if reasoning_lines else llm_response[:500]
    
    def run(self, demand_forecast: Dict[str, Any], inventory: Dict[str, int], 
            price: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute complete procurement allocation workflow.
        
        Args:
            demand_forecast: Forecasted demand per product
            inventory: Current inventory levels  
            price: Pricing data per product
            
        Returns:
            Complete procurement recommendations
        """
        logger.info("Starting procurement allocation workflow")
        
        try:
            # Step 1: Check constraints
            constraint_results = self.check_constraints({
                'demand': demand_forecast,
                'inventory': inventory, 
                'price': price
            })
            
            if not constraint_results.get('budget_feasible') or not constraint_results.get('capacity_feasible'):
                logger.warning("Constraints violated - providing constrained optimization")
            
            # Step 2: Optimize procurement
            optimization_results = self.optimize(demand_forecast, inventory, price)
            
            # Step 3: Generate recommendations
            recommendations = self.generate_recommendation(optimization_results)
            
            # Add constraint information to final result
            recommendations['constraint_analysis'] = constraint_results
            
            logger.info("Procurement allocation workflow completed successfully")
            return recommendations
            
        except Exception as e:
            logger.error(f"Procurement allocation workflow failed: {e}")
            return {
                'recommendations': [],
                'error': str(e),
                'confidence': 0.0
            }