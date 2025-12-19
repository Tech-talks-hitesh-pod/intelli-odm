"""
Procurement Allocation Agent
Optimizes procurement quantities and store allocations based on demand forecasts,
constraints (budget, MOQ, capacity), and business rules.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from datetime import datetime
from utils.ollama_client import OllamaClient
from utils.audit_logger import AuditLogger, LogStatus


class ProcurementAllocationAgent:
    """
    Procurement Allocation Agent
    Optimizes procurement and allocation decisions based on:
    - Demand forecasts
    - Budget constraints
    - Minimum Order Quantities (MOQ)
    - Store capacity constraints
    - Margin targets
    """
    
    def __init__(
        self,
        ollama_client: OllamaClient,
        audit_logger: Optional[AuditLogger] = None,
        budget_constraint: Optional[float] = None,
        moq_constraints: Optional[Dict[str, int]] = None,
        store_capacity: Optional[Dict[str, int]] = None
    ):
        """
        Initialize Procurement Allocation Agent
        
        Args:
            ollama_client: Ollama client for LLM interactions
            audit_logger: Audit logger for tracking operations
            budget_constraint: Total budget constraint (optional)
            moq_constraints: Minimum Order Quantity per article {article: moq}
            store_capacity: Maximum capacity per store {store_id: capacity}
        """
        self.ollama_client = ollama_client
        self.audit_logger = audit_logger
        self.budget_constraint = budget_constraint
        self.moq_constraints = moq_constraints or {}
        self.store_capacity = store_capacity or {}
    
    def check_constraints(
        self,
        demand_forecast: Dict[str, Any],
        inventory_data: pd.DataFrame,
        price_data: pd.DataFrame,
        cost_data: Optional[pd.DataFrame] = None
    ) -> Dict[str, Any]:
        """
        Check constraints against demand forecast
        
        Args:
            demand_forecast: Demand forecast results
            inventory_data: Current inventory data
            price_data: Price data
            cost_data: Cost data (optional)
        
        Returns:
            Constraint validation results
        """
        if self.audit_logger:
            self.audit_logger.log_agent_operation(
                agent_name="ProcurementAllocationAgent",
                description="Checking constraints against demand forecast",
                status=LogStatus.IN_PROGRESS
            )
        
        recommendations = demand_forecast.get("recommendations", {})
        store_allocations = recommendations.get("store_allocations", {})
        articles_to_buy = recommendations.get("articles_to_buy", [])
        
        constraint_results = {
            "budget_check": {"passed": True, "message": "No budget constraint"},
            "moq_check": {},
            "capacity_check": {},
            "violations": []
        }
        
        # Budget constraint check
        if self.budget_constraint is not None:
            total_cost = 0
            cost_map = {}
            
            if cost_data is not None and not cost_data.empty:
                for _, row in cost_data.iterrows():
                    sku = row.get('sku') or row.get('vendor_sku') or row.get('product_id', '')
                    cost = row.get('cost', 0)
                    if sku:
                        cost_map[sku] = cost
            
            for article in articles_to_buy:
                article_allocations = store_allocations.get(article, {})
                article_total_qty = sum(
                    alloc.get('quantity', 0) 
                    for alloc in article_allocations.values()
                )
                
                # Get cost for article
                article_cost = cost_map.get(article, 0)
                if article_cost == 0:
                    # Try to estimate from price data
                    price_rows = price_data[price_data.get('sku', '') == article]
                    if not price_rows.empty:
                        article_cost = price_rows.iloc[0].get('price', 0) * 0.6  # Assume 60% cost
                
                total_cost += article_total_qty * article_cost
            
            if total_cost > self.budget_constraint:
                constraint_results["budget_check"] = {
                    "passed": False,
                    "message": f"Total cost ({total_cost:.2f}) exceeds budget ({self.budget_constraint:.2f})",
                    "total_cost": total_cost,
                    "budget": self.budget_constraint,
                    "excess": total_cost - self.budget_constraint
                }
                constraint_results["violations"].append("budget")
            else:
                constraint_results["budget_check"] = {
                    "passed": True,
                    "message": f"Budget check passed: {total_cost:.2f} <= {self.budget_constraint:.2f}",
                    "total_cost": total_cost,
                    "budget": self.budget_constraint
                }
        
        # MOQ constraint check
        for article in articles_to_buy:
            article_allocations = store_allocations.get(article, {})
            article_total_qty = sum(
                alloc.get('quantity', 0) 
                for alloc in article_allocations.values()
            )
            
            moq = self.moq_constraints.get(article, 0)
            if moq > 0 and article_total_qty < moq:
                constraint_results["moq_check"][article] = {
                    "passed": False,
                    "message": f"Quantity ({article_total_qty}) below MOQ ({moq})",
                    "quantity": article_total_qty,
                    "moq": moq,
                    "shortfall": moq - article_total_qty
                }
                constraint_results["violations"].append(f"moq_{article}")
            else:
                constraint_results["moq_check"][article] = {
                    "passed": True,
                    "quantity": article_total_qty,
                    "moq": moq
                }
        
        # Store capacity check
        for article in articles_to_buy:
            article_allocations = store_allocations.get(article, {})
            for store_id, allocation in article_allocations.items():
                qty = allocation.get('quantity', 0)
                capacity = self.store_capacity.get(store_id, None)
                
                if capacity is not None and qty > capacity:
                    if store_id not in constraint_results["capacity_check"]:
                        constraint_results["capacity_check"][store_id] = {}
                    
                    constraint_results["capacity_check"][store_id][article] = {
                        "passed": False,
                        "message": f"Quantity ({qty}) exceeds capacity ({capacity})",
                        "quantity": qty,
                        "capacity": capacity,
                        "excess": qty - capacity
                    }
                    constraint_results["violations"].append(f"capacity_{store_id}_{article}")
        
        if self.audit_logger:
            self.audit_logger.log_agent_operation(
                agent_name="ProcurementAllocationAgent",
                description=f"Constraint check completed. Violations: {len(constraint_results['violations'])}",
                status=LogStatus.SUCCESS,
                outputs={"constraint_results": constraint_results}
            )
        
        return constraint_results
    
    def optimize(
        self,
        demand_forecast: Dict[str, Any],
        inventory_data: pd.DataFrame,
        price_data: pd.DataFrame,
        cost_data: Optional[pd.DataFrame] = None
    ) -> Dict[str, Any]:
        """
        Optimize procurement and allocation based on constraints
        
        Args:
            demand_forecast: Demand forecast results
            inventory_data: Current inventory data
            price_data: Price data
            cost_data: Cost data (optional)
        
        Returns:
            Optimized allocations
        """
        if self.audit_logger:
            self.audit_logger.log_agent_operation(
                agent_name="ProcurementAllocationAgent",
                description="Optimizing procurement and allocation",
                status=LogStatus.IN_PROGRESS
            )
        
        recommendations = demand_forecast.get("recommendations", {})
        store_allocations = recommendations.get("store_allocations", {}).copy()
        articles_to_buy = recommendations.get("articles_to_buy", [])
        
        # Check constraints first
        constraint_results = self.check_constraints(
            demand_forecast, inventory_data, price_data, cost_data
        )
        
        optimized_allocations = {}
        adjustments = []
        
        # Build cost map
        cost_map = {}
        if cost_data is not None and not cost_data.empty:
            for _, row in cost_data.iterrows():
                sku = row.get('sku') or row.get('vendor_sku') or row.get('product_id', '')
                cost = row.get('cost', 0)
                if sku:
                    cost_map[sku] = cost
        
        # Optimize each article
        for article in articles_to_buy:
            article_allocations = store_allocations.get(article, {}).copy()
            optimized_article = {}
            
            # Get total quantity
            total_qty = sum(alloc.get('quantity', 0) for alloc in article_allocations.values())
            
            # Apply MOQ constraint
            moq = self.moq_constraints.get(article, 0)
            if moq > 0 and total_qty < moq:
                # Scale up to meet MOQ
                scale_factor = moq / total_qty if total_qty > 0 else 1
                adjustments.append({
                    "article": article,
                    "type": "moq_adjustment",
                    "original_qty": total_qty,
                    "adjusted_qty": moq,
                    "scale_factor": scale_factor
                })
                total_qty = moq
            
            # Apply store capacity constraints
            for store_id, allocation in article_allocations.items():
                qty = allocation.get('quantity', 0)
                capacity = self.store_capacity.get(store_id, None)
                
                if capacity is not None and qty > capacity:
                    # Cap at capacity
                    adjusted_qty = capacity
                    adjustments.append({
                        "article": article,
                        "store_id": store_id,
                        "type": "capacity_adjustment",
                        "original_qty": qty,
                        "adjusted_qty": adjusted_qty
                    })
                    qty = adjusted_qty
                
                optimized_article[store_id] = allocation.copy()
                optimized_article[store_id]['quantity'] = qty
            
            # Apply MOQ scaling if needed
            if moq > 0 and total_qty < moq:
                scale_factor = moq / sum(alloc.get('quantity', 0) for alloc in optimized_article.values())
                for store_id in optimized_article:
                    optimized_article[store_id]['quantity'] = int(
                        optimized_article[store_id]['quantity'] * scale_factor
                    )
            
            optimized_allocations[article] = optimized_article
        
        # Budget constraint optimization (if violated)
        if not constraint_results["budget_check"]["passed"]:
            # Scale down all allocations proportionally
            excess = constraint_results["budget_check"]["excess"]
            total_cost = constraint_results["budget_check"]["total_cost"]
            budget = constraint_results["budget_check"]["budget"]
            
            scale_factor = budget / total_cost
            
            for article in optimized_allocations:
                for store_id in optimized_allocations[article]:
                    optimized_allocations[article][store_id]['quantity'] = int(
                        optimized_allocations[article][store_id]['quantity'] * scale_factor
                    )
            
            adjustments.append({
                "type": "budget_adjustment",
                "scale_factor": scale_factor,
                "original_cost": total_cost,
                "adjusted_cost": budget
            })
        
        if self.audit_logger:
            self.audit_logger.log_agent_operation(
                agent_name="ProcurementAllocationAgent",
                description=f"Optimization completed. Adjustments: {len(adjustments)}",
                status=LogStatus.SUCCESS,
                outputs={
                    "optimized_allocations": optimized_allocations,
                    "adjustments": adjustments
                }
            )
        
        return {
            "optimized_allocations": optimized_allocations,
            "adjustments": adjustments,
            "constraint_results": constraint_results
        }
    
    def generate_recommendation(
        self,
        optimization_result: Dict[str, Any],
        original_forecast: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Generate actionable procurement recommendation
        
        Args:
            optimization_result: Result from optimize()
            original_forecast: Original demand forecast
        
        Returns:
            Procurement recommendation
        """
        optimized_allocations = optimization_result["optimized_allocations"]
        adjustments = optimization_result["adjustments"]
        
        # Calculate totals
        total_quantity = 0
        articles = list(optimized_allocations.keys())
        
        for article, stores in optimized_allocations.items():
            article_qty = sum(alloc.get('quantity', 0) for alloc in stores.values())
            total_quantity += article_qty
        
        recommendation = {
            "articles_to_procure": articles,
            "store_allocations": optimized_allocations,
            "total_procurement_quantity": total_quantity,
            "adjustments_applied": adjustments,
            "optimization_summary": {
                "total_articles": len(articles),
                "total_stores": len(set(
                    store_id 
                    for stores in optimized_allocations.values() 
                    for store_id in stores.keys()
                )),
                "adjustments_count": len(adjustments)
            },
            "generated_at": datetime.now().isoformat()
        }
        
        if self.audit_logger:
            self.audit_logger.log_agent_operation(
                agent_name="ProcurementAllocationAgent",
                description="Procurement recommendation generated",
                status=LogStatus.SUCCESS,
                outputs={"recommendation": recommendation}
            )
        
        return recommendation
    
    def run(
        self,
        demand_forecast: Dict[str, Any],
        inventory_data: pd.DataFrame,
        price_data: pd.DataFrame,
        cost_data: Optional[pd.DataFrame] = None
    ) -> Dict[str, Any]:
        """
        Run complete procurement allocation workflow
        
        Args:
            demand_forecast: Demand forecast results
            inventory_data: Current inventory data
            price_data: Price data
            cost_data: Cost data (optional)
        
        Returns:
            Procurement recommendation
        """
        if self.audit_logger:
            self.audit_logger.log_agent_operation(
                agent_name="ProcurementAllocationAgent",
                description="Starting procurement allocation workflow",
                status=LogStatus.IN_PROGRESS
            )
        
        # Check constraints
        constraint_results = self.check_constraints(
            demand_forecast, inventory_data, price_data, cost_data
        )
        
        # Optimize
        optimization_result = self.optimize(
            demand_forecast, inventory_data, price_data, cost_data
        )
        
        # Generate recommendation
        recommendation = self.generate_recommendation(
            optimization_result, demand_forecast
        )
        
        return recommendation
