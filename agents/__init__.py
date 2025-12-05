"""Intelli-ODM Agents Package

This package contains all the specialized AI agents for the Intelli-ODM system:
- DataIngestionAgent: Data validation and preprocessing
- AttributeAnalogyAgent: Product attribute extraction and similarity
- DemandForecastingAgent: Demand prediction and forecasting
- ProcurementAllocationAgent: Procurement optimization and allocation
- OrchestratorAgent: Agent coordination and workflow management
"""

__version__ = "1.0.0"
__author__ = "Intelli-ODM Team"

# Import all agents for easy access
from .data_ingestion_agent import DataIngestionAgent
from .attribute_analogy_agent import AttributeAnalogyAgent
from .demand_forecasting_agent import DemandForecastingAgent
from .procurement_allocation_agent import ProcurementAllocationAgent
from .orchestrator_agent import OrchestratorAgent

__all__ = [
    "DataIngestionAgent",
    "AttributeAnalogyAgent", 
    "DemandForecastingAgent",
    "ProcurementAllocationAgent",
    "OrchestratorAgent"
]