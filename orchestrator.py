# orchestrator.py

from agents.data_ingestion_agent import DataIngestionAgent
from agents.attribute_analogy_agent import AttributeAnalogyAgent
from agents.demand_forecasting_agent import DemandForecastingAgent
from agents.procurement_allocation_agent import ProcurementAllocationAgent
from shared_knowledge_base import SharedKnowledgeBase

class OrchestratorAgent:
    def __init__(self, llama_client, constraint_params):
        self.llama = llama_client
        self.kb = SharedKnowledgeBase()
        self.data_handler = DataIngestionAgent()
        self.attribute_agent = AttributeAnalogyAgent(self.llama, self.kb)
        self.demand_agent = DemandForecastingAgent(self.llama)
        self.procurement_agent = ProcurementAllocationAgent(constraint_params)

    def run_workflow(self, input_data, price_options):
        clean_data = self.data_handler.run(input_data['files'])
        comparables, trends = self.attribute_agent.run(input_data['product_description'])
        forecast, analysis = self.demand_agent.run(comparables, clean_data['sales'], clean_data['inventory'], clean_data['price'], price_options)
        final_rec = self.procurement_agent.run(forecast, clean_data['inventory'], clean_data['price'])
        return final_rec