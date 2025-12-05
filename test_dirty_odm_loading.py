#!/usr/bin/env python3
"""
Test script to verify dirty ODM data loading.
"""

import sys
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

try:
    from config.settings import Settings
    from shared_knowledge_base import SharedKnowledgeBase
    from utils.llm_client import LLMClientFactory
    from agents.orchestrator_agent import OrchestratorAgent
    
    logger.info("Initializing system...")
    
    # Initialize components
    settings = Settings()
    llm_client = LLMClientFactory.create_client({
        'provider': settings.llm_provider,
        'ollama_base_url': settings.ollama_base_url,
        'ollama_model': settings.ollama_model,
        'openai_api_key': settings.openai_api_key,
        'openai_model': settings.openai_model
    })
    knowledge_base = SharedKnowledgeBase()
    
    # Create orchestrator
    orchestrator = OrchestratorAgent(
        llm_client=llm_client,
        knowledge_base=knowledge_base
    )
    
    logger.info("Loading dirty ODM data...")
    
    # Load dirty ODM data
    dirty_input_file = "data/sample/dirty_odm_input.csv"
    historical_file = "data/sample/odm_historical_dataset_5000.csv"
    
    result = orchestrator.load_dirty_odm_data(dirty_input_file, historical_file)
    
    # Print summary
    summary = result.get('summary', {})
    logger.info("=" * 60)
    logger.info("✅ DIRTY ODM DATA LOADED SUCCESSFULLY")
    logger.info("=" * 60)
    logger.info(f"Dirty Input Products: {summary.get('dirty_input_count', 0)}")
    logger.info(f"Historical Products: {summary.get('historical_products_count', 0)}")
    logger.info(f"Historical Records: {summary.get('historical_records_count', 0)}")
    logger.info(f"Cleaning Applied: {summary.get('cleaning_applied', False)}")
    logger.info("=" * 60)
    
    # Verify data in knowledge base
    logger.info("\nVerifying data in knowledge base...")
    all_products = knowledge_base.get_all_products_with_performance()
    logger.info(f"Total products in knowledge base: {len(all_products)}")
    
    # Show sample products
    logger.info("\nSample products in knowledge base:")
    for i, product in enumerate(all_products[:5]):
        logger.info(f"  {i+1}. {product.get('product_id', 'N/A')}: {product.get('description', 'N/A')[:50]}...")
    
    logger.info("\n✅ Test completed successfully!")
    
except Exception as e:
    logger.error(f"❌ Test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

