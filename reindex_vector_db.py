#!/usr/bin/env python3
"""
Script to reindex all products and sales data in the vector database.
This will:
1. Clear existing data from ChromaDB
2. Reload all products from CSV files
3. Re-index everything with fresh embeddings
"""

import logging
import sys
import os
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('reindex.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def clear_vector_database(kb):
    """Clear all data from the vector database."""
    try:
        logger.info("🗑️  Clearing existing vector database...")
        
        # Get collection stats before clearing
        try:
            stats = kb.get_collection_stats()
            products_count = stats.get('products_count', 0)
            logger.info(f"   Current products in DB: {products_count}")
        except:
            logger.info("   Could not get current stats")
        
        # Use the built-in reset method
        kb.reset_collections()
        
        # Recreate collections
        kb._products_collection = kb._get_or_create_collection("products")
        kb._performance_collection = kb._get_or_create_collection("performance")
        
        logger.info("✅ Vector database cleared and ready for reindexing")
        return True
        
    except Exception as e:
        logger.error(f"❌ Failed to clear vector database: {e}")
        return False

def reindex_all_data(kb, data_agent):
    """Reindex all products and sales data."""
    try:
        logger.info("="*80)
        logger.info("REINDEXING VECTOR DATABASE")
        logger.info("="*80)
        
        # Step 1: Clear existing data
        if not clear_vector_database(kb):
            logger.error("Failed to clear database. Aborting reindex.")
            return False
        
        # Step 2: Load and index data
        logger.info("\n📦 Loading and indexing ODM data...")
        
        # Paths to data files
        dirty_input_file = "data/sample/dirty_odm_input.csv"
        historical_file = "data/sample/odm_historical_dataset_5000.csv"
        
        # Check if files exist
        if not os.path.exists(dirty_input_file):
            logger.error(f"❌ Data file not found: {dirty_input_file}")
            return False
        
        if not os.path.exists(historical_file):
            logger.error(f"❌ Data file not found: {historical_file}")
            return False
        
        # Load and index data
        result = data_agent.load_dirty_odm_data(dirty_input_file, historical_file)
        
        if not result:
            logger.error("❌ Failed to load ODM data")
            return False
        
        # Step 3: Verify indexing
        logger.info("\n✅ Verifying indexed data...")
        stats = kb.get_collection_stats()
        products_count = stats.get('products_count', 0)
        
        logger.info("="*80)
        logger.info("REINDEXING COMPLETE")
        logger.info("="*80)
        logger.info(f"✅ Products indexed: {products_count}")
        logger.info(f"✅ Dirty input products: {result.get('summary', {}).get('dirty_input_count', 0)}")
        logger.info(f"✅ Historical products: {result.get('summary', {}).get('historical_products_count', 0)}")
        logger.info(f"✅ Historical records: {result.get('summary', {}).get('historical_records_count', 0)}")
        logger.info("="*80)
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Reindexing failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main function to reindex vector database."""
    logger.info("🚀 Starting vector database reindexing...")
    
    try:
        # Import required modules
        from shared_knowledge_base import SharedKnowledgeBase
        from agents.data_ingestion_agent import DataIngestionAgent
        from utils.llm_client import LLMClientFactory, LLMClient
        from config.settings import settings
        
        # Initialize knowledge base
        logger.info("📚 Initializing knowledge base...")
        kb = SharedKnowledgeBase()
        logger.info("✅ Knowledge base initialized")
        
        # Initialize LLM client (optional, for data agent)
        llm_client = None
        try:
            llm_config = {
                'provider': settings.llm_provider,
                'api_key': settings.openai_api_key if settings.llm_provider == 'openai' else None,
                'model': settings.openai_model if settings.llm_provider == 'openai' else settings.ollama_model,
                'temperature': 0.1,
                'base_url': settings.ollama_base_url if settings.llm_provider == 'ollama' else None,
                'timeout': settings.ollama_timeout if settings.llm_provider == 'ollama' else 300
            }
            llm_client = LLMClientFactory.create_client(llm_config)
            logger.info("✅ LLM client initialized (optional)")
        except Exception as e:
            logger.warning(f"⚠️  LLM client not available: {e} (continuing without it)")
        
        # Initialize data ingestion agent
        logger.info("📥 Initializing data ingestion agent...")
        data_agent = DataIngestionAgent(llm_client, kb)
        logger.info("✅ Data ingestion agent initialized")
        
        # Reindex all data
        success = reindex_all_data(kb, data_agent)
        
        if success:
            logger.info("\n✅ Reindexing completed successfully!")
            logger.info("   You can now use the application with fresh data.")
            return 0
        else:
            logger.error("\n❌ Reindexing failed!")
            return 1
            
    except ImportError as e:
        logger.error(f"❌ Import error: {e}")
        logger.error("   Make sure you're in the correct directory and dependencies are installed")
        return 1
    except Exception as e:
        logger.error(f"❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())

