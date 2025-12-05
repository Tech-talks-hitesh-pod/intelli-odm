#!/usr/bin/env python3
"""Reset and clear the knowledge base."""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

import logging
from shared_knowledge_base import SharedKnowledgeBase
from config.settings import settings, ensure_directories

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

def reset_knowledge_base():
    """Reset the knowledge base by clearing all collections."""
    try:
        ensure_directories()
        kb = SharedKnowledgeBase()
        
        logger.info("Resetting knowledge base...")
        
        # Get current stats before reset
        stats_before = kb.get_collection_stats()
        logger.info(f"Before reset - Products: {stats_before.get('products_count', 0)}, "
                   f"Performance records: {stats_before.get('performance_records_count', 0)}")
        
        # Reset collections
        kb.reset_collections()
        
        # Verify reset
        stats_after = kb.get_collection_stats()
        logger.info(f"After reset - Products: {stats_after.get('products_count', 0)}, "
                   f"Performance records: {stats_after.get('performance_records_count', 0)}")
        
        print("\n✅ Knowledge base reset successfully!")
        print(f"   Cleared {stats_before.get('products_count', 0)} products")
        print(f"   Cleared {stats_before.get('performance_records_count', 0)} performance records")
        print("\n💡 Next steps:")
        print("   1. Generate new test data: python scripts/generate_test_data.py --products 500")
        print("   2. Populate knowledge base: python scripts/populate_knowledge_base.py --data-dir data/sample")
        
        return True
        
    except Exception as e:
        logger.error(f"Failed to reset knowledge base: {e}")
        print(f"\n❌ Error resetting knowledge base: {e}")
        return False

def clear_chromadb_directory():
    """Clear the ChromaDB directory completely."""
    import shutil
    
    try:
        chroma_dir = Path(settings.chromadb_persist_dir)
        
        if chroma_dir.exists():
            logger.info(f"Removing ChromaDB directory: {chroma_dir}")
            shutil.rmtree(chroma_dir)
            print(f"✅ Removed ChromaDB directory: {chroma_dir}")
            return True
        else:
            print(f"ℹ️  ChromaDB directory doesn't exist: {chroma_dir}")
            return True
            
    except Exception as e:
        logger.error(f"Failed to clear ChromaDB directory: {e}")
        print(f"❌ Error clearing ChromaDB directory: {e}")
        return False

def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Reset knowledge base')
    parser.add_argument('--full', action='store_true',
                       help='Also delete the ChromaDB directory completely')
    parser.add_argument('--confirm', action='store_true',
                       help='Skip confirmation prompt')
    
    args = parser.parse_args()
    
    if not args.confirm:
        response = input("⚠️  This will delete all data in the knowledge base. Continue? (yes/no): ")
        if response.lower() not in ['yes', 'y']:
            print("Cancelled.")
            return 0
    
    # Reset collections
    success = reset_knowledge_base()
    
    if not success:
        return 1
    
    # Optionally clear directory
    if args.full:
        clear_chromadb_directory()
    
    return 0

if __name__ == "__main__":
    exit(main())

