#!/usr/bin/env python3
"""List all products in the knowledge base."""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

import logging
import json
from shared_knowledge_base import SharedKnowledgeBase
from config.settings import settings, ensure_directories

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

def list_all_products():
    """List all products in the knowledge base."""
    try:
        ensure_directories()
        kb = SharedKnowledgeBase()
        
        # Get collection stats
        stats = kb.get_collection_stats()
        print(f"\n📊 Knowledge Base Statistics:")
        print(f"   Products: {stats.get('products_count', 0)}")
        print(f"   Performance records: {stats.get('performance_records_count', 0)}")
        print()
        
        # Get all products from the collection
        try:
            products_collection = kb._products_collection
            all_products = products_collection.get(include=["metadatas", "documents"])
            
            if not all_products or not all_products.get('ids'):
                print("ℹ️  No products found in knowledge base.")
                return
            
            print(f"📦 Found {len(all_products['ids'])} products:\n")
            print("=" * 100)
            
            for i, product_id in enumerate(all_products['ids'], 1):
                metadata = all_products['metadatas'][i-1] if all_products.get('metadatas') else {}
                document = all_products['documents'][i-1] if all_products.get('documents') else ''
                
                # Extract product name
                product_name = metadata.get('product_name', '')
                description = metadata.get('description', document)
                
                # Parse attributes
                attributes_str = metadata.get('attributes', '{}')
                try:
                    attributes = json.loads(attributes_str) if isinstance(attributes_str, str) else attributes_str
                except:
                    attributes = {}
                
                print(f"\n{i}. Product ID: {product_id}")
                if product_name:
                    print(f"   Name: {product_name}")
                print(f"   Description: {description[:80]}..." if len(description) > 80 else f"   Description: {description}")
                if attributes:
                    print(f"   Category: {attributes.get('category', 'N/A')}")
                    print(f"   Material: {attributes.get('material', 'N/A')}")
                    print(f"   Color: {attributes.get('color', 'N/A')}")
                
                # Check if it's a generic product
                if product_id.startswith('product_') or product_id.startswith('P') and len(product_id) <= 5:
                    print(f"   ⚠️  WARNING: Generic product ID detected!")
            
            print("\n" + "=" * 100)
            print("\n💡 To reset the knowledge base, run:")
            print("   python scripts/reset_knowledge_base.py")
            print("\n💡 To repopulate with proper data:")
            print("   1. python scripts/generate_test_data.py --products 500")
            print("   2. python scripts/populate_knowledge_base.py --data-dir data/sample")
            
        except Exception as e:
            logger.error(f"Failed to list products: {e}")
            print(f"❌ Error listing products: {e}")
            
    except Exception as e:
        logger.error(f"Failed to access knowledge base: {e}")
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    list_all_products()

