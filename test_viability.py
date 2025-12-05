#!/usr/bin/env python3
"""
Test script to simulate 50 product variations and check viability results.
"""

import sys
import logging
from typing import List, Dict, Any

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import our modules
try:
    from shared_knowledge_base import SharedKnowledgeBase
    from agents.data_ingestion_agent import DataIngestionAgent
    from utils.llm_client import LLMClientFactory, LLMClient
    from config.settings import settings
    from odm_app import analyze_product_viability, extract_product_attributes
    logger.info("✅ Successfully imported all required modules")
except ImportError as e:
    logger.error(f"Import error: {e}")
    sys.exit(1)

def generate_test_products() -> List[str]:
    """Generate 50 product variations for testing."""
    colors = ['red', 'blue', 'black', 'white', 'green', 'yellow', 'purple', 'pink', 
              'brown', 'gray', 'orange', 'navy', 'light blue', 'dark blue', 'light red']
    categories = ['jeans', 'pants', 'shirt', 'dress', 't-shirt', 'jacket', 'kurta', 
                  'shorts', 'hoodie', 'sweater']
    segments = ['mens', 'womens', 'kids', '']
    
    products = []
    
    # Generate combinations
    for color in colors[:10]:  # Use first 10 colors
        for category in categories[:5]:  # Use first 5 categories
            segment = segments[0] if category in ['jeans', 'pants', 'shirt'] else segments[1]
            if segment:
                product = f"{segment} {color} {category}"
            else:
                product = f"{color} {category}"
            products.append(product)
            if len(products) >= 50:
                break
        if len(products) >= 50:
            break
    
    # Add some edge cases
    edge_cases = [
        'blue jeans',
        'red jeans',
        'pink jeans',
        'black jeans',
        'white jeans',
        'mens blue jeans',
        'womens red dress',
        'kids yellow t-shirt',
        'light blue shirt',
        'dark blue jacket'
    ]
    
    # Combine and ensure we have exactly 50
    all_products = edge_cases + products
    return all_products[:50]

def test_viability(kb: SharedKnowledgeBase, products: List[str]) -> List[Dict[str, Any]]:
    """Test viability for all products."""
    results = []
    
    for i, product_desc in enumerate(products, 1):
        logger.info(f"\n{'='*60}")
        logger.info(f"Test {i}/50: {product_desc}")
        logger.info(f"{'='*60}")
        
        try:
            # Find similar products
            similar_products = kb.find_similar_products(
                query_attributes={},
                query_description=product_desc,
                top_k=15
            )
            
            logger.info(f"Found {len(similar_products)} similar products")
            
            # Analyze viability
            viability = analyze_product_viability(product_desc, similar_products)
            
            # Extract attributes
            attrs = extract_product_attributes(product_desc, similar_products)
            
            result = {
                'product': product_desc,
                'viable': viability['viable'],
                'reason': viability['reason'],
                'risk_level': viability['risk_level'],
                'market_acceptance': viability['market_acceptance'],
                'extracted_color': attrs.get('color'),
                'extracted_brick': attrs.get('brick'),
                'extracted_segment': attrs.get('segment'),
                'similar_products_count': len(similar_products)
            }
            
            results.append(result)
            
            # Log result
            status = "✅ VIABLE" if viability['viable'] else "❌ NOT VIABLE"
            logger.info(f"{status}: {viability['reason']}")
            logger.info(f"Extracted: color={attrs.get('color')}, brick={attrs.get('brick')}, segment={attrs.get('segment')}")
            
        except Exception as e:
            logger.error(f"Error testing {product_desc}: {e}")
            results.append({
                'product': product_desc,
                'viable': False,
                'reason': f'Error: {str(e)}',
                'error': True
            })
    
    return results

def print_summary(results: List[Dict[str, Any]]):
    """Print summary of results."""
    print("\n" + "="*80)
    print("SUMMARY OF VIABILITY TESTS")
    print("="*80)
    
    viable_count = sum(1 for r in results if r.get('viable', False))
    not_viable_count = len(results) - viable_count
    error_count = sum(1 for r in results if r.get('error', False))
    
    print(f"\nTotal Products Tested: {len(results)}")
    print(f"✅ Viable: {viable_count}")
    print(f"❌ Not Viable: {not_viable_count}")
    print(f"⚠️  Errors: {error_count}")
    
    print("\n" + "-"*80)
    print("VIABLE PRODUCTS:")
    print("-"*80)
    for r in results:
        if r.get('viable', False):
            print(f"✅ {r['product']:40s} | {r['reason'][:35]}")
    
    print("\n" + "-"*80)
    print("NOT VIABLE PRODUCTS:")
    print("-"*80)
    for r in results:
        if not r.get('viable', False) and not r.get('error', False):
            print(f"❌ {r['product']:40s} | {r['reason'][:35]}")
    
    print("\n" + "-"*80)
    print("ERRORS:")
    print("-"*80)
    for r in results:
        if r.get('error', False):
            print(f"⚠️  {r['product']:40s} | {r.get('reason', 'Unknown error')}")
    
    # Group by extracted attributes
    print("\n" + "-"*80)
    print("BY EXTRACTED COLOR:")
    print("-"*80)
    color_groups = {}
    for r in results:
        color = r.get('extracted_color', 'None')
        if color not in color_groups:
            color_groups[color] = {'viable': 0, 'not_viable': 0}
        if r.get('viable', False):
            color_groups[color]['viable'] += 1
        else:
            color_groups[color]['not_viable'] += 1
    
    for color, counts in sorted(color_groups.items()):
        print(f"{color:20s} | ✅ {counts['viable']:2d} | ❌ {counts['not_viable']:2d}")
    
    print("\n" + "-"*80)
    print("BY EXTRACTED BRICK:")
    print("-"*80)
    brick_groups = {}
    for r in results:
        brick = r.get('extracted_brick', 'None')
        if brick not in brick_groups:
            brick_groups[brick] = {'viable': 0, 'not_viable': 0}
        if r.get('viable', False):
            brick_groups[brick]['viable'] += 1
        else:
            brick_groups[brick]['not_viable'] += 1
    
    for brick, counts in sorted(brick_groups.items()):
        print(f"{brick:20s} | ✅ {counts['viable']:2d} | ❌ {counts['not_viable']:2d}")

def main():
    """Main test function."""
    logger.info("🚀 Starting viability test simulation...")
    
    # Initialize knowledge base
    try:
        kb = SharedKnowledgeBase()
        logger.info("✅ Knowledge base initialized")
        
        # Load ODM data
        data_agent = DataIngestionAgent(None, kb)
        dirty_input_file = "data/sample/dirty_odm_input.csv"
        historical_file = "data/sample/odm_historical_dataset_5000.csv"
        
        logger.info("Loading ODM data...")
        result = data_agent.load_dirty_odm_data(dirty_input_file, historical_file)
        logger.info(f"✅ ODM data loaded: {result.get('summary', {})}")
        
    except Exception as e:
        logger.error(f"❌ Failed to initialize: {e}")
        sys.exit(1)
    
    # Generate test products
    products = generate_test_products()
    logger.info(f"Generated {len(products)} test products")
    
    # Test viability
    results = test_viability(kb, products)
    
    # Print summary
    print_summary(results)
    
    logger.info("\n✅ Test simulation complete!")

if __name__ == "__main__":
    main()

