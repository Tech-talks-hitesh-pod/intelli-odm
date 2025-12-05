#!/usr/bin/env python3
"""
Test script for debug endpoints - can be run without starting the Flask server
"""

import sys
import json
sys.path.append('.')

try:
    from shared_knowledge_base import SharedKnowledgeBase
    from agents.data_ingestion_agent import DataIngestionAgent
    from demo_scenarios import DemoScenarios
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)

def test_vector_db_status():
    """Test vector database status."""
    print("🔍 Testing Vector Database Status...")
    
    try:
        kb = SharedKnowledgeBase()
        
        # Get basic info
        count = kb.get_collection_size()
        stats = kb.get_collection_stats()
        
        print(f"✅ Vector DB Collection Size: {count}")
        print(f"✅ Vector DB Stats: {json.dumps(stats, indent=2)}")
        
        return count > 0
        
    except Exception as e:
        print(f"❌ Vector DB Status Error: {e}")
        return False

def test_load_demo_data():
    """Test loading demo data."""
    print("\n📁 Testing Demo Data Loading...")
    
    try:
        kb = SharedKnowledgeBase()
        agent = DataIngestionAgent(knowledge_base=kb)
        
        # Load demo data
        result = agent.load_and_process_demo_data('data/sample')
        
        # Check results
        products_loaded = len(result.get('products', [])) if result.get('products') is not None else 0
        sales_loaded = len(result.get('sales', [])) if result.get('sales') is not None else 0
        
        print(f"✅ Products Processed: {products_loaded}")
        print(f"✅ Sales Records Processed: {sales_loaded}")
        
        # Check vector DB after loading
        count_after = kb.get_collection_size()
        print(f"✅ Vector DB Size After Loading: {count_after}")
        
        return count_after > 0
        
    except Exception as e:
        print(f"❌ Demo Data Loading Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_similarity_search():
    """Test similarity search."""
    print("\n🔍 Testing Similarity Search...")
    
    try:
        kb = SharedKnowledgeBase()
        
        # Test search
        query = "women denim shorts"
        similar_products = kb.find_similar_products(
            query_attributes={},
            query_description=query,
            top_k=3
        )
        
        print(f"✅ Search Query: '{query}'")
        print(f"✅ Similar Products Found: {len(similar_products)}")
        
        for i, product in enumerate(similar_products[:2], 1):
            print(f"   {i}. {product.get('name', 'Unknown')} (similarity: {product.get('similarity_score', 0):.3f})")
        
        return len(similar_products) > 0
        
    except Exception as e:
        print(f"❌ Similarity Search Error: {e}")
        return False

def test_all_products():
    """Test getting all products."""
    print("\n📦 Testing All Products Retrieval...")
    
    try:
        kb = SharedKnowledgeBase()
        
        # Get all products
        all_products = kb.get_all_products_with_performance()
        
        print(f"✅ Total Products Retrieved: {len(all_products)}")
        
        if all_products:
            sample_product = all_products[0]
            print(f"✅ Sample Product: {sample_product.get('name', 'Unknown')}")
            print(f"   - Product ID: {sample_product.get('product_id', 'N/A')}")
            print(f"   - Sales Units: {sample_product.get('sales', {}).get('total_units', 0)}")
        
        return len(all_products) > 0
        
    except Exception as e:
        print(f"❌ All Products Retrieval Error: {e}")
        return False

def main():
    """Run all tests."""
    print("🧪 Vector Database Debug Test Suite")
    print("=" * 50)
    
    tests = [
        ("Vector DB Status", test_vector_db_status),
        ("Load Demo Data", test_load_demo_data),
        ("Similarity Search", test_similarity_search),
        ("All Products", test_all_products)
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n🔬 Running: {test_name}")
        try:
            result = test_func()
            results.append(result)
            status = "✅ PASS" if result else "❌ FAIL"
            print(f"   {status}")
        except Exception as e:
            print(f"   ❌ ERROR: {e}")
            results.append(False)
    
    # Summary
    passed = sum(results)
    total = len(results)
    
    print("\n" + "=" * 50)
    print(f"📊 Test Results: {passed}/{total} passed")
    
    if passed == total:
        print("🎉 All tests passed! Vector database is working correctly.")
    elif passed > 0:
        print("⚠️ Some tests passed. Vector database is partially working.")
    else:
        print("💥 All tests failed. Vector database needs debugging.")
    
    print("\n🔧 To start the debug API server:")
    print("   python debug_api.py")
    print("\n🌐 Then visit: http://localhost:5001/api/debug/health")

if __name__ == "__main__":
    main()