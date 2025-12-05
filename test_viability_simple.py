#!/usr/bin/env python3
"""
Simplified test script to simulate 50 product variations using the Streamlit app's functions.
Run this after the Streamlit app has loaded data.
"""

import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import functions directly
from odm_app import analyze_product_viability, extract_product_attributes, are_colors_similar

def generate_test_products() -> list:
    """Generate 50 product variations for testing."""
    colors = ['red', 'blue', 'black', 'white', 'green', 'yellow', 'purple', 'pink', 
              'brown', 'gray', 'orange', 'navy', 'light blue', 'dark blue', 'light red',
              'indigo', 'maroon', 'beige', 'teal', 'crimson']
    categories = ['jeans', 'pants', 'shirt', 'dress', 't-shirt', 'jacket', 'kurta', 
                  'shorts', 'hoodie', 'sweater', 'blazer', 'coat']
    segments = ['mens', 'womens', 'kids', '']
    
    products = []
    
    # Generate combinations - prioritize important test cases
    test_cases = [
        'blue jeans',
        'red jeans', 
        'pink jeans',
        'black jeans',
        'white jeans',
        'mens blue jeans',
        'womens red dress',
        'kids yellow t-shirt',
        'light blue shirt',
        'dark blue jacket',
        'navy jeans',
        'indigo jeans',
        'red shirt',
        'blue shirt',
        'black dress',
        'white dress',
        'green kurta',
        'purple kurta',
        'orange hoodie',
        'brown jacket'
    ]
    
    products.extend(test_cases)
    
    # Add more variations
    for color in colors[:15]:
        for category in categories[:8]:
            product = f"{color} {category}"
            if product not in products:
                products.append(product)
            if len(products) >= 50:
                break
        if len(products) >= 50:
            break
    
    return products[:50]

def test_color_similarity():
    """Test color similarity function."""
    print("Testing Color Similarity:")
    print("-" * 60)
    
    test_pairs = [
        ('red', 'light red', True),
        ('red', 'dark red', True),
        ('red', 'blue', False),
        ('blue', 'navy', True),
        ('blue', 'indigo', True),
        ('blue', 'red', False),
        ('pink', 'light pink', True),
        ('pink', 'red', False),
    ]
    
    for color1, color2, expected in test_pairs:
        result = are_colors_similar(color1, color2)
        status = "✅" if result == expected else "❌"
        print(f"{status} {color1:15s} vs {color2:15s} = {result:5s} (expected {expected})")

def simulate_with_mock_data():
    """Simulate viability checks with mock similar products."""
    print("\n" + "="*80)
    print("SIMULATING VIABILITY WITH MOCK DATA")
    print("="*80)
    
    # Mock similar products data (simulating what would come from knowledge base)
    mock_products_blue_jeans = [
        {
            'attributes': {'Colour': 'Blue', 'Brick': 'Jeans', 'Segment': 'Mens'},
            'sales': {'total_units': 1000, 'avg_monthly_units': 100},
            'name': 'Blue Denim Jeans',
            'description': 'Blue jeans for men'
        },
        {
            'attributes': {'Colour': 'Indigo', 'Brick': 'Jeans', 'Segment': 'Mens'},
            'sales': {'total_units': 800, 'avg_monthly_units': 80},
            'name': 'Indigo Jeans',
            'description': 'Indigo colored jeans'
        },
        {
            'attributes': {'Colour': 'Navy', 'Brick': 'Jeans', 'Segment': 'Mens'},
            'sales': {'total_units': 600, 'avg_monthly_units': 60},
            'name': 'Navy Jeans',
            'description': 'Navy blue jeans'
        }
    ]
    
    mock_products_red_jeans = [
        {
            'attributes': {'Colour': 'Blue', 'Brick': 'Jeans', 'Segment': 'Mens'},
            'sales': {'total_units': 1000, 'avg_monthly_units': 100},
            'name': 'Blue Denim Jeans',
            'description': 'Blue jeans for men'
        },
        {
            'attributes': {'Colour': 'Black', 'Brick': 'Jeans', 'Segment': 'Mens'},
            'sales': {'total_units': 500, 'avg_monthly_units': 50},
            'name': 'Black Jeans',
            'description': 'Black colored jeans'
        }
    ]
    
    test_cases = [
        ('blue jeans', mock_products_blue_jeans),
        ('red jeans', mock_products_red_jeans),
        ('light blue jeans', mock_products_blue_jeans),
        ('navy jeans', mock_products_blue_jeans),
    ]
    
    for product_desc, mock_products in test_cases:
        print(f"\nTesting: {product_desc}")
        print("-" * 60)
        
        # Extract attributes
        attrs = extract_product_attributes(product_desc, mock_products)
        print(f"Extracted: color={attrs.get('color')}, brick={attrs.get('brick')}, segment={attrs.get('segment')}")
        
        # Analyze viability
        viability = analyze_product_viability(product_desc, mock_products)
        
        status = "✅ VIABLE" if viability['viable'] else "❌ NOT VIABLE"
        print(f"{status}: {viability['reason']}")
        print(f"Risk Level: {viability['risk_level']}, Market Acceptance: {viability['market_acceptance']}")

def main():
    """Main test function."""
    print("="*80)
    print("PRODUCT VIABILITY TEST SIMULATION")
    print("="*80)
    
    # Test color similarity
    test_color_similarity()
    
    # Simulate with mock data
    simulate_with_mock_data()
    
    # Generate test products list
    products = generate_test_products()
    print(f"\n\nGenerated {len(products)} test product variations:")
    print("-" * 60)
    for i, product in enumerate(products, 1):
        print(f"{i:2d}. {product}")
    
    print("\n" + "="*80)
    print("NOTE: To test with actual data, run the Streamlit app and use the")
    print("predict sales page, or ensure the knowledge base is initialized.")
    print("="*80)

if __name__ == "__main__":
    main()

