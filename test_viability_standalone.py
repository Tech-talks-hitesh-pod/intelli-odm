#!/usr/bin/env python3
"""
Standalone test script to simulate 50 product variations.
Tests the viability logic without requiring streamlit or full system initialization.
"""

import logging
from typing import List, Dict, Any

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Copy the functions from odm_app.py (without streamlit dependency)
def are_colors_similar(color1: str, color2: str) -> bool:
    """Check if two colors are similar (e.g., red and light red, but not red and blue)."""
    color1_lower = color1.lower().strip()
    color2_lower = color2.lower().strip()
    
    # Exact match
    if color1_lower == color2_lower:
        return True
    
    # Extract base colors (remove modifiers like "light", "dark", "bright", etc.)
    base_colors = {
        'red': ['red', 'light red', 'dark red', 'bright red', 'deep red', 'crimson', 'maroon', 'burgundy', 'scarlet'],
        'blue': ['blue', 'light blue', 'dark blue', 'bright blue', 'navy', 'sky blue', 'royal blue', 'indigo'],
        'green': ['green', 'light green', 'dark green', 'bright green', 'olive', 'emerald', 'forest green'],
        'yellow': ['yellow', 'light yellow', 'dark yellow', 'bright yellow', 'gold', 'mustard'],
        'orange': ['orange', 'light orange', 'dark orange', 'bright orange', 'peach', 'coral'],
        'purple': ['purple', 'light purple', 'dark purple', 'bright purple', 'violet', 'lavender'],
        'pink': ['pink', 'light pink', 'dark pink', 'bright pink', 'hot pink', 'rose', 'fuchsia'],
        'black': ['black', 'charcoal', 'ebony'],
        'white': ['white', 'ivory', 'cream', 'off-white', 'beige'],
        'brown': ['brown', 'light brown', 'dark brown', 'tan', 'khaki', 'camel'],
        'gray': ['gray', 'grey', 'light gray', 'dark gray', 'silver', 'charcoal'],
    }
    
    # Find base color for each
    base1 = None
    base2 = None
    
    for base, variants in base_colors.items():
        if any(variant in color1_lower for variant in variants):
            base1 = base
        if any(variant in color2_lower for variant in variants):
            base2 = base
    
    # If both have the same base color, they're similar
    if base1 and base2 and base1 == base2:
        return True
    
    # If one color contains the other (e.g., "red" in "light red")
    if base1 and base1 in color2_lower:
        return True
    if base2 and base2 in color1_lower:
        return True
    
    return False

def extract_product_attributes(description: str, similar_products: List[Dict]) -> Dict[str, Any]:
    """Extract product attributes from description and match with sales data fields."""
    description_lower = description.lower()
    
    # Extract color
    color_keywords = ['red', 'blue', 'black', 'white', 'green', 'yellow', 'purple', 'pink', 'brown', 'gray', 'grey', 
                     'orange', 'navy', 'beige', 'maroon', 'teal', 'crimson', 'burgundy', 'olive', 'emerald', 
                     'gold', 'mustard', 'peach', 'coral', 'violet', 'lavender', 'rose', 'fuchsia', 'charcoal', 
                     'ivory', 'cream', 'tan', 'khaki', 'camel', 'silver', 'indigo', 'sky', 'royal', 'forest']
    desc_color = None
    for color in color_keywords:
        if color in description_lower:
            desc_color = color
            break
    
    # Extract category (Brick)
    brick_keywords = ['jeans', 'pants', 'shirt', 'dress', 't-shirt', 'tshirt', 'top', 'jacket', 'kurta', 'shorts', 
                     'jogger', 'cargo', 'hoodie', 'thermal', 'leggings', 'dungaree', 'pyjama', 'lehenga', 
                     'tracksuit', 'suit', 'blazer', 'coat', 'sweater', 'trousers']
    desc_brick = None
    for brick in brick_keywords:
        if brick in description_lower:
            desc_brick = brick
            break
    
    # Extract segment
    desc_segment = None
    if 'men' in description_lower or 'mens' in description_lower:
        desc_segment = 'Mens'
    elif 'women' in description_lower or 'womens' in description_lower:
        desc_segment = 'Womens'
    elif 'kid' in description_lower or 'boys' in description_lower:
        desc_segment = 'Kids'
    elif 'girl' in description_lower:
        desc_segment = 'Girls'
    
    return {
        'color': desc_color,
        'brick': desc_brick,
        'segment': desc_segment
    }

def analyze_product_viability(product_description: str, similar_products: List[Dict]) -> Dict[str, Any]:
    """Analyze if a product combination is viable based on actual sales data."""
    description_lower = product_description.lower()
    
    # Extract attributes from description
    desc_attrs = extract_product_attributes(product_description, similar_products)
    desc_color = desc_attrs.get('color')
    desc_brick = desc_attrs.get('brick')
    desc_segment = desc_attrs.get('segment')
    
    # Check sales data for exact or similar matches
    exact_matches = 0  # Exact color-category matches
    similar_color_matches = 0  # Similar color (e.g., red vs light red) matches
    category_only_matches = 0  # Same category, different color
    products_with_sales = 0  # Products that actually have sales data
    
    for product in similar_products:
        attrs = product.get('attributes', {})
        # Try multiple field name variations (case insensitive)
        product_color = str(attrs.get('Colour', attrs.get('color', attrs.get('Color', '')))).lower().strip()
        product_brick = str(attrs.get('Brick', attrs.get('brick', attrs.get('category', '')))).lower().strip()
        product_segment = str(attrs.get('Segment', attrs.get('segment', ''))).lower().strip()
        
        # Also check metadata for these fields if not found in attributes
        metadata = product.get('metadata', {})
        if not product_color and metadata:
            product_color = str(metadata.get('Colour', metadata.get('colour', metadata.get('color', '')))).lower().strip()
        if not product_brick and metadata:
            product_brick = str(metadata.get('Brick', metadata.get('brick', metadata.get('category', '')))).lower().strip()
        if not product_segment and metadata:
            product_segment = str(metadata.get('Segment', metadata.get('segment', ''))).lower().strip()
        
        # Check if product has sales data
        sales_data = product.get('sales', {})
        has_sales = sales_data.get('total_units', 0) > 0
        if has_sales:
            products_with_sales += 1
        
        # Match category (Brick) - be more flexible
        brick_match = False
        if desc_brick:
            if product_brick:
                # Direct match or one contains the other
                if desc_brick in product_brick or product_brick in desc_brick:
                    brick_match = True
            # Also check if description contains the brick keyword (for cases where Brick field might be empty)
            if not brick_match and desc_brick in description_lower:
                # If we're looking for jeans and product description/name contains jeans, consider it a match
                product_name = str(product.get('name', '')).lower()
                product_desc = str(product.get('description', '')).lower()
                if desc_brick in product_name or desc_brick in product_desc:
                    brick_match = True
        else:
            # No specific brick mentioned, so any product is a potential match
            brick_match = True
        
        # Match segment if specified - be more lenient
        segment_match = True  # Default to True if not specified
        if desc_segment and product_segment:
            # Check if segments match (case insensitive)
            desc_seg_lower = desc_segment.lower()
            prod_seg_lower = product_segment.lower()
            # Allow partial matches (e.g., "Mens" matches "Men")
            if desc_seg_lower in prod_seg_lower or prod_seg_lower in desc_seg_lower:
                segment_match = True
            else:
                segment_match = False
        
        if not brick_match:
            continue
        if desc_segment and not segment_match:
            continue
        
        category_only_matches += 1
        
        # Match color
        if desc_color:
            if product_color:
                # Exact color match (case insensitive substring match)
                if desc_color in product_color or product_color in desc_color:
                    exact_matches += 1
                # Similar color match (e.g., red ≈ light red, but red ≠ blue)
                elif are_colors_similar(desc_color, product_color):
                    similar_color_matches += 1
            else:
                # Product has no color specified, but we're looking for a specific color - don't count as match
                pass
        else:
            # No color specified in description, so any product with matching category is considered a match
            exact_matches += 1
    
    # Log matching results for debugging
    logger.info(f"Viability check for '{product_description}': color={desc_color}, brick={desc_brick}, segment={desc_segment}")
    logger.info(f"Matches: exact={exact_matches}, similar={similar_color_matches}, category_only={category_only_matches}, with_sales={products_with_sales}, total_products={len(similar_products)}")
    
    # Decision logic based on sales data
    if desc_color and desc_brick:
        # Specific color and category mentioned
        if exact_matches > 0:
            # Found exact color-category matches in sales data
            logger.info(f"✅ Found {exact_matches} exact matches for {desc_color} {desc_brick}")
            return {
                'viable': True,
                'reason': f'Found {exact_matches} exact {desc_color.title()} {desc_brick} products in sales data',
                'risk_level': 'LOW',
                'market_acceptance': 'PROVEN'
            }
        elif similar_color_matches > 0:
            # Found similar color matches (e.g., light red when searching for red)
            logger.info(f"✅ Found {similar_color_matches} similar color matches for {desc_color} {desc_brick}")
            return {
                'viable': True,
                'reason': f'Found {similar_color_matches} similar color {desc_brick} products in sales data (e.g., {desc_color} variants)',
                'risk_level': 'LOW',
                'market_acceptance': 'ACCEPTABLE'
            }
        elif category_only_matches > 0:
            # Found category matches but different colors (e.g., blue jeans when searching for red jeans)
            logger.warning(f"❌ No {desc_color} {desc_brick} found. Found {category_only_matches} {desc_brick} in different colors")
            return {
                'viable': False,
                'reason': f'No {desc_color.title()} {desc_brick} found in sales data. Found {category_only_matches} {desc_brick} products but all in different colors.',
                'risk_level': 'HIGH',
                'market_acceptance': 'UNPROVEN'
            }
        else:
            # No category matches at all
            logger.warning(f"❌ No {desc_brick} products found in sales data")
            return {
                'viable': False,
                'reason': f'No {desc_brick} products found in sales data',
                'risk_level': 'HIGH',
                'market_acceptance': 'UNPROVEN'
            }
    elif desc_brick:
        # Only category mentioned, no specific color
        if category_only_matches > 0:
            return {
                'viable': True,
                'reason': f'Found {category_only_matches} {desc_brick} products in sales data',
                'risk_level': 'LOW' if products_with_sales > 0 else 'MEDIUM',
                'market_acceptance': 'ACCEPTABLE'
            }
        else:
            return {
                'viable': False,
                'reason': f'No {desc_brick} products found in sales data',
                'risk_level': 'HIGH',
                'market_acceptance': 'UNPROVEN'
            }
    else:
        # Generic product, check if we have any similar products
        if len(similar_products) >= 5:
            return {
                'viable': True,
                'reason': f'Found {len(similar_products)} similar products in sales data',
                'risk_level': 'MEDIUM',
                'market_acceptance': 'ACCEPTABLE'
            }
        else:
            return {
                'viable': False,
                'reason': 'Very limited historical evidence of similar products',
                'risk_level': 'HIGH',
                'market_acceptance': 'UNPROVEN'
            }

def generate_test_products() -> List[str]:
    """Generate 50 product variations for testing."""
    colors = ['red', 'blue', 'black', 'white', 'green', 'yellow', 'purple', 'pink', 
              'brown', 'gray', 'orange', 'navy', 'light blue', 'dark blue', 'light red',
              'indigo', 'maroon', 'beige', 'teal', 'crimson']
    categories = ['jeans', 'pants', 'shirt', 'dress', 't-shirt', 'jacket', 'kurta', 
                  'shorts', 'hoodie', 'sweater', 'blazer', 'coat']
    segments = ['mens', 'womens', 'kids', '']
    
    products = []
    
    # Priority test cases
    priority_cases = [
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
        'brown jacket',
        'blue pants',
        'red pants',
        'black pants',
        'white pants',
        'pink pants'
    ]
    
    products.extend(priority_cases)
    
    # Add more variations
    for color in colors[:12]:
        for category in categories[:8]:
            product = f"{color} {category}"
            if product not in products:
                products.append(product)
            if len(products) >= 50:
                break
        if len(products) >= 50:
            break
    
    return products[:50]

def create_mock_products_for_test(product_desc: str) -> List[Dict]:
    """Create mock similar products based on the description."""
    desc_lower = product_desc.lower()
    
    # Simulate what the knowledge base might return
    mock_products = []
    
    # If searching for blue jeans, return blue jeans products
    if 'blue' in desc_lower and 'jeans' in desc_lower:
        mock_products = [
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
    # If searching for red jeans, return only blue/black jeans (different colors)
    elif 'red' in desc_lower and 'jeans' in desc_lower:
        mock_products = [
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
    # Generic: return some products
    else:
        mock_products = [
            {
                'attributes': {'Colour': 'Blue', 'Brick': 'Jeans', 'Segment': 'Mens'},
                'sales': {'total_units': 1000, 'avg_monthly_units': 100},
                'name': 'Blue Denim Jeans',
                'description': 'Blue jeans for men'
            }
        ]
    
    return mock_products

def main():
    """Main test function."""
    print("="*80)
    print("PRODUCT VIABILITY TEST - 50 VARIATIONS")
    print("="*80)
    
    # Generate test products
    products = generate_test_products()
    print(f"\nGenerated {len(products)} test products\n")
    
    results = []
    
    for i, product_desc in enumerate(products, 1):
        print(f"{i:2d}. Testing: {product_desc}")
        
        # Create mock products
        mock_products = create_mock_products_for_test(product_desc)
        
        # Test viability
        try:
            viability = analyze_product_viability(product_desc, mock_products)
            attrs = extract_product_attributes(product_desc, mock_products)
            
            result = {
                'product': product_desc,
                'viable': viability['viable'],
                'reason': viability['reason'],
                'risk_level': viability['risk_level'],
                'market_acceptance': viability['market_acceptance'],
                'extracted_color': attrs.get('color'),
                'extracted_brick': attrs.get('brick'),
                'extracted_segment': attrs.get('segment')
            }
            
            results.append(result)
            
            status = "✅" if viability['viable'] else "❌"
            print(f"    {status} {viability['reason'][:60]}")
            
        except Exception as e:
            print(f"    ⚠️  Error: {e}")
            results.append({
                'product': product_desc,
                'viable': False,
                'reason': f'Error: {str(e)}',
                'error': True
            })
    
    # Print summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    
    viable_count = sum(1 for r in results if r.get('viable', False))
    not_viable_count = sum(1 for r in results if not r.get('viable', False) and not r.get('error', False))
    error_count = sum(1 for r in results if r.get('error', False))
    
    print(f"\nTotal: {len(results)}")
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
    
    # Analysis by color
    print("\n" + "-"*80)
    print("BY COLOR:")
    print("-"*80)
    color_stats = {}
    for r in results:
        color = r.get('extracted_color', 'None')
        if color not in color_stats:
            color_stats[color] = {'viable': 0, 'not_viable': 0}
        if r.get('viable', False):
            color_stats[color]['viable'] += 1
        else:
            color_stats[color]['not_viable'] += 1
    
    for color, stats in sorted(color_stats.items()):
        print(f"{color:20s} | ✅ {stats['viable']:2d} | ❌ {stats['not_viable']:2d}")
    
    print("\n" + "="*80)
    print("Test Complete!")
    print("="*80)

if __name__ == "__main__":
    main()

