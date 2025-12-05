#!/usr/bin/env python3
"""
Create custom demo scenarios for different business cases.
"""

import json
from datetime import datetime, timedelta
import random

def create_custom_scenario(scenario_name, products, inventory_levels, market_conditions):
    """Create a new custom scenario."""
    
    # Generate historical sales data
    sales_data = generate_sales_history(len(products), market_conditions)
    
    scenario = {
        "name": scenario_name,
        "description": f"Custom scenario: {scenario_name}",
        "business_context": {
            "situation": "Custom business scenario",
            "market_trend": market_conditions.get("trend", "stable"),
            "business_goal": "Optimize inventory and procurement",
            "expected_outcome": "Data-driven procurement recommendations"
        },
        "products": products,
        "inventory": inventory_levels,
        "historical_sales": sales_data,
        "market_conditions": market_conditions,
        "expected_insights": [
            f"Market trend: {market_conditions.get('trend', 'stable')}",
            f"Expected investment: ${sum(inventory_levels.values()) * 50:,}",
            f"Products analyzed: {len(products)}",
            "Optimized procurement strategy recommended"
        ]
    }
    
    return scenario

def generate_sales_history(product_count, market_conditions, days=60):
    """Generate realistic sales history."""
    sales_data = []
    base_date = datetime.now() - timedelta(days=days)
    
    trend_factor = {
        "growing": 1.2,
        "declining": 0.8,
        "stable": 1.0,
        "volatile": 1.0
    }.get(market_conditions.get("trend", "stable"), 1.0)
    
    for day in range(days):
        date = base_date + timedelta(days=day)
        
        for product_idx in range(product_count):
            product_id = f"product_{product_idx + 1}"
            
            # Base sales with trend
            base_sales = 10 + (product_idx * 5)  # Varied by product
            trend_sales = base_sales * (trend_factor ** (day / days))
            
            # Add seasonality
            seasonal = 1 + 0.3 * (day / 30) % 2  # Monthly cycle
            
            # Add randomness
            if market_conditions.get("trend") == "volatile":
                random_factor = random.uniform(0.5, 1.5)
            else:
                random_factor = random.uniform(0.8, 1.2)
            
            daily_sales = max(0, int(trend_sales * seasonal * random_factor))
            unit_price = 40 + (product_idx * 10) + random.uniform(-5, 5)
            
            if daily_sales > 0:
                sales_data.append({
                    "product_id": product_id,
                    "date": date.strftime("%Y-%m-%d"),
                    "quantity": daily_sales,
                    "revenue": daily_sales * unit_price,
                    "unit_price": unit_price
                })
    
    return sales_data

def save_scenario(scenario, filename):
    """Save scenario to file."""
    with open(filename, 'w') as f:
        json.dump(scenario, f, indent=2)
    print(f"✅ Scenario saved to {filename}")

def main():
    """Create example custom scenarios."""
    
    print("🎯 Creating Custom Demo Scenarios")
    print("=" * 40)
    
    # Example 1: Holiday Rush
    holiday_products = [
        "Holiday gift sweater, red and green pattern, wool blend, festive design for Christmas season",
        "Winter boots, waterproof leather, warm lining, perfect for cold weather and snow",
        "Gift wrapping accessories set, ribbons, bows, decorative paper, holiday themed",
        "Cozy flannel pajamas, plaid pattern, soft cotton, comfortable for winter nights"
    ]
    
    holiday_inventory = {
        "product_1": 50,   # Low for sweaters (high demand expected)
        "product_2": 80,   # Moderate for boots
        "product_3": 200,  # High for accessories
        "product_4": 60    # Low-moderate for pajamas
    }
    
    holiday_market = {
        "trend": "growing",
        "economic_indicator": 1.15,  # 15% holiday boost
        "seasonal_factor": 1.4,      # 40% increase
        "competitor_pricing": "stable"
    }
    
    holiday_scenario = create_custom_scenario(
        "Holiday Rush Preparation",
        holiday_products,
        holiday_inventory,
        holiday_market
    )
    
    save_scenario(holiday_scenario, "holiday_scenario.json")
    
    # Example 2: Economic Downturn
    budget_products = [
        "Basic cotton t-shirt, plain colors, value pricing, essential wardrobe staple",
        "Simple jeans, classic fit, durable denim, affordable everyday wear",
        "Budget sneakers, comfortable design, synthetic materials, cost-effective footwear",
        "Plain hoodie, fleece material, basic colors, warm and affordable"
    ]
    
    budget_inventory = {
        "product_1": 300,  # High inventory for basics
        "product_2": 200,  # High for jeans
        "product_3": 150,  # Moderate for shoes
        "product_4": 180   # High for hoodies
    }
    
    downturn_market = {
        "trend": "declining",
        "economic_indicator": 0.85,  # 15% economic decline
        "seasonal_factor": 0.9,      # 10% reduced spending
        "competitor_pricing": "aggressive"
    }
    
    downturn_scenario = create_custom_scenario(
        "Economic Downturn Response",
        budget_products,
        budget_inventory,
        downturn_market
    )
    
    save_scenario(downturn_scenario, "economic_downturn_scenario.json")
    
    # Example 3: Back to School
    school_products = [
        "Student backpack, durable fabric, multiple compartments, laptop sleeve included",
        "School uniform polo shirt, navy blue, cotton blend, comfortable fit for students",
        "Casual sneakers for kids, comfortable sole, durable materials, school appropriate",
        "Student lunch bag, insulated design, easy to clean, fun colors for children"
    ]
    
    school_inventory = {
        "product_1": 75,   # Low for backpacks
        "product_2": 120,  # Moderate for shirts
        "product_3": 90,   # Low-moderate for shoes
        "product_4": 110   # Moderate for lunch bags
    }
    
    school_market = {
        "trend": "growing",
        "economic_indicator": 1.1,   # 10% back-to-school boost
        "seasonal_factor": 1.3,      # 30% seasonal increase
        "competitor_pricing": "competitive"
    }
    
    school_scenario = create_custom_scenario(
        "Back to School Rush",
        school_products,
        school_inventory,
        school_market
    )
    
    save_scenario(school_scenario, "back_to_school_scenario.json")
    
    print(f"\n🎉 Created 3 custom scenarios!")
    print(f"📁 Files created:")
    print(f"  - holiday_scenario.json")
    print(f"  - economic_downturn_scenario.json") 
    print(f"  - back_to_school_scenario.json")
    print(f"\n💡 To use these scenarios:")
    print(f"  1. Add them to demo_scenarios.py")
    print(f"  2. Update get_scenario_data() method")
    print(f"  3. Restart the demo")

if __name__ == "__main__":
    main()