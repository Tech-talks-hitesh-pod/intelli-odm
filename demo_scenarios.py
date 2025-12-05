"""
Controlled demo scenarios forpresentation.

This module provides pre-defined, controlled scenarios that demonstrate
specific business cases with predictable outcomes.
"""

from typing import Dict, List, Any
from datetime import datetime, timedelta

class DemoScenarios:
    """Controlled demo scenarios for business presentations."""
    
    @staticmethod
    def get_available_scenarios() -> Dict[str, str]:
        """Get list of available demo scenarios."""
        return {
            "seasonal_demand": "Seasonal Fashion Demand Surge",
            "supply_shortage": "Supply Chain Disruption Response", 
            "new_product": "New Product Line Launch",
            "competitor_analysis": "Competitive Response Strategy",
            "odm_data_ingestion": "ODM Data Ingestion & Forecasting"
        }
    
    @staticmethod
    def get_scenario_data(scenario_name: str) -> Dict[str, Any]:
        """Get controlled data for a specific scenario."""
        scenarios = {
            "seasonal_demand": DemoScenarios._seasonal_demand_scenario(),
            "supply_shortage": DemoScenarios._supply_shortage_scenario(),
            "new_product": DemoScenarios._new_product_scenario(), 
            "competitor_analysis": DemoScenarios._competitor_analysis_scenario(),
            "odm_data_ingestion": DemoScenarios._odm_data_ingestion_scenario()
        }
        
        return scenarios.get(scenario_name, DemoScenarios._seasonal_demand_scenario())
    
    @staticmethod
    def _seasonal_demand_scenario() -> Dict[str, Any]:
        """Scenario 1: Spring/Summer seasonal demand surge."""
        return {
            "name": "Seasonal Fashion Demand Surge",
            "description": "Spring season approaching - expect 40% increase in summer apparel demand",
            "business_context": {
                "season": "Pre-Spring (February)",
                "market_trend": "Increasing demand for summer apparel",
                "business_goal": "Prepare inventory for seasonal surge",
                "expected_outcome": "Proactive procurement to avoid stockouts"
            },
            "products": [
                "Women's floral print summer dress, midi length, cap sleeves, lightweight cotton blend, perfect for spring/summer occasions",
                "Men's lightweight linen shirt, white color, button-down collar, short sleeves, relaxed fit for hot weather comfort", 
                "Women's denim shorts, high-waisted design, distressed details, medium blue wash, comfortable stretch fabric",
                "Unisex cotton t-shirt, pastel colors available, crew neck, short sleeves, organic cotton, eco-friendly production"
            ],
            "inventory": {
                "product_1": 45,   # Summer dress - LOW (will trigger high priority)
                "product_2": 120,  # Linen shirt - MODERATE
                "product_3": 30,   # Denim shorts - LOW (will trigger procurement)
                "product_4": 200   # Cotton t-shirt - ADEQUATE
            },
            "historical_sales": DemoScenarios._generate_seasonal_sales_pattern("spring_surge"),
            "market_conditions": {
                "economic_indicator": 1.05,  # 5% growth
                "competitor_pricing": "stable",
                "seasonal_factor": 1.4  # 40% seasonal increase expected
            },
            "expected_insights": [
                "40% seasonal demand increase predicted for summer apparel",
                "Summer dresses show highest urgency for procurement", 
                "Recommended procurement investment: $75,000-85,000",
                "3-4 high priority items require immediate ordering"
            ]
        }
    
    @staticmethod  
    def _supply_shortage_scenario() -> Dict[str, Any]:
        """Scenario 2: Supply chain disruption requiring alternative sourcing."""
        return {
            "name": "Supply Chain Disruption Response",
            "description": "Primary cotton supplier disrupted - need alternative sourcing strategy",
            "business_context": {
                "situation": "Cotton supply disruption from main supplier",
                "timeline": "2-week shortage expected", 
                "business_goal": "Maintain product availability without stockouts",
                "expected_outcome": "Alternative supplier recommendations with cost optimization"
            },
            "products": [
                "Premium white cotton dress shirt, long sleeves, spread collar, tailored fit, professional business wear",
                "Organic cotton baby onesie, soft fabric, snap closures, hypoallergenic, available in neutral colors",
                "Cotton blend hoodie, pullover style, kangaroo pocket, unisex design, comfortable for casual wear",
                "100% cotton bed sheets, percale weave, queen size, breathable fabric, hotel-quality finish"
            ],
            "inventory": {
                "product_1": 25,   # Dress shirt - CRITICAL (high demand product)
                "product_2": 15,   # Baby onesie - CRITICAL (essential item)
                "product_3": 60,   # Hoodie - LOW
                "product_4": 80    # Bed sheets - MODERATE
            },
            "historical_sales": DemoScenarios._generate_seasonal_sales_pattern("stable_demand"),
            "market_conditions": {
                "economic_indicator": 0.95,  # Slight downturn due to supply issues
                "competitor_pricing": "increasing",  # Others facing same shortage
                "seasonal_factor": 1.0
            },
            "expected_insights": [
                "Alternative supplier diversification recommended",
                "Baby products show highest criticality due to low inventory",
                "Premium pricing strategy viable due to market shortage",
                "Emergency procurement needed within 5-7 days"
            ]
        }
    
    @staticmethod
    def _new_product_scenario() -> Dict[str, Any]:
        """Scenario 3: New sustainable fashion product line launch."""
        return {
            "name": "New Sustainable Product Line Launch", 
            "description": "Launching eco-friendly fashion line - need demand forecasting and initial procurement",
            "business_context": {
                "situation": "New sustainable fashion line launching next month",
                "market_trend": "Growing demand for eco-friendly apparel",
                "business_goal": "Optimize initial inventory without overcommitting",
                "expected_outcome": "Data-driven launch inventory strategy"
            },
            "products": [
                "Eco-friendly women's dress, made from recycled polyester, sleeveless design, knee-length, sustainable fashion",
                "Organic hemp t-shirt, unisex design, natural dye colors, minimal environmental impact production", 
                "Recycled denim jeans, women's skinny fit, made from post-consumer waste, sustainable manufacturing process",
                "Bamboo fiber activewear set, moisture-wicking, anti-bacterial properties, eco-conscious athletic wear"
            ],
            "inventory": {
                "product_1": 0,    # New product - no existing inventory
                "product_2": 0,    # New product
                "product_3": 0,    # New product
                "product_4": 0     # New product
            },
            "historical_sales": DemoScenarios._generate_seasonal_sales_pattern("new_product"),
            "market_conditions": {
                "economic_indicator": 1.08,  # Strong growth in sustainable fashion
                "competitor_pricing": "premium",  # Sustainable fashion commands higher prices
                "seasonal_factor": 1.2  # Strong market interest
            },
            "expected_insights": [
                "Sustainable fashion market shows 25% growth potential",
                "Bamboo activewear predicted highest demand",
                "Initial investment recommendation: $120,000-140,000", 
                "Gradual rollout strategy recommended for risk management"
            ]
        }
    
    @staticmethod
    def _competitor_analysis_scenario() -> Dict[str, Any]:
        """Scenario 4: Competitive response strategy."""
        return {
            "name": "Competitive Response Strategy",
            "description": "Competitor launched similar products at aggressive pricing - need strategic response",
            "business_context": {
                "situation": "Main competitor launched price-competitive line",
                "market_impact": "15% market share loss risk",
                "business_goal": "Maintain market position with strategic pricing and inventory",
                "expected_outcome": "Competitive positioning with optimized procurement"
            },
            "products": [
                "Basic crew neck t-shirt, multiple colors, affordable pricing tier, competitor to fast fashion brands",
                "Essential blue jeans, classic straight fit, standard denim wash, positioned against competitor basics",
                "Simple black leggings, yoga-style fit, polyester blend, competing in activewear basics segment", 
                "White button-down shirt, classic style, cotton blend, professional wear essentials category"
            ],
            "inventory": {
                "product_1": 180,  # T-shirt - HIGH (popular item, need to maintain availability)
                "product_2": 90,   # Jeans - MODERATE
                "product_3": 150,  # Leggings - HIGH (trending item)
                "product_4": 70    # Button-down - MODERATE
            },
            "historical_sales": DemoScenarios._generate_seasonal_sales_pattern("competitive_pressure"),
            "market_conditions": {
                "economic_indicator": 0.98,  # Slight pressure from competition
                "competitor_pricing": "aggressive",  # 20% below market
                "seasonal_factor": 1.1
            },
            "expected_insights": [
                "Price competition requires volume efficiency", 
                "Basic t-shirts and leggings show defensive procurement priority",
                "Cost optimization strategy: $95,000-105,000 investment",
                "Focus on supply chain efficiency over premium positioning"
            ]
        }
    
    @staticmethod
    def _odm_data_ingestion_scenario() -> Dict[str, Any]:
        """Scenario 5: ODM data cleaning and demand forecasting from messy input."""
        return {
            "name": "ODM Data Ingestion & Forecasting",
            "description": "Processing messy ODM data from manufacturers - cleaning, normalizing, and forecasting demand",
            "business_context": {
                "situation": "Received unstructured ODM catalog data from multiple manufacturers",
                "data_quality": "Messy format with abbreviations, inconsistent naming, mixed attributes",
                "business_goal": "Clean and normalize data for accurate demand forecasting",
                "expected_outcome": "Structured product data with forecasted demand and procurement recommendations"
            },
            "products": [
                "Nordic shld jkt deepnavy,nsj-990??,navydeep??,nylon shell heavy matte???,reg?,\"Mens Active TOP?? winter wear looks like glacier shild jkt maybe? price ~4.2k-4.4k, standcollr fullsleev\"",
                "breeze flow linen shrt mint pastel,BLS 204x,mintgrn pstl,pure linen 100pc|soft airy,relaxd,mens casual? solid/half-sleeve spreadcollr summr-vibe ~1800-1900 brunch/resort",
                "TerraMotion cargo joggr,tmc_880,olive drab??,twill+lycra (stretch okish),relax fit??,bottomwear? joggr/cargo hybrid alltime-use mid~1700ish",
                "FESTivEroyale Kurtta EMERALD,FRK-550,emrld??,silc blend shiny,str8cut,mens ethnic top kurta embroidered subtle festive/marriage sleevFULL mandrn price 2400-2600"
            ],
            "inventory": {
                "product_1": 0,    # ODM products - needs initial procurement
                "product_2": 0,    # New ODM data
                "product_3": 0,    # Needs processing
                "product_4": 0     # Fresh from manufacturer
            },
            "historical_sales": DemoScenarios._generate_seasonal_sales_pattern("odm_projection"),
            "market_conditions": {
                "economic_indicator": 1.12,  # Strong growth in ODM segment
                "competitor_pricing": "variable",  # Wide range due to different manufacturers
                "seasonal_factor": 1.15,  # 15% growth in direct manufacturer sourcing
                "data_quality_risk": 0.85   # Some uncertainty due to messy data
            },
            "expected_insights": [
                "15 ODM products successfully cleaned and normalized",
                "Nordic jacket and linen shirt show highest market potential",
                "Ethnic wear segment (kurta) has premium pricing opportunity",
                "Recommended ODM investment: $45,000-55,000 for initial batch",
                "Data quality improved from 30% to 95% structured format"
            ],
            "odm_processing": {
                "input_format": "dirty_odm_input.csv",
                "cleaning_agents": ["attribute_extraction", "price_normalization", "category_classification"],
                "output_format": "structured_product_catalog",
                "quality_improvement": "65% increase in data usability"
            }
        }
    
    @staticmethod
    def _generate_seasonal_sales_pattern(pattern_type: str) -> List[Dict[str, Any]]:
        """Generate controlled historical sales data for different patterns."""
        base_date = datetime.now() - timedelta(days=60)
        sales_data = []
        
        # Define base sales levels for each product in different scenarios
        pattern_configs = {
            "spring_surge": {
                "base_sales": [12, 8, 10, 20],  # Higher for summer items
                "growth_trend": [1.3, 1.1, 1.2, 1.1],  # Spring growth
                "volatility": [0.3, 0.2, 0.25, 0.15]
            },
            "stable_demand": {
                "base_sales": [15, 8, 12, 18],
                "growth_trend": [1.0, 1.0, 1.0, 1.0],  # Stable
                "volatility": [0.2, 0.15, 0.2, 0.1]
            },
            "new_product": {
                "base_sales": [0, 0, 0, 0],  # No historical sales
                "growth_trend": [0, 0, 0, 0],
                "volatility": [0, 0, 0, 0]
            },
            "competitive_pressure": {
                "base_sales": [18, 14, 16, 12], 
                "growth_trend": [0.9, 0.95, 0.92, 0.98],  # Declining due to competition
                "volatility": [0.25, 0.2, 0.3, 0.2]
            },
            "odm_projection": {
                "base_sales": [8, 14, 6, 10],  # ODM products - varied demand
                "growth_trend": [1.15, 1.2, 1.1, 1.18],  # Growing ODM market
                "volatility": [0.35, 0.25, 0.4, 0.3]  # Higher volatility for new ODM products
            }
        }
        
        config = pattern_configs.get(pattern_type, pattern_configs["stable_demand"])
        
        for day in range(60):
            date = base_date + timedelta(days=day)
            
            # Skip if new product (no history)
            if pattern_type == "new_product":
                continue
            
            for product_idx in range(4):
                product_id = f"product_{product_idx + 1}"
                
                # Calculate daily sales with controlled variation
                base = config["base_sales"][product_idx]
                trend = config["growth_trend"][product_idx]
                volatility = config["volatility"][product_idx]
                
                # Add day-of-week effect
                weekday_factor = 1.2 if date.weekday() < 5 else 0.8
                
                # Add trend over time
                time_factor = trend ** (day / 60)  # Gradual trend application
                
                # Add controlled randomness
                import random
                random.seed(day * 1000 + product_idx)  # Deterministic "randomness"
                random_factor = 1 + (random.random() - 0.5) * volatility
                
                daily_sales = max(0, int(base * weekday_factor * time_factor * random_factor))
                
                if daily_sales > 0:
                    # Controlled pricing based on product type and scenario
                    price_bases = [65, 85, 45, 75]  # Different price points per product
                    unit_price = price_bases[product_idx] * (0.9 + random.random() * 0.2)
                    
                    sales_data.append({
                        "product_id": product_id,
                        "date": date.strftime("%Y-%m-%d"),
                        "quantity": daily_sales,
                        "revenue": daily_sales * unit_price,
                        "unit_price": unit_price
                    })
        
        return sales_data
    
    @staticmethod
    def get_scenario_story(scenario_name: str) -> Dict[str, str]:
        """Get the narrative story forpresentation."""
        stories = {
            "seasonal_demand": {
                "setup": "Your retail chain is entering the critical pre-spring period. Weather forecasts predict an early warm season, and social media trends show 40% increased interest in summer fashion. You need to prepare inventory to capture this demand surge without overstocking.",
                "challenge": "How do you balance inventory investment with demand uncertainty? Traditional forecasting might miss the social media trend signals.",
                "solution": "Intelli-ODM analyzes product attributes, trend signals, and comparable product performance to generate data-driven procurement recommendations.",
                "outcome": "Precise identification of high-opportunity products (summer dresses, shorts) with optimized quantity recommendations and supplier strategies."
            },
            "supply_shortage": {
                "setup": "Your primary cotton supplier in Southeast Asia has been disrupted by logistics issues. You have 2 weeks of inventory for your top-selling cotton products. Competitor prices are rising as they face similar challenges.",
                "challenge": "How do you maintain product availability while managing cost increases? Which products should you prioritize with limited alternative sourcing?",
                "solution": "Intelli-ODM evaluates current inventory levels, sales velocity, and margin impact to prioritize emergency procurement decisions.",
                "outcome": "Clear prioritization of critical products (baby items, professional wear) with alternative supplier recommendations and emergency procurement timelines."
            },
            "new_product": {
                "setup": "You're launching a new sustainable fashion line to capture the growing eco-conscious market. Investment capital is limited, and you need to determine optimal launch inventory without overcommitting resources.",
                "challenge": "How do you forecast demand for products with no sales history? What's the right inventory investment for each SKU?",
                "solution": "Intelli-ODM uses comparable product analysis and market trend data to predict demand for new sustainable products.",
                "outcome": "Risk-managed launch strategy with clear investment priorities (bamboo activewear as leader) and phased rollout recommendations."
            },
            "competitor_analysis": {
                "setup": "Your main competitor just launched a basics line priced 20% below your equivalent products. Early market data shows you're losing 15% market share in key categories. You need a strategic response.",
                "challenge": "Should you match pricing, improve efficiency, or differentiate? How do you optimize inventory for a defensive strategy?",
                "solution": "Intelli-ODM analyzes competitive positioning and recommends volume-optimized procurement strategies to maintain margins while defending market share.",
                "outcome": "Defensive procurement strategy focusing on high-velocity basics (t-shirts, leggings) with cost optimization recommendations."
            },
            "odm_data_ingestion": {
                "setup": "Your procurement team just received ODM catalog data from 5 new manufacturers. The data is messy with abbreviations, inconsistent formatting, mixed languages, and unclear attributes. You need to quickly evaluate which products have market potential.",
                "challenge": "How do you extract meaningful insights from unstructured manufacturer data? Which ODM products should you prioritize for initial orders?",
                "solution": "Intelli-ODM's AI agents automatically clean, normalize, and structure the messy data, then analyze market potential using comparable product performance and trend analysis.",
                "outcome": "15 ODM products successfully processed with clear demand forecasts, pricing recommendations, and procurement priorities based on market analysis."
            }
        }
        
        return stories.get(scenario_name, stories["seasonal_demand"])