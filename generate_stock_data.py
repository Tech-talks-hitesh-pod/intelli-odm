#!/usr/bin/env python3
"""
Generate dummy stock data for all stores and all styles.
Creates a CSV file with current stock levels for each store-style combination.
"""

import pandas as pd
import random
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_stores():
    """Load unique stores from historical dataset."""
    try:
        historical_file = "data/sample/odm_historical_dataset_5000.csv"
        df = pd.read_csv(historical_file)
        
        # Get unique stores
        stores = df[['StoreID', 'StoreName', 'City', 'ClimateTag']].drop_duplicates()
        stores = stores.sort_values('StoreID')
        
        logger.info(f"✅ Loaded {len(stores)} unique stores")
        return stores.to_dict('records')
    except Exception as e:
        logger.error(f"❌ Failed to load stores: {e}")
        return []

def load_styles():
    """Load unique styles from historical dataset."""
    try:
        historical_file = "data/sample/odm_historical_dataset_5000.csv"
        df = pd.read_csv(historical_file)
        
        # Get unique styles with their details
        styles = df[['StyleCode', 'StyleName', 'Colour', 'Brick', 'Segment', 'Fabric']].drop_duplicates()
        styles = styles.sort_values('StyleCode')
        
        logger.info(f"✅ Loaded {len(styles)} unique styles")
        return styles.to_dict('records')
    except Exception as e:
        logger.error(f"❌ Failed to load styles: {e}")
        return []

def generate_stock_data(stores, styles):
    """Generate dummy stock data for all store-style combinations."""
    logger.info("📦 Generating stock data...")
    
    stock_records = []
    
    for store in stores:
        store_id = store.get('StoreID', '')
        store_name = store.get('StoreName', '')
        city = store.get('City', '')
        climate = store.get('ClimateTag', '')
        
        for style in styles:
            style_code = style.get('StyleCode', '')
            style_name = style.get('StyleName', '')
            colour = style.get('Colour', '')
            brick = style.get('Brick', '')
            
            # Generate realistic stock levels based on:
            # 1. Product category (some categories sell more)
            # 2. Climate (winter products in cold climates, etc.)
            # 3. Random variation
            
            # Base stock level by category
            base_stock = {
                'Jeans': random.randint(50, 200),
                'T-Shirt': random.randint(100, 300),
                'Shirt': random.randint(80, 250),
                'Dress': random.randint(40, 150),
                'Kurta': random.randint(60, 180),
                'Top': random.randint(70, 200),
                'Shorts': random.randint(30, 120),
                'Jogger': random.randint(40, 150),
            }
            
            # Get base stock for this category
            stock_level = base_stock.get(brick, random.randint(50, 200))
            
            # Adjust based on climate
            if climate:
                climate_lower = str(climate).lower()
                brick_lower = str(brick).lower()
                
                # Winter products in cold climates
                if 'winter' in climate_lower or 'cold' in climate_lower:
                    if any(term in brick_lower for term in ['jacket', 'coat', 'sweater', 'blazer']):
                        stock_level = int(stock_level * 1.5)
                    elif any(term in brick_lower for term in ['t-shirt', 'shorts', 'top']):
                        stock_level = int(stock_level * 0.7)
                
                # Summer products in hot climates
                elif 'hot' in climate_lower or 'humid' in climate_lower:
                    if any(term in brick_lower for term in ['t-shirt', 'shorts', 'top', 'dress']):
                        stock_level = int(stock_level * 1.3)
                    elif any(term in brick_lower for term in ['jacket', 'coat', 'sweater']):
                        stock_level = int(stock_level * 0.6)
            
            # Add random variation (±20%)
            variation = random.uniform(0.8, 1.2)
            stock_level = int(stock_level * variation)
            
            # Ensure minimum stock
            stock_level = max(10, stock_level)
            
            # Calculate reorder point (typically 30% of stock)
            reorder_point = int(stock_level * 0.3)
            
            # Calculate safety stock (typically 20% of stock)
            safety_stock = int(stock_level * 0.2)
            
            # Determine stock status
            if stock_level < reorder_point:
                stock_status = 'LOW_STOCK'
            elif stock_level < reorder_point * 2:
                stock_status = 'REORDER'
            else:
                stock_status = 'IN_STOCK'
            
            stock_records.append({
                'StoreID': store_id,
                'StoreName': store_name,
                'City': city,
                'ClimateTag': climate,
                'StyleCode': style_code,
                'StyleName': style_name,
                'Colour': colour,
                'Brick': brick,
                'Segment': style.get('Segment', ''),
                'Fabric': style.get('Fabric', ''),
                'CurrentStock': stock_level,
                'ReorderPoint': reorder_point,
                'SafetyStock': safety_stock,
                'StockStatus': stock_status,
                'LastUpdated': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
            })
    
    logger.info(f"✅ Generated {len(stock_records)} stock records")
    return stock_records

def main():
    """Main function to generate stock data CSV."""
    logger.info("="*80)
    logger.info("GENERATING DUMMY STOCK DATA")
    logger.info("="*80)
    
    # Load stores and styles
    stores = load_stores()
    if not stores:
        logger.error("❌ No stores found. Cannot generate stock data.")
        return
    
    styles = load_styles()
    if not styles:
        logger.error("❌ No styles found. Cannot generate stock data.")
        return
    
    logger.info(f"\n📊 Data Summary:")
    logger.info(f"   Stores: {len(stores)}")
    logger.info(f"   Styles: {len(styles)}")
    logger.info(f"   Total combinations: {len(stores) * len(styles)}")
    
    # Generate stock data
    stock_records = generate_stock_data(stores, styles)
    
    # Create DataFrame
    df = pd.DataFrame(stock_records)
    
    # Save to CSV
    output_file = "data/sample/store_stock_data.csv"
    Path("data/sample").mkdir(parents=True, exist_ok=True)
    
    df.to_csv(output_file, index=False)
    
    logger.info("="*80)
    logger.info("STOCK DATA GENERATION COMPLETE")
    logger.info("="*80)
    logger.info(f"✅ Saved {len(stock_records)} stock records to: {output_file}")
    logger.info(f"\n📊 Summary Statistics:")
    logger.info(f"   Total records: {len(stock_records)}")
    logger.info(f"   Stores: {len(stores)}")
    logger.info(f"   Styles: {len(styles)}")
    logger.info(f"   Average stock per item: {df['CurrentStock'].mean():.1f}")
    logger.info(f"   Low stock items: {len(df[df['StockStatus'] == 'LOW_STOCK'])}")
    logger.info(f"   Items needing reorder: {len(df[df['StockStatus'] == 'REORDER'])}")
    logger.info(f"   Items in stock: {len(df[df['StockStatus'] == 'IN_STOCK'])}")
    logger.info("="*80)

if __name__ == "__main__":
    main()

