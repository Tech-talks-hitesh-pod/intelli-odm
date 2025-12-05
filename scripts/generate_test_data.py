"""Generate comprehensive test data for Intelli-ODM system."""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
from pathlib import Path
import argparse

class TestDataGenerator:
    """Generate realistic test data for retail analysis."""
    
    def __init__(self, seed=42):
        """Initialize with random seed for reproducibility."""
        np.random.seed(seed)
        random.seed(seed)
        
        # Fashion categories and attributes
        self.categories = {
            'TSHIRT': {
                'materials': ['Cotton', 'Polyester', 'Cotton Blend'],
                'colors': ['White', 'Black', 'Navy', 'Gray', 'Red', 'Blue'],
                'patterns': ['Solid', 'Stripes', 'Printed', 'Logo'],
                'sleeves': ['Short', 'Long', 'Sleeveless'],
                'necklines': ['Crew', 'V-neck', 'Scoop'],
                'price_range': (199, 799)
            },
            'POLO': {
                'materials': ['Cotton', 'Pique Cotton', 'Cotton Blend'],
                'colors': ['White', 'Navy', 'Black', 'Gray', 'Green', 'Blue'],
                'patterns': ['Solid', 'Stripes', 'Embroidered'],
                'sleeves': ['Short'],
                'necklines': ['Polo Collar'],
                'price_range': (299, 999)
            },
            'DRESS': {
                'materials': ['Cotton', 'Polyester', 'Chiffon', 'Crepe'],
                'colors': ['Black', 'Navy', 'Red', 'Blue', 'Floral', 'White'],
                'patterns': ['Solid', 'Floral', 'Printed', 'Abstract'],
                'sleeves': ['Short', 'Long', 'Sleeveless'],
                'necklines': ['Round', 'V-neck', 'Square'],
                'price_range': (599, 1999)
            },
            'JEANS': {
                'materials': ['Denim', 'Stretch Denim'],
                'colors': ['Blue', 'Black', 'Gray', 'White'],
                'patterns': ['Solid', 'Faded', 'Distressed'],
                'sleeves': ['Full Length'],
                'fits': ['Slim', 'Regular', 'Relaxed', 'Skinny'],
                'price_range': (799, 2499)
            },
            'SHIRT': {
                'materials': ['Cotton', 'Linen', 'Cotton Blend'],
                'colors': ['White', 'Blue', 'Black', 'Gray', 'Striped'],
                'patterns': ['Solid', 'Checks', 'Stripes', 'Printed'],
                'sleeves': ['Short', 'Long'],
                'necklines': ['Collar', 'Band Collar'],
                'price_range': (399, 1499)
            }
        }
        
        # Store information
        self.store_tiers = {
            'Tier_1': {'count': 10, 'performance_factor': 1.3},
            'Tier_2': {'count': 15, 'performance_factor': 1.0},
            'Tier_3': {'count': 12, 'performance_factor': 0.7}
        }
        
        # Seasonal patterns
        self.seasonal_factors = {
            'TSHIRT': {1: 0.8, 2: 0.9, 3: 1.2, 4: 1.3, 5: 1.4, 6: 1.2, 
                      7: 1.1, 8: 1.0, 9: 0.9, 10: 0.8, 11: 0.7, 12: 0.8},
            'POLO': {1: 0.7, 2: 0.8, 3: 1.1, 4: 1.3, 5: 1.4, 6: 1.3,
                    7: 1.2, 8: 1.1, 9: 1.0, 10: 0.8, 11: 0.7, 12: 0.7},
            'DRESS': {1: 0.9, 2: 1.0, 3: 1.2, 4: 1.3, 5: 1.2, 6: 1.1,
                     7: 1.0, 8: 0.9, 9: 1.1, 10: 1.2, 11: 1.3, 12: 1.4},
            'JEANS': {1: 1.1, 2: 1.0, 3: 0.9, 4: 0.8, 5: 0.7, 6: 0.8,
                     7: 0.9, 8: 1.0, 9: 1.2, 10: 1.3, 11: 1.2, 12: 1.1},
            'SHIRT': {1: 1.0, 2: 1.0, 3: 1.1, 4: 1.2, 5: 1.1, 6: 1.0,
                     7: 1.0, 8: 1.0, 9: 1.1, 10: 1.2, 11: 1.1, 12: 1.0}
        }
    
    def generate_products(self, num_products=100):
        """Generate product catalog."""
        products = []
        
        for i in range(num_products):
            # Choose random category
            category = random.choice(list(self.categories.keys()))
            cat_attrs = self.categories[category]
            
            # Generate product attributes
            material = random.choice(cat_attrs['materials'])
            color = random.choice(cat_attrs['colors'])
            pattern = random.choice(cat_attrs['patterns'])
            
            # Generate description
            description_parts = [color, material, category.lower()]
            if 'sleeves' in cat_attrs:
                sleeve = random.choice(cat_attrs['sleeves'])
                description_parts.append(f"{sleeve.lower()} sleeve")
            
            if 'necklines' in cat_attrs:
                neckline = random.choice(cat_attrs['necklines'])
                if neckline != 'Collar':
                    description_parts.append(neckline.lower())
            
            description_parts.append(pattern.lower())
            
            products.append({
                'product_id': f'P{i+1:03d}',
                'vendor_sku': f'V-{category[:3]}-{i+1:03d}',
                'description': ' '.join(description_parts),
                'category': category,
                'color': color,
                'material': material,
                'pattern': pattern,
                'price_min': cat_attrs['price_range'][0],
                'price_max': cat_attrs['price_range'][1]
            })
        
        return pd.DataFrame(products)
    
    def generate_stores(self):
        """Generate store information."""
        stores = []
        store_id = 1
        
        for tier, info in self.store_tiers.items():
            for i in range(info['count']):
                stores.append({
                    'store_id': f'store_{store_id:03d}',
                    'tier': tier,
                    'performance_factor': info['performance_factor'],
                    'city': random.choice(['Mumbai', 'Delhi', 'Bangalore', 'Chennai', 'Hyderabad', 'Pune', 'Kolkata']),
                    'size': random.choice(['Small', 'Medium', 'Large'])
                })
                store_id += 1
        
        return pd.DataFrame(stores)
    
    def generate_sales_data(self, products_df, stores_df, start_date='2024-01-01', days=365):
        """Generate realistic sales data."""
        sales_records = []
        start_dt = pd.to_datetime(start_date)
        
        # For each product-store combination
        for _, product in products_df.iterrows():
            category = product['category']
            product_id = product['product_id']
            base_price = random.randint(product['price_min'], product['price_max'])
            
            # Not all products are available in all stores
            available_stores = stores_df.sample(frac=random.uniform(0.6, 0.9))
            
            for _, store in available_stores.iterrows():
                store_id = store['store_id']
                performance_factor = store['performance_factor']
                
                # Base daily sales (units per day)
                base_daily_sales = random.uniform(2, 10) * performance_factor
                
                # Generate daily sales for the period
                for day_offset in range(days):
                    date = start_dt + timedelta(days=day_offset)
                    month = date.month
                    
                    # Apply seasonal factor
                    seasonal_factor = self.seasonal_factors[category][month]
                    
                    # Apply weekly pattern (higher sales on weekends)
                    day_of_week = date.weekday()
                    weekly_factor = 1.2 if day_of_week >= 5 else 1.0
                    
                    # Calculate expected sales
                    expected_sales = base_daily_sales * seasonal_factor * weekly_factor
                    
                    # Add random variation (Poisson distribution)
                    units_sold = max(0, np.random.poisson(expected_sales))
                    
                    # Skip days with zero sales occasionally
                    if units_sold == 0 and random.random() > 0.3:
                        continue
                    
                    # Calculate revenue
                    price_variation = random.uniform(0.9, 1.1)  # ±10% price variation
                    actual_price = base_price * price_variation
                    revenue = units_sold * actual_price
                    
                    sales_records.append({
                        'date': date.strftime('%Y-%m-%d'),
                        'store_id': store_id,
                        'sku': product_id,
                        'units_sold': units_sold,
                        'revenue': round(revenue, 2)
                    })
        
        return pd.DataFrame(sales_records)
    
    def generate_inventory_data(self, products_df, stores_df):
        """Generate current inventory data."""
        inventory_records = []
        
        for _, product in products_df.iterrows():
            product_id = product['product_id']
            
            for _, store in stores_df.iterrows():
                store_id = store['store_id']
                
                # Not all products have inventory in all stores
                if random.random() < 0.8:  # 80% chance of having inventory
                    
                    # Generate realistic inventory levels
                    base_stock = random.randint(10, 200)
                    on_hand = max(0, int(base_stock * random.uniform(0.3, 1.2)))
                    in_transit = random.randint(0, 50) if random.random() < 0.3 else 0
                    
                    inventory_records.append({
                        'store_id': store_id,
                        'sku': product_id,
                        'on_hand': on_hand,
                        'in_transit': in_transit
                    })
        
        return pd.DataFrame(inventory_records)
    
    def generate_pricing_data(self, products_df, stores_df):
        """Generate pricing data."""
        pricing_records = []
        
        for _, product in products_df.iterrows():
            product_id = product['product_id']
            base_price = random.randint(product['price_min'], product['price_max'])
            
            for _, store in stores_df.iterrows():
                store_id = store['store_id']
                tier = store['tier']
                
                # Price variation by store tier
                if tier == 'Tier_1':
                    price_factor = random.uniform(1.0, 1.1)  # Premium pricing
                elif tier == 'Tier_2':
                    price_factor = random.uniform(0.95, 1.05)  # Standard pricing
                else:
                    price_factor = random.uniform(0.85, 0.95)  # Discount pricing
                
                price = round(base_price * price_factor, 0)
                
                # Markdown flag (10% chance)
                markdown_flag = random.random() < 0.1
                if markdown_flag:
                    price *= random.uniform(0.7, 0.9)  # 10-30% markdown
                    price = round(price, 0)
                
                pricing_records.append({
                    'store_id': store_id,
                    'sku': product_id,
                    'price': price,
                    'markdown_flag': markdown_flag
                })
        
        return pd.DataFrame(pricing_records)
    
    def generate_all_data(self, output_dir='data/sample', num_products=100, 
                         start_date='2024-01-01', days=365):
        """Generate complete dataset."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        print("Generating products data...")
        products_df = self.generate_products(num_products)
        
        print("Generating stores data...")
        stores_df = self.generate_stores()
        
        print(f"Generating sales data for {days} days...")
        sales_df = self.generate_sales_data(products_df, stores_df, start_date, days)
        
        print("Generating inventory data...")
        inventory_df = self.generate_inventory_data(products_df, stores_df)
        
        print("Generating pricing data...")
        pricing_df = self.generate_pricing_data(products_df, stores_df)
        
        # Save to CSV files
        print(f"Saving data to {output_path}...")
        products_df.to_csv(output_path / 'products.csv', index=False)
        sales_df.to_csv(output_path / 'sales.csv', index=False)
        inventory_df.to_csv(output_path / 'inventory.csv', index=False)
        pricing_df.to_csv(output_path / 'pricing.csv', index=False)
        stores_df.to_csv(output_path / 'stores.csv', index=False)
        
        # Generate summary
        summary = {
            'generation_date': datetime.now().isoformat(),
            'products_count': len(products_df),
            'stores_count': len(stores_df),
            'sales_records': len(sales_df),
            'inventory_records': len(inventory_df),
            'pricing_records': len(pricing_df),
            'date_range': f"{start_date} to {(pd.to_datetime(start_date) + timedelta(days=days-1)).strftime('%Y-%m-%d')}",
            'categories': list(products_df['category'].value_counts().to_dict().keys())
        }
        
        with open(output_path / 'data_summary.json', 'w') as f:
            import json
            json.dump(summary, f, indent=2)
        
        print("\nData generation complete!")
        print(f"Products: {len(products_df)}")
        print(f"Stores: {len(stores_df)}")
        print(f"Sales records: {len(sales_df):,}")
        print(f"Inventory records: {len(inventory_df):,}")
        print(f"Pricing records: {len(pricing_df):,}")
        print(f"Files saved to: {output_path}")
        
        return {
            'products': products_df,
            'stores': stores_df,
            'sales': sales_df,
            'inventory': inventory_df,
            'pricing': pricing_df,
            'summary': summary
        }


def main():
    """Main function for command line usage."""
    parser = argparse.ArgumentParser(description='Generate test data for Intelli-ODM')
    parser.add_argument('--products', type=int, default=100, help='Number of products to generate')
    parser.add_argument('--days', type=int, default=365, help='Number of days of sales data')
    parser.add_argument('--start-date', default='2024-01-01', help='Start date for sales data (YYYY-MM-DD)')
    parser.add_argument('--output-dir', default='data/sample', help='Output directory')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    
    args = parser.parse_args()
    
    # Generate data
    generator = TestDataGenerator(seed=args.seed)
    generator.generate_all_data(
        output_dir=args.output_dir,
        num_products=args.products,
        start_date=args.start_date,
        days=args.days
    )


if __name__ == "__main__":
    main()