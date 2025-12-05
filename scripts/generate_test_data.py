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
        
        # Realistic fashion product names by category
        self.product_names = {
            'TSHIRT': [
                'Classic Crew Neck T-Shirt', 'Premium Cotton Tee', 'V-Neck Casual T-Shirt',
                'Relaxed Fit T-Shirt', 'Slim Fit Basic Tee', 'Oversized Comfort T-Shirt',
                'Graphic Print T-Shirt', 'Striped Casual Tee', 'Pocket T-Shirt',
                'Henley Style T-Shirt', 'Ribbed Knit Tee', 'Organic Cotton T-Shirt'
            ],
            'POLO': [
                'Classic Polo Shirt', 'Pique Polo T-Shirt', 'Performance Polo',
                'Long Sleeve Polo', 'Striped Polo Shirt', 'Embroidered Polo',
                'Premium Cotton Polo', 'Athletic Fit Polo', 'Business Casual Polo',
                'Color Block Polo', 'Mesh Polo Shirt', 'Golf Polo Shirt'
            ],
            'DRESS': [
                'A-Line Summer Dress', 'Floral Print Maxi Dress', 'Casual Midi Dress',
                'Wrap Style Dress', 'Bodycon Party Dress', 'Shift Dress',
                'Sundress with Pockets', 'Cocktail Dress', 'Office Formal Dress',
                'Boho Chic Dress', 'Linen Blend Dress', 'Chiffon Evening Dress'
            ],
            'JEANS': [
                'Slim Fit Denim Jeans', 'Regular Fit Jeans', 'Skinny Fit Jeans',
                'Relaxed Fit Jeans', 'Straight Leg Jeans', 'Bootcut Jeans',
                'High Waist Jeans', 'Distressed Denim Jeans', 'Stretch Comfort Jeans',
                'Vintage Wash Jeans', 'Black Denim Jeans', 'Wide Leg Jeans'
            ],
            'SHIRT': [
                'Formal Cotton Shirt', 'Casual Linen Shirt', 'Oxford Button-Down Shirt',
                'Checked Pattern Shirt', 'Striped Business Shirt', 'Short Sleeve Shirt',
                'Long Sleeve Casual Shirt', 'Denim Shirt', 'Flannel Shirt',
                'Chambray Shirt', 'Poplin Dress Shirt', 'Relaxed Fit Shirt'
            ],
            'KURTA': [
                'Cotton Kurta', 'Linen Kurta', 'Printed Kurta',
                'Embroidered Kurta', 'Anarkali Kurta', 'Straight Cut Kurta',
                'A-Line Kurta', 'Designer Kurta', 'Party Wear Kurta',
                'Casual Kurta', 'Festive Kurta', 'Chikankari Kurta'
            ],
            'SALWAR': [
                'Cotton Salwar', 'Churidar Salwar', 'Palazzo Salwar',
                'Printed Salwar', 'Solid Color Salwar', 'Designer Salwar',
                'Comfort Fit Salwar', 'Elastic Waist Salwar', 'Ankle Length Salwar',
                'Wide Leg Salwar', 'Trouser Style Salwar', 'Traditional Salwar'
            ],
            'TOP': [
                'Crop Top', 'Peplum Top', 'Off-Shoulder Top',
                'Cold Shoulder Top', 'Ruffle Sleeve Top', 'Tie-Up Top',
                'Wrap Top', 'Tunic Top', 'Blouse Style Top',
                'Knot Front Top', 'Asymmetric Top', 'Bardot Top'
            ]
        }
        
        # Fashion categories and attributes
        self.categories = {
            'TSHIRT': {
                'materials': ['Cotton', 'Polyester', 'Cotton Blend', 'Organic Cotton', 'Modal'],
                'colors': ['White', 'Black', 'Navy', 'Gray', 'Red', 'Blue', 'Green', 'Yellow', 'Pink'],
                'patterns': ['Solid', 'Stripes', 'Printed', 'Logo', 'Graphic', 'Abstract'],
                'sleeves': ['Short', 'Long', 'Sleeveless'],
                'necklines': ['Crew', 'V-neck', 'Scoop', 'Henley'],
                'price_range': (199, 799)
            },
            'POLO': {
                'materials': ['Cotton', 'Pique Cotton', 'Cotton Blend', 'Performance Fabric'],
                'colors': ['White', 'Navy', 'Black', 'Gray', 'Green', 'Blue', 'Red', 'Maroon'],
                'patterns': ['Solid', 'Stripes', 'Embroidered', 'Color Block'],
                'sleeves': ['Short', 'Long'],
                'necklines': ['Polo Collar'],
                'price_range': (299, 999)
            },
            'DRESS': {
                'materials': ['Cotton', 'Polyester', 'Chiffon', 'Crepe', 'Linen', 'Georgette'],
                'colors': ['Black', 'Navy', 'Red', 'Blue', 'Floral', 'White', 'Pink', 'Green', 'Yellow'],
                'patterns': ['Solid', 'Floral', 'Printed', 'Abstract', 'Geometric', 'Polka Dots'],
                'sleeves': ['Short', 'Long', 'Sleeveless', 'Cap Sleeves'],
                'necklines': ['Round', 'V-neck', 'Square', 'Off-Shoulder', 'Halter'],
                'price_range': (599, 1999)
            },
            'JEANS': {
                'materials': ['Denim', 'Stretch Denim', 'Raw Denim'],
                'colors': ['Blue', 'Black', 'Gray', 'White', 'Light Blue', 'Dark Blue'],
                'patterns': ['Solid', 'Faded', 'Distressed', 'Vintage Wash'],
                'sleeves': ['Full Length'],
                'fits': ['Slim', 'Regular', 'Relaxed', 'Skinny', 'Straight', 'Bootcut'],
                'price_range': (799, 2499)
            },
            'SHIRT': {
                'materials': ['Cotton', 'Linen', 'Cotton Blend', 'Poplin', 'Chambray'],
                'colors': ['White', 'Blue', 'Black', 'Gray', 'Striped', 'Checked'],
                'patterns': ['Solid', 'Checks', 'Stripes', 'Printed', 'Denim'],
                'sleeves': ['Short', 'Long'],
                'necklines': ['Collar', 'Band Collar', 'Button-Down'],
                'price_range': (399, 1499)
            },
            'KURTA': {
                'materials': ['Cotton', 'Linen', 'Chiffon', 'Silk', 'Georgette'],
                'colors': ['White', 'Beige', 'Blue', 'Pink', 'Green', 'Red', 'Yellow', 'Printed'],
                'patterns': ['Solid', 'Printed', 'Embroidered', 'Floral', 'Geometric'],
                'sleeves': ['Short', 'Long', 'Three-Quarter'],
                'necklines': ['Round', 'V-neck', 'Mandarin', 'Boat Neck'],
                'price_range': (499, 1999)
            },
            'SALWAR': {
                'materials': ['Cotton', 'Linen', 'Chiffon', 'Georgette'],
                'colors': ['Black', 'Navy', 'Beige', 'White', 'Printed', 'Solid'],
                'patterns': ['Solid', 'Printed', 'Embroidered', 'Striped'],
                'fits': ['Regular', 'Churidar', 'Palazzo', 'Wide Leg'],
                'price_range': (399, 1299)
            },
            'TOP': {
                'materials': ['Cotton', 'Polyester', 'Linen', 'Chiffon'],
                'colors': ['White', 'Black', 'Navy', 'Red', 'Pink', 'Yellow', 'Printed'],
                'patterns': ['Solid', 'Printed', 'Striped', 'Floral'],
                'sleeves': ['Short', 'Sleeveless', 'Three-Quarter', 'Bell Sleeves'],
                'necklines': ['Round', 'V-neck', 'Off-Shoulder', 'Halter'],
                'price_range': (299, 899)
            }
        }
        
        # Store information - 50 stores across India
        self.store_tiers = {
            'Tier_1': {'count': 15, 'performance_factor': 1.3},
            'Tier_2': {'count': 20, 'performance_factor': 1.0},
            'Tier_3': {'count': 15, 'performance_factor': 0.7}
        }
        
        # Indian cities with location data
        self.indian_cities = [
            {'city': 'Mumbai', 'state': 'Maharashtra', 'region': 'West', 'climate': 'Tropical', 'locality': 'Metro', 'lat': 19.0760, 'lon': 72.8777},
            {'city': 'Delhi', 'state': 'Delhi', 'region': 'North', 'climate': 'Semi-arid', 'locality': 'Metro', 'lat': 28.6139, 'lon': 77.2090},
            {'city': 'Bangalore', 'state': 'Karnataka', 'region': 'South', 'climate': 'Tropical', 'locality': 'Metro', 'lat': 12.9716, 'lon': 77.5946},
            {'city': 'Hyderabad', 'state': 'Telangana', 'region': 'South', 'climate': 'Tropical', 'locality': 'Metro', 'lat': 17.3850, 'lon': 78.4867},
            {'city': 'Chennai', 'state': 'Tamil Nadu', 'region': 'South', 'climate': 'Tropical', 'locality': 'Metro', 'lat': 13.0827, 'lon': 80.2707},
            {'city': 'Kolkata', 'state': 'West Bengal', 'region': 'East', 'climate': 'Tropical', 'locality': 'Metro', 'lat': 22.5726, 'lon': 88.3639},
            {'city': 'Pune', 'state': 'Maharashtra', 'region': 'West', 'climate': 'Tropical', 'locality': 'Tier-2', 'lat': 18.5204, 'lon': 73.8567},
            {'city': 'Ahmedabad', 'state': 'Gujarat', 'region': 'West', 'climate': 'Arid', 'locality': 'Tier-2', 'lat': 23.0225, 'lon': 72.5714},
            {'city': 'Jaipur', 'state': 'Rajasthan', 'region': 'North', 'climate': 'Arid', 'locality': 'Tier-2', 'lat': 26.9124, 'lon': 75.7873},
            {'city': 'Surat', 'state': 'Gujarat', 'region': 'West', 'climate': 'Tropical', 'locality': 'Tier-2', 'lat': 21.1702, 'lon': 72.8311},
            {'city': 'Lucknow', 'state': 'Uttar Pradesh', 'region': 'North', 'climate': 'Humid Subtropical', 'locality': 'Tier-2', 'lat': 26.8467, 'lon': 80.9462},
            {'city': 'Kanpur', 'state': 'Uttar Pradesh', 'region': 'North', 'climate': 'Humid Subtropical', 'locality': 'Tier-2', 'lat': 26.4499, 'lon': 80.3319},
            {'city': 'Nagpur', 'state': 'Maharashtra', 'region': 'Central', 'climate': 'Tropical', 'locality': 'Tier-2', 'lat': 21.1458, 'lon': 79.0882},
            {'city': 'Indore', 'state': 'Madhya Pradesh', 'region': 'Central', 'climate': 'Tropical', 'locality': 'Tier-2', 'lat': 22.7196, 'lon': 75.8577},
            {'city': 'Thane', 'state': 'Maharashtra', 'region': 'West', 'climate': 'Tropical', 'locality': 'Tier-2', 'lat': 19.2183, 'lon': 72.9781},
            {'city': 'Bhopal', 'state': 'Madhya Pradesh', 'region': 'Central', 'climate': 'Tropical', 'locality': 'Tier-2', 'lat': 23.2599, 'lon': 77.4126},
            {'city': 'Visakhapatnam', 'state': 'Andhra Pradesh', 'region': 'South', 'climate': 'Tropical', 'locality': 'Tier-2', 'lat': 17.6868, 'lon': 83.2185},
            {'city': 'Patna', 'state': 'Bihar', 'region': 'East', 'climate': 'Humid Subtropical', 'locality': 'Tier-2', 'lat': 25.5941, 'lon': 85.1376},
            {'city': 'Vadodara', 'state': 'Gujarat', 'region': 'West', 'climate': 'Tropical', 'locality': 'Tier-2', 'lat': 22.3072, 'lon': 73.1812},
            {'city': 'Ghaziabad', 'state': 'Uttar Pradesh', 'region': 'North', 'climate': 'Humid Subtropical', 'locality': 'Tier-2', 'lat': 28.6692, 'lon': 77.4538},
            {'city': 'Ludhiana', 'state': 'Punjab', 'region': 'North', 'climate': 'Semi-arid', 'locality': 'Tier-2', 'lat': 30.9010, 'lon': 75.8573},
            {'city': 'Agra', 'state': 'Uttar Pradesh', 'region': 'North', 'climate': 'Semi-arid', 'locality': 'Tier-3', 'lat': 27.1767, 'lon': 78.0081},
            {'city': 'Nashik', 'state': 'Maharashtra', 'region': 'West', 'climate': 'Tropical', 'locality': 'Tier-2', 'lat': 19.9975, 'lon': 73.7898},
            {'city': 'Faridabad', 'state': 'Haryana', 'region': 'North', 'climate': 'Semi-arid', 'locality': 'Tier-2', 'lat': 28.4089, 'lon': 77.3178},
            {'city': 'Meerut', 'state': 'Uttar Pradesh', 'region': 'North', 'climate': 'Humid Subtropical', 'locality': 'Tier-2', 'lat': 28.9845, 'lon': 77.7064},
            {'city': 'Rajkot', 'state': 'Gujarat', 'region': 'West', 'climate': 'Arid', 'locality': 'Tier-2', 'lat': 22.3039, 'lon': 70.8022},
            {'city': 'Varanasi', 'state': 'Uttar Pradesh', 'region': 'North', 'climate': 'Humid Subtropical', 'locality': 'Tier-2', 'lat': 25.3176, 'lon': 82.9739},
            {'city': 'Srinagar', 'state': 'Jammu and Kashmir', 'region': 'North', 'climate': 'Temperate', 'locality': 'Tier-3', 'lat': 34.0837, 'lon': 74.7973},
            {'city': 'Amritsar', 'state': 'Punjab', 'region': 'North', 'climate': 'Semi-arid', 'locality': 'Tier-2', 'lat': 31.6340, 'lon': 74.8723},
            {'city': 'Chandigarh', 'state': 'Chandigarh', 'region': 'North', 'climate': 'Semi-arid', 'locality': 'Tier-2', 'lat': 30.7333, 'lon': 76.7794},
            {'city': 'Coimbatore', 'state': 'Tamil Nadu', 'region': 'South', 'climate': 'Tropical', 'locality': 'Tier-2', 'lat': 11.0168, 'lon': 76.9558},
            {'city': 'Kochi', 'state': 'Kerala', 'region': 'South', 'climate': 'Tropical', 'locality': 'Tier-2', 'lat': 9.9312, 'lon': 76.2673},
            {'city': 'Mysore', 'state': 'Karnataka', 'region': 'South', 'climate': 'Tropical', 'locality': 'Tier-2', 'lat': 12.2958, 'lon': 76.6394},
            {'city': 'Vijayawada', 'state': 'Andhra Pradesh', 'region': 'South', 'climate': 'Tropical', 'locality': 'Tier-2', 'lat': 16.5062, 'lon': 80.6480},
            {'city': 'Jodhpur', 'state': 'Rajasthan', 'region': 'North', 'climate': 'Arid', 'locality': 'Tier-3', 'lat': 26.2389, 'lon': 73.0243},
            {'city': 'Madurai', 'state': 'Tamil Nadu', 'region': 'South', 'climate': 'Tropical', 'locality': 'Tier-2', 'lat': 9.9252, 'lon': 78.1198},
            {'city': 'Raipur', 'state': 'Chhattisgarh', 'region': 'Central', 'climate': 'Tropical', 'locality': 'Tier-2', 'lat': 21.2514, 'lon': 81.6296},
            {'city': 'Kota', 'state': 'Rajasthan', 'region': 'North', 'climate': 'Arid', 'locality': 'Tier-2', 'lat': 25.2138, 'lon': 75.8648},
            {'city': 'Guwahati', 'state': 'Assam', 'region': 'East', 'climate': 'Humid Subtropical', 'locality': 'Tier-2', 'lat': 26.1445, 'lon': 91.7362},
            {'city': 'Bhubaneswar', 'state': 'Odisha', 'region': 'East', 'climate': 'Tropical', 'locality': 'Tier-2', 'lat': 20.2961, 'lon': 85.8245},
            {'city': 'Dehradun', 'state': 'Uttarakhand', 'region': 'North', 'climate': 'Temperate', 'locality': 'Tier-2', 'lat': 30.3165, 'lon': 78.0322},
            {'city': 'Mangalore', 'state': 'Karnataka', 'region': 'South', 'climate': 'Tropical', 'locality': 'Tier-2', 'lat': 12.9141, 'lon': 74.8560},
            {'city': 'Belgaum', 'state': 'Karnataka', 'region': 'South', 'climate': 'Tropical', 'locality': 'Tier-3', 'lat': 15.8497, 'lon': 74.4977},
            {'city': 'Tiruchirappalli', 'state': 'Tamil Nadu', 'region': 'South', 'climate': 'Tropical', 'locality': 'Tier-2', 'lat': 10.7905, 'lon': 78.7047},
            {'city': 'Kozhikode', 'state': 'Kerala', 'region': 'South', 'climate': 'Tropical', 'locality': 'Tier-2', 'lat': 11.2588, 'lon': 75.7804},
            {'city': 'Jamshedpur', 'state': 'Jharkhand', 'region': 'East', 'climate': 'Tropical', 'locality': 'Tier-2', 'lat': 22.8046, 'lon': 86.2029},
            {'city': 'Jabalpur', 'state': 'Madhya Pradesh', 'region': 'Central', 'climate': 'Tropical', 'locality': 'Tier-2', 'lat': 23.1815, 'lon': 79.9864},
            {'city': 'Gwalior', 'state': 'Madhya Pradesh', 'region': 'Central', 'climate': 'Tropical', 'locality': 'Tier-2', 'lat': 26.2183, 'lon': 78.1828},
            {'city': 'Dhanbad', 'state': 'Jharkhand', 'region': 'East', 'climate': 'Tropical', 'locality': 'Tier-2', 'lat': 23.7957, 'lon': 86.4304},
            {'city': 'Warangal', 'state': 'Telangana', 'region': 'South', 'climate': 'Tropical', 'locality': 'Tier-3', 'lat': 18.0000, 'lon': 79.5833}
        ]
        
        # Location-based factors for procurement decisions
        self.location_factors = {
            'climate': {
                'Tropical': {'summer_demand': 1.3, 'winter_demand': 0.8, 'monsoon_impact': 0.9},
                'Arid': {'summer_demand': 1.5, 'winter_demand': 0.7, 'monsoon_impact': 1.0},
                'Semi-arid': {'summer_demand': 1.4, 'winter_demand': 0.8, 'monsoon_impact': 0.95},
                'Humid Subtropical': {'summer_demand': 1.2, 'winter_demand': 1.1, 'monsoon_impact': 0.85},
                'Temperate': {'summer_demand': 1.0, 'winter_demand': 1.3, 'monsoon_impact': 1.0}
            },
            'locality': {
                'Metro': {'purchasing_power': 1.4, 'fashion_consciousness': 1.3, 'competition': 1.2},
                'Tier-2': {'purchasing_power': 1.0, 'fashion_consciousness': 1.0, 'competition': 1.0},
                'Tier-3': {'purchasing_power': 0.7, 'fashion_consciousness': 0.8, 'competition': 0.8}
            }
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
                     7: 1.0, 8: 1.0, 9: 1.1, 10: 1.2, 11: 1.1, 12: 1.0},
            'KURTA': {1: 1.2, 2: 1.3, 3: 1.1, 4: 1.0, 5: 0.9, 6: 0.8,
                     7: 0.7, 8: 0.8, 9: 1.0, 10: 1.2, 11: 1.4, 12: 1.5},  # Festive seasons
            'SALWAR': {1: 1.2, 2: 1.3, 3: 1.1, 4: 1.0, 5: 0.9, 6: 0.8,
                      7: 0.7, 8: 0.8, 9: 1.0, 10: 1.2, 11: 1.4, 12: 1.5},
            'TOP': {1: 0.8, 2: 0.9, 3: 1.2, 4: 1.3, 5: 1.4, 6: 1.3,
                   7: 1.2, 8: 1.1, 9: 1.0, 10: 0.9, 11: 0.8, 12: 0.9}
        }
    
    def generate_products(self, num_products=1000):
        """Generate product catalog with realistic fashion product names."""
        products = []
        
        for i in range(num_products):
            # Choose random category
            category = random.choice(list(self.categories.keys()))
            cat_attrs = self.categories[category]
            
            # Use realistic product name if available
            if category in self.product_names:
                base_name = random.choice(self.product_names[category])
            else:
                base_name = f"{category} Product"
            
            # Generate product attributes
            material = random.choice(cat_attrs['materials'])
            color = random.choice(cat_attrs['colors'])
            pattern = random.choice(cat_attrs['patterns'])
            
            # Build detailed description
            description_parts = [color]
            if pattern != 'Solid':
                description_parts.append(pattern)
            description_parts.append(material)
            description_parts.append(base_name)
            
            if 'sleeves' in cat_attrs:
                sleeve = random.choice(cat_attrs['sleeves'])
                if sleeve != 'Full Length':  # Skip for jeans
                    description_parts.append(f"with {sleeve.lower()} sleeves")
            
            if 'necklines' in cat_attrs:
                neckline = random.choice(cat_attrs['necklines'])
                if neckline not in ['Collar', 'Polo Collar']:
                    description_parts.append(f"{neckline.lower()} neck")
            
            if 'fits' in cat_attrs:
                fit = random.choice(cat_attrs['fits'])
                description_parts.append(f"{fit.lower()} fit")
            
            # Create full description
            full_description = ' '.join(description_parts)
            
            products.append({
                'product_id': f'P{i+1:04d}',
                'vendor_sku': f'V-{category[:3]}-{i+1:04d}',
                'name': base_name,
                'description': full_description,
                'category': category,
                'color': color,
                'material': material,
                'pattern': pattern,
                'price_min': cat_attrs['price_range'][0],
                'price_max': cat_attrs['price_range'][1]
            })
        
        return pd.DataFrame(products)
    
    def generate_stores(self, num_stores=50):
        """Generate store information with location data."""
        stores = []
        store_id = 1
        
        # Select cities for stores (ensure we have 50 stores)
        selected_cities = random.sample(self.indian_cities, min(num_stores, len(self.indian_cities)))
        
        # If we need more stores than cities, repeat some cities
        while len(selected_cities) < num_stores:
            selected_cities.extend(random.sample(self.indian_cities, min(num_stores - len(selected_cities), len(self.indian_cities))))
        
        selected_cities = selected_cities[:num_stores]
        
        for i, city_data in enumerate(selected_cities):
            # Assign tier based on locality
            if city_data['locality'] == 'Metro':
                tier = random.choices(['Tier_1', 'Tier_2'], weights=[0.6, 0.4])[0]
            elif city_data['locality'] == 'Tier-2':
                tier = random.choices(['Tier_2', 'Tier_3'], weights=[0.7, 0.3])[0]
            else:
                tier = 'Tier_3'
            
            tier_info = self.store_tiers[tier]
            
            stores.append({
                'store_id': f'store_{store_id:03d}',
                'tier': tier,
                'performance_factor': tier_info['performance_factor'],
                'city': city_data['city'],
                'state': city_data['state'],
                'region': city_data['region'],
                'climate': city_data['climate'],
                'locality': city_data['locality'],
                'latitude': city_data['lat'],
                'longitude': city_data['lon'],
                'size': random.choice(['Small', 'Medium', 'Large']),
                'purchasing_power': self.location_factors['locality'][city_data['locality']]['purchasing_power'],
                'fashion_consciousness': self.location_factors['locality'][city_data['locality']]['fashion_consciousness'],
                'competition_level': self.location_factors['locality'][city_data['locality']]['competition']
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
        """Generate current inventory data per store with location-based factors."""
        inventory_records = []
        
        for _, product in products_df.iterrows():
            product_id = product['product_id']
            category = product['category']
            
            for _, store in stores_df.iterrows():
                store_id = store['store_id']
                climate = store['climate']
                locality = store['locality']
                purchasing_power = store['purchasing_power']
                tier = store['tier']
                
                # Location-based inventory factors
                climate_factors = self.location_factors['climate'][climate]
                locality_factors = self.location_factors['locality'][locality]
                
                # Base stock varies by tier and location
                base_stock = random.randint(10, 200) * purchasing_power
                
                # Category-specific adjustments based on climate
                if category in ['TSHIRT', 'POLO', 'TOP']:
                    # Summer items - higher in tropical/arid climates
                    if climate in ['Tropical', 'Arid', 'Semi-arid']:
                        base_stock *= 1.2
                    else:
                        base_stock *= 0.9
                elif category in ['JEANS', 'SHIRT']:
                    # All-season items - stable across climates
                    base_stock *= 1.0
                elif category in ['DRESS', 'KURTA', 'SALWAR']:
                    # Fashion items - higher in metro areas
                    if locality == 'Metro':
                        base_stock *= 1.3
                    else:
                        base_stock *= 0.8
                
                # Not all products have inventory in all stores
                # Higher probability in metro areas
                availability_prob = 0.9 if locality == 'Metro' else 0.7 if locality == 'Tier-2' else 0.5
                
                if random.random() < availability_prob:
                    # Generate realistic inventory levels
                    on_hand = max(0, int(base_stock * random.uniform(0.3, 1.2)))
                    in_transit = random.randint(0, 50) if random.random() < 0.3 else 0
                    
                    inventory_records.append({
                        'store_id': store_id,
                        'sku': product_id,
                        'on_hand': on_hand,
                        'in_transit': in_transit,
                        'reorder_point': int(on_hand * 0.3),  # 30% of current stock
                        'safety_stock': int(on_hand * 0.2)  # 20% safety stock
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
    
    def generate_all_data(self, output_dir='data/sample', num_products=1000, 
                         start_date='2023-01-01', days=730, num_stores=50):
        """Generate complete dataset."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        print("Generating products data...")
        products_df = self.generate_products(num_products)
        
        print(f"Generating {num_stores} stores data with location information...")
        stores_df = self.generate_stores(num_stores)
        
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
    parser.add_argument('--products', type=int, default=1000, help='Number of products to generate')
    parser.add_argument('--stores', type=int, default=50, help='Number of stores to generate')
    parser.add_argument('--days', type=int, default=730, help='Number of days of sales data (default: 2 years)')
    parser.add_argument('--start-date', default='2023-01-01', help='Start date for sales data (YYYY-MM-DD)')
    parser.add_argument('--output-dir', default='data/sample', help='Output directory')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    
    args = parser.parse_args()
    
    # Generate data
    generator = TestDataGenerator(seed=args.seed)
    generator.generate_all_data(
        output_dir=args.output_dir,
        num_products=args.products,
        num_stores=args.stores,
        start_date=args.start_date,
        days=args.days
    )


if __name__ == "__main__":
    main()