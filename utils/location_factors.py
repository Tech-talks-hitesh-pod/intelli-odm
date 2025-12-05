"""Location-based factors for procurement decisions."""

import pandas as pd
from typing import Dict, Any, Optional
from pathlib import Path

class LocationFactors:
    """Handle location-based factors for procurement decisions."""
    
    def __init__(self, stores_file: str = "data/sample/stores.csv"):
        """
        Initialize location factors.
        
        Args:
            stores_file: Path to stores CSV file
        """
        self.stores_file = Path(stores_file)
        self.stores_df = None
        self._load_stores()
    
    def _load_stores(self):
        """Load stores data."""
        try:
            if self.stores_file.exists():
                self.stores_df = pd.read_csv(self.stores_file)
            else:
                self.stores_df = pd.DataFrame()
        except Exception as e:
            print(f"Warning: Could not load stores data: {e}")
            self.stores_df = pd.DataFrame()
    
    def get_store_location_factors(self, store_id: str) -> Dict[str, Any]:
        """
        Get location factors for a specific store.
        
        Args:
            store_id: Store identifier
            
        Returns:
            Dictionary of location factors
        """
        if self.stores_df is None or self.stores_df.empty:
            return self._default_factors()
        
        store = self.stores_df[self.stores_df['store_id'] == store_id]
        if store.empty:
            return self._default_factors()
        
        store = store.iloc[0]
        
        return {
            'store_id': store_id,
            'city': store.get('city', 'Unknown'),
            'state': store.get('state', 'Unknown'),
            'region': store.get('region', 'Unknown'),
            'climate': store.get('climate', 'Tropical'),
            'locality': store.get('locality', 'Tier-2'),
            'purchasing_power': store.get('purchasing_power', 1.0),
            'fashion_consciousness': store.get('fashion_consciousness', 1.0),
            'competition_level': store.get('competition_level', 1.0),
            'latitude': store.get('latitude', 0.0),
            'longitude': store.get('longitude', 0.0)
        }
    
    def get_climate_demand_factor(self, climate: str, category: str, month: int) -> float:
        """
        Get demand factor based on climate and category.
        
        Args:
            climate: Climate type
            category: Product category
            month: Month (1-12)
            
        Returns:
            Demand factor multiplier
        """
        # Climate-based seasonal factors
        climate_factors = {
            'Tropical': {
                'summer_months': [3, 4, 5, 6, 7, 8],
                'winter_months': [11, 12, 1, 2],
                'summer_demand': 1.3,
                'winter_demand': 0.8
            },
            'Arid': {
                'summer_months': [3, 4, 5, 6, 7, 8, 9],
                'winter_months': [11, 12, 1, 2],
                'summer_demand': 1.5,
                'winter_demand': 0.7
            },
            'Semi-arid': {
                'summer_months': [3, 4, 5, 6, 7, 8],
                'winter_months': [11, 12, 1, 2],
                'summer_demand': 1.4,
                'winter_demand': 0.8
            },
            'Humid Subtropical': {
                'summer_months': [4, 5, 6, 7, 8, 9],
                'winter_months': [11, 12, 1, 2, 3],
                'summer_demand': 1.2,
                'winter_demand': 1.1
            },
            'Temperate': {
                'summer_months': [5, 6, 7, 8],
                'winter_months': [11, 12, 1, 2, 3, 4],
                'summer_demand': 1.0,
                'winter_demand': 1.3
            }
        }
        
        factors = climate_factors.get(climate, climate_factors['Tropical'])
        
        # Determine if it's summer or winter
        if month in factors['summer_months']:
            base_factor = factors['summer_demand']
        elif month in factors['winter_months']:
            base_factor = factors['winter_demand']
        else:
            base_factor = 1.0
        
        # Category-specific adjustments
        category_adjustments = {
            'TSHIRT': 1.2 if month in factors.get('summer_months', []) else 0.8,
            'POLO': 1.3 if month in factors.get('summer_months', []) else 0.7,
            'TOP': 1.2 if month in factors.get('summer_months', []) else 0.8,
            'JEANS': 1.0,  # All-season
            'SHIRT': 1.0,  # All-season
            'DRESS': 1.1 if month in factors.get('summer_months', []) else 0.9,
            'KURTA': 1.0,  # Cultural, less climate-dependent
            'SALWAR': 1.0  # Cultural, less climate-dependent
        }
        
        category_factor = category_adjustments.get(category, 1.0)
        
        return base_factor * category_factor
    
    def get_locality_demand_factor(self, locality: str, category: str) -> float:
        """
        Get demand factor based on locality type.
        
        Args:
            locality: Locality type (Metro, Tier-2, Tier-3)
            category: Product category
            
        Returns:
            Demand factor multiplier
        """
        locality_factors = {
            'Metro': {
                'purchasing_power': 1.4,
                'fashion_consciousness': 1.3,
                'premium_categories': ['DRESS', 'KURTA', 'SHIRT']
            },
            'Tier-2': {
                'purchasing_power': 1.0,
                'fashion_consciousness': 1.0,
                'premium_categories': []
            },
            'Tier-3': {
                'purchasing_power': 0.7,
                'fashion_consciousness': 0.8,
                'premium_categories': []
            }
        }
        
        factors = locality_factors.get(locality, locality_factors['Tier-2'])
        
        base_factor = factors['purchasing_power']
        
        # Premium categories perform better in metro areas
        if category in factors.get('premium_categories', []):
            if locality == 'Metro':
                base_factor *= 1.2
            else:
                base_factor *= 0.9
        
        return base_factor
    
    def calculate_procurement_adjustment(self, store_id: str, category: str, 
                                        base_quantity: float, month: int = None) -> float:
        """
        Calculate procurement quantity adjustment based on location factors.
        
        Args:
            store_id: Store identifier
            category: Product category
            base_quantity: Base procurement quantity
            month: Current month (1-12), defaults to current month
            
        Returns:
            Adjusted procurement quantity
        """
        if month is None:
            from datetime import datetime
            month = datetime.now().month
        
        location_factors = self.get_store_location_factors(store_id)
        climate = location_factors.get('climate', 'Tropical')
        locality = location_factors.get('locality', 'Tier-2')
        
        # Get climate factor
        climate_factor = self.get_climate_demand_factor(climate, category, month)
        
        # Get locality factor
        locality_factor = self.get_locality_demand_factor(locality, category)
        
        # Combined adjustment
        adjustment = climate_factor * locality_factor
        
        return base_quantity * adjustment
    
    def get_stores_by_region(self, region: str = None) -> pd.DataFrame:
        """
        Get stores filtered by region.
        
        Args:
            region: Region name (North, South, East, West, Central)
            
        Returns:
            DataFrame of stores
        """
        if self.stores_df is None or self.stores_df.empty:
            return pd.DataFrame()
        
        if region:
            return self.stores_df[self.stores_df['region'] == region]
        return self.stores_df
    
    def get_stores_by_climate(self, climate: str = None) -> pd.DataFrame:
        """
        Get stores filtered by climate.
        
        Args:
            climate: Climate type
            
        Returns:
            DataFrame of stores
        """
        if self.stores_df is None or self.stores_df.empty:
            return pd.DataFrame()
        
        if climate:
            return self.stores_df[self.stores_df['climate'] == climate]
        return self.stores_df
    
    def _default_factors(self) -> Dict[str, Any]:
        """Return default factors when store data is not available."""
        return {
            'store_id': 'unknown',
            'city': 'Unknown',
            'state': 'Unknown',
            'region': 'Unknown',
            'climate': 'Tropical',
            'locality': 'Tier-2',
            'purchasing_power': 1.0,
            'fashion_consciousness': 1.0,
            'competition_level': 1.0,
            'latitude': 0.0,
            'longitude': 0.0
        }

