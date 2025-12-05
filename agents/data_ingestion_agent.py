"""Data Ingestion Agent for validation, cleaning, and feature engineering."""

import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from pathlib import Path
from config.agent_configs import DATA_INGESTION_CONFIG

logger = logging.getLogger(__name__)

class DataValidationError(Exception):
    """Raised when data validation fails."""
    pass

class DataIngestionAgent:
    """
    Agent responsible for data ingestion, validation, and preprocessing.
    
    Handles CSV file validation, data cleaning, standardization, 
    and feature engineering for downstream analysis.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """Initialize with configuration."""
        self.config = config or DATA_INGESTION_CONFIG
        
    def validate(self, files: Dict[str, str]) -> Dict[str, pd.DataFrame]:
        """
        Validate input data files for completeness and quality.
        
        Args:
            files: Dictionary mapping file types to file paths
            
        Returns:
            Dictionary of validated DataFrames
            
        Raises:
            DataValidationError: If validation fails
        """
        logger.info("Starting data validation")
        
        errors = []
        warnings = []
        dataframes = {}
        
        # Load files
        for file_type, file_path in files.items():
            try:
                if not Path(file_path).exists():
                    errors.append(f"File not found: {file_path}")
                    continue
                    
                df = pd.read_csv(file_path)
                dataframes[file_type] = df
                logger.info(f"Loaded {file_type}: {len(df)} rows, {len(df.columns)} columns")
                
            except Exception as e:
                errors.append(f"Failed to load {file_type}: {e}")
        
        if errors:
            raise DataValidationError(f"File loading errors: {errors}")
        
        # Validate required columns
        required_columns = self.config["validation"]["required_columns"]
        for file_type, df in dataframes.items():
            if file_type in required_columns:
                missing_cols = set(required_columns[file_type]) - set(df.columns)
                if missing_cols:
                    errors.append(f"{file_type} missing columns: {missing_cols}")
        
        # Check data quality
        for file_type, df in dataframes.items():
            # Check missing data
            missing_pct = (df.isnull().sum().sum() / (df.shape[0] * df.shape[1])) * 100
            if missing_pct > self.config["validation"]["max_missing_percentage"]:
                errors.append(f"{file_type} has {missing_pct:.1f}% missing data (threshold: {self.config['validation']['max_missing_percentage']}%)")
            elif missing_pct > 5:
                warnings.append(f"{file_type} has {missing_pct:.1f}% missing data")
            
            # Validate sales data specifically
            if file_type == "sales":
                self._validate_sales_data(df, errors, warnings)
            
            # Validate pricing data
            elif file_type == "pricing":
                self._validate_pricing_data(df, errors, warnings)
        
        if errors:
            raise DataValidationError(f"Validation errors: {errors}")
        
        if warnings:
            logger.warning(f"Validation warnings: {warnings}")
        
        logger.info("Data validation completed successfully")
        return dataframes
    
    def _validate_sales_data(self, df: pd.DataFrame, errors: List, warnings: List):
        """Validate sales data specifically."""
        # Check for negative sales
        if 'units_sold' in df.columns:
            negative_sales = (df['units_sold'] < 0).sum()
            if negative_sales > 0:
                warnings.append(f"Found {negative_sales} negative sales records")
        
        # Check date format
        if 'date' in df.columns:
            try:
                pd.to_datetime(df['date'])
            except:
                errors.append("Invalid date format in sales data")
        
        # Check for zero revenue with positive units
        if 'units_sold' in df.columns and 'revenue' in df.columns:
            zero_revenue = ((df['units_sold'] > 0) & (df['revenue'] <= 0)).sum()
            if zero_revenue > 0:
                warnings.append(f"Found {zero_revenue} records with positive units but zero revenue")
    
    def _validate_pricing_data(self, df: pd.DataFrame, errors: List, warnings: List):
        """Validate pricing data specifically."""
        if 'price' in df.columns:
            # Check for zero or negative prices
            invalid_prices = (df['price'] <= 0).sum()
            if invalid_prices > 0:
                errors.append(f"Found {invalid_prices} invalid price records (zero or negative)")
            
            # Check for extremely high prices (potential outliers)
            if df['price'].max() > df['price'].median() * 10:
                warnings.append("Detected potential price outliers")
    
    def structure(self, raw_data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """
        Standardize column names, data types, and formats.
        
        Args:
            raw_data: Dictionary of raw DataFrames
            
        Returns:
            Dictionary of structured DataFrames
        """
        logger.info("Starting data structuring")
        
        structured_data = {}
        
        for file_type, df in raw_data.items():
            df_copy = df.copy()
            
            # Standardize column names
            df_copy.columns = df_copy.columns.str.lower().str.replace(' ', '_')
            
            # Type-specific processing
            if file_type == "sales":
                df_copy = self._structure_sales_data(df_copy)
            elif file_type == "inventory":
                df_copy = self._structure_inventory_data(df_copy)
            elif file_type == "pricing":
                df_copy = self._structure_pricing_data(df_copy)
            elif file_type == "products":
                df_copy = self._structure_products_data(df_copy)
            
            structured_data[file_type] = df_copy
            logger.info(f"Structured {file_type} data: {len(df_copy)} records")
        
        logger.info("Data structuring completed")
        return structured_data
    
    def _structure_sales_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Structure sales data."""
        # Parse dates
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'], errors='coerce')
            df = df.dropna(subset=['date'])
            df = df.sort_values('date')
        
        # Ensure numeric types
        for col in ['units_sold', 'revenue']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Remove invalid records
        df = df.dropna(subset=['units_sold'])
        df = df[df['units_sold'] >= 0]
        
        return df
    
    def _structure_inventory_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Structure inventory data."""
        # Ensure numeric types
        for col in ['on_hand', 'in_transit']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                df[col] = df[col].fillna(0)
        
        return df
    
    def _structure_pricing_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Structure pricing data."""
        # Ensure numeric price
        if 'price' in df.columns:
            df['price'] = pd.to_numeric(df['price'], errors='coerce')
            df = df.dropna(subset=['price'])
            df = df[df['price'] > 0]
        
        # Handle markdown flag
        if 'markdown_flag' in df.columns:
            df['markdown_flag'] = df['markdown_flag'].astype(str).str.upper()
            df['markdown_flag'] = df['markdown_flag'].isin(['TRUE', '1', 'YES'])
        
        return df
    
    def _structure_products_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Structure products data."""
        # Clean description
        if 'description' in df.columns:
            df['description'] = df['description'].astype(str).str.strip()
        
        # Standardize category
        if 'category' in df.columns:
            df['category'] = df['category'].astype(str).str.upper().str.strip()
        
        return df
    
    def feature_engineering(self, data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """
        Compute derived features and metrics.
        
        Args:
            data: Dictionary of structured DataFrames
            
        Returns:
            Dictionary containing enhanced DataFrames and computed features
        """
        logger.info("Starting feature engineering")
        
        enhanced_data = data.copy()
        features = {}
        
        # Compute sales features
        if "sales" in data:
            enhanced_data["sales"], sales_features = self._engineer_sales_features(data["sales"])
            features.update(sales_features)
        
        # Compute store features
        if "sales" in data:
            store_tiers = self._classify_stores(data["sales"])
            enhanced_data["store_tiers"] = store_tiers
            features["store_classification"] = store_tiers.to_dict('records')
        
        # Compute inventory features
        if "inventory" in data and "sales" in data:
            enhanced_data["inventory"] = self._engineer_inventory_features(
                data["inventory"], data["sales"]
            )
        
        # Compute pricing features
        if "pricing" in data:
            enhanced_data["pricing"], pricing_features = self._engineer_pricing_features(data["pricing"])
            features.update(pricing_features)
        
        enhanced_data["features"] = features
        logger.info("Feature engineering completed")
        
        return enhanced_data
    
    def _engineer_sales_features(self, sales_df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
        """Engineer sales-related features."""
        df = sales_df.copy()
        features = {}
        
        # Sort by date
        df = df.sort_values(['sku', 'store_id', 'date'])
        
        # Compute velocity (units per day)
        velocity_window = self.config["feature_engineering"]["velocity_window_days"]
        
        velocity_data = []
        for (sku, store_id), group in df.groupby(['sku', 'store_id']):
            group = group.sort_values('date')
            
            # Calculate rolling velocity
            group['velocity_30d'] = group['units_sold'].rolling(
                window=min(len(group), velocity_window), min_periods=1
            ).sum() / velocity_window
            
            # Calculate growth rate
            if len(group) >= 2:
                recent_avg = group['units_sold'].tail(7).mean()
                previous_avg = group['units_sold'].head(7).mean()
                growth_rate = (recent_avg - previous_avg) / (previous_avg + 1e-6)
                group['growth_rate'] = growth_rate
            else:
                group['growth_rate'] = 0.0
            
            velocity_data.append(group)
        
        if velocity_data:
            df = pd.concat(velocity_data, ignore_index=True)
        
        # Compute seasonality indicators
        if 'date' in df.columns:
            df['day_of_week'] = df['date'].dt.dayofweek
            df['week_of_year'] = df['date'].dt.isocalendar().week
            df['month'] = df['date'].dt.month
            df['quarter'] = df['date'].dt.quarter
        
        # Aggregate features
        features["velocity_stats"] = {
            "avg_velocity": df['velocity_30d'].mean() if 'velocity_30d' in df.columns else 0,
            "total_units_sold": df['units_sold'].sum(),
            "total_revenue": df['revenue'].sum() if 'revenue' in df.columns else 0,
            "unique_products": df['sku'].nunique(),
            "unique_stores": df['store_id'].nunique(),
            "date_range": {
                "start": df['date'].min().isoformat() if 'date' in df.columns else None,
                "end": df['date'].max().isoformat() if 'date' in df.columns else None
            }
        }
        
        return df, features
    
    def _classify_stores(self, sales_df: pd.DataFrame) -> pd.DataFrame:
        """Classify stores into performance tiers."""
        # Calculate store performance metrics
        store_metrics = sales_df.groupby('store_id').agg({
            'units_sold': ['sum', 'mean'],
            'revenue': ['sum', 'mean'] if 'revenue' in sales_df.columns else ['count']
        }).round(2)
        
        # Flatten column names
        store_metrics.columns = ['_'.join(col).strip() for col in store_metrics.columns]
        store_metrics = store_metrics.reset_index()
        
        # Calculate performance score
        if 'revenue_sum' in store_metrics.columns:
            store_metrics['performance_score'] = (
                store_metrics['revenue_sum'] / store_metrics['revenue_sum'].max()
            )
        else:
            store_metrics['performance_score'] = (
                store_metrics['units_sold_sum'] / store_metrics['units_sold_sum'].max()
            )
        
        # Assign tiers
        thresholds = self.config["feature_engineering"]["store_tier_thresholds"]
        
        def assign_tier(score):
            if score >= thresholds[0]:
                return "Tier_1"
            elif score >= thresholds[1]:
                return "Tier_2"
            else:
                return "Tier_3"
        
        store_metrics['tier'] = store_metrics['performance_score'].apply(assign_tier)
        
        return store_metrics
    
    def _engineer_inventory_features(self, inventory_df: pd.DataFrame, sales_df: pd.DataFrame) -> pd.DataFrame:
        """Engineer inventory-related features."""
        df = inventory_df.copy()
        
        # Calculate recent sales velocity for days of stock calculation
        recent_sales = sales_df.groupby(['store_id', 'sku']).agg({
            'units_sold': 'mean'
        }).reset_index()
        recent_sales.columns = ['store_id', 'sku', 'avg_daily_sales']
        
        # Merge with inventory
        df = df.merge(recent_sales, on=['store_id', 'sku'], how='left')
        df['avg_daily_sales'] = df['avg_daily_sales'].fillna(1)  # Prevent division by zero
        
        # Calculate days of stock
        df['total_available'] = df.get('on_hand', 0) + df.get('in_transit', 0)
        df['days_of_stock'] = df['total_available'] / df['avg_daily_sales']
        df['days_of_stock'] = df['days_of_stock'].clip(0, 365)  # Cap at 365 days
        
        # Calculate stock ratio
        max_stock = df.groupby('sku')['total_available'].transform('max')
        df['stock_ratio'] = df['total_available'] / (max_stock + 1)
        
        return df
    
    def _engineer_pricing_features(self, pricing_df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
        """Engineer pricing-related features."""
        df = pricing_df.copy()
        features = {}
        
        # Calculate price percentiles by SKU
        df['price_percentile'] = df.groupby('sku')['price'].rank(pct=True)
        
        # Calculate price position within category
        if 'category' in df.columns:
            df['category_price_rank'] = df.groupby('category')['price'].rank(pct=True)
        
        # Identify price bands
        def price_band(percentile):
            if percentile <= 0.33:
                return "Low"
            elif percentile <= 0.67:
                return "Medium"
            else:
                return "High"
        
        df['price_band'] = df['price_percentile'].apply(price_band)
        
        # Calculate markdown impact if available
        if 'markdown_flag' in df.columns:
            markdown_stats = df.groupby('markdown_flag')['price'].agg(['count', 'mean'])
            features["markdown_analysis"] = markdown_stats.to_dict()
        
        # Pricing features
        features["pricing_stats"] = {
            "avg_price": df['price'].mean(),
            "price_range": {
                "min": df['price'].min(),
                "max": df['price'].max(),
                "median": df['price'].median()
            },
            "price_bands": df['price_band'].value_counts().to_dict()
        }
        
        return df, features
    
    def run(self, files: Dict[str, str]) -> Dict[str, Any]:
        """
        Execute complete data ingestion pipeline.
        
        Args:
            files: Dictionary mapping file types to file paths
            
        Returns:
            Complete processed data package
        """
        logger.info("Starting data ingestion pipeline")
        
        try:
            # Step 1: Validate
            validated_data = self.validate(files)
            
            # Step 2: Structure
            structured_data = self.structure(validated_data)
            
            # Step 3: Feature Engineering
            enriched_data = self.feature_engineering(structured_data)
            
            logger.info("Data ingestion pipeline completed successfully")
            return enriched_data
            
        except Exception as e:
            logger.error(f"Data ingestion pipeline failed: {e}")
            raise