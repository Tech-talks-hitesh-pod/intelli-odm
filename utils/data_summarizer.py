"""Data Summarization System for providing test data context to agents."""

import pandas as pd
import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime

logger = logging.getLogger(__name__)

class DataSummarizer:
    """Summarize test data and provide context to agents."""
    
    def __init__(self, data_dir: str = "data/sample"):
        """
        Initialize data summarizer.
        
        Args:
            data_dir: Directory containing test data CSV files
        """
        self.data_dir = Path(data_dir)
        self.data_cache = {}
        self.summary_cache = None
    
    def load_all_data(self) -> Dict[str, pd.DataFrame]:
        """
        Load all test data files.
        
        Returns:
            Dictionary of dataframes with keys: products, sales, inventory, pricing, stores
        """
        if self.data_cache:
            return self.data_cache
        
        try:
            data = {}
            
            # Load products
            products_path = self.data_dir / "products.csv"
            if products_path.exists():
                data['products'] = pd.read_csv(products_path)
                logger.info(f"Loaded {len(data['products'])} products")
            
            # Load sales
            sales_path = self.data_dir / "sales.csv"
            if sales_path.exists():
                data['sales'] = pd.read_csv(sales_path)
                data['sales']['date'] = pd.to_datetime(data['sales']['date'])
                logger.info(f"Loaded {len(data['sales'])} sales records")
            
            # Load inventory
            inventory_path = self.data_dir / "inventory.csv"
            if inventory_path.exists():
                data['inventory'] = pd.read_csv(inventory_path)
                logger.info(f"Loaded {len(data['inventory'])} inventory records")
            
            # Load pricing
            pricing_path = self.data_dir / "pricing.csv"
            if pricing_path.exists():
                data['pricing'] = pd.read_csv(pricing_path)
                logger.info(f"Loaded {len(data['pricing'])} pricing records")
            
            # Load stores
            stores_path = self.data_dir / "stores.csv"
            if stores_path.exists():
                data['stores'] = pd.read_csv(stores_path)
                logger.info(f"Loaded {len(data['stores'])} stores")
            
            self.data_cache = data
            return data
            
        except Exception as e:
            logger.error(f"Failed to load test data: {e}")
            return {}
    
    def generate_summary(self) -> str:
        """
        Generate a comprehensive summary of all test data for model context.
        
        Returns:
            Formatted summary string
        """
        if self.summary_cache:
            return self.summary_cache
        
        data = self.load_all_data()
        
        if not data:
            return "No test data available."
        
        summary_parts = []
        summary_parts.append("=" * 80)
        summary_parts.append("TEST DATA SUMMARY - Historical Product Performance Context")
        summary_parts.append("=" * 80)
        summary_parts.append("")
        
        # Products Summary
        if 'products' in data:
            products_df = data['products']
            summary_parts.append("PRODUCT CATALOG:")
            summary_parts.append(f"  - Total Products: {len(products_df)}")
            
            # Category breakdown
            if 'category' in products_df.columns:
                category_counts = products_df['category'].value_counts()
                summary_parts.append("  - Categories:")
                for cat, count in category_counts.items():
                    summary_parts.append(f"    * {cat}: {count} products")
            
            # Sample products with realistic names
            summary_parts.append("  - Sample Products (with realistic names):")
            sample_products = products_df.head(10)
            for _, product in sample_products.iterrows():
                product_name = product.get('name', product.get('description', 'Unknown'))
                product_id = product.get('product_id', 'N/A')
                category = product.get('category', 'N/A')
                summary_parts.append(f"    * {product_name} (ID: {product_id}, Category: {category})")
            
            summary_parts.append("")
        
        # Sales Summary
        if 'sales' in data:
            sales_df = data['sales']
            summary_parts.append("SALES DATA:")
            summary_parts.append(f"  - Total Sales Records: {len(sales_df):,}")
            
            if 'date' in sales_df.columns:
                date_range = f"{sales_df['date'].min().date()} to {sales_df['date'].max().date()}"
                summary_parts.append(f"  - Date Range: {date_range}")
            
            if 'units_sold' in sales_df.columns:
                total_units = sales_df['units_sold'].sum()
                avg_daily_units = sales_df.groupby('date')['units_sold'].sum().mean()
                summary_parts.append(f"  - Total Units Sold: {total_units:,.0f}")
                summary_parts.append(f"  - Average Daily Units: {avg_daily_units:.1f}")
            
            if 'revenue' in sales_df.columns:
                total_revenue = sales_df['revenue'].sum()
                summary_parts.append(f"  - Total Revenue: ₹{total_revenue:,.2f}")
            
            # Top performing products
            if 'sku' in sales_df.columns and 'units_sold' in sales_df.columns:
                top_products = sales_df.groupby('sku')['units_sold'].sum().nlargest(5)
                summary_parts.append("  - Top 5 Products by Units Sold:")
                for product_id, units in top_products.items():
                    summary_parts.append(f"    * {product_id}: {units:,.0f} units")
            
            summary_parts.append("")
        
        # Inventory Summary
        if 'inventory' in data:
            inventory_df = data['inventory']
            summary_parts.append("INVENTORY DATA:")
            summary_parts.append(f"  - Total Inventory Records: {len(inventory_df):,}")
            
            if 'on_hand' in inventory_df.columns:
                total_on_hand = inventory_df['on_hand'].sum()
                avg_on_hand = inventory_df['on_hand'].mean()
                summary_parts.append(f"  - Total Units On Hand: {total_on_hand:,.0f}")
                summary_parts.append(f"  - Average Units Per Record: {avg_on_hand:.1f}")
            
            summary_parts.append("")
        
        # Pricing Summary
        if 'pricing' in data:
            pricing_df = data['pricing']
            summary_parts.append("PRICING DATA:")
            summary_parts.append(f"  - Total Pricing Records: {len(pricing_df):,}")
            
            if 'price' in pricing_df.columns:
                avg_price = pricing_df['price'].mean()
                min_price = pricing_df['price'].min()
                max_price = pricing_df['price'].max()
                summary_parts.append(f"  - Average Price: ₹{avg_price:,.2f}")
                summary_parts.append(f"  - Price Range: ₹{min_price:,.2f} - ₹{max_price:,.2f}")
            
            summary_parts.append("")
        
        # Stores Summary
        if 'stores' in data:
            stores_df = data['stores']
            summary_parts.append("STORE NETWORK:")
            summary_parts.append(f"  - Total Stores: {len(stores_df)}")
            
            if 'tier' in stores_df.columns:
                tier_counts = stores_df['tier'].value_counts()
                summary_parts.append("  - Store Tiers:")
                for tier, count in tier_counts.items():
                    summary_parts.append(f"    * {tier}: {count} stores")
            
            summary_parts.append("")
        
        # Product Performance Analysis
        if 'products' in data and 'sales' in data:
            summary_parts.append("PRODUCT PERFORMANCE INSIGHTS:")
            
            products_df = data['products']
            sales_df = data['sales']
            
            # Merge to get product names
            if 'sku' in sales_df.columns:
                product_performance = sales_df.groupby('sku').agg({
                    'units_sold': 'sum',
                    'revenue': 'sum'
                }).reset_index()
                
                # Join with products to get names
                if 'product_id' in products_df.columns:
                    product_performance = product_performance.merge(
                        products_df[['product_id', 'name', 'category', 'description']],
                        left_on='sku',
                        right_on='product_id',
                        how='left'
                    )
                    
                    # Top performers by category
                    if 'category' in product_performance.columns:
                        summary_parts.append("  - Top Performers by Category:")
                        for category in product_performance['category'].dropna().unique()[:5]:
                            cat_products = product_performance[product_performance['category'] == category]
                            if not cat_products.empty:
                                top = cat_products.nlargest(1, 'units_sold')
                                if not top.empty:
                                    product_name = top.iloc[0].get('name', top.iloc[0].get('description', 'Unknown'))
                                    units = top.iloc[0]['units_sold']
                                    summary_parts.append(f"    * {category}: {product_name} ({units:,.0f} units)")
            
            summary_parts.append("")
        
        summary_parts.append("=" * 80)
        summary_parts.append("END OF TEST DATA SUMMARY")
        summary_parts.append("=" * 80)
        
        self.summary_cache = "\n".join(summary_parts)
        return self.summary_cache
    
    def get_product_context(self, product_id: Optional[str] = None, 
                          category: Optional[str] = None) -> str:
        """
        Get detailed context for a specific product or category.
        
        Args:
            product_id: Specific product ID
            category: Product category
            
        Returns:
            Context string
        """
        data = self.load_all_data()
        context_parts = []
        
        if product_id and 'products' in data:
            products_df = data['products']
            product = products_df[products_df['product_id'] == product_id]
            if not product.empty:
                product = product.iloc[0]
                context_parts.append(f"Product: {product.get('name', product.get('description', 'Unknown'))}")
                context_parts.append(f"ID: {product_id}")
                context_parts.append(f"Category: {product.get('category', 'N/A')}")
                context_parts.append(f"Description: {product.get('description', 'N/A')}")
                
                # Get sales data
                if 'sales' in data:
                    sales_df = data['sales']
                    product_sales = sales_df[sales_df['sku'] == product_id]
                    if not product_sales.empty:
                        total_units = product_sales['units_sold'].sum()
                        total_revenue = product_sales['revenue'].sum()
                        context_parts.append(f"Total Units Sold: {total_units:,.0f}")
                        context_parts.append(f"Total Revenue: ₹{total_revenue:,.2f}")
        
        elif category and 'products' in data:
            products_df = data['products']
            category_products = products_df[products_df['category'] == category]
            context_parts.append(f"Category: {category}")
            context_parts.append(f"Products in Category: {len(category_products)}")
            
            if 'sales' in data:
                sales_df = data['sales']
                category_ids = category_products['product_id'].tolist()
                category_sales = sales_df[sales_df['sku'].isin(category_ids)]
                if not category_sales.empty:
                    total_units = category_sales['units_sold'].sum()
                    context_parts.append(f"Total Category Units Sold: {total_units:,.0f}")
        
        return "\n".join(context_parts) if context_parts else "No context available."
    
    def get_similar_products_data(self, attributes: Dict[str, Any], 
                                 top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Get data for products with similar attributes.
        
        Args:
            attributes: Product attributes to match
            top_k: Number of similar products to return
            
        Returns:
            List of similar products with their sales data
        """
        data = self.load_all_data()
        similar_products = []
        
        if 'products' not in data:
            return similar_products
        
        products_df = data['products']
        
        # Filter by attributes
        filtered = products_df.copy()
        
        if 'category' in attributes:
            filtered = filtered[filtered['category'] == attributes['category']]
        if 'material' in attributes:
            filtered = filtered[filtered['material'] == attributes['material']]
        if 'color' in attributes:
            filtered = filtered[filtered['color'] == attributes['color']]
        
        # Get top k
        filtered = filtered.head(top_k)
        
        # Add sales data
        if 'sales' in data and not filtered.empty:
            sales_df = data['sales']
            for _, product in filtered.iterrows():
                product_id = product['product_id']
                product_sales = sales_df[sales_df['sku'] == product_id]
                
                product_data = {
                    'product_id': product_id,
                    'name': product.get('name', product.get('description', 'Unknown')),
                    'category': product.get('category', 'N/A'),
                    'description': product.get('description', 'N/A'),
                    'attributes': {
                        'material': product.get('material', ''),
                        'color': product.get('color', ''),
                        'pattern': product.get('pattern', '')
                    }
                }
                
                if not product_sales.empty:
                    product_data['sales'] = {
                        'total_units': int(product_sales['units_sold'].sum()),
                        'total_revenue': float(product_sales['revenue'].sum()),
                        'avg_monthly_units': float(product_sales.groupby(
                            product_sales['date'].dt.to_period('M')
                        )['units_sold'].sum().mean())
                    }
                
                similar_products.append(product_data)
        
        return similar_products

