"""Populate the knowledge base with existing product data."""

import pandas as pd
import logging
from pathlib import Path
import sys
import os

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from shared_knowledge_base import SharedKnowledgeBase
from config.settings import settings, ensure_directories

logger = logging.getLogger(__name__)

class KnowledgeBasePopulator:
    """Populate knowledge base with product and performance data."""
    
    def __init__(self):
        """Initialize knowledge base connection."""
        ensure_directories()
        self.kb = SharedKnowledgeBase()
    
    def populate_from_csv_files(self, data_dir: str = "data/sample"):
        """
        Populate knowledge base from CSV files.
        
        Args:
            data_dir: Directory containing CSV files
        """
        data_path = Path(data_dir)
        
        if not data_path.exists():
            logger.error(f"Data directory not found: {data_path}")
            return False
        
        try:
            # Load data files
            products_df = pd.read_csv(data_path / "products.csv")
            sales_df = pd.read_csv(data_path / "sales.csv") 
            pricing_df = pd.read_csv(data_path / "pricing.csv")
            
            logger.info(f"Loaded {len(products_df)} products, {len(sales_df)} sales records")
            
            # Calculate performance metrics for each product
            performance_metrics = self._calculate_performance_metrics(sales_df, pricing_df)
            
            # Store each product in knowledge base
            stored_count = 0
            for _, product in products_df.iterrows():
                product_id = product['product_id']
                
                # Create structured attributes
                attributes = self._extract_product_attributes(product)
                
                # Get performance data
                perf_data = performance_metrics.get(product_id, {})
                
                # Store in knowledge base
                success = self.kb.store_product(
                    product_id=product_id,
                    attributes=attributes,
                    description=product.get('description', ''),
                    metadata={
                        'performance': perf_data,
                        'price_range': {
                            'min': product.get('price_min', 0),
                            'max': product.get('price_max', 0)
                        }
                    }
                )
                
                if success:
                    stored_count += 1
                    
                    # Also store performance data separately
                    if perf_data:
                        self.kb.store_performance_data(product_id, perf_data)
            
            logger.info(f"Successfully stored {stored_count}/{len(products_df)} products in knowledge base")
            
            # Print statistics
            stats = self.kb.get_collection_stats()
            logger.info(f"Knowledge base stats: {stats}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to populate knowledge base: {e}")
            return False
    
    def _extract_product_attributes(self, product_row) -> dict:
        """Extract structured attributes from product data."""
        attributes = {
            'category': product_row.get('category', ''),
            'material': product_row.get('material', ''),
            'color': product_row.get('color', ''),
            'pattern': product_row.get('pattern', ''),
            'description': product_row.get('description', ''),
        }
        
        # Parse additional attributes from description if available
        description = product_row.get('description', '').lower()
        
        # Extract sleeve type
        if 'short sleeve' in description:
            attributes['sleeve'] = 'Short'
        elif 'long sleeve' in description:
            attributes['sleeve'] = 'Long'
        elif 'sleeveless' in description:
            attributes['sleeve'] = 'Sleeveless'
        else:
            attributes['sleeve'] = 'Unknown'
        
        # Extract neckline
        if 'v-neck' in description:
            attributes['neckline'] = 'V-neck'
        elif 'crew' in description:
            attributes['neckline'] = 'Crew'
        elif 'collar' in description:
            attributes['neckline'] = 'Collar'
        else:
            attributes['neckline'] = 'Unknown'
        
        # Extract style
        if 'casual' in description:
            attributes['style'] = 'Casual'
        elif 'formal' in description:
            attributes['style'] = 'Formal'
        else:
            attributes['style'] = 'Casual'  # Default
        
        # Extract fit
        if 'slim' in description:
            attributes['fit'] = 'Slim'
        elif 'regular' in description:
            attributes['fit'] = 'Regular'
        elif 'relaxed' in description:
            attributes['fit'] = 'Relaxed'
        else:
            attributes['fit'] = 'Regular'  # Default
        
        # Set target gender based on category and description
        if any(word in description for word in ['men', 'mens', 'man']):
            attributes['target_gender'] = 'Men'
        elif any(word in description for word in ['women', 'womens', 'woman', 'ladies']):
            attributes['target_gender'] = 'Women'
        else:
            attributes['target_gender'] = 'Unisex'
        
        return attributes
    
    def _calculate_performance_metrics(self, sales_df: pd.DataFrame, pricing_df: pd.DataFrame) -> dict:
        """Calculate performance metrics for each product."""
        performance_data = {}
        
        # Convert date column
        sales_df['date'] = pd.to_datetime(sales_df['date'])
        
        # Calculate metrics by product
        for product_id in sales_df['sku'].unique():
            product_sales = sales_df[sales_df['sku'] == product_id]
            
            # Basic sales metrics
            total_units = product_sales['units_sold'].sum()
            total_revenue = product_sales['revenue'].sum()
            avg_monthly_sales = total_units / (len(product_sales['date'].dt.to_period('M').unique()) or 1)
            
            # Calculate sell-through rate (simplified)
            # In reality, you'd need inventory data for accurate calculation
            sell_through_rate = min(0.95, total_units / (total_units + 50))  # Simplified
            
            # Get average price
            product_pricing = pricing_df[pricing_df['sku'] == product_id]
            avg_price = product_pricing['price'].mean() if not product_pricing.empty else 0
            
            # Seasonal pattern (simplified)
            monthly_sales = product_sales.groupby(product_sales['date'].dt.month)['units_sold'].sum()
            peak_month = monthly_sales.idxmax() if not monthly_sales.empty else 6
            seasonal_pattern = self._determine_seasonal_pattern(peak_month)
            
            # Store performance metrics
            performance_data[product_id] = {
                'total_units_sold': int(total_units),
                'total_revenue': float(total_revenue),
                'avg_monthly_sales': float(avg_monthly_sales),
                'sell_through_rate': float(sell_through_rate),
                'avg_price': float(avg_price),
                'seasonal_pattern': seasonal_pattern,
                'peak_month': int(peak_month),
                'performance_score': min(1.0, avg_monthly_sales / 100)  # Normalized score
            }
        
        return performance_data
    
    def _determine_seasonal_pattern(self, peak_month: int) -> str:
        """Determine seasonal pattern based on peak month."""
        if peak_month in [3, 4, 5]:
            return 'spring_peak'
        elif peak_month in [6, 7, 8]:
            return 'summer_peak'
        elif peak_month in [9, 10, 11]:
            return 'autumn_peak'
        else:
            return 'winter_peak'
    
    def populate_sample_products(self, count: int = 50):
        """Populate knowledge base with sample products for testing."""
        logger.info(f"Populating knowledge base with {count} sample products")
        
        # Sample product data
        sample_products = [
            {
                'id': f'SAMPLE_{i:03d}',
                'description': f'Sample product {i}',
                'attributes': self._generate_sample_attributes(i),
                'performance': self._generate_sample_performance()
            }
            for i in range(1, count + 1)
        ]
        
        stored_count = 0
        for product in sample_products:
            success = self.kb.store_product(
                product_id=product['id'],
                attributes=product['attributes'],
                description=product['description'],
                metadata={'performance': product['performance']}
            )
            
            if success:
                stored_count += 1
                self.kb.store_performance_data(product['id'], product['performance'])
        
        logger.info(f"Stored {stored_count} sample products")
        return stored_count
    
    def _generate_sample_attributes(self, index: int) -> dict:
        """Generate sample product attributes."""
        categories = ['TSHIRT', 'POLO', 'DRESS', 'JEANS', 'SHIRT']
        materials = ['Cotton', 'Polyester', 'Cotton Blend', 'Denim']
        colors = ['White', 'Black', 'Navy', 'Gray', 'Red', 'Blue']
        
        return {
            'category': categories[index % len(categories)],
            'material': materials[index % len(materials)],
            'color': colors[index % len(colors)],
            'pattern': 'Solid' if index % 2 == 0 else 'Printed',
            'sleeve': 'Short' if index % 3 == 0 else 'Long',
            'neckline': 'Crew' if index % 2 == 0 else 'V-neck',
            'fit': 'Regular',
            'style': 'Casual',
            'target_gender': 'Unisex'
        }
    
    def _generate_sample_performance(self) -> dict:
        """Generate sample performance data."""
        import random
        
        return {
            'total_units_sold': random.randint(100, 1000),
            'total_revenue': random.randint(30000, 300000),
            'avg_monthly_sales': random.randint(20, 200),
            'sell_through_rate': random.uniform(0.6, 0.9),
            'avg_price': random.randint(299, 999),
            'seasonal_pattern': random.choice(['spring_peak', 'summer_peak', 'autumn_peak', 'winter_peak']),
            'performance_score': random.uniform(0.5, 1.0)
        }


def main():
    """Main function for command line usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Populate knowledge base with product data')
    parser.add_argument('--data-dir', default='data/sample', 
                       help='Directory containing CSV files')
    parser.add_argument('--sample-only', action='store_true',
                       help='Only populate with sample data (no CSV files)')
    parser.add_argument('--sample-count', type=int, default=50,
                       help='Number of sample products to generate')
    parser.add_argument('--reset', action='store_true',
                       help='Reset knowledge base before populating')
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    try:
        populator = KnowledgeBasePopulator()
        
        if args.reset:
            logger.info("Resetting knowledge base...")
            populator.kb.reset_collections()
        
        if args.sample_only:
            # Only populate with sample data
            populator.populate_sample_products(args.sample_count)
        else:
            # Try to populate from CSV files first
            success = populator.populate_from_csv_files(args.data_dir)
            
            if not success:
                logger.warning("Failed to populate from CSV files, using sample data instead")
                populator.populate_sample_products(args.sample_count)
        
        # Show final stats
        stats = populator.kb.get_collection_stats()
        print(f"\nKnowledge Base Statistics:")
        print(f"Products: {stats.get('products_count', 0)}")
        print(f"Performance records: {stats.get('performance_records_count', 0)}")
        print(f"Status: {stats.get('status', 'unknown')}")
        
    except Exception as e:
        logger.error(f"Failed to populate knowledge base: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())