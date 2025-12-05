"""Utility for building prompts with data context."""

import logging
from typing import Dict, Any, Optional, List
from utils.data_summarizer import DataSummarizer

logger = logging.getLogger(__name__)

class PromptBuilder:
    """Build prompts with test data context for agents."""
    
    def __init__(self, data_dir: str = "data/sample"):
        """
        Initialize prompt builder.
        
        Args:
            data_dir: Directory containing test data
        """
        self.data_summarizer = DataSummarizer(data_dir)
        self.data_summary = None
    
    def get_data_context(self) -> str:
        """Get data summary context."""
        if self.data_summary is None:
            self.data_summary = self.data_summarizer.generate_summary()
        return self.data_summary
    
    def build_procurement_prompt(self, product_description: str, 
                                 product_attributes: Dict[str, Any],
                                 similar_products_data: Optional[List[Dict]] = None,
                                 location_context: Optional[Dict[str, Any]] = None,
                                 product_analysis: Optional[Dict[str, Any]] = None) -> str:
        """
        Build a procurement recommendation prompt with data context.
        
        Args:
            product_description: Description of the product to evaluate
            product_attributes: Attributes of the product
            similar_products_data: Data about similar products
            
        Returns:
            Formatted prompt string
        """
        data_context = self.get_data_context()
        
        prompt_parts = []
        prompt_parts.append("=" * 80)
        prompt_parts.append("PROCUREMENT RECOMMENDATION REQUEST")
        prompt_parts.append("=" * 80)
        prompt_parts.append("")
        
        # Add comprehensive product analysis if available
        if product_analysis:
            prompt_parts.append("=" * 80)
            prompt_parts.append("COMPREHENSIVE PRODUCT ANALYSIS (Pre-analyzed)")
            prompt_parts.append("=" * 80)
            prompt_parts.append("")
            prompt_parts.append(f"Total Products Analyzed: {product_analysis.get('total_products', 0)}")
            prompt_parts.append("")
            
            # Performance by category
            if product_analysis.get('performance_by_category'):
                prompt_parts.append("PERFORMANCE PATTERNS BY CATEGORY:")
                for category, data in product_analysis['performance_by_category'].items():
                    prompt_parts.append(f"  {category}:")
                    prompt_parts.append(f"    - Products: {data.get('count', 0)}")
                    prompt_parts.append(f"    - Total Units: {data.get('total_units_sold', 0):,}")
                    prompt_parts.append(f"    - Total Revenue: ₹{data.get('total_revenue', 0):,.2f}")
                    prompt_parts.append(f"    - Avg Monthly Units: {data.get('avg_monthly_units', 0):.1f}")
                    
                    # Top performing attributes
                    if data.get('top_attributes', {}).get('material'):
                        top_materials = sorted(
                            data['top_attributes']['material'].items(),
                            key=lambda x: x[1]['units'],
                            reverse=True
                        )[:3]
                        if top_materials:
                            prompt_parts.append("    - Top Materials:")
                            for material, perf in top_materials:
                                prompt_parts.append(f"      * {material}: {perf['units']:,} units")
                    
                    if data.get('top_attributes', {}).get('color'):
                        top_colors = sorted(
                            data['top_attributes']['color'].items(),
                            key=lambda x: x[1]['units'],
                            reverse=True
                        )[:3]
                        if top_colors:
                            prompt_parts.append("    - Top Colors:")
                            for color, perf in top_colors:
                                prompt_parts.append(f"      * {color}: {perf['units']:,} units")
                    prompt_parts.append("")
            
            # Top performers
            if product_analysis.get('top_performers'):
                prompt_parts.append("TOP PERFORMING PRODUCTS:")
                for i, product in enumerate(product_analysis['top_performers'][:5], 1):
                    prompt_parts.append(f"  {i}. {product['name']}")
                    prompt_parts.append(f"     Category: {product['category']}, Material: {product['material']}, Color: {product['color']}")
                    prompt_parts.append(f"     Performance: {product['total_units']:,} units, ₹{product['total_revenue']:,.2f} revenue")
                prompt_parts.append("")
            
            # LLM-generated insights
            if product_analysis.get('llm_insights'):
                prompt_parts.append("KEY INSIGHTS FROM ANALYSIS:")
                prompt_parts.append(product_analysis['llm_insights'])
                prompt_parts.append("")
            
            prompt_parts.append("=" * 80)
            prompt_parts.append("")
        
        # Add historical data context
        prompt_parts.append("HISTORICAL DATA CONTEXT:")
        prompt_parts.append(data_context)
        prompt_parts.append("")
        prompt_parts.append("=" * 80)
        prompt_parts.append("NEW PRODUCT TO EVALUATE:")
        prompt_parts.append("=" * 80)
        prompt_parts.append(f"Description: {product_description}")
        prompt_parts.append(f"Attributes: {product_attributes}")
        prompt_parts.append("")
        
        if similar_products_data:
            prompt_parts.append("SIMILAR PRODUCTS PERFORMANCE:")
            for i, similar in enumerate(similar_products_data[:5], 1):
                name = similar.get('name', 'Unknown')
                sales = similar.get('sales', {})
                prompt_parts.append(f"{i}. {name}")
                if sales:
                    prompt_parts.append(f"   - Total Units Sold: {sales.get('total_units', 0):,}")
                    prompt_parts.append(f"   - Total Revenue: ₹{sales.get('total_revenue', 0):,.2f}")
                    prompt_parts.append(f"   - Avg Monthly Units: {sales.get('avg_monthly_units', 0):.1f}")
            prompt_parts.append("")
        
        if location_context:
            prompt_parts.append("LOCATION-BASED FACTORS:")
            prompt_parts.append(f"   - Total Stores: {location_context.get('total_stores', 0)}")
            if location_context.get('region_distribution'):
                prompt_parts.append("   - Store Distribution by Region:")
                for region, count in location_context['region_distribution'].items():
                    factor = location_context.get('region_demand_factors', {}).get(region, 1.0)
                    prompt_parts.append(f"     * {region}: {count} stores (demand factor: {factor:.2f})")
            if location_context.get('recommendations'):
                prompt_parts.append("   - Location Recommendations:")
                for rec in location_context['recommendations']:
                    prompt_parts.append(f"     * {rec}")
            prompt_parts.append("")
        
        prompt_parts.append("=" * 80)
        prompt_parts.append("YOUR TASK:")
        prompt_parts.append("=" * 80)
        if product_analysis:
            prompt_parts.append("Use the comprehensive product analysis above to understand:")
            prompt_parts.append("  - Which attributes (material, color, category) drive success")
            prompt_parts.append("  - Performance patterns across categories")
            prompt_parts.append("  - Market preferences and trends")
            prompt_parts.append("")
        prompt_parts.append("Based on the pre-analyzed data, historical context, and similar products performance,")
        prompt_parts.append("provide a procurement recommendation:")
        prompt_parts.append("")
        prompt_parts.append("1. Should we procure this product? (Yes/No with reasoning)")
        prompt_parts.append("   - Compare against similar products and category performance")
        prompt_parts.append("   - Consider attribute performance patterns from the analysis")
        prompt_parts.append("2. If yes, how many units should we procure?")
        prompt_parts.append("   - Base on similar products' performance and category averages")
        prompt_parts.append("   - Consider top performers in the same category/attributes")
        prompt_parts.append("3. What is the expected viability/performance score (0-1)?")
        prompt_parts.append("   - Compare against category and attribute benchmarks")
        prompt_parts.append("4. What are the key risks and opportunities?")
        prompt_parts.append("   - Reference patterns from the comprehensive analysis")
        prompt_parts.append("5. Recommended procurement strategy (initial order, reorder points, etc.)")
        prompt_parts.append("")
        prompt_parts.append("Provide your analysis in a structured format, referencing the pre-analyzed patterns.")
        prompt_parts.append("=" * 80)
        
        return "\n".join(prompt_parts)
    
    def build_viability_prompt(self, product_description: str,
                              product_attributes: Dict[str, Any],
                              similar_products_data: Optional[List[Dict]] = None) -> str:
        """
        Build a product viability assessment prompt.
        
        Args:
            product_description: Product description
            product_attributes: Product attributes
            similar_products_data: Similar products data
            
        Returns:
            Formatted prompt
        """
        data_context = self.get_data_context()
        
        prompt_parts = []
        prompt_parts.append("PRODUCT VIABILITY ASSESSMENT")
        prompt_parts.append("")
        prompt_parts.append("HISTORICAL DATA:")
        prompt_parts.append(data_context)
        prompt_parts.append("")
        prompt_parts.append("PRODUCT TO ASSESS:")
        prompt_parts.append(f"Description: {product_description}")
        prompt_parts.append(f"Attributes: {product_attributes}")
        prompt_parts.append("")
        
        if similar_products_data:
            prompt_parts.append("SIMILAR PRODUCTS:")
            for similar in similar_products_data[:3]:
                name = similar.get('name', 'Unknown')
                sales = similar.get('sales', {})
                if sales:
                    prompt_parts.append(f"- {name}: {sales.get('total_units', 0):,} units sold")
        
        prompt_parts.append("")
        prompt_parts.append("Assess the viability of this product based on:")
        prompt_parts.append("1. Similar products' performance")
        prompt_parts.append("2. Market trends from historical data")
        prompt_parts.append("3. Category performance")
        prompt_parts.append("4. Seasonal patterns")
        prompt_parts.append("")
        prompt_parts.append("Provide a viability score (0-1) and detailed reasoning.")
        
        return "\n".join(prompt_parts)

