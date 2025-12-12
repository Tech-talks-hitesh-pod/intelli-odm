# agents/attribute_analogy_agent.py

import pandas as pd
import json
from typing import Dict, List, Any, Optional
from utils.ollama_client import OllamaClient
from utils.audit_logger import AuditLogger, LogStatus

class AttributeAnalogyAgent:
    """
    Attribute Analogy Agent using Ollama (llama3:8b)
    Standardizes product attributes and finds comparable products
    """
    
    def __init__(self, ollama_client: OllamaClient, knowledge_base, audit_logger: Optional[AuditLogger] = None):
        """
        Initialize Attribute Analogy Agent
        
        Args:
            ollama_client: Ollama client instance
            knowledge_base: Shared knowledge base instance
            audit_logger: Optional audit logger for tracking operations
        """
        self.ollama = ollama_client
        self.kb = knowledge_base
        self.audit_logger = audit_logger
        
        # System prompt for attribute standardization
        self.standardize_system_prompt = """You are an expert product attribute analyzer. 
Your task is to extract and standardize product attributes from natural language descriptions.

Product attributes typically include:
- Style Code
- Color
- Segment (e.g., Casual, Formal, Sports)
- Family (e.g., T-Shirts, Jeans, Dresses)
- Class (e.g., Basic, Premium, Designer)
- Brick (e.g., Men's Apparel, Women's Apparel, Kids' Apparel)
- Fabric/Material
- Pattern
- Sleeve Type
- Fit Type
- And other relevant attributes

Return your response as a JSON object with standardized attribute names and values.
Example format:
{
  "style_code": "TSH-001",
  "color": "White",
  "segment": "Casual",
  "family": "T-Shirts",
  "class": "Basic",
  "brick": "Men's Apparel",
  "fabric": "Cotton",
  "pattern": "Solid",
  "sleeve_type": "Short Sleeves"
}"""
        
        # System prompt for finding comparables
        self.comparable_system_prompt = """You are an expert at finding similar products based on attributes.
Given a product's attributes, identify which attributes are most important for similarity matching.

Consider:
- Primary attributes: Family, Segment, Brick (must match closely)
- Secondary attributes: Color, Pattern, Fabric (should match if possible)
- Tertiary attributes: Style details, Fit (nice to have)

Return a JSON object with:
{
  "similarity_score": 0.0-1.0,
  "matching_attributes": ["attribute1", "attribute2"],
  "reasoning": "explanation of similarity"
}"""
    
    def standardize_attributes(self, product_description: str) -> Dict[str, Any]:
        """
        Use Ollama to convert natural language product description to structured attributes
        
        Args:
            product_description: Natural language description of the product
            
        Returns:
            Dictionary of standardized attributes
        """
        try:
            if self.audit_logger:
                self.audit_logger.log_agent_operation(
                    agent_name="AttributeAnalogyAgent",
                    description=f"Standardizing attributes for product: {product_description[:100]}",
                    status=LogStatus.IN_PROGRESS,
                    inputs={"product_description": product_description}
                )
            
            prompt = f"""Extract and standardize all product attributes from the following description:

{product_description}

Return only a valid JSON object with the standardized attributes. Do not include any additional text or explanation."""
            
            # Call Ollama
            response = self.ollama.generate(
                prompt=prompt,
                system_prompt=self.standardize_system_prompt,
                max_tokens=500,
                temperature=0.3  # Lower temperature for more consistent extraction
            )
            
            # Parse JSON response
            try:
                # Try to extract JSON from response (might have markdown code blocks)
                response = response.strip()
                if response.startswith("```json"):
                    response = response[7:]
                if response.startswith("```"):
                    response = response[3:]
                if response.endswith("```"):
                    response = response[:-3]
                response = response.strip()
                
                attributes = json.loads(response)
            except json.JSONDecodeError:
                # Fallback: try to extract key-value pairs manually
                attributes = self._parse_attributes_fallback(response)
            
            # Ensure required fields have defaults
            attributes.setdefault('style_code', 'N/A')
            attributes.setdefault('color', 'N/A')
            attributes.setdefault('segment', 'N/A')
            attributes.setdefault('family', 'N/A')
            attributes.setdefault('class', 'N/A')
            attributes.setdefault('brick', 'N/A')
            
            if self.audit_logger:
                self.audit_logger.log_agent_operation(
                    agent_name="AttributeAnalogyAgent",
                    description=f"Successfully standardized attributes",
                    status=LogStatus.SUCCESS,
                    inputs={"product_description": product_description},
                    outputs={"attributes": attributes}
                )
            
            return attributes
        
        except Exception as e:
            error_msg = str(e)
            if self.audit_logger:
                self.audit_logger.log_agent_operation(
                    agent_name="AttributeAnalogyAgent",
                    description=f"Failed to standardize attributes",
                    status=LogStatus.FAIL,
                    inputs={"product_description": product_description},
                    error=error_msg
                )
            # Return default attributes on error
            return {
                'style_code': 'N/A',
                'color': 'N/A',
                'segment': 'N/A',
                'family': 'N/A',
                'class': 'N/A',
                'brick': 'N/A'
            }
    
    def _parse_attributes_fallback(self, text: str) -> Dict[str, Any]:
        """Fallback parser if JSON parsing fails"""
        attributes = {}
        lines = text.split('\n')
        
        for line in lines:
            if ':' in line:
                key, value = line.split(':', 1)
                key = key.strip().lower().replace(' ', '_').replace('-', '_')
                value = value.strip().strip('"').strip("'").strip(',')
                attributes[key] = value
        
        return attributes
    
    def find_comparable_products(self, attributes: Dict[str, Any], historical_data: Optional[pd.DataFrame] = None) -> List[Dict[str, Any]]:
        """
        Find comparable products using Ollama to analyze attribute similarity
        
        Args:
            attributes: Standardized product attributes
            historical_data: Optional historical sales data to search
            
        Returns:
            List of comparable products with similarity scores
        """
        try:
            if self.audit_logger:
                self.audit_logger.log_agent_operation(
                    agent_name="AttributeAnalogyAgent",
                    description=f"Finding comparable products for attributes: {attributes.get('family', 'N/A')}",
                    status=LogStatus.IN_PROGRESS,
                    inputs={"attributes": attributes}
                )
            
            # Build prompt for finding comparables
            attributes_str = json.dumps(attributes, indent=2)
            
            prompt = f"""Given the following product attributes:
{attributes_str}

Analyze which attributes are most important for finding similar products and provide a similarity matching strategy.

Consider:
1. Which attributes must match exactly (e.g., Family, Brick)?
2. Which attributes should match closely (e.g., Segment, Color)?
3. Which attributes are less critical (e.g., specific style details)?

Return a JSON object with:
{{
  "primary_attributes": ["list of must-match attributes"],
  "secondary_attributes": ["list of should-match attributes"],
  "similarity_criteria": "explanation of matching strategy"
}}"""
            
            # Get similarity strategy from Ollama
            strategy_response = self.ollama.generate(
                prompt=prompt,
                system_prompt=self.comparable_system_prompt,
                max_tokens=300,
                temperature=0.5
            )
            
            # Parse strategy
            try:
                strategy = json.loads(self._extract_json(strategy_response))
            except:
                strategy = {
                    "primary_attributes": ["family", "brick"],
                    "secondary_attributes": ["segment", "color"],
                    "similarity_criteria": "Match on family and brick, prefer similar segment and color"
                }
            
            # Find comparables from knowledge base or historical data
            comparables = []
            
            if historical_data is not None and not historical_data.empty:
                # Search historical data for similar products
                comparables = self._find_comparables_in_data(
                    attributes, historical_data, strategy
                )
            else:
                # Use knowledge base
                comparables = self._find_comparables_in_kb(attributes, strategy)
            
            # Use Ollama to score similarity for each comparable
            scored_comparables = []
            for comp in comparables:
                similarity = self._calculate_similarity_with_llm(attributes, comp, strategy)
                comp['similarity'] = similarity
                scored_comparables.append(comp)
            
            # Sort by similarity
            scored_comparables.sort(key=lambda x: x.get('similarity', 0), reverse=True)
            
            if self.audit_logger:
                self.audit_logger.log_agent_operation(
                    agent_name="AttributeAnalogyAgent",
                    description=f"Found {len(scored_comparables)} comparable products",
                    status=LogStatus.SUCCESS,
                    inputs={"attributes": attributes},
                    outputs={"comparables_count": len(scored_comparables), "top_similarity": scored_comparables[0].get('similarity', 0) if scored_comparables else 0}
                )
            
            return scored_comparables[:10]  # Return top 10
        
        except Exception as e:
            error_msg = str(e)
            if self.audit_logger:
                self.audit_logger.log_agent_operation(
                    agent_name="AttributeAnalogyAgent",
                    description=f"Failed to find comparable products",
                    status=LogStatus.FAIL,
                    inputs={"attributes": attributes},
                    error=error_msg
                )
            return []
    
    def _find_comparables_in_data(self, attributes: Dict[str, Any], data: pd.DataFrame, strategy: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Find comparables in historical data"""
        comparables = []
        
        if 'sku' not in data.columns:
            return comparables
        
        # Filter by primary attributes if available in data
        primary_attrs = strategy.get('primary_attributes', ['family', 'brick'])
        
        filtered_data = data.copy()
        
        # Try to match on available attributes
        for attr in primary_attrs:
            if attr in attributes and attr in data.columns:
                value = attributes[attr]
                if value and value != 'N/A':
                    filtered_data = filtered_data[filtered_data[attr] == value]
        
        # Get unique SKUs
        unique_skus = filtered_data['sku'].unique()[:20]  # Limit to 20 for performance
        
        for sku in unique_skus:
            sku_data = data[data['sku'] == sku].iloc[0]
            comp = {
                'sku': sku,
                'attributes': {}
            }
            
            # Extract attributes from data
            for attr in ['style_code', 'color', 'segment', 'family', 'class', 'brick']:
                if attr in sku_data:
                    comp['attributes'][attr] = sku_data[attr]
            
            comparables.append(comp)
        
        return comparables
    
    def _find_comparables_in_kb(self, attributes: Dict[str, Any], strategy: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Find comparables in knowledge base"""
        # Placeholder - implement KB query logic
        return []
    
    def _calculate_similarity_with_llm(self, target_attrs: Dict[str, Any], comparable: Dict[str, Any], strategy: Dict[str, Any]) -> float:
        """Use Ollama to calculate similarity score"""
        try:
            target_str = json.dumps(target_attrs, indent=2)
            comp_str = json.dumps(comparable.get('attributes', {}), indent=2)
            
            prompt = f"""Compare these two products and calculate a similarity score (0.0 to 1.0):

Target Product:
{target_str}

Comparable Product:
{comp_str}

Matching Strategy:
Primary attributes (must match): {strategy.get('primary_attributes', [])}
Secondary attributes (should match): {strategy.get('secondary_attributes', [])}

Return only a JSON object:
{{
  "similarity_score": 0.0-1.0,
  "reasoning": "brief explanation"
}}"""
            
            response = self.ollama.generate(
                prompt=prompt,
                system_prompt=self.comparable_system_prompt,
                max_tokens=200,
                temperature=0.3
            )
            
            try:
                result = json.loads(self._extract_json(response))
                return float(result.get('similarity_score', 0.5))
            except:
                # Fallback: simple attribute matching
                return self._simple_similarity(target_attrs, comparable.get('attributes', {}))
        
        except:
            return self._simple_similarity(target_attrs, comparable.get('attributes', {}))
    
    def _simple_similarity(self, attrs1: Dict[str, Any], attrs2: Dict[str, Any]) -> float:
        """Simple similarity calculation as fallback"""
        if not attrs2:
            return 0.0
        
        matches = 0
        total = 0
        
        for key in ['family', 'brick', 'segment', 'color']:
            if key in attrs1 and key in attrs2:
                total += 1
                if attrs1[key] == attrs2[key] and attrs1[key] != 'N/A':
                    matches += 1
        
        return matches / total if total > 0 else 0.0
    
    def _extract_json(self, text: str) -> str:
        """Extract JSON from text that might have markdown or extra text"""
        text = text.strip()
        
        # Remove markdown code blocks
        if text.startswith("```json"):
            text = text[7:]
        if text.startswith("```"):
            text = text[3:]
        if text.endswith("```"):
            text = text[:-3]
        
        text = text.strip()
        
        # Find JSON object
        start = text.find('{')
        end = text.rfind('}') + 1
        
        if start >= 0 and end > start:
            return text[start:end]
        
        return text
    
    def analyze_trends(self, attributes: Dict[str, Any], historical_data: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """
        Use Ollama to analyze trends for products with given attributes
        
        Args:
            attributes: Product attributes
            historical_data: Optional historical sales data
            
        Returns:
            Trend analysis results
        """
        try:
            if self.audit_logger:
                self.audit_logger.log_agent_operation(
                    agent_name="AttributeAnalogyAgent",
                    description=f"Analyzing trends for attributes: {attributes.get('family', 'N/A')}",
                    status=LogStatus.IN_PROGRESS,
                    inputs={"attributes": attributes}
                )
            
            attributes_str = json.dumps(attributes, indent=2)
            
            # Prepare historical summary if available
            historical_summary = ""
            if historical_data is not None and not historical_data.empty:
                # Summarize historical performance
                if 'units_sold' in historical_data.columns:
                    total_sales = historical_data['units_sold'].sum()
                    avg_sales = historical_data['units_sold'].mean()
                    historical_summary = f"\nHistorical Performance:\n- Total Units Sold: {total_sales}\n- Average Units Sold: {avg_sales:.2f}"
            
            prompt = f"""Analyze market trends for a product with these attributes:
{attributes_str}
{historical_summary}

Provide trend analysis including:
1. Market demand trend (increasing/stable/decreasing)
2. Seasonal patterns (if any)
3. Competitive positioning
4. Recommendation for procurement

Return a JSON object:
{{
  "demand_trend": "increasing|stable|decreasing",
  "trend_strength": 0.0-1.0,
  "seasonal_patterns": ["pattern1", "pattern2"],
  "competitive_position": "strong|moderate|weak",
  "recommendation": "high|moderate|low",
  "reasoning": "explanation"
}}"""
            
            response = self.ollama.generate(
                prompt=prompt,
                system_prompt="You are a retail market trend analyst. Analyze product trends based on attributes and historical data.",
                max_tokens=400,
                temperature=0.6
            )
            
            try:
                trends = json.loads(self._extract_json(response))
            except:
                trends = {
                    "demand_trend": "stable",
                    "trend_strength": 0.5,
                    "seasonal_patterns": [],
                    "competitive_position": "moderate",
                    "recommendation": "moderate",
                    "reasoning": "Unable to analyze trends"
                }
            
            if self.audit_logger:
                self.audit_logger.log_agent_operation(
                    agent_name="AttributeAnalogyAgent",
                    description=f"Trend analysis completed",
                    status=LogStatus.SUCCESS,
                    inputs={"attributes": attributes},
                    outputs={"trends": trends}
                )
            
            return trends
        
        except Exception as e:
            error_msg = str(e)
            if self.audit_logger:
                self.audit_logger.log_agent_operation(
                    agent_name="AttributeAnalogyAgent",
                    description=f"Failed to analyze trends",
                    status=LogStatus.FAIL,
                    inputs={"attributes": attributes},
                    error=error_msg
                )
            
            return {
                "demand_trend": "unknown",
                "trend_strength": 0.0,
                "seasonal_patterns": [],
                "competitive_position": "unknown",
                "recommendation": "low",
                "reasoning": f"Error: {error_msg}"
            }
    
    def run(self, product_description: str, historical_data: Optional[pd.DataFrame] = None) -> tuple:
        """
        Main run method: standardize attributes, find comparables, analyze trends
        
        Args:
            product_description: Natural language product description
            historical_data: Optional historical sales data
            
        Returns:
            Tuple of (comparables, trends)
        """
        # Step 1: Standardize attributes
        attributes = self.standardize_attributes(product_description)
        
        # Step 2: Find comparable products
        comparables = self.find_comparable_products(attributes, historical_data)
        
        # Step 3: Analyze trends
        trends = self.analyze_trends(attributes, historical_data)
        
        return comparables, trends
