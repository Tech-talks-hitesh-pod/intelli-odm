"""Attribute & Analogy Agent for LLM-powered attribute extraction and product similarity."""

import logging
import json
from typing import Dict, List, Optional, Any, Tuple
from config.agent_configs import ATTRIBUTE_AGENT_CONFIG
from shared_knowledge_base import SharedKnowledgeBase
from utils.llm_client import LLMClient, LLMError, parse_json_response, retry_llm_call

logger = logging.getLogger(__name__)

class AttributeExtractionError(Exception):
    """Raised when attribute extraction fails."""
    pass

class AttributeAnalogyAgent:
    """
    Agent for extracting product attributes and finding analogous products.
    
    Uses LLM reasoning for attribute parsing and vector similarity
    for finding comparable products from the knowledge base.
    """
    
    def __init__(self, llm_client: LLMClient, knowledge_base: SharedKnowledgeBase, 
                 config: Optional[Dict] = None):
        """
        Initialize with LLM client and knowledge base.
        
        Args:
            llm_client: LLM client instance
            knowledge_base: Shared knowledge base instance
            config: Agent configuration
        """
        self.llm_client = llm_client
        self.knowledge_base = knowledge_base
        self.config = config or ATTRIBUTE_AGENT_CONFIG
        
        # Verify LLM availability
        if not self.llm_client.is_available():
            logger.warning("LLM client is not available - some features may not work")
            
        # ODM cleaning prompt (same as data ingestion agent)
        self.odm_cleaning_prompt = self._build_odm_cleaning_prompt()
    
    def extract_attributes(self, product_description: str) -> Dict[str, Any]:
        """
        Extract structured attributes from product description using LLM.
        
        Args:
            product_description: Natural language product description
            
        Returns:
            Dictionary of extracted attributes
            
        Raises:
            AttributeExtractionError: If extraction fails
        """
        logger.info(f"Extracting attributes from: {product_description[:100]}...")
        
        try:
            # Get prompt template
            prompt_template = self.config["prompt_templates"]["attribute_extraction"]
            prompt = prompt_template.format(description=product_description)
            
            # Call LLM with retry logic
            llm_config = self.config["llm"]
            response = retry_llm_call(
                self.llm_client,
                prompt,
                max_retries=llm_config["max_retries"],
                temperature=llm_config["temperature"],
                max_tokens=llm_config["max_tokens"]
            )
            
            # Parse JSON response
            fallback_attributes = {
                "material": "Unknown",
                "color": "Unknown", 
                "pattern": "Unknown",
                "sleeve": "Unknown",
                "neckline": "Unknown",
                "fit": "Unknown",
                "category": "Unknown",
                "style": "Unknown",
                "target_gender": "Unknown"
            }
            
            attributes = parse_json_response(response['response'], fallback_attributes)
            
            # Add confidence score based on how many attributes were extracted
            known_attrs = sum(1 for v in attributes.values() if v != "Unknown" and v)
            total_attrs = len(fallback_attributes)
            confidence = known_attrs / total_attrs
            attributes['confidence'] = confidence
            
            # Validate and clean attributes
            attributes = self._validate_and_clean_attributes(attributes)
            
            logger.info(f"Extracted attributes with confidence {confidence:.2f}: {attributes}")
            return attributes
            
        except LLMError as e:
            logger.error(f"LLM extraction failed: {e}")
            # Return basic attributes parsed from description
            return self._fallback_attribute_extraction(product_description)
        
        except Exception as e:
            logger.error(f"Attribute extraction failed: {e}")
            raise AttributeExtractionError(f"Failed to extract attributes: {e}")
    
    def _validate_and_clean_attributes(self, attributes: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and clean extracted attributes."""
        cleaned = {}
        
        # Define valid values for each attribute
        valid_values = {
            "material": ["Cotton", "Polyester", "Cotton Blend", "Linen", "Denim", "Chiffon", "Crepe", "Silk"],
            "color": ["White", "Black", "Navy", "Gray", "Red", "Blue", "Green", "Yellow", "Pink", "Brown", "Floral"],
            "pattern": ["Solid", "Stripes", "Printed", "Logo", "Embroidered", "Checks", "Floral", "Abstract"],
            "sleeve": ["Short", "Long", "Sleeveless", "3/4", "Full Length"],
            "neckline": ["Crew", "V-neck", "Scoop", "Round", "Collar", "Polo Collar", "Square"],
            "fit": ["Regular", "Slim", "Relaxed", "Oversized", "Skinny", "Straight"],
            "category": ["TSHIRT", "POLO", "DRESS", "JEANS", "SHIRT", "SWEATER", "JACKET"],
            "style": ["Casual", "Formal", "Athletic", "Business", "Party"],
            "target_gender": ["Men", "Women", "Unisex", "Boys", "Girls"]
        }
        
        for key, value in attributes.items():
            if key == 'confidence':
                cleaned[key] = float(value) if value else 0.5
                continue
            
            if not value or value == "Unknown":
                cleaned[key] = "Unknown"
                continue
            
            # Clean the value
            clean_value = str(value).strip().title()
            
            # Validate against known values
            if key in valid_values:
                # Find best match (case insensitive)
                matches = [v for v in valid_values[key] if v.lower() == clean_value.lower()]
                if matches:
                    cleaned[key] = matches[0]
                else:
                    # Look for partial matches
                    partial_matches = [v for v in valid_values[key] if clean_value.lower() in v.lower()]
                    if partial_matches:
                        cleaned[key] = partial_matches[0]
                    else:
                        cleaned[key] = clean_value  # Keep original if no match
            else:
                cleaned[key] = clean_value
        
        return cleaned
    
    def _fallback_attribute_extraction(self, description: str) -> Dict[str, Any]:
        """Fallback attribute extraction using rule-based parsing."""
        logger.info("Using fallback attribute extraction")
        
        description_lower = description.lower()
        
        attributes = {
            "material": "Unknown",
            "color": "Unknown",
            "pattern": "Unknown", 
            "sleeve": "Unknown",
            "neckline": "Unknown",
            "fit": "Unknown",
            "category": "Unknown",
            "style": "Casual",
            "target_gender": "Unisex",
            "confidence": 0.3  # Lower confidence for fallback
        }
        
        # Extract material
        if "cotton" in description_lower:
            attributes["material"] = "Cotton"
        elif "polyester" in description_lower:
            attributes["material"] = "Polyester"
        elif "denim" in description_lower:
            attributes["material"] = "Denim"
        
        # Extract color
        colors = ["white", "black", "navy", "gray", "red", "blue", "green"]
        for color in colors:
            if color in description_lower:
                attributes["color"] = color.title()
                break
        
        # Extract category
        if any(word in description_lower for word in ["t-shirt", "tshirt", "tee"]):
            attributes["category"] = "TSHIRT"
        elif "polo" in description_lower:
            attributes["category"] = "POLO"
        elif "dress" in description_lower:
            attributes["category"] = "DRESS"
        elif "jeans" in description_lower:
            attributes["category"] = "JEANS"
        elif "shirt" in description_lower:
            attributes["category"] = "SHIRT"
        
        # Extract sleeve
        if "short sleeve" in description_lower:
            attributes["sleeve"] = "Short"
        elif "long sleeve" in description_lower:
            attributes["sleeve"] = "Long"
        elif "sleeveless" in description_lower:
            attributes["sleeve"] = "Sleeveless"
        
        # Extract pattern
        if "stripe" in description_lower:
            attributes["pattern"] = "Stripes"
        elif any(word in description_lower for word in ["print", "logo"]):
            attributes["pattern"] = "Printed"
        else:
            attributes["pattern"] = "Solid"
        
        return attributes
    
    def find_comparable_products(self, attributes: Dict[str, Any], 
                                top_n: int = 5) -> List[Dict[str, Any]]:
        """
        Find comparable products from knowledge base using similarity search.
        
        Args:
            attributes: Product attributes
            top_n: Number of similar products to return
            
        Returns:
            List of comparable products with similarity scores
        """
        logger.info(f"Finding comparable products for attributes: {attributes}")
        
        try:
            # Use knowledge base to find similar products
            similar_products = self.knowledge_base.find_similar_products(
                query_attributes=attributes,
                top_k=top_n
            )
            
            # Enhance with match reasons
            enhanced_products = []
            for product in similar_products:
                enhanced_product = product.copy()
                enhanced_product["match_reasons"] = self._generate_match_reasons(
                    attributes, product["attributes"]
                )
                enhanced_products.append(enhanced_product)
            
            logger.info(f"Found {len(enhanced_products)} comparable products")
            return enhanced_products
            
        except Exception as e:
            logger.error(f"Failed to find comparable products: {e}")
            return []
    
    def _generate_match_reasons(self, query_attrs: Dict, product_attrs: Dict) -> List[str]:
        """Generate human-readable match reasons."""
        reasons = []
        
        # Check exact matches
        exact_matches = [
            ("category", "Category"),
            ("material", "Material"), 
            ("color", "Color"),
            ("style", "Style")
        ]
        
        for attr_key, display_name in exact_matches:
            if (query_attrs.get(attr_key, "").lower() == 
                product_attrs.get(attr_key, "").lower() and 
                query_attrs.get(attr_key) != "Unknown"):
                reasons.append(f"{display_name} match")
        
        # Check partial matches
        if (query_attrs.get("target_gender", "").lower() == 
            product_attrs.get("target_gender", "").lower()):
            reasons.append("Target demographic")
        
        if not reasons:
            reasons.append("General similarity")
        
        return reasons
    
    def analyze_trends(self, attributes: Dict[str, Any], 
                      time_window: str = "3M") -> Dict[str, Any]:
        """
        Analyze market trends for the product category.
        
        Args:
            attributes: Product attributes
            time_window: Analysis time window
            
        Returns:
            Trend analysis results
        """
        logger.info(f"Analyzing trends for category: {attributes.get('category', 'Unknown')}")
        
        try:
            # Get trend data from knowledge base
            category = attributes.get('category', 'Unknown')
            kb_trends = self.knowledge_base.query_trends(category, time_window)
            
            # Enhance with LLM analysis
            llm_insights = self._get_llm_trend_insights(attributes, kb_trends)
            
            trend_analysis = {
                "category": category,
                "time_window": time_window,
                "category_trend": "stable",  # Default
                "trend_strength": 0.7,
                "seasonal_pattern": self._analyze_seasonal_patterns(attributes),
                "color_trends": self._analyze_color_trends(attributes),
                "style_preferences": self._analyze_style_preferences(attributes),
                "market_insights": llm_insights,
                "confidence": 0.75,
                "kb_data": kb_trends
            }
            
            logger.info("Trend analysis completed")
            return trend_analysis
            
        except Exception as e:
            logger.error(f"Trend analysis failed: {e}")
            return {
                "category": attributes.get('category', 'Unknown'),
                "time_window": time_window,
                "error": str(e),
                "confidence": 0.0
            }
    
    def _get_llm_trend_insights(self, attributes: Dict, kb_trends: Dict) -> str:
        """Get LLM-generated trend insights."""
        try:
            prompt_template = self.config["prompt_templates"]["trend_analysis"]
            prompt = prompt_template.format(
                attributes=json.dumps(attributes, indent=2),
                time_period="3 months"
            )
            
            response = retry_llm_call(
                self.llm_client,
                prompt,
                max_retries=2,
                temperature=0.3,
                max_tokens=600
            )
            
            return response['response']
            
        except Exception as e:
            logger.warning(f"Failed to get LLM trend insights: {e}")
            return f"Analysis for {attributes.get('category', 'product')} category shows stable market conditions."
    
    def _analyze_seasonal_patterns(self, attributes: Dict) -> Dict[str, Any]:
        """Analyze seasonal patterns based on category."""
        category = attributes.get('category', '').upper()
        
        seasonal_patterns = {
            'TSHIRT': {
                'peak_months': ['March', 'April', 'May'],
                'low_months': ['November', 'December'],
                'seasonal_factor': 1.3
            },
            'DRESS': {
                'peak_months': ['April', 'May', 'December'],
                'low_months': ['August', 'September'],
                'seasonal_factor': 1.2
            },
            'JEANS': {
                'peak_months': ['September', 'October', 'November'],
                'low_months': ['May', 'June'],
                'seasonal_factor': 1.1
            }
        }
        
        return seasonal_patterns.get(category, {
            'peak_months': ['March', 'April'],
            'low_months': ['August', 'September'],
            'seasonal_factor': 1.0
        })
    
    def _analyze_color_trends(self, attributes: Dict) -> Dict[str, Any]:
        """Analyze color trends."""
        color = attributes.get('color', '').lower()
        
        # Simplified color trend analysis
        color_trends = {
            'white': {'popularity': 0.9, 'trend': 'Stable'},
            'black': {'popularity': 0.85, 'trend': 'Increasing'},
            'navy': {'popularity': 0.75, 'trend': 'Stable'},
            'gray': {'popularity': 0.7, 'trend': 'Increasing'}
        }
        
        return {
            color: color_trends.get(color, {'popularity': 0.6, 'trend': 'Stable'})
        }
    
    def _analyze_style_preferences(self, attributes: Dict) -> Dict[str, float]:
        """Analyze style preferences."""
        category = attributes.get('category', '').upper()
        
        style_preferences = {
            'TSHIRT': {'Casual': 0.8, 'Athletic': 0.2},
            'DRESS': {'Casual': 0.6, 'Formal': 0.3, 'Party': 0.1},
            'JEANS': {'Casual': 0.9, 'Business': 0.1},
            'SHIRT': {'Casual': 0.5, 'Formal': 0.4, 'Business': 0.1}
        }
        
        return style_preferences.get(category, {'Casual': 0.7, 'Formal': 0.3})
    
    def run(self, product_description: str) -> Tuple[Dict[str, Any], List[Dict[str, Any]], Dict[str, Any]]:
        """
        Execute complete attribute analysis workflow.
        
        Args:
            product_description: Product description to analyze
            
        Returns:
            Tuple of (attributes, comparables, trends)
        """
        logger.info("Starting attribute analysis workflow")
        
        try:
            # Step 1: Extract attributes
            attributes = self.extract_attributes(product_description)
            
            # Step 2: Find comparable products
            comparables = self.find_comparable_products(attributes)
            
            # Step 3: Analyze trends
            trends = self.analyze_trends(attributes)
            
            logger.info("Attribute analysis workflow completed successfully")
            return attributes, comparables, trends
            
        except Exception as e:
            logger.error(f"Attribute analysis workflow failed: {e}")
            raise
    
    def _build_odm_cleaning_prompt(self) -> str:
        """Build the ODM data cleaning and normalization prompt."""
        return """You are a Data Cleaning & Normalization Agent for an ODM (Own Design Manufacturing) retail system.

Your ONLY job:
- Take MESSY / UNSTRUCTURED product descriptions for NEW ODM ITEMS.
- Clean and normalize them.
- Output them in a STRICT, CONSISTENT, STRUCTURED FORMAT that matches the historical data schema.

For EVERY input product, you MUST output **exactly one JSON object** with the following fields:

Required core fields:

- StyleCode       (string)
- StyleName       (string)
- Colour          (string)
- Fit             (one of: "Slim", "Regular", "Relaxed", "Comfort", "Oversized" or "" if unknown)
- Fabric          (short descriptive string, e.g. "Poly Knit", "Cotton-Linen", "Silk Blend")
- Segment         (one of: "Mens", "Womens", "Kids", "Girls", "Boys" or "" if unknown)
- Family          (one of: "Formal", "Casual", "Active", "Lounge", "Ethnic", "Other")
- Class           (one of: "Top", "Bottom", "Sets")
- Brick           (canonical brick: "Jacket", "Shirt", "T-Shirt", "Kurta", "Kurta Set", "Dress", "Jeans", 
                   "Shorts", "Jogger", "Cargo", "Hoodie", "Thermal", "Leggings", "Dungaree", 
                   "Pyjama Set", "Lehenga Set", "Tracksuit", "Shoes", "Other")
- ProductGroup    (one of: "Knit", "Woven", "Denim", "Other")
- SeasonTag       (one of: "Winter", "Summer", "Festive", "Rain", "All", or "" if unknown)
- BasePrice       (integer in rupees, e.g. 1799; 0 if truly impossible to infer)
- PriceBand       (derived from BasePrice: "Low", "Mid", "Premium", or "" if BasePrice is 0)

Additional helpful fields:

- Pattern         (e.g. "Solid", "Floral", "Printed", "Check", "Stripe", "Embroidered", or "" if unknown)
- Sleeve          ("Full", "Half", "Sleeveless", "NA")
- Neck            ("Crew", "V-Neck", "Mandarin", "Spread Collar", "Stand Collar", "Collared", "NA")
- Occasion        (e.g. "Everyday", "Work", "Festive", "Wedding", "Lounge", "Gym", "Travel", "Brunch", or "")
- MaterialWeight  ("Light", "Medium", "Heavy", or "")
- Stretch         ("No", "Low", "Medium", "High", or "")
- Notes           (short free-text string for any extra info or ambiguity)

Always output VALID JSON only. No explanations or extra text."""

    def extract_odm_attributes(self, product_description: str) -> Dict[str, Any]:
        """
        Extract ODM-specific structured attributes from product description.
        
        Args:
            product_description: Natural language product description (potentially messy)
            
        Returns:
            Dictionary of normalized ODM attributes
        """
        logger.info(f"Extracting ODM attributes from: {product_description[:100]}...")
        
        try:
            # Create full prompt
            full_prompt = f"""{self.odm_cleaning_prompt}

------------------------------------------------------------
INPUT PRODUCT DESCRIPTION:
------------------------------------------------------------

{product_description}

------------------------------------------------------------
OUTPUT (JSON OBJECT ONLY):
------------------------------------------------------------"""

            # Call LLM with low temperature for consistent output
            response = self.llm_client.generate(
                prompt=full_prompt,
                max_tokens=1000,
                temperature=0.1
            )
            
            # Parse JSON response
            import json
            try:
                attributes = json.loads(response)
                
                if not isinstance(attributes, dict):
                    logger.error("LLM response is not a JSON object")
                    return self._get_fallback_odm_attributes()
                
                # Validate required fields
                required_fields = [
                    'StyleCode', 'StyleName', 'Colour', 'Fit', 'Fabric', 'Segment',
                    'Family', 'Class', 'Brick', 'ProductGroup', 'SeasonTag', 
                    'BasePrice', 'PriceBand'
                ]
                
                # Fill missing required fields with defaults
                for field in required_fields:
                    if field not in attributes:
                        if field == 'BasePrice':
                            attributes[field] = 0
                        else:
                            attributes[field] = ""
                
                # Ensure BasePrice is integer
                try:
                    attributes['BasePrice'] = int(attributes['BasePrice'])
                except (ValueError, TypeError):
                    attributes['BasePrice'] = 0
                
                # Auto-derive PriceBand if missing or incorrect
                if attributes['BasePrice'] == 0:
                    attributes['PriceBand'] = ""
                elif attributes['BasePrice'] < 1000:
                    attributes['PriceBand'] = "Low"
                elif attributes['BasePrice'] < 2000:
                    attributes['PriceBand'] = "Mid"
                else:
                    attributes['PriceBand'] = "Premium"
                
                logger.info(f"Successfully extracted ODM attributes: {attributes.get('StyleName', 'Unknown')}")
                return attributes
                
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse LLM JSON response: {e}")
                logger.debug(f"Raw response: {response}")
                return self._get_fallback_odm_attributes()
                
        except Exception as e:
            logger.error(f"ODM attribute extraction failed: {e}")
            return self._get_fallback_odm_attributes()
    
    def _get_fallback_odm_attributes(self) -> Dict[str, Any]:
        """Get fallback ODM attributes when extraction fails."""
        return {
            'StyleCode': '',
            'StyleName': 'Unknown Product',
            'Colour': '',
            'Fit': '',
            'Fabric': '',
            'Segment': '',
            'Family': 'Other',
            'Class': '',
            'Brick': 'Other',
            'ProductGroup': 'Other',
            'SeasonTag': '',
            'BasePrice': 0,
            'PriceBand': '',
            'Pattern': '',
            'Sleeve': 'NA',
            'Neck': 'NA',
            'Occasion': '',
            'MaterialWeight': '',
            'Stretch': '',
            'Notes': 'Extraction failed - manual review needed'
        }
    
    def find_similar_odm_products(self, odm_attributes: Dict[str, Any], limit: int = 5) -> List[Dict[str, Any]]:
        """
        Find similar products using ODM attributes for vector search.
        
        Args:
            odm_attributes: Normalized ODM attributes dictionary
            limit: Maximum number of similar products to return
            
        Returns:
            List of similar products with performance data
        """
        logger.info(f"Finding similar ODM products for: {odm_attributes.get('StyleName', 'Unknown')}")
        
        try:
            # Convert ODM attributes to search query
            search_text_parts = []
            
            # Build descriptive text from ODM attributes
            if odm_attributes.get('StyleName'):
                search_text_parts.append(odm_attributes['StyleName'])
            if odm_attributes.get('Brick'):
                search_text_parts.append(odm_attributes['Brick'])
            if odm_attributes.get('Fabric'):
                search_text_parts.append(odm_attributes['Fabric'])
            if odm_attributes.get('Colour'):
                search_text_parts.append(odm_attributes['Colour'])
            if odm_attributes.get('Pattern'):
                search_text_parts.append(odm_attributes['Pattern'])
            if odm_attributes.get('Fit'):
                search_text_parts.append(odm_attributes['Fit'])
            
            search_text = ' '.join(search_text_parts)
            
            # Use the existing similarity search
            similar_products = self.knowledge_base.find_similar_products(
                query_attributes=odm_attributes,
                limit=limit
            )
            
            logger.info(f"Found {len(similar_products)} similar ODM products")
            return similar_products
            
        except Exception as e:
            logger.error(f"Failed to find similar ODM products: {e}")
            return []