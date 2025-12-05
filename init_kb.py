#!/usr/bin/env python3
"""Initialize knowledge base with sample data."""

import sys
from shared_knowledge_base import SharedKnowledgeBase

def main():
    try:
        kb = SharedKnowledgeBase(persist_directory="data/chroma_db")
        
        # Add some sample products for similarity search
        sample_products = [
            {"id": "sample_tshirt", "attributes": {"category": "TSHIRT", "material": "Cotton", "color": "White"}, "description": "Basic white t-shirt"},
            {"id": "sample_jeans", "attributes": {"category": "JEANS", "material": "Denim", "color": "Blue"}, "description": "Classic blue jeans"},
            {"id": "sample_dress", "attributes": {"category": "DRESS", "material": "Chiffon", "color": "Black"}, "description": "Elegant black dress"}
        ]
        
        for product in sample_products:
            kb.store_product(
                product_id=product["id"],
                attributes=product["attributes"], 
                description=product["description"]
            )
        
        print("✅ Knowledge base initialized with sample data")
        
    except Exception as e:
        print(f"❌ Knowledge base initialization failed: {e}")

if __name__ == "__main__":
    main()
