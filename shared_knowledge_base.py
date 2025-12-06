"""Shared Knowledge Base with ChromaDB vector storage for product similarity and historical performance."""

import logging
import json
import uuid
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
import numpy as np
import pandas as pd
import chromadb
from chromadb.config import Settings as ChromaSettings
from sentence_transformers import SentenceTransformer
from config.settings import settings
from config.agent_configs import KNOWLEDGE_BASE_CONFIG

logger = logging.getLogger(__name__)

class SharedKnowledgeBase:
    """
    Vector database interface for storing and retrieving product information,
    embeddings, and historical performance data.
    """
    
    def __init__(self):
        """Initialize ChromaDB client and embedding model."""
        self._init_chromadb()
        self._init_embedding_model()
        self._products_collection = self._get_or_create_collection("products")
        self._performance_collection = self._get_or_create_collection("performance")
        
    def _init_chromadb(self):
        """Initialize ChromaDB client."""
        try:
            if settings.vector_db_type == "chromadb":
                # Create persistent client
                self.client = chromadb.PersistentClient(
                    path=settings.chromadb_persist_dir,
                    settings=ChromaSettings(
                        anonymized_telemetry=False,
                        allow_reset=True
                    )
                )
                logger.info(f"Connected to ChromaDB at {settings.chromadb_persist_dir}")
            else:
                raise ValueError(f"Unsupported vector DB type: {settings.vector_db_type}")
                
        except Exception as e:
            logger.error(f"Failed to initialize ChromaDB: {e}")
            raise
    
    def _init_embedding_model(self):
        """Initialize sentence transformer model for embeddings."""
        try:
            model_name = KNOWLEDGE_BASE_CONFIG["embeddings"]["model_name"]
            self.embedding_model = SentenceTransformer(model_name)
            logger.info(f"Loaded embedding model: {model_name}")
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            raise
    
    def _get_or_create_collection(self, collection_name: str):
        """Get or create a ChromaDB collection."""
        try:
            collection = self.client.get_or_create_collection(
                name=collection_name,
                metadata={"hnsw:space": "cosine"}
            )
            logger.info(f"Using collection: {collection_name}")
            return collection
        except Exception as e:
            logger.error(f"Failed to create collection {collection_name}: {e}")
            raise
    
    def _generate_embedding(self, text: str) -> np.ndarray:
        """Generate embedding for text using sentence transformer."""
        try:
            embedding = self.embedding_model.encode(text, convert_to_numpy=True)
            return embedding.astype('float32')
        except Exception as e:
            logger.error(f"Failed to generate embedding for text: {e}")
            raise
    
    def _attributes_to_text(self, attributes: Dict[str, Any]) -> str:
        """Convert product attributes to text for embedding."""
        # Create a descriptive text from attributes
        parts = []
        
        # Add key attributes in order of importance
        if attributes.get('category'):
            parts.append(f"Category: {attributes['category']}")
        if attributes.get('material'):
            parts.append(f"Material: {attributes['material']}")
        if attributes.get('color'):
            parts.append(f"Color: {attributes['color']}")
        if attributes.get('pattern'):
            parts.append(f"Pattern: {attributes['pattern']}")
        if attributes.get('sleeve'):
            parts.append(f"Sleeve: {attributes['sleeve']}")
        if attributes.get('fit'):
            parts.append(f"Fit: {attributes['fit']}")
        if attributes.get('style'):
            parts.append(f"Style: {attributes['style']}")
        
        return " | ".join(parts)
    
    def store_product(self, product_id: str, attributes: Dict[str, Any], 
                     description: str = "", metadata: Optional[Dict] = None) -> bool:
        """
        Store product information with vector embeddings.
        
        Args:
            product_id: Unique product identifier
            attributes: Structured product attributes
            description: Product description
            metadata: Additional metadata
            
        Returns:
            bool: Success status
        """
        try:
            # Combine description and attributes for embedding
            # Include both for better semantic matching
            text_parts = []
            if description:
                text_parts.append(description)
            
            # Add attributes text for better similarity matching
            attributes_text = self._attributes_to_text(attributes)
            if attributes_text:
                text_parts.append(attributes_text)
            
            text_for_embedding = " | ".join(text_parts) if text_parts else self._attributes_to_text(attributes)
            
            # Generate embedding from combined text
            logger.info(f"🧠 Generating embedding for product {product_id}")
            logger.debug(f"Embedding text: {text_for_embedding[:100]}...")
            embedding = self._generate_embedding(text_for_embedding)
            logger.info(f"✅ Generated {len(embedding)}-dimensional embedding for product {product_id}")
            
            # Prepare metadata
            store_metadata = {
                "product_id": product_id,
                "attributes": json.dumps(attributes),
                "description": description,
                "created_at": datetime.now().isoformat(),
                **(metadata or {})
            }
            
            # Store in ChromaDB
            logger.info(f"💾 Storing product {product_id} in ChromaDB vector database")
            self._products_collection.add(
                ids=[product_id],
                embeddings=[embedding.tolist()],
                metadatas=[store_metadata],
                documents=[text_for_embedding]
            )
            
            logger.info(f"✅ Successfully stored product {product_id} in vector database")
            return True
            
        except Exception as e:
            logger.error(f"Failed to store product {product_id}: {e}")
            return False
    
    def find_similar_products(self, query_attributes: Dict[str, Any], 
                            query_description: str = "", 
                            top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Find similar products using vector similarity search.
        
        Args:
            query_attributes: Product attributes to search for
            query_description: Product description to search for
            top_k: Number of similar products to return
            
        Returns:
            List of similar products with similarity scores
        """
        # Wrap with LangSmith tracing for vector database queries
        try:
            from langsmith import traceable
            
            # Use wrapper function approach to ensure proper nesting
            @traceable(
                name="Vector_DB_Search",
                run_type="retriever",
                tags=["vector-db", "similarity-search", "chromadb"]
            )
            def _search_wrapper():
                return self._find_similar_products_impl(query_attributes, query_description, top_k)
            
            return _search_wrapper()
        except ImportError:
            # If langsmith not available, use direct call
            return self._find_similar_products_impl(query_attributes, query_description, top_k)
        except Exception as e:
            logger.warning(f"LangSmith tracing failed for vector search: {e}")
            return self._find_similar_products_impl(query_attributes, query_description, top_k)
    
    def _find_similar_products_impl(self, query_attributes: Dict[str, Any], 
                                    query_description: str = "", 
                                    top_k: int = 5) -> List[Dict[str, Any]]:
        """Internal implementation of vector similarity search."""
        try:
            # First try real vector search from the database
            logger.info("Attempting real vector search for similar products")
            
            # Create query text
            search_query = query_description
            if not search_query:
                search_query = self._attributes_to_text(query_attributes)
            
            logger.info(f"Searching for products similar to: {search_query}")
            
            # Check if collection has any data
            try:
                collection_count = self._products_collection.count()
                logger.info(f"Vector database has {collection_count} products")
                
                if collection_count > 0:
                    # Generate query embedding
                    query_embedding = self._generate_embedding(search_query)
                    
                    # Search for similar products with distance threshold
                    # Use larger n_results to filter by similarity threshold later
                    results = self._products_collection.query(
                        query_embeddings=[query_embedding.tolist()],
                        n_results=min(top_k * 3, collection_count),  # Get more candidates to filter
                        include=["metadatas", "documents", "distances"]
                    )
                    
                    # Process real results
                    similar_products = []
                    if results["ids"] and len(results["ids"]) > 0:
                        for i, product_id in enumerate(results["ids"][0]):
                            metadata = results["metadatas"][0][i] if results["metadatas"] else {}
                            description = results["documents"][0][i] if results["documents"] else ""
                            distance = results["distances"][0][i] if results["distances"] else 1.0
                            
                            # Convert distance to similarity (lower distance = higher similarity)
                            # ChromaDB uses cosine distance (0-2 range), normalize to 0-1 similarity
                            # Distance 0 = perfect match (similarity 1.0)
                            # Distance 1 = orthogonal (similarity 0.5)
                            # Distance 2 = opposite (similarity 0.0)
                            similarity_score = max(0.0, 1.0 - (distance / 2.0))
                            
                            # Apply threshold - only include products with similarity > 0.3
                            if similarity_score < 0.3:
                                continue
                            
                            # Parse attributes
                            import json
                            attributes = json.loads(metadata.get("attributes", "{}"))
                            
                            product_data = {
                                'product_id': product_id,
                                'name': metadata.get('product_name', description.split('.')[0] if description else 'Unknown'),
                                'description': description,
                                'attributes': attributes,
                                'similarity_score': similarity_score,
                                'sales': {
                                    'total_units': int(metadata.get('sales_total_units', 0)),
                                    'total_revenue': float(metadata.get('sales_total_revenue', 0)),
                                    'avg_monthly_units': float(metadata.get('sales_avg_monthly', 0))
                                }
                            }
                            
                            similar_products.append(product_data)
                        
                        if similar_products:
                            logger.info(f"Found {len(similar_products)} real similar products from vector database")
                            return similar_products
                    
                logger.info("No products found in vector database")
                return []
                
            except Exception as e:
                logger.warning(f"Vector search failed: {e}")
                return []
            
            # NO FALLBACK: Return empty list if vector search fails
            logger.info("Vector database is empty or search failed - returning empty results")
            return []
            
        except Exception as e:
            logger.error(f"Failed to find similar products: {e}")
            return []
    
    def store_performance_data(self, product_id: str, performance_data: Dict[str, Any]) -> bool:
        """
        Store historical performance data for a product.
        
        Args:
            product_id: Product identifier
            performance_data: Performance metrics and data
            
        Returns:
            bool: Success status
        """
        try:
            # Create unique ID for performance record
            perf_id = f"{product_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            # Prepare metadata
            metadata = {
                "product_id": product_id,
                "performance_data": json.dumps(performance_data),
                "timestamp": datetime.now().isoformat()
            }
            
            # Store performance data (without embedding for now)
            self._performance_collection.add(
                ids=[perf_id],
                embeddings=[[0.0] * 384],  # Placeholder embedding
                metadatas=[metadata],
                documents=[f"Performance data for {product_id}"]
            )
            
            logger.info(f"Stored performance data for product {product_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to store performance data for {product_id}: {e}")
            return False
    
    def get_product_performance(self, product_id: str) -> List[Dict[str, Any]]:
        """
        Retrieve historical performance data for a product.
        
        Args:
            product_id: Product identifier
            
        Returns:
            List of performance records
        """
        try:
            # Search for performance records
            results = self._performance_collection.get(
                where={"product_id": product_id},
                include=["metadatas"]
            )
            
            performance_records = []
            if results["metadatas"]:
                for metadata in results["metadatas"]:
                    performance_data = json.loads(metadata.get("performance_data", "{}"))
                    performance_records.append({
                        "timestamp": metadata.get("timestamp"),
                        "data": performance_data
                    })
            
            logger.info(f"Retrieved {len(performance_records)} performance records for {product_id}")
            return performance_records
            
        except Exception as e:
            logger.error(f"Failed to retrieve performance data for {product_id}: {e}")
            return []
    
    def store_product_with_sales_data(self, product_id: str, attributes: Dict[str, Any],
                                     description: str, sales_data: Dict[str, Any],
                                     inventory_data: Optional[Dict[str, Any]] = None,
                                     pricing_data: Optional[Dict[str, Any]] = None) -> bool:
        """
        Store product with complete sales, inventory, and pricing data.
        This is called after data ingestion and attribute extraction.
        
        Args:
            product_id: Product identifier
            attributes: Extracted attributes from AttributeAnalogyAgent
            description: Product description
            sales_data: Aggregated sales data (total_units, total_revenue, avg_monthly, etc.)
            inventory_data: Current inventory data (on_hand, by_store, etc.)
            pricing_data: Pricing data (avg_price, price_range, etc.)
            
        Returns:
            bool: Success status
        """
        try:
            # Prepare comprehensive metadata
            metadata = {
                "product_name": description.split('.')[0] if '.' in description else description[:50],
                "sales_total_units": int(sales_data.get("total_units", 0)),
                "sales_total_revenue": float(sales_data.get("total_revenue", 0)),
                "sales_avg_monthly": float(sales_data.get("avg_monthly_units", 0)),
                "sales_date_range": json.dumps(sales_data.get("date_range", {})),
            }
            
            # Log sales data being stored
            logger.debug(f"Storing sales metadata for {product_id}: units={metadata['sales_total_units']}, revenue={metadata['sales_total_revenue']}")
            
            if inventory_data:
                metadata.update({
                    "inventory_total": inventory_data.get("total_on_hand", 0),
                    "inventory_by_store": json.dumps(inventory_data.get("by_store", {}))
                })
            
            if pricing_data:
                metadata.update({
                    "price_avg": pricing_data.get("avg_price", 0),
                    "price_min": pricing_data.get("min_price", 0),
                    "price_max": pricing_data.get("max_price", 0)
                })
            
            # Store product with all data
            return self.store_product(
                product_id=product_id,
                attributes=attributes,
                description=description,
                metadata=metadata
            )
            
        except Exception as e:
            logger.error(f"Failed to store product with sales data {product_id}: {e}")
            return False
    
    def get_all_products_with_performance(self) -> List[Dict[str, Any]]:
        """
        Retrieve all products with their performance data from the knowledge base.
        Used for comprehensive analysis before procurement evaluation.
        
        Returns:
            List of products with attributes and performance data
        """
        try:
            logger.info("Retrieving all products from vector database")
            
            # Get all products from vector database
            results = self._products_collection.get(
                include=["metadatas", "documents"]
            )
            
            products = []
            if results["ids"]:
                logger.info(f"Found {len(results['ids'])} products in vector database")
                
                for i, product_id in enumerate(results["ids"]):
                    metadata = results["metadatas"][i]
                    description = results["documents"][i] if results["documents"] else ""
                    
                    # Parse attributes
                    attributes = json.loads(metadata.get("attributes", "{}"))
                    
                    # Extract performance data from metadata
                    product_data = {
                        "product_id": product_id,
                        "name": metadata.get("product_name", description.split('.')[0] if description else product_id),
                        "description": description,
                        "attributes": attributes,
                        "sales": {
                            "total_units": int(metadata.get("sales_total_units", 0)),
                            "total_revenue": float(metadata.get("sales_total_revenue", 0)),
                            "avg_monthly_units": float(metadata.get("sales_avg_monthly", 0))
                        }
                    }
                    
                    # Add inventory if available
                    if metadata.get("inventory_total"):
                        product_data["inventory"] = {
                            "total": int(metadata.get("inventory_total", 0)),
                            "by_store": json.loads(metadata.get("inventory_by_store", "{}"))
                        }
                    
                    # Add pricing if available
                    if metadata.get("price_avg"):
                        product_data["pricing"] = {
                            "avg_price": float(metadata.get("price_avg", 0)),
                            "min_price": float(metadata.get("price_min", 0)),
                            "max_price": float(metadata.get("price_max", 0))
                        }
                    
                    products.append(product_data)
            
            logger.info(f"Retrieved {len(products)} products with performance data from vector DB")
            return products
            
        except Exception as e:
            logger.error(f"Failed to retrieve all products: {e}")
            return []
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about the knowledge base collections."""
        try:
            products_count = self._products_collection.count()
            performance_count = self._performance_collection.count()
            
            return {
                "products_count": products_count,
                "performance_records_count": performance_count,
                "status": "healthy"
            }
            
        except Exception as e:
            logger.error(f"Failed to get collection stats: {e}")
            return {"status": "error", "error": str(e)}
    
    def get_collection_size(self) -> int:
        """Get the number of products in the knowledge base."""
        try:
            return self._products_collection.count()
        except Exception as e:
            logger.error(f"Failed to get collection size: {e}")
            return 0
    
    def reset_collections(self):
        """Reset all collections (use with caution)."""
        try:
            self.client.delete_collection("products")
            self.client.delete_collection("performance")
            
            # Recreate collections
            self._products_collection = self._get_or_create_collection("products")
            self._performance_collection = self._get_or_create_collection("performance")
            
            logger.info("Reset all knowledge base collections")
            
        except Exception as e:
            logger.error(f"Failed to reset collections: {e}")
            raise
    
    def query_trends(self, category: str, time_window: str = "3M") -> Dict[str, Any]:
        """
        Query trend data for a specific category.
        
        Args:
            category: Product category
            time_window: Time window for analysis
            
        Returns:
            Trend analysis data
        """
        try:
            # This is a simplified implementation
            # In production, you'd analyze actual performance data
            
            # Search for products in the category
            results = self._products_collection.get(
                where_document={"$contains": f"Category: {category}"},
                include=["metadatas"]
            )
            
            trend_data = {
                "category": category,
                "time_window": time_window,
                "products_count": len(results["metadatas"]) if results["metadatas"] else 0,
                "trend_direction": "stable",  # Placeholder
                "confidence": 0.7  # Placeholder
            }
            
            return trend_data
            
        except Exception as e:
            logger.error(f"Failed to query trends for category {category}: {e}")
            return {"category": category, "error": str(e)}