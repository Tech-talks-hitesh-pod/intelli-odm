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
            text_for_embedding = description
            if not text_for_embedding:
                text_for_embedding = self._attributes_to_text(attributes)
            
            # Generate embedding
            embedding = self._generate_embedding(text_for_embedding)
            
            # Prepare metadata
            store_metadata = {
                "product_id": product_id,
                "attributes": json.dumps(attributes),
                "description": description,
                "created_at": datetime.now().isoformat(),
                **(metadata or {})
            }
            
            # Store in ChromaDB
            self._products_collection.add(
                ids=[product_id],
                embeddings=[embedding.tolist()],
                metadatas=[store_metadata],
                documents=[text_for_embedding]
            )
            
            logger.info(f"Stored product {product_id} in knowledge base")
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
        try:
            # Create query text
            query_text = query_description
            if not query_text:
                query_text = self._attributes_to_text(query_attributes)
            
            # Generate query embedding
            query_embedding = self._generate_embedding(query_text)
            
            # Search for similar products
            results = self._products_collection.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=top_k,
                include=["metadatas", "documents", "distances"]
            )
            
            # Process results
            similar_products = []
            if results["ids"] and results["ids"][0]:
                for i, product_id in enumerate(results["ids"][0]):
                    metadata = results["metadatas"][0][i]
                    distance = results["distances"][0][i]
                    
                    # Convert distance to similarity score (0-1)
                    similarity_score = max(0.0, 1.0 - distance)
                    
                    # Parse stored attributes
                    stored_attributes = json.loads(metadata.get("attributes", "{}"))
                    
                    # Get product name from metadata if available
                    product_name = metadata.get("product_name", metadata.get("description", ""))
                    if not product_name:
                        # Try to extract from description
                        description = metadata.get("description", "")
                        if description:
                            # Take first part as name
                            product_name = description.split('.')[0] if '.' in description else description.split('\n')[0]
                    
                    similar_products.append({
                        "product_id": product_id,
                        "name": product_name,  # Add name field
                        "similarity_score": similarity_score,
                        "attributes": stored_attributes,
                        "description": metadata.get("description", ""),
                        "metadata": {k: v for k, v in metadata.items() 
                                   if k not in ["product_id", "attributes", "description"]}
                    })
            
            logger.info(f"Found {len(similar_products)} similar products")
            return similar_products
            
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