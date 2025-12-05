# Data Flow Architecture

## Overview

The Intelli-ODM system now follows a clean data flow where:
1. **Data Ingestion** processes and validates data using Pydantic
2. **Attribute Analogy Agent** extracts attributes for all products
3. **All processed data is stored in Vector DB (ChromaDB)**
4. **Procurement/Demand agents pull data from Vector DB** to create prompts

## Data Flow Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                    Demo Data Load                            │
│              (User clicks "Load Scenario")                   │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│              DataIngestionAgent                               │
│  ┌─────────────────────────────────────────────────────┐    │
│  │ 1. Load CSV files (products, sales, inventory, etc) │    │
│  │ 2. Validate with Pydantic models                    │    │
│  │ 3. Structure and clean data                         │    │
│  │ 4. Feature engineering                              │    │
│  └─────────────────────────────────────────────────────┘    │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│         AttributeAnalogyAgent (called by DataIngestion)     │
│  ┌─────────────────────────────────────────────────────┐    │
│  │ For each product:                                    │    │
│  │ - Extract attributes using LLM                       │    │
│  │ - Standardize attributes                             │    │
│  │ - Return structured attributes                       │    │
│  └─────────────────────────────────────────────────────┘    │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│         SharedKnowledgeBase (Vector DB - ChromaDB)           │
│  ┌─────────────────────────────────────────────────────┐    │
│  │ store_product_with_sales_data()                     │    │
│  │ - Store product with attributes                      │    │
│  │ - Store aggregated sales data                       │    │
│  │ - Store inventory data                              │    │
│  │ - Store pricing data                                │    │
│  │ - Generate embeddings for similarity search         │    │
│  └─────────────────────────────────────────────────────┘    │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       │ (Data is now in Vector DB)
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│         New Product Evaluation                               │
│              (User enters new product)                       │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│         ProcurementAllocationAgent                           │
│  ┌─────────────────────────────────────────────────────┐    │
│  │ 1. analyze_all_products_from_kb()                   │    │
│  │    - Pulls all products from Vector DB              │    │
│  │    - Analyzes patterns by category/attributes       │    │
│  │    - Generates LLM insights                         │    │
│  │                                                      │    │
│  │ 2. find_similar_products()                          │    │
│  │    - Vector similarity search in ChromaDB           │    │
│  │    - Returns similar products with scores            │    │
│  │                                                      │    │
│  │ 3. get_all_products_with_performance()             │    │
│  │    - Retrieves sales/inventory/pricing data         │    │
│  │    - For similar products                            │    │
│  └─────────────────────────────────────────────────────┘    │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│              PromptBuilder                                   │
│  ┌─────────────────────────────────────────────────────┐    │
│  │ build_procurement_prompt()                          │    │
│  │ - Includes product analysis from KB                  │    │
│  │ - Includes similar products with performance         │    │
│  │ - Includes location context                         │    │
│  │ - Creates comprehensive prompt for LLM              │    │
│  └─────────────────────────────────────────────────────┘    │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│                    LLM (Ollama/OpenAI)                       │
│              Generates procurement recommendation            │
└─────────────────────────────────────────────────────────────┘
```

## Key Methods

### DataIngestionAgent

1. **`load_and_process_demo_data(data_dir)`**
   - Loads CSV files
   - Validates with Pydantic
   - Structures data
   - Calls AttributeAnalogyAgent for attribute extraction
   - Stores everything in Vector DB via `store_all_products_in_knowledge_base()`

2. **`store_all_products_in_knowledge_base(processed_data)`**
   - Aggregates sales/inventory/pricing data per product
   - Calls `knowledge_base.store_product_with_sales_data()`
   - Stores all products with complete data in ChromaDB

### SharedKnowledgeBase (Vector DB)

1. **`store_product_with_sales_data(product_id, attributes, description, sales_data, inventory_data, pricing_data)`**
   - Stores product with attributes (from AttributeAnalogyAgent)
   - Stores aggregated sales data
   - Stores inventory data
   - Stores pricing data
   - Generates embeddings for similarity search

2. **`get_all_products_with_performance()`**
   - Retrieves all products from Vector DB
   - Returns products with attributes, sales, inventory, pricing data
   - Used by ProcurementAllocationAgent for analysis

3. **`find_similar_products(query_attributes, query_description, top_k)`**
   - Vector similarity search
   - Returns similar products with similarity scores

### ProcurementAllocationAgent

1. **`analyze_all_products_from_kb()`**
   - Pulls all products from Vector DB
   - Analyzes patterns by category/attributes
   - Generates LLM insights
   - Caches results

2. **`evaluate_new_product(product_description, product_attributes)`**
   - Calls `analyze_all_products_from_kb()` to get analysis
   - Calls `knowledge_base.find_similar_products()` for similar products
   - Calls `knowledge_base.get_all_products_with_performance()` for sales data
   - Builds prompt with all data from Vector DB
   - Gets LLM recommendation

## Benefits

1. **Single Source of Truth**: All data stored in Vector DB
2. **Efficient Retrieval**: Vector similarity search for similar products
3. **Complete Context**: Sales, inventory, pricing all stored together
4. **Scalable**: Vector DB handles large product catalogs
5. **Persistent**: Data persists across sessions
6. **Semantic Search**: Embeddings enable semantic similarity matching

## Data Storage in Vector DB

Each product in ChromaDB contains:
- **ID**: product_id
- **Embedding**: Generated from description + attributes
- **Metadata**:
  - product_name
  - attributes (JSON)
  - description
  - sales_total_units
  - sales_total_revenue
  - sales_avg_monthly
  - inventory_total
  - inventory_by_store (JSON)
  - price_avg, price_min, price_max
  - created_at timestamp

## Example Flow

1. User loads demo scenario
2. DataIngestionAgent processes 1000 products
3. AttributeAnalogyAgent extracts attributes for all 1000 products
4. All 1000 products stored in ChromaDB with sales/inventory/pricing
5. User evaluates new product
6. ProcurementAllocationAgent:
   - Pulls all 1000 products from ChromaDB
   - Analyzes patterns
   - Finds 5 similar products via vector search
   - Gets their sales/inventory/pricing from ChromaDB
   - Creates prompt with all this data
   - Gets LLM recommendation

