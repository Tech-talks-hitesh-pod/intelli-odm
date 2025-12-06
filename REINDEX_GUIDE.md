# Vector Database Reindexing Guide

## Overview
This guide explains how to reindex all products and sales data in the ChromaDB vector database.

## Why Reindex?

You may need to reindex when:
- Data files have been updated
- Vector database is corrupted or inconsistent
- You want to refresh embeddings with a new model
- Products are missing from search results

## Method 1: Using Streamlit UI (Recommended)

1. **Start the Streamlit app:**
   ```bash
   streamlit run odm_app.py
   ```

2. **Navigate to Reindex page:**
   - In the sidebar, select "🔄 Reindex Database"

3. **Click "Reindex Database" button:**
   - This will clear all existing data
   - Reload products from CSV files
   - Re-index everything with fresh embeddings

4. **Wait for completion:**
   - Progress will be shown in the UI
   - You'll see success message with updated counts

## Method 2: Using Command Line Script

1. **Run the reindex script:**
   ```bash
   python3 reindex_vector_db.py
   ```

2. **Check the output:**
   - The script will show progress
   - Logs are saved to `reindex.log`
   - Final counts will be displayed

## What Gets Reindexed?

### Data Sources:
1. **`data/sample/dirty_odm_input.csv`**
   - New products to evaluate (15 products)
   - Stored with prefix `ODM_INPUT_`

2. **`data/sample/odm_historical_dataset_5000.csv`**
   - Historical sales data (5000+ records)
   - Grouped by StyleCode
   - Stored with prefix `HIST_`
   - Includes aggregated sales data

### What's Stored:
- Product descriptions
- Attributes (color, category, fabric, etc.)
- Sales data (total units, revenue, monthly averages)
- Vector embeddings for similarity search

## Verification

After reindexing, verify by:

1. **Check Data Summary page:**
   - Should show correct product counts
   - Vector DB status should be "healthy"

2. **Test Search:**
   - Try searching for products
   - Should return relevant results

3. **Test Prediction:**
   - Make a sales prediction
   - Should find similar products

## Troubleshooting

### Issue: "No products found"
- **Solution**: Check that CSV files exist in `data/sample/` directory
- Verify file names match exactly

### Issue: "Failed to store product"
- **Solution**: Check logs for specific error
- Ensure ChromaDB directory is writable
- Check disk space

### Issue: "Embedding model failed"
- **Solution**: Ensure sentence-transformers is installed
- Model downloads automatically on first use

## Manual Database Reset

If you need to manually reset the database:

```python
from shared_knowledge_base import SharedKnowledgeBase

kb = SharedKnowledgeBase()
kb.reset_collections()
```

Then reload data using the Streamlit app or reindex script.

## Data Files Location

Default data files:
- `data/sample/dirty_odm_input.csv`
- `data/sample/odm_historical_dataset_5000.csv`

To use different files, modify:
- `odm_app.py` → `load_odm_data()` function
- `reindex_vector_db.py` → file paths

## Notes

- Reindexing can take 2-5 minutes depending on data size
- All existing data is permanently deleted
- Make sure CSV files are up-to-date before reindexing
- The vector database is stored at: `data/chromadb/` (configurable in settings)

