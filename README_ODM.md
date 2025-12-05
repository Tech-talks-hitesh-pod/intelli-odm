# 🏪 ODM Intelligence Platform

A simplified AI-powered system for Own Design Manufacturing (ODM) product analysis and sales prediction.

## 🎯 Purpose

This system solves the use case: **"If a red t-shirt is selling well at a store, and someone introduces a red shirt with similar attributes, the model should fetch the correlation and predict sales based on historical data."**

## 🚀 Quick Start

1. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Configure Environment:**
   - Copy `.env.example` to `.env` 
   - Add your OpenAI API key
   ```bash
   OPENAI_API_KEY=your_openai_api_key_here
   DEMO_MODE=false
   LLM_PROVIDER=openai
   ```

3. **Run the Application:**
   ```bash
   python -m streamlit run odm_app.py --server.port=8502
   ```

4. **Access the App:**
   - Open http://localhost:8502 in your browser

## 📊 Features

### 1. **Data Summary Dashboard**
- Automatically loads ODM data on startup
- Shows summary of historical products and new input products
- Displays key metrics and visualizations
- Real-time vector database status

### 2. **Product Search**
- Search through all indexed products in the vector database
- Semantic similarity search (e.g., "red shirt", "cotton top")
- Shows similarity scores and sales data for each match

### 3. **Sales Prediction**
- Enter description of a new product
- AI finds similar historical products based on attributes
- Predicts monthly sales with confidence levels
- Detailed AI analysis explaining the reasoning

## 🗃️ Data Files

The system uses two main data files:

1. **`dirty_odm_input.csv`** - New products to evaluate (15 products)
2. **`odm_historical_dataset_5000.csv`** - Historical sales data (5000+ records)

## 🔍 How It Works

1. **Data Loading:**
   - System loads both dirty input and historical ODM data
   - Creates embeddings using sentence transformers
   - Stores products with sales data in ChromaDB vector database

2. **Similarity Search:**
   - Uses vector similarity to find products with similar attributes
   - Considers color, material, style, fit, and product type
   - Returns ranked results with similarity scores

3. **Sales Prediction:**
   - Finds top 5 most similar historical products
   - Calculates weighted average sales based on similarity scores
   - Uses OpenAI GPT to provide detailed analysis and reasoning
   - Provides confidence level based on match quality

## 📋 Use Case Example

**Scenario:** You want to predict sales for "Red cotton t-shirt with short sleeves for men"

**Process:**
1. System searches vector database for similar products
2. Finds historical products like:
   - "Red Cotton Blouse Style Top" (similarity: 0.82)
   - "Cotton Modal T-shirt" (similarity: 0.75)
   - Other similar red/cotton products
3. Calculates predicted sales: ~8-12 units/month
4. Provides AI reasoning based on historical performance

## 🔧 Technical Details

### Architecture
- **Frontend:** Streamlit web interface
- **Vector DB:** ChromaDB for similarity search
- **Embeddings:** SentenceTransformers (all-MiniLM-L6-v2)
- **AI Analysis:** OpenAI GPT-4o-mini
- **Data Processing:** Pandas for ODM data cleaning

### Key Components
- `odm_app.py` - Main Streamlit application
- `shared_knowledge_base.py` - Vector database operations
- `agents/data_ingestion_agent.py` - ODM data processing
- `utils/llm_client.py` - OpenAI integration

### Logging
- Comprehensive logging for embedding creation and storage
- Vector database operations tracking
- Search and prediction process monitoring
- Log files: `odm_app.log`

## 🐛 Troubleshooting

### Common Issues

1. **"No similar products found"**
   - Check if data was loaded properly
   - Verify vector database contains products
   - Try different search terms

2. **Embedding creation fails**
   - Check internet connection (downloads model on first run)
   - Verify sufficient disk space for sentence transformer model

3. **OpenAI API errors**
   - Verify API key is correct and active
   - Check API usage limits
   - Ensure DEMO_MODE=false in .env

### Debug Steps
1. Check logs in `odm_app.log`
2. Verify data files exist in `data/sample/`
3. Test vector database: Go to "Search Products" page
4. Check ChromaDB storage: `data/chroma_db/`

## 📈 Performance

- **Data Loading:** ~30-60 seconds for 5000+ products
- **Search Response:** <2 seconds for similarity search
- **Prediction:** ~5-10 seconds including AI analysis
- **Storage:** ~50MB for embeddings + metadata

## 🔄 Data Updates

To update the system with new ODM data:
1. Replace `dirty_odm_input.csv` with new products
2. Update `odm_historical_dataset_5000.csv` with new sales data
3. Restart the application (auto-reloads data on startup)
4. Vector database updates automatically

## 🎯 Business Value

- **Faster Decision Making:** Get sales predictions in seconds
- **Data-Driven Insights:** Base decisions on historical performance
- **Risk Reduction:** Understand market potential before production
- **Attribute Analysis:** See which product features drive sales