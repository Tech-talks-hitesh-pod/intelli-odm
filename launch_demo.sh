#!/bin/bash

# Activate virtual environment
source venv/bin/activate

# Set environment variables
export PYTHONPATH=$PWD:$PYTHONPATH

echo "🏪 Starting ODM Intelligence Platform..."
echo "========================================"
echo
echo "🌐 Platform will open in your browser at: http://localhost:8502"
echo "📊 Features:"
echo "  • 📈 Data Summary Dashboard"
echo "  • 🔍 Product Search (Vector Database)"
echo "  • 🔮 AI-Powered Sales Prediction"
echo "  • ⚠️  Smart Procurement Recommendations"
echo
echo "🎯 Test Cases:"
echo "  • Search: 'red cotton shirt' - Find similar products"
echo "  • Predict: 'pink jeans' - See CAUTIOUS recommendation"
echo "  • Predict: 'floral dress' - Get sales forecast"
echo
echo "Press Ctrl+C to stop the platform"
echo

# Start ODM Intelligence Platform
streamlit run odm_app.py --server.port=8502 --server.address=0.0.0.0 --theme.base=light
