#!/bin/bash

# Activate virtual environment
source venv/bin/activate

# Set environment variables
export PYTHONPATH=$PWD:$PYTHONPATH

echo "🎯 Starting Intelli-ODM CEO Demo..."
echo "=================================="
echo
echo "🌐 Demo will open in your browser at: http://localhost:8501"
echo "📋 Use the sidebar to select different business scenarios"
echo "🚀 Click 'Run AI Analysis' to see results"
echo
echo "Press Ctrl+C to stop the demo"
echo

# Start Streamlit
streamlit run streamlit_app.py --server.port=8501 --server.address=0.0.0.0 --theme.base=light