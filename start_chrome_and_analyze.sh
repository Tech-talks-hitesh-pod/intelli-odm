#!/bin/bash
# Script to start Chrome with remote debugging and analyze Swadesh pages

echo "🚀 Starting Chrome with remote debugging..."
echo ""

# Check if Chrome is already running on port 9222
if curl -s http://localhost:9222/json > /dev/null 2>&1; then
    echo "✅ Chrome is already running with remote debugging on port 9222"
else
    echo "📱 Starting Chrome..."
    
    # Detect OS and start Chrome accordingly
    if [[ "$OSTYPE" == "darwin"* ]]; then
        # macOS
        /Applications/Google\ Chrome.app/Contents/MacOS/Google\ Chrome --remote-debugging-port=9222 --user-data-dir=/tmp/chrome-debug-profile > /dev/null 2>&1 &
        CHROME_PID=$!
    elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
        # Linux
        google-chrome --remote-debugging-port=9222 --user-data-dir=/tmp/chrome-debug-profile > /dev/null 2>&1 &
        CHROME_PID=$!
    else
        echo "❌ Unsupported OS. Please start Chrome manually with:"
        echo "   chrome --remote-debugging-port=9222"
        exit 1
    fi
    
    echo "⏳ Waiting for Chrome to start..."
    sleep 3
    
    # Verify Chrome started
    if curl -s http://localhost:9222/json > /dev/null 2>&1; then
        echo "✅ Chrome started successfully (PID: $CHROME_PID)"
        echo "   To stop Chrome, run: kill $CHROME_PID"
    else
        echo "❌ Failed to start Chrome. Please start manually:"
        echo "   chrome --remote-debugging-port=9222"
        exit 1
    fi
fi

echo ""
echo "🔍 Starting performance analysis..."
echo ""

# Run the analysis
python3 analyze_swadesh_pages.py

echo ""
echo "✅ Analysis complete!"


