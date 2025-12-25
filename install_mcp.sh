#!/bin/bash
# Installation script for Chrome DevTools MCP Server

echo "Installing Chrome DevTools MCP Server dependencies..."

# Install Python dependencies
pip install websockets httpx

# Try to install MCP package
echo ""
echo "Attempting to install MCP package..."
echo "Note: The MCP Python SDK package name may vary."

# Try different installation methods
if pip install mcp 2>/dev/null; then
    echo "✅ Successfully installed 'mcp' package"
elif pip install @modelcontextprotocol/sdk-python 2>/dev/null; then
    echo "✅ Successfully installed '@modelcontextprotocol/sdk-python' package"
else
    echo "⚠️  Could not automatically install MCP package"
    echo ""
    echo "Please install manually:"
    echo "  1. Check https://github.com/modelcontextprotocol/python-sdk"
    echo "  2. Or try: pip install mcp"
    echo "  3. Or clone and install from source"
fi

echo ""
echo "Installation complete!"
echo ""
echo "Next steps:"
echo "1. Start Chrome with: chrome --remote-debugging-port=9222"
echo "2. Test connection: python test_mcp_server.py"
echo "3. Configure your MCP client (Claude Desktop, Cursor, etc.)"

