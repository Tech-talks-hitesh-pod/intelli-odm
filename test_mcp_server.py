#!/usr/bin/env python3
"""
Test script for the Chrome DevTools MCP Server
Tests basic connectivity and functionality
"""

import asyncio
import sys
import json

try:
    import httpx
    import websockets
except ImportError:
    print("Error: Required packages not installed. Run: pip install httpx websockets")
    sys.exit(1)


async def test_chrome_connection(port=9222):
    """Test connection to Chrome remote debugging"""
    chrome_url = f"http://localhost:{port}"
    
    print(f"Testing connection to Chrome on port {port}...")
    
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(f"{chrome_url}/json")
            response.raise_for_status()
            targets = response.json()
            
            print(f"✅ Successfully connected to Chrome!")
            print(f"Found {len(targets)} target(s):")
            
            for i, target in enumerate(targets[:3], 1):  # Show first 3
                print(f"  {i}. {target.get('title', 'Untitled')} - {target.get('url', 'N/A')}")
            
            if len(targets) > 3:
                print(f"  ... and {len(targets) - 3} more")
            
            # Test WebSocket connection
            if targets:
                ws_url = targets[0].get("webSocketDebuggerUrl")
                if ws_url:
                    print(f"\nTesting WebSocket connection...")
                    try:
                        async with websockets.connect(ws_url, timeout=5) as ws:
                            print("✅ WebSocket connection successful!")
                            return True
                    except Exception as e:
                        print(f"❌ WebSocket connection failed: {e}")
                        return False
                else:
                    print("❌ No WebSocket URL found")
                    return False
            
            return True
            
    except httpx.ConnectError:
        print(f"❌ Failed to connect to Chrome on port {port}")
        print(f"\nMake sure Chrome is running with remote debugging enabled:")
        print(f"  chrome --remote-debugging-port={port}")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


async def test_mcp_imports():
    """Test if MCP packages are installed"""
    print("Testing MCP package imports...")
    
    try:
        from mcp.server import Server
        from mcp.server.stdio import stdio_server
        from mcp.types import Tool, TextContent
        print("✅ MCP packages imported successfully!")
        return True
    except ImportError as e:
        print(f"❌ MCP packages not installed: {e}")
        print("\nInstall with: pip install mcp")
        print("Or if using the official SDK: pip install @modelcontextprotocol/sdk-python")
        return False


async def main():
    """Run all tests"""
    print("=" * 60)
    print("Chrome DevTools MCP Server - Test Suite")
    print("=" * 60)
    print()
    
    # Test MCP imports
    mcp_ok = await test_mcp_imports()
    print()
    
    # Test Chrome connection
    chrome_ok = await test_chrome_connection()
    print()
    
    # Summary
    print("=" * 60)
    if mcp_ok and chrome_ok:
        print("✅ All tests passed! MCP server should work correctly.")
    else:
        print("❌ Some tests failed. Please fix the issues above.")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())

