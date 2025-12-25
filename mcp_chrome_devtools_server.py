#!/usr/bin/env python3
"""
MCP Server for Chrome DevTools Protocol
Connects to Chrome browser via CDP and provides tools for interacting with DevTools
"""

import asyncio
import json
import sys
from typing import Any, Optional
from urllib.parse import urlparse

# Try different MCP package import paths
try:
    from mcp.server import Server
    from mcp.server.stdio import stdio_server
    from mcp.types import Tool, TextContent
except ImportError:
    try:
        # Alternative import path (if package structure differs)
        from mcp import Server
        from mcp.stdio import stdio_server
        from mcp import Tool, TextContent
    except ImportError:
        print("Error: MCP package not installed.")
        print("Install with one of:")
        print("  pip install mcp")
        print("  pip install @modelcontextprotocol/sdk-python")
        print("  npm install -g @modelcontextprotocol/server-python")
        sys.exit(1)

try:
    import websockets
    import httpx
except ImportError:
    print("Error: websockets and httpx packages not installed. Run: pip install websockets httpx")
    sys.exit(1)


class ChromeDevToolsMCP:
    """MCP Server for Chrome DevTools Protocol"""
    
    def __init__(self, chrome_debug_port: int = 9222):
        self.chrome_debug_port = chrome_debug_port
        self.chrome_url = f"http://localhost:{chrome_debug_port}"
        self.ws_url: Optional[str] = None
        self.ws_connection = None
        self.target_id: Optional[str] = None
        
    async def get_chrome_targets(self) -> list:
        """Get list of available Chrome targets"""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(f"{self.chrome_url}/json")
                response.raise_for_status()
                return response.json()
        except Exception as e:
            raise Exception(f"Failed to connect to Chrome: {e}. Make sure Chrome is running with --remote-debugging-port={self.chrome_debug_port}")
    
    async def connect_to_target(self, target_url: Optional[str] = None) -> str:
        """Connect to a Chrome target and return WebSocket URL"""
        targets = await self.get_chrome_targets()
        
        if not targets:
            raise Exception("No Chrome targets available")
        
        # Use first available target or find by URL
        target = None
        if target_url:
            for t in targets:
                if target_url in t.get("url", ""):
                    target = t
                    break
        
        if not target:
            target = targets[0]
        
        self.target_id = target.get("id")
        self.ws_url = target.get("webSocketDebuggerUrl")
        
        if not self.ws_url:
            raise Exception("No WebSocket URL found for target")
        
        return self.ws_url
    
    async def send_cdp_command(self, method: str, params: dict = None) -> dict:
        """Send a CDP command to Chrome"""
        if not self.ws_url:
            await self.connect_to_target()
        
        if params is None:
            params = {}
        
        command = {
            "id": 1,
            "method": method,
            "params": params
        }
        
        try:
            async with websockets.connect(self.ws_url) as ws:
                await ws.send(json.dumps(command))
                response = await ws.recv()
                return json.loads(response)
        except Exception as e:
            raise Exception(f"CDP command failed: {e}")
    
    async def get_dom_tree(self, node_id: int = None) -> dict:
        """Get DOM tree from the page"""
        if node_id is None:
            # Get document node
            result = await self.send_cdp_command("DOM.getDocument", {"depth": -1})
            node_id = result.get("result", {}).get("root", {}).get("nodeId")
        
        result = await self.send_cdp_command("DOM.describeNode", {"nodeId": node_id})
        return result.get("result", {})
    
    async def query_selector(self, selector: str) -> dict:
        """Query DOM element by selector"""
        # First get document
        doc_result = await self.send_cdp_command("DOM.getDocument", {"depth": -1})
        root_node_id = doc_result.get("result", {}).get("root", {}).get("nodeId")
        
        # Query selector
        result = await self.send_cdp_command("DOM.querySelector", {
            "nodeId": root_node_id,
            "selector": selector
        })
        return result.get("result", {})
    
    async def get_console_messages(self) -> list:
        """Get console messages"""
        # Enable console domain
        await self.send_cdp_command("Runtime.enable")
        await self.send_cdp_command("Console.enable")
        
        # Note: This is a simplified version. In production, you'd want to
        # maintain a WebSocket connection to receive console messages in real-time
        return []
    
    async def execute_javascript(self, expression: str) -> dict:
        """Execute JavaScript in the page context"""
        result = await self.send_cdp_command("Runtime.evaluate", {
            "expression": expression,
            "returnByValue": True
        })
        return result.get("result", {})
    
    async def get_network_requests(self) -> list:
        """Get network requests"""
        # Enable network domain
        await self.send_cdp_command("Network.enable")
        
        # Get all requests (simplified - in production you'd track them)
        return []
    
    async def take_screenshot(self) -> str:
        """Take a screenshot of the current page"""
        result = await self.send_cdp_command("Page.captureScreenshot", {
            "format": "png"
        })
        return result.get("result", {}).get("data", "")


# Initialize the Chrome DevTools client
chrome_client = ChromeDevToolsMCP()


# MCP Server setup
server = Server("chrome-devtools-mcp")


@server.list_tools()
async def list_tools() -> list[Tool]:
    """List available tools"""
    return [
        Tool(
            name="connect_chrome",
            description="Connect to Chrome browser running with remote debugging enabled",
            inputSchema={
                "type": "object",
                "properties": {
                    "port": {
                        "type": "number",
                        "description": "Chrome remote debugging port (default: 9222)",
                        "default": 9222
                    },
                    "target_url": {
                        "type": "string",
                        "description": "Optional: URL of the target page to connect to"
                    }
                }
            }
        ),
        Tool(
            name="inspect_element",
            description="Inspect a DOM element by CSS selector",
            inputSchema={
                "type": "object",
                "properties": {
                    "selector": {
                        "type": "string",
                        "description": "CSS selector for the element to inspect"
                    }
                },
                "required": ["selector"]
            }
        ),
        Tool(
            name="get_dom_tree",
            description="Get the DOM tree structure of the current page",
            inputSchema={
                "type": "object",
                "properties": {
                    "depth": {
                        "type": "number",
                        "description": "Depth of the DOM tree to retrieve (-1 for full tree)",
                        "default": 3
                    }
                }
            }
        ),
        Tool(
            name="execute_javascript",
            description="Execute JavaScript code in the page context",
            inputSchema={
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "JavaScript code to execute"
                    }
                },
                "required": ["code"]
            }
        ),
        Tool(
            name="get_console_logs",
            description="Get console messages from the page",
            inputSchema={
                "type": "object",
                "properties": {}
            }
        ),
        Tool(
            name="get_network_requests",
            description="Get network requests made by the page",
            inputSchema={
                "type": "object",
                "properties": {}
            }
        ),
        Tool(
            name="take_screenshot",
            description="Take a screenshot of the current page",
            inputSchema={
                "type": "object",
                "properties": {}
            }
        ),
        Tool(
            name="navigate",
            description="Navigate to a URL",
            inputSchema={
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "URL to navigate to"
                    }
                },
                "required": ["url"]
            }
        ),
        Tool(
            name="get_page_info",
            description="Get information about the current page (title, URL, etc.)",
            inputSchema={
                "type": "object",
                "properties": {}
            }
        )
    ]


@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    """Handle tool calls"""
    try:
        if name == "connect_chrome":
            port = arguments.get("port", 9222)
            target_url = arguments.get("target_url")
            chrome_client.chrome_debug_port = port
            chrome_client.chrome_url = f"http://localhost:{port}"
            ws_url = await chrome_client.connect_to_target(target_url)
            return [TextContent(
                type="text",
                text=f"Connected to Chrome on port {port}. WebSocket URL: {ws_url}"
            )]
        
        elif name == "inspect_element":
            selector = arguments.get("selector")
            result = await chrome_client.query_selector(selector)
            return [TextContent(
                type="text",
                text=json.dumps(result, indent=2)
            )]
        
        elif name == "get_dom_tree":
            depth = arguments.get("depth", 3)
            result = await chrome_client.send_cdp_command("DOM.getDocument", {"depth": depth})
            return [TextContent(
                type="text",
                text=json.dumps(result.get("result", {}), indent=2)
            )]
        
        elif name == "execute_javascript":
            code = arguments.get("code")
            result = await chrome_client.execute_javascript(code)
            return [TextContent(
                type="text",
                text=json.dumps(result, indent=2)
            )]
        
        elif name == "get_console_logs":
            messages = await chrome_client.get_console_messages()
            return [TextContent(
                type="text",
                text=json.dumps({"messages": messages}, indent=2)
            )]
        
        elif name == "get_network_requests":
            requests = await chrome_client.get_network_requests()
            return [TextContent(
                type="text",
                text=json.dumps({"requests": requests}, indent=2)
            )]
        
        elif name == "take_screenshot":
            screenshot_data = await chrome_client.take_screenshot()
            return [TextContent(
                type="text",
                text=f"Screenshot captured (base64 data, {len(screenshot_data)} bytes)"
            )]
        
        elif name == "navigate":
            url = arguments.get("url")
            result = await chrome_client.send_cdp_command("Page.navigate", {"url": url})
            return [TextContent(
                type="text",
                text=f"Navigated to {url}: {json.dumps(result, indent=2)}"
            )]
        
        elif name == "get_page_info":
            # Get page title and URL
            title_result = await chrome_client.execute_javascript("document.title")
            url_result = await chrome_client.execute_javascript("window.location.href")
            return [TextContent(
                type="text",
                text=json.dumps({
                    "title": title_result.get("result", {}).get("value", ""),
                    "url": url_result.get("result", {}).get("value", "")
                }, indent=2)
            )]
        
        else:
            return [TextContent(
                type="text",
                text=f"Unknown tool: {name}"
            )]
    
    except Exception as e:
        return [TextContent(
            type="text",
            text=f"Error: {str(e)}"
        )]


async def main():
    """Main entry point"""
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options()
        )


if __name__ == "__main__":
    asyncio.run(main())

