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
    
    async def get_performance_metrics(self) -> dict:
        """Get performance metrics from the page"""
        # Enable Performance domain
        await self.send_cdp_command("Performance.enable")
        
        # Get metrics
        result = await self.send_cdp_command("Performance.getMetrics", {})
        metrics = result.get("result", {}).get("metrics", [])
        
        # Convert to dict
        metrics_dict = {}
        for metric in metrics:
            metrics_dict[metric.get("name")] = metric.get("value")
        
        return metrics_dict
    
    async def get_core_web_vitals(self) -> dict:
        """Get Core Web Vitals (LCP, FID, CLS)"""
        # Execute JavaScript to get Web Vitals
        vitals_code = """
        (async () => {
            const vitals = {};
            
            // Largest Contentful Paint (LCP)
            try {
                const lcpEntries = performance.getEntriesByType('largest-contentful-paint');
                if (lcpEntries.length > 0) {
                    vitals.LCP = lcpEntries[lcpEntries.length - 1].renderTime || lcpEntries[lcpEntries.length - 1].loadTime;
                }
            } catch(e) {}
            
            // First Input Delay (FID) - requires user interaction, so we'll get it if available
            try {
                const fidEntries = performance.getEntriesByType('first-input');
                if (fidEntries.length > 0) {
                    vitals.FID = fidEntries[0].processingStart - fidEntries[0].startTime;
                }
            } catch(e) {}
            
            // Cumulative Layout Shift (CLS)
            try {
                let clsValue = 0;
                const clsEntries = performance.getEntriesByType('layout-shift');
                clsEntries.forEach(entry => {
                    if (!entry.hadRecentInput) {
                        clsValue += entry.value;
                    }
                });
                vitals.CLS = clsValue;
            } catch(e) {}
            
            // Additional metrics
            vitals.FCP = performance.getEntriesByType('paint').find(entry => entry.name === 'first-contentful-paint')?.startTime;
            vitals.TTFB = performance.timing.responseStart - performance.timing.requestStart;
            vitals.DOMContentLoaded = performance.timing.domContentLoadedEventEnd - performance.timing.navigationStart;
            vitals.Load = performance.timing.loadEventEnd - performance.timing.navigationStart;
            
            return vitals;
        })()
        """
        result = await self.execute_javascript(vitals_code)
        return result.get("result", {}).get("value", {})
    
    async def get_network_metrics(self) -> dict:
        """Get detailed network metrics"""
        # Enable Network domain
        await self.send_cdp_command("Network.enable")
        
        # Get resource timing
        timing_code = """
        (() => {
            const resources = performance.getEntriesByType('resource');
            const metrics = {
                totalRequests: resources.length,
                totalSize: 0,
                totalTime: 0,
                byType: {},
                slowest: [],
                largest: []
            };
            
            resources.forEach(resource => {
                const size = resource.transferSize || 0;
                const time = resource.responseEnd - resource.startTime;
                
                metrics.totalSize += size;
                metrics.totalTime += time;
                
                const type = resource.initiatorType || 'other';
                if (!metrics.byType[type]) {
                    metrics.byType[type] = { count: 0, size: 0, time: 0 };
                }
                metrics.byType[type].count++;
                metrics.byType[type].size += size;
                metrics.byType[type].time += time;
                
                metrics.slowest.push({
                    name: resource.name,
                    time: time,
                    size: size
                });
                
                metrics.largest.push({
                    name: resource.name,
                    size: size,
                    time: time
                });
            });
            
            // Sort and get top 5
            metrics.slowest.sort((a, b) => b.time - a.time).splice(5);
            metrics.largest.sort((a, b) => b.size - a.size).splice(5);
            
            return metrics;
        })()
        """
        result = await self.execute_javascript(timing_code)
        return result.get("result", {}).get("value", {})
    
    async def run_lighthouse_audit(self, categories: list = None) -> dict:
        """Run Lighthouse audit using CDP (simplified version)"""
        if categories is None:
            categories = ["performance", "accessibility", "best-practices", "seo"]
        
        # Get comprehensive performance data
        performance_data = {
            "metrics": await self.get_performance_metrics(),
            "webVitals": await self.get_core_web_vitals(),
            "network": await self.get_network_metrics(),
            "pageInfo": {}
        }
        
        # Get page info
        title_result = await self.execute_javascript("document.title")
        url_result = await self.execute_javascript("window.location.href")
        performance_data["pageInfo"] = {
            "title": title_result.get("result", {}).get("value", ""),
            "url": url_result.get("result", {}).get("value", "")
        }
        
        # Calculate scores (simplified)
        scores = {}
        vitals = performance_data["webVitals"]
        
        # Performance score (simplified calculation)
        lcp = vitals.get("LCP", 0) / 1000  # Convert to seconds
        fcp = vitals.get("FCP", 0) / 1000
        cls = vitals.get("CLS", 0)
        
        perf_score = 100
        if lcp > 4: perf_score -= 25
        elif lcp > 2.5: perf_score -= 15
        if fcp > 3: perf_score -= 20
        elif fcp > 1.8: perf_score -= 10
        if cls > 0.25: perf_score -= 25
        elif cls > 0.1: perf_score -= 15
        
        scores["performance"] = max(0, perf_score)
        
        return {
            "scores": scores,
            "metrics": performance_data,
            "recommendations": self._generate_recommendations(performance_data)
        }
    
    def _generate_recommendations(self, data: dict) -> list:
        """Generate performance recommendations"""
        recommendations = []
        vitals = data.get("webVitals", {})
        network = data.get("network", {})
        
        lcp = vitals.get("LCP", 0) / 1000
        if lcp > 2.5:
            recommendations.append(f"LCP is {lcp:.2f}s (target: <2.5s). Optimize largest content element loading.")
        
        fcp = vitals.get("FCP", 0) / 1000
        if fcp > 1.8:
            recommendations.append(f"FCP is {fcp:.2f}s (target: <1.8s). Reduce render-blocking resources.")
        
        cls = vitals.get("CLS", 0)
        if cls > 0.1:
            recommendations.append(f"CLS is {cls:.3f} (target: <0.1). Fix layout shifts by setting dimensions on images/videos.")
        
        total_size = network.get("totalSize", 0) / (1024 * 1024)  # MB
        if total_size > 5:
            recommendations.append(f"Total page size is {total_size:.2f}MB. Consider code splitting and lazy loading.")
        
        return recommendations


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
        ),
        Tool(
            name="get_performance_metrics",
            description="Get performance metrics from the page (timing, memory, etc.)",
            inputSchema={
                "type": "object",
                "properties": {}
            }
        ),
        Tool(
            name="get_core_web_vitals",
            description="Get Core Web Vitals (LCP, FID, CLS, FCP, TTFB) and page load metrics",
            inputSchema={
                "type": "object",
                "properties": {}
            }
        ),
        Tool(
            name="get_network_metrics",
            description="Get detailed network metrics including request counts, sizes, and timing",
            inputSchema={
                "type": "object",
                "properties": {}
            }
        ),
        Tool(
            name="run_performance_audit",
            description="Run a comprehensive performance audit including Lighthouse-style analysis",
            inputSchema={
                "type": "object",
                "properties": {
                    "categories": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Categories to audit (performance, accessibility, best-practices, seo)",
                        "default": ["performance"]
                    }
                }
            }
        ),
        Tool(
            name="analyze_page_performance",
            description="Complete performance analysis of the current page with recommendations",
            inputSchema={
                "type": "object",
                "properties": {
                    "include_screenshot": {
                        "type": "boolean",
                        "description": "Include screenshot in analysis",
                        "default": False
                    }
                }
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
        
        elif name == "get_performance_metrics":
            metrics = await chrome_client.get_performance_metrics()
            return [TextContent(
                type="text",
                text=json.dumps(metrics, indent=2)
            )]
        
        elif name == "get_core_web_vitals":
            vitals = await chrome_client.get_core_web_vitals()
            return [TextContent(
                type="text",
                text=json.dumps(vitals, indent=2)
            )]
        
        elif name == "get_network_metrics":
            metrics = await chrome_client.get_network_metrics()
            return [TextContent(
                type="text",
                text=json.dumps(metrics, indent=2)
            )]
        
        elif name == "run_performance_audit":
            categories = arguments.get("categories", ["performance"])
            audit_result = await chrome_client.run_lighthouse_audit(categories)
            return [TextContent(
                type="text",
                text=json.dumps(audit_result, indent=2)
            )]
        
        elif name == "analyze_page_performance":
            include_screenshot = arguments.get("include_screenshot", False)
            
            # Get comprehensive analysis
            analysis = {
                "pageInfo": {},
                "webVitals": {},
                "networkMetrics": {},
                "performanceMetrics": {},
                "audit": {},
                "screenshot": None
            }
            
            # Get page info
            title_result = await chrome_client.execute_javascript("document.title")
            url_result = await chrome_client.execute_javascript("window.location.href")
            analysis["pageInfo"] = {
                "title": title_result.get("result", {}).get("value", ""),
                "url": url_result.get("result", {}).get("value", "")
            }
            
            # Get all metrics
            analysis["webVitals"] = await chrome_client.get_core_web_vitals()
            analysis["networkMetrics"] = await chrome_client.get_network_metrics()
            analysis["performanceMetrics"] = await chrome_client.get_performance_metrics()
            analysis["audit"] = await chrome_client.run_lighthouse_audit()
            
            if include_screenshot:
                screenshot_data = await chrome_client.take_screenshot()
                analysis["screenshot"] = f"data:image/png;base64,{screenshot_data}"
            
            return [TextContent(
                type="text",
                text=json.dumps(analysis, indent=2)
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

