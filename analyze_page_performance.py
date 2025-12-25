#!/usr/bin/env python3
"""
Standalone script to analyze page performance using Chrome DevTools
Can analyze the current page in Chrome or navigate to a new URL
"""

import asyncio
import json
import sys
import argparse
from datetime import datetime

try:
    import httpx
    import websockets
except ImportError:
    print("Error: Required packages not installed. Run: pip install httpx websockets")
    sys.exit(1)


class PerformanceAnalyzer:
    """Analyze page performance using Chrome DevTools Protocol"""
    
    def __init__(self, chrome_debug_port: int = 9222):
        self.chrome_debug_port = chrome_debug_port
        self.chrome_url = f"http://localhost:{chrome_debug_port}"
        self.ws_url = None
        self.target_id = None
        
    async def get_chrome_targets(self) -> list:
        """Get list of available Chrome targets"""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(f"{self.chrome_url}/json")
                response.raise_for_status()
                return response.json()
        except Exception as e:
            raise Exception(f"Failed to connect to Chrome: {e}. Make sure Chrome is running with --remote-debugging-port={self.chrome_debug_port}")
    
    async def connect_to_target(self, target_url: str = None) -> str:
        """Connect to a Chrome target"""
        targets = await self.get_chrome_targets()
        
        if not targets:
            raise Exception("No Chrome targets available")
        
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
    
    async def execute_javascript(self, expression: str) -> dict:
        """Execute JavaScript in the page context"""
        result = await self.send_cdp_command("Runtime.evaluate", {
            "expression": expression,
            "returnByValue": True
        })
        return result.get("result", {})
    
    async def navigate(self, url: str):
        """Navigate to a URL"""
        await self.send_cdp_command("Page.enable")
        result = await self.send_cdp_command("Page.navigate", {"url": url})
        
        # Wait for page to load
        await asyncio.sleep(2)
        
        # Wait for network idle
        await asyncio.sleep(1)
        return result
    
    async def get_core_web_vitals(self) -> dict:
        """Get Core Web Vitals"""
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
            
            // First Input Delay (FID)
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
            const paintEntries = performance.getEntriesByType('paint');
            vitals.FCP = paintEntries.find(entry => entry.name === 'first-contentful-paint')?.startTime;
            
            if (performance.timing) {
                vitals.TTFB = performance.timing.responseStart - performance.timing.requestStart;
                vitals.DOMContentLoaded = performance.timing.domContentLoadedEventEnd - performance.timing.navigationStart;
                vitals.Load = performance.timing.loadEventEnd - performance.timing.navigationStart;
            }
            
            return vitals;
        })()
        """
        result = await self.execute_javascript(vitals_code)
        return result.get("result", {}).get("value", {})
    
    async def get_network_metrics(self) -> dict:
        """Get detailed network metrics"""
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
            
            metrics.slowest.sort((a, b) => b.time - a.time).splice(5);
            metrics.largest.sort((a, b) => b.size - a.size).splice(5);
            
            return metrics;
        })()
        """
        result = await self.execute_javascript(timing_code)
        return result.get("result", {}).get("value", {})
    
    async def get_performance_metrics(self) -> dict:
        """Get performance metrics"""
        await self.send_cdp_command("Performance.enable")
        result = await self.send_cdp_command("Performance.getMetrics", {})
        metrics = result.get("result", {}).get("metrics", [])
        
        metrics_dict = {}
        for metric in metrics:
            metrics_dict[metric.get("name")] = metric.get("value")
        
        return metrics_dict
    
    async def analyze_page(self, url: str = None) -> dict:
        """Run complete performance analysis"""
        print("🔍 Starting performance analysis...")
        
        # Connect to Chrome
        await self.connect_to_target(url)
        print("✅ Connected to Chrome")
        
        # Navigate if URL provided
        if url:
            print(f"🌐 Navigating to {url}...")
            await self.navigate(url)
            await asyncio.sleep(3)  # Wait for page to fully load
            print("✅ Page loaded")
        
        # Get page info
        print("📄 Gathering page information...")
        title_result = await self.execute_javascript("document.title")
        url_result = await self.execute_javascript("window.location.href")
        
        # Get metrics
        print("📊 Collecting performance metrics...")
        web_vitals = await self.get_core_web_vitals()
        network_metrics = await self.get_network_metrics()
        performance_metrics = await self.get_performance_metrics()
        
        analysis = {
            "timestamp": datetime.now().isoformat(),
            "pageInfo": {
                "title": title_result.get("result", {}).get("value", ""),
                "url": url_result.get("result", {}).get("value", "")
            },
            "webVitals": web_vitals,
            "networkMetrics": network_metrics,
            "performanceMetrics": performance_metrics,
            "recommendations": []
        }
        
        # Generate recommendations
        print("💡 Generating recommendations...")
        analysis["recommendations"] = self._generate_recommendations(analysis)
        
        return analysis
    
    def _generate_recommendations(self, data: dict) -> list:
        """Generate performance recommendations"""
        recommendations = []
        vitals = data.get("webVitals", {})
        network = data.get("networkMetrics", {})
        
        lcp = vitals.get("LCP", 0) / 1000
        if lcp > 2.5:
            recommendations.append({
                "priority": "high",
                "metric": "LCP",
                "value": f"{lcp:.2f}s",
                "target": "<2.5s",
                "recommendation": "Optimize largest content element loading. Consider image optimization, preloading critical resources, or reducing server response time."
            })
        elif lcp > 0:
            recommendations.append({
                "priority": "info",
                "metric": "LCP",
                "value": f"{lcp:.2f}s",
                "target": "<2.5s",
                "recommendation": "LCP is within acceptable range."
            })
        
        fcp = vitals.get("FCP", 0) / 1000
        if fcp > 1.8:
            recommendations.append({
                "priority": "high",
                "metric": "FCP",
                "value": f"{fcp:.2f}s",
                "target": "<1.8s",
                "recommendation": "Reduce render-blocking resources. Minimize CSS, defer non-critical JavaScript, and optimize font loading."
            })
        
        cls = vitals.get("CLS", 0)
        if cls > 0.1:
            recommendations.append({
                "priority": "high",
                "metric": "CLS",
                "value": f"{cls:.3f}",
                "target": "<0.1",
                "recommendation": "Fix layout shifts by setting explicit dimensions on images and videos, avoid inserting content above existing content, and use transform animations instead of position changes."
            })
        
        total_size = network.get("totalSize", 0) / (1024 * 1024)  # MB
        if total_size > 5:
            recommendations.append({
                "priority": "medium",
                "metric": "Page Size",
                "value": f"{total_size:.2f}MB",
                "target": "<5MB",
                "recommendation": "Consider code splitting, lazy loading, and image optimization to reduce total page size."
            })
        
        total_requests = network.get("totalRequests", 0)
        if total_requests > 100:
            recommendations.append({
                "priority": "medium",
                "metric": "Request Count",
                "value": f"{total_requests}",
                "target": "<100",
                "recommendation": "Reduce number of HTTP requests by combining files, using sprites, or implementing resource bundling."
            })
        
        return recommendations
    
    def print_analysis(self, analysis: dict):
        """Print analysis results in a readable format"""
        print("\n" + "=" * 80)
        print("📊 PERFORMANCE ANALYSIS REPORT")
        print("=" * 80)
        
        # Page Info
        page_info = analysis.get("pageInfo", {})
        print(f"\n📄 Page: {page_info.get('title', 'N/A')}")
        print(f"🔗 URL: {page_info.get('url', 'N/A')}")
        print(f"⏰ Analyzed: {analysis.get('timestamp', 'N/A')}")
        
        # Web Vitals
        print("\n" + "-" * 80)
        print("🎯 CORE WEB VITALS")
        print("-" * 80)
        vitals = analysis.get("webVitals", {})
        
        lcp = vitals.get("LCP", 0) / 1000
        fcp = vitals.get("FCP", 0) / 1000
        cls = vitals.get("CLS", 0)
        fid = vitals.get("FID", 0)
        ttfb = vitals.get("TTFB", 0)
        
        def get_status(value, good, poor):
            if value <= good:
                return "✅ GOOD"
            elif value <= poor:
                return "⚠️  NEEDS IMPROVEMENT"
            else:
                return "❌ POOR"
        
        print(f"LCP (Largest Contentful Paint): {lcp:.2f}s {get_status(lcp, 2.5, 4.0)}")
        print(f"FCP (First Contentful Paint):   {fcp:.2f}s {get_status(fcp, 1.8, 3.0)}")
        print(f"CLS (Cumulative Layout Shift):   {cls:.3f} {get_status(cls, 0.1, 0.25)}")
        if fid > 0:
            print(f"FID (First Input Delay):        {fid:.0f}ms {get_status(fid, 100, 300)}")
        print(f"TTFB (Time to First Byte):      {ttfb:.0f}ms {get_status(ttfb, 800, 1800)}")
        
        # Network Metrics
        print("\n" + "-" * 80)
        print("🌐 NETWORK METRICS")
        print("-" * 80)
        network = analysis.get("networkMetrics", {})
        total_size = network.get("totalSize", 0) / (1024 * 1024)
        total_requests = network.get("totalRequests", 0)
        avg_time = network.get("totalTime", 0) / max(total_requests, 1) / 1000
        
        print(f"Total Requests: {total_requests}")
        print(f"Total Size: {total_size:.2f} MB")
        print(f"Average Request Time: {avg_time:.2f}s")
        
        # Resource breakdown
        by_type = network.get("byType", {})
        if by_type:
            print("\nRequests by Type:")
            for rtype, data in sorted(by_type.items(), key=lambda x: x[1]["count"], reverse=True):
                size_mb = data["size"] / (1024 * 1024)
                print(f"  {rtype:15s}: {data['count']:3d} requests, {size_mb:.2f} MB")
        
        # Slowest requests
        slowest = network.get("slowest", [])
        if slowest:
            print("\n🐌 Slowest Requests:")
            for i, req in enumerate(slowest[:5], 1):
                name = req["name"][:60]
                print(f"  {i}. {name}")
                print(f"     Time: {req['time']/1000:.2f}s, Size: {req['size']/1024:.2f} KB")
        
        # Recommendations
        print("\n" + "-" * 80)
        print("💡 RECOMMENDATIONS")
        print("-" * 80)
        recommendations = analysis.get("recommendations", [])
        if recommendations:
            for i, rec in enumerate(recommendations, 1):
                priority_icon = {"high": "🔴", "medium": "🟡", "info": "ℹ️"}.get(rec.get("priority", "info"), "ℹ️")
                print(f"\n{i}. {priority_icon} {rec.get('metric', 'N/A')}: {rec.get('value', 'N/A')} (Target: {rec.get('target', 'N/A')})")
                print(f"   {rec.get('recommendation', '')}")
        else:
            print("✅ No critical issues found!")
        
        print("\n" + "=" * 80)


async def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Analyze page performance using Chrome DevTools")
    parser.add_argument("--url", type=str, help="URL to analyze (if not provided, analyzes current page)")
    parser.add_argument("--port", type=int, default=9222, help="Chrome remote debugging port (default: 9222)")
    parser.add_argument("--json", action="store_true", help="Output results as JSON")
    parser.add_argument("--output", type=str, help="Save results to file")
    
    args = parser.parse_args()
    
    try:
        analyzer = PerformanceAnalyzer(chrome_debug_port=args.port)
        analysis = await analyzer.analyze_page(url=args.url)
        
        if args.json:
            output = json.dumps(analysis, indent=2)
            print(output)
        else:
            analyzer.print_analysis(analysis)
        
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(analysis, f, indent=2)
            print(f"\n✅ Results saved to {args.output}")
    
    except Exception as e:
        print(f"❌ Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())

