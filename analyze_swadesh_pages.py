#!/usr/bin/env python3
"""
Batch performance analysis for Swadesh Online pages
Analyzes multiple URLs and generates a comprehensive report
"""

import asyncio
import json
import sys
from datetime import datetime
from analyze_page_performance import PerformanceAnalyzer


async def analyze_multiple_pages(urls: list, port: int = 9222):
    """Analyze multiple pages and generate a comprehensive report"""
    
    print("=" * 80)
    print("SWADESH ONLINE - PERFORMANCE ANALYSIS")
    print("=" * 80)
    print(f"Analyzing {len(urls)} pages...")
    print()
    
    analyzer = PerformanceAnalyzer(chrome_debug_port=port)
    results = []
    
    for i, url in enumerate(urls, 1):
        print(f"\n{'='*80}")
        print(f"Page {i}/{len(urls)}: {url}")
        print(f"{'='*80}\n")
        
        try:
            analysis = await analyzer.analyze_page(url=url)
            results.append({
                "url": url,
                "analysis": analysis
            })
            
            # Print summary
            vitals = analysis.get("webVitals", {})
            lcp = vitals.get("LCP", 0) / 1000
            fcp = vitals.get("FCP", 0) / 1000
            cls = vitals.get("CLS", 0)
            
            print(f"\n📊 Quick Summary:")
            print(f"   LCP: {lcp:.2f}s")
            print(f"   FCP: {fcp:.2f}s")
            print(f"   CLS: {cls:.3f}")
            
        except Exception as e:
            print(f"❌ Error analyzing {url}: {e}")
            results.append({
                "url": url,
                "error": str(e)
            })
        
        # Wait between pages
        if i < len(urls):
            print("\n⏳ Waiting 3 seconds before next page...")
            await asyncio.sleep(3)
    
    # Generate comprehensive report
    print("\n\n" + "=" * 80)
    print("COMPREHENSIVE PERFORMANCE REPORT")
    print("=" * 80)
    
    for result in results:
        if "error" in result:
            print(f"\n❌ {result['url']}: {result['error']}")
            continue
        
        url = result["url"]
        analysis = result["analysis"]
        vitals = analysis.get("webVitals", {})
        network = analysis.get("networkMetrics", {})
        
        print(f"\n{'─'*80}")
        print(f"📄 {url}")
        print(f"{'─'*80}")
        
        # Core Web Vitals
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
        
        print(f"\n🎯 Core Web Vitals:")
        print(f"   LCP: {lcp:.2f}s {get_status(lcp, 2.5, 4.0)}")
        print(f"   FCP: {fcp:.2f}s {get_status(fcp, 1.8, 3.0)}")
        print(f"   CLS: {cls:.3f} {get_status(cls, 0.1, 0.25)}")
        if fid > 0:
            print(f"   FID: {fid:.0f}ms {get_status(fid, 100, 300)}")
        print(f"   TTFB: {ttfb:.0f}ms {get_status(ttfb, 800, 1800)}")
        
        # Network Metrics
        total_size = network.get("totalSize", 0) / (1024 * 1024)
        total_requests = network.get("totalRequests", 0)
        
        print(f"\n🌐 Network:")
        print(f"   Total Requests: {total_requests}")
        print(f"   Total Size: {total_size:.2f} MB")
        
        # Recommendations
        recommendations = analysis.get("recommendations", [])
        if recommendations:
            print(f"\n💡 Recommendations:")
            for rec in recommendations:
                priority_icon = {"high": "🔴", "medium": "🟡", "info": "ℹ️"}.get(rec.get("priority", "info"), "ℹ️")
                print(f"   {priority_icon} {rec.get('metric', 'N/A')}: {rec.get('value', 'N/A')} → {rec.get('recommendation', '')}")
    
    # Overall recommendations
    print(f"\n\n{'='*80}")
    print("OVERALL RECOMMENDATIONS")
    print("=" * 80)
    
    all_vitals = []
    all_network = []
    
    for result in results:
        if "error" not in result:
            all_vitals.append(result["analysis"].get("webVitals", {}))
            all_network.append(result["analysis"].get("networkMetrics", {}))
    
    if all_vitals:
        avg_lcp = sum(v.get("LCP", 0) for v in all_vitals) / len(all_vitals) / 1000
        avg_fcp = sum(v.get("FCP", 0) for v in all_vitals) / len(all_vitals) / 1000
        avg_cls = sum(v.get("CLS", 0) for v in all_vitals) / len(all_vitals)
        avg_size = sum(n.get("totalSize", 0) for n in all_network) / len(all_network) / (1024 * 1024)
        avg_requests = sum(n.get("totalRequests", 0) for n in all_network) / len(all_network)
        
        print(f"\n📊 Average Metrics Across All Pages:")
        print(f"   Average LCP: {avg_lcp:.2f}s")
        print(f"   Average FCP: {avg_fcp:.2f}s")
        print(f"   Average CLS: {avg_cls:.3f}")
        print(f"   Average Page Size: {avg_size:.2f} MB")
        print(f"   Average Requests: {avg_requests:.0f}")
        
        print(f"\n💡 Priority Improvements:")
        
        if avg_lcp > 2.5:
            print(f"   🔴 CRITICAL: Optimize Largest Contentful Paint (LCP)")
            print(f"      - Current: {avg_lcp:.2f}s, Target: <2.5s")
            print(f"      - Optimize hero images, preload critical resources")
            print(f"      - Reduce server response time, use CDN")
            print(f"      - Eliminate render-blocking resources")
        
        if avg_fcp > 1.8:
            print(f"   🔴 CRITICAL: Improve First Contentful Paint (FCP)")
            print(f"      - Current: {avg_fcp:.2f}s, Target: <1.8s")
            print(f"      - Minimize render-blocking CSS and JavaScript")
            print(f"      - Optimize font loading (use font-display: swap)")
            print(f"      - Reduce critical path length")
        
        if avg_cls > 0.1:
            print(f"   🔴 CRITICAL: Fix Cumulative Layout Shift (CLS)")
            print(f"      - Current: {avg_cls:.3f}, Target: <0.1")
            print(f"      - Set explicit dimensions on images and videos")
            print(f"      - Reserve space for ads and embeds")
            print(f"      - Avoid inserting content above existing content")
        
        if avg_size > 3:
            print(f"   🟡 MEDIUM: Reduce Page Size")
            print(f"      - Current: {avg_size:.2f} MB, Target: <3 MB")
            print(f"      - Optimize images (WebP, compression)")
            print(f"      - Enable compression (gzip/brotli)")
            print(f"      - Code splitting and lazy loading")
        
        if avg_requests > 80:
            print(f"   🟡 MEDIUM: Reduce Number of Requests")
            print(f"      - Current: {avg_requests:.0f}, Target: <80")
            print(f"      - Combine CSS and JavaScript files")
            print(f"      - Use image sprites where appropriate")
            print(f"      - Implement resource bundling")
    
    # Save results
    output_file = f"swadesh_performance_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_file, 'w') as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "pages": results
        }, f, indent=2)
    
    print(f"\n✅ Full report saved to: {output_file}")
    print("=" * 80)


async def main():
    """Main entry point"""
    urls = [
        "https://www.swadeshonline.com/",
        "https://www.swadeshonline.com/sections/art-decor",
        "https://www.swadeshonline.com/product/dal-decorative-piece-medium-multi-fs-7516762"
    ]
    
    port = 9222
    if len(sys.argv) > 1:
        try:
            port = int(sys.argv[1])
        except ValueError:
            print(f"Invalid port: {sys.argv[1]}, using default 9222")
    
    try:
        await analyze_multiple_pages(urls, port)
    except KeyboardInterrupt:
        print("\n\n⚠️  Analysis interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())


