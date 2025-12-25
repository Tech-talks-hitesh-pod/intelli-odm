#!/usr/bin/env python3
"""
Comprehensive analysis of all three Swadesh pages
Analyzes homepage, art-decor section, and product page
"""

import asyncio
import json
from datetime import datetime
from analyze_page_performance import PerformanceAnalyzer


async def analyze_all_swadesh_pages():
    """Analyze all three Swadesh pages"""
    
    pages = [
        {
            "name": "Homepage",
            "url": "https://www.swadeshonline.com/"
        },
        {
            "name": "Art & Decor Section",
            "url": "https://www.swadeshonline.com/sections/art-decor"
        },
        {
            "name": "Product Page",
            "url": "https://www.swadeshonline.com/product/dal-decorative-piece-medium-multi-fs-7516762"
        }
    ]
    
    print("=" * 80)
    print("SWADESH ONLINE - COMPREHENSIVE PERFORMANCE ANALYSIS")
    print("=" * 80)
    print(f"\n📊 Analyzing {len(pages)} pages...\n")
    
    analyzer = PerformanceAnalyzer(chrome_debug_port=9222)
    all_results = []
    
    try:
        # Check connection
        targets = await analyzer.get_chrome_targets()
        print(f"✅ Connected to Chrome\n")
        
        for i, page in enumerate(pages, 1):
            print(f"{'='*80}")
            print(f"Page {i}/{len(pages)}: {page['name']}")
            print(f"URL: {page['url']}")
            print(f"{'='*80}\n")
            
            try:
                print(f"🌐 Navigating to {page['url']}...")
                analysis = await analyzer.analyze_page(url=page['url'])
                
                result = {
                    "name": page['name'],
                    "url": page['url'],
                    "analysis": analysis,
                    "timestamp": datetime.now().isoformat()
                }
                all_results.append(result)
                
                # Print summary
                vitals = analysis.get("webVitals", {})
                network = analysis.get("networkMetrics", {})
                
                lcp = vitals.get("LCP", 0) / 1000
                fcp = vitals.get("FCP", 0) / 1000
                cls = vitals.get("CLS", 0)
                ttfb = vitals.get("TTFB", 0)
                total_requests = network.get("totalRequests", 0)
                total_size = network.get("totalSize", 0) / (1024 * 1024)
                
                def get_status(value, good, poor, reverse=False):
                    if reverse:
                        value, good, poor = good, value, poor
                    if value <= good:
                        return "✅"
                    elif value <= poor:
                        return "⚠️"
                    else:
                        return "❌"
                
                print(f"\n📊 Quick Summary:")
                print(f"   LCP: {lcp:.2f}s {get_status(lcp, 2.5, 4.0)}")
                print(f"   FCP: {fcp:.2f}s {get_status(fcp, 1.8, 3.0)}")
                print(f"   CLS: {cls:.3f} {get_status(cls, 0.1, 0.25)}")
                print(f"   TTFB: {ttfb:.0f}ms {get_status(ttfb, 800, 1800)}")
                print(f"   Requests: {total_requests} {get_status(total_requests, 100, 150, reverse=True)}")
                print(f"   Size: {total_size:.2f} MB {get_status(total_size, 3, 5, reverse=True)}")
                
            except Exception as e:
                print(f"❌ Error analyzing {page['name']}: {e}")
                all_results.append({
                    "name": page['name'],
                    "url": page['url'],
                    "error": str(e),
                    "timestamp": datetime.now().isoformat()
                })
                import traceback
                traceback.print_exc()
            
            if i < len(pages):
                print("\n⏳ Waiting 5 seconds before next page...\n")
                await asyncio.sleep(5)
        
        # Generate comprehensive report
        print("\n\n" + "=" * 80)
        print("COMPREHENSIVE PERFORMANCE REPORT")
        print("=" * 80)
        
        for result in all_results:
            if "error" in result:
                print(f"\n❌ {result['name']}: {result['error']}")
                continue
            
            analysis = result["analysis"]
            vitals = analysis.get("webVitals", {})
            network = analysis.get("networkMetrics", {})
            
            print(f"\n{'─'*80}")
            print(f"📄 {result['name']}")
            print(f"🔗 {result['url']}")
            print(f"{'─'*80}")
            
            lcp = vitals.get("LCP", 0) / 1000
            fcp = vitals.get("FCP", 0) / 1000
            cls = vitals.get("CLS", 0)
            ttfb = vitals.get("TTFB", 0)
            total_requests = network.get("totalRequests", 0)
            total_size = network.get("totalSize", 0) / (1024 * 1024)
            
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
            print(f"   TTFB: {ttfb:.0f}ms {get_status(ttfb, 800, 1800)}")
            
            print(f"\n🌐 Network Metrics:")
            print(f"   Total Requests: {total_requests}")
            print(f"   Total Size: {total_size:.2f} MB")
            
            # Resource breakdown
            by_type = network.get("byType", {})
            if by_type:
                print(f"\n   Top Resource Types:")
                for rtype, data in sorted(by_type.items(), key=lambda x: x[1]["count"], reverse=True)[:5]:
                    size_mb = data["size"] / (1024 * 1024)
                    print(f"     {rtype:15s}: {data['count']:3d} requests, {size_mb:.2f} MB")
            
            # Recommendations
            recommendations = analysis.get("recommendations", [])
            if recommendations:
                print(f"\n💡 Recommendations:")
                for rec in recommendations[:3]:
                    priority_icon = {"high": "🔴", "medium": "🟡", "info": "ℹ️"}.get(rec.get("priority", "info"), "ℹ️")
                    print(f"   {priority_icon} {rec.get('metric', 'N/A')}: {rec.get('value', 'N/A')}")
        
        # Overall summary
        print(f"\n\n{'='*80}")
        print("OVERALL SUMMARY")
        print("=" * 80)
        
        successful_results = [r for r in all_results if "error" not in r]
        
        if successful_results:
            avg_requests = sum(r["analysis"].get("networkMetrics", {}).get("totalRequests", 0) for r in successful_results) / len(successful_results)
            avg_size = sum(r["analysis"].get("networkMetrics", {}).get("totalSize", 0) for r in successful_results) / len(successful_results) / (1024 * 1024)
            
            print(f"\n📊 Average Metrics Across All Pages:")
            print(f"   Average Requests: {avg_requests:.0f}")
            print(f"   Average Page Size: {avg_size:.2f} MB")
            
            # Collect all issues
            all_issues = []
            for result in successful_results:
                recommendations = result["analysis"].get("recommendations", [])
                for rec in recommendations:
                    if rec.get("priority") in ["high", "medium"]:
                        all_issues.append({
                            "page": result["name"],
                            "metric": rec.get("metric"),
                            "value": rec.get("value"),
                            "recommendation": rec.get("recommendation")
                        })
            
            if all_issues:
                print(f"\n🔴 Priority Issues Found: {len(all_issues)}")
                print("\nTop Issues:")
                for i, issue in enumerate(all_issues[:5], 1):
                    print(f"   {i}. {issue['page']} - {issue['metric']}: {issue['value']}")
        
        # Save results
        output_file = f"swadesh_comprehensive_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(output_file, 'w') as f:
            json.dump({
                "timestamp": datetime.now().isoformat(),
                "pages": all_results
            }, f, indent=2)
        
        print(f"\n✅ Full report saved to: {output_file}")
        print("=" * 80)
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    try:
        asyncio.run(analyze_all_swadesh_pages())
    except KeyboardInterrupt:
        print("\n\n⚠️  Analysis interrupted by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()

