# Page Performance Analysis Guide

This guide explains how to use the Chrome DevTools MCP server to analyze page performance.

## Quick Start

### 1. Start Chrome with Remote Debugging

**macOS:**
```bash
/Applications/Google\ Chrome.app/Contents/MacOS/Google\ Chrome --remote-debugging-port=9222
```

**Linux:**
```bash
google-chrome --remote-debugging-port=9222
```

**Windows:**
```bash
chrome.exe --remote-debugging-port=9222
```

### 2. Analyze a Page

#### Option A: Analyze Current Page in Chrome

If you already have a page open in Chrome:

```bash
python analyze_page_performance.py
```

#### Option B: Analyze a Specific URL

```bash
python analyze_page_performance.py --url https://example.com
```

#### Option C: Save Results to File

```bash
python analyze_page_performance.py --url https://example.com --output results.json
```

#### Option D: Get JSON Output

```bash
python analyze_page_performance.py --url https://example.com --json
```

## What Gets Analyzed

### Core Web Vitals
- **LCP (Largest Contentful Paint)**: Measures loading performance
  - Good: ≤ 2.5s
  - Needs Improvement: 2.5s - 4.0s
  - Poor: > 4.0s

- **FCP (First Contentful Paint)**: Time until first content is rendered
  - Good: ≤ 1.8s
  - Needs Improvement: 1.8s - 3.0s
  - Poor: > 3.0s

- **CLS (Cumulative Layout Shift)**: Measures visual stability
  - Good: ≤ 0.1
  - Needs Improvement: 0.1 - 0.25
  - Poor: > 0.25

- **FID (First Input Delay)**: Measures interactivity
  - Good: ≤ 100ms
  - Needs Improvement: 100ms - 300ms
  - Poor: > 300ms

- **TTFB (Time to First Byte)**: Server response time
  - Good: ≤ 800ms
  - Needs Improvement: 800ms - 1800ms
  - Poor: > 1800ms

### Network Metrics
- Total number of requests
- Total page size
- Average request time
- Requests by type (script, stylesheet, image, etc.)
- Slowest requests
- Largest resources

### Performance Recommendations
The analyzer automatically generates recommendations based on:
- Core Web Vitals scores
- Page size and request count
- Resource loading patterns

## Using with MCP Server

If you're using the MCP server with an AI assistant (like Claude Desktop or Cursor), you can use these commands:

### Available Performance Tools

1. **Get Core Web Vitals**
   ```
   Get Core Web Vitals for the current page
   ```

2. **Get Performance Metrics**
   ```
   Get performance metrics from the page
   ```

3. **Get Network Metrics**
   ```
   Get detailed network metrics
   ```

4. **Run Performance Audit**
   ```
   Run a comprehensive performance audit
   ```

5. **Analyze Page Performance**
   ```
   Analyze the current page performance with recommendations
   ```

## Example Analysis Workflow

1. **Start Chrome with debugging:**
   ```bash
   /Applications/Google\ Chrome.app/Contents/MacOS/Google\ Chrome --remote-debugging-port=9222
   ```

2. **Open the page you want to analyze in Chrome**

3. **Run the analysis:**
   ```bash
   python analyze_page_performance.py
   ```

4. **Review the results:**
   - Check Core Web Vitals scores
   - Review network metrics
   - Read recommendations
   - Focus on high-priority issues first

5. **Make improvements:**
   - Optimize images
   - Reduce render-blocking resources
   - Fix layout shifts
   - Minimize JavaScript and CSS
   - Enable compression and caching

6. **Re-analyze to verify improvements**

## Troubleshooting

### Chrome Not Connecting

If you get connection errors:

1. **Verify Chrome is running with remote debugging:**
   ```bash
   curl http://localhost:9222/json
   ```
   Should return a JSON array of targets.

2. **Check the port:**
   Make sure the port matches (default: 9222)

3. **Firewall:**
   Ensure your firewall isn't blocking localhost connections

### No Performance Data

If metrics are empty or zero:

1. **Wait for page to fully load** - The script waits 3 seconds, but some pages may need more
2. **Check if page has loaded** - Make sure the page is fully rendered
3. **Try navigating to the page first** - Use `--url` flag to navigate and then analyze

### Missing Web Vitals

Some metrics (like FID) require user interaction. If they're missing, that's normal for automated analysis.

## Advanced Usage

### Custom Port

```bash
python analyze_page_performance.py --port 9223 --url https://example.com
```

### Combine with Other Tools

You can combine this with other performance tools:

```bash
# Run analysis
python analyze_page_performance.py --url https://example.com --output results.json

# Process results with other tools
jq '.webVitals.LCP' results.json
```

## Integration with CI/CD

You can integrate this into your CI/CD pipeline:

```bash
#!/bin/bash
# Start Chrome in headless mode
google-chrome --headless --remote-debugging-port=9222 --no-sandbox &

# Wait for Chrome to start
sleep 2

# Run analysis
python analyze_page_performance.py --url https://your-site.com --json > results.json

# Check if LCP is acceptable
LCP=$(jq '.webVitals.LCP' results.json)
if (( $(echo "$LCP > 2500" | bc -l) )); then
    echo "LCP is too high: ${LCP}ms"
    exit 1
fi
```

## Best Practices

1. **Test on real devices** - Performance can vary significantly
2. **Test multiple pages** - Different pages may have different performance characteristics
3. **Test at different times** - Network conditions can affect results
4. **Use throttling** - Test with slower network connections
5. **Monitor over time** - Track performance metrics over time to catch regressions

## Resources

- [Web Vitals](https://web.dev/vitals/)
- [Chrome DevTools Protocol](https://chromedevtools.github.io/devtools-protocol/)
- [Lighthouse](https://developers.google.com/web/tools/lighthouse)
- [PageSpeed Insights](https://pagespeed.web.dev/)

