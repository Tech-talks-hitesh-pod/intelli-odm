# Swadesh Online - Performance Improvement Plan

**Analysis Date:** December 25, 2025  
**Pages Analyzed:** Homepage, Art & Decor Section, Product Page

## Executive Summary

### Current Performance Status

| Metric | Homepage | Art & Decor | Product Page | Average | Target | Status |
|--------|----------|-------------|-------------|---------|--------|--------|
| **HTTP Requests** | 168 | 190 | 138 | 165 | <100 | ❌ Needs Improvement |
| **Page Size** | 0.03 MB | 0.03 MB | 0.02 MB | 0.03 MB | <3 MB | ✅ Good |
| **LCP** | 0.00s* | 0.00s* | 0.00s* | 0.00s* | <2.5s | ✅ Good |
| **FCP** | 0.00s* | 0.00s* | 0.00s* | 0.00s* | <1.8s | ✅ Good |
| **CLS** | 0.000 | 0.000 | 0.000 | 0.000 | <0.1 | ✅ Good |

*Note: Web Vitals showing 0.00s may indicate cached loads or metrics not fully captured. Further investigation recommended.

### Key Findings

1. **High HTTP Request Count** - All pages exceed the recommended 100 requests
2. **Excessive Preload Links** - 50+ link requests per page (preload/prefetch)
3. **Multiple Third-Party Scripts** - Analytics, tracking, and personalization scripts
4. **Good Page Size** - Total page size is well optimized
5. **Fast Load Times** - Average request time is good (0.03s)

---

## Priority 1: Reduce HTTP Requests (High Impact)

### Current State
- **Homepage:** 168 requests
- **Art & Decor:** 190 requests  
- **Product Page:** 138 requests
- **Average:** 165 requests

### Target
- Reduce to <100 requests per page
- **Potential improvement:** 40-65% reduction

### Action Items

#### 1.1 Optimize Preload/Prefetch Links (Estimated: -50 requests)
**Current:** 50-52 link requests per page

**Actions:**
- Audit all `<link rel="preload">` and `<link rel="prefetch">` tags
- Remove unnecessary preloads for non-critical resources
- Keep only critical above-the-fold resources
- Use `rel="preload"` only for fonts, critical CSS, and hero images
- Implement conditional preloading based on user intent

**Implementation:**
```html
<!-- Keep only critical resources -->
<link rel="preload" href="/fonts/critical.woff2" as="font" crossorigin>
<link rel="preload" href="/css/critical.css" as="style">

<!-- Remove non-critical preloads -->
<!-- <link rel="preload" href="/images/below-fold.jpg" as="image"> -->
```

**Expected Impact:** Reduce link requests from 50+ to 10-15 per page

#### 1.2 Bundle JavaScript Files (Estimated: -15 requests)
**Current:** 25 script requests per page

**Actions:**
- Combine multiple small JavaScript files into bundles
- Use code splitting for route-based chunks
- Implement lazy loading for non-critical scripts
- Use webpack/rollup for bundling

**Implementation:**
```javascript
// Before: Multiple separate files
<script src="/js/utils.js"></script>
<script src="/js/helpers.js"></script>
<script src="/js/validators.js"></script>

// After: Single bundled file
<script src="/js/bundle.js"></script>
```

**Expected Impact:** Reduce script requests from 25 to 5-8 per page

#### 1.3 Optimize Image Loading (Estimated: -10 requests)
**Current:** 21-47 image requests per page

**Actions:**
- Implement lazy loading for below-the-fold images
- Use responsive images with srcset
- Combine small images into sprites where appropriate
- Use modern formats (WebP, AVIF) with fallbacks

**Implementation:**
```html
<!-- Lazy load images -->
<img src="placeholder.jpg" 
     data-src="image.jpg" 
     loading="lazy" 
     alt="Description">

<!-- Responsive images -->
<img srcset="image-400.jpg 400w,
             image-800.jpg 800w,
             image-1200.jpg 1200w"
     sizes="(max-width: 600px) 400px,
            (max-width: 1200px) 800px,
            1200px"
     src="image-800.jpg"
     alt="Description">
```

**Expected Impact:** Reduce initial image requests by 30-40%

#### 1.4 Consolidate CSS Files (Estimated: -5 requests)
**Current:** Multiple CSS files

**Actions:**
- Combine CSS files into single or minimal files
- Extract critical CSS and inline it
- Load non-critical CSS asynchronously
- Use CSS-in-JS or component-based CSS

**Expected Impact:** Reduce CSS requests to 1-2 per page

---

## Priority 2: Optimize Third-Party Scripts (Medium Impact)

### Current Third-Party Scripts Identified
1. **Facebook Privacy Sandbox Pixel** - 0.29s load time
2. **Google Ads Tracking** - 0.26s load time
3. **Dynamic Yield Personalization** - 0.21s load time
4. **Sentry Error Tracking** - 0.23s load time
5. **Google Analytics** - Multiple requests

### Action Items

#### 2.1 Defer Non-Critical Third-Party Scripts
**Actions:**
- Load analytics and tracking scripts after page load
- Use `async` or `defer` attributes
- Implement consent-based loading for GDPR compliance
- Use tag managers to consolidate scripts

**Implementation:**
```html
<!-- Defer non-critical scripts -->
<script src="analytics.js" defer></script>
<script src="tracking.js" defer></script>

<!-- Or load after page load -->
<script>
window.addEventListener('load', function() {
    // Load third-party scripts
    loadScript('analytics.js');
    loadScript('tracking.js');
});
</script>
```

**Expected Impact:** Reduce initial blocking time by 0.5-1s

#### 2.2 Consolidate Tracking Scripts
**Actions:**
- Use Google Tag Manager to manage all tracking scripts
- Consolidate multiple analytics into single implementation
- Remove duplicate tracking pixels

**Expected Impact:** Reduce tracking-related requests by 50%

---

## Priority 3: Implement Resource Hints Strategically (Medium Impact)

### Current State
- 50+ preload/prefetch links per page
- Many may be unnecessary or poorly prioritized

### Action Items

#### 3.1 Audit and Prioritize Resource Hints
**Actions:**
- Keep only critical resource hints
- Use `dns-prefetch` for external domains
- Use `preconnect` for critical third-party resources
- Remove prefetch for resources unlikely to be needed

**Implementation:**
```html
<!-- Critical: DNS prefetch for external domains -->
<link rel="dns-prefetch" href="https://cdn.swadeshonline.com">
<link rel="dns-prefetch" href="https://fonts.googleapis.com">

<!-- Critical: Preconnect for important resources -->
<link rel="preconnect" href="https://api.swadeshonline.com">

<!-- Remove: Non-critical prefetches -->
<!-- <link rel="prefetch" href="/products/unlikely-page"> -->
```

**Expected Impact:** Reduce link requests while maintaining performance

---

## Priority 4: Image Optimization (Low-Medium Impact)

### Current State
- 21-47 images per page
- Total size is good (0.02-0.03 MB), but optimization can improve

### Action Items

#### 4.1 Implement Modern Image Formats
**Actions:**
- Convert images to WebP format (30-50% smaller)
- Use AVIF for supported browsers (50-70% smaller)
- Provide fallbacks for older browsers

**Implementation:**
```html
<picture>
  <source srcset="image.avif" type="image/avif">
  <source srcset="image.webp" type="image/webp">
  <img src="image.jpg" alt="Description">
</picture>
```

#### 4.2 Implement Lazy Loading
**Actions:**
- Use native `loading="lazy"` attribute
- Implement intersection observer for custom lazy loading
- Load images only when they're about to enter viewport

**Expected Impact:** Reduce initial page load by 20-30%

---

## Priority 5: Code Splitting and Lazy Loading (Medium Impact)

### Action Items

#### 5.1 Implement Route-Based Code Splitting
**Actions:**
- Split JavaScript by routes/pages
- Load page-specific code only when needed
- Use dynamic imports for heavy components

**Implementation:**
```javascript
// Route-based code splitting
const ProductPage = lazy(() => import('./pages/ProductPage'));
const ArtDecorPage = lazy(() => import('./pages/ArtDecorPage'));

// Component-based lazy loading
const HeavyComponent = lazy(() => import('./components/HeavyComponent'));
```

#### 5.2 Lazy Load Non-Critical Components
**Actions:**
- Lazy load modals, popups, and below-the-fold content
- Defer loading of recommendation widgets
- Load social sharing buttons on demand

**Expected Impact:** Reduce initial JavaScript bundle size by 40-60%

---

## Implementation Roadmap

### Phase 1: Quick Wins (Week 1-2)
- [ ] Audit and remove unnecessary preload links
- [ ] Defer non-critical third-party scripts
- [ ] Implement image lazy loading
- [ ] **Expected Improvement:** 30-40% reduction in requests

### Phase 2: Optimization (Week 3-4)
- [ ] Bundle JavaScript files
- [ ] Consolidate CSS files
- [ ] Optimize image formats (WebP)
- [ ] **Expected Improvement:** Additional 20-30% reduction

### Phase 3: Advanced Optimization (Week 5-6)
- [ ] Implement code splitting
- [ ] Consolidate tracking scripts
- [ ] Fine-tune resource hints
- [ ] **Expected Improvement:** Additional 10-20% reduction

### Phase 4: Monitoring and Refinement (Ongoing)
- [ ] Set up performance monitoring
- [ ] A/B test optimizations
- [ ] Continuous monitoring and improvement

---

## Expected Results

### Before Optimization
- **Average Requests:** 165
- **Page Size:** 0.03 MB
- **Load Time:** ~0.5-1s (estimated)

### After Optimization (Target)
- **Average Requests:** <100 (40% reduction)
- **Page Size:** 0.02-0.03 MB (maintained)
- **Load Time:** <0.5s (50% improvement)

### Success Metrics
- ✅ HTTP requests reduced to <100 per page
- ✅ LCP < 2.5s (when measured on fresh load)
- ✅ FCP < 1.8s
- ✅ CLS < 0.1
- ✅ PageSpeed Insights score > 90

---

## Monitoring and Validation

### Tools to Use
1. **Chrome DevTools** - Network tab, Performance tab
2. **Lighthouse** - Automated performance audits
3. **PageSpeed Insights** - Real-world performance data
4. **WebPageTest** - Detailed waterfall analysis
5. **Custom Monitoring Script** - (See `monitor_swadesh_performance.py`)

### Key Metrics to Track
- HTTP request count
- Total page size
- Core Web Vitals (LCP, FCP, CLS, FID, TTFB)
- Time to Interactive (TTI)
- First Byte Time (TTFB)
- Total Blocking Time (TBT)

---

## Additional Recommendations

### 1. CDN Optimization
- Ensure all static assets are served from CDN
- Use CDN caching headers effectively
- Implement CDN-level compression

### 2. Caching Strategy
- Implement service workers for offline support
- Use browser caching headers (Cache-Control, ETag)
- Cache API responses appropriately

### 3. Server Optimization
- Enable HTTP/2 or HTTP/3
- Implement server-side compression (gzip/brotli)
- Optimize server response times

### 4. Progressive Enhancement
- Implement critical CSS inline
- Use skeleton screens for loading states
- Progressive image loading

---

## Conclusion

The Swadesh website has good fundamentals (small page size, fast request times) but needs optimization in HTTP request count. By implementing the recommendations in this plan, we can achieve:

- **40-65% reduction in HTTP requests**
- **Improved Core Web Vitals scores**
- **Better user experience**
- **Higher search engine rankings**

The improvements should be implemented in phases, starting with quick wins and progressing to more advanced optimizations. Continuous monitoring is essential to validate improvements and identify new optimization opportunities.

---

**Next Steps:**
1. Review and approve this improvement plan
2. Set up performance monitoring (use `monitor_swadesh_performance.py`)
3. Begin Phase 1 implementation
4. Schedule weekly performance reviews

