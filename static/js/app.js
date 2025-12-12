// app.js
let uploadedFiles = {};
let storeMappings = [];
let forecastResults = null;

// Initialize on page load
document.addEventListener('DOMContentLoaded', function() {
    setupFileUploads();
    loadAvailableStores();
});

function setupFileUploads() {
    const fileInputs = ['sales_data', 'inventory_data', 'price_data', 'cost_data', 'new_articles_data'];
    
    fileInputs.forEach(inputId => {
        const input = document.getElementById(inputId);
        if (input) {
            input.addEventListener('change', function(e) {
                if (e.target.files.length > 0) {
                    uploadFile(inputId, e.target.files[0]);
                }
            });
        }
    });
}

async function uploadFile(fieldName, file) {
    const formData = new FormData();
    formData.append(fieldName, file);
    
    try {
        const response = await fetch('/api/upload', {
            method: 'POST',
            body: formData
        });
        
        const data = await response.json();
        
        if (data.success) {
            uploadedFiles[fieldName] = data.files[fieldName];
            showSuccess(`File uploaded: ${file.name}`);
        } else {
            showError(data.error || 'Upload failed');
        }
    } catch (error) {
        showError('Error uploading file: ' + error.message);
    }
}

async function runForecast() {
    // Validate required files
    if (!uploadedFiles['sales_data']) {
        showError('Please upload Historical Sales Data');
        return;
    }
    
    if (!uploadedFiles['new_articles_data']) {
        showError('Please upload New Articles Data');
        return;
    }
    
    // Show loading
    document.getElementById('loading').style.display = 'block';
    document.getElementById('error-message').style.display = 'none';
    hideAllTabs();
    
    // Get parameters
    const params = {
        file_paths: uploadedFiles,
        margin_target: parseFloat(document.getElementById('margin_target').value) || 30,
        variance_threshold: parseFloat(document.getElementById('variance_threshold').value) || 5,
        forecast_horizon_days: parseInt(document.getElementById('forecast_horizon_days').value) || 60,
        max_quantity_per_store: parseInt(document.getElementById('max_quantity_per_store').value) || 500,
        universe_of_stores: document.getElementById('universe_of_stores').value ? 
            parseInt(document.getElementById('universe_of_stores').value) : null,
        price_options: [200, 300, 400, 500, 600, 700, 800, 900, 1000],
        store_mappings: storeMappings
    };
    
    try {
        const response = await fetch('/api/forecast', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(params)
        });
        
        const data = await response.json();
        
        document.getElementById('loading').style.display = 'none';
        
        if (data.success) {
            forecastResults = data.results;
            displayResults(data.results);
            showSuccess('Forecast completed successfully!');
        } else {
            showError(data.error || 'Forecast failed');
        }
    } catch (error) {
        document.getElementById('loading').style.display = 'none';
        showError('Error running forecast: ' + error.message);
    }
}

function displayResults(results) {
    // Display top banner
    displayTopBanner(results.top_banner || {});
    
    // Display recommendations
    displayRecommendations(results.recommendations || {});
    
    // Display article metrics
    displayArticleMetrics(results.article_level_metrics || {});
    
    // Display store allocations
    displayStoreAllocations(results.store_allocations || {});
    
    // Display factor analysis
    displayFactorAnalysis(results.factor_analysis || {});
    
    // Show results tabs
    document.getElementById('results-tabs').style.display = 'flex';
    showTab('recommendations');
}

function displayTopBanner(banner) {
    const bannerCard = document.getElementById('top-banner');
    const bannerContent = document.getElementById('banner-content');
    
    if (!banner || Object.keys(banner).length === 0) {
        bannerCard.style.display = 'none';
        return;
    }
    
    bannerCard.style.display = 'block';
    
    bannerContent.innerHTML = `
        <div class="banner-item">
            <div class="banner-item-label">Total Unique SKUs</div>
            <div class="banner-item-value">${banner.total_unique_skus || 0}</div>
        </div>
        <div class="banner-item">
            <div class="banner-item-label">Total Quantity Bought</div>
            <div class="banner-item-value">${formatNumber(banner.total_quantity_bought || 0)}</div>
        </div>
        <div class="banner-item">
            <div class="banner-item-label">Total Stores</div>
            <div class="banner-item-value">${banner.total_stores || 0}</div>
        </div>
        <div class="banner-item">
            <div class="banner-item-label">Total Buy Cost</div>
            <div class="banner-item-value">₹${formatNumber(banner.total_buy_cost || 0)}</div>
        </div>
        <div class="banner-item">
            <div class="banner-item-label">Total Sales Value</div>
            <div class="banner-item-value">₹${formatNumber(banner.total_sales_value || 0)}</div>
        </div>
        <div class="banner-item">
            <div class="banner-item-label">Avg Margin vs Target</div>
            <div class="banner-item-value">
                ${((banner.average_margin_achieved || 0) * 100).toFixed(1)}% / ${((banner.target_margin || 0) * 100).toFixed(1)}%
                ${banner.margin_vs_target?.meets_target ? '✅' : '⚠️'}
            </div>
        </div>
    `;
}

function displayRecommendations(recommendations) {
    const content = document.getElementById('recommendations-content');
    
    if (!recommendations || Object.keys(recommendations).length === 0) {
        content.innerHTML = '<p>No recommendations available</p>';
        return;
    }
    
    let html = '<div class="metric-card">';
    html += `<h4>Summary</h4>`;
    html += `<p><strong>Articles to Buy:</strong> ${recommendations.articles_to_buy?.length || 0}</p>`;
    html += `<p><strong>Total Procurement Quantity:</strong> ${formatNumber(recommendations.total_procurement_quantity || 0)} units</p>`;
    html += `<p><strong>Total Stores:</strong> ${recommendations.total_stores || 0}</p>`;
    
    if (recommendations.store_universe_validation) {
        const validation = recommendations.store_universe_validation;
        html += `<p><strong>Store Universe Validation:</strong> `;
        if (validation.valid) {
            html += `<span style="color: green;">✅ ${validation.message}</span>`;
        } else {
            html += `<span style="color: red;">⚠️ ${validation.message}</span>`;
        }
        html += `</p>`;
    }
    
    html += '</div>';
    
    if (recommendations.optimization_summary) {
        html += '<div class="metric-card">';
        html += '<h4>Optimization Summary</h4>';
        html += `<pre style="white-space: pre-wrap; font-family: inherit;">${recommendations.optimization_summary}</pre>`;
        html += '</div>';
    }
    
    content.innerHTML = html;
}

function displayArticleMetrics(metrics) {
    const content = document.getElementById('article-metrics-content');
    
    if (!metrics || Object.keys(metrics).length === 0) {
        content.innerHTML = '<p>No article metrics available</p>';
        return;
    }
    
    let html = '<table><thead><tr>';
    html += '<th>Article</th><th>Style Code</th><th>Color</th><th>Segment</th>';
    html += '<th>MRP</th><th>Avg Price</th><th>Discount</th><th>Margin</th>';
    html += '<th>Stores</th><th>ROS</th><th>STR</th><th>Net Sales</th>';
    html += '</tr></thead><tbody>';
    
    for (const [article, data] of Object.entries(metrics)) {
        const details = data.article_details || {};
        html += '<tr>';
        html += `<td><strong>${article}</strong></td>`;
        html += `<td>${details.style_code || 'N/A'}</td>`;
        html += `<td>${details.color || 'N/A'}</td>`;
        html += `<td>${details.segment || 'N/A'}</td>`;
        html += `<td>₹${(data.mrp || 0).toFixed(2)}</td>`;
        html += `<td>₹${(data.average_selling_price || 0).toFixed(2)}</td>`;
        html += `<td>${(data.average_discount || 0).toFixed(1)}%</td>`;
        html += `<td>${((data.margin_pct || 0) * 100).toFixed(1)}%</td>`;
        html += `<td>${data.total_store_exposure || 0}</td>`;
        html += `<td>${(data.ros || 0).toFixed(2)}/day</td>`;
        html += `<td>${((data.str || 0) * 100).toFixed(1)}%</td>`;
        html += `<td>₹${formatNumber(data.net_sales_value || 0)}</td>`;
        html += '</tr>';
    }
    
    html += '</tbody></table>';
    content.innerHTML = html;
}

function displayStoreAllocations(allocations) {
    const content = document.getElementById('store-allocations-content');
    
    if (!allocations || Object.keys(allocations).length === 0) {
        content.innerHTML = '<p>No store allocations available</p>';
        return;
    }
    
    let html = '';
    
    for (const [article, stores] of Object.entries(allocations)) {
        html += `<div class="metric-card"><h4>Article: ${article}</h4>`;
        html += '<table><thead><tr>';
        html += '<th>Store ID</th><th>Quantity</th><th>Expected ST</th><th>Expected ROS</th>';
        html += '<th>Margin %</th><th>Score</th><th>Action</th>';
        html += '</tr></thead><tbody>';
        
        for (const [storeId, data] of Object.entries(stores)) {
            html += '<tr>';
            html += `<td>${storeId}</td>`;
            html += `<td><input type="number" class="editable-quantity" value="${data.quantity || 0}" 
                     onchange="updateStoreQuantity('${article}', '${storeId}', this.value)"></td>`;
            html += `<td>${((data.expected_sell_through || 0) * 100).toFixed(1)}%</td>`;
            html += `<td>${(data.expected_rate_of_sale || 0).toFixed(2)}/day</td>`;
            html += `<td>${((data.margin_pct || 0) * 100).toFixed(1)}%</td>`;
            html += `<td>${(data.optimization_score || 0).toFixed(2)}</td>`;
            html += `<td>${data.recommendation || 'N/A'}</td>`;
            html += '</tr>';
        }
        
        html += '</tbody></table></div>';
    }
    
    content.innerHTML = html;
}

function displayFactorAnalysis(analysis) {
    const content = document.getElementById('factor-analysis-content');
    
    if (!analysis || Object.keys(analysis).length === 0) {
        content.innerHTML = '<p>No factor analysis available</p>';
        return;
    }
    
    let html = '';
    
    // Correlation Analysis
    if (analysis.correlation_matrix) {
        html += '<div class="metric-card"><h4>Correlation Analysis</h4>';
        if (analysis.correlation_matrix.top_correlations) {
            html += '<ul>';
            for (const [attr, value] of Object.entries(analysis.correlation_matrix.top_correlations)) {
                html += `<li><strong>${attr}:</strong> ${value.toFixed(3)}</li>`;
            }
            html += '</ul>';
        }
        html += '</div>';
    }
    
    // PCA Results
    if (analysis.pca_results) {
        html += '<div class="metric-card"><h4>PCA Results</h4>';
        html += `<p><strong>Components:</strong> ${analysis.pca_results.n_components || 'N/A'}</p>`;
        if (analysis.pca_results.explained_variance_ratio) {
            html += `<p><strong>Explained Variance:</strong> ${analysis.pca_results.explained_variance_ratio.map(v => (v * 100).toFixed(1) + '%').join(', ')}</p>`;
        }
        html += '</div>';
    }
    
    // Factor Importance
    if (analysis.factor_importance) {
        html += '<div class="metric-card"><h4>Factor Importance</h4>';
        html += '<table><thead><tr><th>Factor</th><th>Importance</th></tr></thead><tbody>';
        analysis.factor_importance.slice(0, 10).forEach(([factor, importance]) => {
            html += `<tr><td>${factor}</td><td>${(importance * 100).toFixed(2)}%</td></tr>`;
        });
        html += '</tbody></table></div>';
    }
    
    content.innerHTML = html || '<p>No factor analysis data available</p>';
}

function showTab(tabName) {
    // Hide all tabs
    const tabs = ['recommendations', 'article-metrics', 'store-allocations', 'factor-analysis', 'audit-trail'];
    tabs.forEach(tab => {
        document.getElementById(`${tab}-tab`).style.display = 'none';
        document.querySelector(`[onclick="showTab('${tab}')"]`)?.classList.remove('active');
    });
    
    // Show selected tab
    document.getElementById(`${tabName}-tab`).style.display = 'block';
    document.querySelector(`[onclick="showTab('${tabName}')"]`)?.classList.add('active');
}

function hideAllTabs() {
    const tabs = ['recommendations', 'article-metrics', 'store-allocations', 'factor-analysis', 'audit-trail'];
    tabs.forEach(tab => {
        document.getElementById(`${tab}-tab`).style.display = 'none';
    });
}

function addStoreMapping() {
    const newStoreId = document.getElementById('new_store_id').value;
    const referenceStoreId = document.getElementById('reference_store_id').value;
    
    if (!newStoreId || !referenceStoreId) {
        showError('Please enter both new store ID and reference store');
        return;
    }
    
    storeMappings.push({
        new_store_id: newStoreId,
        reference_store_id: referenceStoreId
    });
    
    updateStoreMappingsList();
    
    // Clear inputs
    document.getElementById('new_store_id').value = '';
    document.getElementById('reference_store_id').value = '';
}

function updateStoreMappingsList() {
    const list = document.getElementById('store-mappings-list');
    list.innerHTML = '';
    
    storeMappings.forEach((mapping, index) => {
        const item = document.createElement('div');
        item.className = 'store-mapping-item';
        item.innerHTML = `
            <span><strong>${mapping.new_store_id}</strong> → ${mapping.reference_store_id}</span>
            <button class="btn btn-secondary" onclick="removeStoreMapping(${index})">
                <i class="fas fa-times"></i>
            </button>
        `;
        list.appendChild(item);
    });
}

function removeStoreMapping(index) {
    storeMappings.splice(index, 1);
    updateStoreMappingsList();
}

async function loadAvailableStores() {
    // This would load stores from uploaded sales data
    // For now, placeholder
}

function updateStoreQuantity(article, storeId, newQuantity) {
    // This would call the HITL workflow to update quantity
    console.log(`Updating ${article} at ${storeId} to ${newQuantity}`);
    // In production, make API call to update quantity
}

function clearAll() {
    uploadedFiles = {};
    storeMappings = [];
    forecastResults = null;
    
    // Clear file inputs
    ['sales_data', 'inventory_data', 'price_data', 'cost_data', 'new_articles_data'].forEach(id => {
        document.getElementById(id).value = '';
    });
    
    // Reset parameters
    document.getElementById('margin_target').value = 30;
    document.getElementById('variance_threshold').value = 5;
    document.getElementById('forecast_horizon_days').value = 60;
    document.getElementById('max_quantity_per_store').value = 500;
    document.getElementById('universe_of_stores').value = '';
    
    // Clear results
    document.getElementById('top-banner').style.display = 'none';
    document.getElementById('results-tabs').style.display = 'none';
    hideAllTabs();
    updateStoreMappingsList();
    
    showSuccess('All data cleared');
}

function showError(message) {
    const errorDiv = document.getElementById('error-message');
    errorDiv.textContent = message;
    errorDiv.style.display = 'block';
    setTimeout(() => {
        errorDiv.style.display = 'none';
    }, 5000);
}

function showSuccess(message) {
    // Create temporary success message
    const successDiv = document.createElement('div');
    successDiv.className = 'success-message';
    successDiv.textContent = message;
    document.querySelector('.main-content').insertBefore(successDiv, document.querySelector('.left-panel'));
    setTimeout(() => {
        successDiv.remove();
    }, 3000);
}

function formatNumber(num) {
    return new Intl.NumberFormat('en-IN').format(num);
}

