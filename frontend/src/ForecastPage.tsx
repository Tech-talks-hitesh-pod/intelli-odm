import React from 'react';
import './ForecastModern.css';

interface ForecastPageProps {
  params: any;
  setParams: (params: any) => void;
  loading: boolean;
  error: string | null;
  isStreaming: boolean;
  previousRuns: any[];
  auditLogs: any[];
  streamingLogs: any[];
  results: any;
  currentRunId: string | null;
  selectedRunId: string | null;
  setSelectedRunId: (id: string | null) => void;
  forecastTab: 'logs' | 'summary';
  setForecastTab: (tab: 'logs' | 'summary') => void;
  logsDisplayLimit: number;
  setLogsDisplayLimit: (limit: number | ((prev: number) => number)) => void;
  handleForecast: () => void;
  loadForecastRun: (runId: string) => void;
  CollapsibleJsonTile: React.FC<any>;
}

const ForecastPage: React.FC<ForecastPageProps> = ({
  params,
  setParams,
  loading,
  error,
  isStreaming,
  previousRuns,
  auditLogs,
  streamingLogs,
  results,
  currentRunId,
  selectedRunId,
  setSelectedRunId,
  forecastTab,
  setForecastTab,
  logsDisplayLimit,
  setLogsDisplayLimit,
  handleForecast,
  loadForecastRun,
  CollapsibleJsonTile,
}) => {
  const clearCurrentRun = () => {
    setSelectedRunId(null);
    setForecastTab('logs');
    if (currentRunId) {
      loadForecastRun(currentRunId);
    }
  };

  const selectPreviousRun = () => {
    if (previousRuns.length > 0 && !selectedRunId) {
      loadForecastRun(previousRuns[0].run_id);
    }
  };

  const currentLogs = auditLogs.length > 0 ? auditLogs : streamingLogs;
  const totalLogs = currentLogs.length;

  return (
    <div className="forecast-modern">
      {/* Hero Banner */}
      <div className="forecast-hero-banner">
        <div className="hero-pattern"></div>
        <div className="hero-content-wrapper">
          <div className="hero-left">
            <h1 className="hero-main-title">
              <span className="hero-emoji">🎯</span>
              Demand Forecasting Engine
            </h1>
            <p className="hero-description">
              Powered by Multi-Agent AI System with Ollama LLaMA3
            </p>
          </div>
          <div className="hero-stats-row">
            <div className="hero-stat-card">
              <div className="stat-value">{previousRuns.length}</div>
              <div className="stat-label">Total Runs</div>
            </div>
            <div className="hero-stat-card status">
              <div className="stat-value">
                {loading ? <span className="status-active">Active</span> : <span className="status-ready">Ready</span>}
              </div>
              <div className="stat-label">System Status</div>
            </div>
            <div className="hero-stat-card">
              <div className="stat-value">{totalLogs}</div>
              <div className="stat-label">Current Logs</div>
            </div>
          </div>
        </div>
      </div>

      {/* Main Content Area */}
      <div className="forecast-main-layout">
        {/* Configuration Panel */}
        <div className="config-panel-modern">
          <div className="panel-header-modern">
            <h2>
              <span className="header-icon">⚙️</span>
              Configuration Parameters
            </h2>
          </div>

          <div className="config-content">
            {/* Margin Target */}
            <div className="config-group">
              <div className="config-header">
                <label className="config-title">Margin Target</label>
                <span className="config-display">{params.margin_target}%</span>
              </div>
              <div className="modern-range-wrapper">
                <input
                  type="range"
                  min="0"
                  max="100"
                  step="1"
                  value={params.margin_target}
                  onChange={(e) => setParams({ ...params, margin_target: parseFloat(e.target.value) })}
                  className="modern-range"
                />
                <div className="range-track">
                  <div className="range-progress" style={{ width: `${params.margin_target}%` }}></div>
                </div>
                <div className="range-labels">
                  <span>0%</span>
                  <span>50%</span>
                  <span>100%</span>
                </div>
              </div>
            </div>

            {/* Variance Threshold */}
            <div className="config-group">
              <div className="config-header">
                <label className="config-title">Variance Threshold</label>
                <span className="config-display">{params.variance_threshold}%</span>
              </div>
              <div className="modern-range-wrapper">
                <input
                  type="range"
                  min="0"
                  max="100"
                  step="1"
                  value={params.variance_threshold}
                  onChange={(e) => setParams({ ...params, variance_threshold: parseFloat(e.target.value) })}
                  className="modern-range"
                />
                <div className="range-track">
                  <div className="range-progress" style={{ width: `${params.variance_threshold}%` }}></div>
                </div>
                <div className="range-labels">
                  <span>0%</span>
                  <span>50%</span>
                  <span>100%</span>
                </div>
              </div>
            </div>

            {/* Forecast Horizon */}
            <div className="config-group">
              <div className="config-header">
                <label className="config-title">Forecast Horizon</label>
                <span className="config-display">{params.forecast_horizon_days} Days</span>
              </div>
              <div className="modern-range-wrapper">
                <input
                  type="range"
                  min="1"
                  max="180"
                  step="1"
                  value={params.forecast_horizon_days}
                  onChange={(e) => setParams({ ...params, forecast_horizon_days: parseInt(e.target.value) })}
                  className="modern-range"
                />
                <div className="range-track">
                  <div className="range-progress" style={{ width: `${(params.forecast_horizon_days / 180) * 100}%` }}></div>
                </div>
                <div className="range-labels">
                  <span>1</span>
                  <span>90</span>
                  <span>180</span>
                </div>
              </div>
            </div>

            {/* Max Quantity */}
            <div className="config-group">
              <div className="config-header">
                <label className="config-title">Max Qty per Store</label>
                <span className="config-display">{params.max_quantity_per_store}</span>
              </div>
              <div className="modern-range-wrapper">
                <input
                  type="range"
                  min="1"
                  max="2000"
                  step="10"
                  value={params.max_quantity_per_store}
                  onChange={(e) => setParams({ ...params, max_quantity_per_store: parseInt(e.target.value) })}
                  className="modern-range"
                />
                <div className="range-track">
                  <div className="range-progress" style={{ width: `${(params.max_quantity_per_store / 2000) * 100}%` }}></div>
                </div>
                <div className="range-labels">
                  <span>1</span>
                  <span>1000</span>
                  <span>2000</span>
                </div>
              </div>
            </div>

            {/* Universe of Stores */}
            <div className="config-group">
              <label className="config-title">Universe of Stores</label>
              <input
                type="text"
                className="modern-input"
                value={params.universe_of_stores}
                onChange={(e) => setParams({ ...params, universe_of_stores: e.target.value })}
                placeholder="Optional - e.g., 20"
              />
            </div>

            {/* Action Button */}
            <button
              className={`modern-action-btn ${loading ? 'loading' : ''}`}
              onClick={handleForecast}
              disabled={loading}
            >
              {loading ? (
                <>
                  <span className="btn-spinner"></span>
                  <span>Processing Forecast...</span>
                </>
              ) : (
                <>
                  <span className="btn-icon">🚀</span>
                  <span>Run Demand Forecast</span>
                </>
              )}
            </button>

            {error && (
              <div className="modern-error">
                <span className="error-icon">⚠️</span>
                {error}
              </div>
            )}

            {isStreaming && (
              <div className="streaming-status">
                <span className="stream-dot"></span>
                Real-time streaming active...
              </div>
            )}
          </div>
        </div>

        {/* Results Panel */}
        <div className="results-panel-modern">
          {/* Tab Navigation */}
          <div className="results-nav">
            <div className="nav-tabs">
              <button
                className={`nav-tab ${selectedRunId === null ? 'active' : ''}`}
                onClick={clearCurrentRun}
              >
                <span className="tab-icon">📊</span>
                <span className="tab-text">Current Run</span>
                {currentRunId && <span className="tab-badge live"></span>}
              </button>
              <button
                className={`nav-tab ${selectedRunId !== null ? 'active' : ''}`}
                onClick={selectPreviousRun}
              >
                <span className="tab-icon">📂</span>
                <span className="tab-text">History</span>
                {previousRuns.length > 0 && <span className="tab-counter">{previousRuns.length}</span>}
              </button>
            </div>
          </div>

          {/* Current Run View */}
          {selectedRunId === null && (
            <div className="results-content">
              {/* Sub-tabs */}
              <div className="content-tabs">
                <button
                  className={`content-tab ${forecastTab === 'summary' ? 'active' : ''}`}
                  onClick={() => setForecastTab('summary')}
                >
                  Dashboard
                </button>
                <button
                  className={`content-tab ${forecastTab === 'logs' ? 'active' : ''}`}
                  onClick={() => setForecastTab('logs')}
                >
                  Audit Logs
                  {totalLogs > 0 && <span className="content-badge">{totalLogs}</span>}
                </button>
              </div>

              {/* Dashboard View */}
              {forecastTab === 'summary' && (
                <div className="dashboard-modern">
                  {results ? (
                    <>
                      {/* KPI Grid */}
                      <div className="kpi-grid-modern">
                        <div className="kpi-tile gradient-purple">
                          <div className="kpi-icon-wrap">
                            <span className="kpi-icon">📦</span>
                          </div>
                          <div className="kpi-data">
                            <div className="kpi-number">{results.top_banner?.total_unique_skus || 0}</div>
                            <div className="kpi-label">Unique SKUs</div>
                          </div>
                        </div>

                        <div className="kpi-tile gradient-blue">
                          <div className="kpi-icon-wrap">
                            <span className="kpi-icon">📊</span>
                          </div>
                          <div className="kpi-data">
                            <div className="kpi-number">{results.top_banner?.total_quantity_bought || 0}</div>
                            <div className="kpi-label">Total Quantity</div>
                          </div>
                        </div>

                        <div className="kpi-tile gradient-green">
                          <div className="kpi-icon-wrap">
                            <span className="kpi-icon">🏪</span>
                          </div>
                          <div className="kpi-data">
                            <div className="kpi-number">{results.top_banner?.total_stores || 0}</div>
                            <div className="kpi-label">Stores</div>
                          </div>
                        </div>

                        <div className="kpi-tile gradient-orange">
                          <div className="kpi-icon-wrap">
                            <span className="kpi-icon">💰</span>
                          </div>
                          <div className="kpi-data">
                            <div className="kpi-number">₹{(results.top_banner?.total_buy_cost || 0).toLocaleString()}</div>
                            <div className="kpi-label">Investment</div>
                          </div>
                        </div>

                        <div className="kpi-tile gradient-pink">
                          <div className="kpi-icon-wrap">
                            <span className="kpi-icon">💵</span>
                          </div>
                          <div className="kpi-data">
                            <div className="kpi-number">₹{(results.top_banner?.total_sales_value || 0).toLocaleString()}</div>
                            <div className="kpi-label">Revenue</div>
                          </div>
                        </div>

                        <div className="kpi-tile gradient-teal">
                          <div className="kpi-icon-wrap">
                            <span className="kpi-icon">📈</span>
                          </div>
                          <div className="kpi-data">
                            <div className="kpi-number">
                              {((results.top_banner?.average_margin_achieved || 0) * 100).toFixed(1)}%
                            </div>
                            <div className="kpi-label">Margin</div>
                            <div className="kpi-sub">Target: {((results.top_banner?.target_margin || params.margin_target/100) * 100).toFixed(1)}%</div>
                          </div>
                        </div>
                      </div>

                      {/* Recommendations - User Friendly View */}
                      <div className="recommendations-section">
                        {results.recommendations && typeof results.recommendations === 'object' && (
                          <>
                            {/* Executive Summary */}
                            {results.recommendations.optimization_summary && (
                              <div className="exec-summary-card">
                                <div className="exec-header">
                                  <span className="exec-icon">📋</span>
                                  <h3>Executive Summary</h3>
                                </div>
                                <div className="exec-body">
                                  {results.recommendations.optimization_summary.split('\n').map((line: string, idx: number) => {
                                    if (line.includes('Article:')) {
                                      return <div key={idx} className="exec-article-header">{line}</div>;
                                    } else if (line.includes('⚠️')) {
                                      return <div key={idx} className="exec-warning">{line}</div>;
                                    } else if (line.includes(':') && !line.startsWith(' ')) {
                                      return <div key={idx} className="exec-metric">{line}</div>;
                                    } else if (line.trim()) {
                                      return <div key={idx} className="exec-line">{line}</div>;
                                    }
                                    return null;
                                  })}
                                </div>
                              </div>
                            )}

                            {/* Risk Assessment */}
                            {results.recommendations.risk_assessment && (
                              <div className="risk-assessment-card">
                                <div className="risk-header">
                                  <span className="risk-icon">⚠️</span>
                                  <h3>Risk Assessment</h3>
                                </div>
                                <div className="risk-badges">
                                  {results.recommendations.risk_assessment.low_sell_through_articles?.length > 0 && (
                                    <div className="risk-badge warning">
                                      <span className="risk-badge-icon">📉</span>
                                      <span className="risk-badge-text">
                                        {results.recommendations.risk_assessment.low_sell_through_articles.length} articles with low sell-through risk
                                      </span>
                                    </div>
                                  )}
                                  {results.recommendations.risk_assessment.margin_risk_articles?.length > 0 && (
                                    <div className="risk-badge danger">
                                      <span className="risk-badge-icon">💰</span>
                                      <span className="risk-badge-text">
                                        {results.recommendations.risk_assessment.margin_risk_articles.length} articles with margin risk
                                      </span>
                                    </div>
                                  )}
                                  {results.recommendations.risk_assessment.high_confidence_articles?.length > 0 && (
                                    <div className="risk-badge success">
                                      <span className="risk-badge-icon">✅</span>
                                      <span className="risk-badge-text">
                                        {results.recommendations.risk_assessment.high_confidence_articles.length} high confidence articles
                                      </span>
                                    </div>
                                  )}
                                  {(!results.recommendations.risk_assessment.low_sell_through_articles?.length && 
                                    !results.recommendations.risk_assessment.margin_risk_articles?.length) && (
                                    <div className="risk-badge success">
                                      <span className="risk-badge-icon">✅</span>
                                      <span className="risk-badge-text">No significant risks identified</span>
                                    </div>
                                  )}
                                </div>
                              </div>
                            )}

                            {/* Articles to Buy */}
                            {results.recommendations.articles_to_buy && (
                              <div className="articles-section">
                                <div className="section-header">
                                  <h3>🛍️ Recommended Articles ({results.recommendations.articles_to_buy.length})</h3>
                                </div>
                                <div className="articles-grid">
                                  {results.recommendations.articles_to_buy.map((sku: string) => {
                                    const metrics = results.recommendations.article_level_metrics?.[sku];
                                    const allocations = results.recommendations.store_allocations?.[sku];
                                    
                                    return (
                                      <div key={sku} className="article-card">
                                        <div className="article-header">
                                          <span className="article-sku">{sku}</span>
                                          {metrics?.margin_meets_target && (
                                            <span className="article-status success">✓ Meets Target</span>
                                          )}
                                        </div>
                                        
                                        {metrics?.article_details && (
                                          <div className="article-details">
                                            <div className="detail-row">
                                              <span className="detail-label">Style</span>
                                              <span className="detail-value">{metrics.article_details.style_code}</span>
                                            </div>
                                            <div className="detail-row">
                                              <span className="detail-label">Color</span>
                                              <span className="detail-value">{metrics.article_details.color}</span>
                                            </div>
                                            <div className="detail-row">
                                              <span className="detail-label">Segment</span>
                                              <span className="detail-value">{metrics.article_details.segment}</span>
                                            </div>
                                            <div className="detail-row">
                                              <span className="detail-label">Category</span>
                                              <span className="detail-value">{metrics.article_details.family} / {metrics.article_details.brick}</span>
                                            </div>
                                          </div>
                                        )}

                                        {metrics && (
                                          <div className="article-metrics">
                                            <div className="metric-item">
                                              <span className="metric-label">MRP</span>
                                              <span className="metric-value">₹{metrics.mrp?.toFixed(0) || 0}</span>
                                            </div>
                                            <div className="metric-item">
                                              <span className="metric-label">Quantity</span>
                                              <span className="metric-value highlight">{metrics.total_quantity?.toFixed(0) || 0}</span>
                                            </div>
                                            <div className="metric-item">
                                              <span className="metric-label">Stores</span>
                                              <span className="metric-value">{metrics.total_store_exposure || 0}</span>
                                            </div>
                                            <div className="metric-item">
                                              <span className="metric-label">Margin</span>
                                              <span className="metric-value success">{((metrics.margin_pct || 0) * 100).toFixed(0)}%</span>
                                            </div>
                                            <div className="metric-item">
                                              <span className="metric-label">ROS</span>
                                              <span className="metric-value">{metrics.ros?.toFixed(2) || 0}/day</span>
                                            </div>
                                            <div className="metric-item">
                                              <span className="metric-label">Net Sales</span>
                                              <span className="metric-value">₹{metrics.net_sales_value?.toLocaleString(undefined, {maximumFractionDigits: 0}) || 0}</span>
                                            </div>
                                          </div>
                                        )}

                                        {/* Store Allocations Table */}
                                        {allocations && Object.keys(allocations).length > 0 && (
                                          <div className="store-allocations">
                                            <div className="allocations-header">Store Allocations</div>
                                            <table className="allocations-table">
                                              <thead>
                                                <tr>
                                                  <th>Store</th>
                                                  <th>Qty</th>
                                                  <th>Sell-Thru</th>
                                                  <th>ROS</th>
                                                  <th>Action</th>
                                                </tr>
                                              </thead>
                                              <tbody>
                                                {Object.entries(allocations).map(([store, data]: [string, any]) => (
                                                  <tr key={store}>
                                                    <td className="store-name">{store}</td>
                                                    <td className="qty-cell">{data.quantity?.toFixed(0) || 0}</td>
                                                    <td>
                                                      <div className="mini-progress">
                                                        <div 
                                                          className="mini-progress-bar"
                                                          style={{ width: `${(data.expected_sell_through || 0) * 100}%` }}
                                                        ></div>
                                                        <span>{((data.expected_sell_through || 0) * 100).toFixed(0)}%</span>
                                                      </div>
                                                    </td>
                                                    <td>{data.expected_rate_of_sale?.toFixed(2) || 0}</td>
                                                    <td>
                                                      <span className={`action-badge ${data.recommendation?.includes('buy') ? 'buy' : 'skip'}`}>
                                                        {data.recommendation?.replace('_', ' ') || 'N/A'}
                                                      </span>
                                                    </td>
                                                  </tr>
                                                ))}
                                              </tbody>
                                            </table>
                                          </div>
                                        )}
                                      </div>
                                    );
                                  })}
                                </div>
                              </div>
                            )}

                            {/* Priority Stores */}
                            {results.recommendations.priority_stores && results.recommendations.priority_stores.length > 0 && (
                              <div className="priority-stores-card">
                                <div className="priority-header">
                                  <span className="priority-icon">🏪</span>
                                  <h3>Priority Stores</h3>
                                </div>
                                <div className="priority-list">
                                  {results.recommendations.priority_stores.map((store: any[], idx: number) => (
                                    <div key={idx} className="priority-item">
                                      <span className="priority-rank">#{idx + 1}</span>
                                      <span className="priority-store-name">{store[0]}</span>
                                      <div className="priority-score-bar">
                                        <div 
                                          className="priority-score-fill"
                                          style={{ width: `${(store[1] || 0) * 100}%` }}
                                        ></div>
                                      </div>
                                      <span className="priority-score">{((store[1] || 0) * 100).toFixed(0)}%</span>
                                      <span className="priority-qty">{store[2]?.toFixed(0) || 0} units</span>
                                    </div>
                                  ))}
                                </div>
                              </div>
                            )}

                            {/* Expected Metrics Summary */}
                            {results.recommendations.expected_metrics && (
                              <div className="expected-metrics-card">
                                <div className="metrics-header">
                                  <span className="metrics-icon">📈</span>
                                  <h3>Expected Performance</h3>
                                </div>
                                <div className="metrics-grid">
                                  <div className="metric-box">
                                    <div className="metric-box-value">
                                      ₹{results.recommendations.expected_metrics.total_revenue?.toLocaleString(undefined, {maximumFractionDigits: 0}) || 0}
                                    </div>
                                    <div className="metric-box-label">Total Revenue</div>
                                  </div>
                                  <div className="metric-box">
                                    <div className="metric-box-value">
                                      ₹{results.recommendations.expected_metrics.total_margin_value?.toLocaleString(undefined, {maximumFractionDigits: 0}) || 0}
                                    </div>
                                    <div className="metric-box-label">Margin Value</div>
                                  </div>
                                  <div className="metric-box">
                                    <div className="metric-box-value">
                                      {((results.recommendations.expected_metrics.avg_sell_through_rate || 0) * 100).toFixed(1)}%
                                    </div>
                                    <div className="metric-box-label">Sell-Through Rate</div>
                                  </div>
                                  <div className="metric-box">
                                    <div className="metric-box-value">
                                      {((results.recommendations.expected_metrics.avg_margin_pct || 0) * 100).toFixed(0)}%
                                    </div>
                                    <div className="metric-box-label">Avg Margin</div>
                                  </div>
                                  <div className="metric-box">
                                    <div className="metric-box-value">
                                      {results.recommendations.expected_metrics.total_expected_units_sold?.toFixed(0) || 0}
                                    </div>
                                    <div className="metric-box-label">Expected Units Sold</div>
                                  </div>
                                  <div className="metric-box">
                                    <div className="metric-box-value">
                                      {(results.recommendations.expected_metrics.avg_optimization_score * 100)?.toFixed(0) || 0}%
                                    </div>
                                    <div className="metric-box-label">Optimization Score</div>
                                  </div>
                                </div>
                              </div>
                            )}
                          </>
                        )}
                      </div>
                    </>
                  ) : (
                    <div className="empty-dashboard">
                      <div className="empty-icon-large">📊</div>
                      <h3>No Forecast Data Available</h3>
                      <p>{loading ? 'Forecast is being processed...' : 'Configure and run a forecast to see results'}</p>
                    </div>
                  )}
                </div>
              )}

              {/* AI Decision Journey View */}
              {forecastTab === 'logs' && (
                <div className="ai-journey-modern">
                  {totalLogs > 0 ? (
                    <>
                      <div className="journey-header">
                        <h3>
                          <span className="journey-icon">🤖</span>
                          AI Decision Journey
                        </h3>
                        <span className="journey-counter">
                          {totalLogs} decision points analyzed
                        </span>
                      </div>

                      {/* Timeline View */}
                      <div className="ai-timeline">
                        {currentLogs.slice(0, logsDisplayLimit).map((log: any, idx: number) => {
                          const agentName = log.agent_name || 'System';
                          const isAttributeAgent = agentName.includes('Attribute');
                          const isDemandAgent = agentName.includes('Demand');
                          const isOrchestrator = agentName.includes('Orchestrator');
                          
                          return (
                            <div key={`timeline-${idx}`} className={`timeline-item ${idx % 2 === 0 ? 'left' : 'right'}`}>
                              <div className="timeline-marker">
                                <span className="marker-icon">
                                  {isAttributeAgent ? '🔍' : isDemandAgent ? '📊' : isOrchestrator ? '🎯' : '⚙️'}
                                </span>
                              </div>
                              <div className="timeline-content">
                                <div className="timeline-header">
                                  <span className="timeline-agent">{agentName}</span>
                                  <span className="timeline-time">
                                    {new Date(log.date_time || Date.now()).toLocaleTimeString()}
                                  </span>
                                </div>
                                <div className="timeline-body">
                                  {log.description && (
                                    <p className="timeline-description">{log.description}</p>
                                  )}
                                  
                                  {/* Extract key insights from log data */}
                                  {log.input && (
                                    <div className="insight-card">
                                      <span className="insight-label">📥 Input Analysis</span>
                                      <div className="insight-content">
                                        {typeof log.input === 'object' ? (
                                          <div className="insight-details">
                                            {log.input.sku && <span className="detail-chip">SKU: {log.input.sku}</span>}
                                            {log.input.quantity && <span className="detail-chip">Qty: {log.input.quantity}</span>}
                                            {log.input.attributes && <span className="detail-chip">Attributes: {Object.keys(log.input.attributes).length}</span>}
                                          </div>
                                        ) : (
                                          <span>{String(log.input).substring(0, 100)}...</span>
                                        )}
                                      </div>
                                    </div>
                                  )}
                                  
                                  {log.output && (
                                    <div className="insight-card output">
                                      <span className="insight-label">✨ AI Decision</span>
                                      <div className="insight-content">
                                        {typeof log.output === 'object' ? (
                                          <div className="insight-details">
                                            {log.output.recommendation && (
                                              <div className="decision-highlight">
                                                💡 {log.output.recommendation}
                                              </div>
                                            )}
                                            {log.output.confidence && (
                                              <div className="confidence-indicator">
                                                <div className="confidence-bar-mini">
                                                  <div 
                                                    className="confidence-fill-mini"
                                                    style={{ width: `${log.output.confidence}%` }}
                                                  ></div>
                                                </div>
                                                <span className="confidence-value">{log.output.confidence}% confident</span>
                                              </div>
                                            )}
                                            {log.output.similar_products && (
                                              <div className="similar-products">
                                                <span className="similar-label">Similar Items Found:</span>
                                                <div className="similar-chips">
                                                  {log.output.similar_products.slice(0, 3).map((prod: any, pidx: number) => (
                                                    <span key={pidx} className="similar-chip">
                                                      {prod.sku || prod}
                                                    </span>
                                                  ))}
                                                </div>
                                              </div>
                                            )}
                                          </div>
                                        ) : (
                                          <span>{String(log.output).substring(0, 100)}...</span>
                                        )}
                                      </div>
                                    </div>
                                  )}

                                  {/* Performance Metrics */}
                                  {log.metrics && (
                                    <div className="metrics-row">
                                      {log.metrics.processing_time && (
                                        <span className="metric-badge">
                                          ⏱️ {log.metrics.processing_time}ms
                                        </span>
                                      )}
                                      {log.metrics.accuracy && (
                                        <span className="metric-badge">
                                          🎯 {log.metrics.accuracy}% accurate
                                        </span>
                                      )}
                                      {log.metrics.items_processed && (
                                        <span className="metric-badge">
                                          📦 {log.metrics.items_processed} items
                                        </span>
                                      )}
                                    </div>
                                  )}
                                </div>
                                
                                {/* Expandable Raw Data */}
                                <details className="timeline-raw-data">
                                  <summary>View Complete Data</summary>
                                  <pre className="raw-data-content">
                                    {JSON.stringify(log, null, 2)}
                                  </pre>
                                </details>
                              </div>
                            </div>
                          );
                        })}
                      </div>

                      {/* Agent Summary Cards */}
                      <div className="agents-summary">
                        <h4>AI Agents Activity Summary</h4>
                        <div className="agent-cards-grid">
                          {(() => {
                            const agentStats: any = {};
                            currentLogs.forEach((log: any) => {
                              const agent = log.agent_name || 'System';
                              if (!agentStats[agent]) {
                                agentStats[agent] = { count: 0, lastActive: log.date_time };
                              }
                              agentStats[agent].count++;
                            });
                            
                            return Object.entries(agentStats).map(([agent, stats]: [string, any]) => (
                              <div key={agent} className="agent-stat-card">
                                <div className="agent-stat-icon">
                                  {agent.includes('Attribute') ? '🔍' : 
                                   agent.includes('Demand') ? '📊' : 
                                   agent.includes('Orchestrator') ? '🎯' : '⚙️'}
                                </div>
                                <div className="agent-stat-content">
                                  <span className="agent-name">{agent}</span>
                                  <span className="agent-actions">{stats.count} actions</span>
                                  <span className="agent-last-active">
                                    Last: {new Date(stats.lastActive).toLocaleTimeString()}
                                  </span>
                                </div>
                              </div>
                            ));
                          })()}
                        </div>
                      </div>

                      {totalLogs > logsDisplayLimit && (
                        <div className="journey-pagination">
                          <button
                            onClick={() => setLogsDisplayLimit(prev => Math.min(prev + 10, totalLogs))}
                            className="journey-btn primary"
                          >
                            Show More Journey Steps
                          </button>
                          <button
                            onClick={() => setLogsDisplayLimit(totalLogs)}
                            className="journey-btn secondary"
                          >
                            View Complete Journey
                          </button>
                        </div>
                      )}
                    </>
                  ) : (
                    <div className="empty-dashboard">
                      <div className="empty-icon-large">🤖</div>
                      <h3>No AI Journey Yet</h3>
                      <p>{loading ? 'AI is processing...' : 'Run a forecast to see the AI decision-making process'}</p>
                    </div>
                  )}
                </div>
              )}
            </div>
          )}

          {/* History View */}
          {selectedRunId !== null && (
            <div className="results-content">
              <div className="history-header-bar">
                <button 
                  onClick={() => setSelectedRunId(null)}
                  className="back-btn-modern"
                >
                  ← Back to Current Run
                </button>
                <h3>Historical Run: {selectedRunId}</h3>
              </div>

              {/* Sub-tabs for historical run */}
              <div className="content-tabs">
                <button
                  className={`content-tab ${forecastTab === 'summary' ? 'active' : ''}`}
                  onClick={() => setForecastTab('summary')}
                >
                  Dashboard
                </button>
                <button
                  className={`content-tab ${forecastTab === 'logs' ? 'active' : ''}`}
                  onClick={() => setForecastTab('logs')}
                >
                  Audit Logs
                  {totalLogs > 0 && <span className="content-badge">{totalLogs}</span>}
                </button>
              </div>

              {/* Historical Dashboard View */}
              {forecastTab === 'summary' && (
                <div className="dashboard-modern">
                  {results ? (
                    <>
                      {/* KPI Grid */}
                      <div className="kpi-grid-modern">
                        <div className="kpi-tile gradient-purple">
                          <div className="kpi-icon-wrap">
                            <span className="kpi-icon">📦</span>
                          </div>
                          <div className="kpi-data">
                            <div className="kpi-number">{results.top_banner?.total_unique_skus || 0}</div>
                            <div className="kpi-label">Unique SKUs</div>
                          </div>
                        </div>

                        <div className="kpi-tile gradient-blue">
                          <div className="kpi-icon-wrap">
                            <span className="kpi-icon">📊</span>
                          </div>
                          <div className="kpi-data">
                            <div className="kpi-number">{results.top_banner?.total_quantity_bought || 0}</div>
                            <div className="kpi-label">Total Quantity</div>
                          </div>
                        </div>

                        <div className="kpi-tile gradient-green">
                          <div className="kpi-icon-wrap">
                            <span className="kpi-icon">🏪</span>
                          </div>
                          <div className="kpi-data">
                            <div className="kpi-number">{results.top_banner?.total_stores || 0}</div>
                            <div className="kpi-label">Stores</div>
                          </div>
                        </div>

                        <div className="kpi-tile gradient-orange">
                          <div className="kpi-icon-wrap">
                            <span className="kpi-icon">💰</span>
                          </div>
                          <div className="kpi-data">
                            <div className="kpi-number">₹{(results.top_banner?.total_buy_cost || 0).toLocaleString()}</div>
                            <div className="kpi-label">Investment</div>
                          </div>
                        </div>

                        <div className="kpi-tile gradient-pink">
                          <div className="kpi-icon-wrap">
                            <span className="kpi-icon">💵</span>
                          </div>
                          <div className="kpi-data">
                            <div className="kpi-number">₹{(results.top_banner?.total_sales_value || 0).toLocaleString()}</div>
                            <div className="kpi-label">Revenue</div>
                          </div>
                        </div>

                        <div className="kpi-tile gradient-teal">
                          <div className="kpi-icon-wrap">
                            <span className="kpi-icon">📈</span>
                          </div>
                          <div className="kpi-data">
                            <div className="kpi-number">
                              {((results.top_banner?.average_margin_achieved || 0) * 100).toFixed(1)}%
                            </div>
                            <div className="kpi-label">Margin</div>
                            <div className="kpi-sub">Target: {((results.top_banner?.target_margin || params.margin_target/100) * 100).toFixed(1)}%</div>
                          </div>
                        </div>
                      </div>

                      {/* Recommendations - User Friendly View */}
                      <div className="recommendations-section">
                        {results.recommendations && typeof results.recommendations === 'object' && (
                          <>
                            {/* Executive Summary */}
                            {results.recommendations.optimization_summary && (
                              <div className="exec-summary-card">
                                <div className="exec-header">
                                  <span className="exec-icon">📋</span>
                                  <h3>Executive Summary</h3>
                                </div>
                                <div className="exec-body">
                                  {results.recommendations.optimization_summary.split('\n').map((line: string, idx: number) => {
                                    if (line.includes('Article:')) {
                                      return <div key={idx} className="exec-article-header">{line}</div>;
                                    } else if (line.includes('⚠️')) {
                                      return <div key={idx} className="exec-warning">{line}</div>;
                                    } else if (line.includes(':') && !line.startsWith(' ')) {
                                      return <div key={idx} className="exec-metric">{line}</div>;
                                    } else if (line.trim()) {
                                      return <div key={idx} className="exec-line">{line}</div>;
                                    }
                                    return null;
                                  })}
                                </div>
                              </div>
                            )}

                            {/* Risk Assessment */}
                            {results.recommendations.risk_assessment && (
                              <div className="risk-assessment-card">
                                <div className="risk-header">
                                  <span className="risk-icon">⚠️</span>
                                  <h3>Risk Assessment</h3>
                                </div>
                                <div className="risk-badges">
                                  {results.recommendations.risk_assessment.low_sell_through_articles?.length > 0 && (
                                    <div className="risk-badge warning">
                                      <span className="risk-badge-icon">📉</span>
                                      <span className="risk-badge-text">
                                        {results.recommendations.risk_assessment.low_sell_through_articles.length} articles with low sell-through risk
                                      </span>
                                    </div>
                                  )}
                                  {results.recommendations.risk_assessment.margin_risk_articles?.length > 0 && (
                                    <div className="risk-badge danger">
                                      <span className="risk-badge-icon">💰</span>
                                      <span className="risk-badge-text">
                                        {results.recommendations.risk_assessment.margin_risk_articles.length} articles with margin risk
                                      </span>
                                    </div>
                                  )}
                                  {results.recommendations.risk_assessment.high_confidence_articles?.length > 0 && (
                                    <div className="risk-badge success">
                                      <span className="risk-badge-icon">✅</span>
                                      <span className="risk-badge-text">
                                        {results.recommendations.risk_assessment.high_confidence_articles.length} high confidence articles
                                      </span>
                                    </div>
                                  )}
                                  {(!results.recommendations.risk_assessment.low_sell_through_articles?.length && 
                                    !results.recommendations.risk_assessment.margin_risk_articles?.length) && (
                                    <div className="risk-badge success">
                                      <span className="risk-badge-icon">✅</span>
                                      <span className="risk-badge-text">No significant risks identified</span>
                                    </div>
                                  )}
                                </div>
                              </div>
                            )}

                            {/* Articles to Buy */}
                            {results.recommendations.articles_to_buy && (
                              <div className="articles-section">
                                <div className="section-header">
                                  <h3>🛍️ Recommended Articles ({results.recommendations.articles_to_buy.length})</h3>
                                </div>
                                <div className="articles-grid">
                                  {results.recommendations.articles_to_buy.map((sku: string) => {
                                    const metrics = results.recommendations.article_level_metrics?.[sku];
                                    const allocations = results.recommendations.store_allocations?.[sku];
                                    
                                    return (
                                      <div key={sku} className="article-card">
                                        <div className="article-header">
                                          <span className="article-sku">{sku}</span>
                                          {metrics?.margin_meets_target && (
                                            <span className="article-status success">✓ Meets Target</span>
                                          )}
                                        </div>
                                        
                                        {metrics?.article_details && (
                                          <div className="article-details">
                                            <div className="detail-row">
                                              <span className="detail-label">Style</span>
                                              <span className="detail-value">{metrics.article_details.style_code}</span>
                                            </div>
                                            <div className="detail-row">
                                              <span className="detail-label">Color</span>
                                              <span className="detail-value">{metrics.article_details.color}</span>
                                            </div>
                                            <div className="detail-row">
                                              <span className="detail-label">Segment</span>
                                              <span className="detail-value">{metrics.article_details.segment}</span>
                                            </div>
                                            <div className="detail-row">
                                              <span className="detail-label">Category</span>
                                              <span className="detail-value">{metrics.article_details.family} / {metrics.article_details.brick}</span>
                                            </div>
                                          </div>
                                        )}

                                        {metrics && (
                                          <div className="article-metrics">
                                            <div className="metric-item">
                                              <span className="metric-label">MRP</span>
                                              <span className="metric-value">₹{metrics.mrp?.toFixed(0) || 0}</span>
                                            </div>
                                            <div className="metric-item">
                                              <span className="metric-label">Quantity</span>
                                              <span className="metric-value highlight">{metrics.total_quantity?.toFixed(0) || 0}</span>
                                            </div>
                                            <div className="metric-item">
                                              <span className="metric-label">Stores</span>
                                              <span className="metric-value">{metrics.total_store_exposure || 0}</span>
                                            </div>
                                            <div className="metric-item">
                                              <span className="metric-label">Margin</span>
                                              <span className="metric-value success">{((metrics.margin_pct || 0) * 100).toFixed(0)}%</span>
                                            </div>
                                            <div className="metric-item">
                                              <span className="metric-label">ROS</span>
                                              <span className="metric-value">{metrics.ros?.toFixed(2) || 0}/day</span>
                                            </div>
                                            <div className="metric-item">
                                              <span className="metric-label">Net Sales</span>
                                              <span className="metric-value">₹{metrics.net_sales_value?.toLocaleString(undefined, {maximumFractionDigits: 0}) || 0}</span>
                                            </div>
                                          </div>
                                        )}

                                        {/* Store Allocations Table */}
                                        {allocations && Object.keys(allocations).length > 0 && (
                                          <div className="store-allocations">
                                            <div className="allocations-header">Store Allocations</div>
                                            <table className="allocations-table">
                                              <thead>
                                                <tr>
                                                  <th>Store</th>
                                                  <th>Qty</th>
                                                  <th>Sell-Thru</th>
                                                  <th>ROS</th>
                                                  <th>Action</th>
                                                </tr>
                                              </thead>
                                              <tbody>
                                                {Object.entries(allocations).map(([store, data]: [string, any]) => (
                                                  <tr key={store}>
                                                    <td className="store-name">{store}</td>
                                                    <td className="qty-cell">{data.quantity?.toFixed(0) || 0}</td>
                                                    <td>
                                                      <div className="mini-progress">
                                                        <div 
                                                          className="mini-progress-bar"
                                                          style={{ width: `${(data.expected_sell_through || 0) * 100}%` }}
                                                        ></div>
                                                        <span>{((data.expected_sell_through || 0) * 100).toFixed(0)}%</span>
                                                      </div>
                                                    </td>
                                                    <td>{data.expected_rate_of_sale?.toFixed(2) || 0}</td>
                                                    <td>
                                                      <span className={`action-badge ${data.recommendation?.includes('buy') ? 'buy' : 'skip'}`}>
                                                        {data.recommendation?.replace('_', ' ') || 'N/A'}
                                                      </span>
                                                    </td>
                                                  </tr>
                                                ))}
                                              </tbody>
                                            </table>
                                          </div>
                                        )}
                                      </div>
                                    );
                                  })}
                                </div>
                              </div>
                            )}

                            {/* Priority Stores */}
                            {results.recommendations.priority_stores && results.recommendations.priority_stores.length > 0 && (
                              <div className="priority-stores-card">
                                <div className="priority-header">
                                  <span className="priority-icon">🏪</span>
                                  <h3>Priority Stores</h3>
                                </div>
                                <div className="priority-list">
                                  {results.recommendations.priority_stores.map((store: any[], idx: number) => (
                                    <div key={idx} className="priority-item">
                                      <span className="priority-rank">#{idx + 1}</span>
                                      <span className="priority-store-name">{store[0]}</span>
                                      <div className="priority-score-bar">
                                        <div 
                                          className="priority-score-fill"
                                          style={{ width: `${(store[1] || 0) * 100}%` }}
                                        ></div>
                                      </div>
                                      <span className="priority-score">{((store[1] || 0) * 100).toFixed(0)}%</span>
                                      <span className="priority-qty">{store[2]?.toFixed(0) || 0} units</span>
                                    </div>
                                  ))}
                                </div>
                              </div>
                            )}

                            {/* Expected Metrics Summary */}
                            {results.recommendations.expected_metrics && (
                              <div className="expected-metrics-card">
                                <div className="metrics-header">
                                  <span className="metrics-icon">📈</span>
                                  <h3>Expected Performance</h3>
                                </div>
                                <div className="metrics-grid">
                                  <div className="metric-box">
                                    <div className="metric-box-value">
                                      ₹{results.recommendations.expected_metrics.total_revenue?.toLocaleString(undefined, {maximumFractionDigits: 0}) || 0}
                                    </div>
                                    <div className="metric-box-label">Total Revenue</div>
                                  </div>
                                  <div className="metric-box">
                                    <div className="metric-box-value">
                                      ₹{results.recommendations.expected_metrics.total_margin_value?.toLocaleString(undefined, {maximumFractionDigits: 0}) || 0}
                                    </div>
                                    <div className="metric-box-label">Margin Value</div>
                                  </div>
                                  <div className="metric-box">
                                    <div className="metric-box-value">
                                      {((results.recommendations.expected_metrics.avg_sell_through_rate || 0) * 100).toFixed(1)}%
                                    </div>
                                    <div className="metric-box-label">Sell-Through Rate</div>
                                  </div>
                                  <div className="metric-box">
                                    <div className="metric-box-value">
                                      {((results.recommendations.expected_metrics.avg_margin_pct || 0) * 100).toFixed(0)}%
                                    </div>
                                    <div className="metric-box-label">Avg Margin</div>
                                  </div>
                                  <div className="metric-box">
                                    <div className="metric-box-value">
                                      {results.recommendations.expected_metrics.total_expected_units_sold?.toFixed(0) || 0}
                                    </div>
                                    <div className="metric-box-label">Expected Units Sold</div>
                                  </div>
                                  <div className="metric-box">
                                    <div className="metric-box-value">
                                      {(results.recommendations.expected_metrics.avg_optimization_score * 100)?.toFixed(0) || 0}%
                                    </div>
                                    <div className="metric-box-label">Optimization Score</div>
                                  </div>
                                </div>
                              </div>
                            )}
                          </>
                        )}
                      </div>
                    </>
                  ) : (
                    <div className="empty-dashboard">
                      <div className="empty-icon-large">📊</div>
                      <h3>No Results for This Run</h3>
                      <p>Unable to load results for the selected historical run</p>
                    </div>
                  )}
                </div>
              )}

              {/* Historical AI Journey View */}
              {forecastTab === 'logs' && (
                <div className="ai-journey-modern">
                  {totalLogs > 0 ? (
                    <>
                      <div className="journey-header">
                        <h3>
                          <span className="journey-icon">🤖</span>
                          Historical AI Decision Journey
                        </h3>
                        <span className="journey-counter">
                          {totalLogs} historical decision points
                        </span>
                      </div>

                      {/* Historical Timeline - Similar structure but styled differently */}
                      <div className="ai-timeline historical">
                        {currentLogs.slice(0, logsDisplayLimit).map((log: any, idx: number) => {
                          const agentName = log.agent_name || 'System';
                          const isAttributeAgent = agentName.includes('Attribute');
                          const isDemandAgent = agentName.includes('Demand');
                          const isOrchestrator = agentName.includes('Orchestrator');
                          
                          return (
                            <div key={`hist-timeline-${idx}`} className={`timeline-item ${idx % 2 === 0 ? 'left' : 'right'}`}>
                              <div className="timeline-marker">
                                <span className="marker-icon">
                                  {isAttributeAgent ? '🔍' : isDemandAgent ? '📊' : isOrchestrator ? '🎯' : '⚙️'}
                                </span>
                              </div>
                              <div className="timeline-content">
                                <div className="timeline-header">
                                  <span className="timeline-agent">{agentName}</span>
                                  <span className="timeline-time">
                                    {new Date(log.date_time || Date.now()).toLocaleString()}
                                  </span>
                                </div>
                                <div className="timeline-body">
                                  {log.description && (
                                    <p className="timeline-description">{log.description}</p>
                                  )}
                                  
                                  {/* Historical insights */}
                                  {log.input && (
                                    <div className="insight-card">
                                      <span className="insight-label">📥 Historical Input</span>
                                      <div className="insight-content">
                                        {typeof log.input === 'object' ? (
                                          <div className="insight-details">
                                            {log.input.sku && <span className="detail-chip">SKU: {log.input.sku}</span>}
                                            {log.input.quantity && <span className="detail-chip">Qty: {log.input.quantity}</span>}
                                          </div>
                                        ) : (
                                          <span>{String(log.input).substring(0, 100)}...</span>
                                        )}
                                      </div>
                                    </div>
                                  )}
                                  
                                  {log.output && (
                                    <div className="insight-card output">
                                      <span className="insight-label">✨ Historical Decision</span>
                                      <div className="insight-content">
                                        {typeof log.output === 'object' && log.output.recommendation && (
                                          <div className="decision-highlight">
                                            💡 {log.output.recommendation}
                                          </div>
                                        )}
                                      </div>
                                    </div>
                                  )}
                                </div>
                                
                                <details className="timeline-raw-data">
                                  <summary>View Complete Data</summary>
                                  <pre className="raw-data-content">
                                    {JSON.stringify(log, null, 2)}
                                  </pre>
                                </details>
                              </div>
                            </div>
                          );
                        })}
                      </div>

                      {totalLogs > logsDisplayLimit && (
                        <div className="journey-pagination">
                          <button
                            onClick={() => setLogsDisplayLimit(prev => Math.min(prev + 10, totalLogs))}
                            className="journey-btn primary"
                          >
                            Show More Historical Steps
                          </button>
                        </div>
                      )}
                    </>
                  ) : (
                    <div className="empty-dashboard">
                      <div className="empty-icon-large">🤖</div>
                      <h3>No Historical Journey</h3>
                      <p>No AI decision journey available for this run</p>
                    </div>
                  )}
                </div>
              )}

              {/* Previous Runs List */}
              <div className="history-section">
                <h4>Other Historical Runs</h4>
                <div className="history-grid-modern">
                  {previousRuns.map((run: any) => (
                    <div
                      key={run.run_id}
                      className={`history-card-modern ${selectedRunId === run.run_id ? 'selected' : ''}`}
                      onClick={() => loadForecastRun(run.run_id)}
                    >
                      <div className="history-card-header">
                        <span className="history-date">{new Date(run.timestamp).toLocaleDateString()}</span>
                        <span className="history-time">{new Date(run.timestamp).toLocaleTimeString()}</span>
                      </div>
                      <div className="history-stats">
                        <div className="history-stat">
                          <span className="stat-icon">📦</span>
                          <span>{run.summary?.total_skus || 0} SKUs</span>
                        </div>
                        <div className="history-stat">
                          <span className="stat-icon">📊</span>
                          <span>{run.summary?.total_quantity || 0} Qty</span>
                        </div>
                        <div className="history-stat">
                          <span className="stat-icon">🏪</span>
                          <span>{run.summary?.total_stores || 0} Stores</span>
                        </div>
                        <div className="history-stat">
                          <span className="stat-icon">📋</span>
                          <span>{run.log_count || 0} Logs</span>
                        </div>
                      </div>
                      <div className="history-id">ID: {run.run_id.slice(-8)}</div>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default ForecastPage;