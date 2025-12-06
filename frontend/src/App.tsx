import React, { useState } from 'react';
import axios from 'axios';
import './App.css';

const API_BASE_URL = 'http://localhost:8000';

interface UploadedFiles {
  sales_data?: string;
  inventory_data?: string;
  price_data?: string;
  cost_data?: string;
  new_articles_data?: string;
}

function App() {
  const [files, setFiles] = useState<UploadedFiles>({});
  const [params, setParams] = useState({
    margin_target: 30,
    variance_threshold: 5,
    forecast_horizon_days: 60,
    max_quantity_per_store: 500,
    universe_of_stores: '',
  });
  const [loading, setLoading] = useState(false);
  const [results, setResults] = useState<any>(null);
  const [error, setError] = useState<string | null>(null);
  const [auditLogs, setAuditLogs] = useState<any[]>([]);
  const [activeTab, setActiveTab] = useState<'forecast' | 'audit'>('forecast');

  const handleFileUpload = async (fieldName: string, file: File) => {
    const formData = new FormData();
    formData.append(fieldName, file);

    try {
      const response = await axios.post(`${API_BASE_URL}/api/upload`, formData, {
        headers: { 'Content-Type': 'multipart/form-data' },
      });

      if (response.data.success) {
        setFiles((prev) => ({
          ...prev,
          [fieldName]: response.data.files[fieldName],
        }));
      }
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Upload failed');
    }
  };

  const handleForecast = async () => {
    if (!files.sales_data || !files.new_articles_data) {
      setError('Please upload Sales Data and New Articles Data (required)');
      return;
    }

    setLoading(true);
    setError(null);

    try {
      const formData = new FormData();
      formData.append('file_paths', JSON.stringify(files));
      formData.append('margin_target', params.margin_target.toString());
      formData.append('variance_threshold', params.variance_threshold.toString());
      formData.append('forecast_horizon_days', params.forecast_horizon_days.toString());
      formData.append('max_quantity_per_store', params.max_quantity_per_store.toString());
      if (params.universe_of_stores) {
        formData.append('universe_of_stores', params.universe_of_stores);
      }

      const response = await axios.post(`${API_BASE_URL}/api/forecast`, formData, {
        headers: { 'Content-Type': 'multipart/form-data' },
      });

      if (response.data.success) {
        setResults(response.data.results);
        if (response.data.run_id) {
          loadAuditLogs(response.data.run_id);
        }
      }
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Forecast failed');
    } finally {
      setLoading(false);
    }
  };

  const loadAuditLogs = async (runId: string) => {
    try {
      const response = await axios.get(`${API_BASE_URL}/api/audit-logs/${runId}`);
      if (response.data.success) {
        setAuditLogs(response.data.logs);
      }
    } catch (err) {
      console.error('Failed to load audit logs', err);
    }
  };

  return (
    <div className="app">
      <header className="header">
        <h1>📊 Demand Forecasting & Allocation Engine</h1>
        <p className="subtitle">Multi-Agent LLM System using Ollama (llama3:8b)</p>
      </header>

      <div className="tabs">
        <button
          className={activeTab === 'forecast' ? 'active' : ''}
          onClick={() => setActiveTab('forecast')}
        >
          Forecast
        </button>
        <button
          className={activeTab === 'audit' ? 'active' : ''}
          onClick={() => setActiveTab('audit')}
        >
          Audit Logs
        </button>
      </div>

      {activeTab === 'forecast' && (
        <div className="main-content">
          <div className="left-panel">
            <section className="card">
              <h2>📤 Step 1: Upload Data</h2>
              <div className="upload-section">
                {[
                  { key: 'sales_data', label: 'Historical Sales Data', required: true },
                  { key: 'inventory_data', label: 'Inventory Data', required: false },
                  { key: 'price_data', label: 'Price Data', required: false },
                  { key: 'cost_data', label: 'Cost Data', required: false },
                  { key: 'new_articles_data', label: 'New Articles Data', required: true },
                ].map(({ key, label, required }) => (
                  <div key={key} className="upload-item">
                    <label>
                      {label} {required && <span className="required">*</span>}
                    </label>
                    <input
                      type="file"
                      accept=".csv,.xlsx,.xls"
                      onChange={(e) => {
                        const file = e.target.files?.[0];
                        if (file) handleFileUpload(key, file);
                      }}
                    />
                    {files[key as keyof UploadedFiles] && (
                      <small className="success">✓ Uploaded</small>
                    )}
                  </div>
                ))}
              </div>
            </section>

            <section className="card">
              <h2>⚙️ Step 2: Configure Parameters</h2>
              <div className="form-group">
                <label>Margin Target (%)</label>
                <input
                  type="number"
                  value={params.margin_target}
                  onChange={(e) =>
                    setParams({ ...params, margin_target: parseFloat(e.target.value) })
                  }
                  min="0"
                  max="100"
                />
              </div>
              <div className="form-group">
                <label>Variance Threshold (%)</label>
                <input
                  type="number"
                  value={params.variance_threshold}
                  onChange={(e) =>
                    setParams({ ...params, variance_threshold: parseFloat(e.target.value) })
                  }
                  min="0"
                  max="100"
                />
              </div>
              <div className="form-group">
                <label>Forecast Horizon (Days)</label>
                <input
                  type="number"
                  value={params.forecast_horizon_days}
                  onChange={(e) =>
                    setParams({ ...params, forecast_horizon_days: parseInt(e.target.value) })
                  }
                  min="1"
                />
              </div>
              <div className="form-group">
                <label>Max Quantity per Store</label>
                <input
                  type="number"
                  value={params.max_quantity_per_store}
                  onChange={(e) =>
                    setParams({ ...params, max_quantity_per_store: parseInt(e.target.value) })
                  }
                  min="1"
                />
              </div>
              <div className="form-group">
                <label>Universe of Stores (Optional)</label>
                <input
                  type="number"
                  value={params.universe_of_stores}
                  onChange={(e) =>
                    setParams({ ...params, universe_of_stores: e.target.value })
                  }
                  min="1"
                />
              </div>
            </section>

            <button
              className="btn btn-primary btn-large"
              onClick={handleForecast}
              disabled={loading}
            >
              {loading ? '⏳ Running Forecast...' : '▶️ Run Demand Forecast'}
            </button>

            {error && <div className="error-message">{error}</div>}
          </div>

          <div className="right-panel">
            {results && (
              <>
                {results.top_banner && (
                  <section className="card banner-card">
                    <h2>📊 Summary</h2>
                    <div className="banner-grid">
                      <div className="banner-item">
                        <div className="banner-item-label">Total Unique SKUs</div>
                        <div className="banner-item-value">
                          {results.top_banner.total_unique_skus || 0}
                        </div>
                      </div>
                      <div className="banner-item">
                        <div className="banner-item-label">Total Quantity</div>
                        <div className="banner-item-value">
                          {results.top_banner.total_quantity_bought || 0}
                        </div>
                      </div>
                      <div className="banner-item">
                        <div className="banner-item-label">Total Stores</div>
                        <div className="banner-item-value">
                          {results.top_banner.total_stores || 0}
                        </div>
                      </div>
                      <div className="banner-item">
                        <div className="banner-item-label">Total Buy Cost</div>
                        <div className="banner-item-value">
                          ₹{results.top_banner.total_buy_cost?.toLocaleString() || 0}
                        </div>
                      </div>
                      <div className="banner-item">
                        <div className="banner-item-label">Total Sales Value</div>
                        <div className="banner-item-value">
                          ₹{results.top_banner.total_sales_value?.toLocaleString() || 0}
                        </div>
                      </div>
                      <div className="banner-item">
                        <div className="banner-item-label">Avg Margin vs Target</div>
                        <div className="banner-item-value">
                          {((results.top_banner.average_margin_achieved || 0) * 100).toFixed(1)}% /{' '}
                          {((results.top_banner.target_margin || 0) * 100).toFixed(1)}%
                        </div>
                      </div>
                    </div>
                  </section>
                )}

                <section className="card">
                  <h3>Recommendations</h3>
                  <pre>{JSON.stringify(results.recommendations, null, 2)}</pre>
                </section>
              </>
            )}
          </div>
        </div>
      )}

      {activeTab === 'audit' && (
        <div className="audit-logs">
          <h2>📋 Audit Logs</h2>
          {auditLogs.length === 0 ? (
            <p>No audit logs available. Run a forecast first.</p>
          ) : (
            <table>
              <thead>
                <tr>
                  <th>Agent Name</th>
                  <th>Date & Time</th>
                  <th>Description</th>
                  <th>Status</th>
                </tr>
              </thead>
              <tbody>
                {auditLogs.map((log, idx) => (
                  <tr key={idx}>
                    <td>{log.agent_name}</td>
                    <td>{new Date(log.date_time).toLocaleString()}</td>
                    <td>{log.description}</td>
                    <td>
                      <span
                        className={`status ${
                          log.status === 'Success' ? 'success' : 'fail'
                        }`}
                      >
                        {log.status}
                      </span>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </div>
      )}
    </div>
  );
}

export default App;

