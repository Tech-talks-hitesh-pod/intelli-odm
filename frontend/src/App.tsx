import React, { useState, useEffect, useMemo } from 'react';
import axios from 'axios';
import './App.css';

const API_BASE_URL = 'http://localhost:8000';

const DATA_TYPES = [
  { key: 'sales_data', label: 'Sales Data', icon: '📊', color: '#667eea' },
  { key: 'inventory_data', label: 'Inventory Data', icon: '📦', color: '#f093fb' },
  { key: 'price_data', label: 'Price Data', icon: '💰', color: '#4facfe' },
  { key: 'cost_data', label: 'Cost Data', icon: '💵', color: '#43e97b' },
  { key: 'new_articles_data', label: 'New Articles', icon: '🆕', color: '#fa709a', required: true }
];

type MainTab = 'data' | 'forecast' | 'history';

interface DataState {
  data: any[];
  columns: string[];
  loading: boolean;
  validation: any;
  uploading: boolean;
}

// Collapsible JSON Tile Component - Optimized with memoization and truncation
const CollapsibleJsonTile: React.FC<{ data: any; title: string }> = React.memo(({ data, title }) => {
  const [isExpanded, setIsExpanded] = useState(false);
  
  // Memoize JSON stringification and truncate if too large to prevent UI freezing
  const jsonString = useMemo(() => {
    try {
      const json = JSON.stringify(data, null, 2);
      // Truncate if larger than 50KB to prevent UI freezing
      const maxLength = 50000;
      if (json.length > maxLength) {
        return json.substring(0, maxLength) + `\n\n... (truncated, ${json.length - maxLength} characters remaining)`;
      }
      return json;
    } catch (e) {
      return String(data);
    }
  }, [data]);
  
  return (
    <div className="json-tile">
      <button
        className="json-tile-header"
        onClick={() => setIsExpanded(!isExpanded)}
        type="button"
      >
        <span>{title}</span>
        <span>{isExpanded ? '▼' : '▶'}</span>
      </button>
      {isExpanded && (
        <div className="json-tile-content">
          <pre>{jsonString}</pre>
        </div>
      )}
    </div>
  );
}, (prevProps, nextProps) => {
  // Custom comparison - only re-render if key fields change
  return prevProps.title === nextProps.title && 
         prevProps.data?.date_time === nextProps.data?.date_time &&
         prevProps.data?.agent_name === nextProps.data?.agent_name &&
         prevProps.data?.description === nextProps.data?.description;
});

// Set display name for React DevTools
CollapsibleJsonTile.displayName = 'CollapsibleJsonTile';

function App() {
  const [mainTab, setMainTab] = useState<MainTab>('data');
  const [dataTab, setDataTab] = useState<string>(DATA_TYPES[0].key);
  const [dataStates, setDataStates] = useState<{ [key: string]: DataState }>({});
  const [history, setHistory] = useState<{ [key: string]: any[] }>({});
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
  const [generatedSampleData, setGeneratedSampleData] = useState<any[]>([]);
  const [showGeneratedPreview, setShowGeneratedPreview] = useState(false);
  const [streamingLogs, setStreamingLogs] = useState<any[]>([]);
  const [isStreaming, setIsStreaming] = useState(false);
  const [forecastTab, setForecastTab] = useState<'logs' | 'summary'>('logs');
  const [previousRuns, setPreviousRuns] = useState<any[]>([]);
  const [selectedRunId, setSelectedRunId] = useState<string | null>(null);
  const [currentRunId, setCurrentRunId] = useState<string | null>(null);
  const [eventSource, setEventSource] = useState<EventSource | null>(null);
  const [logsDisplayLimit, setLogsDisplayLimit] = useState<number>(50); // Show first 50 logs initially
  const [newArticlesRowCount, setNewArticlesRowCount] = useState<number>(10); // Number of rows for new articles generation

  // Load sample data on mount and when tab changes
  useEffect(() => {
    loadSampleData(dataTab);
    loadHistory(dataTab);
  }, [dataTab]);

  // Load previous runs when forecast tab is active
  useEffect(() => {
    if (mainTab === 'forecast') {
      loadPreviousRuns();
    }
  }, [mainTab]);

  // Cleanup event source on unmount
  useEffect(() => {
    return () => {
      if (eventSource) {
        eventSource.close();
      }
    };
  }, [eventSource]);

  const loadSampleData = async (dataType: string) => {
    if (!dataStates[dataType]) {
      setDataStates(prev => ({
        ...prev,
        [dataType]: { data: [], columns: [], loading: true, validation: null, uploading: false }
      }));
    } else {
      setDataStates(prev => ({
        ...prev,
        [dataType]: { ...prev[dataType], loading: true }
      }));
    }

    try {
      const response = await axios.get(`${API_BASE_URL}/api/data/${dataType}`);
      if (response.data.success) {
        setDataStates(prev => ({
          ...prev,
          [dataType]: {
            data: response.data.data || [],
            columns: response.data.columns || [],
            loading: false,
            validation: null,
            uploading: false
          }
        }));
      }
    } catch (err: any) {
      console.error(`Failed to load ${dataType}:`, err);
      setDataStates(prev => ({
        ...prev,
        [dataType]: {
          data: [],
          columns: [],
          loading: false,
          validation: null,
          uploading: false
        }
      }));
    }
  };

  const loadHistory = async (dataType: string) => {
    try {
      const response = await axios.get(`${API_BASE_URL}/api/data/${dataType}/history`);
      if (response.data.success) {
        setHistory(prev => ({
          ...prev,
          [dataType]: response.data.history || []
        }));
      }
    } catch (err) {
      console.error(`Failed to load history for ${dataType}:`, err);
    }
  };

  const handleFileUpload = async (dataType: string, file: File) => {
    setDataStates(prev => ({
      ...prev,
      [dataType]: { ...prev[dataType], uploading: true, validation: null }
    }));

    const formData = new FormData();
    formData.append('file', file);

    try {
      const response = await axios.post(
        `${API_BASE_URL}/api/data/${dataType}/upload`,
        formData,
        { headers: { 'Content-Type': 'multipart/form-data' } }
      );

      if (response.data.success) {
        // Update data with new uploaded data
        setDataStates(prev => ({
          ...prev,
          [dataType]: {
            data: response.data.data || [],
            columns: response.data.columns || [],
            loading: false,
            validation: response.data.validation,
            uploading: false
          }
        }));
        
        // Reload history
        loadHistory(dataType);
        
        // Show success message
        setError(null);
      } else {
        setDataStates(prev => ({
          ...prev,
          [dataType]: {
            ...prev[dataType],
            validation: response.data.validation,
            uploading: false
          }
        }));
        setError(response.data.message || 'Upload failed');
      }
    } catch (err: any) {
      setDataStates(prev => ({
        ...prev,
        [dataType]: { ...prev[dataType], uploading: false }
      }));
      setError(err.response?.data?.detail || 'Upload failed');
    }
  };

  const handleDownloadTemplate = async (dataType: string) => {
    try {
      const resp = await axios.get(`${API_BASE_URL}/api/templates/${dataType}`, {
        responseType: 'blob',
      });
      const csvText = await resp.data.text?.() || await resp.data;
      const blob = new Blob([csvText], { type: 'text/csv' });
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `${dataType}_template.csv`;
      document.body.appendChild(a);
      a.click();
      window.URL.revokeObjectURL(url);
      a.remove();
    } catch (err) {
      setError('Could not download template');
    }
  };

  const handleRestore = async (dataType: string, version: string) => {
    try {
      const response = await axios.get(`${API_BASE_URL}/api/data/${dataType}/restore/${version}`);
      if (response.data.success) {
        setDataStates(prev => ({
          ...prev,
          [dataType]: {
            data: response.data.data || [],
            columns: response.data.columns || [],
            loading: false,
            validation: null,
            uploading: false
          }
        }));
        loadHistory(dataType);
        setError(null);
      }
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Restore failed');
    }
  };

  const generateSampleData = async (dataType: string) => {
    try {
      // For new articles, include row count as query parameter
      const url = dataType === 'new_articles_data'
        ? `${API_BASE_URL}/api/data/${dataType}/generate?num_rows=${newArticlesRowCount}`
        : `${API_BASE_URL}/api/data/${dataType}/generate`;
      
      const response = await axios.post(url);
      if (response.data.success) {
        setGeneratedSampleData(response.data.data || []);
        setShowGeneratedPreview(true);
        setError(null);
      }
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Failed to generate sample data');
    }
  };

  const handleUseGeneratedDataForAll = async () => {
    if (generatedSampleData.length === 0) {
      setError('No generated data to use');
      return;
    }

    // Convert to CSV based on data type
    const headers = Object.keys(generatedSampleData[0]);
    const csv = [
      headers.join(','),
      ...generatedSampleData.map((r) =>
        headers.map((h) => {
          const val = r[h];
          if (typeof val === 'string' && (val.includes(',') || val.includes('"'))) {
            return `"${val.replace(/"/g, '""')}"`;
          }
          return val || '';
        }).join(',')
      ),
    ].join('\n');

    const file = new File([csv], `generated_${dataTab}.csv`, { type: 'text/csv' });
    await handleFileUpload(dataTab, file);
    
    if (!error) {
      setGeneratedSampleData([]);
      setShowGeneratedPreview(false);
    }
  };

  const generateSampleNewArticles = async () => {
    // First, get existing SKUs from other data types to ensure 60% matching
    const existingSkus = new Set<string>();
    try {
      // Get SKUs from sales, inventory, price data
      const salesResp = await axios.get(`${API_BASE_URL}/api/data/sales_data`);
      if (salesResp.data.success && salesResp.data.data) {
        salesResp.data.data.forEach((row: any) => {
          if (row.sku) existingSkus.add(row.sku);
        });
      }
      const invResp = await axios.get(`${API_BASE_URL}/api/data/inventory_data`);
      if (invResp.data.success && invResp.data.data) {
        invResp.data.data.forEach((row: any) => {
          if (row.sku) existingSkus.add(row.sku);
        });
      }
      const priceResp = await axios.get(`${API_BASE_URL}/api/data/price_data`);
      if (priceResp.data.success && priceResp.data.data) {
        priceResp.data.data.forEach((row: any) => {
          if (row.sku) existingSkus.add(row.sku);
        });
      }
    } catch (err) {
      console.warn('Could not fetch existing SKUs for matching', err);
    }

    const existingSkuList = Array.from(existingSkus);
    const matchingCount = Math.ceil(10 * 0.6); // 60% of 10 = 6
    const matchingSkus = existingSkuList.slice(0, matchingCount);
    const newSkus = Array.from({ length: 10 - matchingSkus.length }, (_, i) => `VS-${String(100 + i).padStart(3, '0')}`);

    // Define the complete schema with all required and optional fields
    const baseSchema = {
      product_id: '',
      vendor_sku: '',
      description: '',
      category: '',
      color: '',
      material: '',
      size_set: '',
      brick: '',
      class: '',
      segment: '',
      family: '',
      brand: ''
    };

    // Generate sample products with 60% matching SKUs
    const sampleProducts = [
      {
        product_id: 'P100',
        vendor_sku: matchingSkus[0] || 'VS-001',
        description: 'White cotton t-shirt, chest print, short sleeve',
        category: 'TSHIRT',
        color: 'White',
        material: 'Cotton',
        size_set: 'S,M,L,XL',
        brick: 'T-SHIRT',
        class: 'Top Wear',
        segment: 'Mens wear',
        family: 'Active Wear',
        brand: 'GAP'
      },
      {
        product_id: 'P101',
        vendor_sku: matchingSkus[1] || 'VS-002',
        description: 'Black polo shirt, solid, half sleeves',
        category: 'POLO',
        color: 'Black',
        material: 'Cotton',
        size_set: 'M,L,XL',
        brick: 'POLO',
        class: 'Top Wear',
        segment: 'Mens wear',
        family: 'Classic Wear',
        brand: 'BENETTON'
      },
      {
        product_id: 'P102',
        vendor_sku: matchingSkus[2] || 'VS-003',
        description: 'Blue denim jeans, slim fit',
        category: 'JEANS',
        color: 'Blue',
        material: 'Denim',
        size_set: '30,32,34,36',
        brick: 'JEANS',
        class: 'Bottom Wear',
        segment: 'Mens wear',
        family: 'Denim Wear',
        brand: 'LEVIS'
      },
      {
        product_id: 'P103',
        vendor_sku: matchingSkus[3] || 'VS-004',
        description: 'Red hoodie, zip-up, fleece lining',
        category: 'HOODIE',
        color: 'Red',
        material: 'Polyester',
        size_set: 'S,M,L,XL',
        brick: 'HOODIE',
        class: 'Top Wear',
        segment: 'Mens wear',
        family: 'Casual Wear',
        brand: 'NIKE'
      },
      {
        product_id: 'P104',
        vendor_sku: matchingSkus[4] || 'VS-005',
        description: 'Green cargo pants, multiple pockets',
        category: 'PANTS',
        color: 'Green',
        material: 'Cotton Blend',
        size_set: '30,32,34,36',
        brick: 'PANTS',
        class: 'Bottom Wear',
        segment: 'Mens wear',
        family: 'Casual Wear',
        brand: 'ADIDAS'
      },
      {
        product_id: 'P105',
        vendor_sku: matchingSkus[5] || 'VS-006',
        description: 'Navy blue blazer, single breasted',
        category: 'BLAZER',
        color: 'Navy Blue',
        material: 'Wool Blend',
        size_set: '38,40,42,44',
        brick: 'BLAZER',
        class: 'Top Wear',
        segment: 'Mens wear',
        family: 'Formal Wear',
        brand: 'ZARA'
      },
      {
        product_id: 'P106',
        vendor_sku: newSkus[0] || 'VS-007',
        description: 'Gray sweatpants, elastic waist',
        category: 'SWEATPANTS',
        color: 'Gray',
        material: 'Cotton',
        size_set: 'S,M,L,XL',
        brick: 'SWEATPANTS',
        class: 'Bottom Wear',
        segment: 'Mens wear',
        family: 'Active Wear',
        brand: 'PUMA'
      },
      {
        product_id: 'P107',
        vendor_sku: newSkus[1] || 'VS-008',
        description: 'Yellow rain jacket, waterproof',
        category: 'JACKET',
        color: 'Yellow',
        material: 'Nylon',
        size_set: 'M,L,XL',
        brick: 'JACKET',
        class: 'Top Wear',
        segment: 'Mens wear',
        family: 'Outerwear',
        brand: 'COLUMBIA'
      },
      {
        product_id: 'P108',
        vendor_sku: newSkus[2] || 'VS-009',
        description: 'Beige chinos, straight fit',
        category: 'CHINOS',
        color: 'Beige',
        material: 'Cotton',
        size_set: '30,32,34,36',
        brick: 'CHINOS',
        class: 'Bottom Wear',
        segment: 'Mens wear',
        family: 'Casual Wear',
        brand: 'H&M'
      },
      {
        product_id: 'P109',
        vendor_sku: newSkus[3] || 'VS-010',
        description: 'Purple crew neck sweater, wool blend',
        category: 'SWEATER',
        color: 'Purple',
        material: 'Wool Blend',
        size_set: 'S,M,L,XL',
        brick: 'SWEATER',
        class: 'Top Wear',
        segment: 'Mens wear',
        family: 'Casual Wear',
        brand: 'TOMMY HILFIGER'
      }
    ];

    // Ensure all rows have all fields by merging with base schema
    const cleanedProducts = sampleProducts.map(product => {
      const cleaned = { ...baseSchema, ...product };
      // Ensure no undefined or null values - use empty string as fallback
      Object.keys(cleaned).forEach(key => {
        if (cleaned[key] === undefined || cleaned[key] === null) {
          cleaned[key] = '';
        }
      });
      return cleaned;
    });

    setGeneratedSampleData(cleanedProducts);
    setShowGeneratedPreview(true);
  };

  const handleUseGeneratedData = async () => {
    if (generatedSampleData.length === 0) {
      setError('No generated data to use');
      return;
    }

    // Clean and validate data before converting to CSV
    const baseSchema = {
      product_id: '',
      vendor_sku: '',
      description: '',
      category: '',
      color: '',
      material: '',
      size_set: '',
      brick: '',
      class: '',
      segment: '',
      family: '',
      brand: ''
    };

    // Ensure all rows have all required fields
    const cleanedData = generatedSampleData.map((row, index) => {
      const cleaned: any = { ...baseSchema };
      
      // Copy all existing values
      Object.keys(row).forEach(key => {
        const value = (row as any)[key];
        // Ensure no undefined, null, or empty values for required fields
        if (key === 'product_id' || key === 'vendor_sku' || key === 'description') {
          cleaned[key] = (value && String(value).trim()) || `AUTO_${index + 1}`;
        } else {
          cleaned[key] = (value && String(value).trim()) || '';
        }
      });

      // Ensure required fields are never empty
      if (!cleaned.product_id || cleaned.product_id === '') {
        cleaned.product_id = `P${100 + index}`;
      }
      if (!cleaned.vendor_sku || cleaned.vendor_sku === '') {
        cleaned.vendor_sku = `VS-${String(index + 1).padStart(3, '0')}`;
      }
      if (!cleaned.description || cleaned.description === '') {
        cleaned.description = `Sample product ${index + 1}`;
      }

      return cleaned;
    });

    // Use the standard schema order for headers to ensure consistency
    const standardHeaders = [
      'product_id',
      'vendor_sku',
      'description',
      'category',
      'color',
      'material',
      'size_set',
      'brick',
      'class',
      'segment',
      'family',
      'brand'
    ];
    
    // Ensure all standard headers are present, and add any extra ones
    const allHeaders = new Set<string>(standardHeaders);
    cleanedData.forEach((row: any) => {
      Object.keys(row).forEach(key => allHeaders.add(key));
    });
    const headers = Array.from(allHeaders);
    
    // Ensure all rows have all headers (fill missing with empty string)
    const finalData = cleanedData.map((row: any) => {
      const completeRow: any = {};
      headers.forEach(header => {
        completeRow[header] = row[header] !== undefined && row[header] !== null 
          ? String(row[header]).trim() 
          : '';
      });
      return completeRow;
    });

    // Convert to CSV with proper escaping - use finalData instead of cleanedData
    const csv = [
      headers.join(','),
      ...finalData.map((r: any) =>
        headers.map((h) => {
          const val = r[h] || '';
          const strVal = String(val);
          // Escape values that contain commas, quotes, or newlines
          if (strVal.includes(',') || strVal.includes('"') || strVal.includes('\n')) {
            return `"${strVal.replace(/"/g, '""')}"`;
          }
          return strVal;
        }).join(',')
      ),
    ].join('\n');

    // Create file from CSV
    const file = new File([csv], 'generated_new_articles.csv', { type: 'text/csv' });

    // Upload and validate using the same upload handler
    await handleFileUpload('new_articles_data', file);

    // Clear generated data and hide preview only if upload was successful
    // (The upload handler will update the state, so we check if there's no error)
    if (!error) {
      setGeneratedSampleData([]);
      setShowGeneratedPreview(false);
    }
  };

  const loadPreviousRuns = async () => {
    try {
      const response = await axios.get(`${API_BASE_URL}/api/forecast-runs`);
      if (response.data.success) {
        setPreviousRuns(response.data.runs || []);
      }
    } catch (err) {
      console.error('Failed to load previous runs', err);
    }
  };

  const loadForecastRun = async (runId: string) => {
    try {
      setError(null);
      setLoading(true);
      setIsStreaming(false); // Clear streaming state when loading existing run
      setLogsDisplayLimit(50); // Reset to initial limit
      
      const response = await axios.get(`${API_BASE_URL}/api/forecast-runs/${runId}`, {
        timeout: 10000 // 10 second timeout
      });
      
      if (response.data.success) {
        setResults(response.data.results || null);
        const logs = response.data.logs || [];
        // Limit logs to prevent UI freezing - show most recent first
        const sortedLogs = logs.sort((a: any, b: any) => 
          new Date(b.date_time || 0).getTime() - new Date(a.date_time || 0).getTime()
        );
        setAuditLogs(sortedLogs);
        setStreamingLogs(sortedLogs);
        setSelectedRunId(runId);
        // If no results, show logs tab instead
        if (!response.data.results || Object.keys(response.data.results).length === 0) {
          setForecastTab('logs');
        } else {
          setForecastTab('summary');
        }
      }
    } catch (err: any) {
      const errorMsg = err.response?.data?.detail || err.message || 'Failed to load forecast run';
      setError(typeof errorMsg === 'string' ? errorMsg : JSON.stringify(errorMsg));
      console.error('Error loading forecast run:', err);
      // Still try to load logs directly if forecast-runs endpoint fails
      try {
        const logsResponse = await axios.get(`${API_BASE_URL}/api/audit-logs/${runId}`, {
          timeout: 10000
        });
        if (logsResponse.data.success) {
          const logs = logsResponse.data.logs || [];
          // Sort and limit logs
          const sortedLogs = logs.sort((a: any, b: any) => 
            new Date(b.date_time || 0).getTime() - new Date(a.date_time || 0).getTime()
          );
          setAuditLogs(sortedLogs);
          setStreamingLogs(sortedLogs);
          setSelectedRunId(runId);
          setForecastTab('logs');
        }
      } catch (logErr) {
        console.error('Error loading logs directly:', logErr);
      }
    } finally {
      setLoading(false);
    }
  };

  const handleForecast = async () => {
    setError(null);
    setLoading(true);
    setIsStreaming(true);
    setStreamingLogs([]);
    setAuditLogs([]);
    setForecastTab('logs');
    setSelectedRunId(null);
    setResults(null);
    setLogsDisplayLimit(50); // Reset to initial limit

    // Close existing event source if any
    if (eventSource) {
      eventSource.close();
      setEventSource(null);
    }

      let pollInterval: number | null = null;
    let runId: string | null = null;

    try {
      const formData = new FormData();
      formData.append('margin_target', params.margin_target.toString());
      formData.append('variance_threshold', params.variance_threshold.toString());
      formData.append('forecast_horizon_days', params.forecast_horizon_days.toString());
      formData.append('max_quantity_per_store', params.max_quantity_per_store.toString());
      if (params.universe_of_stores) {
        formData.append('universe_of_stores', params.universe_of_stores);
      }

      // Start forecast - it now returns run_id immediately
      const startResponse = await axios.post(`${API_BASE_URL}/api/forecast`, formData, {
        headers: { 'Content-Type': 'multipart/form-data' },
      });

      if (startResponse.data.success && startResponse.data.run_id) {
        runId = startResponse.data.run_id;
        setCurrentRunId(runId);
        
        // Start polling immediately as primary method (more reliable than SSE)
        const pollLogs = async () => {
          try {
            const logsResp = await axios.get(`${API_BASE_URL}/api/audit-logs/${runId}`);
            if (logsResp.data.success) {
              const newLogs = logsResp.data.logs || [];
              if (newLogs.length > 0) {
                setAuditLogs(newLogs);
                setStreamingLogs(newLogs);
                
                // Check if forecast is complete
                const lastLog = newLogs[newLogs.length - 1];
                const desc = lastLog.description?.toLowerCase() || '';
                if (desc.includes('forecast completed') || desc.includes('forecast failed') || 
                    desc.includes('completed successfully')) {
                  if (pollInterval) {
                    window.clearInterval(pollInterval);
                    pollInterval = null;
                  }
                  if (es) {
                    es.close();
                    setEventSource(null);
                  }
                  setIsStreaming(false);
                  setLoading(false);
                  loadForecastRun(runId!);
                  loadPreviousRuns();
                }
              }
            }
          } catch (pollErr) {
            console.error('Error polling logs', pollErr);
          }
        };
        
        // Poll immediately and then every 1 second
        pollLogs();
        pollInterval = window.setInterval(pollLogs, 1000);
        
        // Also set up SSE as secondary method
        let es: EventSource | null = null;
        try {
          es = new EventSource(`${API_BASE_URL}/api/audit-logs/${runId}/stream`);
          setEventSource(es);

          es.onmessage = (event) => {
            try {
              const data = JSON.parse(event.data);
              console.log('SSE message received:', data.type);
              
              if (data.type === 'connected') {
                console.log('SSE connected for run:', data.run_id);
              } else if (data.type === 'log') {
                const newLog = data.data;
                setAuditLogs(prev => {
                  const exists = prev.some((l: any) => 
                    l.date_time === newLog.date_time && 
                    l.agent_name === newLog.agent_name &&
                    l.description === newLog.description
                  );
                  if (!exists) {
                    console.log('Adding new log via SSE:', newLog.description);
                    return [...prev, newLog];
                  }
                  return prev;
                });
                setStreamingLogs(prev => {
                  const exists = prev.some((l: any) => 
                    l.date_time === newLog.date_time && 
                    l.agent_name === newLog.agent_name &&
                    l.description === newLog.description
                  );
                  if (!exists) {
                    return [...prev, newLog];
                  }
                  return prev;
                });
              } else if (data.type === 'complete') {
                console.log('Forecast completed via SSE');
                if (pollInterval) {
                  window.clearInterval(pollInterval);
                  pollInterval = null;
                }
                if (es) {
                  es.close();
                  setEventSource(null);
                }
                setIsStreaming(false);
                setLoading(false);
                loadForecastRun(runId!);
                loadPreviousRuns();
              } else if (data.type === 'error') {
                console.error('SSE error:', data.error);
              }
            } catch (err) {
              console.error('Error parsing SSE message', err);
            }
          };

          es.onerror = (err) => {
            console.error('SSE connection error', err);
            // Don't close, polling will handle it
          };
        } catch (sseErr) {
          console.error('Failed to create SSE connection, using polling only:', sseErr);
        }
      } else {
        setError(startResponse.data.detail || 'Forecast failed to start');
        setIsStreaming(false);
        setLoading(false);
      }
    } catch (err: any) {
      if (pollInterval) {
        window.clearInterval(pollInterval);
      }
      if (eventSource) {
        eventSource.close();
        setEventSource(null);
      }
      const errorMsg = err.response?.data?.detail || err.message || 'Forecast failed';
      setError(typeof errorMsg === 'string' ? errorMsg : JSON.stringify(errorMsg));
      setIsStreaming(false);
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

  const currentData = dataStates[dataTab] || { data: [], columns: [], loading: false, validation: null, uploading: false };
  const currentHistory = history[dataTab] || [];

  return (
    <div className="app">
      <header className="header">
        <h1>📊 Demand Forecasting & Allocation Engine</h1>
        <p className="subtitle">Multi-Agent LLM System using Ollama (llama3:8b)</p>
      </header>

      <div className="main-tabs">
        <button
          className={`main-tab ${mainTab === 'data' ? 'active' : ''}`}
          onClick={() => setMainTab('data')}
        >
          📁 Data Management
        </button>
        <button
          className={`main-tab ${mainTab === 'forecast' ? 'active' : ''}`}
          onClick={() => setMainTab('forecast')}
        >
          🚀 Forecast
        </button>
        <button
          className={`main-tab ${mainTab === 'history' ? 'active' : ''}`}
          onClick={() => setMainTab('history')}
        >
          📜 History
        </button>
      </div>

      {mainTab === 'data' && (
        <div className="data-management">
          <div className="data-type-tabs">
            {DATA_TYPES.map(({ key, label, icon, color }) => (
              <button
                key={key}
                className={`data-type-tab ${dataTab === key ? 'active' : ''}`}
                onClick={() => setDataTab(key)}
                style={{ '--tab-color': color } as React.CSSProperties}
              >
                <span className="tab-icon">{icon}</span>
                {label}
                {DATA_TYPES.find(dt => dt.key === key)?.required && (
                  <span className="required-badge">*</span>
                )}
              </button>
            ))}
          </div>

          <div className="data-content">
            <div className="data-actions">
              <div className="upload-section">
                <input
                  type="file"
                  accept=".csv,.xlsx,.xls"
                  id={`file-upload-${dataTab}`}
                  onChange={(e) => {
                    const file = e.target.files?.[0];
                    if (file) handleFileUpload(dataTab, file);
                  }}
                  style={{ display: 'none' }}
                />
                <label htmlFor={`file-upload-${dataTab}`} className="upload-button">
                  {currentData.uploading 
                    ? '⏳ Uploading...' 
                    : `📤 Upload ${DATA_TYPES.find(dt => dt.key === dataTab)?.label || 'Data'}`}
                </label>
                {dataTab === 'new_articles_data' && (
                  <div className="row-count-slider-container">
                    <label className="slider-label">
                      Number of Rows: <strong>{newArticlesRowCount}</strong>
                    </label>
                    <input
                      type="range"
                      min="5"
                      max="50"
                      step="5"
                      value={newArticlesRowCount}
                      onChange={(e) => setNewArticlesRowCount(parseInt(e.target.value))}
                      className="slider"
                    />
                    <div className="slider-labels">
                      <span>5</span>
                      <span>50</span>
                    </div>
                  </div>
                )}
                <button
                  className="action-button generate-button"
                  onClick={() => {
                    if (dataTab === 'new_articles_data') {
                      generateSampleData(dataTab);
                    } else {
                      generateSampleData(dataTab);
                    }
                  }}
                >
                  🎲 Generate Sample Data
                </button>
                <button
                  className="action-button"
                  onClick={() => handleDownloadTemplate(dataTab)}
                >
                  📥 Download Template
                </button>
                <button
                  className="action-button"
                  onClick={() => loadSampleData(dataTab)}
                >
                  🔄 Refresh
                </button>
              </div>

              {showGeneratedPreview && generatedSampleData.length > 0 && (
                <div className="generated-preview">
                  <div className="preview-header">
                    <h3>✨ Generated Sample Data Preview</h3>
                    <div className="preview-actions">
                      <button
                        className="use-button"
                        onClick={dataTab === 'new_articles_data' ? handleUseGeneratedData : handleUseGeneratedDataForAll}
                        disabled={currentData.uploading}
                      >
                        {currentData.uploading ? '⏳ Processing...' : '✅ Use This Data'}
                      </button>
                      <button
                        className="cancel-button"
                        onClick={() => {
                          setShowGeneratedPreview(false);
                          setGeneratedSampleData([]);
                        }}
                      >
                        ❌ Cancel
                      </button>
                    </div>
                  </div>
                  <div className="preview-table-container">
                    {generatedSampleData.length > 0 && (
                      <>
                        <table className="data-table">
                          <thead>
                            <tr>
                              {Object.keys(generatedSampleData[0]).map((col) => (
                                <th key={col}>{col}</th>
                              ))}
                            </tr>
                          </thead>
                          <tbody>
                            {generatedSampleData.map((row, i) => (
                              <tr key={i}>
                                {Object.keys(generatedSampleData[0]).map((col) => (
                                  <td key={col}>
                                    {dataTab === 'new_articles_data' ? (
                                      <input
                                        type="text"
                                        value={row[col] || ''}
                                        onChange={(e) => {
                                          const newData = [...generatedSampleData];
                                          newData[i] = { ...newData[i], [col]: e.target.value };
                                          setGeneratedSampleData(newData);
                                        }}
                                        className="editable-cell"
                                      />
                                    ) : (
                                      <span>{row[col] ?? ''}</span>
                                    )}
                                  </td>
                                ))}
                              </tr>
                            ))}
                          </tbody>
                        </table>
                        <div className="preview-footer">
                          <p>💡 {dataTab === 'new_articles_data' ? 'You can edit the data above before using it. ' : ''}Click "Use This Data" to validate and replace current sample data.</p>
                        </div>
                      </>
                    )}
                  </div>
                </div>
              )}

              {currentData.validation && (
                <div className={`validation-result ${currentData.validation.valid ? 'valid' : 'invalid'}`}>
                  <h3>{currentData.validation.valid ? '✅ Validation Passed' : '❌ Validation Failed'}</h3>
                  {currentData.validation.data_quality_score && (
                    <p>Data Quality Score: {currentData.validation.data_quality_score}/100</p>
                  )}
                  {currentData.validation.errors?.length > 0 && (
                    <div>
                      <strong>Errors:</strong>
                      <ul>
                        {currentData.validation.errors.map((err: string, i: number) => (
                          <li key={i}>{err}</li>
                        ))}
                      </ul>
                    </div>
                  )}
                  {currentData.validation.warnings?.length > 0 && (
                    <div>
                      <strong>Warnings:</strong>
                      <ul>
                        {currentData.validation.warnings.map((warn: string, i: number) => (
                          <li key={i}>{warn}</li>
                        ))}
                      </ul>
                    </div>
                  )}
                  {currentData.validation.recommendations?.length > 0 && (
                    <div>
                      <strong>Recommendations:</strong>
                      <ul>
                        {currentData.validation.recommendations.map((rec: string, i: number) => (
                          <li key={i}>{rec}</li>
                        ))}
                      </ul>
                    </div>
                  )}
                </div>
              )}
            </div>

            <div className="data-table-container">
              {currentData.loading ? (
                <div className="loading">Loading data...</div>
              ) : currentData.data.length === 0 ? (
                <div className="empty-state">No data available</div>
              ) : (
                <div className="table-wrapper">
                  <table className="data-table">
                    <thead>
                      <tr>
                        {currentData.columns.map((col) => (
                          <th key={col}>{col}</th>
                        ))}
                      </tr>
                    </thead>
                    <tbody>
                      {currentData.data.slice(0, 100).map((row: any, i: number) => (
                        <tr key={i}>
                          {currentData.columns.map((col) => (
                            <td key={col}>{row[col] ?? ''}</td>
                          ))}
                        </tr>
                      ))}
                    </tbody>
                  </table>
                  {currentData.data.length > 100 && (
                    <div className="table-footer">
                      Showing first 100 of {currentData.data.length} rows
                    </div>
                  )}
                </div>
              )}
            </div>
          </div>
        </div>
      )}

      {mainTab === 'forecast' && (
        <div className="forecast-content">
          <div className="forecast-left">
            <section className="card">
              <h2>⚙️ Configure Parameters</h2>
              <div className="form-group">
                <label>Margin Target: {params.margin_target}%</label>
                <input
                  type="range"
                  min="0"
                  max="100"
                  step="1"
                  value={params.margin_target}
                  onChange={(e) =>
                    setParams({ ...params, margin_target: parseFloat(e.target.value) })
                  }
                  className="slider"
                />
                <div className="slider-labels">
                  <span>0%</span>
                  <span>100%</span>
                </div>
              </div>
              <div className="form-group">
                <label>Variance Threshold: {params.variance_threshold}%</label>
                <input
                  type="range"
                  min="0"
                  max="100"
                  step="1"
                  value={params.variance_threshold}
                  onChange={(e) =>
                    setParams({ ...params, variance_threshold: parseFloat(e.target.value) })
                  }
                  className="slider"
                />
                <div className="slider-labels">
                  <span>0%</span>
                  <span>100%</span>
                </div>
              </div>
              <div className="form-group">
                <label>Forecast Horizon: {params.forecast_horizon_days} Days</label>
                <input
                  type="range"
                  min="1"
                  max="180"
                  step="1"
                  value={params.forecast_horizon_days}
                  onChange={(e) =>
                    setParams({ ...params, forecast_horizon_days: parseInt(e.target.value) })
                  }
                  className="slider"
                />
                <div className="slider-labels">
                  <span>1</span>
                  <span>180</span>
                </div>
              </div>
              <div className="form-group">
                <label>Max Quantity per Store: {params.max_quantity_per_store}</label>
                <input
                  type="range"
                  min="1"
                  max="2000"
                  step="10"
                  value={params.max_quantity_per_store}
                  onChange={(e) =>
                    setParams({ ...params, max_quantity_per_store: parseInt(e.target.value) })
                  }
                  className="slider"
                />
                <div className="slider-labels">
                  <span>1</span>
                  <span>2000</span>
                </div>
              </div>
              <div className="form-group">
                <label>Universe of Stores (Optional)</label>
                <input
                  type="text"
                  value={params.universe_of_stores}
                  onChange={(e) =>
                    setParams({ ...params, universe_of_stores: e.target.value })
                  }
                  placeholder="e.g., 20 or leave empty"
                />
              </div>
              <button
                className="btn btn-primary btn-large"
                onClick={handleForecast}
                disabled={loading}
              >
                {loading ? '⏳ Running Forecast...' : '▶️ Run Demand Forecast'}
              </button>
              {error && <div className="error-message">{error}</div>}
            </section>
          </div>

          <div className="forecast-right">
            {/* Run Selection Tabs */}
            <div className="run-selection-tabs">
              <button
                className={`run-tab ${selectedRunId === null ? 'active' : ''}`}
                onClick={() => {
                  setSelectedRunId(null);
                  setForecastTab('logs');
                  // Show current run logs if available
                  if (currentRunId) {
                    loadForecastRun(currentRunId);
                  } else {
                    // Clear logs if no current run
                    setAuditLogs([]);
                    setStreamingLogs([]);
                    setResults(null);
                  }
                }}
              >
                🔄 Current Run
                {currentRunId && (
                  <span className="tab-badge-small">{currentRunId.split('_').pop()}</span>
                )}
              </button>
              <button
                className={`run-tab ${selectedRunId !== null ? 'active' : ''}`}
                onClick={() => {
                  // Switch to previous runs view - select first run if none selected
                  if (previousRuns.length > 0) {
                    if (!selectedRunId) {
                      loadForecastRun(previousRuns[0].run_id);
                    }
                  }
                }}
              >
                📜 Previous Runs
                {previousRuns.length > 0 && (
                  <span className="tab-badge-small">{previousRuns.length}</span>
                )}
              </button>
            </div>

            {/* Previous Runs List - Only show when Previous Runs tab is active */}
            {selectedRunId !== null && previousRuns.length > 0 && (
              <section className="card previous-runs-section">
                <h3>📜 Previous Runs</h3>
                <div className="previous-runs-list">
                  {previousRuns.map((run: any) => (
                    <div
                      key={run.run_id}
                      className={`previous-run-item ${selectedRunId === run.run_id ? 'selected' : ''}`}
                      onClick={() => loadForecastRun(run.run_id)}
                    >
                      <div className="run-header">
                        <strong>{new Date(run.timestamp).toLocaleString()}</strong>
                        <span className="run-id">{run.run_id}</span>
                      </div>
                      <div className="run-summary">
                        <span>SKUs: {run.summary?.total_skus || 0}</span>
                        <span>Qty: {run.summary?.total_quantity || 0}</span>
                        <span>Stores: {run.summary?.total_stores || 0}</span>
                        <span>Logs: {run.log_count || 0}</span>
                      </div>
                    </div>
                  ))}
                </div>
              </section>
            )}

            {/* Forecast Results Tabs - Always Visible */}
            <div className="forecast-results">
              <div className="forecast-results-tabs">
                <button
                  className={`forecast-tab ${forecastTab === 'logs' ? 'active' : ''}`}
                  onClick={() => setForecastTab('logs')}
                >
                  📋 Audit Logs
                  {(auditLogs.length > 0 || streamingLogs.length > 0) && (
                    <span className="tab-badge">
                      {auditLogs.length || streamingLogs.length}
                    </span>
                  )}
                </button>
                <button
                  className={`forecast-tab ${forecastTab === 'summary' ? 'active' : ''}`}
                  onClick={() => setForecastTab('summary')}
                >
                  📊 Summary
                </button>
              </div>

              {forecastTab === 'logs' && (
                <div className="forecast-tab-content">
                  {isStreaming && (
                    <div className="streaming-indicator">
                      <span className="pulse">●</span> Streaming logs in real-time...
                    </div>
                  )}
                  {loading && auditLogs.length === 0 && streamingLogs.length === 0 ? (
                    <div className="empty-state">
                      Loading logs...
                    </div>
                  ) : (auditLogs.length === 0 && streamingLogs.length === 0) ? (
                    <div className="empty-state">
                      No audit logs available. Run a forecast to see logs.
                    </div>
                  ) : (
                    <div className="audit-logs-list">
                      <div className="logs-header">
                        <span className="logs-count">
                          Showing {Math.min(logsDisplayLimit, (auditLogs.length > 0 ? auditLogs.length : streamingLogs.length))} of {(auditLogs.length > 0 ? auditLogs.length : streamingLogs.length)} logs
                        </span>
                      </div>
                      {(() => {
                        const logsToDisplay = (auditLogs.length > 0 ? auditLogs : streamingLogs)
                          .slice(0, logsDisplayLimit);
                        return logsToDisplay.map((log: any, idx: number) => (
                          <CollapsibleJsonTile 
                            key={`${log.date_time || idx}-${log.agent_name || 'log'}-${log.description?.substring(0, 20) || idx}`} 
                            data={log} 
                            title={`${log.agent_name || 'Log'} - ${new Date(log.date_time || Date.now()).toLocaleString()}`} 
                          />
                        ));
                      })()}
                      {((auditLogs.length > 0 ? auditLogs.length : streamingLogs.length) > logsDisplayLimit) && (
                        <div className="load-more-logs">
                          <button
                            onClick={() => setLogsDisplayLimit(prev => Math.min(prev + 50, (auditLogs.length > 0 ? auditLogs.length : streamingLogs.length)))}
                            className="load-more-button"
                          >
                            Load More Logs ({((auditLogs.length > 0 ? auditLogs.length : streamingLogs.length) - logsDisplayLimit)} remaining)
                          </button>
                          <button
                            onClick={() => setLogsDisplayLimit((auditLogs.length > 0 ? auditLogs.length : streamingLogs.length))}
                            className="load-all-button"
                          >
                            Load All ({auditLogs.length > 0 ? auditLogs.length : streamingLogs.length} total)
                          </button>
                        </div>
                      )}
                    </div>
                  )}
                </div>
              )}

              {forecastTab === 'summary' && (
                <div className="forecast-tab-content">
                  {results ? (
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
                  ) : (
                    <div className="empty-state">
                      {loading ? 'Running forecast...' : 'No results yet. Run a forecast to see summary.'}
                    </div>
                  )}
                </div>
              )}
            </div>
          </div>
        </div>
      )}

      {mainTab === 'history' && (
        <div className="history-content">
          <div className="history-tabs">
            {DATA_TYPES.map(({ key, label, icon }) => (
              <button
                key={key}
                className={`history-tab ${dataTab === key ? 'active' : ''}`}
                onClick={() => {
                  setDataTab(key);
                  loadHistory(key);
                }}
              >
                <span className="tab-icon">{icon}</span>
                {label}
              </button>
            ))}
          </div>

          <div className="history-list">
            {currentHistory.length === 0 ? (
              <div className="empty-state">No backup history available</div>
            ) : (
              currentHistory.map((backup, idx) => (
                <div key={idx} className="history-item">
                  <div className="history-item-header">
                    <h3>Version {backup.version}</h3>
                    <span className="history-date">
                      {new Date(backup.timestamp).toLocaleString()}
                    </span>
                  </div>
                  <div className="history-item-details">
                    <p><strong>Rows:</strong> {backup.row_count}</p>
                    <p><strong>File:</strong> {backup.original_filename}</p>
                    {backup.backup_exists && (
                      <button
                        className="restore-button"
                        onClick={() => handleRestore(dataTab, backup.version)}
                      >
                        🔄 Restore This Version
                      </button>
                    )}
                  </div>
                </div>
              ))
            )}
          </div>
        </div>
      )}
    </div>
  );
}

export default App;
