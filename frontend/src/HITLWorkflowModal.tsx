import React, { useState, useEffect } from 'react';
import './HITLWorkflowModal.css';

interface HITLWorkflowModalProps {
  isOpen: boolean;
  onClose: () => void;
  runId: string | null;
  onFinalize: () => void;
}

interface ApprovalItem {
  level: string;
  article: string;
  store_id?: string;
  type: string;
  original: number;
  edited: number;
  variance_pct: number;
  status: string;
  edited_by: string;
  edited_at: string;
}

interface HITLWorkflowData {
  variance_threshold: number;
  variance_threshold_pct: number;
  available_stores: string[];
  store_mappings: Record<string, any>;
  approval_queue: ApprovalItem[];
  aggregate_edits: Record<string, any>;
  store_level_edits: Record<string, Record<string, any>>;
}

const HITLWorkflowModal: React.FC<HITLWorkflowModalProps> = ({
  isOpen,
  onClose,
  runId,
  onFinalize
}) => {
  const [workflowData, setWorkflowData] = useState<HITLWorkflowData | null>(null);
  const [loading, setLoading] = useState(false);
  const [activeTab, setActiveTab] = useState<'approvals' | 'edits' | 'mappings'>('approvals');
  const [editingItem, setEditingItem] = useState<{ article: string; store_id?: string; original: number } | null>(null);
  const [editValue, setEditValue] = useState<string>('');

  useEffect(() => {
    if (isOpen && runId) {
      loadWorkflowData();
    }
  }, [isOpen, runId]);

  const loadWorkflowData = async () => {
    if (!runId) return;
    
    setLoading(true);
    try {
      const response = await fetch(`http://localhost:8000/api/hitl/${runId}`);
      if (response.ok) {
        const data = await response.json();
        setWorkflowData(data);
      }
    } catch (error) {
      console.error('Error loading HITL workflow:', error);
    } finally {
      setLoading(false);
    }
  };

  const handleApprove = async (article: string, storeId?: string) => {
    if (!runId) return;
    
    try {
      const formData = new FormData();
      formData.append('article', article);
      if (storeId) formData.append('store_id', storeId);
      formData.append('approver_id', 'user');
      
      const response = await fetch(`http://localhost:8000/api/hitl/${runId}/approve`, {
        method: 'POST',
        body: formData
      });
      
      if (response.ok) {
        await loadWorkflowData();
      }
    } catch (error) {
      console.error('Error approving item:', error);
    }
  };

  const handleReject = async (article: string, storeId?: string) => {
    if (!runId) return;
    
    try {
      const formData = new FormData();
      formData.append('article', article);
      if (storeId) formData.append('store_id', storeId);
      formData.append('reason', 'Rejected by user');
      formData.append('user_id', 'user');
      
      const response = await fetch(`http://localhost:8000/api/hitl/${runId}/reject`, {
        method: 'POST',
        body: formData
      });
      
      if (response.ok) {
        await loadWorkflowData();
      }
    } catch (error) {
      console.error('Error rejecting item:', error);
    }
  };

  const handleEdit = async () => {
    if (!runId || !editingItem) return;
    
    try {
      const formData = new FormData();
      formData.append('article', editingItem.article);
      formData.append('edited_quantity', editValue);
      formData.append('original_quantity', editingItem.original.toString());
      formData.append('user_id', 'user');
      
      if (editingItem.store_id) {
        formData.append('store_id', editingItem.store_id);
        formData.append('total_forecasted_quantity', '1000'); // TODO: Get from results
      }
      
      const endpoint = editingItem.store_id 
        ? `http://localhost:8000/api/hitl/${runId}/edit-store`
        : `http://localhost:8000/api/hitl/${runId}/edit-aggregate`;
      
      const response = await fetch(endpoint, {
        method: 'POST',
        body: formData
      });
      
      if (response.ok) {
        setEditingItem(null);
        setEditValue('');
        await loadWorkflowData();
      }
    } catch (error) {
      console.error('Error editing item:', error);
    }
  };

  const handleFinalize = async () => {
    if (!runId) return;
    
    try {
      const response = await fetch(`http://localhost:8000/api/hitl/${runId}/finalize`, {
        method: 'POST'
      });
      
      if (response.ok) {
        onFinalize();
        onClose();
      }
    } catch (error) {
      console.error('Error finalizing workflow:', error);
    }
  };

  if (!isOpen) return null;

  return (
    <div className="hitl-modal-overlay" onClick={onClose}>
      <div className="hitl-modal-content" onClick={(e) => e.stopPropagation()}>
        <div className="hitl-modal-header">
          <h2>🔄 Human-in-the-Loop Workflow</h2>
          <button className="hitl-close-btn" onClick={onClose}>×</button>
        </div>

        {loading ? (
          <div className="hitl-loading">Loading workflow data...</div>
        ) : workflowData ? (
          <>
            <div className="hitl-tabs">
              <button
                className={`hitl-tab ${activeTab === 'approvals' ? 'active' : ''}`}
                onClick={() => setActiveTab('approvals')}
              >
                ⚠️ Approvals ({workflowData.approval_queue.length})
              </button>
              <button
                className={`hitl-tab ${activeTab === 'edits' ? 'active' : ''}`}
                onClick={() => setActiveTab('edits')}
              >
                ✏️ Edits
              </button>
              <button
                className={`hitl-tab ${activeTab === 'mappings' ? 'active' : ''}`}
                onClick={() => setActiveTab('mappings')}
              >
                🗺️ Store Mappings
              </button>
            </div>

            <div className="hitl-tab-content">
              {activeTab === 'approvals' && (
                <div className="hitl-approvals">
                  {workflowData.approval_queue.length === 0 ? (
                    <div className="hitl-empty">No items requiring approval</div>
                  ) : (
                    workflowData.approval_queue.map((item, idx) => (
                      <div key={idx} className="hitl-approval-item">
                        <div className="hitl-item-header">
                          <div>
                            <strong>{item.article}</strong>
                            {item.store_id && <span className="hitl-store-badge">{item.store_id}</span>}
                            <span className={`hitl-level-badge ${item.level}`}>
                              {item.level === 'aggregate' ? '📊 Aggregate' : '🏪 Store Level'}
                            </span>
                          </div>
                          <div className="hitl-variance">
                            Variance: <span className={item.variance_pct > workflowData.variance_threshold_pct ? 'high' : 'low'}>
                              {item.variance_pct.toFixed(1)}%
                            </span>
                          </div>
                        </div>
                        <div className="hitl-item-details">
                          <div className="hitl-quantity-comparison">
                            <span>Original: <strong>{item.original.toFixed(0)}</strong></span>
                            <span>→</span>
                            <span>Edited: <strong>{item.edited.toFixed(0)}</strong></span>
                            <span className="hitl-diff">
                              ({item.edited > item.original ? '+' : ''}{(item.edited - item.original).toFixed(0)})
                            </span>
                          </div>
                          <div className="hitl-actions">
                            <button
                              className="hitl-btn hitl-btn-approve"
                              onClick={() => handleApprove(item.article, item.store_id)}
                            >
                              ✅ Approve
                            </button>
                            <button
                              className="hitl-btn hitl-btn-reject"
                              onClick={() => handleReject(item.article, item.store_id)}
                            >
                              ❌ Reject
                            </button>
                          </div>
                        </div>
                      </div>
                    ))
                  )}
                </div>
              )}

              {activeTab === 'edits' && (
                <div className="hitl-edits">
                  <div className="hitl-section">
                    <h3>Aggregate Edits</h3>
                    {Object.keys(workflowData.aggregate_edits).length === 0 ? (
                      <div className="hitl-empty">No aggregate edits</div>
                    ) : (
                      Object.entries(workflowData.aggregate_edits).map(([article, edit]: [string, any]) => (
                        <div key={article} className="hitl-edit-item">
                          <div className="hitl-edit-header">
                            <strong>{article}</strong>
                            <span className={`hitl-status-badge ${edit.approval_status}`}>
                              {edit.approval_status}
                            </span>
                          </div>
                          <div className="hitl-edit-details">
                            <span>Original: {edit.original_quantity.toFixed(0)}</span>
                            <span>Edited: {edit.edited_quantity.toFixed(0)}</span>
                            <span>Variance: {edit.variance_pct.toFixed(1)}%</span>
                          </div>
                        </div>
                      ))
                    )}
                  </div>

                  <div className="hitl-section">
                    <h3>Store Level Edits</h3>
                    {Object.keys(workflowData.store_level_edits).length === 0 ? (
                      <div className="hitl-empty">No store level edits</div>
                    ) : (
                      Object.entries(workflowData.store_level_edits).map(([article, stores]: [string, any]) => (
                        <div key={article} className="hitl-article-edits">
                          <strong>{article}</strong>
                          {Object.entries(stores).map(([storeId, edit]: [string, any]) => (
                            <div key={storeId} className="hitl-store-edit">
                              <span className="hitl-store-name">{storeId}</span>
                              <span>Original: {edit.original_quantity.toFixed(0)}</span>
                              <span>Edited: {edit.edited_quantity.toFixed(0)}</span>
                              <span className={`hitl-status-badge ${edit.approval_status}`}>
                                {edit.approval_status}
                              </span>
                            </div>
                          ))}
                        </div>
                      ))
                    )}
                  </div>
                </div>
              )}

              {activeTab === 'mappings' && (
                <div className="hitl-mappings">
                  <div className="hitl-section">
                    <h3>Store Mappings</h3>
                    {Object.keys(workflowData.store_mappings).length === 0 ? (
                      <div className="hitl-empty">No store mappings</div>
                    ) : (
                      Object.entries(workflowData.store_mappings).map(([newStore, mapping]: [string, any]) => (
                        <div key={newStore} className="hitl-mapping-item">
                          <strong>{newStore}</strong> → <strong>{mapping.reference_store_id}</strong>
                          <span className="hitl-mapping-date">
                            Created: {new Date(mapping.created_at).toLocaleString()}
                          </span>
                        </div>
                      ))
                    )}
                  </div>
                </div>
              )}
            </div>

            <div className="hitl-modal-footer">
              <button className="hitl-btn hitl-btn-secondary" onClick={onClose}>
                Close
              </button>
              <button className="hitl-btn hitl-btn-primary" onClick={handleFinalize}>
                ✅ Finalize & Apply Changes
              </button>
            </div>
          </>
        ) : (
          <div className="hitl-error">Failed to load workflow data</div>
        )}
      </div>
    </div>
  );
};

export default HITLWorkflowModal;
