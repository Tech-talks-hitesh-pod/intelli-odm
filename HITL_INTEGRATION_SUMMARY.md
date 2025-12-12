# Human-in-the-Loop (HITL) Integration Summary

## Overview

The Demand Forecasting Agent now includes a complete Human-in-the-Loop workflow system that enables human oversight, approval workflows, and audit trails while keeping all existing capabilities intact.

## Key Features Implemented

### 1. ✅ Store Mapping (Step 1)
- **Function**: `add_new_store_mapping(new_store_id, reference_store_id, user_id)`
- **Purpose**: Add new store IDs before forecasting with reference to existing stores
- **UI Requirements**: Text input + Dropdown for reference store selection
- **Audit Trail**: Automatically tracked

### 2. ✅ Audit Trail (Step 2)
- **Function**: `get_audit_trail(article, store_id, action_type)`
- **Purpose**: Complete metadata tracking of all actions
- **Features**: Filterable, exportable, includes timestamps and user IDs
- **Storage**: In-memory (can be persisted via `export_metadata()`)

### 3. ✅ Editable Aggregate Quantities (Step 3)
- **Function**: `edit_aggregate_quantity(article, edited_quantity, user_id)`
- **Purpose**: Allow users to edit quantities at aggregate level
- **UI Requirements**: Editable fields in article-level metrics table
- **Output**: Edit record with variance calculation

### 4. ✅ Auto-Approve or Flag for Approval (Step 4)
- **Rule**: Variance < 5% = Auto-approve, Variance ≥ 5% = Flag for approval
- **Auto-Approval**: Automatically re-runs allocation algorithm
- **Flagged Items**: Added to approval queue for category head review

### 5. ✅ Store-Level Editable Allocations (Step 5)
- **Function**: `edit_store_level_quantity(article, store_id, edited_quantity, user_id)`
- **Purpose**: Edit quantities at store-style level
- **UI Requirements**: Expandable table with all article attributes (style_code, color, segment, family, class, brick)
- **Validation**: Cannot exceed total forecasted quantity

### 6. ✅ Dynamic Rebalancing (Step 6)
- **Function**: Automatic in `edit_store_level_quantity()`
- **Purpose**: Automatically rebalance other stores when one is edited
- **Algorithm**: Proportional scaling with rounding adjustment
- **Constraint**: Total quantity remains constant

### 7. ✅ Approval Workflow (Step 7)
- **Function**: `approve_item(article, store_id, approver_id)`
- **Function**: `generate_final_output_with_approvals()`
- **Purpose**: Separate approved and pending items
- **Output**: Two sections - "Approved Allocations" and "Need Approval"

### 8. ✅ Audit Trail for Final Output (Step 8)
- **Function**: `get_system_vs_human_comparison(article)`
- **Purpose**: Track system suggested vs human finalized quantities
- **Display**: Complete comparison at aggregate and store level
- **Location**: Audit Trail tab in UI

### 9. ✅ Variance Highlights (Step 9)
- **Function**: `get_variance_highlights()`
- **Purpose**: Highlight all variances for quick attention
- **Categories**: Critical (>10%), Moderate (5-10%), Auto-Approved (<5%)
- **UI**: Color-coded highlights (Red/Yellow/Green)

---

## API Methods Added to DemandForecastingAgent

```python
# Store Mapping
agent.add_new_store_mapping(new_store_id, reference_store_id, user_id)
agent.get_available_stores_for_mapping(sales_data)

# Aggregate Editing
agent.edit_aggregate_quantity(article, edited_quantity, user_id)

# Store-Level Editing
agent.edit_store_level_quantity(article, store_id, edited_quantity, user_id)

# Approval Workflow
agent.get_approval_queue()
agent.approve_item(article, store_id, approver_id)
agent.generate_final_output_with_approvals()

# Audit Trail
agent.get_audit_trail(article, store_id, action_type)
agent.get_system_vs_human_comparison(article)
agent.export_hitl_metadata()

# Variance Highlights
agent.get_variance_highlights()
```

---

## Workflow Diagram

```
┌─────────────────────────────────────────────────────────────┐
│ Step 1: Add New Store Mapping (Before Forecasting)          │
│ - User enters new store ID                                  │
│ - User selects reference store from dropdown                │
│ - Mapping saved with audit trail                            │
└────────────────────┬──────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│ Step 2: Run Forecasting                                     │
│ - Store mappings automatically applied                      │
│ - Forecasts generated for new stores based on reference      │
│ - Audit trail maintained                                    │
└────────────────────┬──────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│ Step 3: Display Results with Editable Aggregate Quantities │
│ - Article-level table with editable fields                  │
│ - Real-time variance calculation                            │
│ - Status indicators (Auto-Approved / Needs Approval)        │
└────────────────────┬──────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│ Step 4: Auto-Approve or Flag                                │
│ - Variance < 5%: Auto-approve → Re-run allocation          │
│ - Variance ≥ 5%: Flag for approval → Wait                  │
└────────────────────┬──────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│ Step 5: Display Store-Level Allocations                    │
│ - Expandable table with all article attributes              │
│ - Editable quantity fields                                  │
│ - Total quantity validation                                 │
└────────────────────┬──────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│ Step 6: Dynamic Rebalancing                                 │
│ - User edits store-level quantity                           │
│ - Other stores automatically rebalanced                     │
│ - Total quantity constraint enforced                        │
└────────────────────┬──────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│ Step 7: Approval and Final Output                           │
│ - Store-level variance check (<5% auto-approve, ≥5% flag)  │
│ - Generate final output with approved/pending separation    │
│ - "Need Approval" section for pending items                │
└────────────────────┬──────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│ Step 8: Audit Trail                                         │
│ - System vs Human comparison                                │
│ - Complete action history                                   │
│ - Exportable metadata                                        │
└────────────────────┬──────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│ Step 9: Variance Highlights                                 │
│ - Critical variances (>10%) - Red highlight                │
│ - Moderate variances (5-10%) - Yellow highlight            │
│ - Quick attention for approvers                             │
└─────────────────────────────────────────────────────────────┘
```

---

## Data Flow

```
Input Data
    ↓
[Store Mapping] ← User Input (Step 1)
    ↓
Forecasting Agent
    ↓
[Forecast Results] ← System Generated
    ↓
[Aggregate Edit] ← User Input (Step 3)
    ↓
[Variance Check] ← Automatic (Step 4)
    ├─ < 5% → Auto-Approve → Re-run Allocation
    └─ ≥ 5% → Flag for Approval → Wait
    ↓
[Store-Level Edit] ← User Input (Step 5)
    ↓
[Dynamic Rebalancing] ← Automatic (Step 6)
    ↓
[Approval Workflow] ← Category Head (Step 7)
    ↓
[Final Output] ← Approved + Pending Separation
    ↓
[Audit Trail] ← Complete History (Step 8)
    ↓
[Variance Highlights] ← Quick Review (Step 9)
```

---

## Example Usage

```python
# Initialize with HITL enabled
agent = DemandForecastingAgent(
    llama_client=llama_client,
    enable_hitl=True,
    variance_threshold=0.05  # 5%
)

# Step 1: Add store mapping
agent.add_new_store_mapping("store_026", "store_001", "user123")

# Step 2: Run forecasting (mappings applied automatically)
results, sensitivity = agent.run(...)

# Step 3: User edits aggregate quantity
edit = agent.edit_aggregate_quantity("TS-114", 2400, "user123")
# Variance: 9.09% → Flagged for approval

# Step 4: Category head approves
agent.approve_item("TS-114", None, "category_head_001")
# Allocation re-run automatically

# Step 5: User edits store-level quantity
store_edit = agent.edit_store_level_quantity("TS-114", "store_001", 250, "user123")
# Other stores rebalanced automatically

# Step 7: Generate final output
final = agent.generate_final_output_with_approvals()
# Returns: approved_allocations, pending_approvals, variances

# Step 8: View audit trail
audit = agent.get_audit_trail(article="TS-114")
comparison = agent.get_system_vs_human_comparison("TS-114")

# Step 9: View highlights
highlights = agent.get_variance_highlights()
```

---

## Output Structure with HITL

```python
{
    'forecast_results': {...},  # Original forecast results
    'recommendations': {
        'article_level_metrics': {...},
        'store_allocations': {...},
        # ... other recommendations
    },
    'hitl_metadata': {
        'enabled': True,
        'available_stores': ['store_001', 'store_002', ...],
        'store_mappings': {
            'store_026': {
                'reference_store_id': 'store_001',
                'created_at': '2025-12-05T10:30:00',
                'created_by': 'user123'
            }
        },
        'variance_threshold': 0.05
    },
    'final_output': {  # After HITL workflow
        'approved_allocations': {...},
        'pending_approvals': {...},
        'variances': [...],
        'summary': {...}
    }
}
```

---

## UI Integration Points

1. **Store Mapping UI**: Before forecasting starts
2. **Results Display UI**: After forecasting completes
3. **Edit UI**: Inline editing in tables
4. **Approval UI**: Separate approval queue interface
5. **Audit Trail UI**: Dedicated tab/page
6. **Variance Highlights UI**: Dashboard/widget

---

## Validation Rules

| Rule | Description | Enforcement |
|------|-------------|-------------|
| Store Mapping | New store ID must be unique | ValueError if duplicate |
| Aggregate Edit | No upper limit | User discretion |
| Store Edit | Cannot exceed total forecasted | ValueError if exceeds |
| Variance Threshold | 5% (configurable) | Automatic flagging |
| Rebalancing | Total must remain constant | Automatic adjustment |

---

## Status Indicators

- ✅ **Auto-Approved**: Variance < 5%, automatically processed
- ⚠️ **Needs Approval**: Variance ≥ 5%, waiting for category head
- ✅ **Approved**: Category head approved
- ❌ **Rejected**: Category head rejected (future enhancement)

---

## All Capabilities Intact

✅ Model Selection (Task 1)  
✅ Factor Analysis (Task 2)  
✅ Store-Level Forecasting (Task 3)  
✅ Optimization (Rate of Sale, Sell-Through, Margin)  
✅ Fallback Mechanisms  
✅ Pydantic Validation  
✅ Token Optimization  
✅ Article-Level Metrics  
✅ Store Universe Validation  
✅ **NEW**: Human-in-the-Loop Workflow  
✅ **NEW**: Approval System  
✅ **NEW**: Audit Trail  
✅ **NEW**: Dynamic Rebalancing  

---

## Files Created/Modified

1. **`agents/hitl_workflow.py`**: New HITL workflow module
2. **`agents/demand_forecasting_agent.py`**: Enhanced with HITL integration
3. **`HITL_WORKFLOW_GUIDE.md`**: Complete workflow documentation
4. **`HITL_INTEGRATION_SUMMARY.md`**: This summary document

---

## Next Steps for UI Development

1. Implement store mapping interface
2. Create editable tables for aggregate and store-level quantities
3. Build approval queue interface
4. Develop audit trail viewer
5. Add variance highlighting widgets
6. Implement real-time rebalancing preview
7. Add export functionality for audit trail

