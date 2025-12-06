# Human-in-the-Loop (HITL) Workflow Guide

This document describes the complete Human-in-the-Loop workflow implementation for the Demand Forecasting Agent, keeping all capabilities intact.

## Overview

The HITL workflow enables human oversight and approval at key decision points:
1. **Store Mapping**: Add new stores before forecasting
2. **Aggregate Editing**: Edit quantities at aggregate level
3. **Store-Level Editing**: Edit allocations at store-style level
4. **Approval Workflow**: Automatic approval (<5% variance) or flag for approval (≥5% variance)
5. **Dynamic Rebalancing**: Automatic rebalancing when quantities are edited
6. **Audit Trail**: Complete tracking of all changes

---

## Step-by-Step Workflow

### Step 1: Add New Store Mapping (Before Forecasting)

**Purpose**: Map new store IDs to existing stores for reference

**UI Components Required**:
- Text input field for new store ID
- Dropdown list of available stores (from input data)
- "Add Mapping" button

**API Call**:
```python
# Get available stores for dropdown
available_stores = agent.get_available_stores_for_mapping(sales_data)

# Add new store mapping
mapping = agent.add_new_store_mapping(
    new_store_id="store_026",
    reference_store_id="store_001",  # Selected from dropdown
    user_id="user123"
)
```

**Output**:
```python
{
    'new_store_id': 'store_026',
    'reference_store_id': 'store_001',
    'created_at': '2025-12-05T10:30:00',
    'created_by': 'user123',
    'status': 'active'
}
```

**Audit Trail Entry**: Automatically created

---

### Step 2: Audit Trail (Metadata)

**Purpose**: Track all actions for compliance and review

**UI Components Required**:
- Audit Trail tab/page
- Filter options (by article, store, action type, date range)
- Export functionality

**API Call**:
```python
# Get full audit trail
audit_trail = agent.get_audit_trail()

# Get filtered audit trail
filtered_audit = agent.get_audit_trail(
    article="TS-114",
    store_id="store_001",
    action_type="aggregate_quantity_edited"
)

# Export metadata
metadata = agent.export_hitl_metadata()
```

**Audit Trail Structure**:
```python
[
    {
        'timestamp': '2025-12-05T10:30:00',
        'action': 'store_mapping_added',
        'user_id': 'user123',
        'details': {
            'new_store_id': 'store_026',
            'reference_store_id': 'store_001',
            ...
        }
    },
    {
        'timestamp': '2025-12-05T10:35:00',
        'action': 'aggregate_quantity_edited',
        'user_id': 'user123',
        'details': {
            'article': 'TS-114',
            'original_quantity': 2200,
            'edited_quantity': 2400,
            'variance_pct': 9.09,
            ...
        }
    }
]
```

---

### Step 3: Display Output with Editable Aggregate Quantities

**Purpose**: Show forecast results with editable quantity fields at aggregate level

**UI Components Required**:
- Table displaying article-level metrics
- Editable quantity field for each article
- "Save Changes" button
- Real-time variance calculation display

**Display Table**:

| Article | Style Code | Color | Total Qty (Editable) | Original Qty | Variance | Status |
|---------|------------|-------|---------------------|-------------|----------|--------|
| TS-114 | TSH-001 | White | `[2400]` | 2200 | +9.09% | ⚠️ Needs Approval |
| TS-203 | TSH-002 | Black | `[1800]` | 1800 | 0.00% | ✅ Auto-Approved |
| TS-301 | TSH-003 | Blue | `[1200]` | 1200 | 0.00% | ✅ Auto-Approved |

**API Call**:
```python
# Edit aggregate quantity
edit_result = agent.edit_aggregate_quantity(
    article="TS-114",
    edited_quantity=2400,
    user_id="user123"
)
```

**Output**:
```python
{
    'article': 'TS-114',
    'original_quantity': 2200,
    'edited_quantity': 2400,
    'variance': 0.0909,
    'variance_pct': 9.09,
    'approval_status': 'needs_approval',  # > 5% variance
    'approval_required': True,
    'edited_by': 'user123',
    'edited_at': '2025-12-05T10:40:00'
}
```

---

### Step 4: Auto-Approve or Flag for Approval

**Rules**:
- **Variance < 5%**: Auto-approve → Re-run allocation algorithm
- **Variance ≥ 5%**: Flag for approval → Wait for category head approval

**API Call** (Automatic):
```python
# If variance < 5%, automatically:
# 1. Auto-approve
# 2. Re-run allocation algorithm
# 3. Update store-level allocations

# If variance >= 5%:
# 1. Flag for approval
# 2. Add to approval queue
# 3. Wait for approval before re-running
```

**Approval Queue**:
```python
approval_queue = agent.get_approval_queue()
# Returns items with variance >= 5%
```

---

### Step 5: Display Store-Level Allocation with Editable Fields

**Purpose**: Show allocations at store-style level with editable quantities

**UI Components Required**:
- Expandable table showing store-level allocations
- Editable quantity fields for each store-style combination
- Total quantity validation (cannot exceed aggregate)
- Real-time rebalancing preview

**Display Table**:

| Article | Store ID | Style Code | Color | Segment | Family | Class | Brick | Quantity (Editable) | Original | Variance | Status |
|---------|----------|------------|-------|---------|--------|-------|-------|-------------------|----------|----------|--------|
| TS-114 | store_001 | TSH-001 | White | Casual | T-Shirts | Basic | Men's Apparel | `[250]` | 220 | +13.64% | ⚠️ Needs Approval |
| TS-114 | store_002 | TSH-001 | White | Casual | T-Shirts | Basic | Men's Apparel | `[200]` | 200 | 0.00% | ✅ Auto-Approved |
| TS-114 | store_003 | TSH-001 | White | Casual | T-Shirts | Basic | Men's Apparel | `[150]` | 150 | 0.00% | ✅ Auto-Approved |

**API Call**:
```python
# Edit store-level quantity
edit_result = agent.edit_store_level_quantity(
    article="TS-114",
    store_id="store_001",
    edited_quantity=250,
    user_id="user123"
)
```

**Output**:
```python
{
    'article': 'TS-114',
    'store_id': 'store_001',
    'original_quantity': 220,
    'edited_quantity': 250,
    'variance': 0.1364,
    'variance_pct': 13.64,
    'approval_status': 'needs_approval',
    'approval_required': True,
    'edited_by': 'user123',
    'edited_at': '2025-12-05T10:45:00'
}
```

---

### Step 6: Dynamic Rebalancing

**Purpose**: Automatically rebalance other stores when one store is edited

**Rules**:
- Total quantity cannot exceed aggregate forecasted quantity
- Other stores are proportionally adjusted
- Rounding differences are handled

**API Call** (Automatic):
```python
# When store-level quantity is edited:
# 1. Validate: edited_qty <= total_forecasted_qty
# 2. Calculate remaining quantity
# 3. Rebalance other stores proportionally
# 4. Update allocations

# Rebalancing happens automatically in edit_store_level_quantity()
```

**Rebalancing Example**:
```
Original Allocation:
  store_001: 220 units
  store_002: 200 units
  store_003: 150 units
  Total: 570 units

User edits store_001 to 250 units:
  store_001: 250 units (edited)
  store_002: 200 → 185 units (rebalanced)
  store_003: 150 → 135 units (rebalanced)
  Total: 570 units (unchanged)
```

---

### Step 7: Approval and Final Output Generation

**Purpose**: Separate approved and pending items, generate final output

**Rules**:
- **Variance < 5%**: Auto-approve
- **Variance ≥ 5%**: Needs approval from category head
- Generate separate outputs for approved and pending items

**API Call**:
```python
# Approve an item
approval = agent.approve_item(
    article="TS-114",
    store_id="store_001",  # None for aggregate level
    approver_id="category_head_001"
)

# Generate final output
final_output = agent.generate_final_output_with_approvals()
```

**Final Output Structure**:
```python
{
    'approved_allocations': {
        'TS-114': {
            'store_002': {
                'quantity': 200,
                'was_edited': False,
                'edit_record': None,
                # ... other allocation details
            },
            'store_003': {
                'quantity': 150,
                'was_edited': False,
                # ...
            }
        }
    },
    'pending_approvals': {
        'TS-114': {
            'store_001': {
                'quantity': 250,
                'was_edited': True,
                'edit_record': {
                    'original_quantity': 220,
                    'edited_quantity': 250,
                    'variance_pct': 13.64,
                    'approval_status': 'needs_approval'
                },
                # ...
            }
        }
    },
    'variances': [
        {
            'article': 'TS-114',
            'store_id': 'store_001',
            'type': 'store_level',
            'original': 220,
            'edited': 250,
            'variance_pct': 13.64,
            'status': 'needs_approval'
        }
    ],
    'summary': {
        'total_articles': 3,
        'approved_articles': 2,
        'pending_articles': 1,
        'total_variances': 1,
        'variances_requiring_approval': 1
    }
}
```

**UI Display**:
- **Approved Allocations**: Display in main table (green highlight)
- **Need Approval**: Display in separate "Pending Approvals" section (yellow/red highlight)

---

### Step 8: Audit Trail for Final Output

**Purpose**: Track system suggested vs human finalized quantities

**API Call**:
```python
# Get system vs human comparison for an article
comparison = agent.get_system_vs_human_comparison(article="TS-114")
```

**Comparison Structure**:
```python
{
    'article': 'TS-114',
    'aggregate_level': {
        'system_suggested': 2200,
        'human_finalized': 2400,
        'variance': 0.0909,
        'variance_pct': 9.09,
        'status': 'approved'
    },
    'store_level': {
        'store_001': {
            'system_suggested': 220,
            'human_finalized': 250,
            'variance': 0.1364,
            'variance_pct': 13.64,
            'status': 'needs_approval'
        },
        'store_002': {
            'system_suggested': 200,
            'human_finalized': 200,
            'variance': 0.0,
            'variance_pct': 0.0,
            'status': 'auto_approved'
        }
    }
}
```

**Audit Trail Tab Display**:

| Timestamp | Action | Article | Store | System Qty | Human Qty | Variance | User | Status |
|-----------|--------|---------|-------|------------|-----------|----------|------|--------|
| 2025-12-05 10:40:00 | Aggregate Edit | TS-114 | - | 2200 | 2400 | +9.09% | user123 | Approved |
| 2025-12-05 10:45:00 | Store Edit | TS-114 | store_001 | 220 | 250 | +13.64% | user123 | Pending |
| 2025-12-05 10:50:00 | Approval | TS-114 | store_001 | 220 | 250 | +13.64% | category_head | Approved |

---

### Step 9: Variance Highlights

**Purpose**: Highlight all variances for quick attention

**API Call**:
```python
# Get variance highlights
highlights = agent.get_variance_highlights()
```

**Highlights Structure**:
```python
{
    'aggregate_variances': [
        {
            'article': 'TS-114',
            'original': 2200,
            'edited': 2400,
            'variance_pct': 9.09,
            'status': 'needs_approval'
        }
    ],
    'store_level_variances': [
        {
            'article': 'TS-114',
            'store_id': 'store_001',
            'original': 220,
            'edited': 250,
            'variance_pct': 13.64,
            'status': 'needs_approval'
        }
    ],
    'critical_variances': [  # > 10%
        {
            'article': 'TS-114',
            'store_id': 'store_001',
            'variance_pct': 13.64,
            ...
        }
    ],
    'moderate_variances': [  # 5-10%
        {
            'article': 'TS-114',
            'variance_pct': 9.09,
            ...
        }
    ]
}
```

**UI Display**:
- **Critical Variances (>10%)**: Red highlight, top priority
- **Moderate Variances (5-10%)**: Yellow highlight, medium priority
- **Auto-Approved (<5%)**: Green highlight, no action needed

---

## Complete Workflow Example

```python
from agents.demand_forecasting_agent import DemandForecastingAgent
from agents.hitl_workflow import HITLWorkflow

# Initialize agent with HITL enabled
agent = DemandForecastingAgent(
    llama_client=llama_client,
    enable_hitl=True,
    variance_threshold=0.05  # 5%
)

# Step 1: Add new store mapping (before forecasting)
available_stores = agent.get_available_stores_for_mapping(sales_data)
mapping = agent.add_new_store_mapping(
    new_store_id="store_026",
    reference_store_id="store_001",
    user_id="user123"
)

# Step 2: Run forecasting (store mappings applied automatically)
results, sensitivity = agent.run(
    comparables, sales_data, inventory_data, price_data,
    price_options, product_attributes, forecast_horizon_days=60
)

# Step 3: Display results with editable aggregate quantities
# User edits TS-114 quantity from 2200 to 2400 (9.09% variance)
edit_result = agent.edit_aggregate_quantity(
    article="TS-114",
    edited_quantity=2400,
    user_id="user123"
)
# Result: Flagged for approval (variance > 5%)

# Step 4: Category head approves aggregate edit
approval = agent.approve_item(
    article="TS-114",
    store_id=None,  # Aggregate level
    approver_id="category_head_001"
)

# Step 5: Re-run allocation (automatic after approval)
# Allocations rebalanced for TS-114

# Step 6: User edits store-level quantity
store_edit = agent.edit_store_level_quantity(
    article="TS-114",
    store_id="store_001",
    edited_quantity=250,  # Original: 220 (13.64% variance)
    user_id="user123"
)
# Other stores automatically rebalanced

# Step 7: Generate final output
final_output = agent.generate_final_output_with_approvals()

# Step 8: View audit trail
audit_trail = agent.get_audit_trail(article="TS-114")
comparison = agent.get_system_vs_human_comparison(article="TS-114")

# Step 9: View variance highlights
highlights = agent.get_variance_highlights()
```

---

## UI Integration Checklist

### Required UI Components:

- [ ] **Store Mapping Section**
  - [ ] Text input for new store ID
  - [ ] Dropdown for reference store selection
  - [ ] "Add Mapping" button
  - [ ] List of mapped stores

- [ ] **Forecast Results Display**
  - [ ] Article-level metrics table
  - [ ] Editable quantity fields (aggregate level)
  - [ ] Real-time variance calculation
  - [ ] Status indicators (Auto-Approved / Needs Approval)

- [ ] **Store-Level Allocation Display**
  - [ ] Expandable table showing store-style combinations
  - [ ] All article attributes displayed (style_code, color, segment, family, class, brick)
  - [ ] Editable quantity fields
  - [ ] Total quantity validation
  - [ ] Rebalancing preview

- [ ] **Approval Queue**
  - [ ] List of items needing approval
  - [ ] Approve/Reject buttons
  - [ ] Approval comments field

- [ ] **Final Output Display**
  - [ ] Approved allocations section (green highlight)
  - [ ] Pending approvals section (yellow/red highlight)
  - [ ] "Need Approval" badge for pending items

- [ ] **Audit Trail Tab**
  - [ ] Complete audit log
  - [ ] Filters (article, store, action, date)
  - [ ] System vs Human comparison view
  - [ ] Export functionality

- [ ] **Variance Highlights**
  - [ ] Critical variances panel (red)
  - [ ] Moderate variances panel (yellow)
  - [ ] Quick action buttons

---

## Data Structures

### Store Mapping
```python
{
    'new_store_id': str,
    'reference_store_id': str,
    'created_at': str (ISO format),
    'created_by': str,
    'status': 'active'
}
```

### Edit Record
```python
{
    'article': str,
    'store_id': str (optional),
    'original_quantity': float,
    'edited_quantity': float,
    'variance': float,
    'variance_pct': float,
    'approval_status': 'auto_approved' | 'needs_approval' | 'approved' | 'rejected',
    'approval_required': bool,
    'edited_by': str,
    'edited_at': str (ISO format),
    'approved_by': str (optional),
    'approved_at': str (optional)
}
```

### Final Output
```python
{
    'approved_allocations': Dict[str, Dict[str, Dict]],
    'pending_approvals': Dict[str, Dict[str, Dict]],
    'variances': List[Dict],
    'summary': Dict,
    'generated_at': str (ISO format)
}
```

---

## Validation Rules

1. **Store Mapping**: New store ID must be unique
2. **Aggregate Edit**: No upper limit (user discretion)
3. **Store-Level Edit**: Cannot exceed total forecasted quantity
4. **Variance Threshold**: 5% (configurable)
5. **Rebalancing**: Total quantity must remain constant

---

## Error Handling

- **Store already mapped**: Raise ValueError with clear message
- **Quantity exceeds total**: Raise ValueError with validation message
- **Article not found**: Raise ValueError with article identifier
- **HITL not enabled**: Raise ValueError suggesting to enable HITL

---

## Notes

- All capabilities remain intact
- HITL workflow is optional (can be disabled)
- Audit trail is persistent (can be exported/imported)
- Rebalancing is automatic and proportional
- Variance calculation: `|edited - original| / original * 100`

