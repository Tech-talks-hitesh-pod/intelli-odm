# agents/hitl_workflow.py

"""
Human-in-the-Loop (HITL) Workflow Module
Manages approval workflows, audit trails, and dynamic rebalancing
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from datetime import datetime
from enum import Enum
import json

class ApprovalStatus(Enum):
    """Approval status enumeration"""
    PENDING = "pending"
    AUTO_APPROVED = "auto_approved"
    APPROVED = "approved"
    REJECTED = "rejected"
    NEEDS_APPROVAL = "needs_approval"

class VarianceType(Enum):
    """Variance type enumeration"""
    AGGREGATE = "aggregate"
    STORE_LEVEL = "store_level"
    STYLE_LEVEL = "style_level"

class HITLWorkflow:
    """
    Human-in-the-Loop Workflow Manager
    Handles store mapping, approvals, audit trails, and rebalancing
    """
    
    def __init__(self, variance_threshold: float = 0.05):
        """
        Initialize HITL Workflow
        
        Args:
            variance_threshold: Variance threshold for auto-approval (default 5% = 0.05)
        """
        self.variance_threshold = variance_threshold
        self.store_mappings = {}  # {new_store_id: reference_store_id}
        self.audit_trail = []
        self.approval_queue = []
        self.aggregate_edits = {}  # {article: edited_quantity}
        self.store_level_edits = {}  # {article: {store_id: edited_quantity}}
        self.final_allocations = {}
        
    def add_new_store_mapping(self, new_store_id: str, reference_store_id: str, 
                             user_id: str = "system") -> Dict[str, Any]:
        """
        Add a new store ID with mapping to reference store
        
        Args:
            new_store_id: New store identifier
            reference_store_id: Existing store to use as reference
            user_id: User who added the mapping
        
        Returns:
            Mapping record with audit information
        """
        if new_store_id in self.store_mappings:
            raise ValueError(f"Store {new_store_id} already exists in mappings")
        
        mapping_record = {
            'new_store_id': new_store_id,
            'reference_store_id': reference_store_id,
            'created_at': datetime.now().isoformat(),
            'created_by': user_id,
            'status': 'active'
        }
        
        self.store_mappings[new_store_id] = mapping_record
        
        # Add to audit trail
        self._add_audit_entry(
            action='store_mapping_added',
            details=mapping_record,
            user_id=user_id
        )
        
        return mapping_record
    
    def get_available_stores(self, sales_data: pd.DataFrame) -> List[str]:
        """
        Get list of available stores from input data for dropdown
        
        Args:
            sales_data: Sales data DataFrame
        
        Returns:
            List of store IDs
        """
        if sales_data is None or sales_data.empty:
            return []
        
        if 'store_id' in sales_data.columns:
            return sorted(sales_data['store_id'].unique().tolist())
        return []
    
    def apply_store_mappings_to_forecast(self, store_forecasts: Dict[str, Dict],
                                        sales_data: pd.DataFrame,
                                        inventory_data: pd.DataFrame,
                                        price_data: pd.DataFrame) -> Dict[str, Dict]:
        """
        Apply store mappings to forecasts - create forecasts for new stores
        based on reference stores
        
        Args:
            store_forecasts: Original store-level forecasts
            sales_data: Sales data
            inventory_data: Inventory data
            price_data: Price data
        
        Returns:
            Updated forecasts with new stores included
        """
        updated_forecasts = store_forecasts.copy()
        
        for new_store_id, mapping in self.store_mappings.items():
            reference_store_id = mapping['reference_store_id']
            
            if reference_store_id in store_forecasts:
                # Copy reference store forecasts for new store
                reference_forecasts = store_forecasts[reference_store_id]
                new_store_forecasts = {}
                
                for article, forecast in reference_forecasts.items():
                    # Create new forecast based on reference
                    new_forecast = forecast.copy()
                    new_forecast['store_id'] = new_store_id
                    new_forecast['reference_store_id'] = reference_store_id
                    new_forecast['is_mapped_store'] = True
                    new_forecast['mapping_created_at'] = mapping['created_at']
                    
                    # Adjust quantities slightly (can be customized)
                    # For now, use same quantities as reference
                    new_store_forecasts[article] = new_forecast
                
                updated_forecasts[new_store_id] = new_store_forecasts
                
                # Add to audit trail
                self._add_audit_entry(
                    action='store_forecast_created_from_mapping',
                    details={
                        'new_store_id': new_store_id,
                        'reference_store_id': reference_store_id,
                        'articles': list(new_store_forecasts.keys())
                    }
                )
        
        return updated_forecasts
    
    def edit_aggregate_quantity(self, article: str, edited_quantity: float,
                               original_quantity: float, user_id: str = "user") -> Dict[str, Any]:
        """
        Edit aggregate quantity for an article
        
        Args:
            article: Article/SKU identifier
            edited_quantity: User-edited quantity
            original_quantity: Original system-forecasted quantity
            user_id: User making the edit
        
        Returns:
            Edit record with variance and approval status
        """
        variance = abs(edited_quantity - original_quantity) / original_quantity if original_quantity > 0 else 0
        variance_pct = variance * 100
        
        edit_record = {
            'article': article,
            'original_quantity': original_quantity,
            'edited_quantity': edited_quantity,
            'variance': variance,
            'variance_pct': variance_pct,
            'edited_at': datetime.now().isoformat(),
            'edited_by': user_id,
            'approval_status': ApprovalStatus.NEEDS_APPROVAL.value if variance > self.variance_threshold else ApprovalStatus.AUTO_APPROVED.value,
            'approval_required': variance > self.variance_threshold
        }
        
        self.aggregate_edits[article] = edit_record
        
        # Add to audit trail
        self._add_audit_entry(
            action='aggregate_quantity_edited',
            details=edit_record,
            user_id=user_id
        )
        
        return edit_record
    
    def rebalance_store_allocations(self, article: str, total_quantity: float,
                                   store_allocations: Dict[str, Dict],
                                   original_store_allocations: Dict[str, Dict]) -> Dict[str, Dict]:
        """
        Rebalance store allocations when aggregate quantity changes
        
        Args:
            article: Article identifier
            total_quantity: New total quantity (may be edited)
            store_allocations: Current store allocations
            original_store_allocations: Original store allocations
        
        Returns:
            Rebalanced store allocations
        """
        # Calculate original total
        original_total = sum(s.get('quantity', 0) for s in original_store_allocations.values())
        
        if original_total == 0:
            return store_allocations
        
        # Calculate scaling factor
        scale_factor = total_quantity / original_total
        
        # Rebalance proportionally
        rebalanced = {}
        allocated_total = 0
        
        for store_id, allocation in store_allocations.items():
            original_qty = allocation.get('quantity', 0)
            scaled_qty = original_qty * scale_factor
            
            # Round to nearest integer
            new_qty = max(0, round(scaled_qty))
            allocated_total += new_qty
            
            rebalanced[store_id] = allocation.copy()
            rebalanced[store_id]['quantity'] = new_qty
            rebalanced[store_id]['original_quantity'] = original_qty
            rebalanced[store_id]['scaling_factor'] = scale_factor
        
        # Adjust for rounding differences
        difference = total_quantity - allocated_total
        if abs(difference) > 0 and rebalanced:
            # Add/subtract difference to store with highest quantity
            max_store = max(rebalanced.items(), key=lambda x: x[1]['quantity'])
            rebalanced[max_store[0]]['quantity'] += int(difference)
        
        # Add to audit trail
        self._add_audit_entry(
            action='store_allocations_rebalanced',
            details={
                'article': article,
                'original_total': original_total,
                'new_total': total_quantity,
                'scale_factor': scale_factor,
                'rebalanced_allocations': {k: v['quantity'] for k, v in rebalanced.items()}
            }
        )
        
        return rebalanced
    
    def edit_store_level_quantity(self, article: str, store_id: str, 
                                 edited_quantity: float,
                                 original_quantity: float,
                                 total_forecasted_quantity: float,
                                 user_id: str = "user") -> Dict[str, Any]:
        """
        Edit store-level quantity allocation
        
        Args:
            article: Article identifier
            store_id: Store identifier
            edited_quantity: User-edited quantity
            original_quantity: Original allocated quantity
            total_forecasted_quantity: Total forecasted quantity for article
            user_id: User making the edit
        
        Returns:
            Edit record with validation and rebalancing info
        """
        # Validate: edited quantity cannot exceed total
        if edited_quantity > total_forecasted_quantity:
            raise ValueError(
                f"Edited quantity ({edited_quantity}) exceeds total forecasted quantity ({total_forecasted_quantity})"
            )
        
        variance = abs(edited_quantity - original_quantity) / original_quantity if original_quantity > 0 else 0
        variance_pct = variance * 100
        
        edit_record = {
            'article': article,
            'store_id': store_id,
            'original_quantity': original_quantity,
            'edited_quantity': edited_quantity,
            'variance': variance,
            'variance_pct': variance_pct,
            'edited_at': datetime.now().isoformat(),
            'edited_by': user_id,
            'approval_status': ApprovalStatus.NEEDS_APPROVAL.value if variance > self.variance_threshold else ApprovalStatus.AUTO_APPROVED.value,
            'approval_required': variance > self.variance_threshold
        }
        
        if article not in self.store_level_edits:
            self.store_level_edits[article] = {}
        self.store_level_edits[article][store_id] = edit_record
        
        # Add to audit trail
        self._add_audit_entry(
            action='store_level_quantity_edited',
            details=edit_record,
            user_id=user_id
        )
        
        return edit_record
    
    def rebalance_after_store_edit(self, article: str, edited_store_id: str,
                                   edited_quantity: float,
                                   store_allocations: Dict[str, Dict],
                                   total_forecasted_quantity: float) -> Dict[str, Dict]:
        """
        Dynamically rebalance allocations after store-level edit
        
        Args:
            article: Article identifier
            edited_store_id: Store that was edited
            edited_quantity: New quantity for edited store
            store_allocations: Current store allocations
            total_forecasted_quantity: Total forecasted quantity
        
        Returns:
            Rebalanced allocations
        """
        # Calculate current total (excluding edited store)
        other_stores_total = sum(
            s.get('quantity', 0) 
            for store_id, s in store_allocations.items() 
            if store_id != edited_store_id
        )
        
        # Calculate remaining quantity
        remaining_quantity = total_forecasted_quantity - edited_quantity
        
        if remaining_quantity < 0:
            raise ValueError("Remaining quantity cannot be negative after edit")
        
        # Rebalance other stores proportionally
        rebalanced = {}
        rebalanced[edited_store_id] = store_allocations[edited_store_id].copy()
        rebalanced[edited_store_id]['quantity'] = edited_quantity
        rebalanced[edited_store_id]['was_edited'] = True
        
        if other_stores_total > 0:
            scale_factor = remaining_quantity / other_stores_total
        else:
            scale_factor = 1.0
        
        allocated_other = 0
        for store_id, allocation in store_allocations.items():
            if store_id != edited_store_id:
                original_qty = allocation.get('quantity', 0)
                scaled_qty = original_qty * scale_factor
                new_qty = max(0, round(scaled_qty))
                allocated_other += new_qty
                
                rebalanced[store_id] = allocation.copy()
                rebalanced[store_id]['quantity'] = new_qty
                rebalanced[store_id]['original_quantity'] = original_qty
                rebalanced[store_id]['rebalanced'] = True
        
        # Adjust for rounding
        difference = remaining_quantity - allocated_other
        if abs(difference) > 0 and len(rebalanced) > 1:
            # Find store with highest quantity (excluding edited store)
            other_stores = {k: v for k, v in rebalanced.items() if k != edited_store_id}
            if other_stores:
                max_store = max(other_stores.items(), key=lambda x: x[1]['quantity'])
                rebalanced[max_store[0]]['quantity'] += int(difference)
        
        # Add to audit trail
        self._add_audit_entry(
            action='dynamic_rebalance_after_store_edit',
            details={
                'article': article,
                'edited_store_id': edited_store_id,
                'edited_quantity': edited_quantity,
                'remaining_quantity': remaining_quantity,
                'scale_factor': scale_factor,
                'rebalanced_allocations': {k: v['quantity'] for k, v in rebalanced.items()}
            }
        )
        
        return rebalanced
    
    def get_approval_queue(self) -> List[Dict[str, Any]]:
        """
        Get items requiring approval
        
        Returns:
            List of items needing approval
        """
        approval_items = []
        
        # Aggregate level approvals
        for article, edit in self.aggregate_edits.items():
            if edit['approval_required']:
                approval_items.append({
                    'level': 'aggregate',
                    'article': article,
                    'type': VarianceType.AGGREGATE.value,
                    'original': edit['original_quantity'],
                    'edited': edit['edited_quantity'],
                    'variance_pct': edit['variance_pct'],
                    'status': edit['approval_status'],
                    'edited_by': edit['edited_by'],
                    'edited_at': edit['edited_at']
                })
        
        # Store level approvals
        for article, store_edits in self.store_level_edits.items():
            for store_id, edit in store_edits.items():
                if edit['approval_required']:
                    approval_items.append({
                        'level': 'store_level',
                        'article': article,
                        'store_id': store_id,
                        'type': VarianceType.STORE_LEVEL.value,
                        'original': edit['original_quantity'],
                        'edited': edit['edited_quantity'],
                        'variance_pct': edit['variance_pct'],
                        'status': edit['approval_status'],
                        'edited_by': edit['edited_by'],
                        'edited_at': edit['edited_at']
                    })
        
        return approval_items
    
    def approve_item(self, article: str, store_id: Optional[str] = None,
                    approver_id: str = "category_head") -> Dict[str, Any]:
        """
        Approve an item (aggregate or store-level)
        
        Args:
            article: Article identifier
            store_id: Store ID (None for aggregate level)
            approver_id: Approver user ID
        
        Returns:
            Approval record
        """
        if store_id is None:
            # Aggregate level approval
            if article not in self.aggregate_edits:
                raise ValueError(f"No aggregate edit found for article {article}")
            
            edit = self.aggregate_edits[article]
            edit['approval_status'] = ApprovalStatus.APPROVED.value
            edit['approved_by'] = approver_id
            edit['approved_at'] = datetime.now().isoformat()
            
            approval_record = edit.copy()
        else:
            # Store level approval
            if article not in self.store_level_edits or store_id not in self.store_level_edits[article]:
                raise ValueError(f"No store-level edit found for article {article}, store {store_id}")
            
            edit = self.store_level_edits[article][store_id]
            edit['approval_status'] = ApprovalStatus.APPROVED.value
            edit['approved_by'] = approver_id
            edit['approved_at'] = datetime.now().isoformat()
            
            approval_record = edit.copy()
        
        # Add to audit trail
        self._add_audit_entry(
            action='item_approved',
            details={
                'article': article,
                'store_id': store_id,
                'approver_id': approver_id,
                'approval_record': approval_record
            },
            user_id=approver_id
        )
        
        return approval_record
    
    def generate_final_output(self, store_allocations: Dict[str, Dict[str, Dict]],
                              article_level_metrics: Dict[str, Dict]) -> Dict[str, Any]:
        """
        Generate final output with approved and pending items separated
        
        Args:
            store_allocations: Store-level allocations
            article_level_metrics: Article-level metrics
        
        Returns:
            Final output with approved and pending items
        """
        approved_allocations = {}
        pending_approvals = {}
        variances = []
        
        for article, stores in store_allocations.items():
            article_approved = {}
            article_pending = {}
            
            for store_id, allocation in stores.items():
                # Check if this store-article combination was edited
                was_edited = False
                edit_record = None
                
                if article in self.store_level_edits and store_id in self.store_level_edits[article]:
                    edit_record = self.store_level_edits[article][store_id]
                    was_edited = True
                
                allocation_data = allocation.copy()
                allocation_data['was_edited'] = was_edited
                allocation_data['edit_record'] = edit_record
                
                if was_edited and edit_record:
                    # Check approval status
                    if edit_record['approval_status'] == ApprovalStatus.APPROVED.value:
                        article_approved[store_id] = allocation_data
                    elif edit_record['approval_status'] == ApprovalStatus.AUTO_APPROVED.value:
                        article_approved[store_id] = allocation_data
                    else:
                        article_pending[store_id] = allocation_data
                        # Add to variances
                        variances.append({
                            'article': article,
                            'store_id': store_id,
                            'type': VarianceType.STORE_LEVEL.value,
                            'original': edit_record['original_quantity'],
                            'edited': edit_record['edited_quantity'],
                            'variance_pct': edit_record['variance_pct'],
                            'status': edit_record['approval_status']
                        })
                else:
                    # Not edited, auto-approved
                    article_approved[store_id] = allocation_data
            
            if article_approved:
                approved_allocations[article] = article_approved
            if article_pending:
                pending_approvals[article] = article_pending
        
        # Add aggregate level variances
        for article, edit in self.aggregate_edits.items():
            if edit['approval_status'] not in [ApprovalStatus.APPROVED.value, ApprovalStatus.AUTO_APPROVED.value]:
                variances.append({
                    'article': article,
                    'store_id': None,
                    'type': VarianceType.AGGREGATE.value,
                    'original': edit['original_quantity'],
                    'edited': edit['edited_quantity'],
                    'variance_pct': edit['variance_pct'],
                    'status': edit['approval_status']
                })
        
        final_output = {
            'approved_allocations': approved_allocations,
            'pending_approvals': pending_approvals,
            'variances': variances,
            'summary': {
                'total_articles': len(store_allocations),
                'approved_articles': len(approved_allocations),
                'pending_articles': len(pending_approvals),
                'total_variances': len(variances),
                'variances_requiring_approval': len([v for v in variances if v['status'] == ApprovalStatus.NEEDS_APPROVAL.value])
            },
            'generated_at': datetime.now().isoformat()
        }
        
        return final_output
    
    def get_audit_trail(self, article: Optional[str] = None,
                       store_id: Optional[str] = None,
                       action_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get audit trail with optional filters
        
        Args:
            article: Filter by article
            store_id: Filter by store
            action_type: Filter by action type
        
        Returns:
            Filtered audit trail
        """
        filtered = self.audit_trail
        
        if article:
            filtered = [e for e in filtered if e.get('details', {}).get('article') == article]
        
        if store_id:
            filtered = [e for e in filtered if e.get('details', {}).get('store_id') == store_id]
        
        if action_type:
            filtered = [e for e in filtered if e.get('action') == action_type]
        
        return filtered
    
    def get_system_vs_human_comparison(self, article: str) -> Dict[str, Any]:
        """
        Get comparison of system suggested vs human finalized quantities
        
        Args:
            article: Article identifier
        
        Returns:
            Comparison data
        """
        comparison = {
            'article': article,
            'aggregate_level': {},
            'store_level': {}
        }
        
        # Aggregate level comparison
        if article in self.aggregate_edits:
            edit = self.aggregate_edits[article]
            comparison['aggregate_level'] = {
                'system_suggested': edit['original_quantity'],
                'human_finalized': edit['edited_quantity'],
                'variance': edit['variance'],
                'variance_pct': edit['variance_pct'],
                'status': edit['approval_status']
            }
        
        # Store level comparison
        if article in self.store_level_edits:
            for store_id, edit in self.store_level_edits[article].items():
                comparison['store_level'][store_id] = {
                    'system_suggested': edit['original_quantity'],
                    'human_finalized': edit['edited_quantity'],
                    'variance': edit['variance'],
                    'variance_pct': edit['variance_pct'],
                    'status': edit['approval_status']
                }
        
        return comparison
    
    def _add_audit_entry(self, action: str, details: Dict[str, Any], 
                        user_id: str = "system"):
        """
        Add entry to audit trail
        
        Args:
            action: Action performed
            details: Action details
            user_id: User who performed the action
        """
        audit_entry = {
            'timestamp': datetime.now().isoformat(),
            'action': action,
            'user_id': user_id,
            'details': details
        }
        
        self.audit_trail.append(audit_entry)
    
    def export_metadata(self) -> Dict[str, Any]:
        """
        Export all metadata for persistence
        
        Returns:
            Complete metadata dictionary
        """
        return {
            'store_mappings': self.store_mappings,
            'aggregate_edits': self.aggregate_edits,
            'store_level_edits': self.store_level_edits,
            'audit_trail': self.audit_trail,
            'variance_threshold': self.variance_threshold,
            'exported_at': datetime.now().isoformat()
        }
    
    def import_metadata(self, metadata: Dict[str, Any]):
        """
        Import metadata from previous session
        
        Args:
            metadata: Metadata dictionary
        """
        self.store_mappings = metadata.get('store_mappings', {})
        self.aggregate_edits = metadata.get('aggregate_edits', {})
        self.store_level_edits = metadata.get('store_level_edits', {})
        self.audit_trail = metadata.get('audit_trail', [])
        self.variance_threshold = metadata.get('variance_threshold', 0.05)

