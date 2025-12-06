"""
Audit Logger for Multi-Agent System
Captures inputs, outputs, and status of each agent operation
"""

import json
import os
from datetime import datetime
from typing import Dict, List, Any, Optional
from pathlib import Path
from enum import Enum

class LogStatus(str, Enum):
    """Status enumeration for audit logs"""
    SUCCESS = "Success"
    FAIL = "Fail"
    IN_PROGRESS = "In Progress"
    WARNING = "Warning"

class AuditLogger:
    """Centralized audit logging system for multi-agent operations"""
    
    def __init__(self, log_dir: str = "audit-logs"):
        """
        Initialize audit logger
        
        Args:
            log_dir: Directory to store audit logs
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        self.current_run_id = None
    
    def start_run(self, run_id: Optional[str] = None) -> str:
        """
        Start a new forecast run session
        
        Args:
            run_id: Optional run ID, will generate if not provided
            
        Returns:
            Run ID for this session
        """
        if run_id is None:
            run_id = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        self.current_run_id = run_id
        return run_id
    
    def log_agent_operation(
        self,
        agent_name: str,
        description: str,
        status: LogStatus,
        inputs: Optional[Dict[str, Any]] = None,
        outputs: Optional[Dict[str, Any]] = None,
        error: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Log an agent operation
        
        Args:
            agent_name: Name of the agent (e.g., "DemandForecastingAgent")
            description: Description of the operation
            status: Status of the operation (Success/Fail/In Progress/Warning)
            inputs: Input data for the operation
            outputs: Output data from the operation
            error: Error message if status is Fail
            metadata: Additional metadata
            
        Returns:
            Log entry dictionary
        """
        log_entry = {
            "agent_name": agent_name,
            "date_time": datetime.now().isoformat(),
            "description": description,
            "status": status.value,
            "run_id": self.current_run_id,
            "inputs": self._sanitize_for_json(inputs) if inputs else None,
            "outputs": self._sanitize_for_json(outputs) if outputs else None,
            "error": error,
            "metadata": metadata or {}
        }
        
        # Save to file
        self._save_log_entry(log_entry)
        
        return log_entry
    
    def _sanitize_for_json(self, data: Any) -> Any:
        """Convert data to JSON-serializable format"""
        if isinstance(data, dict):
            return {k: self._sanitize_for_json(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [self._sanitize_for_json(item) for item in data]
        elif isinstance(data, (int, float, str, bool, type(None))):
            return data
        elif hasattr(data, '__dict__'):
            return str(data)
        else:
            return str(data)
    
    def _save_log_entry(self, log_entry: Dict[str, Any]):
        """Save log entry to JSON file"""
        if not self.current_run_id:
            self.start_run()
        
        # Create run-specific log file
        log_file = self.log_dir / f"{self.current_run_id}.json"
        
        # Load existing logs or create new list
        if log_file.exists():
            try:
                with open(log_file, 'r') as f:
                    logs = json.load(f)
            except:
                logs = []
        else:
            logs = []
        
        # Append new log entry
        logs.append(log_entry)
        
        # Save back to file
        with open(log_file, 'w') as f:
            json.dump(logs, f, indent=2, default=str)
    
    def get_run_logs(self, run_id: str) -> List[Dict[str, Any]]:
        """
        Get all logs for a specific run
        
        Args:
            run_id: Run ID to retrieve logs for
            
        Returns:
            List of log entries
        """
        log_file = self.log_dir / f"{run_id}.json"
        
        if not log_file.exists():
            return []
        
        try:
            with open(log_file, 'r') as f:
                return json.load(f)
        except:
            return []
    
    def get_all_runs(self) -> List[str]:
        """
        Get list of all run IDs
        
        Returns:
            List of run IDs
        """
        runs = []
        for log_file in self.log_dir.glob("*.json"):
            run_id = log_file.stem
            runs.append(run_id)
        
        return sorted(runs, reverse=True)
    
    def get_agent_logs(self, agent_name: str, run_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get logs for a specific agent
        
        Args:
            agent_name: Name of the agent
            run_id: Optional run ID to filter by
            
        Returns:
            List of log entries for the agent
        """
        if run_id:
            logs = self.get_run_logs(run_id)
        else:
            # Get from all runs
            logs = []
            for run_id in self.get_all_runs():
                run_logs = self.get_run_logs(run_id)
                logs.extend(run_logs)
        
        # Filter by agent name
        return [log for log in logs if log.get('agent_name') == agent_name]

