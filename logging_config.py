import logging
import sys
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(name)s | %(levelname)s | %(message)s',
    handlers=[
        logging.FileHandler('logs/intelli_odm.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

# Agent execution tracker
class AgentTracker:
    def __init__(self):
        self.executions = []
    
    def log_agent_start(self, agent_name, action):
        entry = {
            'timestamp': datetime.now().isoformat(),
            'agent': agent_name,
            'action': action,
            'status': 'started'
        }
        self.executions.append(entry)
        logging.info(f"🤖 {agent_name}: {action} - Started")
    
    def log_agent_complete(self, agent_name, action, result=None):
        for entry in reversed(self.executions):
            if entry['agent'] == agent_name and entry['action'] == action and entry['status'] == 'started':
                entry['status'] = 'completed'
                entry['result'] = result
                break
        logging.info(f"✅ {agent_name}: {action} - Completed")
    
    def log_agent_error(self, agent_name, action, error):
        for entry in reversed(self.executions):
            if entry['agent'] == agent_name and entry['action'] == action and entry['status'] == 'started':
                entry['status'] = 'error'
                entry['error'] = str(error)
                break
        logging.error(f"❌ {agent_name}: {action} - Error: {error}")
    
    def get_execution_log(self):
        return self.executions

# Global tracker instance
tracker = AgentTracker()
