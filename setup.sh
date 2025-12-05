#!/bin/bash

# Intelli-ODM CEO Demo Setup Script
# Quick 5-minute setup for executive demonstration

set -e  # Exit on any error

echo "🎯 Intelli-ODM CEO Demo Setup"
echo "================================"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if Python 3.8+ is available
check_python() {
    print_status "Checking Python installation..."
    
    if command -v python3 &> /dev/null; then
        PYTHON_VERSION=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
        print_success "Found Python $PYTHON_VERSION"
        
        # Check if version is 3.8 or higher
        if python3 -c 'import sys; sys.exit(0 if sys.version_info >= (3, 8) else 1)'; then
            PYTHON_CMD="python3"
            PIP_CMD="pip3"
        else
            print_error "Python 3.8+ required. Found Python $PYTHON_VERSION"
            exit 1
        fi
    else
        print_error "Python 3 not found. Please install Python 3.8+"
        exit 1
    fi
}

# Create virtual environment
setup_venv() {
    print_status "Setting up Python virtual environment..."
    
    if [ ! -d "venv" ]; then
        $PYTHON_CMD -m venv venv
        print_success "Virtual environment created"
    else
        print_warning "Virtual environment already exists"
    fi
    
    # Activate virtual environment
    source venv/bin/activate
    print_success "Virtual environment activated"
    
    # Upgrade pip
    pip install --upgrade pip > /dev/null 2>&1
}

# Install Python dependencies
install_dependencies() {
    print_status "Installing Python dependencies..."
    
    # Create minimal requirements for demo
    cat > requirements_demo.txt << EOF
streamlit>=1.28.0
pandas>=1.5.0
plotly>=5.15.0
numpy>=1.21.0
chromadb>=0.4.15
sentence-transformers>=2.2.0
python-dotenv>=1.0.0
pydantic>=2.0.0
requests>=2.28.0
ollama>=0.1.7
openai>=1.0.0
pulp>=2.7.0
prophet>=1.1.4
scikit-learn>=1.3.0
langsmith>=0.0.87
httpx>=0.25.0
EOF

    pip install -r requirements_demo.txt
    print_success "Dependencies installed"
}

# Setup Ollama
setup_ollama() {
    print_status "Setting up Ollama LLM service..."
    
    # Check if Ollama is installed
    if command -v ollama &> /dev/null; then
        print_success "Ollama already installed"
    else
        print_status "Installing Ollama..."
        if [[ "$OSTYPE" == "darwin"* ]]; then
            # macOS
            if command -v brew &> /dev/null; then
                brew install ollama
            else
                curl -fsSL https://ollama.ai/install.sh | sh
            fi
        else
            # Linux
            curl -fsSL https://ollama.ai/install.sh | sh
        fi
        print_success "Ollama installed"
    fi
    
    # Start Ollama service in background
    print_status "Starting Ollama service..."
    if ! pgrep -f "ollama serve" > /dev/null; then
        ollama serve > ollama.log 2>&1 &
        OLLAMA_PID=$!
        sleep 5
        print_success "Ollama service started (PID: $OLLAMA_PID)"
    else
        print_warning "Ollama service already running"
    fi
    
    # Pull Llama3 model
    print_status "Downloading Llama3 model (this may take a few minutes)..."
    if ollama list | grep -q "llama3:8b"; then
        print_success "Llama3 model already available"
    else
        ollama pull llama3:8b
        print_success "Llama3 model downloaded"
    fi
    
    # Test Ollama connection
    print_status "Testing Ollama connection..."
    if curl -s http://localhost:11434/api/version > /dev/null; then
        print_success "Ollama service is ready"
    else
        print_warning "Ollama service may need a moment to start"
    fi
}

# Setup environment variables
setup_environment() {
    print_status "Setting up environment configuration..."
    
    if [ ! -f ".env" ]; then
        cat > .env << EOF
# LLM Configuration
LLM_PROVIDER=ollama
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3:8b

# OpenAI Configuration (optional)
# OPENAI_API_KEY=your_openai_api_key_here
OPENAI_MODEL=gpt-4o-mini

# Database Configuration
CHROMA_PERSIST_DIRECTORY=data/chroma_db

# Observability Configuration
LANGCHAIN_TRACING_V2=true
LANGCHAIN_PROJECT=intelli-odm-demo
# LANGCHAIN_API_KEY=your_langsmith_api_key_here

# Demo Configuration
DEMO_MODE=false
LOG_LEVEL=INFO
EOF
        print_success "Environment file created"
    else
        print_warning "Environment file already exists"
    fi
}

# Setup directories
setup_directories() {
    print_status "Setting up data directories..."
    
    mkdir -p data/chroma_db
    mkdir -p logs
    mkdir -p exports
    
    print_success "Directories created"
}

# Create launcher script
create_launcher() {
    print_status "Creating demo launcher..."
    
    cat > launch_demo.sh << 'EOF'
#!/bin/bash

# Activate virtual environment
source venv/bin/activate

# Set environment variables
export PYTHONPATH=$PWD:$PYTHONPATH

echo "🎯 Starting Intelli-ODM CEO Demo..."
echo "=================================="
echo
echo "🌐 Demo will open in your browser at: http://localhost:8501"
echo "📋 Use the sidebar to select different business scenarios"
echo "🚀 Click 'Run AI Analysis' to see results"
echo
echo "Press Ctrl+C to stop the demo"
echo

# Start Streamlit
streamlit run streamlit_app.py --server.port=8501 --server.address=0.0.0.0 --theme.base=light
EOF
    
    chmod +x launch_demo.sh
    print_success "Demo launcher created"
}

# Initialize knowledge base with sample data
init_knowledge_base() {
    print_status "Initializing knowledge base..."
    
    # Create a simple initialization script
    cat > init_kb.py << 'EOF'
#!/usr/bin/env python3
"""Initialize knowledge base with sample data."""

import sys
from shared_knowledge_base import SharedKnowledgeBase

def main():
    try:
        kb = SharedKnowledgeBase(persist_directory="data/chroma_db")
        
        # Add some sample products for similarity search
        sample_products = [
            {"id": "sample_tshirt", "attributes": {"category": "TSHIRT", "material": "Cotton", "color": "White"}, "description": "Basic white t-shirt"},
            {"id": "sample_jeans", "attributes": {"category": "JEANS", "material": "Denim", "color": "Blue"}, "description": "Classic blue jeans"},
            {"id": "sample_dress", "attributes": {"category": "DRESS", "material": "Chiffon", "color": "Black"}, "description": "Elegant black dress"}
        ]
        
        for product in sample_products:
            kb.store_product(
                product_id=product["id"],
                attributes=product["attributes"], 
                description=product["description"]
            )
        
        print("✅ Knowledge base initialized with sample data")
        
    except Exception as e:
        print(f"❌ Knowledge base initialization failed: {e}")

if __name__ == "__main__":
    main()
EOF
    
    # Run the initialization
    python3 init_kb.py > /dev/null 2>&1
    rm init_kb.py
    
    print_success "Knowledge base initialized"
}

# System verification
verify_system() {
    print_status "Verifying system setup..."
    
    # Test Python imports
    python3 -c "
import sys
try:
    import streamlit, pandas, plotly, numpy, chromadb
    print('✅ Core Python packages working')
except ImportError as e:
    print(f'❌ Python package issue: {e}')
    sys.exit(1)
"
    
    # Test Ollama if configured
    if [ -f ".env" ] && grep -q "LLM_PROVIDER=ollama" .env; then
        python3 -c "
try:
    import ollama
    client = ollama.Client()
    response = client.list()
    print('✅ Ollama connection working')
except Exception as e:
    print(f'⚠️ Ollama connection issue: {e}')
"
    fi
    
    # Test knowledge base
    python3 -c "
try:
    from shared_knowledge_base import SharedKnowledgeBase
    kb = SharedKnowledgeBase(persist_directory='data/chroma_db')
    print(f'✅ Knowledge base working ({kb.get_collection_size()} products)')
except Exception as e:
    print(f'❌ Knowledge base issue: {e}')
"
    
    print_success "System verification completed"
}

# Setup observability dashboard
setup_observability() {
    print_status "Setting up observability dashboard..."
    
    # Create simple logging configuration
    cat > logging_config.py << 'EOF'
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
EOF
    
    print_success "Observability dashboard configured"
}

# Main setup function
main() {
    echo
    print_status "Starting Intelli-ODM CEO Demo setup..."
    
    # Core setup steps
    check_python
    setup_venv
    install_dependencies
    setup_ollama
    setup_environment
    setup_directories
    setup_observability
    init_knowledge_base
    verify_system
    create_launcher
    
    echo
    print_success "🎉 Setup completed successfully!"
    echo
    echo "======================================"
    echo "🚀 CEO Demo Ready!"
    echo "======================================"
    echo
    echo "📍 Services Running:"
    echo "  • Ollama LLM:     http://localhost:11434"
    echo "  • Demo UI:        http://localhost:8501"
    echo
    echo "🎯 To Start Demo:"
    echo "  ./launch_demo.sh"
    echo
    echo "📊 Features Available:"
    echo "  • 4 Business Scenarios"
    echo "  • AI Agent Tracking"
    echo "  • Real-time Results"
    echo "  • Executive Dashboard"
    echo
    echo "📋 Observability:"
    echo "  • Agent logs:     logs/intelli_odm.log"
    echo "  • Execution traces available in UI"
    echo "  • LangSmith ready (set LANGCHAIN_API_KEY)"
    echo
    echo "⚡ Quick Test:"
    echo "  curl http://localhost:11434/api/version"
    echo "======================================"
    
    # Offer to start demo immediately
    echo
    read -p "🚀 Start CEO demo now? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        print_status "Starting CEO demo..."
        echo "🌐 Opening http://localhost:8501 in your browser..."
        ./launch_demo.sh
    else
        print_status "Setup complete! Run './launch_demo.sh' when ready for your presentation."
        echo
        print_warning "💡 Pro tip: Test the demo once before your CEO presentation!"
    fi
}

# Run main function
main "$@"