# Setup Guide - Demand Forecasting & Allocation Engine

## Prerequisites

1. **Python 3.9+** installed
2. **Node.js 18+** and npm installed
3. **Ollama** installed and running with `llama3:8b` model

### Install Ollama

```bash
# macOS
brew install ollama

# Linux
curl -fsSL https://ollama.ai/install.sh | sh

# Windows - Download from https://ollama.ai
```

### Pull llama3:8b Model

```bash
ollama pull llama3:8b
```

### Start Ollama

```bash
ollama serve
```

Verify it's running:
```bash
curl http://localhost:11434/api/tags
```

## Installation Steps

### 1. Install Python Dependencies

```bash
pip install -r requirements.txt
```

### 2. Install Frontend Dependencies

```bash
cd frontend
npm install
cd ..
```

### 3. Create Required Directories

```bash
mkdir -p uploads audit-logs
```

## Running the Application

### Option 1: Using the Run Script (Recommended)

```bash
./run.sh
```

This will start both backend and frontend automatically.

### Option 2: Manual Start

#### Start Backend

```bash
cd backend
python -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

#### Start Frontend (in a new terminal)

```bash
cd frontend
npm run dev
```

## Access the Application

- **Frontend**: http://localhost:3000
- **Backend API**: http://localhost:8000
- **API Docs**: http://localhost:8000/docs

## Project Structure

```
intelli-odm/
├── backend/
│   └── main.py              # FastAPI backend
├── frontend/
│   ├── src/
│   │   ├── App.tsx          # React main component
│   │   └── main.tsx         # React entry point
│   └── package.json
├── agents/
│   ├── demand_forecasting_agent.py    # Main forecasting agent (uses Ollama)
│   ├── attribute_analogy_agent.py     # Attribute analysis agent (uses Ollama)
│   └── data_ingestion_agent.py
├── utils/
│   ├── ollama_client.py     # Ollama client wrapper
│   └── audit_logger.py      # Audit logging system
├── uploads/                 # Uploaded data files
├── audit-logs/              # JSON audit log files
└── requirements.txt
```

## Key Features

### 1. Multi-Agent System
- **Data Ingestion Agent**: Validates and structures input data
- **Attribute Analogy Agent**: Uses Ollama (llama3:8b) to standardize attributes and find comparables
- **Demand Forecasting Agent**: Uses Ollama (llama3:8b) for model selection and forecasting

### 2. Audit Logging
All agent operations are logged with:
- Agent Name
- Date & Time
- Description
- Status (Success/Fail/In Progress/Warning)
- Inputs and Outputs (JSON format)

Logs are stored in `audit-logs/` directory as JSON files, one per run.

### 3. Ollama Integration
- Uses `llama3:8b` model running locally
- Prompt-based interactions for:
  - Model selection
  - Attribute standardization
  - Comparable product finding
  - Trend analysis

## API Endpoints

### Upload Files
```
POST /api/upload
Content-Type: multipart/form-data
```

### Run Forecast
```
POST /api/forecast
Content-Type: multipart/form-data
```

### Get Audit Logs
```
GET /api/audit-logs/{run_id}
GET /api/audit-logs
GET /api/audit-logs/agent/{agent_name}
```

## Troubleshooting

### Ollama Connection Issues
- Ensure Ollama is running: `ollama serve`
- Check if model is available: `ollama list`
- Verify connection: `curl http://localhost:11434/api/tags`

### Port Conflicts
- Backend default port: 8000 (change in `backend/main.py`)
- Frontend default port: 3000 (change in `frontend/vite.config.ts`)

### Import Errors
- Ensure you're in the project root directory
- Check that all dependencies are installed
- Verify Python path includes project root

## Development

### Backend Development
```bash
cd backend
python -m uvicorn main:app --reload
```

### Frontend Development
```bash
cd frontend
npm run dev
```

## Production Deployment

For production:
1. Build frontend: `cd frontend && npm run build`
2. Serve static files from FastAPI
3. Use production WSGI server (Gunicorn)
4. Set up proper authentication
5. Configure environment variables

## Notes

- All audit logs are stored in JSON format in `audit-logs/` directory
- Each forecast run gets a unique run_id
- Logs can be viewed in the UI under "Audit Logs" tab
- Ollama must be running before starting the application

