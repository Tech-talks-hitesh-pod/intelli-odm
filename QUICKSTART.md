# Quick Start - Intelli-ODM

**Get up and running in 5 minutes!**

---

## 1️⃣ Clone & Navigate
```bash
git clone https://github.com/your-org/intelli-odm.git
cd intelli-odm
```

---

## 2️⃣ Run Setup Script

### Windows (PowerShell)
```powershell
.\setup.ps1
```

### Windows (CMD)
```cmd
setup.bat
```

### macOS / Linux
```bash
chmod +x setup.sh
./setup.sh
```

---

## 3️⃣ Install Ollama

**Download & Install:**
- Windows: https://ollama.com/download/windows
- macOS: `brew install ollama` or https://ollama.com/download/mac
- Linux: `curl -fsSL https://ollama.com/install.sh | sh`

**Pull Model:**
```bash
ollama pull llama3:8b
```

---

## 4️⃣ Activate Environment

### Windows
```cmd
.venv\Scripts\activate
```

### macOS / Linux
```bash
source .venv/bin/activate
```

---

## 5️⃣ Verify Setup

```bash
python -c "import pandas, numpy, sklearn, chromadb; print('✅ Ready!')"
ollama list
```

---

## 🎯 You're Ready!

The core system is installed and ready to use!

**Optional: Time-Series Forecasting**
```bash
# Only needed if you want Prophet-based time-series forecasting
# The system also supports analogy-based and regression forecasting

# macOS/Linux:
pip install prophet

# Windows (recommended):
conda install -c conda-forge prophet
```

**Next Steps:**
- Read `Readme.md` for full documentation
- Check `INSTALL.md` for troubleshooting
- Copy `config.example` to `.env` for configuration

---

## 🚀 Common Commands

```bash
# Run tests
pytest

# Format code
black . && isort .

# Check code quality
flake8

# Start Jupyter notebook
jupyter notebook

# Run the system
python orchestrator.py
```

---

**Need Help?** See `INSTALL.md` or `SETUP_SUMMARY.md` for detailed instructions.

