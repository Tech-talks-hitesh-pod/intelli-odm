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

### Or Use Make
```bash
make install
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

**Next Steps:**
- Read `Readme.md` for full documentation
- Check `INSTALL.md` for troubleshooting
- Copy `config.example` to `.env` for configuration
- Run `make help` to see all available commands

---

## 🚀 Common Commands

```bash
make test        # Run tests
make format      # Format code
make lint        # Check code quality
make jupyter     # Start Jupyter notebook
python orchestrator.py  # Run the system
```

---

**Need Help?** See `INSTALL.md` or `SETUP_SUMMARY.md` for detailed instructions.

