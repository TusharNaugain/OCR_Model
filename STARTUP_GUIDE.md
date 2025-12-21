# 🚀 Quick Start Guide - Frontend & Backend

## Directory Structure

```
OCR_PROJECT/
├── main.py                    # Command-line OCR script
├── document_classifier.py     # Classification engine
├── parallel_processor.py      # Batch processing
└── ocr-web-app/              # Web application
    ├── backend/              # FastAPI server
    │   └── api.py
    └── frontend/             # Next.js web interface
        ├── package.json
        └── app/
```

---

## 🎯 Method 1: Use the Startup Script (Easiest)

```bash
cd /Users/tushar/OCR/OCR_PROJECT
./start_app.sh
```

This will start **both** backend and frontend automatically!

---

## 🔧 Method 2: Start Backend & Frontend Separately

### Step 1: Start Backend (Terminal 1)

```bash
# Navigate to backend directory
cd /Users/tushar/OCR/OCR_PROJECT/ocr-web-app/backend

# Install Python dependencies (first time only)
pip3 install fastapi uvicorn python-multipart

# Start FastAPI server
python3 api.py
```

**Expected output:**
```
INFO:     Started server process [12345]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8000
```

✅ Backend is now running on `http://localhost:8000`

---

### Step 2: Start Frontend (Terminal 2 - NEW TERMINAL)

```bash
# Navigate to frontend directory
cd /Users/tushar/OCR/OCR_PROJECT/ocr-web-app/frontend

# Install Node dependencies (first time only)
npm install

# Start Next.js development server
npm run dev
```

**Expected output:**
```
> ocr-web-app@0.1.0 dev
> next dev

  ▲ Next.js 14.x.x
  - Local:        http://localhost:3000
  - Ready in 2.3s
```

✅ Frontend is now running on `http://localhost:3000`

---

## 🌐 Access the Application

Open your browser and go to:
```
http://localhost:3000
```

You should see the OCR upload interface!

---

## 🔍 Troubleshooting

### Issue: "pip3: command not found"
```bash
# Use pip instead
pip install fastapi uvicorn python-multipart
```

### Issue: "npm: command not found"
```bash
# Install Node.js first
# Visit: https://nodejs.org/
# Or use Homebrew:
brew install node
```

### Issue: "Port 3000 already in use"
```bash
# Kill the process using port 3000
lsof -ti:3000 | xargs kill -9

# Or use a different port
npm run dev -- -p 3001
```

### Issue: "Port 8000 already in use"
```bash
# Kill the process using port 8000
lsof -ti:8000 | xargs kill -9
```

### Issue: Backend can't import modules
```bash
# Install all Python dependencies
cd /Users/tushar/OCR/OCR_PROJECT
pip3 install -r requirements.txt
```

---

## 📝 Quick Test

Once both are running:

1. **Open browser**: `http://localhost:3000`
2. **Upload a document**: Click upload button
3. **Check backend logs**: You should see processing activity
4. **View results**: Results appear in the frontend

---

## 🛑 Stopping the Services

### Stop Backend
Press `Ctrl+C` in the backend terminal

### Stop Frontend
Press `Ctrl+C` in the frontend terminal

---

## 🔄 Development Workflow

### Making Backend Changes
1. Edit `ocr-web-app/backend/api.py`
2. Stop backend (`Ctrl+C`)
3. Restart: `python3 api.py`

### Making Frontend Changes
Frontend auto-reloads! Just save your changes and refresh the browser.

---

## 📊 API Endpoints (for testing)

Once backend is running, you can test:

```bash
# Check if backend is alive
curl http://localhost:8000/

# Upload a file
curl -X POST -F "files=@document.pdf" http://localhost:8000/api/ocr/upload

# Check job status
curl http://localhost:8000/api/ocr/status/{job_id}

# Get results
curl http://localhost:8000/api/ocr/results/{job_id}
```

---

## 🎯 Summary Commands

### Start Everything (Easy Way)
```bash
./start_app.sh
```

### Start Backend Only
```bash
cd ocr-web-app/backend && python3 api.py
```

### Start Frontend Only
```bash
cd ocr-web-app/frontend && npm run dev
```

### Start Both (Two Terminals)
**Terminal 1:**
```bash
cd ocr-web-app/backend && python3 api.py
```

**Terminal 2:**
```bash
cd ocr-web-app/frontend && npm run dev
```

---

## ✨ You're Ready!

Your OCR system is now running with:
- 🤖 AI-powered document classification
- 📄 7+ document types supported
- ⚡ Parallel batch processing
- 🌐 Beautiful web interface

Happy processing! 🚀
