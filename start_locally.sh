#!/bin/bash

# Function to kill background processes on exit
cleanup() {
    echo -e "\nStopping servers..."
    # Kill all child processes of this script
    pkill -P $$
    exit
}

# Trap SIGINT (Ctrl+C) and call cleanup
trap cleanup SIGINT SIGTERM

echo "==========================================="
echo "   Starting OCR Web App Locally"
echo "==========================================="

# 1. Start Backend
echo "[1/2] Starting Backend (Port 8000)..."
cd ocr-web-app/backend || { echo "Backend directory not found"; exit 1; }

# Check for uvicorn
if ! command -v uvicorn &> /dev/null; then
    echo "⚠️  'uvicorn' not found. Attempting to install requirements..."
    pip install -r requirements.txt
    if [ $? -ne 0 ]; then
        echo "❌ Failed to install backend dependencies. Please check python setup."
        exit 1
    fi
fi

# Run Backend in background
uvicorn api:app --host 0.0.0.0 --port 8000 --reload &
BACKEND_PID=$!

# Wait a moment for backend to initialize
sleep 2

# 2. Start Frontend
echo "[2/2] Starting Frontend (Port 3000)..."
cd ../frontend || { echo "Frontend directory not found"; kill $BACKEND_PID; exit 1; }

# Install modules if missing
if [ ! -d "node_modules" ]; then
    echo "📦 Node modules not found. Installing..."
    npm install
fi

# Run Frontend in background
npm run dev &
FRONTEND_PID=$!

cd ../..

echo "==========================================="
echo "✅ App is running!"
echo "   Frontend: http://localhost:3000"
echo "   Backend:  http://localhost:8000"
echo "==========================================="
echo "Press Ctrl+C to stop both servers."

# Wait for both processes
wait
