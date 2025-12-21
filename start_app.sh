#!/bin/bash
export PATH="/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin:$PATH"

# Kill any existing processes on ports
echo "🧹 Cleaning up ports..."
if command -v lsof >/dev/null 2>&1; then
    lsof -ti:8000 | xargs kill -9 2>/dev/null
    lsof -ti:3000 | xargs kill -9 2>/dev/null
    lsof -ti:3001 | xargs kill -9 2>/dev/null
else
    echo "⚠️  'lsof' not found. Skipping port cleanup. If ports are in use, restart might fail."
fi

echo "🚀 Starting SmartScan OCR System..."

# Start Backend in background
echo "📦 Starting Backend Server..."
cd ocr-web-app/backend
python3 api.py > ../../backend.log 2>&1 &
BACKEND_PID=$!
echo "✅ Backend started (PID: $BACKEND_PID)"

# Wait for backend to initialize
sleep 3

# Start Frontend
echo "💻 Starting Frontend..."
cd ../frontend
echo "   - URL: http://localhost:3000"
npm run dev

# Cleanup on exit
trap "kill $BACKEND_PID" EXIT
