FROM python:3.10-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    poppler-utils \
    libgl1 \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create directories for uploads and results
RUN mkdir -p uploads results

# Expose the port
EXPOSE 8000

# Create a robust startup script
RUN echo '#!/bin/bash\n\
    echo "=== SEARCHING FOR APP === "\n\
    find / -name "api.py" 2>/dev/null\n\
    APP_DIR=$(dirname $(find /app -name "api.py" -print -quit))\n\
    echo "Found app in: $APP_DIR"\n\
    cd $APP_DIR\n\
    echo "=== CONTENTS === "\n\
    ls -la\n\
    echo "=== STARTING === "\n\
    exec uvicorn api:app --host 0.0.0.0 --port 8000' > /run.sh && chmod +x /run.sh

# Run the application
CMD ["/run.sh"]
