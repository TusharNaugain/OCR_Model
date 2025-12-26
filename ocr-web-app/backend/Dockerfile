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

# Create a startup script to debug runtime state
RUN echo '#!/bin/bash\n\
echo "=== DEBUG INFO ==="\n\
pwd\n\
ls -la\n\
echo "=== END DEBUG ===\n"\n\
exec uvicorn api:app --host 0.0.0.0 --port 8000' > /start.sh && chmod +x /start.sh

# Run the application using the script
CMD ["/start.sh"]
