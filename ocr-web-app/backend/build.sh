#!/usr/bin/env bash
# Install system dependencies for OCR

# Update package list
apt-get update

# Install Tesseract OCR
apt-get install -y tesseract-ocr

# Install Poppler (for PDF processing)
apt-get install -y poppler-utils

echo "System dependencies installed successfully"
