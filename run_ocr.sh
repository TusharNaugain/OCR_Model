#!/bin/bash

# Gemini OCR System - Quick Start Script
# This script sets up your environment and runs the OCR system

# Set Gemini API Key
export GEMINI_API_KEY="AIzaSyAFCE6jHDGolkSiaGsTCwg3GONDiuzFwaU"
export GEMINI_MODEL="gemini-2.0-flash-exp"

# Optional: Set OCR engine (default is gemini-enhanced when API key is set)
# export OCR_ENGINE="gemini-enhanced"

# Run the OCR system
echo "🚀 Starting Gemini-Enhanced OCR System..."
echo "📋 FREE Tier Active: 1,500 requests/day"
echo ""

python3 main.py "$@"
