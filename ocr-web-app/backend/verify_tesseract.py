import sys
import os
from pathlib import Path

# Add project root to sys.path
# Script is in ocr-web-app/backend/verify_tesseract.py
# Root is OCR_PROJECT
root_dir = str(Path(__file__).parent.parent.parent)
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

print(f"Project Root: {root_dir}")

try:
    import pytesseract
    from ocr_utils import _tesseract_cmd # It's private but we can check it
    
    print(f"Configured Tesseract CMD: {pytesseract.pytesseract.tesseract_cmd}")
    print(f"Internal tesseract_cmd var: {_tesseract_cmd}")
    
    import subprocess
    version = subprocess.check_output([pytesseract.pytesseract.tesseract_cmd, "--version"]).decode()
    print(f"Tesseract Version: {version.splitlines()[0]}")
    print("✅ Tesseract is accessible!")
except Exception as e:
    print(f"❌ Verification failed: {e}")
    # Print sys.path for debugging
    print("sys.path:")
    for p in sys.path:
        print(f"  {p}")
