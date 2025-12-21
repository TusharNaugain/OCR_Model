
print("Starting imports...")
try:
    print("Importing sys, os...")
    import sys, os
    print("Importing pathlib...")
    from pathlib import Path
    
    print("Importing fitz (PyMuPDF)...")
    import fitz
    print("✓ fitz imported")
    
    print("Importing PIL...")
    from PIL import Image
    print("✓ PIL imported")
    
    print("Importing pytesseract...")
    import pytesseract
    print("✓ pytesseract imported")
    
    print("Importing cv2...")
    import cv2
    print("✓ cv2 imported")
    
    print("Importing numpy...")
    import numpy as np
    print("✓ numpy imported")
    
    print("Importing google.generativeai...")
    import google.generativeai as genai
    print("✓ google.generativeai imported")
    
    print("ALL IMPORTS SUCCESSFUL")
    
except Exception as e:
    print(f"\n❌ IMPORT FAILED: {e}")
