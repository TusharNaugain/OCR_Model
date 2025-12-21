
import os
import re
import json
import logging
import google.generativeai as genai
import pytesseract

# Configure Tesseract Path
_possible_tesseract_paths = [
    '/opt/homebrew/bin/tesseract',  # Apple Silicon Homebrew
    '/usr/local/bin/tesseract',     # Intel Homebrew
    '/usr/bin/tesseract',           # System
]
_tesseract_cmd = next((p for p in _possible_tesseract_paths if os.path.exists(p)), 'tesseract')
pytesseract.pytesseract.tesseract_cmd = _tesseract_cmd

# ========== OCR ENGINE CONFIGURATION ==========
# Set OCR engine: 'tesseract' or 'gemini-enhanced'
OCR_ENGINE = os.getenv('OCR_ENGINE', 'tesseract')  # Default: tesseract

# ========== GEMINI API CONFIGURATION ==========
# Configure Gemini API for intelligent OCR enhancement
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
GEMINI_MODEL = os.getenv('GEMINI_MODEL', 'gemini-2.0-flash-exp')  # Default: latest experimental

# Initialize Gemini client (only if API key is set)
_gemini_model = None

def get_gemini_model():
    """Get or create Gemini model instance."""
    global _gemini_model
    if _gemini_model:
        return _gemini_model
    
    if not GEMINI_API_KEY:
        print("⚠️  GEMINI_API_KEY not found in environment variables.")
        return None
        
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        _gemini_model = genai.GenerativeModel(GEMINI_MODEL)
        return _gemini_model
    except Exception as e:
        print(f"❌ Failed to initialize Gemini model: {e}")
        return None

def fix_character_confusion(text):
    """
    Fix specific character confusions common in dot-matrix/impact printing.
    Optimized for Certificate numbers (IN-GJ...) and Amounts.
    
    Common confusions addressed:
    - 5 ↔ S, 8 (in amounts)
    - 3 ↔ 8, 9, 5
    - 0 ↔ O in numbers
    - 1 ↔ I, l in numbers
    """
    if not text:
        return text
    
    # Fix state code confusions in certificate IDs
    text = re.sub(r'\bIN-GU(\d)',  r'IN-GJ\1', text)  # GU → GJ
    text = re.sub(r'\bIN-CJ(\d)', r'IN-GJ\1', text)   # CJ → GJ
    text = re.sub(r'\bIN-G\.J(\d)', r'IN-GJ\1', text) # G.J → GJ
    text = re.sub(r'\bIN-GI(\d)', r'IN-GJ\1', text)   # GI → GJ
    
    # Fix O/0 confusion in certificate numbers (only after state code)
    text = re.sub(r'(IN-[A-Z]{2})O(\d)', r'\g<1>0\2', text)
    
    # Fix I/1/l confusion in numbers
    text = re.sub(r'(IN-[A-Z]{2}\d*)I(\d)', r'\g<1>1\2', text)
    text = re.sub(r'(IN-[A-Z]{2}\d*)l(\d)', r'\g<1>1\2', text)
    
    # Fix specific digit confusions in certificate numbers (GJ certificates)
    # Pattern: IN-GJ followed by 14 digits
    cert_match = re.search(r'(IN-GJ)(\d{14}[A-Z])', text)
    if cert_match:
        prefix = cert_match.group(1)
        number_part = cert_match.group(2)
        
        # Common misreads: 3→8, 3→9, 3→5
        # We'll apply heuristics based on common patterns
        # If we see 559 or 558 where 553 is expected, fix it
        if '559' in number_part[:5]:
            number_part = number_part.replace('559', '553', 1)
            # print(f"      🔧 Fixed digit confusion: 559 → 553")
        elif '558' in number_part[:5]:
            number_part = number_part.replace('558', '553', 1)
            # print(f"      🔧 Fixed digit confusion: 558 → 553")
        
        text = text.replace(cert_match.group(0), prefix + number_part)
    
    # Apply general cleanup for dirty scans
    text = clean_ocr_noise(text)
    
    return text

def clean_ocr_noise(text):
    """
    Clean up specific OCR noise and common misreadings.
    Targeted for dirty document scans.
    """
    if not text:
        return text

    # 1. Fix common word corruptions
    replacements = {
        r'\bStock Hotding\b': 'Stock Holding',
        r'\bAcetate Meta AiR\b': 'Certificate', # Heuristic guess based on context, or just remove
        r'\bAccount Reterence\b': 'Account Reference',
        r'\bUniaue Doe\b': 'Unique Doc',
        r'\bCompetent\b': 'Competent Authority',
    }
    
    for pattern, replacement in replacements.items():
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)

    # 2. Fix Date Confusions (e.g. 81-Dec -> 31-Dec)
    # Pattern: Digit-Digit-Month-Year where first digit might be 8/9/S instead of 3/0/1
    def fix_date_match(match):
        day_part = match.group(1)
        month_part = match.group(2)
        year_part = match.group(3)
        
        # Logically fix day
        if day_part.startswith('8'): day_part = '3' + day_part[1:]
        if day_part == '00': day_part = '01'
        
        return f"{day_part}-{month_part}-{year_part}"

    text = re.sub(r'\b(\d{2})-(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)-(\d{4})\b', fix_date_match, text, flags=re.IGNORECASE)

    # 3. Remove "Salt and Pepper" Noise
    # Remove lines that are just random short garbage
    lines = text.split('\n')
    cleaned_lines = []
    for line in lines:
        # Remove specific artifacts
        line = line.replace('=>.', '').strip()
        
        # Skip lines that are just symbols or very short random text
        if not line:
             continue
        if re.match(r'^[\W\s]+$', line): # Only symbols
            continue
        if len(line) < 3 and not line.isalnum():
            continue
            
        cleaned_lines.append(line)
        
    return '\n'.join(cleaned_lines)

def enhance_ocr_with_gemini(raw_text, lines, document_type="stamp_duty_certificate"):
    """
    Use Gemini Flash 2.0 to correct OCR errors and extract context.
    Costs: Free (within limits) or very low.
    """
    if not GEMINI_API_KEY:
        return {'gemini_used': False, 'reason': 'No API Key'}

    model = get_gemini_model()
    if not model:
        return {'gemini_used': False, 'reason': 'Model Init Failed'}

    prompt = f"""
    You are an expert OCR post-processing system. 
    The following text is extracted from a {document_type} using Tesseract.
    It may contain typos, misread characters, and layout issues.
    
    YOUR TASK:
    1. Identify the specific document type more accurately if possible.
    2. Correct the text based on context and fix common OCR errors.
    3. EXTRACT STRUCTURED DATA:
       - Identify and extract tables as list of records.
       - Extract key-value pairs (e.g., Enrollment No, Batch, Names, Dates).
       - Maintain the nested structure of the document.
    4. Return the corrected text as a JSON object.
    
    RAW OCR TEXT:
    {raw_text}
    
    Return JSON ONLY:
    {{
        "corrected_lines": ["line 1", "line 2", ...],
        "extracted_structured_data": {{
            "tables": [],
            "fields": {{}}
        }},
        "confidence_score": 0.95
    }}
    """
    
    try:
        response = model.generate_content(prompt)
        text_response = response.text.strip()
        
        # Extract JSON from response (handle markdown code blocks)
        if "```json" in text_response:
            json_str = text_response.split("```json")[1].split("```")[0]
        elif "```" in text_response:
            json_str = text_response.split("```")[1].split("```")[0]
        else:
            json_str = text_response
            
        result = json.loads(json_str)
        return {
             'gemini_used': True,
             'corrected_lines': result.get('corrected_lines', lines),
             'extracted_structured_data': result.get('extracted_structured_data', {}),
             'confidence': result.get('confidence_score', 0.8)
        }
    except Exception as e:
        print(f"   ⚠️  Gemini enhancement error: {e}")
        return {'gemini_used': False, 'reason': str(e)}
