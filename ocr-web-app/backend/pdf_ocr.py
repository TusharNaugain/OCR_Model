import os
import re
# import cv2  <-- Moved inside functions
import fitz  # PyMuPDF
import numpy as np
# import pytesseract <-- Moved inside functions
from PIL import Image, ImageEnhance, ImageOps
from pathlib import Path
# from pytesseract import Output <-- Moved inside functions
from ocr_utils import OCR_ENGINE, GEMINI_API_KEY, fix_character_confusion, enhance_ocr_with_gemini

# Global flags (can be tuned via env vars in the future)
ULTRA_FAST_MODE = os.environ.get('ULTRA_FAST_MODE') == '1'
FAST_MODE = os.environ.get('FAST_MODE') == '1'

def enhance_image_for_digits(img):
    """
    Specific enhancement for digit recognition (used in header scan).
    """
    # Convert PIL to numpy
    import cv2
    img_np = np.array(img.convert('L'))
    
    # Method 1: Adaptive Threshold with denoise
    denoised = cv2.fastNlMeansDenoising(img_np, None, 10, 7, 21)
    adaptive = cv2.adaptiveThreshold(
        denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 11, 2
    )
    
    # Method 2: Morphological operations to preserve digit structure
    kernel = np.ones((2, 2), np.uint8)
    morph = cv2.morphologyEx(adaptive, cv2.MORPH_CLOSE, kernel)
    
    # Method 3: Sharpen for better digit edges
    kernel_sharpen = np.array([[-1, -1, -1],
                                [-1,  9, -1],
                                [-1, -1, -1]])
    sharpened = cv2.filter2D(morph, -1, kernel_sharpen)
    
    return Image.fromarray(sharpened)

def find_certificate_cv(pil_img):
    """
    Use OpenCV to find potential certificate regions based on morphology.
    Returns a list of cropped PIL images of candidate regions.
    """
    # Convert PIL to CV2
    import cv2
    img_cv = np.array(pil_img)
    
    # Handle Grayscale vs RGB
    if len(img_cv.shape) == 2:
        gray = img_cv
    else:
        # Check if RGB or BGR - PIL is RGB, OpenCV expects BGR usually, 
        # but for grayscale conversion it matters less if we just want intensity.
        # However, standard practice:
        gray = cv2.cvtColor(img_cv, cv2.COLOR_RGB2GRAY)
    
    # 2. Adaptive Thresholding
    thresh = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY_INV, 11, 2
    )
    
    # 3. Morphological Dilation
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 3))
    dilated = cv2.dilate(thresh, kernel, iterations=1)
    
    # 4. Find Contours
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    candidates = []
    height, width = gray.shape
    
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        
        # Filter for Certificate-like Geometry
        aspect_ratio = w / float(h)
        if (y < height * 0.6) and (w > 100) and (h > 15) and (aspect_ratio > 4.0):
             # Add padding
             pad_x = 10
             pad_y = 5
             x1 = max(0, x - pad_x)
             y1 = max(0, y - pad_y)
             x2 = min(width, x + w + pad_x)
             y2 = min(height, y + h + pad_y)
             
             crop = pil_img.crop((x1, y1, x2, y2))
             candidates.append(crop)
             
    return candidates

def process_single_page_ocr_robust(pdf_path, page_num, output_dir=None, doc_handle=None, prior_doc_type=None):
    """
    Process a single page with ROBUST OCR logic (mirrors main.py).
    Includes:
    - Phase 1: High-Res Header Scan
    - Phase 2: Full Page Scan with Advanced Preprocessing
    - Phase 3: Fallback Layers (OpenCV, Nuclear Threshold)
    
    Args:
        prior_doc_type: If provided, skips classification and assumes this type.
    
    Returns standard result dictionary.
    """
    import cv2
    import pytesseract
    from pytesseract import Output
    from document_processors.base_processor import DocumentType

    print(f"\n🔍 [RobustOCR] Processing Page {page_num}...")
    
    # Open PDF and get page
    should_close = False
    if doc_handle:
        doc = doc_handle
    else:
        doc = fitz.open(pdf_path)
        should_close = True
    
    if page_num > len(doc):
        # ... (error handling)
        if should_close: doc.close()
        return None
    
    page = doc[page_num - 1]  # 0-indexed
    
    header_lines = []
    ocr_lines = []
    skip_full_page = False

    # ... (Phase 1 remains largely same, just standardizing imports if needed) ...

    # === PHASE 1: CERTIFICATE HEADER SCAN ===
    if ULTRA_FAST_MODE:
        # ... (fast mode logic)
        pix = page.get_pixmap(dpi=120)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        # ...
        custom_config = '--psm 6 --oem 3 -c tessedit_char_whitelist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-:. "'
        text = pytesseract.image_to_string(img, config=custom_config)
        text = fix_character_confusion(text)
        header_lines = [l.strip() for l in text.split('\n') if l.strip()]
        skip_full_page = True 
        ocr_lines = header_lines
    
    else:
        # Normal/Robust Mode: Super-Resolution Header Scan
        print(f"   📄 [RobustOCR] Scanning certificate header (Super-Res Mode)...")
        page_rect = page.rect
        header_clip = fitz.Rect(0, 0, page_rect.width, page_rect.height * 0.33)
        header_pix = page.get_pixmap(dpi=150, clip=header_clip)
        header_img = Image.frombytes("RGB", [header_pix.width, header_pix.height], header_pix.samples)
        
        # SUPER-RESOLUTION STEP
        original_size = header_img.size
        new_size = (original_size[0] * 3, original_size[1] * 3)
        header_img = header_img.resize(new_size, resample=Image.Resampling.LANCZOS)
        
        header_img_enhanced = enhance_image_for_digits(header_img)
        header_text = pytesseract.image_to_string(header_img_enhanced, config='--psm 6 --oem 3')
        header_text = fix_character_confusion(header_text)
        header_lines = [l.strip() for l in header_text.split('\n') if l.strip()]
    
    # === PHASE 2: FULL PAGE SCAN ===
    lines = []
    if not skip_full_page:
        print(f"   📄 [RobustOCR] Converting full page an preprocessing...")
        pix = page.get_pixmap(dpi=150)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        
        # ... (preprocessing logic)
        img_processed = img.convert('L')
        # ... (omitting lengthy preprocessing lines for brevity in replacement if unchanged, but need to be careful)
        # To be safe and avoid "omission" errors in replace_file_content, I'll copy the preprocessing logic exactly as is or target smaller chunks.
        # Actually, let's just target the function signature and the classification logic block further down.
        # It is safer to make TWO edits. One for signature, one for logic.
        
    # Wait, the user wants me to fix the signature first.
    pass

# Resetting strategy: distinct edits.

    import cv2
    import pytesseract
    from pytesseract import Output
    print(f"\n🔍 [RobustOCR] Processing Page {page_num}...")
    
    # Open PDF and get page
    should_close = False
    if doc_handle:
        doc = doc_handle
    else:
        doc = fitz.open(pdf_path)
        should_close = True
    
    if page_num > len(doc):
        print(f"❌ Page {page_num} does not exist (PDF has {len(doc)} pages)")
        if should_close: doc.close()
        return None
    
    page = doc[page_num - 1]  # 0-indexed
    
    header_lines = []
    ocr_lines = []
    skip_full_page = False

    # === PHASE 1: CERTIFICATE HEADER SCAN ===
    if ULTRA_FAST_MODE:
        # Ultra-Fast logic (skipped for backend usually, but keeping for parity)
        pix = page.get_pixmap(dpi=120)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        width, height = img.size
        img = img.crop((0, 0, width, int(height * 0.33)))
        img = img.convert('L')
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(2.0)
        custom_config = '--psm 6 --oem 3 -c tessedit_char_whitelist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-:. "'
        text = pytesseract.image_to_string(img, config=custom_config)
        text = fix_character_confusion(text)
        header_lines = [l.strip() for l in text.split('\n') if l.strip()]
        skip_full_page = True 
        ocr_lines = header_lines # In ultra-fast, this is it
    
    else:
        # Normal/Robust Mode: Super-Resolution Header Scan
        print(f"   📄 [RobustOCR] Scanning certificate header (Super-Res Mode)...")
        page_rect = page.rect
        header_clip = fitz.Rect(0, 0, page_rect.width, page_rect.height * 0.33)
        header_pix = page.get_pixmap(dpi=150, clip=header_clip)
        header_img = Image.frombytes("RGB", [header_pix.width, header_pix.height], header_pix.samples)
        
        # SUPER-RESOLUTION STEP: 300% Upscale
        original_size = header_img.size
        new_size = (original_size[0] * 3, original_size[1] * 3)
        header_img = header_img.resize(new_size, resample=Image.Resampling.LANCZOS)
        
        # Enhanced preprocessing
        header_img_enhanced = enhance_image_for_digits(header_img)
        
        # OCR header with PSM 6
        header_text = pytesseract.image_to_string(header_img_enhanced, config='--psm 6 --oem 3')
        header_text = fix_character_confusion(header_text)
        header_lines = [l.strip() for l in header_text.split('\n') if l.strip()]
    
    # === PHASE 2: FULL PAGE SCAN ===
    lines = []
    if not skip_full_page:
        print(f"   📄 [RobustOCR] Converting full page an preprocessing...")
        pix = page.get_pixmap(dpi=150)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        
        # --- Advanced Preprocessing from main.py ---
        img_processed = img.convert('L')
        img_np = np.array(img_processed)
        img_filtered = cv2.bilateralFilter(img_np, 9, 75, 75)
        img_pil = Image.fromarray(img_filtered)
        img_pil = ImageEnhance.Contrast(img_pil).enhance(1.8)
        img_pil = ImageEnhance.Sharpness(img_pil).enhance(1.5)
        img_np = np.array(img_pil)
        kernel = np.ones((2, 2), np.uint8)
        img_np = cv2.morphologyEx(img_np, cv2.MORPH_CLOSE, kernel)
        img_processed = Image.fromarray(img_np)
        
        # DEBUG SAVE
        if output_dir:
            try:
                debug_path = Path(output_dir) / f"debug_page_{page_num}_prep_backend.png"
                img_processed.save(debug_path)
                print(f"   🐛 DEBUG: Saved preprocessed page to {debug_path}")
            except Exception:
                pass

        # Run Tesseract
        print(f"   🔤 [RobustOCR] Running Tesseract OCR...")
        ocr_data = pytesseract.image_to_data(img_processed, output_type=Output.DICT)
        
        # Extract Lines
        current_line = []
        current_line_num = -1
        for i in range(len(ocr_data['text'])):
            conf = int(ocr_data['conf'][i])
            text = ocr_data['text'][i].strip()
            line_num = ocr_data['line_num'][i]
            
            if conf > 0 and text:
                if line_num != current_line_num:
                    if current_line: lines.append(' '.join(current_line))
                    current_line = [text]
                    current_line_num = line_num
                else:
                    current_line.append(text)
        if current_line: lines.append(' '.join(current_line))
    
    # Gemini Enhancement (Optional)
    engine = OCR_ENGINE.lower()
    if engine == 'gemini-enhanced' or (GEMINI_API_KEY and engine != 'tesseract'):
        gemini_result = enhance_ocr_with_gemini('\n'.join(lines), lines)
        if gemini_result.get('gemini_used'):
            lines = gemini_result['corrected_lines']
            print("   ✅ Using Gemini-enhanced results")

    # Fix confusions
    lines = [fix_character_confusion(line) for line in lines]
    
    # === MERGE HIGH-DPI HEADER LINES ===
    # Prioritize specialized header scan results
    if header_lines and not skip_full_page:
        for hl in header_lines:
            clean_hl = hl.replace('|', 'I').replace('lN-', 'IN-').replace('1N-', 'IN-')
            if re.search(r'IN-[A-Z]{2}', clean_hl) or 'Certificate' in clean_hl:
                lines = [l for l in lines if clean_hl not in l and l not in clean_hl]
                lines.insert(0, clean_hl)
                print(f"      ✅ [High-DPI Merge]: {clean_hl}")

    # === CLASSIFY DOCUMENT (Early) ===
    full_text_check = '\n'.join(lines)
    
    if prior_doc_type:
        doc_type = prior_doc_type
        confidence = 1.0
        is_certificate_regex = False
        print(f"   🧠 [Classifier] Using prior classification: {doc_type.value}")
    else:
        try:
            from document_classifier import classify_document
            from document_processors.base_processor import DocumentType
            
            # Check for forceful "Certificate" override via regex first (legacy compatibility)
            is_certificate_regex = re.search(r'IN-[A-Z]{2}.*\d{5}|Stamp Duty', full_text_check, re.IGNORECASE)
            
            if is_certificate_regex:
                 doc_type = DocumentType.FINANCIAL
                 confidence = 1.0
                 print(f"   🧠 [Classifier] heuristic-override -> FINANCIAL (Certificate)")
            else:
                 doc_type, confidence = classify_document(full_text_check, lines)
                 print(f"   🧠 [Classifier] Detected: {doc_type.value} (Confidence: {confidence})")
        except ImportError:
            print("   ⚠️  [Classifier] Module missing, defaulting to legacy.")
            doc_type = DocumentType.UNKNOWN
            confidence = 0.0
            is_certificate_regex = False

    # === FALLBACKS (CONDITIONAL) ===
    # Only run expensive fallbacks if it LOOKS like a certificate but is missing the number
    should_run_fallbacks = (
        not ULTRA_FAST_MODE and 
        (doc_type == DocumentType.FINANCIAL or doc_type == DocumentType.UNKNOWN) and
        not re.search(r'IN-[A-Z]{2}.*\d{5}', full_text_check, re.IGNORECASE)
    )

    if should_run_fallbacks:
        # === FALLBACK: OPENCV SMART DETECTION ===
        print("   ⚠️  Certificate MISSING. Engaging OPENCV SMART DETECTION...")
        try:
             # Re-use 'img' from Phase 2 (RGB) for finding contours
             candidates = find_certificate_cv(img)
             
             for i, crop in enumerate(candidates):
                 # Super-Res Crop
                 original_size = crop.size
                 new_size = (original_size[0] * 3, original_size[1] * 3)
                 crop_scaled = crop.resize(new_size, resample=Image.Resampling.LANCZOS)
                 crop_gray = crop_scaled.convert('L')
                 crop_enh = ImageEnhance.Contrast(crop_gray).enhance(2.0)
                 crop_enh = ImageEnhance.Sharpness(crop_enh).enhance(3.0)
                 
                 text = pytesseract.image_to_string(crop_enh, config='--psm 7')
                 clean_text = text.strip().replace('|', 'I').replace('lN-', 'IN-')
                 
                 if clean_text and (re.search(r'IN-[A-Z]{2}', clean_text) or re.search(r'[A-Z]{2}\d{10,}', clean_text)):
                      if clean_text not in lines:
                          print(f"      ✅ OPENCV RECOVERED: {clean_text}")
                          lines.insert(0, clean_text)
                          break
        except Exception as e:
             print(f"      ⚠️  OpenCV Fallback failed: {e}")

        # === FALLBACK: NUCLEAR THRESHOLD SCAN ===
        full_text_check = '\n'.join(lines) # Re-check after OpenCV
        if not re.search(r'IN-[A-Z]{2}.*\d{5}', full_text_check, re.IGNORECASE):
            print("   ⚠️  Certificate MISSING. Engaging MULTI-THRESHOLD NUCLEAR mode...")
            try:
                width, height = img.size
                header_crop = img.crop((0, 0, width, int(height * 0.50)))
                header_gray = header_crop.convert('L')
                
                strategies = [
                    ('Normal-120', header_gray.point(lambda p: p > 120 and 255)),
                    ('Normal-150', header_gray.point(lambda p: p > 150 and 255)),
                    ('Normal-180', header_gray.point(lambda p: p > 180 and 255)),
                    ('Inverted-150', ImageOps.invert(header_gray).point(lambda p: p > 150 and 255))
                ]
                
                recovered_any = False
                for name, bin_img in strategies:
                    if recovered_any: break
                    text = pytesseract.image_to_string(bin_img, config='--psm 6 --oem 3')
                    for hl in text.split('\n'):
                        clean_hl = hl.strip().replace('|', 'I').replace('l', 'I')
                        if (re.search(r'IN-[A-Z]{2}', clean_hl) or 'Cert' in clean_hl):
                             if clean_hl not in lines:
                                print(f"      ✅ RECOVERED ({name}): {clean_hl}")
                                lines.insert(0, clean_hl)
                                recovered_any = True
                                break
            except Exception as e:
                print(f"      ⚠️  Nuclear Fallback failed: {e}")
    else:
        if doc_type == DocumentType.LEGAL:
            print("   ⏩ [Optimization] Skipping certificate fallbacks for Rent Agreement.")

    # === EXTRACT FIELDS (DYNAMIC) ===
    full_text = '\n'.join(lines)
    
    # 2. Dispatch Extraction
    # (Classifier already ran earlier, using doc_type and is_certificate_regex from above)

    # 2. Dispatch Extraction
    extracted_fields = {}
    
    if doc_type == DocumentType.LEGAL:
        try:
            from document_processors.legal_processor import LegalProcessor
            print("   ⚖️  [Extraction] Using LegalProcessor for Rent Agreement/Contract")
            processor = LegalProcessor()
            extracted_fields = processor.extract_fields(full_text, lines)
        except Exception as e:
            print(f"   ❌ [Extraction] LegalProcessor failed: {e}")
            extracted_fields = {'error': 'Legal extraction failed'}
            
    elif is_certificate_regex or doc_type == DocumentType.FINANCIAL or 'certificate' in full_text.lower():
        # Keep existing certificate logic
        print("   💰 [Extraction] Using Standard Certificate Extraction")
        extracted_fields = extract_certificate_fields(lines)
        
    else:
        # Generic Fallback
        print(f"   ℹ️  [Extraction] Generic fallback for {doc_type.value}")
        extracted_fields = {
            'document_type': doc_type.value
        }
        
    # User Request: Always include full text
    extracted_fields['full_text'] = full_text



    if should_close:
        doc.close()

    # Calculate Avg Confidence
    # (Simplified for robustness, just setting a default if unavailable)
    avg_conf = 0.8 # Placeholder if not calculated from all parts
    
    result = {
        'page_number': page_num,
        'lines': lines,
        'full_text': '\n'.join(lines),
        'line_count': len(lines),
        'avg_confidence': avg_conf,
        'document_type': 'stamp_duty_certificate',
        'extracted_fields': extracted_fields
    }
    
    return result

    return extracted

def fix_common_ocr_errors(text):
    """
    Fix specific recurring OCR errors for this dataset.
    """
    if not text: return text
    
    # Date Corrections (Day 43 -> 13)
    text = re.sub(r'43-(Jun|Jul|Jan)', r'13-\1', text)
    text = re.sub(r'4([0-9])-(Jun|Jul|Jan)', r'1\1-\2', text) # Generalize 4X -> 1X for dates if >> 31
    
    return text

def extract_certificate_fields(lines):
    """
    Extract specific key-value pairs where the value is to the right of the key.
    """
    fields_to_extract = [
        "Certificate No.",
        "Certificate Issued Date",
        "Account Reference",
        "Unique Doc. Reference",
        "Purchased by",
        "Description of Document",
        "Description",
        "Consideration Price (Rs.) First Party",
        "Second Party",
        "Stamp Duty Paid By"
    ]
    # Sort keys by length descending to prevent "Description" matching inside "Description of Document"
    fields_to_extract.sort(key=len, reverse=True)
    
    extracted = {}
    
    # helper to clean value
    def clean_val(v):
        v = v.strip().lstrip(':').replace('|', '').strip()
        return fix_common_ocr_errors(v)

    for i, line in enumerate(lines):
        for key in fields_to_extract:
            if key.lower() in line.lower():
                # Normalize key for JSON
                json_key = key.lower().replace('.', '').replace(' ', '_').replace('(', '').replace(')', '')
                
                # Avoid overwriting if we already found a good value
                if json_key in extracted and len(extracted[json_key]) > 5:
                    continue

                value_found = ""
                
                # 1. Try Same Line
                pattern = re.compile(re.escape(key), re.IGNORECASE)
                match = pattern.search(line)
                if match:
                    same_line_val = line[match.end():].strip()
                    # If it's a date field, look for date pattern specifically
                    if "date" in json_key:
                        cleaned_val = fix_common_ocr_errors(same_line_val)
                        date_match = re.search(r'(\d{2}-[A-Za-z]{3}-\d{4})', cleaned_val)
                        if date_match:
                            value_found = date_match.group(1)
                        else:
                             # Check for noisy date (e.g. 43-Jun-2023)
                             noisy_match = re.search(r'(\d{2}-[A-Za-z]{3}-\d{4})', fix_common_ocr_errors(same_line_val))
                             if noisy_match:
                                 value_found = noisy_match.group(1)
                    else:
                        value_found = clean_val(same_line_val)
                        if "of document" in value_found.lower(): value_found = ""

                # 2. Try Next Line (if strictly empty or looks like noise)
                # Heuristic: If value is empty OR (it's a date field and we didn't find a date)
                should_check_next = not value_found
                if "date" in json_key and not re.search(r'\d{2}-[A-Za-z]{3}-\d{4}', value_found):
                    should_check_next = True
                    
                if should_check_next and i + 1 < len(lines):
                    next_line = lines[i+1].strip()
                    # Determine if next line is a separate key or the value
                    # If next line contains one of our other keys, it's not the value.
                    is_next_line_key = any(k.lower() in next_line.lower() for k in fields_to_extract)
                    
                    if not is_next_line_key:
                        if "date" in json_key:
                            cleaned_next = fix_common_ocr_errors(next_line)
                            date_match = re.search(r'(\d{2}-[A-Za-z]{3}-\d{4})', cleaned_next)
                            if date_match:
                                value_found = date_match.group(1)
                        else:
                            # For non-dates, assume next line is the value if reasonably short
                            # Fix for Stamp Duty Paid By consuming Stamp Duty Amount line
                            if "stamp duty amount" in next_line.lower():
                                value_found = "" # Do not consume next line
                            else:
                                value_found = clean_val(next_line)
                
                
                if value_found:
                    extracted[json_key] = value_found
    
    # === FALLBACK: Extract common fields from entire text using patterns ===
    full_text = '\n'.join(lines)
    
    # Certificate Number (IN-GJ format) - Relaxed Regex for Noisy OCR
    # Matches IN-GJ... followed by digits/chars, tolerating some noise
    if 'certificate_no' not in extracted or not extracted.get('certificate_no'):
        # Standard strict: IN-[A-Z]{2}\d{14}
        # Relaxed: IN-[A-Z]{2} then alphanumeric
        cert_match = re.search(r'(IN-[A-Z]{2}[A-Z0-9\s-]{14,20})', full_text)
        if cert_match:
            raw_cert = cert_match.group(1).replace(' ', '').replace('-', '')
            # Reformat to IN-GJ...
            if len(raw_cert) > 4:
                extracted['certificate_no'] = raw_cert[:2] + '-' + raw_cert[2:4] + raw_cert[4:]
    
    # Date (DD-MMM-YYYY format)
    if 'date' not in extracted or not extracted.get('date'):
        # Run fix on full text first
        cleaned_full = fix_common_ocr_errors(full_text)
        date_match = re.search(r'(\d{2}-[A-Za-z]{3}-\d{4})', cleaned_full)
        if date_match:
            extracted['date'] = date_match.group(1)
    
    # Amount (look for standalone numbers like 500)
    if 'amount' not in extracted or not extracted.get('amount'):
        # Look for lines with just a number (likely the amount)
        for line in lines:
            if line.strip().isdigit() and 100 <= int(line.strip()) <= 100000:
                extracted['amount'] = line.strip()
                break
    
    # Stamp Duty Type (Article 5(h) pattern) - Fixed to capture full text
    if 'stamp_duty_type' not in extracted or not extracted.get('stamp_duty_type'):
        article_match = re.search(r'(Article \d+\([a-z]\) [^\\n]+)', full_text, re.IGNORECASE)
        if article_match:
            extracted['stamp_duty_type'] = article_match.group(1).strip()
    
    # Account Reference (gj followed by 8-11 digits) - More precise
    if 'account' not in extracted or not extracted.get('account'):
        account_match = re.search(r'(gj\d{8,11})', full_text, re.IGNORECASE)
        if account_match:
            # Extract just the first 10-11 digits after 'gj'
            account_full = account_match.group(1).lower()
            extracted['account'] = account_full[:10] if len(account_full) > 10 else account_full
    
    # Branch (GJ-XXX pattern)
    if 'branch' not in extracted or not extracted.get('branch'):
        extracted['branch'] = 'GJ-XXX'  # Default for Gujarat
    
    # Generated On (extract from date if available)
    if 'generated_on' not in extracted and 'date' in extracted:
        try:
            from datetime import datetime
            date_obj = datetime.strptime(extracted['date'], '%d-%b-%Y')
            extracted['generated_on'] = date_obj.strftime('%d-%m-%y')
        except:
            pass
    
    # Party Name (ORBIS TRUSTEESHIP pattern)
    if 'first_party_name' not in extracted or not extracted.get('first_party_name'):
        party_match = re.search(r'(ORBIS TRUSTEESHIP SERVICES PRIVATE LIMITED)', full_text)
        if party_match:
            extracted['first_party_name'] = party_match.group(1)
            extracted['party_name'] = party_match.group(1)  # Also set party_name
    
    return extracted
