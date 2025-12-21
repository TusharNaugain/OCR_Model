def process_single_image_ocr(image_path, display_num=None, output_dir=None):
    """
    Process a standalone image file with OCR (PNG, JPG, JPEG).
    Returns the OCR result data similar to process_single_page_ocr.
    """
    from pathlib import Path
    from PIL import Image, ImageEnhance
    import pytesseract
    from pytesseract import Output
    import cv2
    import numpy as np
    from ocr_utils import (
        OCR_ENGINE, 
        GEMINI_API_KEY, 
        fix_character_confusion,
        enhance_ocr_with_gemini
    )
    
    # Import new document processing components
    try:
        from document_classifier import classify_document
        from document_processors import (
            FinancialProcessor, IDCardProcessor, FormProcessor,
            LegalProcessor, HealthcareProcessor, HistoricalProcessor,
            LogisticsProcessor, DocumentType
        )
        use_specialized_extraction = True
    except ImportError:
        use_specialized_extraction = False
    
    print(f"\n🔍 Processing Image: {Path(image_path).name}...")
    
    # Load image
    img = Image.open(image_path)
    
    # Advanced Preprocessing with OpenCV
    print(f"   🖼️  Applying advanced preprocessing...")
    cv_img = cv2.imread(image_path)
    if cv_img is not None:
        processed_cv = preprocess_image_cv2(cv_img)
        # Convert back to PIL for consistency if needed or use directly
        img_processed = Image.fromarray(processed_cv)
    else:
        # Fallback to basic PIL processing
        img_processed = img.convert('L')
        img_processed = ImageEnhance.Contrast(img_processed).enhance(1.5)
    
    # DEBUG: Save preprocessed image to verify grayscaling
    if output_dir:
        try:
            debug_path = Path(output_dir) / f"debug_prep_{Path(image_path).name}"
            img_processed.save(debug_path)
            print(f"   🐛 DEBUG: Saved preprocessed image to {debug_path}")
        except Exception as e:
            print(f"   ⚠️ Could not save debug image: {e}")
    
    # Run Tesseract OCR
    print(f"   🔤 Running Tesseract OCR...")
    ocr_data = pytesseract.image_to_data(img_processed, output_type=Output.DICT)
    
    # Extract text lines
    lines = []
    current_line = []
    current_line_num = -1
    
    for i in range(len(ocr_data['text'])):
        conf = int(ocr_data['conf'][i])
        text = ocr_data['text'][i].strip()
        line_num = ocr_data['line_num'][i]
        
        if conf > 0 and text:
            if line_num != current_line_num:
                if current_line:
                    lines.append(' '.join(current_line))
                current_line = [text]
                current_line_num = line_num
            else:
                current_line.append(text)
    
    if current_line:
        lines.append(' '.join(current_line))
    
    # Apply Gemini enhancement if enabled
    engine = OCR_ENGINE.lower()
    if engine == 'gemini-enhanced' or (GEMINI_API_KEY and engine != 'tesseract'):
        # Use Gemini to enhance and correct OCR results
        full_text = '\n'.join(lines)
        gemini_result = enhance_ocr_with_gemini(full_text, lines)
        
        if gemini_result.get('gemini_used'):
            lines = gemini_result['corrected_lines']
            print(f"   ✅ Using Gemini-enhanced results (confidence: {gemini_result.get('confidence', 0):.1%})")
        else:
            print(f"   ℹ️  Using Tesseract results (Gemini unavailable)")
    
    # Apply character confusion fixes
    print(f"   🔧 Applying character confusion fixes to {len(lines)} lines...")
    lines = [fix_character_confusion(line) for line in lines]
    
    # Compute average confidence
    confidences = [int(c) for c in ocr_data['conf'] if int(c) > 0]
    avg_conf = sum(confidences) / len(confidences) if confidences else 0
    
    # Prepare full text
    full_text = '\n'.join(lines)
    
    result = {
        'page_number': 1,  # Images are treated as single page
        'lines': lines,
        'full_text': full_text,
        'line_count': len(lines),
        'confidence': avg_conf / 100.0,
        'avg_confidence': avg_conf / 100.0,
        'source_file': Path(image_path).name
    }
    
    # Add specialized extraction if available
    if use_specialized_extraction:
        try:
            # Classify document
            doc_type, class_confidence = classify_document(full_text, lines, img)
            result['document_type'] = doc_type.value
            result['classification_confidence'] = class_confidence
            
            print(f"   📋 Classified as: {doc_type.value} (confidence: {class_confidence:.1%})")
            
            # Use appropriate processor
            processors = {
                DocumentType.FINANCIAL: FinancialProcessor(),
                DocumentType.ID_CARD: IDCardProcessor(),
                DocumentType.FORM: FormProcessor(),
                DocumentType.LEGAL: LegalProcessor(),
                DocumentType.HEALTHCARE: HealthcareProcessor(),
                DocumentType.HISTORICAL: HistoricalProcessor(),
                DocumentType.LOGISTICS: LogisticsProcessor(),
            }
            
            if doc_type in processors:
                processor = processors[doc_type]
                specialized_fields = processor.extract_fields(full_text, lines, img)
                is_valid, errors = processor.validate_fields(specialized_fields)
                
                result['extracted_fields'] = specialized_fields
                result['validation'] = {
                    'is_valid': is_valid,
                    'errors': errors
                }
                
                print(f"   ✅ Extracted {len(specialized_fields)} specialized fields")
                if not is_valid:
                    print(f"   ⚠️  Validation errors: {', '.join(errors)}")
        except Exception as e:
            print(f"   ⚠️  Specialized extraction failed: {e}")
    
    print(f"   ✅ Processing complete: {len(lines)} lines (avg confidence: {avg_conf:.1f}%)")
    
    return result


def preprocess_image_cv2(image):
    """
    Advanced preprocessing using OpenCV:
    1. Grayscale
    2. Bilateral Filter (Noise reduction while preserving edges)
    3. Adaptive Thresholding
    4. Dilation/Erosion (Cleanup)
    """
    import cv2
    import numpy as np
    
    # 1. Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # 2. Bilateral Filtering for noise reduction
    # d=9, sigmaColor=75, sigmaSpace=75
    denoised = cv2.bilateralFilter(gray, 9, 75, 75)
    
    # 3. Adaptive Thresholding
    # Gaussian thresholding handles uneven lighting better
    thresh = cv2.adaptiveThreshold(
        denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY, 11, 2
    )
    
    # 4. Morphological operations (Optional cleaning)
    kernel = np.ones((1, 1), np.uint8)
    processed = cv2.dilate(thresh, kernel, iterations=1)
    processed = cv2.erode(processed, kernel, iterations=1)
    
    return processed



