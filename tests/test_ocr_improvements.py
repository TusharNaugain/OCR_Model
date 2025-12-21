import sys
import os
import cv2
import numpy as np
from pathlib import Path
from PIL import Image

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from image_ocr import preprocess_image_cv2, process_single_image_ocr
from document_processors.form_processor import FormProcessor

def test_preprocessing():
    print("Testing Preprocessing...")
    # Create a dummy image
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    cv2.putText(img, "TEST", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    processed = preprocess_image_cv2(img)
    assert processed is not None
    assert len(processed.shape) == 2 # Should be grayscale
    print("✅ Preprocessing check passed")

def test_form_processor_academic():
    print("Testing Academic Form Extraction...")
    processor = FormProcessor()
    text = """
    UNIVERSITY OF TECHNOLOGY
    ENROLLMENT NO: 123456789
    CANDIDATE NAME: JOHN DOE
    BATCH: 2021-2025
    SEMESTER: 5
    
    SUBJECTS            INTERNAL EXTERNAL
    Mathematics         25       70
    Physics             20       65
    Computer Science    28       72
    """
    lines = text.strip().split('\n')
    
    fields = processor.extract_fields(text, lines)
    
    print(f"Extracted fields: {fields}")
    
    assert fields['form_type'] == 'academic_form'
    assert fields['enrollment_no'] == '123456789'
    assert fields['candidate_name'] == 'JOHN DOE'
    assert fields['batch_semester'] == '2021-2025'
    
    # Check results table
    results = fields.get('academic_results', [])
    assert len(results) >= 3
    assert results[0]['subject'] == 'Mathematics'
    assert results[0]['marks_obtained'] == '25'
    print("✅ Academic form extraction check passed")

def test_overall_pipeline_basic():
    print("Testing Overall Pipeline (Basic Check)...")
    # Use existing image if possible
    sample_img = "tesscrate_input/PHOTO-2025-10-08-14-31-28.jpg"
    if os.path.exists(sample_img):
        result = process_single_image_ocr(sample_img)
        print(f"OCR Confidence: {result['confidence']:.2%}")
        assert result['confidence'] > 0
        print("✅ Overall pipeline basic check passed")
    else:
        print("⚠️  Sample image not found, skipping pipeline check")

if __name__ == "__main__":
    try:
        test_preprocessing()
        test_form_processor_academic()
        test_overall_pipeline_basic()
        print("\n✨ All tests passed!")
    except Exception as e:
        print(f"\n❌ Tests failed: {e}")
        sys.exit(1)
