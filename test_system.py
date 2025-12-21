#!/usr/bin/env python3
"""
Quick Test Script for Multi-Document OCR System
"""

import sys
from pathlib import Path

# Test imports
print("Testing imports...")

try:
    from document_classifier import classify_document, DocumentType
    print("✅ Document classifier imported")
except Exception as e:
    print(f"❌ Document classifier import failed: {e}")

try:
    from document_processors import (
        BaseDocumentProcessor,
        FinancialProcessor,
        IDCardProcessor,
        FormProcessor,
        LegalProcessor,
        HealthcareProcessor,
        HistoricalProcessor,
        LogisticsProcessor
    )
    print("✅ All document processors imported")
except Exception as e:
    print(f"❌ Document processors import failed: {e}")

try:
    from parallel_processor import ParallelOCRProcessor
    print("✅ Parallel processor imported")
except Exception as e:
    print(f"❌ Parallel processor import failed: {e}")

# Test instantiation
print("\nTesting processor instantiation...")

try:
    processors = {
        'financial': FinancialProcessor(),
        'id_card': IDCardProcessor(),
        'form': FormProcessor(),
        'legal': LegalProcessor(),
        'healthcare': HealthcareProcessor(),
        'historical': HistoricalProcessor(),
        'logistics': LogisticsProcessor(),
    }
    print(f"✅ Created {len(processors)} processors successfully")
    
    # Test expected fields
    for name, processor in processors.items():
        fields = processor.get_expected_fields()
        print(f"   - {name}: {len(fields)} expected fields")
    
except Exception as e:
    print(f"❌ Processor instantiation failed: {e}")

# Test classification with sample text
print("\nTesting document classification...")

test_documents = {
    'invoice': "INVOICE #12345\nDate: 01/15/2024\nTotal Amount: $1,234.56\nSubtotal: $1,100.00\nTax: $134.56",
    'passport': "PASSPORT\nPassport No: AB1234567\nName: JOHN DOE\nDate of Birth: 01/01/1990\nNationality: USA\nExpiry Date: 01/01/2030",
    'form': "APPLICATION FORM\nName: _______________\nAddress: _______________\n[X] Yes  [ ] No\nSignature: _______________",
}

for doc_name, sample_text in test_documents.items():
    try:
        lines = sample_text.split('\n')
        doc_type, confidence = classify_document(sample_text, lines, None)
        print(f"✅ '{doc_name}' classified as: {doc_type.value} (confidence: {confidence:.1%})")
    except Exception as e:
        print(f"❌ Classification of '{doc_name}' failed: {e}")

print("\n✨ Basic tests complete!")
print("\nNext steps:")
print("1. Upload test documents via the web interface")
print("2. Check that documents are automatically classified")
print("3. Verify specialized fields are extracted correctly")
print("4. Test batch upload with multiple documents")
