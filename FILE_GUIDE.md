# Quick Reference: File Purposes

## 🎯 Core OCR Files

| File | Purpose | When to Use |
|------|---------|-------------|
| **main.py** | PDF processing pipeline | Process multi-page PDFs from command line |
| **image_ocr.py** | Image processing + classification | Process JPG/PNG images with auto-classification |
| **ocr_utils.py** | Shared utilities | Used internally by other files |

## 🤖 Intelligence Layer

| File | Purpose | What it Does |
|------|---------|--------------|
| **document_classifier.py** | Auto-detect document type | Determines if document is invoice, passport, form, etc. |
| **parallel_processor.py** | Batch processing | Processes 100+ docs in parallel (2-4x faster) |

## 📋 Document Processors (document_processors/)

| File | Handles | Key Fields Extracted |
|------|---------|---------------------|
| **base_processor.py** | Common utilities | Image preprocessing, validation, cleaning |
| **financial_processor.py** | Invoices, receipts | Amounts, dates, vendor, line items |
| **id_card_processor.py** | Passports, licenses | Name, DOB, ID#, MRZ parsing |
| **form_processor.py** | Tax forms, surveys | Field-value pairs, checkboxes |
| **legal_processor.py** | Contracts, court docs | Case#, parties, court name |
| **healthcare_processor.py** | Medical records | Patient info, medications, diagnosis |
| **historical_processor.py** | Old books, newspapers | Title, author, archive# |
| **logistics_processor.py** | Shipping, orders | Tracking#, PO#, carrier |

## 🌐 Web Application

| File | Purpose | What it Does |
|------|---------|--------------|
| **ocr-web-app/backend/api.py** | REST API | Handles file uploads, returns results |
| **ocr-web-app/frontend/** | Web UI | User interface for uploading documents |

---

## 🔄 Processing Flow (Simplified)

```
Upload Document
    ↓
Tesseract OCR → Extract Text
    ↓
Gemini AI (optional) → Fix Errors
    ↓
Classifier → "This is an invoice"
    ↓
FinancialProcessor → Extract amounts, dates, vendor
    ↓
Validation → Check data quality
    ↓
Return JSON Results
```

---

## 💡 Real-World Example

**You upload:** `invoice.pdf`

**What happens:**
1. **image_ocr.py** converts it to image & runs OCR
2. **document_classifier.py** detects it's a financial document
3. **financial_processor.py** extracts invoice number, amounts, dates
4. **Validation** checks if subtotal + tax = total
5. **api.py** returns complete JSON with all fields

**You get:**
```json
{
  "document_type": "financial",
  "extracted_fields": {
    "invoice_number": "INV-12345",
    "total_amount": "$1,234.56",
    "date": "01/15/2024"
  }
}
```

---

## 📊 When to Use What

### Processing a Single PDF
```bash
python3 main.py document.pdf
```

### Processing Images
```python
from image_ocr import process_single_image_ocr
result = process_single_image_ocr("photo.jpg")
```

### Batch Processing (100+ files)
```python
from parallel_processor import ParallelOCRProcessor
processor = ParallelOCRProcessor(max_workers=4)
results = processor.process_with_classification(files)
```

### Via Web Interface
1. Open `http://localhost:3000`
2. Upload files
3. Get results automatically

---

## 🎯 Key Takeaways

✅ **main.py** = Old PDFs processing (legacy)  
✅ **image_ocr.py** = New smart processing with classification  
✅ **Processors** = Specialized extractors for each document type  
✅ **Classifier** = Auto-detects what kind of document it is  
✅ **Parallel** = Makes batching super fast  
✅ **API** = Powers the web interface  

The system now **automatically figures out** what type of document you're processing and **extracts the right fields** - no manual configuration needed! 🚀
