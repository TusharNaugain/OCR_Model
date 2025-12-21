# Example Simplified JSON Output

This is the new cleaner JSON format:

```json
{
  "source": {
    "file": "document_name.pdf",
    "page_number": 1,
    "is_image": false
  },
  "ocr_result": {
    "lines": [
      "Line 1 of extracted text",
      "Line 2 of extracted text",
      "Line 3 of extracted text"
    ],
    "full_text": "Line 1 of extracted text\nLine 2...",
    "confidence": 0.89
  },
  "extracted_fields": {
    "certificate_no": "IN-GJ55531064934629X",
    "stamp_duty_amount": "500",
    "stamp_duty_type": "Article 5(h) Agreement"
  },
  "csv_match": {
    "matched": true,
    "certificate_number": "IN-GJ55531064934629X",
    "match_score": 100
  }
}
```

**Key Changes:**
- ✅ Clean, structured format
- ✅ `lines` array containing all extracted text
- ✅ CSV match only included when CSV exists
- ✅ Removed unnecessary metadata fields
- ✅ Source info includes file type (PDF vs image)
