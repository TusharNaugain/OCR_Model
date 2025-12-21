# FastAPI Backend for SmartScan OCR

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run the server:
```bash
python main.py
```

The API will be available at `http://localhost:8000`

## API Documentation

Once running, visit:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## Endpoints

### POST /api/ocr/upload
Upload files for OCR processing.

**Request:** Multipart form data with files

**Response:**
```json
{
  "job_id": "uuid",
  "status": "pending",
  "message": "Files uploaded successfully",
  "file_count": 2
}
```

### GET /api/ocr/status/{job_id}
Get processing status.

**Response:**
```json
{
  "job_id": "uuid",
  "status": "processing",
  "progress": 45,
  "message": "Processing file 2 of 5...",
  "file_count": 5,
  "processed_count": 2
}
```

### GET /api/ocr/results/{job_id}
Get OCR results (only when status is "completed").

**Response:**
```json
{
  "job_id": "uuid",
  "status": "completed",
  "results": [...]
}
```
