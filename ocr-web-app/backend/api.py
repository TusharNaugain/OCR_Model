from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional, List
import uvicorn
import os
import uuid
import json
import shutil
from pathlib import Path
from datetime import datetime
import subprocess

app = FastAPI(
    title="SmartScan OCR API",
    description="AI-Powered OCR with Gemini Enhancement",
    version="1.0.0"
)

# CORS configuration for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for deployment
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Storage configuration
UPLOAD_DIR = Path("uploads")
RESULTS_DIR = Path("results")
UPLOAD_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

# Job status tracking
jobs = {}

class JobStatus(BaseModel):
    job_id: str
    status: str  # pending, processing, completed, failed
    progress: int  # 0-100
    message: str
    created_at: str
    completed_at: Optional[str] = None
    file_count: int = 0
    processed_count: int = 0

class OCRResult(BaseModel):
    job_id: str
    status: str
    results: List[dict]
    error: Optional[str] = None

# Imports are now local


# Import from lighter modules to avoid heavy dependencies
from ocr_utils import OCR_ENGINE, get_gemini_model
from image_ocr import process_single_image_ocr
from pdf_ocr import process_single_page_ocr_robust as process_single_page_ocr




def run_ocr_process(job_id: str, files: List[Path]):
    """Run OCR process on uploaded files with parallel processing and document classification"""
    try:
        jobs[job_id]["status"] = "processing"
        jobs[job_id]["message"] = "Processing files with OCR..."
        
        # Create job directory
        job_dir = UPLOAD_DIR / job_id
        output_dir = RESULTS_DIR / job_id
        output_dir.mkdir(exist_ok=True)
        
        results = []
        total_files = len(files)
        
        # Try to use parallel processor if available
        try:
            from parallel_processor import ParallelOCRProcessor
            use_parallel = total_files > 1  # Use parallel for multiple files
        except ImportError:
            use_parallel = False
        
        if use_parallel and total_files > 1:
            # Use parallel processing for better performance
            print(f"Using parallel processing for {total_files} files")
            processor = ParallelOCRProcessor(max_workers=max(1, min(4, total_files)))
            
            # Process all files in parallel
            batch_result = processor.process_with_classification(
                files,
                None,  # Will use internal processing
                output_dir,
                classify_first=True
            )
            
            # Update job progress
            jobs[job_id]["processed_count"] = total_files
            jobs[job_id]["progress"] = 90
            
            # Format results
            for item in batch_result['results']:
                if item.get('status') == 'success':
                    ocr_res = item.get('ocr_result', {})
                    result = {
                        "file": Path(item['document']).name,
                        "status": "success",
                        "document_type": item.get('document_type', 'unknown'),
                        "classification_confidence": item.get('classification_confidence', 0),
                        "extracted_text": {
                            "lines": ocr_res.get('lines', []),
                            "full_text": ocr_res.get('full_text', ''),
                            "confidence": ocr_res.get('confidence', 0)
                        },
                        "extracted_fields": item.get('specialized_fields', {}),
                        "validation": item.get('validation', {})
                    }
                    results.append(result)
                else:
                    results.append({
                        "file": Path(item['document']).name,
                        "status": "error",
                        "error": item.get('error', 'Unknown error')
                    })
            
        else:
            # Process files sequentially (fallback or single file)
            for idx, file_path in enumerate(files):
                # Update progress
                jobs[job_id]["processed_count"] = idx
                jobs[job_id]["progress"] = int((idx / total_files) * 90)
                jobs[job_id]["message"] = f"Processing file {idx + 1} of {total_files}..."
                
                try:
                    # Determine file type
                    is_pdf = file_path.suffix.lower() == '.pdf'
                    
                    if is_pdf:
                        # Process ALL pages
                        import fitz
                        doc = fitz.open(file_path)
                        total_pages = len(doc)
                        doc.close()
                        
                        pdf_pages_results = []
                        for pg in range(1, total_pages + 1):
                            jobs[job_id]["message"] = f"Processing file {idx + 1}/{total_files} (Page {pg}/{total_pages})..."
                            
                            page_res = process_single_page_ocr(
                                str(file_path), 
                                page_num=pg, 
                                output_dir=output_dir, 
                                doc_handle=None
                            )
                            if page_res:
                                pdf_pages_results.append(page_res)
                        
                        # Use results from Page 1 for high-level summary, but include all pages
                        if pdf_pages_results:
                            ocr_res = pdf_pages_results[0].copy()
                            # Aggregate full text from all pages
                            full_multipage_text = "\n\n".join([p.get('full_text', '') for p in pdf_pages_results])
                            ocr_res['full_text'] = full_multipage_text
                            ocr_res['pages'] = pdf_pages_results
                        else:
                             ocr_res = {'lines': [], 'full_text': '', 'confidence': 0}
                             
                    else:
                        # For images
                        ocr_res = process_single_image_ocr(
                            str(file_path),
                            display_num=idx + 1,
                            output_dir=output_dir
                        )
                        ocr_res['pages'] = [ocr_res] # Treat image as 1-page doc
                    
                    
                    # Format results to match CLI output structure
                    # USER REQUEST: Save ONLY extracted_fields per page (no wrapper)
                    page_results = []
                    for page_data in ocr_res.get('pages', []):
                        # Save only the extracted fields, exactly like CLI output
                        page_result = page_data.get('extracted_fields', {})
                        page_results.append(page_result)
                    
                    # Add all page results to main results
                    results.extend(page_results)
                    
                except Exception as e:
                    print(f"Error processing file {file_path}: {e}")
                    results.append({
                        "file": file_path.name,
                        "status": "error",
                        "error": str(e)
                    })
        
        # Save results
        result_file = output_dir / "results.json"
        with open(result_file, 'w') as f:
            json.dump(results, f, indent=2, default=str, ensure_ascii=False)
        
        jobs[job_id]["status"] = "completed"
        jobs[job_id]["progress"] = 100
        jobs[job_id]["message"] = "OCR processing complete!"
        jobs[job_id]["completed_at"] = datetime.now().isoformat()
        jobs[job_id]["processed_count"] = total_files
        jobs[job_id]["results_path"] = str(result_file)
        
    except Exception as e:
        jobs[job_id]["status"] = "failed"
        jobs[job_id]["message"] = f"Error: {str(e)}"
        jobs[job_id]["error"] = str(e)


@app.get("/")
async def root():
    return {
        "name": "SmartScan OCR API",
        "version": "1.0.0",
        "status": "running",
        "endpoints": {
            "upload": "/api/ocr/upload",
            "status": "/api/ocr/status/{job_id}",
            "results": "/api/ocr/results/{job_id}"
        }
    }

@app.post("/api/ocr/upload")
async def upload_files(
    background_tasks: BackgroundTasks,
    files: List[UploadFile] = File(...)
):
    """Upload files for OCR processing"""
    try:
        # Generate job ID
        job_id = str(uuid.uuid4())
        
        # Create job directory
        job_dir = UPLOAD_DIR / job_id
        job_dir.mkdir(parents=True, exist_ok=True)
        
        # Save uploaded files
        saved_files = []
        for file in files:
            file_path = job_dir / file.filename
            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            saved_files.append(file_path)
        
        # Initialize job status
        jobs[job_id] = {
            "job_id": job_id,
            "status": "pending",
            "progress": 0,
            "message": "Files uploaded, queued for processing...",
            "created_at": datetime.now().isoformat(),
            "file_count": len(saved_files),
            "processed_count": 0,
            "files": [f.name for f in saved_files]
        }
        
        # Start background processing
        background_tasks.add_task(run_ocr_process, job_id, saved_files)
        
        return {
            "job_id": job_id,
            "status": "pending",
            "message": "Files uploaded successfully. Processing started.",
            "file_count": len(saved_files)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/ocr/status/{job_id}")
async def get_status(job_id: str):
    """Get processing status for a job"""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    return JobStatus(**jobs[job_id])

@app.get("/api/ocr/results/{job_id}")
async def get_results(job_id: str):
    """Get OCR results for a completed job"""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = jobs[job_id]
    
    if job["status"] != "completed":
        raise HTTPException(
            status_code=400, 
            detail=f"Job not completed yet. Current status: {job['status']}"
        )
    
    # Load results from file
    result_file = Path(job["results_path"])
    if not result_file.exists():
        raise HTTPException(status_code=404, detail="Results file not found")
    
    with open(result_file, 'r') as f:
        results = json.load(f)
    
    return OCRResult(
        job_id=job_id,
        status=job["status"],
        results=results
    )

@app.delete("/api/ocr/job/{job_id}")
async def delete_job(job_id: str):
    """Delete a job and its files"""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    # Delete files
    job_dir = UPLOAD_DIR / job_id
    result_dir = RESULTS_DIR / job_id
    
    if job_dir.exists():
        shutil.rmtree(job_dir)
    if result_dir.exists():
        shutil.rmtree(result_dir)
    
    # Remove from jobs
    del jobs[job_id]
    
    return {"message": "Job deleted successfully"}

if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
