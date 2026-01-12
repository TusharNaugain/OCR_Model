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
import pandas as pd
from io import BytesIO
from fastapi.responses import StreamingResponse
import sentry_sdk

sentry_sdk.init(
    dsn=os.getenv("SENTRY_DSN", "https://0d0b514ea6edd21fc3616f5919196888@o4510350699659264.ingest.us.sentry.io/4510641231757312"),
    # Set traces_sample_rate to 1.0 to capture 100%
    # of transactions for performance monitoring.
    traces_sample_rate=1.0,
    # Set profiles_sample_rate to 1.0 to profile 100%
    # of sampled transactions.
    # We recommend adjusting this value in production.
    profiles_sample_rate=1.0,
)


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
from comparison_utils import compare_ocr_with_reference




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
                        detected_doc_type = None
                        
                        # 1. Process Page 1 Synchronously (for Classification)
                        jobs[job_id]["message"] = f"Analyzing document type (Page 1)..."
                        
                        pdf_pages_results = [None] * total_pages
                        
                        # Process Page 1
                        page1_res = process_single_page_ocr(
                            str(file_path), 1, output_dir=output_dir, doc_handle=None, prior_doc_type=None
                        )
                        
                        if page1_res:
                            pdf_pages_results[0] = page1_res
                            
                            # Detect Type
                            if page1_res.get('extracted_fields'):
                                doc_type_str = page1_res['extracted_fields'].get('document_type', '')
                                if doc_type_str:
                                    print(f"🔒 [API] Locking classification to: {doc_type_str}")
                                    from document_processors.base_processor import DocumentType
                                    try:
                                        detected_doc_type = DocumentType(doc_type_str)
                                    except:
                                        pass
                        
                        # 2. Process Page 2..N in Parallel
                        if total_pages > 1:
                            print(f"🚀 [API] Parallelizing pages 2-{total_pages}...")
                            from concurrent.futures import ThreadPoolExecutor, as_completed
                            import multiprocessing
                            
                            max_workers = min(total_pages - 1, multiprocessing.cpu_count() * 2) 
                            # Safe limit
                            if max_workers < 1: max_workers = 1
                            
                            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                                future_to_page = {
                                    executor.submit(
                                        process_single_page_ocr, 
                                        str(file_path), 
                                        p, 
                                        output_dir=output_dir, 
                                        doc_handle=None,
                                        prior_doc_type=detected_doc_type
                                    ): p for p in range(2, total_pages + 1)
                                }
                                
                                for i, future in enumerate(as_completed(future_to_page)):
                                    pg = future_to_page[future]
                                    jobs[job_id]["message"] = f"Processing parallel batch ({i+1}/{total_pages-1})..."
                                    try:
                                        res = future.result()
                                        if res:
                                            pdf_pages_results[pg - 1] = res
                                    except Exception as exc:
                                        print(f"❌ Page {pg} Failed: {exc}")

                        # Filter out Nones
                        pdf_pages_results = [p for p in pdf_pages_results if p is not None]

                        # === GLOBAL EXTRACTION (Optimization) ===
                        # If Legal/Rent Agreement OR Certificate, attempt to merge.
                        
                        final_doc_result = {}
                        
                        is_mergeable = False
                        if pdf_pages_results:
                             first_page_type = pdf_pages_results[0].get('extracted_fields', {}).get('document_type', '')
                             # Allow merging for Legal OR Certificates (since users upload multi-page stamp papers)
                             if first_page_type in ['rent_agreement', 'legal_document', 'contract', 'stamp_duty_certificate', 'financial']:
                                 is_mergeable = True

                        if is_mergeable:
                            print(f"✨ [API] Performing GLOBAL EXTRACTION for {first_page_type}...")
                            # 1. Aggregate Text
                            combined_text = "\n\n".join([p.get('full_text', '') for p in pdf_pages_results])
                            combined_lines = []
                            for p in pdf_pages_results:
                                combined_lines.extend(p.get('lines', []))
                                
                            # 2. Run Appropriate Processor on Full Text
                            try:
                                if first_page_type in ['stamp_duty_certificate', 'financial']:
                                     # GLOBAL CONSENSUS:
                                     # User uploaded a mulit-page stamp paper. Merging into 1 result.
                                     # Heuristic: If it contains 'Agreement', use LegalProcessor.
                                     # Else, treat as single Certificate (e.g. 1st page has details).
                                     
                                     print("   🌍 [Global] Merging Stamp Duty/Financial document...")
                                     
                                     has_agreement_text = "agreement" in combined_text.lower() or "contract" in combined_text.lower()
                                     
                                     if has_agreement_text:
                                         print("      -> Detected Agreement text. Using LegalProcessor.")
                                         from document_processors.legal_processor import LegalProcessor
                                         processor = LegalProcessor()
                                         global_fields = processor.extract_fields(combined_text, combined_lines)
                                         # Ensure we keep the Doc Type as 'stamp_duty' if identified
                                         if first_page_type == 'stamp_duty_certificate':
                                              global_fields['document_type'] = 'stamp_duty_certificate'
                                     else:
                                         # It's a pure certificate but multi-page?
                                         # Just extract using standard logic on the combined text (or Page 1 preference)
                                         print("      -> Pure Certificate. consolidating...")
                                         from pdf_ocr import extract_certificate_fields
                                         global_fields = extract_certificate_fields(combined_lines)

                                else:
                                    # Standard Legal Processor
                                    from document_processors.legal_processor import LegalProcessor
                                    processor = LegalProcessor()
                                    global_fields = processor.extract_fields(combined_text, combined_lines)
                                
                                global_fields['full_text'] = combined_text # Ensure full text is present
                                
                                # 3. Create Single Result Object
                                final_doc_result = global_fields
                                results.append(final_doc_result)
                                
                            except Exception as e:
                                print(f"Global extraction failed: {e}")
                                # Fallback to page 1
                                results.append(pdf_pages_results[0]['extracted_fields'])

                        else:
                            # STANDARD BEHAVIOR (Others)
                            # Return list of page results as before
                            extracted_pages = [p.get('extracted_fields', {}) for p in pdf_pages_results]
                            results.extend(extracted_pages)
                             
                    else:
                        # For images
                        ocr_res = process_single_image_ocr(
                            str(file_path),
                            display_num=idx + 1,
                            output_dir=output_dir
                        )
                        results.append(ocr_res.get('extracted_fields', {}))
                    
                    
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

@app.post("/api/ocr/compare")
async def compare_documents(
    reference_file: UploadFile = File(...),
    document_file: UploadFile = File(...)
):
    """
    Compare an uploaded document (PDF/Image) against a Reference File (Excel/CSV).
    Performs OCR on document and compares with Reference.
    """
    temp_dir = UPLOAD_DIR / f"temp_compare_{uuid.uuid4()}"
    temp_dir.mkdir(exist_ok=True)
    
    try:
        # 1. Save Files
        ref_path = temp_dir / reference_file.filename
        doc_path = temp_dir / document_file.filename
        
        with open(ref_path, "wb") as f:
            shutil.copyfileobj(reference_file.file, f)
        with open(doc_path, "wb") as f:
            shutil.copyfileobj(document_file.file, f)
            
        # 2. Run OCR on Document (Synchronous for now, assuming 1 file)
        # Determine type
        extracted_data = {}
        is_pdf = doc_path.suffix.lower() == '.pdf'
        
        if is_pdf:
            import fitz
            doc = fitz.open(doc_path)
            # Just process first page for now for comparison, or merge if multi-page?
            # User requirement implies "A document". Let's try page 1 or merge all.
            # For robustness, let's just do page 1 as it usually contains the summary/cert details.
            # TODO: Add multi-page support if needed.
            
            # Use Robust OCR
            res = process_single_page_ocr(str(doc_path), 1, output_dir=temp_dir)
            if res and 'extracted_fields' in res:
                extracted_data = res['extracted_fields']
            
            doc.close()
            
        else:
            # Image
            res = process_single_image_ocr(str(doc_path), output_dir=temp_dir)
            if res and 'extracted_fields' in res:
                extracted_data = res['extracted_fields']
        
        if not extracted_data:
             return JSONResponse(status_code=400, content={"error": "OCR failed to extract data from document"})

        # 3. Validation / Comparison
        comparison_result = compare_ocr_with_reference(extracted_data, str(ref_path))
        
        # 4. Cleanup
        shutil.rmtree(temp_dir)
        
        return {
            "status": "success",
            "ocr_data": extracted_data,
            "comparison": comparison_result
        }

    except Exception as e:
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
        return JSONResponse(status_code=500, content={"error": str(e)})

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

@app.get("/api/ocr/export/{job_id}")
async def export_results(job_id: str, format: str = "csv"):
    """
    Export OCR results to CSV or Excel.
    """
    data_to_export = []
    
    # 1. Try filesystem (Standard OCR)
    results_path = RESULTS_DIR / job_id / "results.json"
    if results_path.exists():
        try:
             with open(results_path, 'r') as f:
                 raw = json.load(f)
                 # Unpack if it's wrapped in structure
                 if isinstance(raw, list): data_to_export = raw
                 elif isinstance(raw, dict) and "results" in raw: data_to_export = raw["results"]
                 elif isinstance(raw, dict) and "ocr_data" in raw: data_to_export = raw["ocr_data"] 
        except:
            pass
            
    # 2. If valid data, export
    if not data_to_export:
        raise HTTPException(status_code=404, detail="Job results not found or empty")

    # Normalize
    df = pd.DataFrame(data_to_export)
    
    # USER REQUEST: Add spacing (blank row) between every page/record for better readability
    # Method: Create a new index with gaps, or just reconstruct the list
    if not df.empty:
        # Create a list interleaved with empty rows
        interleaved = []
        empty_row = {col: "" for col in df.columns}
        
        for _, row in df.iterrows():
            interleaved.append(row.to_dict())
            interleaved.append(empty_row) # Add blank row
            
        # Remove the very last empty row if desired, or keep it. keeping for consistency
        df = pd.DataFrame(interleaved)
    
    # Clean up column names 
    df.columns = [str(c).replace('_', ' ').title() for c in df.columns]

    stream = BytesIO()
    
    if format.lower() == 'xlsx':
        # Excel
        with pd.ExcelWriter(stream, engine='openpyxl') as writer:
            df.to_excel(writer, index=False, sheet_name='OCR Data')
        media_type = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        filename = f"ocr_results_{job_id}.xlsx"
    else:
        # CSV
        df.to_csv(stream, index=False)
        media_type = "text/csv"
        filename = f"ocr_results_{job_id}.csv"
        
    stream.seek(0)
    
    return StreamingResponse(
        stream, 
        media_type=media_type, 
        headers={"Content-Disposition": f"attachment; filename={filename}"}
    )

if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
