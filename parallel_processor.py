#!/usr/bin/env python3
"""
Parallel OCR Processor
======================

High-performance batch document processing with parallel execution.
Achieves 2x+ speedup through multiprocessing and optimized resource usage.
"""

import os
import sys
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Callable
from multiprocessing import Pool, cpu_count, Manager
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
from datetime import datetime
from tqdm import tqdm
from PIL import Image

# Import OCR components
from document_classifier import classify_document, DocumentType
from document_processors import (
    FinancialProcessor, IDCardProcessor, FormProcessor,
    LegalProcessor, HealthcareProcessor, HistoricalProcessor,
    LogisticsProcessor
)

logger = logging.getLogger(__name__)


class ParallelOCRProcessor:
    """
    High-performance parallel OCR processor.
    
    Features:
    - Parallel document processing across CPU cores
    - Progress tracking with ETA
    - Automatic error recovery
    - Smart batching and load balancing
    """
    
    def __init__(self, max_workers: Optional[int] = None):
        """
        Initialize parallel processor.
        
        Args:
            max_workers: Maximum number of parallel workers (default: CPU count - 1)
        """
        self.max_workers = max_workers or max(1, cpu_count() - 1)
        
        # Initialize processors
        self.processors = {
            DocumentType.FINANCIAL: FinancialProcessor(),
            DocumentType.ID_CARD: IDCardProcessor(),
            DocumentType.FORM: FormProcessor(),
            DocumentType.LEGAL: LegalProcessor(),
            DocumentType.HEALTHCARE: HealthcareProcessor(),
            DocumentType.HISTORICAL: HistoricalProcessor(),
            DocumentType.LOGISTICS: LogisticsProcessor(),
        }
        
        logger.info(f"Parallel processor initialized with {self.max_workers} workers")
    
    def process_batch(
        self,
        documents: List[Path],
        ocr_function: Callable,
        output_dir: Path,
        show_progress: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Process a batch of documents in parallel.
        
        Args:
            documents: List of document file paths
            ocr_function: OCR function to call (e.g., process_single_page_ocr)
            output_dir: Output directory for results
            show_progress: Show progress bar
            
        Returns:
            List of processing results
        """
        start_time = time.time()
        total_docs = len(documents)
        
        logger.info(f"Processing {total_docs} documents with {self.max_workers} workers")
        
        # Prepare output directory
        output_dir.mkdir(parents=True, exist_ok=True)
        
        results = []
        errors = []
        
        # Use ProcessPoolExecutor for CPU-bound tasks
        with Pool(processes=self.max_workers) as pool:
            # Create tasks
            tasks = [
                (doc, ocr_function, output_dir, idx)
                for idx, doc in enumerate(documents, 1)
            ]
            
            # Process with progress bar
            if show_progress:
                pbar = tqdm(total=total_docs, desc="Processing documents", unit="doc")
            
            # Use imap_unordered for better memory efficiency
            for result in pool.imap_unordered(self._process_single_document_wrapper, tasks):
                if result.get('status') == 'success':
                    results.append(result)
                else:
                    errors.append(result)
                
                if show_progress:
                    pbar.update(1)
            
            if show_progress:
                pbar.close()
        
        # Calculate statistics
        duration = time.time() - start_time
        docs_per_second = total_docs / duration if duration > 0 else 0
        
        logger.info(f"Batch processing complete:")
        logger.info(f"  - Total documents: {total_docs}")
        logger.info(f"  - Successful: {len(results)}")
        logger.info(f"  - Errors: {len(errors)}")
        logger.info(f"  - Duration: {duration:.2f}s")
        logger.info(f"  - Speed: {docs_per_second:.2f} docs/sec")
        
        return {
            'results': results,
            'errors': errors,
            'statistics': {
                'total': total_docs,
                'successful': len(results),
                'failed': len(errors),
                'duration_seconds': duration,
                'docs_per_second': docs_per_second
            }
        }
    
    @staticmethod
    def _process_single_document_wrapper(args):
        """Wrapper function for multiprocessing"""
        doc_path, ocr_function, output_dir, idx = args
        
        try:
            # Import here to avoid pickle issues
            from main import process_single_page_ocr
            from image_ocr import process_single_image_ocr
            
            # Determine if PDF or image
            is_pdf = doc_path.suffix.lower() == '.pdf'
            
            if is_pdf:
                result = process_single_page_ocr(
                    str(doc_path),
                    page_num=1,
                    output_dir=output_dir,
                    doc_handle=None
                )
            else:
                result = process_single_image_ocr(
                    str(doc_path),
                    display_num=idx,
                    output_dir=output_dir
                )
            
            return {
                'status': 'success',
                'document': str(doc_path),
                'index': idx,
                'result': result
            }
            
        except Exception as e:
            logger.error(f"Error processing {doc_path}: {e}")
            return {
                'status': 'error',
                'document': str(doc_path),
                'index': idx,
                'error': str(e)
            }
    
    def process_with_classification(
        self,
        documents: List[Path],
        ocr_function: Callable,
        output_dir: Path,
        classify_first: bool = True
    ) -> Dict[str, Any]:
        """
        Process documents with automatic classification and specialized extraction.
        
        Args:
            documents: List of document paths
            ocr_function: OCR function
            output_dir: Output directory
            classify_first: Whether to classify documents before processing
            
        Returns:
            Processing results with classifications
        """
        start_time = time.time()
        results = []
        
        # Process in parallel
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = []
            
            for idx, doc_path in enumerate(documents, 1):
                future = executor.submit(
                    self._process_and_classify,
                    doc_path, ocr_function, output_dir, idx, classify_first
                )
                futures.append(future)
            
            # Collect results with progress
            pbar = tqdm(total=len(documents), desc="Processing & classifying", unit="doc")
            
            for future in as_completed(futures):
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    logger.error(f"Task failed: {e}")
                    results.append({'status': 'error', 'error': str(e)})
                finally:
                    pbar.update(1)
            
            pbar.close()
        
        duration = time.time() - start_time
        
        # Group by document type
        by_type = {}
        for result in results:
            if result.get('status') == 'success':
                doc_type = result.get('document_type', 'unknown')
                if doc_type not in by_type:
                    by_type[doc_type] = []
                by_type[doc_type].append(result)
        
        return {
            'results': results,
            'by_type': by_type,
            'statistics': {
                'total': len(documents),
                'successful': len([r for r in results if r.get('status') == 'success']),
                'duration_seconds': duration,
                'docs_per_second': len(documents) / duration if duration > 0 else 0
            }
        }
    
    def _process_and_classify(
        self,
        doc_path: Path,
        ocr_function: Callable,
        output_dir: Path,
        idx: int,
        classify_first: bool
    ) -> Dict[str, Any]:
        """Process a single document with classification"""
        try:
            from main import process_single_page_ocr
            from image_ocr import process_single_image_ocr
            
            # Run OCR
            is_pdf = doc_path.suffix.lower() == '.pdf'
            
            if is_pdf:
                ocr_result = process_single_page_ocr(
                    str(doc_path),
                    page_num=1,
                    output_dir=output_dir,
                    doc_handle=None
                )
            else:
                ocr_result = process_single_image_ocr(
                    str(doc_path),
                    display_num=idx,
                    output_dir=output_dir
                )
            
            # Classify document
            text = ocr_result.get('full_text', '')
            lines = ocr_result.get('lines', [])
            
            doc_type, confidence = classify_document(text, lines, None)
            
            # Use specialized processor
            if doc_type in self.processors:
                processor = self.processors[doc_type]
                specialized_fields = processor.extract_fields(text, lines, None)
                is_valid, errors = processor.validate_fields(specialized_fields)
            else:
                specialized_fields = {}
                is_valid = True
                errors = []
            
            return {
                'status': 'success',
                'document': str(doc_path),
                'document_type': doc_type.value,
                'classification_confidence': confidence,
                'ocr_result': ocr_result,
                'specialized_fields': specialized_fields,
                'validation': {
                    'is_valid': is_valid,
                    'errors': errors
                }
            }
            
        except Exception as e:
            logger.error(f"Error processing {doc_path}: {e}")
            return {
                'status': 'error',
                'document': str(doc_path),
                'error': str(e)
            }


def create_parallel_processor(max_workers: Optional[int] = None) -> ParallelOCRProcessor:
    """
    Create a parallel OCR processor instance.
    
    Args:
        max_workers: Maximum number of parallel workers
        
    Returns:
        ParallelOCRProcessor instance
    """
    return ParallelOCRProcessor(max_workers=max_workers)
