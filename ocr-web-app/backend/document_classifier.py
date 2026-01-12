#!/usr/bin/env python3
"""
Document Classifier
===================

AI-powered document type classification using Gemini Vision.
Falls back to heuristic detection if Gemini is unavailable.
"""

import os
import re
import logging
from typing import Dict, Tuple, Optional, List
from PIL import Image
# import google.generativeai as genai (Lazy loaded)

from document_processors.base_processor import DocumentType

logger = logging.getLogger(__name__)

# Gemini configuration
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
GEMINI_MODEL = os.getenv('GEMINI_MODEL', 'gemini-2.0-flash-exp')

# Classification cache to avoid re-processing
_classification_cache = {}


class DocumentClassifier:
    """
    Intelligent document classifier using AI and heuristics.
    """
    
    def __init__(self, use_gemini: bool = True):
        self.use_gemini = use_gemini and GEMINI_API_KEY is not None
        self.model = None
        
        if self.use_gemini:
            try:
                import google.generativeai as genai
                genai.configure(api_key=GEMINI_API_KEY)
                self.model = genai.GenerativeModel(GEMINI_MODEL)
                logger.info(f"Gemini classifier initialized with model: {GEMINI_MODEL}")
            except Exception as e:
                logger.warning(f"Failed to initialize Gemini: {e}. Falling back to heuristic classification.")
                self.use_gemini = False
    
    def classify(self, text: str, lines: List[str], image: Optional[Image.Image] = None) -> Tuple[DocumentType, float]:
        """
        Classify document type from OCR text and optional image.
        
        Args:
            text: Full OCR text
            lines: List of text lines
            image: Optional PIL Image for visual classification
            
        Returns:
            Tuple of (DocumentType, confidence_score)
        """
        # Check cache first
        text_hash = hash(text[:500])  # Hash first 500 chars for cache key
        if text_hash in _classification_cache:
            logger.debug("Using cached classification")
            return _classification_cache[text_hash]
        
        # Try Gemini classification if available and image provided
        if self.use_gemini and image and self.model:
            result = self._classify_with_gemini(text, image)
            if result:
                _classification_cache[text_hash] = result
                return result
        
        # Fall back to heuristic classification
        result = self._classify_heuristic(text, lines)
        _classification_cache[text_hash] = result
        return result
    
    def _classify_with_gemini(self, text: str, image: Image.Image) -> Optional[Tuple[DocumentType, float]]:
        """
        Use Gemini Vision to classify document type.
        
        Args:
            text: OCR text
            image: PIL Image
            
        Returns:
            Tuple of (DocumentType, confidence) or None if classification fails
        """
        try:
            prompt = """
            Analyze this document image and classify it into one of these categories:
            
            1. FINANCIAL - invoices, receipts, checks, bank statements, loan applications
            2. LEGAL - contracts, deeds, court records, legal filings
            3. ID_CARD - passports, driver's licenses, identification cards
            4. HEALTHCARE - patient records, insurance claims, prescriptions, test results
            5. HISTORICAL - old books, historical newspapers, handwritten notes, archival documents
            6. FORM - tax documents, surveys, employee records, application forms
            7. LOGISTICS - shipping labels, purchase orders, delivery notes, bills of lading
            
            Consider:
            - Document layout and structure
            - Headers and logos
            - Field types and labels
            - Visual formatting
            
            Respond with ONLY a JSON object:
            {
                "document_type": "FINANCIAL",
                "confidence": 0.95,
                "reasoning": "Contains invoice number, line items, and total amount"
            }
            """
            
            response = self.model.generate_content([prompt, image])
            response_text = response.text.strip()
            
            # Extract JSON from response
            import json
            if "```json" in response_text:
                json_str = response_text.split("```json")[1].split("```")[0]
            elif "```" in response_text:
                json_str = response_text.split("```")[1].split("```")[0]
            else:
                json_str = response_text
            
            result = json.loads(json_str)
            
            # Parse document type
            doc_type_str = result.get('document_type', 'UNKNOWN').upper()
            try:
                doc_type = DocumentType[doc_type_str]
            except KeyError:
                logger.warning(f"Unknown document type from Gemini: {doc_type_str}")
                return None
            
            confidence = float(result.get('confidence', 0.5))
            
            logger.info(f"Gemini classification: {doc_type.value} (confidence: {confidence:.2f})")
            logger.debug(f"Reasoning: {result.get('reasoning', 'N/A')}")
            
            return (doc_type, confidence)
            
        except Exception as e:
            logger.warning(f"Gemini classification failed: {e}")
            return None
    
    def _classify_heuristic(self, text: str, lines: List[str]) -> Tuple[DocumentType, float]:
        """
        Heuristic-based classification using text patterns.
        
        Args:
            text: Full OCR text
            lines: List of text lines
            
        Returns:
            Tuple of (DocumentType, confidence)
        """
        text_lower = text.lower()
        text_upper = text.upper()
        
        # Financial document patterns
        financial_keywords = ['invoice', 'receipt', 'total amount', 'subtotal', 'tax', 'payment', 
                            'balance', 'account number', 'bank statement', 'transaction', 'check']
        financial_score = sum(1 for kw in financial_keywords if kw in text_lower)
        
        # Legal document patterns
        legal_keywords = ['contract', 'agreement', 'whereas', 'party', 'defendant', 'plaintiff',
                         'court', 'case no', 'deed', 'witness', 'notary', 'legal']
        legal_score = sum(1 for kw in legal_keywords if kw in text_lower)
        
        # ID card patterns
        id_keywords = ['passport', 'license', 'identification', 'date of birth', 'dob', 
                      'nationality', 'issued by', 'expires', 'id number', 'driver']
        id_score = sum(1 for kw in id_keywords if kw in text_lower)
        
        # Check for MRZ (Machine Readable Zone) in passports
        if any(re.match(r'^[A-Z0-9<]{44}$', line) for line in lines):
            id_score += 5
        
        # Healthcare patterns
        healthcare_keywords = ['patient', 'doctor', 'physician', 'diagnosis', 'prescription',
                              'medication', 'dosage', 'insurance', 'claim', 'medical']
        healthcare_score = sum(1 for kw in healthcare_keywords if kw in text_lower)
        
        # Historical document patterns
        historical_indicators = [
            len([c for c in text if c.isupper()]) / (len(text) + 1) < 0.1,  # Mostly lowercase
            any(year < 2000 for year in self._extract_years(text)),  # Old dates
            'archive' in text_lower or 'historical' in text_lower
        ]
        historical_score = sum(historical_indicators) * 2
        
        # Form patterns
        form_keywords = ['form', 'application', 'survey', 'questionnaire', '☐', '☑', 
                        'checkbox', 'yes/no', 'signature']
        form_score = sum(1 for kw in form_keywords if kw in text_lower)
        
        # Check for form-like structure (multiple fields with labels)
        if len(re.findall(r'[\w\s]+:\s*_+', text)) > 5:
            form_score += 3
        
        # Logistics patterns
        logistics_keywords = ['shipping', 'delivery', 'tracking', 'purchase order', 'po number',
                             'shipment', 'carrier', 'freight', 'warehouse', 'inventory']
        logistics_score = sum(1 for kw in logistics_keywords if kw in text_lower)
        
        # Calculate scores
        scores = {
            DocumentType.FINANCIAL: financial_score,
            DocumentType.LEGAL: legal_score,
            DocumentType.ID_CARD: id_score,
            DocumentType.HEALTHCARE: healthcare_score,
            DocumentType.HISTORICAL: historical_score,
            DocumentType.FORM: form_score,
            DocumentType.LOGISTICS: logistics_score,
        }
        
        # Get highest score
        if max(scores.values()) == 0:
            return (DocumentType.UNKNOWN, 0.0)
        
        doc_type = max(scores, key=scores.get)
        max_score = scores[doc_type]
        total_score = sum(scores.values())
        
        # Calculate confidence
        confidence = min(max_score / (total_score + 1), 1.0)
        
        logger.info(f"Heuristic classification: {doc_type.value} (confidence: {confidence:.2f})")
        logger.debug(f"Scores: {scores}")
        
        return (doc_type, round(confidence, 2))
    
    def _extract_years(self, text: str) -> list:
        """Extract year values from text"""
        years = []
        for match in re.finditer(r'\b(19\d{2}|20\d{2})\b', text):
            try:
                years.append(int(match.group(1)))
            except ValueError:
                pass
        return years


# Global classifier instance
_classifier = None


def get_classifier() -> DocumentClassifier:
    """Get or create global DocumentClassifier instance"""
    global _classifier
    if _classifier is None:
        _classifier = DocumentClassifier()
    return _classifier


def classify_document(text: str, lines: List[str], image: Optional[Image.Image] = None) -> Tuple[DocumentType, float]:
    """
    Convenience function to classify a document.
    
    Args:
        text: Full OCR text
        lines: List of text lines
        image: Optional PIL Image
        
    Returns:
        Tuple of (DocumentType, confidence)
    """
    classifier = get_classifier()
    return classifier.classify(text, lines, image)
