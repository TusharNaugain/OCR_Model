#!/usr/bin/env python3
"""Historical Document Processor"""
import re
from typing import Dict, List, Any, Optional, Tuple
from PIL import Image
from document_processors.base_processor import BaseDocumentProcessor, DocumentType

class HistoricalProcessor(BaseDocumentProcessor):
    def __init__(self):
        super().__init__(DocumentType.HISTORICAL)
    
    def get_expected_fields(self) -> List[str]:
        return ['document_type', 'title', 'date', 'author', 'archive_number', 'text_content']
    
    def extract_fields(self, text: str, lines: List[str], image: Optional[Image.Image] = None) -> Dict[str, Any]:
        fields = {
            'document_type': self._detect_historical_type(text),
            'title': self._extract_title(lines),
            'date': self._extract_historical_date(text),
            'author': self._extract_author(text),
            'archive_number': self._extract_archive_number(text),
            'text_content': text[:500],  # Store first 500 chars
        }
        return {k: self.clean_field_value(v) if isinstance(v, str) else v for k, v in fields.items()}
    
    def validate_fields(self, fields: Dict[str, Any]) -> Tuple[bool, List[str]]:
        errors = []
        if not fields.get('text_content'):
            errors.append("No text content extracted")
        return (len(errors) == 0, errors)
    
    def _detect_historical_type(self, text: str) -> str:
        t = text.lower()
        if 'newspaper' in t:
            return 'newspaper'
        elif 'book' in t or 'chapter' in t:
            return 'book'
        elif 'letter' in t or 'correspondence' in t:
            return 'letter'
        return 'historical_document'
    
    def _extract_title(self, lines: List[str]) -> str:
        # Title is often the first non-empty line
        for line in lines[:10]:
            if len(line.strip()) > 5:
                return line.strip()
        return ""
    
    def _extract_historical_date(self, text: str) -> str:
        # Look for old date formats
        patterns = [
            r'(\d{1,2}\s+(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4})',
            r'(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})',
        ]
        for pattern in patterns:
            match = re.search(pattern, text, re.I)
            if match:
                return match.group(1)
        return ""
    
    def _extract_author(self, text: str) -> str:
        match = re.search(r'(?:by|author|written by)[:\s]*(.+?)(?:\n|$)', text, re.I)
        return match.group(1).strip() if match else ""
    
    def _extract_archive_number(self, text: str) -> str:
        match = re.search(r'(?:archive|catalog|reference)\s*(?:no|number)[:\s]*([A-Z0-9-]+)', text, re.I)
        return match.group(1).strip() if match else ""
