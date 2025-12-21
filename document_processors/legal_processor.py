#!/usr/bin/env python3
"""Legal Document Processor"""
import re
from typing import Dict, List, Any, Optional, Tuple
from PIL import Image
from document_processors.base_processor import BaseDocumentProcessor, DocumentType

class LegalProcessor(BaseDocumentProcessor):
    def __init__(self):
        super().__init__(DocumentType.LEGAL)
    
    def get_expected_fields(self) -> List[str]:
        return ['document_type', 'case_number', 'date', 'parties', 'court_name', 'signatures']
    
    def extract_fields(self, text: str, lines: List[str], image: Optional[Image.Image] = None) -> Dict[str, Any]:
        fields = {
            'document_type': self._detect_legal_type(text),
            'case_number': self._extract_case_number(text),
            'date': self._extract_date(text),
            'parties': self._extract_parties(text),
            'court_name': self._extract_court(text),
            'signatures': self._detect_signatures(text),
        }
        return {k: self.clean_field_value(v) if isinstance(v, str) else v for k, v in fields.items()}
    
    def validate_fields(self, fields: Dict[str, Any]) -> Tuple[bool, List[str]]:
        errors = []
        if not fields.get('document_type'):
            errors.append("Missing document type")
        return (len(errors) == 0, errors)
    
    def _detect_legal_type(self, text: str) -> str:
        t = text.lower()
        if 'contract' in t or 'agreement' in t:
            return 'contract'
        elif 'deed' in t:
            return 'deed'
        elif 'court' in t or 'case' in t:
            return 'court_document'
        return 'legal_document'
    
    def _extract_case_number(self, text: str) -> str:
        match = re.search(r'case\s*(?:no|number)[:\s]*([A-Z0-9-]+)', text, re.I)
        return match.group(1).strip() if match else ""
    
    def _extract_date(self, text: str) -> str:
        match = re.search(r'date[:\s]*(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})', text, re.I)
        return match.group(1).strip() if match else ""
    
    def _extract_parties(self, text: str) -> List[str]:
        parties = []
        for pattern in [r'(?:plaintiff|defendant|party)[:\s]*(.+?)(?:\n|$)', r'(?:between|among)[:\s]*(.+?)(?:and|&)']:
            for match in re.finditer(pattern, text, re.I):
                parties.append(match.group(1).strip())
        return parties
    
    def _extract_court(self, text: str) -> str:
        match = re.search(r'(?:court of|in the)[:\s]*(.+?)(?:\n|$)', text, re.I)
        return match.group(1).strip() if match else ""
    
    def _detect_signatures(self, text: str) -> bool:
        return bool(re.search(r'signature|signed|seal', text, re.I))
