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
        # Improved Regex to capture Party Names
        # Looks for "Between [Name] ... AND [Name]" structure common in agreements
        
        # 1. Try to find "Between... And" block
        # Capture up to 'hereinafter' or 'Whereas' to stop correctly
        # AND must be preceded by newline to avoid matching "representatives and assignees"
        between_match = re.search(r'Between\s+([\s\S]+?)(?:^|\n)\s*(?:AND|&)\s+([\s\S]+?)(?:hereinafter|Whereas)', text, re.I)
        if between_match:
            p1_raw = between_match.group(1)
            p2_raw = between_match.group(2)
            
            p1 = p1_raw.split('hereinafter')[0].strip()
            p2 = p2_raw.split('hereinafter')[0].strip()
            
            parties.append(self._clean_party_name(p1))
            parties.append(self._clean_party_name(p2))
            return parties

        # Fallback patterns
        # Added comma, parens, hyphens to char set
        parts_re = r'[\w\s\./,\-\(\)]+' 
        
        patterns = [
            fr'between\s+({parts_re})(?:\s+and|\s*,)',
            fr'(?:^|\n)\s*(?:AND|&)\s+({parts_re})\s+(?:hereinafter\s+)?(?:called|known|referred)(?:\s+as|\s+the)?',
            fr'tenant\s*:\s*({parts_re})',
            fr'landlord\s*:\s*({parts_re})'
        ]
        
        for p in patterns:
            matches = re.finditer(p, text, re.I)
            for m in matches:
                name = m.group(1).strip()
                if len(name) > 3 and 'hereinafter' not in name.lower():
                     parties.append(self._clean_party_name(name))
                     
        return list(dict.fromkeys(parties))[:2] # Unique & limit to 2

    def _clean_party_name(self, text: str) -> str:
        # Helper to truncate party name if it accidentally captured the whole address
        # Stop at "R/O" or "S/O" if meaningful name exists before it
        
        # Split by common delimiters
        common_delimiters = [r'\sR/O\s', r'\sS/O\s', r'\sD/O\s', r'\sW/O\s', r',', r'\n']
        
        shortest = text
        for d in common_delimiters:
            parts = re.split(d, shortest, flags=re.IGNORECASE)
            if len(parts) > 1 and len(parts[0]) > 3:
                shortest = parts[0]
                
        return shortest.strip().strip('.').strip()

    def _extract_court(self, text: str) -> str:
        match = re.search(r'(?:court of|in the)[:\s]*(.+?)(?:\n|$)', text, re.I)
        return match.group(1).strip() if match else ""
    
    def _detect_signatures(self, text: str) -> bool:
        return bool(re.search(r'signature|signed|seal', text, re.I))

    def _extract_rent_details(self, text: str) -> Dict[str, str]:
        """Extract rent amount and security deposit"""
        details = {}
        
        # Monthly Rent
        rent_match = re.search(r'monthly\s+rent\s+of\s+(?:Rs\.?|INR)\s*([\d,]+)', text, re.I)
        if rent_match:
            details['monthly_rent'] = rent_match.group(1)
            
        # Security Deposit
        sec_match = re.search(r'(?:Rs\.?|INR)\s*([\d,]+|NIL)\s*.*received\s+as\s+Security', text, re.I | re.DOTALL)
        if sec_match:
            details['security_deposit'] = sec_match.group(1)
            
        return details

    def _extract_agreement_period(self, text: str) -> Dict[str, str]:
        """Extract agreement dates and duration"""
        dates = {}
        
        # Commencement Date
        start_match = re.search(r'commenced\s+from\s+(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})', text, re.I)
        if start_match:
            dates['start_date'] = start_match.group(1)
            
        # Duration
        duration_match = re.search(r'period\s+of\s+(\d+\s+months?)', text, re.I)
        if duration_match:
            dates['duration'] = duration_match.group(1)
            
        return dates

    def extract_fields(self, text: str, lines: List[str], image: Optional[Image.Image] = None) -> Dict[str, Any]:
        base_fields = {
            'document_type': self._detect_legal_type(text),
            'case_number': self._extract_case_number(text),
            'date': self._extract_date(text),
            'court_name': self._extract_court(text),
            'signatures': self._detect_signatures(text),
        }
        
        # Add Rent Agreement specific fields
        if base_fields['document_type'] == 'rent_agreement':
            base_fields.update(self._extract_rent_details(text))
            base_fields.update(self._extract_agreement_period(text))
            
            # Parties extraction optimized for Rent Agreement
            parties = self._extract_parties(text)
            if len(parties) >= 2:
                base_fields['first_party'] = parties[0]
                base_fields['second_party'] = parties[1]
            elif len(parties) == 1:
                base_fields['first_party'] = parties[0]
        else:
             base_fields['parties'] = self._extract_parties(text)

        return {k: self.clean_field_value(v) if isinstance(v, str) else v for k, v in base_fields.items()}

    def _detect_legal_type(self, text: str) -> str:
        t = text.lower()
        if 'rent agreement' in t:
            return 'rent_agreement'
        elif 'contract' in t or 'agreement' in t:
            return 'contract'
        elif 'deed' in t:
            return 'deed'
        elif 'court' in t or 'case' in t:
            return 'court_document'
        return 'legal_document'

