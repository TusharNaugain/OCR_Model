#!/usr/bin/env python3
"""Healthcare Document Processor"""
import re
from typing import Dict, List, Any, Optional, Tuple
from PIL import Image
from document_processors.base_processor import BaseDocumentProcessor, DocumentType

class HealthcareProcessor(BaseDocumentProcessor):
    def __init__(self):
        super().__init__(DocumentType.HEALTHCARE)
    
    def get_expected_fields(self) -> List[str]:
        return ['document_type', 'patient_name', 'patient_id', 'dob', 'medications', 'diagnosis', 'physician']
    
    def extract_fields(self, text: str, lines: List[str], image: Optional[Image.Image] = None) -> Dict[str, Any]:
        fields = {
            'document_type': self._detect_healthcare_type(text),
            'patient_name': self._extract_patient_name(text),
            'patient_id': self._extract_patient_id(text),
            'dob': self._extract_dob(text),
            'medications': self._extract_medications(text),
            'diagnosis': self._extract_diagnosis(text),
            'physician': self._extract_physician(text),
        }
        return {k: self.clean_field_value(v) if isinstance(v, str) else v for k, v in fields.items()}
    
    def validate_fields(self, fields: Dict[str, Any]) -> Tuple[bool, List[str]]:
        errors = []
        if not fields.get('patient_name'):
            errors.append("Missing patient name")
        return (len(errors) == 0, errors)
    
    def _detect_healthcare_type(self, text: str) -> str:
        t = text.lower()
        if 'prescription' in t:
            return 'prescription'
        elif 'claim' in t or 'insurance' in t:
            return 'insurance_claim'
        elif 'test result' in t or 'lab' in t:
            return 'test_results'
        return 'patient_record'
    
    def _extract_patient_name(self, text: str) -> str:
        match = re.search(r'patient(?:\s+name)?[:\s]*(.+?)(?:\n|$)', text, re.I)
        return match.group(1).strip() if match else ""
    
    def _extract_patient_id(self, text: str) -> str:
        match = re.search(r'patient\s*(?:id|number)[:\s]*([A-Z0-9-]+)', text, re.I)
        return match.group(1).strip() if match else ""
    
    def _extract_dob(self, text: str) -> str:
        match = re.search(r'(?:dob|date of birth)[:\s]*(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})', text, re.I)
        return match.group(1).strip() if match else ""
    
    def _extract_medications(self, text: str) -> List[str]:
        medications = []
        for match in re.finditer(r'(?:medication|drug|prescription)[:\s]*(.+?)(?:\n|$)', text, re.I):
            medications.append(match.group(1).strip())
        return medications
    
    def _extract_diagnosis(self, text: str) -> str:
        match = re.search(r'diagnosis[:\s]*(.+?)(?:\n|$)', text, re.I)
        return match.group(1).strip() if match else ""
    
    def _extract_physician(self, text: str) -> str:
        match = re.search(r'(?:doctor|physician|dr\.?)[:\s]*(.+?)(?:\n|$)', text, re.I)
        return match.group(1).strip() if match else ""
