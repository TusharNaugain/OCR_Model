#!/usr/bin/env python3
"""
Form Processor
==============

Specialized processor for structured forms:
- Tax documents
- Employee records
- Surveys and questionnaires
- Application forms
"""

import re
from typing import Dict, List, Any, Optional, Tuple
from PIL import Image
from document_processors.base_processor import BaseDocumentProcessor, DocumentType


class FormProcessor(BaseDocumentProcessor):
    """
    Process structured forms and extract field-value pairs.
    """
    
    def __init__(self):
        super().__init__(DocumentType.FORM)
    
    def get_expected_fields(self) -> List[str]:
        return [
            'form_type',
            'form_number',
            'enrollment_no',
            'batch_semester',
            'form_date',
            'applicant_name',
            'candidate_name',
            'academic_results',
            'field_value_pairs',  # Dynamic list of all form fields
            'checkboxes',  # List of checked items
            'signature_present',
        ]
    
    def extract_fields(self, text: str, lines: List[str], image: Optional[Image.Image] = None) -> Dict[str, Any]:
        """Extract form fields"""
        fields = {}
        
        # Detect form type
        fields['form_type'] = self._detect_form_type(text)
        
        # Extract basic info
        fields['form_number'] = self._extract_form_number(text)
        fields['enrollment_no'] = self._extract_enrollment_no(text)
        fields['batch_semester'] = self._extract_batch_semester(text)
        fields['form_date'] = self._extract_date(text)
        
        applicant_name = self._extract_applicant_name(text)
        fields['applicant_name'] = applicant_name
        fields['candidate_name'] = self._extract_candidate_name(text) or applicant_name
        
        # Extract academic results table if applicable
        if fields['form_type'] == 'academic_form':
            fields['academic_results'] = self._extract_academic_results(text, lines)
        
        # Extract all field-value pairs
        fields['field_value_pairs'] = self._extract_field_value_pairs(text, lines)
        
        # Detect checkboxes
        fields['checkboxes'] = self._extract_checkboxes(text, lines)
        
        # Check for signature
        fields['signature_present'] = self._detect_signature(text)
        
        # Clean all field values
        for key, value in fields.items():
            if isinstance(value, str) and key != 'form_type':
                fields[key] = self.clean_field_value(value)
        
        return fields
    
    def validate_fields(self, fields: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Validate extracted form fields"""
        errors = []
        
        # Check if we extracted at least some field-value pairs
        if not fields.get('field_value_pairs') or len(fields['field_value_pairs']) == 0:
            errors.append("No form fields extracted")
        
        # Check for applicant name (common in most forms)
        if not fields.get('applicant_name'):
            errors.append("Missing applicant/signatory name")
        
        is_valid = len(errors) == 0
        return (is_valid, errors)
    
    # ========== Helper Methods ==========
    
    def _detect_form_type(self, text: str) -> str:
        """Detect type of form"""
        text_lower = text.lower()
        
        if 'tax' in text_lower or 'w-2' in text_lower or 'w2' in text_lower or '1040' in text_lower:
            return 'tax_form'
        elif 'enrollment' in text_lower or 'roll no' in text_lower or 'marksheet' in text_lower or 'semester' in text_lower:
            return 'academic_form'
        elif 'employment' in text_lower or 'employee' in text_lower:
            return 'employment_form'
        elif 'survey' in text_lower or 'questionnaire' in text_lower:
            return 'survey'
        elif 'application' in text_lower:
            return 'application_form'
        else:
            return 'generic_form'
    
    def _extract_form_number(self, text: str) -> str:
        """Extract form number/ID"""
        patterns = [
            r'form\s*(?:no|number|#)[:\s]*([A-Z0-9-]+)',
            r'(?:^|\n)form\s+([A-Z0-9-]+)(?:\s|$)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
            if match:
                return match.group(1).strip()
        
        return ""
    
    def _extract_date(self, text: str) -> str:
        """Extract form date"""
        patterns = [
            r'date[:\s]*(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})',
            r'(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1).strip()
        
        return ""
    
    def _extract_applicant_name(self, text: str) -> str:
        """Extract applicant/signatory name"""
        patterns = [
            r'(?:name|applicant|employee|signatory)[:\s]*(.+?)(?:\n|$)',
            r'(?:full name)[:\s]*(.+?)(?:\n|$)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
            if match:
                name = match.group(1).strip()
                # Filter out invalid names
                if len(name) > 2 and len(name) < 100:
                    return name
        
        return ""
    
    def _extract_field_value_pairs(self, text: str, lines: List[str]) -> List[Dict[str, str]]:
        """
        Extract all field-value pairs from form.
        Forms typically have patterns like:
        - Field Name: Value
        - Field Name ___Value___
        - Field Name [ ] Value
        """
        pairs = []
        
        # Pattern 1: Field: Value
        for match in re.finditer(r'([A-Za-z][A-Za-z\s]{2,30})[:\s]+([^\n]{1,200})', text):
            field = match.group(1).strip()
            value = match.group(2).strip()
            
            # Filter out noise
            if len(field) > 2 and len(value) > 0 and not value.startswith('_'):
                pairs.append({
                    'field': field,
                    'value': value
                })
        
        # Pattern 2: Field ___Value___
        for match in re.finditer(r'([A-Za-z][A-Za-z\s]{2,30})[\s_]{3,}([^\s_][^\n]{0,200})', text):
            field = match.group(1).strip()
            value = match.group(2).strip()
            
            if len(field) > 2 and len(value) > 0:
                # Check if not already added
                if not any(p['field'] == field for p in pairs):
                    pairs.append({
                        'field': field,
                        'value': value
                    })
        
        return pairs
    
    def _extract_checkboxes(self, text: str, lines: List[str]) -> List[Dict[str, Any]]:
        """
        Extract checkbox states from form.
        Looks for patterns like:
        - [X] Option
        - ☑ Option
        - [✓] Option
        """
        checkboxes = []
        
        # Unicode checkbox patterns
        checked_patterns = ['☑', '☒', '✓', '✔', '[X]', '[x]', '[✓]']
        unchecked_patterns = ['☐', '[ ]']
        
        for line in lines:
            # Check for checked boxes
            for pattern in checked_patterns:
                if pattern in line:
                    label = line.split(pattern, 1)[1].strip() if pattern in line else ""
                    if label and len(label) < 200:
                        checkboxes.append({
                            'label': label,
                            'checked': True
                        })
            
            # Check for unchecked boxes
            for pattern in unchecked_patterns:
                if pattern in line:
                    label = line.split(pattern, 1)[1].strip() if pattern in line else ""
                    if label and len(label) < 200:
                        # Only add if not already added as checked
                        if not any(cb['label'] == label for cb in checkboxes):
                            checkboxes.append({
                                'label': label,
                                'checked': False
                            })
        
        return checkboxes
    
    def _detect_signature(self, text: str) -> bool:
        """Detect if form contains signature field"""
        signature_keywords = ['signature', 'signed', 'signatory', 'sign here']
        
        text_lower = text.lower()
        return any(kw in text_lower for kw in signature_keywords)

    def _extract_enrollment_no(self, text: str) -> str:
        """Extract enrollment number or roll number"""
        patterns = [
            r'(?:enrollment|roll)\s*(?:no|number|#)[:\s]*([A-Z0-9-]+)',
            r'enr\s*(?:no|number)[:\s]*([A-Z0-9-]+)',
        ]
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1).strip()
        return ""

    def _extract_batch_semester(self, text: str) -> str:
        """Extract batch or semester info"""
        patterns = [
            r'(?:batch|semester|sem)[:\s]*([A-Z0-9-\s]+?)(?:\n|$)',
            r'(?:20\d{2}-\d{2,4})', # Year ranges like 2021-25
        ]
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                try:
                    return match.group(1).strip()
                except IndexError:
                    return match.group(0).strip()
        return ""

    def _extract_candidate_name(self, text: str) -> str:
        """Extract candidate name with specialized patterns"""
        patterns = [
            r'(?:candidate|student)\s*name[:\s]*(.+?)(?:\n|$)',
            r'name\s*of\s*the\s*candidate[:\s]*(.+?)(?:\n|$)',
        ]
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1).strip()
        return ""

    def _extract_academic_results(self, text: str, lines: List[str]) -> List[Dict[str, str]]:
        """Extract academic subject and score tables"""
        results = []
        # Simple heuristic: look for lines with subject names and numbers
        # This is a placeholder for more robust table extraction
        subject_pattern = r'^\s*([A-Z][A-Za-z0-9\s]{2,40})\s+(\d{1,3})\s+(\d{1,3})'
        for line in lines:
            match = re.search(subject_pattern, line)
            if match:
                results.append({
                    'subject': match.group(1).strip(),
                    'marks_obtained': match.group(2),
                    'total_marks': match.group(3)
                })
        return results
