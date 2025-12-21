#!/usr/bin/env python3
"""
ID Card Processor
=================

Specialized processor for identification documents:
- Passports
- Driver's licenses
- National ID cards
- Government-issued IDs
"""

import re
from typing import Dict, List, Any, Optional, Tuple
from PIL import Image
from document_processors.base_processor import BaseDocumentProcessor, DocumentType


class IDCardProcessor(BaseDocumentProcessor):
    """
    Process ID cards and extract personal information.
    """
    
    def __init__(self):
        super().__init__(DocumentType.ID_CARD)
    
    def get_expected_fields(self) -> List[str]:
        return [
            'id_type',  # passport, drivers_license, national_id
            'id_number',
            'full_name',
            'first_name',
            'last_name',
            'date_of_birth',
            'nationality',
            'sex',
            'issue_date',
            'expiry_date',
            'issuing_authority',
            'address',
            'mrz_line1',  # Machine Readable Zone for passports
            'mrz_line2',
        ]
    
    def extract_fields(self, text: str, lines: List[str], image: Optional[Image.Image] = None) -> Dict[str, Any]:
        """Extract ID card fields"""
        fields = {}
        
        # Detect ID type
        fields['id_type'] = self._detect_id_type(text)
        
        # Extract MRZ if present (passport)
        mrz_data = self._extract_mrz(lines)
        if mrz_data:
            fields.update(mrz_data)
        
        # Extract standard fields
        if not fields.get('id_number'):
            fields['id_number'] = self._extract_id_number(text, fields['id_type'])
        
        if not fields.get('full_name'):
            fields['full_name'] = self._extract_name(text, lines)
        
        if not fields.get('date_of_birth'):
            fields['date_of_birth'] = self._extract_dob(text)
        
        fields['sex'] = self._extract_sex(text)
        fields['nationality'] = self._extract_nationality(text)
        fields['issue_date'] = self._extract_issue_date(text)
        fields['expiry_date'] = self._extract_expiry_date(text)
        fields['issuing_authority'] = self._extract_authority(text)
        fields['address'] = self._extract_address(text, lines)
        
        # Parse full name into first/last if not already done
        if fields['full_name'] and not (fields.get('first_name') and fields.get('last_name')):
            name_parts = fields['full_name'].split()
            if len(name_parts) >= 2:
                fields['first_name'] = name_parts[0]
                fields['last_name'] = ' '.join(name_parts[1:])
        
        # Clean all field values
        for key, value in fields.items():
            if isinstance(value, str):
                fields[key] = self.clean_field_value(value)
        
        return fields
    
    def validate_fields(self, fields: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Validate extracted ID fields"""
        errors = []
        
        # Check required fields
        if not fields.get('id_number'):
            errors.append("Missing ID number")
        
        if not fields.get('full_name'):
            errors.append("Missing name")
        
        if not fields.get('date_of_birth'):
            errors.append("Missing date of birth")
        elif not self.is_valid_date(str(fields['date_of_birth'])):
            errors.append(f"Invalid DOB format: {fields['date_of_birth']}")
        
        # Validate expiry date if present
        if fields.get('expiry_date') and not self.is_valid_date(str(fields['expiry_date'])):
            errors.append(f"Invalid expiry date format: {fields['expiry_date']}")
        
        is_valid = len(errors) == 0
        return (is_valid, errors)
    
    # ========== Helper Methods ==========
    
    def _detect_id_type(self, text: str) -> str:
        """Detect type of ID document"""
        text_lower = text.lower()
        
        if 'passport' in text_lower:
            return 'passport'
        elif 'driver' in text_lower or 'driving' in text_lower:
            return 'drivers_license'
        elif 'national id' in text_lower or 'identity card' in text_lower:
            return 'national_id'
        else:
            return 'id_card'
    
    def _extract_mrz(self, lines: List[str]) -> Optional[Dict[str, str]]:
        """
        Extract and parse Machine Readable Zone (MRZ) from passport.
        MRZ is 2 lines of 44 characters each for TD-3 passports.
        """
        # Look for MRZ pattern: lines with mostly uppercase letters, numbers, and '<' symbols
        mrz_lines = []
        
        for line in lines:
            line_clean = line.strip().replace(' ', '')
            # MRZ lines are typically 44 characters, all uppercase/numbers/<
            if len(line_clean) >= 40 and all(c.isupper() or c.isdigit() or c == '<' for c in line_clean):
                mrz_lines.append(line_clean)
        
        if len(mrz_lines) >= 2:
            # Parse MRZ
            mrz1 = mrz_lines[-2][:44]  # Take last 2 lines
            mrz2 = mrz_lines[-1][:44]
            
            try:
                # Parse MRZ Line 1: P<COUNTRYNAME<SURNAME<<FIRSTNAME
                doc_type = mrz1[0]  # P for passport
                country = mrz1[2:5].replace('<', '')
                
                # Extract name from line 1
                name_part = mrz1[5:].split('<<')
                surname = name_part[0].replace('<', ' ').strip()
                given_names = name_part[1].replace('<', ' ').strip() if len(name_part) > 1 else ''
                
                # Parse MRZ Line 2: PASSPORT_NO<CHECK<NATIONALITY<DOB<SEX<EXPIRY<PERSONAL
                passport_no = mrz2[:9].replace('<', '').strip()
                nationality = mrz2[10:13].replace('<', '')
                dob = mrz2[13:19]  # YYMMDD
                sex = mrz2[20]
                expiry = mrz2[21:27]  # YYMMDD
                
                # Format dates from YYMMDD to DD/MM/YYYY
                def format_mrz_date(yymmdd):
                    if len(yymmdd) == 6:
                        yy, mm, dd = yymmdd[:2], yymmdd[2:4], yymmdd[4:6]
                        # Assume 20xx for years
                        yyyy = f"20{yy}" if int(yy) < 50 else f"19{yy}"
                        return f"{dd}/{mm}/{yyyy}"
                    return yymmdd
                
                return {
                    'id_number': passport_no,
                    'full_name': f"{given_names} {surname}".strip(),
                    'first_name': given_names,
                    'last_name': surname,
                    'nationality': nationality,
                    'date_of_birth': format_mrz_date(dob),
                    'sex': sex,
                    'expiry_date': format_mrz_date(expiry),
                    'mrz_line1': mrz1,
                    'mrz_line2': mrz2,
                }
            except Exception as e:
                self.logger.warning(f"MRZ parsing failed: {e}")
        
        return None
    
    def _extract_id_number(self, text: str, id_type: str) -> str:
        """Extract ID/passport/license number"""
        patterns = []
        
        if id_type == 'passport':
            patterns = [
                r'passport\s*(?:no|number)[:\s]*([A-Z0-9]{6,12})',
                r'(?:^|\n)([A-Z]\d{7,8})(?:\s|$)',  # US passport pattern
            ]
        elif id_type == 'drivers_license':
            patterns = [
                r'(?:license|dl)\s*(?:no|number|#)[:\s]*([A-Z0-9-]{8,20})',
                r'(?:^|\n)([A-Z]\d{7,14})(?:\s|$)',
            ]
        else:
            patterns = [
                r'(?:id|identification)\s*(?:no|number)[:\s]*([A-Z0-9-]{6,20})',
                r'(?:^|\n)([A-Z0-9]{6,20})(?:\s|$)',
            ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
            if match:
                return match.group(1).strip()
        
        return ""
    
    def _extract_name(self, text: str, lines: List[str]) -> str:
        """Extract full name"""
        patterns = [
            r'(?:name|surname|full name)[:\s]*(.+?)(?:\n|$)',
            r'(?:^|\n)([A-Z][a-z]+\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)(?:\n|$)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.MULTILINE)
            if match:
                name = match.group(1).strip()
                # Filter out obvious non-names
                if len(name) > 3 and len(name) < 100 and not any(c.isdigit() for c in name):
                    return name
        
        return ""
    
    def _extract_dob(self, text: str) -> str:
        """Extract date of birth"""
        patterns = [
            r'(?:date of birth|dob|birth date)[:\s]*(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})',
            r'(?:date of birth|dob|birth date)[:\s]*(\d{1,2}\s+[A-Za-z]+\s+\d{4})',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1).strip()
        
        return ""
    
    def _extract_sex(self, text: str) -> str:
        """Extract sex/gender"""
        patterns = [
            r'(?:sex|gender)[:\s]*(M|F|Male|Female)',
            r'(?:^|\n)(M|F)(?:\s|$)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
            if match:
                sex = match.group(1).upper()
                return 'M' if sex in ['M', 'MALE'] else 'F'
        
        return ""
    
    def _extract_nationality(self, text: str) -> str:
        """Extract nationality"""
        patterns = [
            r'(?:nationality|country)[:\s]*([A-Z]{2,3}|[A-Z][a-z]+)',
            r'(?:citizen of|national of)[:\s]*([A-Z][a-z]+)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1).strip()
        
        return ""
    
    def _extract_issue_date(self, text: str) -> str:
        """Extract issue date"""
        patterns = [
            r'(?:date of issue|issue date|issued)[:\s]*(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})',
            r'(?:date of issue|issue date|issued)[:\s]*(\d{1,2}\s+[A-Za-z]+\s+\d{4})',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1).strip()
        
        return ""
    
    def _extract_expiry_date(self, text: str) -> str:
        """Extract expiry date"""
        patterns = [
            r'(?:expiry date|expires|exp|valid until)[:\s]*(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})',
            r'(?:expiry date|expires|exp|valid until)[:\s]*(\d{1,2}\s+[A-Za-z]+\s+\d{4})',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1).strip()
        
        return ""
    
    def _extract_authority(self, text: str) -> str:
        """Extract issuing authority"""
        patterns = [
            r'(?:issued by|issuing authority|authority)[:\s]*(.+?)(?:\n|$)',
            r'(?:government of|state of|department of)[:\s]*(.+?)(?:\n|$)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
            if match:
                return match.group(1).strip()
        
        return ""
    
    def _extract_address(self, text: str, lines: List[str]) -> str:
        """Extract address"""
        patterns = [
            r'(?:address|residence)[:\s]*(.+?)(?:\n\n|$)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
            if match:
                address = match.group(1).strip()
                # Clean up address (remove extra newlines)
                address = ' '.join(address.split())
                return address
        
        return ""
