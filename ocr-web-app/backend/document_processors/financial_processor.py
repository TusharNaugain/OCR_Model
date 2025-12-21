#!/usr/bin/env python3
"""
Financial Document Processor
=============================

Specialized processor for financial documents:
- Invoices
- Receipts  
- Checks
- Bank statements
- Loan applications
"""

import re
from typing import Dict, List, Any, Optional, Tuple
from PIL import Image
from document_processors.base_processor import BaseDocumentProcessor, DocumentType


class FinancialProcessor(BaseDocumentProcessor):
    """
    Process financial documents and extract key fields.
    """
    
    def __init__(self):
        super().__init__(DocumentType.FINANCIAL)
    
    def get_expected_fields(self) -> List[str]:
        return [
            'document_type',  # invoice, receipt, check, statement
            'document_number',
            'date',
            'vendor_name',
            'vendor_address',
            'customer_name',
            'customer_address',
            'line_items',
            'subtotal',
            'tax',
            'total_amount',
            'payment_method',
            'account_number',
        ]
    
    def extract_fields(self, text: str, lines: List[str], image: Optional[Image.Image] = None) -> Dict[str, Any]:
        """Extract financial document fields"""
        fields = {}
        
        # Determine sub-type
        fields['document_type'] = self._detect_financial_subtype(text)
        
        # Extract common fields
        fields['document_number'] = self._extract_document_number(text, lines)
        fields['date'] = self._extract_date(text, lines)
        
        # Extract parties
        fields['vendor_name'] = self._extract_vendor(text, lines)
        fields['customer_name'] = self._extract_customer(text, lines)
        
        # Extract amounts
        amounts = self._extract_amounts(text, lines)
        fields.update(amounts)
        
        # Extract line items (for invoices/receipts)
        fields['line_items'] = self._extract_line_items(text, lines)
        
        # Extract payment info
        fields['payment_method'] = self._extract_payment_method(text)
        fields['account_number'] = self._extract_account_number(text)
        
        # Clean all field values
        for key, value in fields.items():
            if isinstance(value, str):
                fields[key] = self.clean_field_value(value)
        
        return fields
    
    def validate_fields(self, fields: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Validate extracted financial fields"""
        errors = []
        
        # Check required fields
        if not fields.get('total_amount'):
            errors.append("Missing total amount")
        elif not self.is_valid_amount(str(fields['total_amount'])):
            errors.append(f"Invalid total amount: {fields['total_amount']}")
        
        if not fields.get('date'):
            errors.append("Missing date")
        elif not self.is_valid_date(str(fields['date'])):
            errors.append(f"Invalid date format: {fields['date']}")
        
        # Validate amounts if present
        if fields.get('subtotal') and fields.get('tax') and fields.get('total_amount'):
            try:
                subtotal = self._parse_amount(fields['subtotal'])
                tax = self._parse_amount(fields['tax'])
                total = self._parse_amount(fields['total_amount'])
                
                # Check if subtotal + tax ≈ total (allow 1% error for rounding)
                calculated_total = subtotal + tax
                if abs(calculated_total - total) / total > 0.01:
                    errors.append(f"Amount mismatch: {subtotal} + {tax} ≠ {total}")
            except:
                pass
        
        is_valid = len(errors) == 0
        return (is_valid, errors)
    
    # ========== Helper Methods ==========
    
    def _detect_financial_subtype(self, text: str) -> str:
        """Detect specific financial document type"""
        text_lower = text.lower()
        
        if 'invoice' in text_lower:
            return 'invoice'
        elif 'receipt' in text_lower:
            return 'receipt'
        elif 'check' in text_lower or 'cheque' in text_lower:
            return 'check'
        elif 'statement' in text_lower:
            return 'bank_statement'
        elif 'loan' in text_lower or 'application' in text_lower:
            return 'loan_application'
        else:
            return 'financial_document'
    
    def _extract_document_number(self, text: str, lines: List[str]) -> str:
        """Extract invoice/receipt/check number"""
        patterns = [
            r'(?:invoice|receipt|check)\s*(?:no|#|number)[:\s]*([A-Z0-9-]+)',
            r'(?:no|#)\s*([A-Z0-9-]{5,})',
            r'document\s*(?:no|number)[:\s]*([A-Z0-9-]+)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1).strip()
        
        return ""
    
    def _extract_date(self, text: str, lines: List[str]) -> str:
        """Extract document date"""
        # Common date patterns
        patterns = [
            r'date[:\s]*(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})',
            r'(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})',
            r'(\d{4}-\d{2}-\d{2})',
            r'(\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{4})',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1).strip()
        
        return ""
    
    def _extract_vendor(self, text: str, lines: List[str]) -> str:
        """Extract vendor/seller name"""
        # Usually the first few lines, often in larger text
        # Look for patterns
        patterns = [
            r'(?:from|vendor|seller)[:\s]*(.+?)(?:\n|$)',
            r'bill\s+from[:\s]*(.+?)(?:\n|$)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1).strip()
        
        # Fallback: first non-empty line if it looks like a business name
        if lines:
            first_line = lines[0].strip()
            if len(first_line) > 5 and len(first_line) < 100:
                return first_line
        
        return ""
    
    def _extract_customer(self, text: str, lines: List[str]) -> str:
        """Extract customer/buyer name"""
        patterns = [
            r'(?:to|customer|buyer|bill to)[:\s]*(.+?)(?:\n|$)',
            r'sold\s+to[:\s]*(.+?)(?:\n|$)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1).strip()
        
        return ""
    
    def _extract_amounts(self, text: str, lines: List[str]) -> Dict[str, str]:
        """Extract subtotal, tax, and total amounts"""
        amounts = {
            'subtotal': '',
            'tax': '',
            'total_amount': '',
        }
        
        # Patterns for amounts
        currency_pattern = r'[$₹€£]?\s*[\d,]+\.?\d*'
        
        # Subtotal
        subtotal_match = re.search(
            rf'(?:sub\s*total|subtotal)[:\s]*({currency_pattern})',
            text, re.IGNORECASE
        )
        if subtotal_match:
            amounts['subtotal'] = subtotal_match.group(1).strip()
        
        # Tax
        tax_match = re.search(
            rf'(?:tax|vat|gst)[:\s]*({currency_pattern})',
            text, re.IGNORECASE
        )
        if tax_match:
            amounts['tax'] = tax_match.group(1).strip()
        
        # Total (most important)
        total_patterns = [
            rf'(?:total|grand\s*total|amount\s*due)[:\s]*({currency_pattern})',
            rf'(?:^|\n)\s*total[:\s]*({currency_pattern})',
        ]
        
        for pattern in total_patterns:
            total_match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
            if total_match:
                amounts['total_amount'] = total_match.group(1).strip()
                break
        
        # If no total found, look for largest amount
        if not amounts['total_amount']:
            all_amounts = re.findall(currency_pattern, text)
            if all_amounts:
                # Parse and find maximum
                parsed = []
                for amt in all_amounts:
                    try:
                        parsed.append((amt, self._parse_amount(amt)))
                    except:
                        pass
                if parsed:
                    amounts['total_amount'] = max(parsed, key=lambda x: x[1])[0]
        
        return amounts
    
    def _parse_amount(self, amount_str: str) -> float:
        """Parse amount string to float"""
        # Remove currency symbols and whitespace
        cleaned = re.sub(r'[₹$€£\s]', '', amount_str)
        # Remove comma thousand separators
        cleaned = cleaned.replace(',', '')
        return float(cleaned)
    
    def _extract_line_items(self, text: str, lines: List[str]) -> List[Dict[str, str]]:
        """Extract line items from invoice/receipt"""
        line_items = []
        
        # Look for table-like structures
        # This is a simplified version - could be enhanced with table detection
        in_items_section = False
        
        for line in lines:
            line_clean = line.strip()
            
            # Check if this looks like a line item
            # Usually has: description + quantity + price
            if re.search(r'.+\s+\d+\s+[$₹€£]?\s*[\d,]+\.?\d*', line_clean):
                # Try to parse
                parts = line_clean.split()
                if len(parts) >= 3:
                    # Last part is likely price
                    price = parts[-1]
                    # Second to last might be quantity
                    qty = parts[-2] if len(parts) > 2 and parts[-2].isdigit() else '1'
                    # Rest is description
                    desc = ' '.join(parts[:-2]) if len(parts) > 2 else parts[0]
                    
                    line_items.append({
                        'description': desc,
                        'quantity': qty,
                        'price': price
                    })
        
        return line_items
    
    def _extract_payment_method(self, text: str) -> str:
        """Extract payment method"""
        methods = ['cash', 'credit card', 'debit card', 'check', 'wire transfer', 'paypal', 'online']
        
        text_lower = text.lower()
        for method in methods:
            if method in text_lower:
                return method
        
        return ""
    
    def _extract_account_number(self, text: str) -> str:
        """Extract account/card number (masked)"""
        patterns = [
            r'account[:\s]*([X*\d][\dX* ]{8,})',
            r'card[:\s]*([X*\d][\dX* ]{12,16})',
            r'(\*{4}\s*\d{4})',  # Last 4 digits pattern
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1).strip()
        
        return ""
