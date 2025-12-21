#!/usr/bin/env python3
"""Logistics Document Processor"""
import re
from typing import Dict, List, Any, Optional, Tuple
from PIL import Image
from document_processors.base_processor import BaseDocumentProcessor, DocumentType

class LogisticsProcessor(BaseDocumentProcessor):
    def __init__(self):
        super().__init__(DocumentType.LOGISTICS)
    
    def get_expected_fields(self) -> List[str]:
        return ['document_type', 'tracking_number', 'po_number', 'shipment_date', 'origin', 'destination', 'carrier']
    
    def extract_fields(self, text: str, lines: List[str], image: Optional[Image.Image] = None) -> Dict[str, Any]:
        fields = {
            'document_type': self._detect_logistics_type(text),
            'tracking_number': self._extract_tracking(text),
            'po_number': self._extract_po_number(text),
            'shipment_date': self._extract_date(text),
            'origin': self._extract_origin(text),
            'destination': self._extract_destination(text),
            'carrier': self._extract_carrier(text),
        }
        return {k: self.clean_field_value(v) if isinstance(v, str) else v for k, v in fields.items()}
    
    def validate_fields(self, fields: Dict[str, Any]) -> Tuple[bool, List[str]]:
        errors = []
        if not fields.get('tracking_number') and not fields.get('po_number'):
            errors.append("Missing tracking or PO number")
        return (len(errors) == 0, errors)
    
    def _detect_logistics_type(self, text: str) -> str:
        t = text.lower()
        if 'purchase order' in t or 'po' in t:
            return 'purchase_order'
        elif 'shipping label' in t or 'shipment' in t:
            return 'shipping_label'
        elif 'delivery' in t:
            return 'delivery_note'
        return 'logistics_document'
    
    def _extract_tracking(self, text: str) -> str:
        match = re.search(r'tracking\s*(?:no|number|#)[:\s]*([A-Z0-9]{8,30})', text, re.I)
        return match.group(1).strip() if match else ""
    
    def _extract_po_number(self, text: str) -> str:
        match = re.search(r'(?:po|purchase order)\s*(?:no|number|#)[:\s]*([A-Z0-9-]+)', text, re.I)
        return match.group(1).strip() if match else ""
    
    def _extract_date(self, text: str) -> str:
        match = re.search(r'(?:date|ship date)[:\s]*(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})', text, re.I)
        return match.group(1).strip() if match else ""
    
    def _extract_origin(self, text: str) -> str:
        match = re.search(r'(?:from|origin|ship from)[:\s]*(.+?)(?:\n|$)', text, re.I)
        return match.group(1).strip() if match else ""
    
    def _extract_destination(self, text: str) -> str:
        match = re.search(r'(?:to|destination|deliver to)[:\s]*(.+?)(?:\n|$)', text, re.I)
        return match.group(1).strip() if match else ""
    
    def _extract_carrier(self, text: str) -> str:
        carriers = ['ups', 'fedex', 'dhl', 'usps', 'amazon']
        t = text.lower()
        for carrier in carriers:
            if carrier in t:
                return carrier.upper()
        match = re.search(r'carrier[:\s]*(.+?)(?:\n|$)', text, re.I)
        return match.group(1).strip() if match else ""
