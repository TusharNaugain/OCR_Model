"""
Document Processors Package
===========================

Specialized OCR processors for different document types.
Each processor implements document-specific field extraction and validation.
"""

from .base_processor import BaseDocumentProcessor, DocumentType
from .financial_processor import FinancialProcessor
from .id_card_processor import IDCardProcessor
from .form_processor import FormProcessor
from .legal_processor import LegalProcessor
from .healthcare_processor import HealthcareProcessor
from .historical_processor import HistoricalProcessor
from .logistics_processor import LogisticsProcessor

__all__ = [
    'BaseDocumentProcessor',
    'DocumentType',
    'FinancialProcessor',
    'IDCardProcessor',
    'FormProcessor',
    'LegalProcessor',
    'HealthcareProcessor',
    'HistoricalProcessor',
    'LogisticsProcessor',
]
