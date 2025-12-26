#!/usr/bin/env python3
"""
Base Document Processor
========================

Abstract base class for all document processors.
Provides common preprocessing utilities and interface definition.
"""

from abc import ABC, abstractmethod
from enum import Enum
from typing import Dict, List, Any, Optional, Tuple
# import cv2  <-- Moved inside functions
import numpy as np
from PIL import Image
import logging

logger = logging.getLogger(__name__)


class DocumentType(Enum):
    """Supported document types"""
    FINANCIAL = "financial"
    LEGAL = "legal"
    ID_CARD = "id_card"
    HEALTHCARE = "healthcare"
    HISTORICAL = "historical"
    FORM = "form"
    LOGISTICS = "logistics"
    UNKNOWN = "unknown"


class BaseDocumentProcessor(ABC):
    """
    Abstract base class for document processors.
    
    Each processor must implement:
    - extract_fields(): Extract document-specific fields
    - validate_fields(): Validate extracted data
    - get_expected_fields(): Return list of expected field names
    """
    
    def __init__(self, document_type: DocumentType):
        self.document_type = document_type
        self.logger = logging.getLogger(f"{__name__}.{document_type.value}")
    
    @abstractmethod
    def extract_fields(self, text: str, lines: List[str], image: Optional[Image.Image] = None) -> Dict[str, Any]:
        """
        Extract document-specific fields from OCR text.
        
        Args:
            text: Full OCR text
            lines: List of text lines
            image: Original image (optional, for visual analysis)
            
        Returns:
            Dictionary of extracted fields
        """
        pass
    
    @abstractmethod
    def validate_fields(self, fields: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """
        Validate extracted fields.
        
        Args:
            fields: Extracted fields dictionary
            
        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        pass
    
    @abstractmethod
    def get_expected_fields(self) -> List[str]:
        """
        Get list of expected field names for this document type.
        
        Returns:
            List of field names
        """
        pass
    
    # ========== Common Preprocessing Utilities ==========
    
    def deskew_image(self, image: np.ndarray) -> np.ndarray:
        """
        Deskew image using Hough line transform.
        
        Args:
            image: Input image (numpy array)
            
        Returns:
            Deskewed image
        """
        try:
            # Convert to grayscale if needed
            if len(image.shape) == 3:
                import cv2
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image.copy()
            
            # Apply edge detection
            edges = cv2.Canny(gray, 50, 150, apertureSize=3)
            
            # Detect lines
            lines = cv2.HoughLinesP(edges, 1, np.pi/180, 100, minLineLength=100, maxLineGap=10)
            
            if lines is None:
                return image
            
            # Calculate angles
            angles = []
            for line in lines:
                x1, y1, x2, y2 = line[0]
                angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
                angles.append(angle)
            
            # Get median angle
            median_angle = np.median(angles)
            
            # Rotate image
            if abs(median_angle) > 0.5:  # Only rotate if angle is significant
                (h, w) = image.shape[:2]
                center = (w // 2, h // 2)
                M = cv2.getRotationMatrix2D(center, median_angle, 1.0)
                rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
                self.logger.info(f"Deskewed image by {median_angle:.2f} degrees")
                return rotated
            
            return image
            
        except Exception as e:
            self.logger.warning(f"Deskew failed: {e}")
            return image
    
    def denoise_image(self, image: np.ndarray, strength: str = 'medium') -> np.ndarray:
        """
        Apply denoising to image.
        
        Args:
            image: Input image
            strength: 'light', 'medium', or 'heavy'
            
        Returns:
            Denoised image
        """
        try:
            if strength == 'light':
                h = 3
            elif strength == 'heavy':
                h = 15
            else:  # medium
                h = 10
            
            if len(image.shape) == 3:
                import cv2
                denoised = cv2.fastNlMeansDenoisingColored(image, None, h, h, 7, 21)
            else:
                denoised = cv2.fastNlMeansDenoising(image, None, h, 7, 21)
            
            return denoised
            
        except Exception as e:
            self.logger.warning(f"Denoising failed: {e}")
            return image
    
    def enhance_contrast(self, image: np.ndarray, method: str = 'clahe') -> np.ndarray:
        """
        Enhance image contrast.
        
        Args:
            image: Input image
            method: 'clahe', 'histogram', or 'adaptive'
            
        Returns:
            Contrast-enhanced image
        """
        try:
            # Convert to grayscale if needed
            if len(image.shape) == 3:
                import cv2
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image.copy()
            
            if method == 'clahe':
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                enhanced = clahe.apply(gray)
            elif method == 'histogram':
                enhanced = cv2.equalizeHist(gray)
            else:  # adaptive
                enhanced = cv2.adaptiveThreshold(
                    gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
                )
            
            # Convert back to color if input was color
            if len(image.shape) == 3:
                enhanced = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
            
            return enhanced
            
        except Exception as e:
            self.logger.warning(f"Contrast enhancement failed: {e}")
            return image
    
    def sharpen_image(self, image: np.ndarray) -> np.ndarray:
        """
        Sharpen image using unsharp masking.
        
        Args:
            image: Input image
            
        Returns:
            Sharpened image
        """
        try:
            kernel = np.array([[-1,-1,-1],
                             [-1, 9,-1],
                             [-1,-1,-1]])
            import cv2
            sharpened = cv2.filter2D(image, -1, kernel)
            return sharpened
            
        except Exception as e:
            self.logger.warning(f"Sharpening failed: {e}")
            return image
    
    def optimize_resolution(self, image: Image.Image, target_dpi: int = 300) -> Image.Image:
        """
        Optimize image resolution for OCR.
        
        Args:
            image: PIL Image
            target_dpi: Target DPI (default 300 for optimal OCR)
            
        Returns:
            Optimized PIL Image
        """
        try:
            # Get current DPI (default to 72 if not set)
            current_dpi = image.info.get('dpi', (72, 72))[0]
            
            if current_dpi < target_dpi:
                # Upscale
                scale_factor = target_dpi / current_dpi
                new_size = (int(image.width * scale_factor), int(image.height * scale_factor))
                image = image.resize(new_size, Image.Resampling.LANCZOS)
                self.logger.info(f"Upscaled image from {current_dpi} to {target_dpi} DPI")
            elif current_dpi > target_dpi * 1.5:
                # Downscale if significantly higher
                scale_factor = target_dpi / current_dpi
                new_size = (int(image.width * scale_factor), int(image.height * scale_factor))
                image = image.resize(new_size, Image.Resampling.LANCZOS)
                self.logger.info(f"Downscaled image from {current_dpi} to {target_dpi} DPI")
            
            return image
            
        except Exception as e:
            self.logger.warning(f"Resolution optimization failed: {e}")
            return image
    
    # ========== Common Field Validation Utilities ==========
    
    def is_valid_date(self, date_str: str) -> bool:
        """Check if string is a valid date"""
        import re
        from datetime import datetime
        
        # Common date patterns
        patterns = [
            r'\d{2}/\d{2}/\d{4}',  # DD/MM/YYYY or MM/DD/YYYY
            r'\d{2}-\d{2}-\d{4}',  # DD-MM-YYYY or MM-DD-YYYY
            r'\d{4}-\d{2}-\d{2}',  # YYYY-MM-DD
            r'\d{2}\.\d{2}\.\d{4}',  # DD.MM.YYYY
        ]
        
        for pattern in patterns:
            if re.match(pattern, date_str):
                return True
        
        return False
    
    def is_valid_amount(self, amount_str: str) -> bool:
        """Check if string is a valid monetary amount"""
        import re
        
        # Remove currency symbols and whitespace
        cleaned = re.sub(r'[₹$€£\s,]', '', amount_str)
        
        # Check if remaining is a valid number
        try:
            float(cleaned)
            return True
        except ValueError:
            return False
    
    def clean_field_value(self, value: str) -> str:
        """Clean field value by removing extra whitespace and special characters"""
        if not value:
            return ""
        
        # Remove extra whitespace
        value = ' '.join(value.split())
        
        # Remove common OCR artifacts
        value = value.replace('|', 'I').replace('_', '')
        
        return value.strip()
    
    def calculate_confidence(self, fields: Dict[str, Any]) -> float:
        """
        Calculate overall confidence score based on extracted fields.
        
        Args:
            fields: Extracted fields dictionary
            
        Returns:
            Confidence score (0-1)
        """
        expected_fields = self.get_expected_fields()
        
        if not expected_fields:
            return 1.0
        
        # Count how many expected fields were extracted
        extracted_count = sum(1 for field in expected_fields if fields.get(field))
        
        # Calculate percentage
        confidence = extracted_count / len(expected_fields)
        
        return round(confidence, 2)
