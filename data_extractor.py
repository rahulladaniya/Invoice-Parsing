import os
import numpy as np
from paddleocr import PaddleOCR
from PIL import Image

class OCRExtractor:
    def __init__(self):
        self.ocr = PaddleOCR(use_angle_cls=True, lang='en')
        
    def extract_text(self, cropped_images):
        """
        Extract text from a list of cropped images using PaddleOCR.
        :param cropped_images: List of tuples (Image, label)
        :return: Dictionary with labels as keys and extracted text as values
        """
        ocr_results = {}
        
        for i, (cropped_img, label) in enumerate(cropped_images):
            try:
                # Convert PIL Image to numpy array for PaddleOCR
                image_array = np.array(cropped_img)
                
                # Perform OCR
                result = self.ocr.ocr(image_array, cls=True)
                
                # Extract text from the OCR result
                text = '\n'.join([line[1][0] for line in result[0]])
                
                ocr_results[label] = text
            except Exception as e:
                ocr_results[label] = f"Error processing image: {e}"
                
        return ocr_results
