# This file is given a below part of the image (the 60 pixels directly below a product on the shelf)where price tag is there on the shelf. It scans that that for text, reads the text, and then uses a filter (Regex) to extract only the actual prices, ignoring things like "500ml" or "1kg".

import logging
from typing import List
import numpy as np
logger = logging.getLogger(__name__)
class pricetagreader:
    """
    class to handle text extraction from image strips using OpenCV.
    """
    def __init__(self, languages: List[str]=['en']):
        logger.info("Initializing OCRExtractor with EasyOCR...")
        try:
            import easyocr
            self.reader=easyocr.Reader(languages,gpu=False)
            # GPU set to False for maximum compatibility on any machine
            logger.info("EasyOCR initialized successfully....")
        except ImportError:
            logger.error("easyocr library not found. Please install it using: pip install easyocr")
            raise
        except Exception as e:
            logger.error(f"Failed to load EasyOCR: {e}")
            raise

    def extract_text_from_crop(self, crop_img: np.ndarray) -> str:
        """
        Runs OCR on a specific cropped region of interest (e.g. below a product).
        """
        try:
            if crop_img is None or crop_img.size == 0:
                return ""
            results = self.reader.readtext(crop_img)
            # result will contain a list, it includes the coordinates of the text, the text itself, and the confidence score.
            import re
            raw_text = " ".join([res[1] for res in results]).lower()
            # extracting the text from all tags and join them in lowercase using list comprehension
            cleaned_text = re.sub(r'\b\d+(?:\.\d+)?\s*(?:ml|l|gm|g|kg|mm|oz|pc|pcs)\b', '', raw_text)
            # re.sub(pattern, replacement, string)
            #replace the given pattern with blank space such that It finds strings like "500ml" or "1.5 kg" and completely deletes them from the text!
            numbers = re.findall(r'\d+(?:\.\d+)?', cleaned_text)
            # re.findall(pattern,text) used to find all matching patterns in a string
            text_found = " ".join(numbers).strip()
            # join elements of list return it removing extra space
            return text_found
            
        except Exception as e:
            logger.warning(f"OCR extraction failed on crop: {e}")
            return ""