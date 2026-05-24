# after extracting the coordinates of the image out yolo detector gives us in this file we will crop that coordinates of the bounding box and given to clip classifier which considers the color of the product aslo and classify what bottle is it, what chips bag is , etc

import logging
import config
from typing import List
import cv2
import numpy as np
logger = logging.getLogger(__name__)

class BrandClassifier:
    """
    Classifies generic product crops into specific brands using CLIP.
    """
    def __init__(self,model_name=config.clipmodel_name):
        logger.info(f"Initializing BrandClassifier with {model_name}...")
        try:
            from transformers import CLIPProcessor, CLIPModel
            self.processor = CLIPProcessor.from_pretrained(model_name)
            self.model = CLIPModel.from_pretrained(model_name)
            logger.info("CLIP model loaded successfully..")
        except ImportError:
            logger.error("transformers libraray not found")
            raise
        self.brands=config.clip_totalbrands
        self.target_brands=config.target_brands

    def classify(self,image:np.ndarray, box:List[int]) -> str:
        """
        Takes the full image and bounding box, crops it, and classifies the brand.
        """
        x1,y1,x2,y2=box
        h,w=image.shape[:2]
        # as image will be in this format (height,width,channel) so we want only first two coordinates
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        # coordinates doesnot go off edges
        cutout_img = image[y1:y2, x1:x2]
        if cutout_img.size == 0:
            return "Other"
        # opencv works with BGR format but almost every libraray works with RGB format
        cutout_imgrgb = cv2.cvtColor(cutout_img, cv2.COLOR_BGR2RGB)
        try:
            from PIL import Image
            # model compatible image loading(RGB consistency,easier preprocessing etc)
            pil_image = Image.fromarray(cutout_imgrgb)
        except ImportError:
            logger.error("Pillow library not found.")
            return "Other"
        # classifications
        inputs = self.processor(text=self.brands, images=pil_image, return_tensors="pt", padding=True)
        # return_tensors="pt" (turn my images in pytorch matrix  as it dont reads images)
        # Our brand words are different lengths (EX :"Amul" has 4 letters). A math matrix must be a perfect rectangle to work. padding=True tells Python to add blank spaces to the end of short words so they perfectly line up with the long words in the matrix!
        outputs = self.model(**inputs)
        # Our inputs variable is actually a Python Dictionary holding our image matrix and text matrix. By putting ** in front of it, it "Unpacks" the dictionary and hands the items to the AI model one-by-one. It's a Python shortcut for writing self.model(images=..., text=...).
        probs = outputs.logits_per_image.softmax(dim=1)
        # In Machine Learning, "Logits" is the fancy academic word for "Raw Score". Before the AI gives us nice percentages (like 95%), it outputs raw, messy math scores (like 14.5 for Pepsi and -2.1 for Amul). This line just grabs those raw scores so the softmax function can convert them into clean 0-100% percentages
        predicted_idx = probs.argmax().item()
        # fetching the number at maximum index position
        predicted_brand = self.brands[predicted_idx]
        # fectching the brand name using max index
        if probs[0][predicted_idx] < config.CLIP_confthreshold:
            return "Other"
        # checking confidence threshold if less than required added to other category
        if predicted_brand not in self.target_brands:
            return "Other"
        # even if the brand is not what we want to classify keep in other category
        return predicted_brand