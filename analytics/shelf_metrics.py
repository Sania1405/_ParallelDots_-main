# This file is the master controller such that connects the pipeline.It tells YOLO to find the items, tells CLIP to identify them, tells EasyOCR to read the prices, and then calculates the final business math (like Shelf Space Percentage).
import config
import cv2
import numpy as np
import logging
from typing import Dict, Any, List, Tuple
from collections import defaultdict
from detection.yolo_detector import Shelfscanner
from classification.clip_classifier import BrandClassifier
from ocr.easy_ocr import pricetagreader
logger = logging.getLogger(__name__)

class shelfmetrics:
    """
    connect and manages the ML Pipeline for Retail Shelf Intelligence.
    Combines Detection, Classification, OCR, and Shelf Segmentation.
    """
    def __init__(self):
        logger.info("Initializing ShelfAnalyzer Pipeline...")
        self.detector = Shelfscanner(config.yolomodel_path)
        self.brand_classifier = BrandClassifier(model_name="openai/clip-vit-base-patch32")
        self.ocr_extractor =pricetagreader(languages=['en'])

    def process_image(self, image_path: str) -> Tuple[Dict[str, Any], np.ndarray]:
        """
        Processes a single shelf image and returns the business metrics and the annotated image.
        """
        #Loading Image using OpenCV
        image = cv2.imread(image_path)
        if image is None:
            logger.error(f"Failed to load image: {image_path}")
            return {}, np.zeros((10,10,3), dtype=np.uint8)
        # If accidentally given the code a broken image path, OpenCV (cv2.imread) returns None. If the code tries to process None, the whole program crashes.Instead,we use numpy to create a tiny, fake 10x10 black square image to  return these empty/fake things so the main pipeline so it will skip empty image rather than crashing pipeline
        annotated_image = image.copy()
        # copying image so that original one will not destroy as if needed to reuse original one so creates an issue
        logger.info("Running Generic Object Detection with YOLO...")
        detections = self.detector.detect(image_path)
        metrics = {
            "image_name": image_path.split("/")[-1].split("\\")[-1],
            "total_products": len(detections),
            "brands": defaultdict(int),
            "shelf_space_pixels": defaultdict(int),
            "total_shelf_pixels": 0,
            "ocr_labels": []
        }
        # defaultdict(int) used in dictionary if any brand will not present rather giving error, intialize it.
        for det in detections:
            x1, y1, x2, y2 = det["box"]
            det["y_center"] = (y1 + y2) / 2
            det["x_center"] = (x1 + x2) / 2
            # YOLO finds boxes randomly. It doesn't know what a "shelf" is.For every box, we add the Top (y1) and Bottom (y2) together and divide by 2 to find the exact middle of the box (y_center).so we can compare next product if its wirithin the range of previous one that means both are on the same shelf else not
        detections.sort(key=lambda d: d["y_center"])
            # runns internal loop to extract y_centre for each index and at last sort the whole list using timsort(extract key-compare-shift-insert)
        shelves = []
        if detections:
            current_shelf = [detections[0]]
            for det in detections[1:]:
                # saving the very first product that is in the top shelf in cureent shelf and looping over rest of the elements to see if rest elements belong to the same shelf or not
                # If y_center is within 60 pixels, it's on the same shelf
                if abs(det["y_center"] - current_shelf[-1]["y_center"]) < config.shelfheight_comparison:
                    # current_shelf[-1] fetching the y_centre (detections is a dictionary  in list)
                    current_shelf.append(det)
                    # product belongs to same shelf
                else:
                    shelves.append(current_shelf)
                    # otherwise add the products that were in previous shelf
                    current_shelf = [det]
                    # starting adding products in new shelf as distance between both are more.
            shelves.append(current_shelf)
        
        def calculate_similarity(crop1, crop2):
            # comparison between the crops of proucts
            if crop1 is None or crop2 is None or crop1.size == 0 or crop2.size == 0:
                return 0.0
            hsv1 = cv2.cvtColor(crop1, cv2.COLOR_BGR2HSV)
            # rgb converted to hsv (Hue(pure color), Saturation(intensity), and Value(brightness))for better comparison and use to track a color in different lightning
            hsv2 = cv2.cvtColor(crop2, cv2.COLOR_BGR2HSV)
            hist1 = cv2.calcHist([hsv1], [0, 1], None, [50, 60], [0, 180, 0, 256])
            hist2 = cv2.calcHist([hsv2], [0, 1], None, [50, 60], [0, 180, 0, 256])
            # cv2.calcHist()-A 2D histogram creates a grid or matrix where the rows represent different shades of Hue and the columns represent different levels of Saturation.
            #(example) hist2 = cv2.calcHist([source image], [channels we want to analyze currenlty we r not considering brightness to compare product in light or dark shadow], None(mask,want OpenCV to look at 100% of the pixels inside cropped image.), [no. of grids to analyze the cropped image], [range of hue and saturation to make bins inside this range])
            cv2.normalize(hist1, hist1, 0, 1, cv2.NORM_MINMAX)
            cv2.normalize(hist2, hist2, 0, 1, cv2.NORM_MINMAX)
            # normalizing both images [0,1] for better comparison even if in case of their physical larger size(more pixels)
            # grid to grid comparison and coorelation value will be returned
            # if 1.0 (Perfect Match): The two color prints are absolutely identical
            # 0.0 (Unrelated): The colors have absolutely nothing in common
            # -1.0 (Inverse Match): The histograms are completely opposite
            return cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
        for shelf in shelves:
            # Sort left-to-right on the shelf
            shelf.sort(key=lambda d: d["x_center"])
            
            previous_brand = None
            previous_crop = None
            
            for det in shelf:
                x1, y1, x2, y2 = det["box"]
                
                h, w = image.shape[:2]
                cx1, cy1 = max(0, x1), max(0, y1)
                # min and max is used to prevent the program being crashed if bounding box id detected out of the edge
                cx2, cy2 = min(w, x2), min(h, y2)
                # if bounding box coordinates detected are larger than the actual image dimensions
                current_crop = image[cy1:cy2, cx1:cx2]
                # cropping rows y1 to y2 and columns x1 to x2
                
                # Check visual similarity with previous product
                is_similar = False
                if previous_crop is not None:
                    sim_score = calculate_similarity(previous_crop, current_crop)
                    # got correlation value
                    if sim_score > config.viusualsimilarity_threshold:  # checking visual similarity
                        is_similar = True
                        
                if is_similar:
                    # Same block, copy the brand!
                    final_brand = previous_brand
                else:
                    # New block, run the AI classifier
                    final_brand = self.brand_classifier.classify(image, det["box"])
                    previous_brand = final_brand
                    previous_crop = current_crop
                    
                metrics["brands"][final_brand] += 1
                
                # Estimate shelf space by the bounding box width
                box_width = x2 - x1
                metrics["shelf_space_pixels"][final_brand] += box_width
                metrics["total_shelf_pixels"] += box_width
                
                # Extract price tag directly below the product
                roi_y1 = min(h, y2)
                roi_y2 = min(h, y2 + 60) # Search area 60px below
                roi_x1 = max(0, x1)
                roi_x2 = min(w, x2)
                
                price_crop = image[roi_y1:roi_y2, roi_x1:roi_x2]
                price_text = self.ocr_extractor.extract_text_from_crop(price_crop)
                if price_text:
                    metrics["ocr_labels"].append(price_text)
                
                # Draw green box for products - much thinner!
                cv2.rectangle(annotated_image, (x1, y1), (x2, y2), (0, 255, 0), 1)
                label = f"{final_brand}"
                cv2.putText(annotated_image, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        # Convert defaultdicts to standard dicts for clean JSON output
        metrics["brands"] = dict(metrics["brands"])
        
        # Calculate percentage of shelf space per brand and extract and  removing it because we dont need it in final metrics
        total_px = metrics.pop("total_shelf_pixels")
        metrics["shelf_space_percentage"] = {}
        metrics["availability_status"] = {}
        # initialize empty dictionaries
        
        for brand, px in metrics.pop("shelf_space_pixels").items():
            if total_px > 0:
                pct = (px / total_px) * 100
                metrics["shelf_space_percentage"][brand] = f"{pct:.1f}%"
                # showing only one digit after decimal
                if pct >= 20.0:
                    metrics["availability_status"][brand] = "High Presence"
                elif pct >= 5.0:
                    metrics["availability_status"][brand] = "Medium Presence"
                else:
                    metrics["availability_status"][brand] = "Low Presence"
                    # checking availability of product
            else:
                metrics["shelf_space_percentage"][brand] = "0%"
                metrics["availability_status"][brand] = "Unknown"
                # handles divide-by-zero edge case when no detections,image failed
        
        # Ensure unique OCR labels
        metrics["ocr_labels"] = list(set(metrics["ocr_labels"]))
        # only unique prices
        
        return metrics, annotated_image