# This file is used to detect the genric shapes like bottles/packets using "Spatial Geometry" (Shapes).It will only detect the shapes. We are not using any other version of yolo like yolov8 because it is trained on coco dataset may able to detect bottles but will not able to know about packets of lays so we are using yolo-world instead because it knows that common words as YOLO-World is a special hybrid model. It took the fast "eyes" of YOLOv8 and attached them to the massive "brain" of a language model.

import logging
import config
from typing import List, Dict, Any
# type hinting will tell ehat is the expected data type
logger = logging.getLogger(__name__)

class Shelfscanner:
    def __init__(self,model_name:str=config.yolomodel_path):
        """
        Initializes the YOLO-World model.
        """
        logger.info(f"initializing the shelf scanner with {model_name}...")
        # using error handling to handle errors gracefully instead of crashing the entire program
        try:
            from ultralytics import YOLOWorld
            self.model=YOLOWorld(model_name)
            logger.info("Yolo-world model loaded successfully...")
        except ImportError:
            logger.error("Ultralytics library not found, kindly download it using :pip install ultralytics")
            raise
        except Exception as e:
            logger.error(f"failed to load Yolo-world model {e}")
            raise
        # giving the yolo model to detect these custom classes
        self.model.set_classes(config.yolo_search_words)
        self.search_words = config.yolo_search_words

    def detect(self,image_path: str) -> List[Dict[str,Any]]:
        """
        Runs detection on the given image and returns a list of bounding boxes and class names.
        """
        logger.info(f"Running detection on {image_path}")
        try:
            results=self.model.predict(image_path,conf=config.YOLO_confthreshold,verbose=False)
            # conf:confidence threshold to keep the products whose confidence score is more than the given
            # verbose=false to suppresses extra logs making cleaner console
            items_found=[]
            if len(results)>0:
                boxes=results[0].boxes
                # extracting informatrion related to bounding box 
                for box in boxes:
                    x1,y1,x2,y2=box.xyxy[0].cpu().numpy()
                    # PyTorch forces arrays into tensors even if there is only 1 item so extracting the first list using [0]
                    # .cpu() used becazse yolo may store tensors on gpu but numpy works on cpu
                    # .numpy() converts tensors to numpy arrays
                    word_id=int(box.cls[0].cpu().item())
                    # fetching the classid which is stored in tensor converted into tensor converted to standard number
                    word_name=self.search_words[word_id]
                    # fectching the name of the word detected and confidence score, after that adding box coordiantes,class name and class if to the items_found list
                    conf=float(box.conf[0].cpu().item())
                    items_found.append({
                        "box":[int(x1),int(y1),int(x2),int(y2)],
                        "object_name" : word_name,
                        "confidence" : conf
                    })
            logger.info(f"Found {len(items_found)} objects.")
            return items_found
        except Exception as  e:
            logger.error(f"Detection failed on {image_path}: {e}")
            return []