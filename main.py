import os
import cv2
import json
import logging
from analytics.shelf_metrics import shelfmetrics

# Set up the console logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    logger.info("Starting Retail Shelf Intelligence Pipeline...")
    # Start the Master Controller using your new name
    engine = shelfmetrics()
    # The list of images we want to scan
    input_images = ["input_images/img_1.jpg", "input_images/img_2.jpg", "input_images/img_3.jpg","input_images/testem.jpeg"]
    # Creating the output folder
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)
    all_results = []
    for image_file in input_images:
        # Safety check to prevent crashing if a file is missing
        if not os.path.exists(image_file):
            logger.warning(f"File {image_file} not found. Skipping...")
            continue
            
        logger.info(f"Processing {image_file}...")
        # Running the pipeline 
        metrics, annotated_image = engine.process_image(image_path=image_file)
        
        # Save the painted green-box image
        base_name = os.path.basename(image_file)
        output_image_path = os.path.join(output_dir, f"analyzed_{base_name}")
        cv2.imwrite(output_image_path, annotated_image)
        logger.info(f"Saved annotated image to {output_image_path}")
        
        # Add the JSON math to our master list
        all_results.append(metrics)
        
    # Save the final JSON file format
    output_json_path = os.path.join(output_dir, "shelf_metrics.json")
    with open(output_json_path, 'w') as f:
        json.dump(all_results, f, indent=4)
        
    logger.info(f"Pipeline complete! All metrics saved to {output_json_path}")

# Run the program!
if __name__ == "__main__":
    main()