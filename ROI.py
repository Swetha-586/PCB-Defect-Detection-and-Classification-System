import os
import cv2
import xml.etree.ElementTree as ET
from tqdm import tqdm

def process_module_2_direct(base_path):
    # Paths based on your folders
    image_dir = os.path.join(base_path, "images")
    annotation_base = os.path.join(base_path, "Annotations")
    output_roi_dir = os.path.join(base_path, "ROI_Dataset")
    
    # Exact category names from your project
    defect_types = ["Missing_hole", "Mouse_bite", "Open_circuit", "Short", "Spur", "Spurious_copper"]
    
    # Padding: Adds a small margin around the defect for better context
    PADDING = 5

    print("Starting Module 2: Direct ROI Extraction from Annotations...")

    for defect in defect_types:
        defect_img_path = os.path.join(image_dir, defect)
        defect_anno_path = os.path.join(annotation_base, defect)
        
        if not os.path.exists(defect_img_path) or not os.path.exists(defect_anno_path):
            continue

        os.makedirs(os.path.join(output_roi_dir, defect), exist_ok=True)

        # Get list of images in the defect folder
        image_files = [f for f in os.listdir(defect_img_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        for filename in tqdm(image_files, desc=f"Processing {defect}"):
            base_name = os.path.splitext(filename)[0]
            
            # Check for XML inside the defect subfolder
            xml_path = os.path.join(defect_anno_path, f"{base_name}.xml")
            
            if not os.path.exists(xml_path):
                continue

            # Load the original PCB image
            img = cv2.imread(os.path.join(defect_img_path, filename))
            if img is None: continue
            
            h_img, w_img = img.shape[:2]

            # Parse XML coordinates
            tree = ET.parse(xml_path)
            root = tree.getroot()

            for i, obj in enumerate(root.findall('object')):
                bbox = obj.find('bndbox')
                
                # Convert to int, applying padding while staying within image boundaries
                xmin = max(0, int(float(bbox.find('xmin').text)) - PADDING)
                ymin = max(0, int(float(bbox.find('ymin').text)) - PADDING)
                xmax = min(w_img, int(float(bbox.find('xmax').text)) + PADDING)
                ymax = min(h_img, int(float(bbox.find('ymax').text)) + PADDING)

                # Direct Crop from original image
                roi_crop = img[ymin:ymax, xmin:xmax]

                # Save to ROI_Dataset
                roi_name = f"{base_name}_roi_{i}.jpg"
                save_path = os.path.join(output_roi_dir, defect, roi_name)
                cv2.imwrite(save_path, roi_crop)

    print(f"\n[SUCCESS] Natural ROIs extracted to: {output_roi_dir}")

if __name__ == "__main__":
    process_module_2_direct(".")