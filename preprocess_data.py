import cv2
import numpy as np
import os
from pathlib import Path

def preprocess_and_save_rois(template_path, test_path, save_dir, class_name):
    # 1. Load and Grayscale
    t_img = cv2.imread(template_path, 0)
    s_img = cv2.imread(test_path, 0)
    color_s = cv2.imread(test_path)
    
    # 2. Alignment (Friend's logic: Align test to template)
    # If they are already aligned, this just ensures they are the same size
    h, w = t_img.shape
    s_img = cv2.resize(s_img, (w, h))
    color_s = cv2.resize(color_s, (w, h))

    # 3. Subtraction (The "Main Problem" Solver)
    diff = cv2.absdiff(t_img, s_img)
    _, mask = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)
    
    # 4. Extract ROIs
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    os.makedirs(save_dir, exist_ok=True)
    for i, cnt in enumerate(contours):
        x, y, w_roi, h_roi = cv2.boundingRect(cnt)
        if w_roi * h_roi < 50: continue # Ignore tiny noise pixels
        
        # Center the defect in a 128x128 crop
        crop = color_s[y:y+h_roi, x:x+w_roi]
        resized_crop = cv2.resize(crop, (128, 128))
        
        cv2.imwrite(f"{save_dir}/{class_name}_{i}.jpg", resized_crop)

print("Preprocessing logic ready. This converts full boards into 128x128 training tiles.")