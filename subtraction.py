import cv2
import numpy as np
import os

def align_images(template, test_img):
    """
    Precision alignment (ECC) to ensure template and test images 
    overlap perfectly before subtraction.
    """
    warp_mode = cv2.MOTION_TRANSLATION
    warp_matrix = np.eye(2, 3, dtype=np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 50, 1e-10)

    try:
        (_, warp_matrix) = cv2.findTransformECC(template, test_img, warp_matrix, warp_mode, criteria)
        aligned_img = cv2.warpAffine(test_img, warp_matrix, (template.shape[1], template.shape[0]), 
                                    flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
        return aligned_img
    except:
        # Fallback to basic resize if alignment fails
        return cv2.resize(test_img, (template.shape[1], template.shape[0]))

def process_subtraction(base_path):
    # Output directory for your masks
    output_base_dir = os.path.join(base_path, "Image_subtraction_Outputs")
    
    # Define exact defect folders inside 'images'
    defect_types = ["Missing_hole", "Mouse_bite", "Open_circuit", "Short", "Spur", "Spurious_copper"]
    
    for defect in defect_types:
        folder_path = os.path.join(base_path, "images", defect)
        if not os.path.exists(folder_path):
            print(f"Skipping: {defect} folder not found.")
            continue
            
        print(f"Processing Category: {defect}")
        
        for filename in os.listdir(folder_path):
            if filename.lower().endswith((".jpg", ".jpeg", ".png")):
                # Logic: filename '01_test.jpg' -> template '01.JPG'
                template_id = filename.split('_')[0]
                template_path = os.path.join(base_path, "PCB_USED", f"{template_id}.JPG")
                test_path = os.path.join(folder_path, filename)

                if not os.path.exists(template_path):
                    print(f" ! Template missing for {filename}")
                    continue

                # Load in Grayscale for subtraction
                template = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
                test_img = cv2.imread(test_path, cv2.IMREAD_GRAYSCALE)
                
                if template is None or test_img is None: continue

                # 1. Align images to remove background shift
                aligned_test = align_images(template, test_img)
                
                # 2. Gaussian Blur to reduce sensor noise
                t_blur = cv2.GaussianBlur(template, (5, 5), 0)
                s_blur = cv2.GaussianBlur(aligned_test, (5, 5), 0)

                # 3. Absolute Difference
                diff_map = cv2.absdiff(t_blur, s_blur)

                # 4. Otsu's Thresholding to identify the defect
                _, thresh = cv2.threshold(diff_map, 35, 255, cv2.THRESH_BINARY)
                
                # 5. Morphological Opening to remove tiny noise spots
                kernel = np.ones((5, 5), np.uint8)
                mask = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

                # Save the mask
                output_dir = os.path.join(output_base_dir, defect)
                os.makedirs(output_dir, exist_ok=True)
                cv2.imwrite(os.path.join(output_dir, f"mask_{filename}"), mask)

    print("\n[SUCCESS] All subtraction masks saved in 'Image_subtraction_Outputs'.")

if __name__ == "__main__":
    # Path to your main project folder
    process_subtraction(".")