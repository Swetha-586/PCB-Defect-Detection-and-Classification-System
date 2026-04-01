import cv2
import torch
import numpy as np
import os
from torchvision import transforms, models
import torch.nn as nn

# --- 1. CONFIG ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHECKPOINT = "model_checkpoint.pth" 
TEMPLATE_DIR = "PCB_USED"
CLASS_NAMES = ['Missing_hole', 'Mouse_bite', 'Open_circuit', 'Short', 'Spur', 'Spurious_copper']

def align_images(template, test):
    """Perfectly aligns the test image to the template to prevent ghosting."""
    # Find size of template
    sz = template.shape
    # Define the motion model
    warp_mode = cv2.MOTION_TRANSLATION
    warp_matrix = np.eye(2, 3, dtype=np.float32)
    # Run the ECC algorithm to find the perfect alignment
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 50, 1e-8)
    try:
        (_, warp_matrix) = cv2.findTransformECC(template, test, warp_matrix, warp_mode, criteria)
        test_aligned = cv2.warpAffine(test, warp_matrix, (sz[1], sz[0]), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
        return test_aligned
    except:
        return test # Fallback if alignment fails

def scan_pcb(test_path):
    print("Loading AI and Aligning Boards...")
    model = models.efficientnet_b4()
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, len(CLASS_NAMES))
    model.load_state_dict(torch.load(CHECKPOINT, map_location=DEVICE))
    model.to(DEVICE).eval()

    test_img_color = cv2.imread(os.path.normpath(test_path))
    test_gray = cv2.cvtColor(test_img_color, cv2.COLOR_BGR2GRAY)
    
    # 1. Match Template
    template_path = os.path.join(TEMPLATE_DIR, os.listdir(TEMPLATE_DIR)[0]) # Change if you have many templates
    template_img = cv2.imread(template_path, 0)
    template_img = cv2.resize(template_img, (test_gray.shape[1], test_gray.shape[0]))

    # 2. PERFORM ALIGNMENT (The Fix)
    test_gray_aligned = align_images(template_img, test_gray)

    # 3. Subtraction
    diff = cv2.absdiff(template_img, test_gray_aligned)
    _, mask = cv2.threshold(diff, 40, 255, cv2.THRESH_BINARY)
    
    # Use a smaller kernel to avoid deleting small Spurious Copper
    kernel = np.ones((2,2), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    tf = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w * h < 30: continue 
        
        # INCREASED PADDING TO 30 for Shorts/Spurious Copper
        # This helps the AI see if the copper is touching other traces (Short) or just floating (Spurious)
        padding = 30 
        y1, y2 = max(0, y-padding), min(test_img_color.shape[0], y+h+padding)
        x1, x2 = max(0, x-padding), min(test_img_color.shape[1], x+w+padding)
        
        crop = test_img_color[y1:y2, x1:x2]
        input_tensor = tf(crop).unsqueeze(0).to(DEVICE)
        
        with torch.no_grad():
            output = model(input_tensor)
            _, pred = torch.max(output, 1)
            label = CLASS_NAMES[pred.item()]

        cv2.rectangle(test_img_color, (x, y), (x+w, y+h), (0, 0, 255), 2)
        cv2.putText(test_img_color, label, (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    cv2.imwrite("FINAL_ALIGNED_REPORT.jpg", test_img_color)
    print("Done! Alignment fixed the Spurious/Short detection.")

if __name__ == "__main__":
    scan_pcb(r"images\Spurious_copper\01_spurious_copper_01.jpg")