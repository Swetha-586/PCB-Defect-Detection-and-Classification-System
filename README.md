🔍 VisionCore AI: 
Hybrid PCB Defect Detection System
VisionCore AI is an industrial-grade automated inspection system designed to replace manual PCB review with a high-speed, intelligent pipeline. By merging Classical Computer Vision (OpenCV) for sub-pixel localization and EfficientNet-B4 Deep Learning (PyTorch) for classification, the system identifies manufacturing flaws in seconds with superior reliability.

🛠️ Problem Statement
Manual Bottlenecks: Human inspection is slow, inconsistent, and prone to fatigue.

Error Margin: Undetected defects (15–20% in standard lines) lead to massive financial losses.

Cost of Failure: A single defective board passing through assembly can cost between $50 and $500.

Scalability: Manual processes cannot keep up with high-speed modern manufacturing demands.

💡 Proposed Solution
Hybrid Pipeline: Combines deterministic image registration with probabilistic AI classification.

Real-time Interface: A Flask-based web dashboard for instant "Upload-to-Report" analysis.

Industrial Accuracy: Utilizing EfficientNet-B4 to achieve high-precision classification across 6 defect types.

Traceability: Generates annotated visual outputs and downloadable CSV prediction logs for QC records.

🚀 Key Features
Automated Identification: Detects 6 core PCB defects: Open, Short, Mousebite, Spur, Pinhole, and Spurious Copper.

Intelligent Localization: Employs sub-pixel alignment and Otsu thresholding to pinpoint minute flaws.

Interactive Dashboard: Real-time UI displaying prediction confidence scores and defect coordinates.

Exportable Documentation: One-click download for annotated results and historical inspection logs.

Deployment Ready: Optimized for seamless integration into modern smart factory workflows.

📁 Project Structure
The repository is organized to separate the core logic from the web interface:

Plaintext
VisionCore_AI/
├── PCB_USED/                   # Source "Golden" and "Test" board images
├── Image_subtraction_Outputs/  # Generated binary masks and difference maps
├── ROI_Dataset/                # Cropped 128x128 patches for AI training
├── static/                     # UI assets (CSS, JS, results)
├── templates/                  # Flask HTML templates
├── rotate.py                   # Image registration & alignment logic
├── subtraction.py              # Difference analysis & thresholding
├── ROI.py                      # Region of Interest extraction
├── train_final.py              # EfficientNet-B4 training & optimization
├── app.py                      # Main Flask Web Application
├── final_scanner.py            # End-to-end inference script
├── accuracy_curve.png          # Model training performance
└── confusion_matrix.png        # Class-wise accuracy breakdown
🏗️ System Architecture
VisionCore AI operates through a four-stage hybrid process:

Registration (rotate.py): Synchronizes the test board with a "Golden" template to eliminate alignment errors.

Difference Analysis (subtraction.py): Isolates anomalies using mathematical subtraction and binary thresholding.

Localization (ROI.py): Scans the mask and extracts 128x128 patches around potential defects.

Classification (train_final.py): Identifies the specific defect type using an EfficientNet-B4 backbone.

💻 Technology Stack
Language: Python 3.9+

Backend: Flask

Computer Vision: OpenCV, NumPy

Deep Learning: PyTorch, Torchvision

Frontend: HTML5, CSS3, JavaScript

Metrics: Scikit-learn, Matplotlib

🛠️ Setup & Installation
Prerequisites
Python 3.9+

NVIDIA GPU (Recommended for training)

Installation
Clone the Repository

Bash
git clone <your-repository-url>
cd VisionCore_AI
Install Dependencies

Bash
pip install -r requirements.txt
Launch the Web Application

Bash
python app.py
Open http://localhost:5000 in your browser.

📊 Performance
Architecture: EfficientNet-B4

Input Resolution: 128 x 128 pixels

Optimization: Fine-tuned for industrial PCB defect classes.

Results: High-contrast bounding boxes and categorized prediction logs.

🤝 Acknowledgments
We express our sincere gratitude to our mentor for her invaluable guidance and support throughout the development of VisionCore AI. This project showcases the power of Hybrid AI in solving critical real-world industrial challenges.
