# Advanced AI Diagnostics Engine for Vehicle Damage Detection

A YOLOv5-based AI engine designed to automatically detect external vehicle damages from images. This engine can identify multiple types of car damages, provide confidence scores for severity, and can be integrated into web or desktop applications for automated service estimates.

## Features

- **Damage Detection**: Detects external car damages including:
  - Front/Rear Glass Damage
  - Headlight Damage
  - Bumper Scratches & Dents
  - Side Mirror Damage
- **Confidence Scoring**: Each detected damage includes a confidence score, useful for estimating severity.
- **YOLOv5 Model**: Trained with a custom dataset for high accuracy on vehicle damage detection.
- **Integration Ready**: Can be used with Flask or other web frameworks, easily integrated into FullStack applications.

## Tech Stack

- **AI/ML Framework**: PyTorch, YOLOv5
- **Programming Language**: Python 3.x
- **Computer Vision**: OpenCV
- **Web Integration**: Flask API (for connecting with web applications)
- **Dataset**: Custom-labeled vehicle damage images

## Installation & Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/advanced-ai-diagnostics.gitxt
2.Install required Python packages:
 pip install -r requirements.txt
 
3.Ensure best.pt (trained YOLOv5 model) is in the root directory.

4. Run the Flask API:
   ```bash
   python app.py
 5.The engine is now ready to receive vehicle images via API for damage detection.  






   
