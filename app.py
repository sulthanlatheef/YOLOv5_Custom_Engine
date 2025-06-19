from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import torch
from PIL import Image

app = Flask(__name__)
CORS(app)

# Health‑check root
@app.route('/', methods=['GET'])
def index():
    return jsonify({"status": "App is running successfully on Render!"})

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'best.pt')
YOLOV5_DIR = os.path.join(BASE_DIR, 'yolov5')

# Custom output labels
custom_labels = {
    0: "Smart Diagnostics confirmed denting on the body panel, requiring repair work. Kindly share your vehicle details in the next step so I can generate an overall estimate for your concern.",
    1: "Smart Diagnostics confirmed cracks or chips on the front windscreen needing replacement. Kindly share your vehicle details in the next step so I can generate an overall estimate for your concern.",
    2: "Smart Diagnostics confirmed damage to the headlight assembly leading to a replacement need. Kindly share your vehicle details in the next step so I can generate an overall estimate for your concern.",
    3: "Smart Diagnostics confirmed breakage on the rear windscreen, suggesting replacement. Kindly share your vehicle details in the next step so I can generate an overall estimate for your concern.",
    4: "Custom Output for Class 4",
    5: "Smart Diagnostics confirmed damage to the side mirror’s glass or casing leading to a replacement. Kindly share your vehicle details in the next step so I can generate an overall estimate for your concern.",
    6: "Custom Output for Class 6",
    7: "Smart Diagnostics confirmed damage to the taillight housing leading to a need for replacement. Kindly share your vehicle details in the next step so I can generate an overall estimate for your concern.",
    8: "Smart Diagnostics confirmed denting on the bonnet panel, suggesting body repairs. Kindly allow me to analyze the severity so we can generate an overall estimate for your concern.",
    9: "Custom Output for Class 9",
    10: "Smart Diagnostics confirmed denting on the outer door panel needing professional repair. Kindly allow me to analyze the severity so we can generate an overall estimate for your concern.",
    11: "Smart Diagnostics confirmed dents and deformations on the fender area requiring correction. Kindly allow me to analyze the severity so we can generate an overall estimate for your concern.",
    12: "Smart Diagnostics confirmed dents on the front bumper suggesting replacement or repair. Kindly allow me to analyze the severity so we can generate an overall estimate for your concern.",
    13: "Custom Output for Class 13",
    14: "Smart Diagnostics confirmed denting on the quarter panel requiring professional restoration. Kindly allow me to analyze the severity so we can generate an overall estimate for your concern.",
    15: "Smart Diagnostics confirmed dents on the rear bumper suggesting repair or replacement. Kindly allow me to analyze the severity so we can generate an overall estimate for your concern.",
    16: "Smart Diagnostics confirmed denting on the roof panel requiring restoration work. Kindly allow me to analyze the severity so we can generate an overall estimate for your concern."
}

# Load the YOLOv5 model from the local clone (only once)
model = torch.hub.load(
    repo_or_dir=YOLOV5_DIR,
    model='custom',
    path=MODEL_PATH,
    source='local',
    force_reload=False
)
model.conf = 0.30  # set confidence threshold

@app.route('/ping')
def ping():
    return "pong", 200

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    try:
        img = Image.open(request.files['image'].stream).convert('RGB')
    except Exception as e:
        return jsonify({"error": "Invalid image file", "exception": str(e)}), 400

    # Run inference
    results = model(img, size=480)
    detections = results.xyxyn[0]  # tensor with [x1,y1,x2,y2,conf,cls] rows

    predictions = []
    for det in detections:
        cls_idx = int(det[5].item())
        conf    = float(det[4].item())
        predictions.append({
            "custom":   custom_labels.get(cls_idx, "Unknown Custom Output"),
            "original": model.names.get(cls_idx, "Unknown"),
            "confidence": conf
        })

    if not predictions:
        predictions = [{
            "custom":   "Please upload slightly closer version of the issue. If problem persists, kindly consider the Regular Service (prime care) option.",
            "original": "Unable to detect an issue",
            "confidence": 0.0
        }]

    return jsonify({"predictions": predictions})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port)
