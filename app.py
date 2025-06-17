from flask import Flask, request, jsonify
from flask_cors import CORS
import os
from PIL import Image
from ultralytics import YOLO

app = Flask(__name__)
CORS(app)

# Root route for Render health check
@app.route('/', methods=['GET'])
def index():
    return jsonify({"status": "App is running successfully on Render!"})

# Get current directory and model path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, 'best.pt')

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

# Load model once at startup
model = YOLO(model_path)
model.conf = 0.30  # set confidence threshold

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    try:
        img = Image.open(request.files['image'].stream).convert('RGB')
    except Exception as e:
        return jsonify({"error": "Invalid image file", "exception": str(e)}), 400

    # Run inference
    results = model(img, imgsz=480)  # returns a list with one Results object
    boxes = results[0].boxes  # Boxes object

    predictions = []
    for box in boxes:
        cls = int(box.cls[0])   # class index
        conf = float(box.conf[0])
        predictions.append({
            "custom": custom_labels.get(cls, "Unknown Custom Output"),
            "original": model.names.get(cls, "Unknown"),
            "confidence": conf
        })

    if not predictions:
        predictions = [{
            "custom": "Please upload slightly closer version of the issue. If problem persists, kindly consider the Regular Service (prime care) option.",
            "original": "Unable to detect an issue",
            "confidence": 0.0
        }]

    return jsonify({"predictions": predictions})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port)
