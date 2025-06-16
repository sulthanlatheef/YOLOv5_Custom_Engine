from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
from PIL import Image

app = Flask(__name__)
CORS(app)

# Path to the custom weights file
# Dynamically get the current directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Construct relative path to the model file
model_path = os.path.join(BASE_DIR, 'best.pt')

# Load the YOLOv5 model with custom weights
model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path)
model.conf = 0.30

# Define a custom output mapping for all 17 classes (indices 0 to 16)
custom_labels = {
    0: "Smart Diagnostics confirmed denting on the body panel, requiring repair work. Kindly share your vehicle details in the next step so I can generate an overall estimate for your concern.",
    1: "Smart Diagnostics confirmed cracks or chips on the front windscreen needing replacement. Kindly share your vehicle details in the next step so I can generate an overall estimate for your concern.",
    2: " Smart Diagnostics confirmed damage to the headlight assembly leading to a replacement need. Kindly share your vehicle details in the next step so I can generate an overall estimate for your concern.",
    3: " Smart Diagnostics confirmed breakage on the rear windscreen, suggesting replacement. Kindly share your vehicle details in the next step so I can generate an overall estimate for your concern.",
    4: "Custom Output for Class 4",
    5: " Smart Diagnostics confirmed damage to the side mirror’s glass or casing leading to a replacement. Kindly share your vehicle details in the next step so I can generate an overall estimate for your concern.",
    6: "Custom Output for Class 6",
    7: "Smart Diagnostics confirmed damage to the taillight housing leading to a need for replacement. Kindly share your vehicle details in the next step so I can generate an overall estimate for your concern.",
    8: "Smart Diagnostics confirmed denting on the bonnet panel, suggesting body repairs. Kindly allow me to analyze the severity so we can generate an overall estimate for your concern.",
    9: "Custom Output for Class 9",
    10: "Smart Diagnostics confirmed denting on the outer door panel needing professional repair. Kindly allow me to analyze the severity so we can generate an overall estimate for your concern.",
    11: "Smart Diagnostics confirmed dents and deformations on the fender area requiring correction. Kindly allow me to analyze the severity so we can generate an overall estimate for your concern.",
    12: " Smart Diagnostics confirmed dents on the front bumper suggesting replacement or repair. Kindly allow me to analyze the severity so we can generate an overall estimate for your concern.",
    13: "Custom Output for Class 13",
    14: "Smart Diagnostics confirmed denting on the quarter panel requiring professional restoration.Kindly allow me to analyze the severity so we can generate an overall estimate for your concern.",
    15: " Smart Diagnostics confirmed dents on the rear bumper suggesting repair or replacement. Kindly allow me to analyze the severity so we can generate an overall estimate for your concern.",
    16: " Smart Diagnostics confirmed denting on the roof panel requiring restoration work. Kindly allow me to analyze the severity so we can generate an overall estimate for your concern."
}

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files['image']
    try:
        image = Image.open(file.stream).convert('RGB')
    except Exception as e:
        return jsonify({"error": "Invalid image file", "exception": str(e)}), 400

    # Let YOLOv5 handle resizing, letterboxing, and scaling
    results = model(image, size=480)

    predictions = []
    try:
        # YOLOv5 returns a tensor with columns: [x1, y1, x2, y2, confidence, class]
        detections = results.xyxyn[0]
        for detection in detections:
            cls_index = int(detection[5])
            confidence_score = float(detection[4])
            custom = custom_labels.get(cls_index, "Unknown Custom Output")
            original = model.names[cls_index] if cls_index in model.names else "Unknown"
            predictions.append({
                "custom": custom,
                "original": original,
                "confidence": confidence_score  # raw confidence (assumed to be between 0 and 1)
            })
    except Exception as e:
        return jsonify({"error": "Failed to extract predictions", "exception": str(e)}), 500

    if not predictions:
        predictions = [{"custom": "Please upload slightly closer version of the issue,If problem persists kindly consider the Regular Service (prime care) option", "original": "Unable to detect an issue", "confidence": 0.0}]

    return jsonify({"predictions": predictions})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)

