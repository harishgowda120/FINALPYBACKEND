from flask import Flask, request, jsonify
from flask_cors import CORS
from deepface import DeepFace
import base64
import cv2
import numpy as np

app = Flask(__name__)
CORS(app)

@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "Backend running successfully!"})

@app.route("/detect_emotion", methods=["POST"])
def detect_emotion():
    try:
        data = request.get_json()
        if "image" not in data:
            return jsonify({"error": "Image missing"}), 400

        img_base64 = data["image"]

        # Convert Base64 → OpenCV image
        img_bytes = base64.b64decode(img_base64)
        np_arr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if img is None:
            return jsonify({"error": "Invalid image"}), 400

        # Emotion detection
        result = DeepFace.analyze(img, actions=["emotion"], enforce_detection=False)

        return jsonify({
            "emotion": result[0]["dominant_emotion"]
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
