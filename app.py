import os
os.environ['TF_USE_LEGACY_KERAS'] = '0'

from flask import Flask, request, jsonify
from flask_cors import CORS
from deepface import DeepFace
import cv2
import numpy as np
import time

app = Flask(__name__)
CORS(app)

# Emotion explanations
emotion_explanations = {
    "happy": "The person looks cheerful and content.",
    "sad": "The person seems unhappy or disappointed.",
    "angry": "The person appears frustrated or upset.",
    "surprise": "The person looks shocked or amazed.",
    "fear": "The person seems scared or anxious.",
    "disgust": "The person looks displeased or uncomfortable.",
    "neutral": "The person has a calm and relaxed expression."
}

@app.route('/detect_emotion', methods=['POST'])
def detect_emotion():
    if 'image' not in request.files:
        return jsonify({"error": "No image provided"}), 400

    file = request.files['image']
    np_img = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

    if img is None:
        return jsonify({"error": "Invalid image"}), 400

    try:
        start_time = time.time()

        # DeepFace emotion detection
        result = DeepFace.analyze(
            img_path=img,
            actions=['emotion'],
            enforce_detection=False,
            detector_backend='mtcnn'
        )

        dominant = result[0]['dominant_emotion']
        explanation = emotion_explanations.get(
            dominant,
            "Emotion detected but no explanation available."
        )

        return jsonify({
            "emotion": dominant,
            "explanation": explanation,
            "processing_time_sec": round(time.time() - start_time, 2)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "DeepFace Emotion API is running!"})


if __name__ == '__main__':
    port = int(os.environ.get("PORT", 8000))
    app.run(host="0.0.0.0", port=port)
