from flask import Flask, request, jsonify
from deepface import DeepFace
import os

app = Flask(__name__)

# Ensure these directories exist
os.makedirs("/opt/render/.deepface/weights", exist_ok=True)

@app.route("/detect_emotion", methods=["POST"])
def detect_emotion():
    # Check if file part exists
    if "image" not in request.files:
        return jsonify({"error": "No file part in the request"}), 400
    
    file = request.files["image"]
    
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400
    
    try:
        # Save the file temporarily
        file_path = os.path.join("/tmp", file.filename)
        file.save(file_path)
        
        # Analyze emotion
        result = DeepFace.analyze(img_path=file_path, actions=["emotion"])
        
        # Delete temp file if needed
        os.remove(file_path)
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 10000)))
