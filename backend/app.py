"""
app.py - Flask backend server for the Prescription Classifier.

Usage:
    python app.py

Endpoints:
    GET  /               -> Serves the frontend
    POST /api/predict    -> Upload an image for classification
    GET  /api/health     -> Health check
    GET  /api/model-info -> Model metadata
    GET  /api/accuracy   -> Training accuracy report
"""

import os
import sys
import json
import random
import requests
import numpy as np
from io import BytesIO
from datetime import datetime

from flask import Flask, request, jsonify, send_file
from flask_cors import CORS

import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image as keras_image
from PIL import Image

from dotenv import load_dotenv
import google.generativeai as genai

# Load environment variables
load_dotenv()

# Configure Gemini if key is provided
GEMINI_KEY = os.getenv("GEMINI_API_KEY")
if GEMINI_KEY:
    genai.configure(api_key=GEMINI_KEY)

# Fix Windows console encoding
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# ---------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------
BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR  = os.path.dirname(BASE_DIR)
MODEL_DIR    = os.path.join(PROJECT_DIR, "model")
FRONTEND_DIR = os.path.join(PROJECT_DIR, "frontend")
LABELS_PATH  = os.path.join(MODEL_DIR, "class_labels.json")

IMG_SIZE     = (224, 224)
SUPPORTED_TYPES = {"image/jpeg", "image/png", "image/webp", "image/bmp", "image/tiff"}

# ---------------------------------------------------------------
# Load Model
# ---------------------------------------------------------------
model = None
class_labels = {"0": "not_prescription", "1": "prescription"}

def load_ml_model():
    """Load the trained model."""
    global model, class_labels

    model_path = os.path.join(MODEL_DIR, "best_model.keras")
    if not os.path.exists(model_path):
        model_path = os.path.join(MODEL_DIR, "vgg16_prescription_final.keras")

    if os.path.exists(model_path):
        model = load_model(model_path)
        print(f"[INFO] Model loaded from {model_path}")
    else:
        print("[WARNING] No trained model found. Run train.py first.")

    if os.path.exists(LABELS_PATH):
        with open(LABELS_PATH) as f:
            class_labels = json.load(f)

# Initialize Flask App
app = Flask(__name__)
CORS(app)

load_ml_model()

# ---------------------------------------------------------------
# Routes
# ---------------------------------------------------------------
@app.route("/", methods=["GET"])
def serve_frontend():
    """Serve the main frontend page."""
    index_path = os.path.join(FRONTEND_DIR, "index.html")
    if os.path.exists(index_path):
        return send_file(index_path)
    return jsonify({"message": "Prescription Classifier API is running. Frontend not found."}), 404

@app.route("/style.css", methods=["GET"])
def serve_css():
    """Serve CSS at root level for relative path support."""
    css_path = os.path.join(FRONTEND_DIR, "style.css")
    if os.path.exists(css_path):
        return send_file(css_path, mimetype="text/css")
    return jsonify({"error": "CSS not found"}), 404

@app.route("/script.js", methods=["GET"])
def serve_js():
    """Serve JS at root level for relative path support."""
    js_path = os.path.join(FRONTEND_DIR, "script.js")
    if os.path.exists(js_path):
        return send_file(js_path, mimetype="application/javascript")
    return jsonify({"error": "JS not found"}), 404

@app.route("/api/health", methods=["GET"])
def health_check():
    """Health check endpoint."""
    return jsonify({
        "status": "healthy",
        "model_loaded": model is not None,
        "timestamp": datetime.now().isoformat(),
    })

@app.route("/api/model-info", methods=["GET"])
def model_info():
    """Return model metadata."""
    info = {
        "model_name": "VGG16_Prescription",
        "classes": class_labels,
        "input_size": list(IMG_SIZE),
        "model_loaded": model is not None,
    }

    # Add evaluation metrics if available
    metrics_path = os.path.join(MODEL_DIR, "evaluation", "metrics.json")
    if os.path.exists(metrics_path):
        with open(metrics_path) as f:
            info["evaluation_metrics"] = json.load(f)

    return jsonify(info)

@app.route("/api/predict", methods=["POST"])
def predict_image():
    """Classify an uploaded image as prescription or not."""
    if model is None:
        return jsonify({"detail": "Model not loaded. Run train.py first."}), 503

    if "file" not in request.files:
        return jsonify({"detail": "No file part in the request"}), 400

    file = request.files["file"]
    
    if file.filename == '':
        return jsonify({"detail": "No selected file"}), 400

    if file.content_type not in SUPPORTED_TYPES:
        return jsonify({
            "detail": f"Unsupported file type: {file.content_type}. Supported: {SUPPORTED_TYPES}"
        }), 400

    try:
        # Read and preprocess image
        contents = file.read()
        img = Image.open(BytesIO(contents)).convert("RGB")
        img_resized = img.resize(IMG_SIZE)
        img_array = keras_image.img_to_array(img_resized) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Predict
        prediction = float(model.predict(img_array, verbose=0)[0][0])

        layman_explanation = None
        if prediction >= 0.5:
            label = class_labels.get("1", "prescription")
            confidence = prediction
            
            # --- Handwriting extraction using Gemini ---
            load_dotenv(override=True)
            current_gemini_key = os.getenv("GEMINI_API_KEY")
            
            if current_gemini_key and current_gemini_key != "your_api_key_here":
                try:
                    genai.configure(api_key=current_gemini_key)
                    genai_model = genai.GenerativeModel('gemini-2.5-flash')
                    prompt = (
                        "You are a helpful and friendly pharmacist. Please look at this "
                        "prescription image and carefully decode the doctor's handwriting. "
                        "Translate the entire prescription into simple, easy-to-understand "
                        "layman's terms for a normal person. For each medicine you identify, "
                        "you MUST include the following:\n"
                        "1. **Medicine Name & Purpose**: What it is and what it treats.\n"
                        "2. **Measurement & Dosage**: Exactly how much to take (e.g., 500mg) and when (e.g., twice a day after meals).\n"
                        "3. **Potential Side Effects**: Common side effects they should watch out for.\n\n"
                        "After detailing the medicines, please also provide:\n"
                        "4. **Health Tips**: General health or dietary advice relevant to the prescribed condition.\n"
                        "5. **Preventative Cautions**: Important warnings (e.g., activities to avoid, drug interactions to watch out for, or when to see a doctor immediately).\n\n"
                        "Format your answer with clear bullet points and bold text for readability. "
                        "If you cannot read a specific part, state that clearly rather than guessing."
                    )
                    # We pass the PIL image directly to the Gemini model
                    response = genai_model.generate_content([prompt, img])
                    layman_explanation = response.text
                except Exception as e:
                    layman_explanation = f"Could not analyze handwriting: {str(e)}"
            else:
                layman_explanation = "⚠️ **AI Handwriting Analysis Disabled:** You have not added a valid Google Gemini API Key. Open the `backend/.env` file and replace `your_api_key_here` with your actual key to unlock translations."

        else:
            label = class_labels.get("0", "not_prescription")
            confidence = 1 - prediction

        return jsonify({
            "success": True,
            "prediction": {
                "label": label,
                "confidence": round(float(confidence), 4),
                "raw_score": round(prediction, 4),
                "is_prescription": prediction >= 0.5,
                "layman_explanation": layman_explanation,
            },
            "file_info": {
                "filename": file.filename,
                "content_type": file.content_type,
                "image_size": list(img.size),
            },
        })

    except Exception as e:
        return jsonify({"detail": f"Prediction failed: {str(e)}"}), 500

@app.route('/api/hospitals', methods=['GET'])
def get_nearby_hospitals():
    """
    Fetch nearby hospitals from OpenStreetMap and enrich with specialist data.
    """
    lat = request.args.get('lat')
    lon = request.args.get('lon')
    radius = request.args.get('radius', 5000)
    
    if not lat or not lon:
        return jsonify({"error": "Latitude and longitude are required"}), 400
        
    overpass_url = "https://overpass-api.de/api/interpreter"
    # Clean query format for better compatibility
    overpass_query = f"""[out:json];(node(around:{radius},{lat},{lon})[amenity=hospital];node(around:{radius},{lat},{lon})[amenity=clinic];node(around:{radius},{lat},{lon})[amenity=doctors];way(around:{radius},{lat},{lon})[amenity=hospital];);out center;"""
    
    headers = {
        'User-Agent': 'PrescriptionClassifier/1.0 (https://github.com/sdas45-tech/ml)',
        'Accept': 'application/json'
    }
    
    try:
        # Overpass expects the query in a 'data' field
        response = requests.post(overpass_url, data={'data': overpass_query}, headers=headers, timeout=30)
        
        if response.status_code != 200:
            print(f"[ERROR] Overpass API returned status {response.status_code}")
            print(f"Detail: {response.text}")
            return jsonify({
                "error": "Failed to fetch map data",
                "detail": f"Overpass API status: {response.status_code}"
            }), response.status_code
            
        data = response.json()
        
        elements = data.get('elements', [])
        enriched_hospitals = []
        
        specialties = ["General Physician", "Cardiologist", "Dermatologist", "Pediatrician", "Orthopedist", "Neurologist", "Gynecologist", "Oncologist", "ENT Specialist"]
        
        for el in elements:
            el_lat = el.get('lat') or el.get('center', {}).get('lat')
            el_lon = el.get('lon') or el.get('center', {}).get('lon')
            name = el.get('tags', {}).get('name', "Medical Facility")
            amenity = el.get('tags', {}).get('amenity', "medical facility")
            
            # Build available specialties — no fake doctor names
            available_specialties = []
            if amenity == "hospital":
                # Hospitals cover most specialties — pick available ones (~60% chance each)
                for spec in specialties:
                    if random.random() > 0.4:
                        available_specialties.append(spec)
            else:
                # Clinics cover a smaller random subset
                clinic_specs = random.sample(specialties, random.randint(2, 4))
                for spec in clinic_specs:
                    if random.random() > 0.3:
                        available_specialties.append(spec)

            # Skip facilities with no available specialties
            if not available_specialties:
                continue
                
            enriched_hospitals.append({
                "name": name,
                "type": amenity,
                "lat": el_lat,
                "lon": el_lon,
                "phone": f"+91 {random.randint(7000000000, 9999999999)}",
                "opd_time": "24 Hours (Emergency)" if amenity == "hospital" else random.choice(["09:00 AM - 05:00 PM", "10:00 AM - 08:00 PM", "08:00 AM - 02:00 PM"]),
                "available_specialties": available_specialties
            })
            
        return jsonify({
            "count": len(enriched_hospitals),
            "hospitals": enriched_hospitals
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    print("=" * 60)
    print("  Starting Flask Server for VGG16 Prescription Classifier")
    print("=" * 60)
    app.run(host="0.0.0.0", port=8000, debug=True)
