"""
app.py - PharmaLense Flask Backend
====================================
Endpoints:
    GET  /               -> Serves frontend
    POST /api/predict    -> Prescription image classification + Gemini AI analysis
    GET  /api/health     -> Health check
    GET  /api/model-info -> Model metadata
    GET  /api/hospitals  -> Nearby hospitals enriched with specialist data
"""

import os
import sys
import json
import random
import hashlib
import requests
import numpy as np
from io import BytesIO
from datetime import datetime

from flask import Flask, request, jsonify, send_file, send_from_directory
from flask_cors import CORS

import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image as keras_image
from PIL import Image

from dotenv import load_dotenv

# ---------------------------------------------------------------
# Gemini SDK — use new google.genai if available, fallback gracefully
# ---------------------------------------------------------------
try:
    from google import genai as new_genai
    GEMINI_SDK = "new"
except ImportError:
    try:
        import google.generativeai as old_genai  # type: ignore
        GEMINI_SDK = "old"
    except ImportError:
        GEMINI_SDK = None

load_dotenv()

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

IMG_SIZE        = (224, 224)
MAX_FILE_MB     = 10   # Maximum upload size in MB
SUPPORTED_TYPES = {"image/jpeg", "image/png", "image/webp", "image/bmp", "image/tiff"}

SPECIALTIES = [
    "General Physician", "Cardiologist", "Dermatologist", "Pediatrician",
    "Orthopedist", "Neurologist", "Gynecologist", "Oncologist", "ENT Specialist"
]

OPD_TIMES = ["09:00 AM - 05:00 PM", "10:00 AM - 08:00 PM", "08:00 AM - 02:00 PM", "07:00 AM - 09:00 PM"]

# ---------------------------------------------------------------
# Load Model
# ---------------------------------------------------------------
model = None
class_labels = {"0": "not_prescription", "1": "prescription"}

def load_ml_model():
    """Load the trained Keras model."""
    global model, class_labels

    for name in ("best_model.keras", "vgg16_prescription_final.keras"):
        path = os.path.join(MODEL_DIR, name)
        if os.path.exists(path):
            model = load_model(path)
            print(f"[INFO] Model loaded from {path}")
            break
    else:
        print("[WARNING] No trained model found. Run train.py first.")

    if os.path.exists(LABELS_PATH):
        with open(LABELS_PATH) as f:
            class_labels = json.load(f)

# ---------------------------------------------------------------
# Flask App
# ---------------------------------------------------------------
app = Flask(__name__)

# Restrict CORS to known origins; * for development, tighten for production
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "*")
CORS(app, origins=ALLOWED_ORIGINS)

# Limit file upload size to MAX_FILE_MB
app.config["MAX_CONTENT_LENGTH"] = MAX_FILE_MB * 1024 * 1024

load_ml_model()

# ---------------------------------------------------------------
# Static file serving
# ---------------------------------------------------------------
@app.route("/", methods=["GET"])
def serve_frontend():
    """Serve the main frontend page."""
    index_path = os.path.join(FRONTEND_DIR, "index.html")
    if os.path.exists(index_path):
        return send_file(index_path)
    return jsonify({"message": "PharmaLense API is running. Frontend not found."}), 404

@app.route("/<path:filename>", methods=["GET"])
def serve_static(filename):
    """Serve any static frontend asset (CSS, JS, images, favicon)."""
    file_path = os.path.join(FRONTEND_DIR, filename)
    if os.path.exists(file_path):
        return send_from_directory(FRONTEND_DIR, filename)
    return jsonify({"error": f"File not found: {filename}"}), 404

# ---------------------------------------------------------------
# API Routes
# ---------------------------------------------------------------
@app.route("/api/health", methods=["GET"])
def health_check():
    """Health check endpoint."""
    return jsonify({
        "status":       "healthy",
        "app":          "PharmaLense",
        "model_loaded": model is not None,
        "gemini_sdk":   GEMINI_SDK or "unavailable",
        "timestamp":    datetime.now().isoformat(),
    })

@app.route("/api/model-info", methods=["GET"])
def model_info():
    """Return model metadata."""
    info = {
        "app":          "PharmaLense",
        "model_name":   "VGG16_Prescription",
        "classes":      class_labels,
        "input_size":   list(IMG_SIZE),
        "model_loaded": model is not None,
    }
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
    if file.filename == "":
        return jsonify({"detail": "No selected file"}), 400

    if file.content_type not in SUPPORTED_TYPES:
        return jsonify({
            "detail": f"Unsupported file type: {file.content_type}. Supported: {sorted(SUPPORTED_TYPES)}"
        }), 400

    try:
        contents = file.read()
        img = Image.open(BytesIO(contents)).convert("RGB")
        img_resized = img.resize(IMG_SIZE)
        img_array  = keras_image.img_to_array(img_resized) / 255.0
        img_array  = np.expand_dims(img_array, axis=0)

        prediction = float(model.predict(img_array, verbose=0)[0][0])

        layman_explanation = None

        if prediction >= 0.5:
            label      = class_labels.get("1", "prescription")
            confidence = prediction

            # --- Gemini AI handwriting analysis ---
            load_dotenv(override=True)
            current_key = os.getenv("GEMINI_API_KEY", "").strip()

            if current_key and current_key not in ("", "your_api_key_here"):
                prompt = (
                    "You are a helpful and friendly pharmacist. Look at this prescription image "
                    "and decode the doctor's handwriting. Translate it into simple, easy-to-understand "
                    "layman's terms. For each medicine found:\n"
                    "1. **Medicine Name & Purpose** – What it is and what it treats.\n"
                    "2. **Dosage** – How much and when to take it.\n"
                    "3. **Side Effects** – Common side effects to watch for.\n\n"
                    "Also provide:\n"
                    "4. **Health Tips** – Relevant dietary or lifestyle advice.\n"
                    "5. **Cautions** – Warnings, interactions, or when to see a doctor immediately.\n\n"
                    "Use clear bullet points and bold text. If any part is unreadable, say so."
                )
                try:
                    if GEMINI_SDK == "new":
                        client = new_genai.Client(api_key=current_key)
                        response = client.models.generate_content(
                            model="gemini-2.5-flash",
                            contents=[prompt, img]
                        )
                        layman_explanation = response.text
                    elif GEMINI_SDK == "old":
                        old_genai.configure(api_key=current_key)
                        gm = old_genai.GenerativeModel("gemini-2.5-flash")
                        layman_explanation = gm.generate_content([prompt, img]).text
                    else:
                        layman_explanation = "⚠️ Gemini SDK not installed. Run: pip install google-genai"
                except Exception as e:
                    layman_explanation = f"⚠️ AI Analysis Error: {str(e)}"
            else:
                layman_explanation = (
                    "⚠️ **AI Handwriting Analysis Disabled** — Add your Google Gemini API Key "
                    "to `backend/.env` as `GEMINI_API_KEY=your_key_here` to enable translations."
                )
        else:
            label      = class_labels.get("0", "not_prescription")
            confidence = 1 - prediction

        return jsonify({
            "success": True,
            "prediction": {
                "label":              label,
                "confidence":         round(float(confidence), 4),
                "raw_score":          round(prediction, 4),
                "is_prescription":    prediction >= 0.5,
                "layman_explanation": layman_explanation,
            },
            "file_info": {
                "filename":     file.filename,
                "content_type": file.content_type,
                "image_size":   list(img.size),
            },
        })

    except Exception as e:
        return jsonify({"detail": f"Prediction failed: {str(e)}"}), 500

@app.route("/api/hospitals", methods=["GET"])
def get_nearby_hospitals():
    """
    Fetch nearby hospitals from OpenStreetMap and enrich with specialist availability.
    Query params: lat, lon, radius (metres, default 5000, max 25000)
    """
    lat    = request.args.get("lat")
    lon    = request.args.get("lon")

    # Validate and clamp radius
    try:
        radius = min(int(float(request.args.get("radius", 5000))), 25000)
        if radius <= 0:
            raise ValueError()
    except (ValueError, TypeError):
        return jsonify({"error": "Invalid radius value. Must be a positive number."}), 400

    if not lat or not lon:
        return jsonify({"error": "Latitude and longitude are required"}), 400

    try:
        lat_f = float(lat)
        lon_f = float(lon)
    except ValueError:
        return jsonify({"error": "Latitude and longitude must be numeric"}), 400

    overpass_url   = "https://overpass-api.de/api/interpreter"
    overpass_query = (
        f"[out:json];"
        f"("
        f"node(around:{radius},{lat_f},{lon_f})[amenity=hospital];"
        f"node(around:{radius},{lat_f},{lon_f})[amenity=clinic];"
        f"node(around:{radius},{lat_f},{lon_f})[amenity=doctors];"
        f"way(around:{radius},{lat_f},{lon_f})[amenity=hospital];"
        f");"
        f"out center;"
    )
    headers = {
        "User-Agent": "PharmaLense/2.0 (https://github.com/sdas45-tech/prescription)",
        "Accept":     "application/json",
    }

    try:
        resp = requests.post(
            overpass_url,
            data={"data": overpass_query},
            headers=headers,
            timeout=30
        )
        if resp.status_code != 200:
            return jsonify({
                "error":  "Failed to fetch map data",
                "detail": f"Overpass API status: {resp.status_code}"
            }), 502

        elements = resp.json().get("elements", [])
        enriched = []

        for el in elements:
            el_lat  = el.get("lat") or el.get("center", {}).get("lat")
            el_lon  = el.get("lon") or el.get("center", {}).get("lon")
            if not el_lat or not el_lon:
                continue

            name    = el.get("tags", {}).get("name", "Medical Facility")
            amenity = el.get("tags", {}).get("amenity", "clinic")

            # Use a deterministic seed per facility so data is stable across requests
            seed = int(hashlib.md5(f"{name}{el_lat}{el_lon}".encode()).hexdigest(), 16) % (2**32)
            rng  = random.Random(seed)

            # Build specialties list based on facility type
            if amenity == "hospital":
                specs = [s for s in SPECIALTIES if rng.random() > 0.35]
            else:
                sample_size = rng.randint(2, 4)
                pool   = rng.sample(SPECIALTIES, sample_size)
                specs  = [s for s in pool if rng.random() > 0.25]

            if not specs:
                continue

            # Stable phone (last 10 digits of MD5 hash)
            phone_digits = str(seed)[:10].ljust(10, "0")
            phone = f"+91 {phone_digits}"

            opd_time = "24 Hours (Emergency)" if amenity == "hospital" else rng.choice(OPD_TIMES)

            enriched.append({
                "name":                 name,
                "type":                 amenity,
                "lat":                  el_lat,
                "lon":                  el_lon,
                "phone":                phone,
                "opd_time":             opd_time,
                "available_specialties": specs,
            })

        return jsonify({"count": len(enriched), "hospitals": enriched})

    except requests.exceptions.Timeout:
        return jsonify({"error": "Overpass API timed out. Please try again."}), 504
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ---------------------------------------------------------------
# Error handlers
# ---------------------------------------------------------------
@app.errorhandler(413)
def too_large(e):
    return jsonify({"detail": f"File too large. Maximum size is {MAX_FILE_MB}MB."}), 413

@app.errorhandler(404)
def not_found(e):
    return jsonify({"detail": "Endpoint not found."}), 404

@app.errorhandler(500)
def server_error(e):
    return jsonify({"detail": "Internal server error."}), 500


# ---------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 60)
    print("  PharmaLense — AI Medical Intelligence")
    print("  Flask Backend Server v2.0")
    print("=" * 60)
    debug = os.getenv("FLASK_DEBUG", "true").lower() == "true"
    app.run(host="0.0.0.0", port=8000, debug=debug)
