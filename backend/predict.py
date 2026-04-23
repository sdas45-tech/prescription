"""
predict.py - Predict whether an image is a Prescription or Not using the trained VGG16 model.

Usage:
    python predict.py                          # Interactive mode - select images one by one
    python predict.py <image_path>             # Predict a single image
    python predict.py <directory_path>         # Predict all images in a directory
    python predict.py <image_path> --json      # Output as JSON
"""

import os
import sys
import json
import argparse
import numpy as np
from pathlib import Path

import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image as keras_image

# Fix Windows console encoding
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# ---------------------------------------------------------------
# Configuration (aligned with train.py)
# ---------------------------------------------------------------
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(BASE_DIR)
MODEL_DIR   = os.path.join(PROJECT_DIR, "model")
DEFAULT_MODEL_PATH = os.path.join(MODEL_DIR, "best_model.keras")
LABELS_PATH = os.path.join(MODEL_DIR, "class_labels.json")

IMG_SIZE    = (224, 224)
SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}


# ---------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------
def load_class_labels():
    if os.path.exists(LABELS_PATH):
        with open(LABELS_PATH) as f:
            return json.load(f)
    return {"0": "not_prescription", "1": "prescription"}


def preprocess_image(img_path):
    img = keras_image.load_img(img_path, target_size=IMG_SIZE)
    img_array = keras_image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array


def predict_single(model, img_path, labels):
    img_array = preprocess_image(img_path)
    prediction = model.predict(img_array, verbose=0)[0][0]

    if prediction >= 0.5:
        label = labels.get("1", "prescription")
        confidence = prediction
    else:
        label = labels.get("0", "not_prescription")
        confidence = 1 - prediction

    return label, float(confidence)


def predict_batch(model, dir_path, labels):
    results = []
    files = sorted(Path(dir_path).rglob("*"))
    for fp in files:
        if fp.suffix.lower() in SUPPORTED_EXTS:
            label, conf = predict_single(model, str(fp), labels)
            results.append({"file": str(fp), "label": label, "confidence": conf})
    return results


# ---------------------------------------------------------------
# Pretty Printing (ASCII-safe for Windows)
# ---------------------------------------------------------------
def print_result(file_path, label, confidence):
    tag = "[RX]" if label == "prescription" else "[--]"
    bar_filled = int(confidence * 30)
    bar = "#" * bar_filled + "." * (30 - bar_filled)
    print(f"  {tag}  {os.path.basename(file_path)}")
    print(f"      Prediction : {label.upper()}")
    print(f"      Confidence : {confidence:.2%}  [{bar}]")
    print()


# ---------------------------------------------------------------
# Interactive Mode - Select images one by one
# ---------------------------------------------------------------
def interactive_mode(model, labels):
    """Let the user pick images one by one from a directory listing."""
    print("\n" + "=" * 60)
    print("  INTERACTIVE MODE - Select images one by one")
    print("=" * 60)

    # Ask user for a directory to scan
    print("\n  Enter the path to a folder containing images")
    print("  (or press Enter to use the test dataset):\n")
    user_dir = input("  Folder path: ").strip()

    if not user_dir:
        user_dir = os.path.join(PROJECT_DIR, "classification_dataset", "test")
        print(f"  Using default: {user_dir}")

    if not os.path.isdir(user_dir):
        print(f"\n  [ERROR] Directory not found: {user_dir}")
        return

    # Collect all image files (recursively)
    all_images = []
    for fp in sorted(Path(user_dir).rglob("*")):
        if fp.suffix.lower() in SUPPORTED_EXTS:
            all_images.append(str(fp))

    if not all_images:
        print("\n  [WARNING] No supported images found in that directory.")
        return

    print(f"\n  Found {len(all_images)} image(s). Select one to predict:\n")

    # Main selection loop
    while True:
        # Display numbered list
        print("-" * 60)
        for i, img_path in enumerate(all_images, 1):
            # Show relative path for readability
            rel = os.path.relpath(img_path, user_dir)
            print(f"  {i:3d}. {rel}")

        print(f"\n  Enter a number (1-{len(all_images)}) to predict")
        print("  Enter 'q' to quit\n")

        choice = input("  Your choice: ").strip().lower()

        if choice in ("q", "quit", "exit"):
            print("\n  Goodbye!")
            break

        try:
            idx = int(choice) - 1
            if 0 <= idx < len(all_images):
                selected = all_images[idx]
                print(f"\n  Predicting: {os.path.basename(selected)}")
                print("-" * 40)
                label, conf = predict_single(model, selected, labels)
                print_result(selected, label, conf)
            else:
                print(f"\n  [ERROR] Please enter a number between 1 and {len(all_images)}")
        except ValueError:
            print("\n  [ERROR] Invalid input. Enter a number or 'q' to quit.")

        # Ask if they want to continue
        again = input("  Predict another? (y/n): ").strip().lower()
        if again in ("n", "no", "q", "quit"):
            print("\n  Goodbye!")
            break
        print()


# ---------------------------------------------------------------
# Main
# ---------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Predict Prescription vs Not-Prescription")
    parser.add_argument("input", nargs="?", default=None,
                        help="Path to an image file or directory (omit for interactive mode)")
    parser.add_argument("--model", default=DEFAULT_MODEL_PATH, help="Path to the trained .keras model")
    parser.add_argument("--json", action="store_true", help="Output results as JSON")
    args = parser.parse_args()

    # Resolve model path
    model_path = args.model
    if not os.path.exists(model_path):
        fallback = os.path.join(MODEL_DIR, "vgg16_prescription_final.keras")
        if os.path.exists(fallback):
            model_path = fallback
        else:
            print(f"[ERROR] Model not found: {model_path}")
            print("        Run train.py first to create a trained model.")
            sys.exit(1)

    print("=" * 60)
    print("  VGG16 Prescription Classifier - Prediction")
    print("=" * 60)
    print(f"  Model : {model_path}\n")
    print("  Loading model...")

    model = load_model(model_path)
    labels = load_class_labels()
    print("  Model loaded successfully!\n")

    # -----------------------------------------------------------
    # No input provided -> Interactive mode (select one by one)
    # -----------------------------------------------------------
    if args.input is None:
        interactive_mode(model, labels)
        return

    input_path = args.input

    # -----------------------------------------------------------
    # Single file prediction
    # -----------------------------------------------------------
    if os.path.isfile(input_path):
        label, conf = predict_single(model, input_path, labels)
        if args.json:
            print(json.dumps({"file": input_path, "label": label, "confidence": conf}, indent=2))
        else:
            print_result(input_path, label, conf)

    # -----------------------------------------------------------
    # Directory batch prediction
    # -----------------------------------------------------------
    elif os.path.isdir(input_path):
        results = predict_batch(model, input_path, labels)
        if not results:
            print("[WARNING] No supported images found in the directory.")
            sys.exit(0)

        if args.json:
            print(json.dumps(results, indent=2))
        else:
            print(f"  Found {len(results)} image(s)\n")
            prescription_count = 0
            for r in results:
                print_result(r["file"], r["label"], r["confidence"])
                if r["label"] == "prescription":
                    prescription_count += 1

            print("-" * 60)
            print(f"  Summary: {prescription_count}/{len(results)} predicted as PRESCRIPTION")
            print(f"           {len(results) - prescription_count}/{len(results)} predicted as NOT PRESCRIPTION")
    else:
        print(f"[ERROR] Path not found: {input_path}")
        sys.exit(1)

    print("\n[DONE] Prediction complete!")


if __name__ == "__main__":
    main()
