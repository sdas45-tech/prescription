"""
evaluate.py - Evaluate the trained VGG16 prescription classifier on the test set.

Usage:
    python evaluate.py
    python evaluate.py --model ../model/best_model.keras
    python evaluate.py --test-dir ../classification_dataset/test

Output:
    model/evaluation/classification_report.txt
    model/evaluation/metrics.json
    model/evaluation/confusion_matrix.png
    model/evaluation/roc_curve.png
    model/evaluation/per_class_accuracy.png
    model/evaluation/score_distribution.png
"""

import os
import sys
import json
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_curve,
    auc,
    accuracy_score,
    precision_recall_fscore_support,
)

# Fix Windows console encoding
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# ---------------------------------------------------------------
# Configuration (aligned with train.py)
# ---------------------------------------------------------------
BASE_DIR        = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR     = os.path.dirname(BASE_DIR)
MODEL_DIR       = os.path.join(PROJECT_DIR, "model")
DATASET_DIR     = os.path.join(PROJECT_DIR, "classification_dataset")
TEST_DIR        = os.path.join(DATASET_DIR, "test")
EVAL_DIR        = os.path.join(MODEL_DIR, "evaluation")

DEFAULT_MODEL   = os.path.join(MODEL_DIR, "best_model.keras")
LABELS_PATH     = os.path.join(MODEL_DIR, "class_labels.json")

IMG_SIZE        = (224, 224)
BATCH_SIZE      = 16
CLASS_NAMES     = ["not_prescription", "prescription"]


def load_class_labels():
    if os.path.exists(LABELS_PATH):
        with open(LABELS_PATH) as f:
            return json.load(f)
    return {"0": "not_prescription", "1": "prescription"}


def build_test_generator(test_dir):
    test_datagen = ImageDataGenerator(rescale=1.0 / 255)
    test_gen = test_datagen.flow_from_directory(
        test_dir,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="binary",
        classes=CLASS_NAMES,
        shuffle=False,
    )
    return test_gen


# ---------------------------------------------------------------
# Evaluation Plots
# ---------------------------------------------------------------
def plot_confusion_matrix(y_true, y_pred, save_path):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
        linewidths=0.5, linecolor="gray", annot_kws={"size": 16},
    )
    plt.title("Confusion Matrix", fontsize=16, fontweight="bold")
    plt.xlabel("Predicted Label", fontsize=13)
    plt.ylabel("True Label", fontsize=13)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  [PLOT] Confusion matrix saved -> {save_path}")


def plot_roc_curve(y_true, y_scores, save_path):
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color="#3b82f6", linewidth=2.5, label=f"ROC Curve (AUC = {roc_auc:.4f})")
    plt.plot([0, 1], [0, 1], color="gray", linestyle="--", linewidth=1)
    plt.fill_between(fpr, tpr, alpha=0.1, color="#3b82f6")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate", fontsize=13)
    plt.ylabel("True Positive Rate", fontsize=13)
    plt.title("ROC Curve", fontsize=16, fontweight="bold")
    plt.legend(loc="lower right", fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  [PLOT] ROC curve saved -> {save_path}")
    return roc_auc


def plot_per_class_accuracy(y_true, y_pred, save_path):
    cm = confusion_matrix(y_true, y_pred)
    per_class_acc = cm.diagonal() / cm.sum(axis=1)

    colors = ["#ef4444", "#22c55e"]
    plt.figure(figsize=(8, 5))
    bars = plt.bar(CLASS_NAMES, per_class_acc, color=colors, edgecolor="white", linewidth=1.5)
    for bar, acc in zip(bars, per_class_acc):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                 f"{acc:.2%}", ha="center", fontsize=13, fontweight="bold")
    plt.ylim(0, 1.15)
    plt.title("Per-Class Accuracy", fontsize=16, fontweight="bold")
    plt.ylabel("Accuracy", fontsize=13)
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  [PLOT] Per-class accuracy chart saved -> {save_path}")


def plot_prediction_distribution(y_scores, save_path):
    plt.figure(figsize=(8, 5))
    plt.hist(y_scores, bins=50, color="#8b5cf6", edgecolor="white", alpha=0.85)
    plt.axvline(x=0.5, color="red", linestyle="--", linewidth=1.5, label="Threshold (0.5)")
    plt.title("Prediction Score Distribution", fontsize=16, fontweight="bold")
    plt.xlabel("Prediction Score (-> Prescription)", fontsize=13)
    plt.ylabel("Count", fontsize=13)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  [PLOT] Score distribution saved -> {save_path}")


# ---------------------------------------------------------------
# Main
# ---------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Evaluate VGG16 Prescription Classifier")
    parser.add_argument("--model",    default=DEFAULT_MODEL, help="Path to the trained .keras model")
    parser.add_argument("--test-dir", default=TEST_DIR,      help="Path to the test dataset directory")
    args = parser.parse_args()

    model_path = args.model
    if not os.path.exists(model_path):
        fallback = os.path.join(MODEL_DIR, "vgg16_prescription_final.keras")
        if os.path.exists(fallback):
            model_path = fallback
        else:
            print(f"[ERROR] Model not found: {model_path}")
            print("        Run train.py first.")
            sys.exit(1)

    os.makedirs(EVAL_DIR, exist_ok=True)

    print("=" * 60)
    print("  VGG16 Prescription Classifier - Evaluation")
    print("=" * 60)
    print(f"  Model    : {model_path}")
    print(f"  Test Dir : {args.test_dir}\n")

    model = load_model(model_path)

    test_gen = build_test_generator(args.test_dir)
    print(f"  [INFO] Test samples : {test_gen.samples}")
    print(f"  [INFO] Classes      : {test_gen.class_indices}\n")

    # Predict
    y_scores = model.predict(test_gen, verbose=1).flatten()
    y_pred = (y_scores >= 0.5).astype(int)
    y_true = test_gen.classes

    # Metrics
    acc = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary")

    print("\n" + "-" * 60)
    print("  EVALUATION RESULTS")
    print("-" * 60)
    print(f"  Overall Accuracy  : {acc:.4f}  ({acc:.2%})")
    print(f"  Precision         : {precision:.4f}")
    print(f"  Recall            : {recall:.4f}")
    print(f"  F1 Score          : {f1:.4f}")
    print("-" * 60)

    report = classification_report(y_true, y_pred, target_names=CLASS_NAMES, digits=4)
    print("\n  Classification Report:\n")
    print(report)

    # Save text report
    report_path = os.path.join(EVAL_DIR, "classification_report.txt")
    with open(report_path, "w") as f:
        f.write("VGG16 Prescription Classifier - Evaluation Report\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Model      : {model_path}\n")
        f.write(f"Test Dir   : {args.test_dir}\n")
        f.write(f"Samples    : {test_gen.samples}\n\n")
        f.write(f"Overall Accuracy  : {acc:.4f}  ({acc:.2%})\n")
        f.write(f"Precision         : {precision:.4f}\n")
        f.write(f"Recall            : {recall:.4f}\n")
        f.write(f"F1 Score          : {f1:.4f}\n\n")
        f.write("Classification Report:\n")
        f.write(report)
    print(f"\n  [FILE] Report saved -> {report_path}")

    # Save metrics JSON
    metrics = {
        "accuracy": float(acc),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "total_samples": int(test_gen.samples),
        "model": model_path,
        "test_dir": args.test_dir,
    }
    metrics_path = os.path.join(EVAL_DIR, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"  [FILE] Metrics JSON saved -> {metrics_path}")

    # Generate plots
    print("\n  Generating evaluation plots...\n")
    plot_confusion_matrix(y_true, y_pred, os.path.join(EVAL_DIR, "confusion_matrix.png"))
    roc_auc = plot_roc_curve(y_true, y_scores, os.path.join(EVAL_DIR, "roc_curve.png"))
    plot_per_class_accuracy(y_true, y_pred, os.path.join(EVAL_DIR, "per_class_accuracy.png"))
    plot_prediction_distribution(y_scores, os.path.join(EVAL_DIR, "score_distribution.png"))

    metrics["roc_auc"] = float(roc_auc)
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print("\n" + "=" * 60)
    print(f"  [DONE] Evaluation complete!  AUC = {roc_auc:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
