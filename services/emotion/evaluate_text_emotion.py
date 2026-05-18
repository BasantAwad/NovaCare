import sys
import os
import json
import time
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report
)

# Add project root and services to system path
sys.path.insert(0, r"c:\Users\Pc\NovaCare-1")
sys.path.insert(0, r"c:\Users\Pc\NovaCare-1\services\emotion")

from services.emotion.text_predictor import TextEmotionAnalyzer

def main():
    print("=" * 60)
    # 1. Initialize analyzer
    print("Initializing TextEmotionAnalyzer...")
    device = "cpu"
    analyzer = TextEmotionAnalyzer(use_pretrained=True, device=device)
    
    if analyzer.model is None:
        print("Error: Could not load text emotion model. Aborting.")
        return
        
    # 2. Load dataset
    print("\nLoading GoEmotions test dataset split...")
    from datasets import load_dataset
    try:
        # Load test split
        dataset = load_dataset("go_emotions", split="test")
        print(f"Loaded GoEmotions test split with {len(dataset)} samples.")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    # Select a slice of 200 samples for fast, reliable CPU evaluation
    n_samples = 200
    sub_dataset = dataset.select(range(n_samples))
    print(f"Selecting first {n_samples} samples for CPU evaluation...")

    texts = sub_dataset["text"]
    ground_truth_lists = sub_dataset["labels"] # List of label indices per sample
    
    # 3. Run evaluation
    print(f"\nRunning inference on {n_samples} samples on {device}...")
    predictions_fine = []
    predictions_basic = []
    ground_truth_fine_first = []
    ground_truth_basic_first = []
    
    latencies = []
    
    for i in range(n_samples):
        text = texts[i]
        gt_indices = ground_truth_lists[i]
        
        # Take the first label index as the primary category for single-label comparison
        gt_idx = gt_indices[0] if gt_indices else 27 # Default to neutral (27)
        gt_label_fine = analyzer.EMOTION_LABELS[gt_idx]
        gt_label_basic = analyzer.EMOTION_MAPPING.get(gt_label_fine, "neutral")
        
        ground_truth_fine_first.append(gt_label_fine)
        ground_truth_basic_first.append(gt_label_basic)
        
        # Measure latency
        start_time = time.perf_counter()
        pred_res = analyzer.predict(text, expand_slang=True)
        end_time = time.perf_counter()
        
        latencies.append((end_time - start_time) * 1000.0) # in ms
        
        predictions_fine.append(pred_res.get("fine_grained_emotion", "neutral"))
        predictions_basic.append(pred_res.get("emotion", "neutral"))
        
        if (i + 1) % 50 == 0:
            print(f"  Processed {i + 1}/{n_samples} samples...")

    avg_latency = sum(latencies) / len(latencies)
    fps = 1000.0 / avg_latency if avg_latency > 0 else 0.0
    print(f"\nInference completed in {sum(latencies)/1000.0:.2f}s total.")
    print(f"  Average Latency: {avg_latency:.2f} ms")
    print(f"  Inference Speed: {fps:.2f} FPS")

    # 4. Calculate metrics (Basic Categories - 7 Mapped Classes)
    basic_classes = ["happy", "sad", "angry", "fear", "disgust", "surprise", "neutral"]
    
    y_true_basic = ground_truth_basic_first
    y_pred_basic = predictions_basic
    
    # Standard metrics
    accuracy = accuracy_score(y_true_basic, y_pred_basic)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true_basic, y_pred_basic, average="weighted", zero_division=0
    )
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true_basic, y_pred_basic, average="macro", zero_division=0
    )
    
    print("\n" + "=" * 60)
    print("RECONSTRUCTED METRICS (7 Basic Categories)")
    print("=" * 60)
    print(f"Accuracy:        {accuracy * 100:.2f}%")
    print(f"Precision (W):   {precision * 100:.2f}%")
    print(f"Recall (W):      {recall * 100:.2f}%")
    print(f"F1-Score (W):    {f1 * 100:.2f}%")
    print(f"F1-Score (M):    {f1_macro * 100:.2f}%")
    print("=" * 60)
    
    # Save CSV metrics
    metrics_data = {
        "Metric": ["Accuracy", "Weighted Precision", "Weighted Recall", "Weighted F1-Score", "Macro F1-Score", "Avg Latency (ms)", "Throughput (FPS)"],
        "Value": [accuracy, precision, recall, f1, f1_macro, avg_latency, fps]
    }
    df = pd.DataFrame(metrics_data)
    csv_path = r"C:\Users\Pc\.gemini\antigravity\brain\331ab3b0-b2e1-44dd-b5fe-05c56a606d80\artifacts\text_emotion_metrics.csv"
    df.to_csv(csv_path, index=False)
    print(f"Saved CSV metrics to: {csv_path}")
    
    # Save classification report
    report = classification_report(y_true_basic, y_pred_basic, target_names=None, zero_division=0)
    report_path = r"C:\Users\Pc\.gemini\antigravity\brain\331ab3b0-b2e1-44dd-b5fe-05c56a606d80\artifacts\text_emotion_classification_report.txt"
    with open(report_path, "w") as f:
        f.write(report)
    print(f"Saved classification report to: {report_path}")
    
    # 5. Plot confusion matrix
    cm = confusion_matrix(y_true_basic, y_pred_basic, labels=basic_classes)
    
    # Normalize by row totals
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    cm_norm = np.nan_to_num(cm_norm) # Replace division-by-zero NaNs with 0
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm_norm,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        xticklabels=basic_classes,
        yticklabels=basic_classes,
        vmin=0,
        vmax=1
    )
    plt.xlabel("Predicted Emotion")
    plt.ylabel("True Emotion")
    plt.title("Reconstructed Confusion Matrix (7 Basic Mapped Categories)")
    plt.tight_layout()
    
    png_path = r"C:\Users\Pc\.gemini\antigravity\brain\331ab3b0-b2e1-44dd-b5fe-05c56a606d80\artifacts\text_emotion_confusion_matrix.png"
    plt.savefig(png_path, dpi=150)
    plt.close()
    print(f"Saved Confusion Matrix PNG to: {png_path}")
    print("\n[SUCCESS] Metric reconstruction successfully complete!")

if __name__ == "__main__":
    main()
