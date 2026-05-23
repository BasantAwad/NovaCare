import os
import sys
import time
import argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Add services/asl/ to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import data_config, CHECKPOINT_DIR
from models import create_model

def load_model(checkpoint_path: str, device: str = "cpu") -> tuple:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model_type = checkpoint.get("config", {}).get("model_type", "attention")
    model = create_model(model_type)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model, checkpoint

def apply_jitter(X: np.ndarray, std: float) -> np.ndarray:
    """Add Gaussian coordinate noise to landmarks"""
    if std == 0:
        return X.copy()
    noise = np.random.normal(0, std, X.shape)
    return X + noise

def apply_rotation(X: np.ndarray, angle_deg: float) -> np.ndarray:
    """Rotate landmarks around z-axis in image plane"""
    if angle_deg == 0:
        return X.copy()
    
    angle = np.radians(angle_deg)
    cos_a, sin_a = np.cos(angle), np.sin(angle)
    R = np.array([
        [cos_a, -sin_a, 0],
        [sin_a, cos_a, 0],
        [0, 0, 1]
    ], dtype=np.float32)
    
    X_rot = X.reshape(-1, 21, 3).copy()
    for i in range(len(X_rot)):
        X_rot[i] = X_rot[i] @ R.T
    return X_rot.reshape(-1, 63)

def apply_occlusion(X: np.ndarray, num_joints_dropped: int) -> np.ndarray:
    """Simulate joint occlusions by setting coordinates to zero"""
    if num_joints_dropped == 0:
        return X.copy()
    
    X_occ = X.reshape(-1, 21, 3).copy()
    num_samples = len(X_occ)
    
    for i in range(num_samples):
        # Randomly choose joints to drop
        drop_indices = np.random.choice(21, num_joints_dropped, replace=False)
        X_occ[i, drop_indices] = 0.0 # Set coordinates to zero
        
    return X_occ.reshape(-1, 63)

def evaluate_stress(model, X: np.ndarray, y: np.ndarray, device: str) -> float:
    X_tensor = torch.FloatTensor(X).to(device)
    y_tensor = torch.LongTensor(y).to(device)
    
    with torch.no_grad():
        logits = model(X_tensor)
        preds = logits.argmax(dim=-1)
        acc = (preds == y_tensor).float().mean().item()
    return acc * 100.0

def main():
    print("=" * 60)
    print("ASL LANDMARK ROBUSTNESS STRESS TESTER")
    print("=" * 60)
    
    parser = argparse.ArgumentParser(description="Stress Test ASL Landmark Classifier")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=r"services/asl/checkpoints/best_model.pt",
        help="Path to model checkpoint"
    )
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()
    
    if not os.path.exists(args.checkpoint):
        print(f"[ERROR] Checkpoint not found: {args.checkpoint}")
        return
        
    # Load model and data
    model, checkpoint = load_model(args.checkpoint, args.device)
    print(f"Loaded model from {args.checkpoint} (Epoch: {checkpoint.get('epoch', 'N/A')})")
    
    data_dir = r"services/asl/data/processed"
    X_test_path = os.path.join(data_dir, "X_test.npy")
    y_test_path = os.path.join(data_dir, "y_test.npy")
    
    if not os.path.exists(X_test_path):
        print(f"[ERROR] Test data not found under: {data_dir}")
        return
        
    X_test = np.load(X_test_path)
    y_test = np.load(y_test_path)
    print(f"Loaded test dataset split: {len(X_test)} samples")
    
    # 1. Stress Test: Jitter (Noise STD)
    jitter_levels = [0.0, 0.005, 0.01, 0.02, 0.04, 0.07, 0.1]
    jitter_accs = []
    print("\n--- Stress Level 1: Gaussian Coordinate Jitter Noise ---")
    for lvl in jitter_levels:
        X_stressed = apply_jitter(X_test, lvl)
        acc = evaluate_stress(model, X_stressed, y_test, args.device)
        jitter_accs.append(acc)
        print(f"  Noise STD (sigma) = {lvl:<6} | Accuracy = {acc:.2f}%")
        
    # 2. Stress Test: Rotation Angle (degrees)
    rotation_levels = [0.0, 5.0, 10.0, 15.0, 20.0, 30.0, 45.0]
    rotation_accs = []
    print("\n--- Stress Level 2: Random Hand Coordinate Rotation ---")
    for lvl in rotation_levels:
        X_stressed = apply_rotation(X_test, lvl)
        acc = evaluate_stress(model, X_stressed, y_test, args.device)
        rotation_accs.append(acc)
        print(f"  Angle (degrees) = {lvl:<6} | Accuracy = {acc:.2f}%")
        
    # 3. Stress Test: Joint Occlusion (dropped joints)
    occlusion_levels = [0, 1, 2, 3, 5, 8, 12]
    occlusion_accs = []
    print("\n--- Stress Level 3: Joint Occlusion (Dropped Joints) ---")
    for lvl in occlusion_levels:
        X_stressed = apply_occlusion(X_test, lvl)
        acc = evaluate_stress(model, X_stressed, y_test, args.device)
        occlusion_accs.append(acc)
        print(f"  Dropped Joints = {lvl:<6} | Accuracy = {acc:.2f}%")
        
    # Plot degradation curves
    plt.figure(figsize=(15, 4))
    
    # Subplot 1: Jitter
    plt.subplot(1, 3, 1)
    plt.plot(jitter_levels, jitter_accs, 'o-', color='crimson', linewidth=2)
    plt.axhline(y=90.0, color='gray', linestyle='--', alpha=0.5)
    plt.xlabel('Noise Std Dev (Jitter)')
    plt.ylabel('Accuracy (%)')
    plt.title('Coordinate Jitter Noise')
    plt.ylim(0, 105)
    plt.grid(True, alpha=0.3)
    
    # Subplot 2: Rotation
    plt.subplot(1, 3, 2)
    plt.plot(rotation_levels, rotation_accs, 'o-', color='darkorange', linewidth=2)
    plt.axhline(y=90.0, color='gray', linestyle='--', alpha=0.5)
    plt.xlabel('Rotation Angle (deg)')
    plt.title('Hand Rotation Sensitivity')
    plt.ylim(0, 105)
    plt.grid(True, alpha=0.3)
    
    # Subplot 3: Occlusion
    plt.subplot(1, 3, 3)
    plt.plot(occlusion_levels, occlusion_accs, 'o-', color='royalblue', linewidth=2)
    plt.axhline(y=90.0, color='gray', linestyle='--', alpha=0.5)
    plt.xlabel('Num Joint Occlusions')
    plt.title('Joint Occlusion Tolerance')
    plt.ylim(0, 105)
    plt.grid(True, alpha=0.3)
    
    plt.suptitle("Robustness Degradation Curves (ASL Landmark Attention Model)", fontsize=13, weight='bold')
    plt.tight_layout()
    
    checkpoint_dir = Path(args.checkpoint).parent
    png_path = checkpoint_dir / "robustness_degradation.png"
    plt.savefig(png_path, dpi=150)
    print(f"\nSaved degradation curves to: {png_path}")
    
    # Save results to CSV
    csv_path = r"C:/Users/Pc/.gemini/antigravity/brain/331ab3b0-b2e1-44dd-b5fe-05c56a606d80/artifacts/asl_robustness_degradation.csv"
    with open(csv_path, "w") as f:
        f.write("TestType,StressLevel,Accuracy\n")
        for lvl, acc in zip(jitter_levels, jitter_accs):
            f.write(f"Jitter,{lvl},{acc:.4f}\n")
        for lvl, acc in zip(rotation_levels, rotation_accs):
            f.write(f"Rotation,{lvl},{acc:.4f}\n")
        for lvl, acc in zip(occlusion_levels, occlusion_accs):
            f.write(f"Occlusion,{lvl},{acc:.4f}\n")
    print(f"[SUCCESS] Robustness metrics saved to: {csv_path}!")

if __name__ == "__main__":
    main()
