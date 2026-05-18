import os
import sys
import time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report, f1_score

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Add services/asl/ to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from models import create_model, count_parameters
from training.train_real_asl import RealASLLandmarkDataset, RealWorldLandmarkAugmentor

def train_and_eval(model_type: str, X_train, y_train, X_val, y_val, X_test, y_test, epochs=3, batch_size=64, device="cpu"):
    print(f"\n--- Training Model: {model_type.upper()} ---")
    
    augmentor = RealWorldLandmarkAugmentor(p=0.5)
    train_dataset = RealASLLandmarkDataset(X_train, y_train, augmentor)
    val_dataset = RealASLLandmarkDataset(X_val, y_val, None)
    test_dataset = RealASLLandmarkDataset(X_test, y_test, None)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Update number of classes to 26 for ASL Alphabet
    config = {
        "input_dim": 63,
        "hidden_dim": 256,
        "num_heads": 4,
        "num_layers": 2,
        "num_classes": 26, # 26 ASL alphabet letters
        "dropout": 0.3
    }
    
    model = create_model(model_type, config).to(device)
    param_count = count_parameters(model)
    
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
    
    start_train_time = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            total_loss += loss.item()
            correct += logits.argmax(dim=-1).eq(y).sum().item()
            total += y.size(0)
            
        # Validation
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                val_correct += logits.argmax(dim=-1).eq(y).sum().item()
                val_total += y.size(0)
        print(f"  Epoch {epoch}/{epochs} | Train Loss: {total_loss/len(train_loader):.4f} | Train Acc: {(correct/total)*100.0:.2f}% | Val Acc: {(val_correct/val_total)*100.0:.2f}%")
        
    train_duration = time.time() - start_train_time
    
    # Save model temporarily to measure disk size
    temp_path = Path(f"temp_{model_type}.pt")
    torch.save(model.state_dict(), temp_path)
    file_size_mb = temp_path.stat().st_size / (1024 * 1024)
    temp_path.unlink()
    
    # Evaluate on Test Split
    model.eval()
    all_preds = []
    all_targets = []
    
    # Latency benchmarking
    latencies = []
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            for single_x in x:
                single_x = single_x.unsqueeze(0)
                t0 = time.perf_counter()
                logits = model(single_x)
                t1 = time.perf_counter()
                latencies.append((t1 - t0) * 1000.0) # in ms
                
            logits = model(x)
            all_preds.extend(logits.argmax(dim=-1).cpu().numpy())
            all_targets.extend(y.numpy())
            
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    
    test_acc = (all_preds == all_targets).mean() * 100.0
    weighted_f1 = f1_score(all_targets, all_preds, average='weighted') * 100.0
    macro_f1 = f1_score(all_targets, all_preds, average='macro') * 100.0
    avg_latency = np.mean(latencies)
    throughput_fps = 1000.0 / avg_latency
    
    print(f"Evaluation results for {model_type.upper()}:")
    print(f"  Test Accuracy: {test_acc:.2f}% | Weighted F1: {weighted_f1:.2f}% | Macro F1: {macro_f1:.2f}%")
    print(f"  Avg Latency:   {avg_latency:.3f} ms | Throughput:   {throughput_fps:.2f} FPS")
    
    return {
        "model_type": model_type,
        "param_count": param_count,
        "file_size_mb": file_size_mb,
        "train_time_sec": train_duration,
        "test_acc": test_acc,
        "weighted_f1": weighted_f1,
        "macro_f1": macro_f1,
        "avg_latency_ms": avg_latency,
        "throughput_fps": throughput_fps
    }

def main():
    print("=" * 60)
    print("ASL ARCHITECTURAL ARCHITECTURE COMPARATIVE BENCHMARK")
    print("=" * 60)
    
    data_dir = r"services/asl/data/processed"
    X_train = np.load(os.path.join(data_dir, "X_train.npy"))
    y_train = np.load(os.path.join(data_dir, "y_train.npy"))
    X_val = np.load(os.path.join(data_dir, "X_val.npy"))
    y_val = np.load(os.path.join(data_dir, "y_val.npy"))
    X_test = np.load(os.path.join(data_dir, "X_test.npy"))
    y_test = np.load(os.path.join(data_dir, "y_test.npy"))
    
    print(f"Loaded real dataset split with {len(X_train)} train, {len(X_test)} test samples.")
    
    results = {}
    for mtype in ["attention", "lite"]:
        results[mtype] = train_and_eval(mtype, X_train, y_train, X_val, y_val, X_test, y_test, epochs=3, batch_size=128)
        
    # Save comparison to CSV
    csv_path = r"C:/Users/Pc/.gemini/antigravity/brain/331ab3b0-b2e1-44dd-b5fe-05c56a606d80/artifacts/asl_architectural_comparison.csv"
    with open(csv_path, "w") as f:
        f.write("ModelType,ParamCount,FileSizeMB,TrainTimeSec,TestAcc,WeightedF1,MacroF1,AvgLatencyMs,ThroughputFPS\n")
        for mtype, r in results.items():
            f.write(f"{mtype},{r['param_count']},{r['file_size_mb']:.4f},{r['train_time_sec']:.2f},{r['test_acc']:.4f},{r['weighted_f1']:.4f},{r['macro_f1']:.4f},{r['avg_latency_ms']:.4f},{r['throughput_fps']:.2f}\n")
    print(f"\n[SUCCESS] Architectural comparison saved to: {csv_path}")
    
    # Plot comparison bar chart
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    mtypes = ["Self-Attention\n(Transformer)", "Lite MLP"]
    colors = ["crimson", "royalblue"]
    
    # Subplot 1: Test Accuracy
    accs = [results["attention"]["test_acc"], results["lite"]["test_acc"]]
    axes[0].bar(mtypes, accs, color=colors, width=0.4)
    axes[0].set_ylabel("Accuracy (%)")
    axes[0].set_title("Test Accuracy comparison")
    axes[0].set_ylim(0, 105)
    for i, v in enumerate(accs):
        axes[0].text(i, v + 2, f"{v:.2f}%", ha='center', weight='bold')
        
    # Subplot 2: CPU Latency
    lats = [results["attention"]["avg_latency_ms"], results["lite"]["avg_latency_ms"]]
    axes[1].bar(mtypes, lats, color=colors, width=0.4)
    axes[1].set_ylabel("Inference Latency (ms)")
    axes[1].set_title("CPU Latency (Lower is Better)")
    axes[1].set_ylim(0, max(lats) * 1.3)
    for i, v in enumerate(lats):
        axes[1].text(i, v + (max(lats)*0.03), f"{v:.3f} ms", ha='center', weight='bold')
        
    # Subplot 3: Param Count
    params = [results["attention"]["param_count"] / 1e6, results["lite"]["param_count"] / 1e6]
    axes[2].bar(mtypes, params, color=colors, width=0.4)
    axes[2].set_ylabel("Parameters (Millions)")
    axes[2].set_title("Model size comparison")
    axes[2].set_ylim(0, max(params) * 1.3)
    for i, v in enumerate(params):
        axes[2].text(i, v + (max(params)*0.03), f"{v:.2f}M", ha='center', weight='bold')
        
    plt.suptitle("ASL Fingerspelling Model Architecture Comparison (Real ASL Dataset)", fontsize=13, weight='bold')
    plt.tight_layout()
    
    png_path = r"C:/Users/Pc/.gemini/antigravity/brain/331ab3b0-b2e1-44dd-b5fe-05c56a606d80/artifacts/asl_architectural_comparison.png"
    plt.savefig(png_path, dpi=150)
    print(f"Saved architectural comparison plot to: {png_path}")
    print("=" * 60)

if __name__ == "__main__":
    main()
