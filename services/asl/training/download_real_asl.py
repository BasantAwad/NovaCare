import os
import sys
import urllib.request
from pathlib import Path
import numpy as np
from sklearn.model_selection import train_test_split

def main():
    print("=" * 60)
    print("Downloading & Processing Real ASL Alphabet Keypoint Dataset")
    print("=" * 60)
    
    url = "https://raw.githubusercontent.com/Muhib-Mehdi/ASL-Recognition-System/main/model/keypoint_classifier/keypoint.csv"
    data_dir = r"c:\Users\Pc\NovaCare-1\services\asl\data\processed"
    os.makedirs(data_dir, exist_ok=True)
    
    print(f"Downloading from raw source: {url}...")
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req) as response:
            content = response.read().decode('utf-8')
    except Exception as e:
        print(f"[ERROR] Failed to download keypoint dataset: {e}")
        return
        
    lines = content.splitlines()
    print(f"Successfully downloaded {len(lines)} raw rows.")
    
    X_list = []
    y_list = []
    
    # Parse lines and clean data
    for idx, line in enumerate(lines):
        if not line.strip():
            continue
        parts = line.split(',')
        if len(parts) != 43: # 1 label + 42 keypoint coordinates
            continue
        try:
            label = int(parts[0])
            coords = [float(x) for x in parts[1:]]
            
            # Convert 2D coordinates (21, 2) to 3D by padding Z=0.0
            coords_2d = np.array(coords, dtype=np.float32).reshape(21, 2)
            coords_3d = np.hstack([coords_2d, np.zeros((21, 1), dtype=np.float32)]) # (21, 3)
            
            # Wrist centering (origin at joint 0)
            wrist = coords_3d[0].copy()
            coords_3d -= wrist
            
            # Middle-finger base scale normalization
            mid_base_dist = np.linalg.norm(coords_3d[9])
            if mid_base_dist > 0.001:
                coords_3d /= mid_base_dist
                
            X_list.append(coords_3d.flatten())
            y_list.append(label)
        except ValueError:
            continue
            
    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.int64)
    print(f"Parsed and cleaned {len(X)} valid samples.")
    
    # Stratified split to prevent train/test contamination and handle class imbalance
    print("\nPerforming stratified train/validation/test splits (70/15/15)...")
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=0.15, random_state=42, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=0.1765, random_state=42, stratify=y_train_val
    ) # 0.1765 * 0.85 approx 0.15
    
    # Save splits
    np.save(os.path.join(data_dir, "X_train.npy"), X_train)
    np.save(os.path.join(data_dir, "y_train.npy"), y_train)
    np.save(os.path.join(data_dir, "X_val.npy"), X_val)
    np.save(os.path.join(data_dir, "y_val.npy"), y_val)
    np.save(os.path.join(data_dir, "X_test.npy"), X_test)
    np.save(os.path.join(data_dir, "y_test.npy"), y_test)
    
    print("\nData Splits Saved Successfully:")
    print(f"  X_train shape: {X_train.shape} | y_train shape: {y_train.shape}")
    print(f"  X_val shape:   {X_val.shape}   | y_val shape:   {y_val.shape}")
    print(f"  X_test shape:  {X_test.shape}  | y_test shape:  {y_test.shape}")
    print("=" * 60)

if __name__ == "__main__":
    main()
