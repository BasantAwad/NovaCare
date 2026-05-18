import os
import numpy as np

def main():
    print("=" * 60)
    print("Generating Synthetic ASL Landmark Dataset (Distinct Classes)")
    print("=" * 60)
    
    # 1. Create directories
    data_dir = r"c:\Users\Pc\NovaCare-1\services\asl\data\processed"
    os.makedirs(data_dir, exist_ok=True)
    
    num_classes = 29
    num_landmarks = 21
    num_coords = 3
    
    # Samples per split per class
    n_train = 60
    n_val = 15
    n_test = 15
    
    X_train_list, y_train_list = [], []
    X_val_list, y_val_list = [], []
    X_test_list, y_test_list = [], []
    
    # Generate templates for each class to make them fully learnable
    for c in range(num_classes):
        # Varying frequencies and scales across classes makes them structurally unique
        freq_factor = 1.0 + (c * 0.12)
        scale_factor = 0.5 + (c * 0.05)
        
        base_x = np.sin(np.linspace(0, np.pi * freq_factor, num_landmarks)) * scale_factor
        base_y = np.cos(np.linspace(0, np.pi * freq_factor * 0.8, num_landmarks)) * scale_factor
        base_z = np.linspace(-0.2, 0.2, num_landmarks) * (1.0 + c * 0.03)
        
        template = np.stack([base_x, base_y, base_z], axis=1).flatten() # (63,)
        
        # Train split
        for _ in range(n_train):
            noise = np.random.normal(0, 0.01, template.shape)
            X_train_list.append(template + noise)
            y_train_list.append(c)
            
        # Val split
        for _ in range(n_val):
            noise = np.random.normal(0, 0.01, template.shape)
            X_val_list.append(template + noise)
            y_val_list.append(c)
            
        # Test split
        for _ in range(n_test):
            noise = np.random.normal(0, 0.01, template.shape)
            X_test_list.append(template + noise)
            y_test_list.append(c)
            
    # Convert to numpy arrays
    X_train = np.array(X_train_list, dtype=np.float32)
    y_train = np.array(y_train_list, dtype=np.int64)
    X_val = np.array(X_val_list, dtype=np.float32)
    y_val = np.array(y_val_list, dtype=np.int64)
    X_test = np.array(X_test_list, dtype=np.float32)
    y_test = np.array(y_test_list, dtype=np.int64)
    
    # Normalize scales (Wrist-relative centering and Middle finger division)
    for X in [X_train, X_val, X_test]:
        for i in range(len(X)):
            sample = X[i].reshape(21, 3)
            wrist = sample[0].copy()
            # Centering (translation invariance)
            sample -= wrist
            # Middle finger base distance scaling (scale invariance)
            mid_base_dist = np.linalg.norm(sample[9])
            if mid_base_dist > 0.001:
                sample /= mid_base_dist
            X[i] = sample.flatten()
            
    # 2. Save arrays
    np.save(os.path.join(data_dir, "X_train.npy"), X_train)
    np.save(os.path.join(data_dir, "y_train.npy"), y_train)
    np.save(os.path.join(data_dir, "X_val.npy"), X_val)
    np.save(os.path.join(data_dir, "y_val.npy"), y_val)
    np.save(os.path.join(data_dir, "X_test.npy"), X_test)
    np.save(os.path.join(data_dir, "y_test.npy"), y_test)
    
    print(f"Generated successfully and saved in: {data_dir}")
    print(f"  X_train shape: {X_train.shape} | y_train shape: {y_train.shape}")
    print(f"  X_val shape:   {X_val.shape}   | y_val shape:   {y_val.shape}")
    print(f"  X_test shape:  {X_test.shape}  | y_test shape:  {y_test.shape}")
    print("=" * 60)

if __name__ == "__main__":
    main()
