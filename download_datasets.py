"""
NovaCare - Dataset Download Script
Downloads Kaggle datasets for training AI models.
Run this script once to download all required datasets.
"""
import os

# You need to install kagglehub: pip install kagglehub
# Also set up Kaggle API credentials: https://www.kaggle.com/docs/api

DATASETS = [
    {
        "name": "ananthu017/emotion-detection-fer",
        "description": "FER Emotion Detection Dataset (Face images)",
        "use": "emotion_detector.py training"
    },
    {
        "name": "bhavikjikadara/emotions-dataset",
        "description": "Emotions Dataset (Face images alternative)",
        "use": "emotion_detector.py training"
    },
    {
        "name": "pashupatigupta/emotion-detection-from-text",
        "description": "Text-based Emotion Detection Dataset",
        "use": "text_emotion.py training"
    },
    {
        "name": "tapakah68/facial-emotion-recognition",
        "description": "Facial Emotion Recognition Dataset",
        "use": "emotion_detector.py training"
    }
]

def download_datasets():
    """Download all required datasets from Kaggle"""
    try:
        import kagglehub
    except ImportError:
        print("ERROR: kagglehub not installed. Run: pip install kagglehub")
        return

    download_dir = os.path.join(os.path.dirname(__file__), 'datasets')
    os.makedirs(download_dir, exist_ok=True)

    print("=" * 60)
    print("NovaCare Dataset Downloader")
    print("=" * 60)
    
    paths = {}
    for dataset in DATASETS:
        print(f"\nDownloading: {dataset['name']}")
        print(f"Description: {dataset['description']}")
        print(f"Used for: {dataset['use']}")
        
        try:
            path = kagglehub.dataset_download(dataset['name'])
            paths[dataset['name']] = path
            print(f"✓ Downloaded to: {path}")
        except Exception as e:
            print(f"✗ Error: {e}")
            paths[dataset['name']] = None

    print("\n" + "=" * 60)
    print("Download Summary")
    print("=" * 60)
    
    # Save paths to config file
    config_path = os.path.join(download_dir, 'dataset_paths.txt')
    with open(config_path, 'w') as f:
        for name, path in paths.items():
            status = "✓" if path else "✗"
            print(f"{status} {name}")
            f.write(f"{name}={path}\n")
    
    print(f"\nPaths saved to: {config_path}")
    print("\nNext steps:")
    print("1. Check the downloaded datasets")
    print("2. Run train_models.py to train AI models")
    
    return paths


if __name__ == "__main__":
    download_datasets()
