"""
NovaCare - Model Training Script
Train AI models using downloaded Kaggle datasets.
"""
import os
import sys

# Add parent to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def load_dataset_paths():
    """Load dataset paths from config file"""
    config_path = os.path.join(os.path.dirname(__file__), 'datasets', 'dataset_paths.txt')
    paths = {}
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            for line in f:
                if '=' in line:
                    name, path = line.strip().split('=', 1)
                    paths[name] = path if path != 'None' else None
    return paths


def train_emotion_detector(dataset_path):
    """Train facial emotion detection model"""
    print("\n" + "=" * 60)
    print("Training Emotion Detector (Face)")
    print("=" * 60)
    
    if not dataset_path:
        print("ERROR: No dataset path provided. Run download_datasets.py first.")
        return False
    
    try:
        from ai.emotion_detector import EmotionDetector
        detector = EmotionDetector()
        
        # Look for train folder in dataset
        train_path = dataset_path
        if os.path.exists(os.path.join(dataset_path, 'train')):
            train_path = os.path.join(dataset_path, 'train')
        elif os.path.exists(os.path.join(dataset_path, 'images')):
            train_path = os.path.join(dataset_path, 'images')
        
        print(f"Training on: {train_path}")
        history = detector.train(train_path, epochs=25, batch_size=64)
        print("✓ Training complete!")
        return True
    except Exception as e:
        print(f"✗ Training failed: {e}")
        return False


def train_text_emotion(dataset_path):
    """Train text emotion classifier"""
    print("\n" + "=" * 60)
    print("Training Text Emotion Analyzer")
    print("=" * 60)
    
    if not dataset_path:
        print("ERROR: No dataset path provided. Run download_datasets.py first.")
        return False
    
    try:
        from ai.text_emotion import TextEmotionAnalyzer
        analyzer = TextEmotionAnalyzer()
        
        # Look for CSV file in dataset
        csv_path = None
        for root, dirs, files in os.walk(dataset_path):
            for file in files:
                if file.endswith('.csv'):
                    csv_path = os.path.join(root, file)
                    break
            if csv_path:
                break
        
        if not csv_path:
            print("ERROR: No CSV file found in dataset")
            return False
        
        print(f"Training on: {csv_path}")
        accuracy = analyzer.train(csv_path)
        print(f"✓ Training complete! Accuracy: {accuracy:.2%}")
        return True
    except Exception as e:
        print(f"✗ Training failed: {e}")
        return False


def main():
    print("=" * 60)
    print("NovaCare Model Training")
    print("=" * 60)
    
    paths = load_dataset_paths()
    if not paths:
        print("No dataset paths found. Run download_datasets.py first.")
        return
    
    print(f"Found {len(paths)} dataset paths")
    
    # Train emotion detector (face)
    fer_path = paths.get('ananthu017/emotion-detection-fer') or paths.get('tapakah68/facial-emotion-recognition')
    if fer_path:
        train_emotion_detector(fer_path)
    else:
        print("Skipping face emotion training - no dataset available")
    
    # Train text emotion
    text_path = paths.get('pashupatigupta/emotion-detection-from-text')
    if text_path:
        train_text_emotion(text_path)
    else:
        print("Skipping text emotion training - no dataset available")
    
    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    print("\nTrained models saved to: ai/trained_models/")


if __name__ == "__main__":
    main()
