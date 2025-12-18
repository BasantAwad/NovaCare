"""
NovaCare - Unified Model Training Script
Train all AI models for the NovaCare system.
"""
import os
import sys
import argparse

# Add parent to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def train_medical_qa():
    """Train the Medical QA model using HuggingFace Med_Dataset"""
    print("\n" + "="*60)
    print("TRAINING: Medical Question Answering Model")
    print("="*60)
    
    try:
        from ai.medical_qa import MedicalQA
        qa = MedicalQA()
        
        success = qa.train(
            dataset_name="Med-dataset/Med_Dataset",
            epochs=3,
            batch_size=8
        )
        
        if success:
            print("✓ Medical QA training complete!")
        else:
            print("✗ Medical QA training failed")
            
    except Exception as e:
        print(f"✗ Error: {e}")


def train_conversational_ai(dataset_path=None):
    """Train the Conversational AI for emotional support"""
    print("\n" + "="*60)
    print("TRAINING: Conversational AI (Emotional Support)")
    print("="*60)
    
    try:
        from ai.conversational_ai import ConversationalAI
        ai = ConversationalAI()
        
        success = ai.train(
            dataset_path=dataset_path,
            epochs=3,
            batch_size=4
        )
        
        if success:
            print("✓ Conversational AI training complete!")
        else:
            print("✗ Conversational AI training failed")
            
    except Exception as e:
        print(f"✗ Error: {e}")


def train_emotion_face(dataset_path=None):
    """Train the facial emotion detection model using unified EmotionAnalyzer"""
    print("\n" + "="*60)
    print("TRAINING: Facial Emotion Detection")
    print("="*60)
    
    if not dataset_path:
        print("Please provide a dataset path using --emotion-dataset")
        print("Expected format: Directory with subdirs for each emotion (angry, happy, sad, etc.)")
        return
    
    try:
        from ai.emotion_analyzer import EmotionAnalyzer
        analyzer = EmotionAnalyzer()
        
        history = analyzer.train(dataset_path, mode='face', epochs=25, batch_size=64)
        print("✓ Face emotion training complete!")
        
    except Exception as e:
        print(f"✗ Error: {e}")


def train_emotion_text(dataset_path=None):
    """Train the text emotion classifier using unified EmotionAnalyzer"""
    print("\n" + "="*60)
    print("TRAINING: Text Emotion Classifier")
    print("="*60)
    
    if not dataset_path:
        print("Please provide a CSV dataset path using --text-emotion-dataset")
        print("Expected format: CSV with 'text' and 'emotion' columns")
        return
    
    try:
        from ai.emotion_analyzer import EmotionAnalyzer
        analyzer = EmotionAnalyzer()
        
        accuracy = analyzer.train(dataset_path, mode='text')
        print(f"✓ Text emotion training complete! Accuracy: {accuracy:.2%}")
        
    except Exception as e:
        print(f"✗ Error: {e}")


def download_datasets():
    """Download required datasets from Kaggle"""
    print("\n" + "="*60)
    print("DOWNLOADING: Kaggle Datasets")
    print("="*60)
    
    try:
        import kagglehub
        
        datasets = [
            ("ananthu017/emotion-detection-fer", "Facial Emotion (FER)"),
            ("pashupatigupta/emotion-detection-from-text", "Text Emotion"),
        ]
        
        for dataset_name, desc in datasets:
            print(f"\nDownloading: {desc}...")
            try:
                path = kagglehub.dataset_download(dataset_name)
                print(f"✓ Downloaded to: {path}")
            except Exception as e:
                print(f"✗ Failed: {e}")
                
    except ImportError:
        print("kagglehub not installed. Run: pip install kagglehub")


def main():
    parser = argparse.ArgumentParser(description="NovaCare Model Training")
    parser.add_argument("--all", action="store_true", help="Train all models")
    parser.add_argument("--medical", action="store_true", help="Train Medical QA model")
    parser.add_argument("--conversation", action="store_true", help="Train Conversational AI")
    parser.add_argument("--emotion-face", action="store_true", help="Train Emotion Analyzer (face)")
    parser.add_argument("--emotion-text", action="store_true", help="Train Emotion Analyzer (text)")
    parser.add_argument("--download", action="store_true", help="Download Kaggle datasets")
    
    parser.add_argument("--emotion-dataset", type=str, help="Path to facial emotion dataset")
    parser.add_argument("--text-emotion-dataset", type=str, help="Path to text emotion CSV")
    parser.add_argument("--conversation-dataset", type=str, help="Path to conversation JSON")
    
    args = parser.parse_args()
    
    print("="*60)
    print("NovaCare AI Model Training System")
    print("="*60)
    
    if args.download:
        download_datasets()
        return
    
    if args.all or args.medical:
        train_medical_qa()
    
    if args.all or args.conversation:
        train_conversational_ai(args.conversation_dataset)
    
    if args.emotion_face:
        train_emotion_face(args.emotion_dataset)
    
    if args.emotion_text:
        train_emotion_text(args.text_emotion_dataset)
    
    if not any([args.all, args.medical, args.conversation, args.emotion_face, args.emotion_text]):
        print("\nNo training option selected. Use --help to see options.")
        print("\nExamples:")
        print("  python train_models.py --all                    # Train medical + conversation")
        print("  python train_models.py --medical                # Train medical QA only")
        print("  python train_models.py --emotion-face --emotion-dataset /path/to/fer")
        print("  python train_models.py --emotion-text --text-emotion-dataset /path/to/csv")
        print("  python train_models.py --download               # Download Kaggle datasets")
    
    print("\n" + "="*60)
    print("Training session complete!")
    print("="*60)


if __name__ == "__main__":
    main()
