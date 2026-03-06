"""
Utility Functions for Multimodal Emotion AI
Shared helper functions for video processing and result aggregation.
"""

import os
import cv2
import numpy as np
from moviepy.editor import VideoFileClip
from typing import Tuple, List, Dict, Optional
import tempfile


# ============================================================================
# Constants
# ============================================================================
EMOTION_LABELS_FACE = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
EMOTION_LABELS_AUDIO = ['neutral', 'calm', 'happy', 'sad', 'angry', 'fearful', 'disgust', 'surprised']
SAMPLE_RATE = 16000  # Required for Wav2Vec 2.0


# ============================================================================
# Video Processing Functions
# ============================================================================
def extract_audio_from_video(video_path: str, output_path: Optional[str] = None) -> str:
    """
    Extract audio track from a video file.
    
    Args:
        video_path: Path to the input video file.
        output_path: Optional path for the output audio file.
                     If None, creates a temp file.
    
    Returns:
        Path to the extracted audio file (WAV format).
    """
    if output_path is None:
        output_path = tempfile.mktemp(suffix='.wav')
    
    video = VideoFileClip(video_path)
    audio = video.audio
    audio.write_audiofile(output_path, fps=SAMPLE_RATE, verbose=False, logger=None)
    video.close()
    
    return output_path


def extract_frames_from_video(
    video_path: str,
    interval_seconds: float = 1.0,
    output_dir: Optional[str] = None
) -> List[np.ndarray]:
    """
    Extract frames from a video at specified intervals.
    
    Args:
        video_path: Path to the input video file.
        interval_seconds: Time interval between frame captures.
        output_dir: Optional directory to save frames as images.
    
    Returns:
        List of frames as numpy arrays (BGR format).
    """
    frames = []
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = int(fps * interval_seconds)
    
    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_count % frame_interval == 0:
            frames.append(frame)
            
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
                frame_path = os.path.join(output_dir, f"frame_{frame_count:06d}.jpg")
                cv2.imwrite(frame_path, frame)
        
        frame_count += 1
    
    cap.release()
    return frames


def split_video(video_path: str) -> Tuple[str, List[np.ndarray]]:
    """
    Split a video into its audio track and frames.
    
    Args:
        video_path: Path to the input video file.
    
    Returns:
        Tuple of (audio_path, list_of_frames)
    """
    audio_path = extract_audio_from_video(video_path)
    frames = extract_frames_from_video(video_path)
    return audio_path, frames


# ============================================================================
# Result Aggregation Functions
# ============================================================================
def aggregate_predictions(
    text_result: Dict,
    audio_result: Dict,
    face_result: Dict
) -> Dict:
    """
    Aggregate predictions from all three models to determine final emotion.
    
    Args:
        text_result: Prediction from text model {"emotion": str, "confidence": float}
        audio_result: Prediction from audio model {"emotion": str, "confidence": float}
        face_result: Prediction from face model {"emotion": str, "confidence": float}
    
    Returns:
        Aggregated result with final emotion and analysis.
    """
    results = {
        'text': text_result,
        'audio': audio_result,
        'face': face_result
    }
    
    # Get all predicted emotions
    emotions = [
        text_result.get('emotion', 'unknown'),
        audio_result.get('emotion', 'unknown'),
        face_result.get('emotion', 'unknown')
    ]
    
    # Get confidences
    confidences = [
        text_result.get('confidence', 0.0),
        audio_result.get('confidence', 0.0),
        face_result.get('confidence', 0.0)
    ]
    
    # Check for unanimous agreement
    unique_emotions = set(e for e in emotions if e != 'unknown')
    
    if len(unique_emotions) == 1:
        # All models agree
        final_emotion = unique_emotions.pop()
        agreement = "unanimous"
        analysis = f"All three modalities detected '{final_emotion}' emotion."
    elif len(unique_emotions) == 2:
        # Majority vote or conflict
        from collections import Counter
        emotion_counts = Counter(e for e in emotions if e != 'unknown')
        most_common = emotion_counts.most_common(1)[0]
        
        if most_common[1] >= 2:
            # Majority (2/3) agree
            final_emotion = most_common[0]
            agreement = "majority"
            analysis = f"Majority (2/3) detected '{final_emotion}'. Possible mixed signals."
        else:
            # All different - use weighted average by confidence
            weighted_emotions = list(zip(emotions, confidences))
            weighted_emotions = [(e, c) for e, c in weighted_emotions if e != 'unknown']
            final_emotion = max(weighted_emotions, key=lambda x: x[1])[0]
            agreement = "conflict"
            analysis = "Conflicting signals detected. Using highest confidence prediction."
    else:
        # All different emotions
        weighted_emotions = list(zip(emotions, confidences))
        weighted_emotions = [(e, c) for e, c in weighted_emotions if e != 'unknown']
        if weighted_emotions:
            final_emotion = max(weighted_emotions, key=lambda x: x[1])[0]
        else:
            final_emotion = 'unknown'
        agreement = "conflict"
        analysis = "All modalities detected different emotions. Possible complex emotional state."
    
    # Detect potential sarcasm or deception
    special_cases = detect_special_cases(text_result, audio_result, face_result)
    if special_cases:
        analysis += f" Note: {special_cases}"
    
    return {
        'final_emotion': final_emotion,
        'agreement': agreement,
        'analysis': analysis,
        'individual_results': results,
        'average_confidence': sum(confidences) / len(confidences) if confidences else 0.0
    }


def detect_special_cases(
    text_result: Dict,
    audio_result: Dict,
    face_result: Dict
) -> Optional[str]:
    """
    Detect special cases like sarcasm, deception, or mixed emotions.
    
    Returns:
        String describing the special case, or None.
    """
    text_emotion = text_result.get('emotion', 'unknown').lower()
    audio_emotion = audio_result.get('emotion', 'unknown').lower()
    face_emotion = face_result.get('emotion', 'unknown').lower()
    
    # Potential sarcasm: positive words but negative tone/expression
    positive_words = {'happy', 'joy', 'love', 'excited', 'grateful'}
    negative_expressions = {'angry', 'sad', 'disgust', 'fear', 'contempt'}
    
    if text_emotion in positive_words:
        if audio_emotion in negative_expressions or face_emotion in negative_expressions:
            return "Potential sarcasm detected (positive words with negative tone/expression)."
    
    # Masked emotion: neutral face but strong audio/text emotion
    if face_emotion == 'neutral':
        if audio_emotion in negative_expressions or text_emotion in negative_expressions:
            return "Possible suppressed/masked emotion detected."
    
    # Confusion: very low confidence across all modalities
    avg_conf = (
        text_result.get('confidence', 0) +
        audio_result.get('confidence', 0) +
        face_result.get('confidence', 0)
    ) / 3
    
    if avg_conf < 0.4:
        return "Low confidence across modalities - emotional state may be ambiguous."
    
    return None


if __name__ == "__main__":
    # Test the utility functions
    print("Utility functions loaded successfully!")
    print(f"Face emotion labels: {EMOTION_LABELS_FACE}")
    print(f"Audio emotion labels: {EMOTION_LABELS_AUDIO}")
    print(f"Target sample rate: {SAMPLE_RATE} Hz")
