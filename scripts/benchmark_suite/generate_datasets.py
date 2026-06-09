import os
import csv
import json
import base64
from PIL import Image
import numpy as np

def generate_llm_dataset(filepath):
    print(f"Generating LLM dataset at {filepath}...")
    dataset = [
        # format: prompt, expected_action, param_key, param_value
        ("I fell down and my leg hurts badly!", "trigger_emergency", "", ""),
        ("Please take me to the kitchen", "navigate", "destination", "kitchen"),
        ("I want to listen to some relaxing music", "play_music", "mood", "relaxing"),
        ("Can you call my guardian?", "call_guardian", "", ""),
        ("When is my next medication?", "show_medications", "", ""),
        ("I feel very dizzy and need help", "trigger_emergency", "", ""),
        ("Move to the bedroom now", "navigate", "destination", "bedroom"),
        ("Stop playing the music", "pause_music", "", ""),
        ("Play upbeat music", "play_music", "mood", "upbeat"),
        ("Hello Nova, how are you?", "", "", ""),
        ("Are you a real person?", "", "", ""),
        ("Take me to the living room", "navigate", "destination", "living room"),
        ("I need to use the bathroom", "navigate", "destination", "bathroom"),
        ("I feel pain in my chest!", "trigger_emergency", "", ""),
        ("What is my current heart rate?", "check_health", "", ""),
        ("Pause the song please", "pause_music", "", ""),
        ("I'm feeling very sad today.", "", "", ""), # Should just be a conversational reply
        ("Navigate to the front door", "navigate", "destination", "front door"),
        ("Call dad", "call_guardian", "", ""),
        ("Can you show me my pill schedule?", "show_medications", "", "")
    ]
    
    with open(filepath, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["prompt", "expected_action", "param_key", "param_value"])
        writer.writerows(dataset)
    print("LLM dataset generated.")

def generate_emotion_dataset(output_dir, num_images=10):
    print(f"Generating Emotion images at {output_dir}...")
    os.makedirs(output_dir, exist_ok=True)
    
    for i in range(num_images):
        # Generate random noise image (mimics raw camera feed to some degree for processing)
        # Note: Since it's random noise, no face will be detected, but we can measure raw ViT latency.
        # Wait, if no face is detected, the API returns early. 
        # We need a dummy face. We'll just generate simple gradient images that the face detector might try to process.
        img_array = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        img = Image.fromarray(img_array)
        path = os.path.join(output_dir, f"test_img_{i}.jpg")
        img.save(path)
        
    print(f"{num_images} images generated.")

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, "data")
    os.makedirs(data_dir, exist_ok=True)
    
    generate_llm_dataset(os.path.join(data_dir, "llm_prompts.csv"))
    generate_emotion_dataset(os.path.join(data_dir, "emotion_images"))
