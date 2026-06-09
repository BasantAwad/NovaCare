import os
import time
import base64
import requests
import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

API_URL = "http://localhost:5000/api/emotion/detect"

def run_benchmark(data_dir, output_dir):
    print("Starting Emotion Detection Benchmark...")
    os.makedirs(output_dir, exist_ok=True)
    
    latencies = []
    confidences = []
    successes = 0
    failures = 0
    
    image_files = [f for f in os.listdir(data_dir) if f.endswith('.jpg')]
    if not image_files:
        print("No test images found.")
        return

    print(f"Testing {len(image_files)} images against {API_URL}...")
    
    for img_file in image_files:
        with open(os.path.join(data_dir, img_file), "rb") as f:
            b64_img = base64.b64encode(f.read()).decode('utf-8')
            
        start_time = time.time()
        try:
            resp = requests.post(API_URL, json={"image": b64_img}, timeout=5)
            latency = (time.time() - start_time) * 1000 # ms
            
            if resp.status_code == 200:
                data = resp.json()
                latencies.append(latency)
                confidences.append(data.get('confidence', 0.0))
                successes += 1
            else:
                failures += 1
        except Exception as e:
            failures += 1
            print(f"Request failed: {e}")

    if not latencies:
        print("No successful requests to benchmark.")
        return
        
    avg_latency = np.mean(latencies)
    fps = 1000.0 / avg_latency if avg_latency > 0 else 0
    
    print("\n--- Emotion Benchmark Results ---")
    print(f"Successes: {successes}, Failures: {failures}")
    print(f"Average Latency: {avg_latency:.2f} ms")
    print(f"Estimated Throughput: {fps:.2f} FPS")
    print(f"Average Confidence: {np.mean(confidences):.2%}")
    
    # Plotting Latency Distribution
    plt.figure(figsize=(10, 5))
    sns.histplot(latencies, bins=10, kde=True, color='purple')
    plt.axvline(avg_latency, color='red', linestyle='dashed', linewidth=2, label=f'Avg: {avg_latency:.2f}ms')
    plt.title('Emotion Detection Inference Latency Distribution')
    plt.xlabel('Latency (ms)')
    plt.ylabel('Frequency')
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'emotion_latency_dist.png'))
    plt.close()
    
    # Plotting Confidence Distribution
    plt.figure(figsize=(10, 5))
    sns.histplot(confidences, bins=10, kde=True, color='teal')
    plt.title('Emotion Model Confidence Distribution')
    plt.xlabel('Confidence Score (0-1)')
    plt.ylabel('Frequency')
    plt.savefig(os.path.join(output_dir, 'emotion_confidence_dist.png'))
    plt.close()

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    run_benchmark(os.path.join(base_dir, "data", "emotion_images"), os.path.join(base_dir, "..", "..", "docs", "benchmarks"))
