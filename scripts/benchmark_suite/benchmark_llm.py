import os
import time
import requests
import csv
import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

API_URL = "http://localhost:5000/api/chat"

def run_benchmark(dataset_file, output_dir):
    print("Starting LLM Benchmark...")
    os.makedirs(output_dir, exist_ok=True)
    
    latencies = []
    y_true_actions = []
    y_pred_actions = []
    
    # Read the synthetic dataset
    with open(dataset_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        data = list(reader)
        
    print(f"Testing {len(data)} prompts against {API_URL}...")
    
    for row in data:
        prompt = row['prompt']
        expected_action = row['expected_action'].strip()
        
        start_time = time.time()
        try:
            # We explicitly test the 'fast' local profile for consistent latency benchmarking, 
            # or allow it to fallback to cloud if local is off.
            resp = requests.post(API_URL, json={"message": prompt, "llm_profile": "fast"}, timeout=30)
            latency = time.time() - start_time
            
            if resp.status_code == 200:
                resp_data = resp.json()
                latencies.append(latency)
                
                actions = resp_data.get('actions', [])
                pred_action = actions[0]['name'] if actions else ""
                
                y_true_actions.append(expected_action)
                y_pred_actions.append(pred_action)
                print(f"Prompt: '{prompt[:30]}...' | Latency: {latency:.2f}s | Expected: '{expected_action}' | Got: '{pred_action}'")
            else:
                print(f"Failed prompt '{prompt[:30]}...': Status {resp.status_code}")
                
        except Exception as e:
            print(f"Error on prompt '{prompt[:30]}...': {e}")
            
    if not latencies:
        print("No successful LLM requests.")
        return
        
    avg_latency = np.mean(latencies)
    
    # Calculate ML Metrics for Tool Calling
    # To compute precision/recall properly for multi-class or binary, we use weighted avg
    # We treat empty string "" as a class (no action)
    accuracy = accuracy_score(y_true_actions, y_pred_actions)
    precision = precision_score(y_true_actions, y_pred_actions, average='weighted', zero_division=0)
    recall = recall_score(y_true_actions, y_pred_actions, average='weighted', zero_division=0)
    f1 = f1_score(y_true_actions, y_pred_actions, average='weighted', zero_division=0)
    
    print("\n--- LLM Benchmark Results ---")
    print(f"Average End-to-End Latency: {avg_latency:.2f} seconds")
    print(f"Tool Calling Accuracy: {accuracy:.2%}")
    print(f"Tool Calling Precision: {precision:.2%}")
    print(f"Tool Calling Recall: {recall:.2%}")
    print(f"Tool Calling F1-Score: {f1:.2%}")
    
    # Plotting Latency Scatter
    plt.figure(figsize=(10, 5))
    sns.scatterplot(x=range(len(latencies)), y=latencies, color='blue', s=100)
    plt.axhline(avg_latency, color='red', linestyle='dashed', label=f'Avg: {avg_latency:.2f}s')
    plt.title('LLM Response Latency per Request')
    plt.xlabel('Request Sequence')
    plt.ylabel('Latency (seconds)')
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'llm_latency_scatter.png'))
    plt.close()
    
    # Plotting ML Metrics Bar Chart
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    scores = [accuracy, precision, recall, f1]
    
    plt.figure(figsize=(8, 5))
    sns.barplot(x=metrics, y=scores, palette='viridis')
    plt.title('LLM Tool Extraction Performance (Heuristics + JSON)')
    plt.ylim(0, 1.05)
    for i, score in enumerate(scores):
        plt.text(i, score + 0.02, f"{score:.2%}", ha='center')
    plt.ylabel('Score')
    plt.savefig(os.path.join(output_dir, 'llm_ml_metrics.png'))
    plt.close()

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    run_benchmark(
        os.path.join(base_dir, "data", "llm_prompts.csv"), 
        os.path.join(base_dir, "..", "..", "docs", "benchmarks")
    )
