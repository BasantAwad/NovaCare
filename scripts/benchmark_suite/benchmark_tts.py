import os
import time
import requests
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

API_URL = "http://localhost:8002/tts"

def run_benchmark(output_dir):
    print("Starting TTS Benchmark...")
    os.makedirs(output_dir, exist_ok=True)
    
    # Sentences with increasing word counts
    sentences = [
        "Hello Nova.",
        "How are you doing today?",
        "Please navigate to the kitchen.",
        "I need you to call my guardian immediately, I am not feeling well.",
        "The quick brown fox jumps over the lazy dog and the robot watches carefully.",
        "NovaCare is an integrated AI-powered healthcare companion and robotic assistant designed to empower independence.",
        "I would like to listen to some relaxing music while I drink my water and check my vitals.",
        "This is a much longer sentence designed to test the real time factor of the pocket text to speech engine running on the edge hardware.",
        "By measuring the ratio of the time it takes to synthesize the audio against the actual duration of the generated audio file we can ensure the robot speaks fluidly without long awkward pauses.",
        "According to your medication schedule you need to take aspirin at nine am and then drink two glasses of water to stay hydrated throughout the day while your battery charges."
    ]
    
    word_counts = []
    latencies = []
    rtfs = [] # Real Time Factor
    
    print(f"Testing {len(sentences)} sentences against {API_URL}...")
    
    # Warmup request
    print("Sending warmup request to initialize TTS model...")
    try:
        requests.post(API_URL, data={"text": "Warmup"}, timeout=30)
    except Exception:
        pass
        
    for sentence in sentences:
        words = len(sentence.split())
        word_counts.append(words)
        
        start_time = time.time()
        try:
            resp = requests.post(API_URL, data={"text": sentence}, timeout=30)
            latency = time.time() - start_time
            
            if resp.status_code == 200:
                audio_bytes = resp.content
                latencies.append(latency)
                
                # Estimate audio duration (assuming 24kHz, 16-bit PCM WAV)
                # Header is 44 bytes. 2 bytes per sample. 24000 samples per sec. -> 48000 bytes per second.
                audio_len_bytes = len(audio_bytes) - 44
                if audio_len_bytes > 0:
                    audio_duration = audio_len_bytes / 48000.0
                    rtf = latency / audio_duration if audio_duration > 0 else 0
                    rtfs.append(rtf)
                    print(f"Words: {words} | Latency: {latency:.2f}s | Audio Duration: {audio_duration:.2f}s | RTF: {rtf:.2f}")
                else:
                    rtfs.append(0)
                    print(f"Words: {words} | Latency: {latency:.2f}s | Failed to estimate audio duration")
            else:
                print(f"Failed to synthesize {words} words: {resp.status_code}")
                latencies.append(0)
                rtfs.append(0)
                
        except Exception as e:
            print(f"TTS request failed: {e}")
            latencies.append(0)
            rtfs.append(0)

    if not any(latencies):
        print("No successful TTS requests.")
        return
        
    avg_latency = np.mean([l for l in latencies if l > 0])
    avg_rtf = np.mean([r for r in rtfs if r > 0])
    
    print("\n--- TTS Benchmark Results ---")
    print(f"Average Synthesis Latency: {avg_latency:.2f} seconds")
    print(f"Average Real-Time Factor (RTF): {avg_rtf:.2f} (Lower is better, < 1.0 is real-time)")
    
    # Plotting Latency vs Word Count
    plt.figure(figsize=(10, 5))
    sns.regplot(x=word_counts, y=latencies, color='purple', scatter_kws={'s':100})
    plt.title('TTS Synthesis Latency vs. Text Length')
    plt.xlabel('Word Count')
    plt.ylabel('Synthesis Latency (seconds)')
    plt.savefig(os.path.join(output_dir, 'tts_latency_scatter.png'))
    plt.close()
    
    # Plotting RTF
    plt.figure(figsize=(8, 5))
    sns.barplot(x=[f"{w} words" for w in word_counts], y=rtfs, palette='coolwarm')
    plt.axhline(1.0, color='red', linestyle='dashed', label='Real-time Threshold (1.0)')
    plt.title('TTS Real-Time Factor (RTF)')
    plt.xlabel('Input Size')
    plt.ylabel('RTF (Synthesis Time / Audio Duration)')
    plt.xticks(rotation=45, ha='right')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'tts_rtf_bar.png'))
    plt.close()

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    run_benchmark(os.path.join(base_dir, "..", "..", "docs", "benchmarks"))
