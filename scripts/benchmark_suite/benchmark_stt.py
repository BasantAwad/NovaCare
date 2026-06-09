import os
import time
import requests
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import tempfile
import speech_recognition as sr

# Required for Word Error Rate (WER) calculation
try:
    import jiwer
except ImportError:
    print("Installing jiwer for Word Error Rate calculation...")
    os.system("pip install jiwer")
    import jiwer

API_URL = "http://localhost:8002/tts"
STT_LANG = "en-US"

def synthesize_audio(text, output_path):
    """Generate audio using Kyutai Pocket TTS to test the STT engine."""
    try:
        resp = requests.post(API_URL, data={"text": text}, timeout=30)
        if resp.status_code == 200:
            with open(output_path, "wb") as f:
                f.write(resp.content)
            return True
    except Exception as e:
        print(f"TTS synthesis failed for STT benchmark: {e}")
    return False

def run_benchmark(output_dir):
    print("Starting STT (Speech-to-Text) Benchmark...")
    os.makedirs(output_dir, exist_ok=True)
    
    sentences = [
        "Hello Nova.",
        "How are you doing today?",
        "Please navigate to the kitchen.",
        "I need you to call my guardian immediately.",
        "The quick brown fox jumps over the lazy dog.",
        "NovaCare is an AI powered healthcare companion.",
        "I would like to listen to some relaxing music.",
        "My heart rate has been feeling a bit high today.",
        "What time is my next doctor appointment?",
        "Remind me to take my aspirin at nine AM."
    ]
    
    recognizer = sr.Recognizer()
    
    word_counts = []
    latencies = []
    wers = [] # Word Error Rate
    transcriptions = []
    
    for idx, sentence in enumerate(sentences):
        words = len(sentence.split())
        word_counts.append(words)
        
        # 1. Synthesize the audio file using our TTS
        temp_wav = os.path.join(tempfile.gettempdir(), f"stt_bench_{idx}.wav")
        success = synthesize_audio(sentence, temp_wav)
        
        if not success:
            print(f"[{idx+1}/{len(sentences)}] Failed to synthesize audio for testing.")
            latencies.append(0)
            wers.append(1.0)
            continue
            
        # 2. Run STT on the generated audio file
        start_time = time.time()
        recognized_text = ""
        try:
            with sr.AudioFile(temp_wav) as source:
                audio = recognizer.record(source)
                recognized_text = recognizer.recognize_google(audio, language=STT_LANG)
                latency = time.time() - start_time
                
                # Calculate Word Error Rate (WER). Lower is better (0.0 = perfect match)
                # Remove punctuation for fairer comparison
                import string
                gt_clean = sentence.translate(str.maketrans('', '', string.punctuation)).lower()
                pred_clean = recognized_text.translate(str.maketrans('', '', string.punctuation)).lower()
                
                error_rate = jiwer.wer(gt_clean, pred_clean)
                
                latencies.append(latency)
                wers.append(error_rate)
                transcriptions.append((sentence, recognized_text))
                
                print(f"[{idx+1}/{len(sentences)}] Words: {words} | Latency: {latency:.2f}s | WER: {error_rate:.2f}")
                print(f"   Ground Truth: {gt_clean}")
                print(f"   Transcribed : {pred_clean}")
                
        except Exception as e:
            print(f"[{idx+1}/{len(sentences)}] STT failed: {e}")
            latencies.append(0)
            wers.append(1.0)
        finally:
            if os.path.exists(temp_wav):
                os.remove(temp_wav)

    if not any(latencies):
        print("No successful STT requests.")
        return
        
    avg_latency = np.mean([l for l in latencies if l > 0])
    avg_wer = np.mean(wers)
    accuracy = (1.0 - avg_wer) * 100
    
    print("\n============================================================")
    print("  STT Benchmark Results")
    print("============================================================")
    print(f"Total Sentences:     {len(sentences)}")
    print(f"Average Latency:     {avg_latency:.2f} seconds")
    print(f"Average WER:         {avg_wer:.3f} (Lower is better)")
    print(f"Estimated Accuracy:  {max(0, accuracy):.1f}%")
    print("============================================================")
    
    # 1. Plot Latency vs Word Count
    plt.figure(figsize=(10, 5))
    sns.regplot(x=word_counts, y=[l for l in latencies], color='green', scatter_kws={'s':100})
    plt.title('STT Transcription Latency vs. Text Length')
    plt.xlabel('Word Count')
    plt.ylabel('Latency (seconds)')
    plt.savefig(os.path.join(output_dir, 'stt_latency_scatter.png'))
    plt.close()
    
    # 2. Plot WER
    plt.figure(figsize=(10, 5))
    sns.barplot(x=[f"{w} words" for w in word_counts], y=wers, palette='Reds')
    plt.title('STT Word Error Rate (WER) by Sentence Length')
    plt.xlabel('Input Size')
    plt.ylabel('Word Error Rate (0.0 = Perfect)')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'stt_wer_bar.png'))
    plt.close()
    
    print(f"\nCharts saved to {output_dir}/")

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(base_dir, "..", "..", "docs", "benchmarks")
    run_benchmark(output_dir)
