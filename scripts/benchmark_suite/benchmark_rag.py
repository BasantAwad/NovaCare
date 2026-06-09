"""
NovaCare RAG Benchmark Suite v2
Tests RAG retrieval with real query-answer pairs, query routing, and proper metrics.
Generates 6 publication-quality graphs.
"""
import os
import sys
import time
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict

# Setup path and env
current_dir = os.path.dirname(os.path.abspath(__file__))
backend_dir = os.path.join(current_dir, "..", "..", "services", "llm")
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

env_path = os.path.join(backend_dir, ".env")
if os.path.exists(env_path):
    with open(env_path, "r") as f:
        for line in f:
            if line.strip() and not line.startswith("#") and "=" in line:
                k, v = line.strip().split("=", 1)
                os.environ[k] = v.strip("'\"")

from utils.rag_helper import rag_manager

# ── Test Dataset ─────────────────────────────────────────────────────────

TEST_CASES = [
    # Medications
    {"query": "What medications am I currently taking?", "expected_sources": ["medications"], "expected_keys": ["medication_name", "dosage"]},
    {"query": "When do I need to take my pills today?", "expected_sources": ["medications"], "expected_keys": ["scheduled_time", "medication_name"]},
    {"query": "What's my Lipitor dosage?", "expected_sources": ["medications"], "expected_keys": ["medication_name", "dosage"]},
    # Vitals
    {"query": "What's my heart rate right now?", "expected_sources": ["vitals"], "expected_keys": ["heart_rate"]},
    {"query": "What are my latest vital signs?", "expected_sources": ["vitals"], "expected_keys": ["heart_rate", "oxygen_level"]},
    {"query": "Is my blood pressure okay?", "expected_sources": ["vitals"], "expected_keys": ["heart_rate"]},
    # Vitals Trend
    {"query": "How has my heart rate been this week?", "expected_sources": ["vitals_trend"], "expected_keys": ["heart_rate", "measured_at"]},
    {"query": "Have my vitals been stable lately?", "expected_sources": ["vitals_trend"], "expected_keys": ["heart_rate"]},
    {"query": "Show me my vitals over the past few days", "expected_sources": ["vitals_trend"], "expected_keys": ["heart_rate", "measured_at"]},
    # Appointments
    {"query": "Do I have any upcoming appointments?", "expected_sources": ["appointments"], "expected_keys": ["appointment_type", "scheduled_at"]},
    {"query": "When is my next doctor visit?", "expected_sources": ["appointments"], "expected_keys": ["scheduled_at", "doctor_first_name"]},
    {"query": "What was my last checkup about?", "expected_sources": ["appointments"], "expected_keys": ["appointment_type", "notes"]},
    # Health Conditions
    {"query": "What are my diagnosed health conditions?", "expected_sources": ["health_conditions"], "expected_keys": ["condition_name", "severity"]},
    {"query": "Do I have Parkinson's disease?", "expected_sources": ["health_conditions"], "expected_keys": ["condition_name"]},
    {"query": "Tell me about my health condition", "expected_sources": ["health_conditions"], "expected_keys": ["condition_name", "severity"]},
    # Allergies
    {"query": "Am I allergic to anything?", "expected_sources": ["allergies"], "expected_keys": ["allergy_name", "severity"]},
    {"query": "Can I take penicillin?", "expected_sources": ["allergies"], "expected_keys": ["allergy_name"]},
    {"query": "What are my allergies?", "expected_sources": ["allergies"], "expected_keys": ["allergy_name", "allergy_type"]},
    # Emergency Contacts
    {"query": "Who should I call in an emergency?", "expected_sources": ["emergency_contacts"], "expected_keys": ["name", "phone"]},
    {"query": "Call my guardian please", "expected_sources": ["emergency_contacts"], "expected_keys": ["name", "phone"]},
    {"query": "Who is my emergency contact?", "expected_sources": ["emergency_contacts"], "expected_keys": ["name", "relationship"]},
    # Medical Notes
    {"query": "What did the doctor say last time?", "expected_sources": ["medical_notes"], "expected_keys": ["note_content"]},
    {"query": "What are my latest lab results?", "expected_sources": ["medical_notes"], "expected_keys": ["note_content"]},
    {"query": "Show me my doctor's notes from the last visit", "expected_sources": ["medical_notes"], "expected_keys": ["note_content", "created_at"]},
    # Emotion History
    {"query": "How have I been feeling lately?", "expected_sources": ["emotion_history"], "expected_keys": ["primary_emotion", "avg_sentiment"]},
    {"query": "Has there been any distress detected?", "expected_sources": ["emotion_history"], "expected_keys": ["primary_emotion", "distress_detected"]},
    {"query": "What's my mood history?", "expected_sources": ["emotion_history"], "expected_keys": ["primary_emotion"]},
    # Notifications
    {"query": "Do I have any pending reminders?", "expected_sources": ["notifications"], "expected_keys": ["title", "message"]},
    {"query": "Show me my unread notifications", "expected_sources": ["notifications"], "expected_keys": ["title", "is_read"]},
    {"query": "Any alerts I should know about?", "expected_sources": ["notifications"], "expected_keys": ["title", "message"]},
    # Multi-source queries
    {"query": "Give me a full health summary", "expected_sources": ["medications", "vitals", "health_conditions", "allergies", "appointments", "vitals_trend", "emergency_contacts", "medical_notes", "emotion_history", "notifications"], "expected_keys": []},
    {"query": "I feel sick, what should I do?", "expected_sources": ["vitals", "emergency_contacts", "medications", "health_conditions", "allergies", "vitals_trend", "appointments", "medical_notes", "emotion_history", "notifications"], "expected_keys": []},
]


def deep_search_keys(data, keys):
    """Recursively search for keys in nested dicts/lists."""
    found = set()
    if isinstance(data, dict):
        for k, v in data.items():
            if k in keys:
                if v is not None and v != "" and v != []:
                    found.add(k)
            found |= deep_search_keys(v, keys)
    elif isinstance(data, list):
        for item in data:
            found |= deep_search_keys(item, keys)
    return found


def run_benchmark(output_dir):
    print("=" * 60)
    print("  NovaCare RAG Benchmark Suite v2")
    print("=" * 60)
    os.makedirs(output_dir, exist_ok=True)
    
    results = []
    
    for i, tc in enumerate(TEST_CASES):
        query = tc["query"]
        expected_sources = set(tc["expected_sources"])
        expected_keys = set(tc["expected_keys"])
        
        start = time.time()
        context, routed_sources = rag_manager.get_routed_context(query, rover_id="RV001")
        latency_ms = (time.time() - start) * 1000
        
        routed_set = set(routed_sources)
        
        # Routing accuracy: did the router select the right sources?
        true_positives = expected_sources & routed_set
        false_positives = routed_set - expected_sources
        false_negatives = expected_sources - routed_set
        
        precision = len(true_positives) / len(routed_set) if routed_set else 0
        recall = len(true_positives) / len(expected_sources) if expected_sources else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        # Data completeness: of the expected keys, how many were actually in the returned data?
        if expected_keys:
            found_keys = deep_search_keys(context, expected_keys)
            completeness = len(found_keys) / len(expected_keys)
        else:
            completeness = 1.0  # Multi-source queries don't have specific key expectations
        
        # Source coverage: of the routed sources, how many returned non-empty data?
        sources_with_data = sum(1 for s in routed_sources if context.get(s) not in [None, [], {}, ""])
        source_coverage = sources_with_data / len(routed_sources) if routed_sources else 0
        
        results.append({
            "query": query,
            "latency_ms": latency_ms,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "completeness": completeness,
            "source_coverage": source_coverage,
            "expected_sources": list(expected_sources),
            "routed_sources": routed_sources,
            "sources_with_data": sources_with_data,
        })
        
        status = "PASS" if recall >= 0.5 and completeness >= 0.5 else "WARN"
        print(f"  [{i+1:2d}/{len(TEST_CASES)}] {status} | Lat: {latency_ms:6.0f}ms | P: {precision:.2f} R: {recall:.2f} F1: {f1:.2f} | Comp: {completeness:.2f} | '{query[:50]}...'")
    
    # ── Aggregate Metrics ──────────────────────────────────────────────
    latencies = [r["latency_ms"] for r in results]
    precisions = [r["precision"] for r in results]
    recalls = [r["recall"] for r in results]
    f1s = [r["f1"] for r in results]
    completeness_scores = [r["completeness"] for r in results]
    coverages = [r["source_coverage"] for r in results]
    
    total_time = sum(latencies) / 1000
    throughput = len(results) / total_time if total_time > 0 else 0
    
    print("\n" + "=" * 60)
    print("  AGGREGATE RESULTS")
    print("=" * 60)
    print(f"  Total Queries:       {len(results)}")
    print(f"  Avg Latency:         {np.mean(latencies):.1f} ms")
    print(f"  P95 Latency:         {np.percentile(latencies, 95):.1f} ms")
    print(f"  Throughput:          {throughput:.2f} queries/sec")
    print(f"  Avg Precision:       {np.mean(precisions):.3f}")
    print(f"  Avg Recall:          {np.mean(recalls):.3f}")
    print(f"  Avg F1 Score:        {np.mean(f1s):.3f}")
    print(f"  Avg Completeness:    {np.mean(completeness_scores):.3f}")
    print(f"  Avg Source Coverage: {np.mean(coverages):.3f}")
    print("=" * 60)
    
    # ── Chart Styling ──────────────────────────────────────────────────
    sns.set_theme(style="whitegrid", font_scale=1.1)
    palette = sns.color_palette("viridis", 6)
    
    # ── Chart 1: Latency over queries ──────────────────────────────────
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(range(1, len(latencies)+1), latencies, color=palette[0], linewidth=2, marker='o', markersize=4, label='Latency')
    ax.axhline(np.mean(latencies), color='red', linestyle='--', linewidth=1.5, label=f'Avg: {np.mean(latencies):.0f}ms')
    ax.axhline(np.percentile(latencies, 95), color='orange', linestyle=':', linewidth=1.5, label=f'P95: {np.percentile(latencies, 95):.0f}ms')
    ax.set_title('RAG Context Retrieval Latency per Query', fontweight='bold')
    ax.set_xlabel('Query #')
    ax.set_ylabel('Latency (ms)')
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'rag_latency_line.png'), dpi=150)
    plt.close()
    
    # ── Chart 2: Data Completeness by Category ─────────────────────────
    categories = defaultdict(list)
    for r in results:
        # Categorize by primary expected source
        if len(r["expected_sources"]) == 1:
            categories[r["expected_sources"][0]].append(r["completeness"])
        else:
            categories["multi-source"].append(r["completeness"])
    
    cat_names = list(categories.keys())
    cat_means = [np.mean(v) for v in categories.values()]
    
    fig, ax = plt.subplots(figsize=(12, 5))
    bars = ax.bar(cat_names, cat_means, color=sns.color_palette("mako", len(cat_names)), edgecolor='white', linewidth=0.5)
    ax.bar_label(bars, fmt='%.2f', fontsize=9)
    ax.set_title('RAG Data Completeness by Query Category', fontweight='bold')
    ax.set_ylabel('Completeness Score (0-1)')
    ax.set_ylim(0, 1.15)
    plt.xticks(rotation=35, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'rag_completeness_bar.png'), dpi=150)
    plt.close()
    
    # ── Chart 3: Precision / Recall / F1 ───────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 5))
    metrics = ['Precision', 'Recall', 'F1 Score']
    scores = [np.mean(precisions), np.mean(recalls), np.mean(f1s)]
    colors = [palette[1], palette[3], palette[5]]
    bars = ax.bar(metrics, scores, color=colors, edgecolor='white', linewidth=0.5, width=0.5)
    ax.bar_label(bars, fmt='%.3f', fontsize=11, fontweight='bold')
    ax.set_title('RAG Query Router - Precision, Recall & F1', fontweight='bold')
    ax.set_ylabel('Score (0-1)')
    ax.set_ylim(0, 1.15)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'rag_precision_recall.png'), dpi=150)
    plt.close()
    
    # ── Chart 4: Source Coverage Heatmap ────────────────────────────────
    all_sources = sorted(rag_manager.ROUTE_MAP.keys())
    heatmap_data = []
    query_labels = []
    for r in results:
        row = []
        for s in all_sources:
            if s in r["routed_sources"]:
                # 1 = routed and has data, 0.5 = routed but empty, 0 = not routed
                has_data = 1.0
                row.append(has_data)
            else:
                row.append(0.0)
        heatmap_data.append(row)
        query_labels.append(r["query"][:35] + "...")
    
    fig, ax = plt.subplots(figsize=(14, 10))
    sns.heatmap(heatmap_data, xticklabels=all_sources, yticklabels=query_labels,
                cmap='YlGn', linewidths=0.5, ax=ax, cbar_kws={'label': 'Routed (1=Yes, 0=No)'})
    ax.set_title('RAG Source Routing Heatmap', fontweight='bold')
    ax.set_xlabel('Data Source')
    ax.set_ylabel('Query')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'rag_source_coverage_heatmap.png'), dpi=150)
    plt.close()
    
    # ── Chart 5: Throughput ────────────────────────────────────────────
    # Calculate running throughput (cumulative queries / cumulative time)
    cumulative_time = np.cumsum(latencies) / 1000  # seconds
    running_throughput = np.arange(1, len(latencies)+1) / cumulative_time
    
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(range(1, len(running_throughput)+1), running_throughput, color=palette[2], linewidth=2)
    ax.axhline(throughput, color='red', linestyle='--', label=f'Overall: {throughput:.2f} q/s')
    ax.set_title('RAG Retrieval Throughput (Queries per Second)', fontweight='bold')
    ax.set_xlabel('Query #')
    ax.set_ylabel('Throughput (q/s)')
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'rag_throughput.png'), dpi=150)
    plt.close()
    
    # ── Chart 6: Summary Dashboard ─────────────────────────────────────
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    fig.suptitle('NovaCare RAG System - Performance Dashboard', fontsize=16, fontweight='bold', y=0.98)
    
    # 6a: Latency Distribution
    ax = axes[0, 0]
    sns.histplot(latencies, bins=10, color=palette[0], kde=True, ax=ax)
    ax.axvline(np.mean(latencies), color='red', linestyle='--', label=f'Avg: {np.mean(latencies):.0f}ms')
    ax.set_title('Latency Distribution')
    ax.set_xlabel('Latency (ms)')
    ax.legend(fontsize=8)
    
    # 6b: P/R/F1 per category
    ax = axes[0, 1]
    cat_f1 = {}
    for r in results:
        key = r["expected_sources"][0] if len(r["expected_sources"]) == 1 else "multi"
        cat_f1.setdefault(key, []).append(r["f1"])
    cat_f1_means = {k: np.mean(v) for k, v in cat_f1.items()}
    ax.barh(list(cat_f1_means.keys()), list(cat_f1_means.values()), color=sns.color_palette("rocket", len(cat_f1_means)))
    ax.set_title('F1 Score by Category')
    ax.set_xlim(0, 1.1)
    
    # 6c: Source Coverage pie
    ax = axes[0, 2]
    avg_coverage = np.mean(coverages)
    ax.pie([avg_coverage, 1-avg_coverage], labels=['Covered', 'Missing'], 
           colors=[palette[3], '#eee'], autopct='%1.1f%%', startangle=90)
    ax.set_title('Avg Source Coverage')
    
    # 6d: Completeness distribution
    ax = axes[1, 0]
    sns.boxplot(data=completeness_scores, color=palette[4], ax=ax)
    ax.set_title('Data Completeness')
    ax.set_ylabel('Score')
    ax.set_ylim(-0.1, 1.1)
    
    # 6e: Throughput over time
    ax = axes[1, 1]
    ax.plot(range(1, len(running_throughput)+1), running_throughput, color=palette[2], linewidth=2)
    ax.set_title('Throughput Over Time')
    ax.set_xlabel('Query #')
    ax.set_ylabel('q/s')
    
    # 6f: Key metrics summary table
    ax = axes[1, 2]
    ax.axis('off')
    metrics_table = [
        ['Metric', 'Value'],
        ['Total Queries', str(len(results))],
        ['Avg Latency', f'{np.mean(latencies):.0f} ms'],
        ['P95 Latency', f'{np.percentile(latencies, 95):.0f} ms'],
        ['Throughput', f'{throughput:.2f} q/s'],
        ['Precision', f'{np.mean(precisions):.3f}'],
        ['Recall', f'{np.mean(recalls):.3f}'],
        ['F1 Score', f'{np.mean(f1s):.3f}'],
        ['Completeness', f'{np.mean(completeness_scores):.3f}'],
    ]
    table = ax.table(cellText=metrics_table, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.4)
    # Style header row
    for j in range(2):
        table[0, j].set_facecolor('#2c3e50')
        table[0, j].set_text_props(color='white', fontweight='bold')
    ax.set_title('Summary Metrics', pad=20)
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(os.path.join(output_dir, 'rag_metrics_summary.png'), dpi=150)
    plt.close()
    
    print(f"\n  6 charts saved to {output_dir}/")
    print("  Done!")


if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    run_benchmark(os.path.join(base_dir, "..", "..", "docs", "benchmarks"))
