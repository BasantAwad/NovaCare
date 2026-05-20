#!/usr/bin/env python3
"""
Measure round-trip latency for NovaCare edge TTS (proxy) and optional Pocket TTS upstream.

Run on the Jetson (or any host where Pocket + proxy are listening):

  pip install requests
  python benchmark_tts_latency.py
  python benchmark_tts_latency.py --url http://127.0.0.1:8765 --samples 5

Reports: cold/warm median time to full WAV response (short phrase). Use results to decide
whether Pocket TTS is acceptable on Jetson Nano or a lighter fallback is needed.
"""
from __future__ import annotations

import argparse
import statistics
import time
from typing import List

import requests


def timed_post(url: str, text: str, timeout: float) -> tuple[float, int]:
    t0 = time.perf_counter()
    r = requests.post(
        f"{url.rstrip('/')}/api/speak",
        json={"text": text},
        timeout=timeout,
    )
    elapsed = time.perf_counter() - t0
    r.raise_for_status()
    n = len(r.content)
    return elapsed, n


def main() -> None:
    p = argparse.ArgumentParser(description="Benchmark edge TTS HTTP latency")
    p.add_argument(
        "--url",
        default="http://127.0.0.1:8765",
        help="NovaCare edge-tts-proxy base URL",
    )
    p.add_argument("--samples", type=int, default=5, help="Number of timed requests after warmup")
    p.add_argument("--timeout", type=float, default=180.0, help="Per-request timeout seconds")
    p.add_argument(
        "--phrase",
        default="Hello. I am NovaBot. How can I help you today?",
        help="Short English phrase for synthesis",
    )
    args = p.parse_args()

    base = args.url.rstrip("/")
    print(f"Target: {base}/api/speak")
    print("Warmup (1 request, not scored)...")
    timed_post(base, args.phrase, args.timeout)

    times: List[float] = []
    for i in range(args.samples):
        dt, nbytes = timed_post(base, args.phrase, args.timeout)
        times.append(dt)
        print(f"  sample {i + 1}: {dt:.3f}s, {nbytes} bytes WAV")

    print("---")
    print(f"median: {statistics.median(times):.3f}s")
    print(f"min:    {min(times):.3f}s")
    print(f"max:    {max(times):.3f}s")
    print("Log these numbers in your Jetson sizing notes (thermal throttling affects max).")


if __name__ == "__main__":
    main()
