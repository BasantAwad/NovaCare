"""
Telemetry for Optimized Runtime
===============================
Provides system resource monitoring (CPU and Memory) to ensure
the runtime stays within strict constraints (e.g., <1% CPU for Bug2 vs SLAM).
"""

import os
import psutil
from typing import Dict, Any

class SystemTelemetry:
    def __init__(self):
        self.process = psutil.Process(os.getpid())
        # First call to cpu_percent usually returns 0.0, so initialize it
        self.process.cpu_percent(interval=None)

    def get_metrics(self) -> Dict[str, Any]:
        """
        Return current CPU and Memory usage of the orchestrator process.
        Returns:
            dict containing CPU percentage, Memory MB, and Threads count.
        """
        try:
            # interval=None calculates CPU usage since the last call
            cpu_usage = self.process.cpu_percent(interval=None)
            memory_info = self.process.memory_info()
            memory_mb = memory_info.rss / (1024 * 1024)
            num_threads = self.process.num_threads()

            return {
                "cpu_percent": round(cpu_usage, 2),
                "memory_mb": round(memory_mb, 2),
                "threads": num_threads
            }
        except Exception as e:
            return {"error": str(e)}

_telemetry_instance = None

def get_telemetry() -> SystemTelemetry:
    global _telemetry_instance
    if _telemetry_instance is None:
        _telemetry_instance = SystemTelemetry()
    return _telemetry_instance
