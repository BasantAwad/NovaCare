import time
import logging
import psutil
import os
from typing import Dict, Any

logger = logging.getLogger(__name__)

class PerformanceMonitor:
    """
    Monitors system resources and runtime metrics.
    """
    def __init__(self):
        self.process = psutil.Process(os.getpid())

    def get_system_metrics(self) -> Dict[str, Any]:
        """Get CPU and RAM usage"""
        return {
            "cpu_percent": psutil.cpu_percent(interval=1),
            "memory_percent": psutil.virtual_memory().percent,
            "process_memory_mb": self.process.memory_info().rss / (1024 * 1024),
            "temperature": self._get_temp()
        }

    def _get_temp(self) -> float:
        """Attempt to read CPU temperature (Jetson/Linux specific)"""
        try:
            with open("/sys/class/thermal/thermal_zone0/temp", "r") as f:
                return int(f.read().strip()) / 1000.0
        except:
            return 0.0

def benchmark_inference(func: callable, data: Any, iterations: int = 100):
    """Benchmark an inference function"""
    start_time = time.time()
    for _ in range(iterations):
        func(data)
    end_time = time.time()
    
    total_time = end_time - start_time
    avg_time = total_time / iterations
    fps = 1.0 / avg_time
    
    print(f"--- Benchmark Result ---")
    print(f"Iterations: {iterations}")
    print(f"Total Time: {total_time:.4f}s")
    print(f"Avg Time:   {avg_time:.4f}s")
    print(f"Est. FPS:   {fps:.2f}")
    
    return avg_time

if __name__ == "__main__":
    monitor = PerformanceMonitor()
    print("Initial Metrics:", monitor.get_system_metrics())
