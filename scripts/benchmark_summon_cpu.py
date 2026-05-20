"""
Benchmark CPU Usage for Summon Controller
=========================================
Tests the optimized runtime's CPU utilization during a simulated
Summon operations.
"""

import sys
import os
import time
import asyncio

# Ensure project root is in path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from optimized_runtime.orchestrator.runtime_orchestrator import RuntimeOrchestrator
from optimized_runtime.summon.protocol import SummonMessageType
from optimized_runtime.summon.summon_controller import get_summon_controller

async def run_benchmark():
    print("="*50)
    print(" Benchmarking Summon Controller CPU Usage")
    print("="*50)
    
    # Initialize orchestrator
    orchestrator = RuntimeOrchestrator()
    await orchestrator.initialize()
    
    # Give it a second to settle
    await asyncio.sleep(1.0)
    
    # Baseline
    metrics = orchestrator.get_metrics()
    print(f"[Baseline] CPU: {metrics['system_resources']['cpu_percent']}% | Mem: {metrics['system_resources']['memory_mb']}MB")
    
    # Simulate a summon request via the controller
    controller = get_summon_controller()
    print("\n[Simulating Summon Request]")
    await controller._handle_summon_request({
        "type": SummonMessageType.SUMMON_REQUEST.value,
        "request_id": "bench-123",
        "user_id": "test_user",
        "ble_mac": "00:11:22:33:44:55",
        "timestamp": time.time()
    })
    
    print("Navigating (Simulated) for 10 seconds...")
    samples = []
    
    for i in range(10):
        # Simulate RSSI update
        await controller._handle_rssi_update({
            "type": SummonMessageType.RSSI_UPDATE.value,
            "rssi": -60 + (i % 5),
            "timestamp": time.time()
        })
        await asyncio.sleep(1.0)
        metrics = orchestrator.get_metrics()
        cpu = metrics['system_resources']['cpu_percent']
        mem = metrics['system_resources']['memory_mb']
        samples.append(cpu)
        print(f"  Sec {i+1}: CPU {cpu}% | Mem {mem}MB | State: {controller.state_machine.state.value}")
        
    # Cancel summon
    await controller._handle_summon_cancel({"reason": "benchmark_complete"})
    
    avg_cpu = sum(samples) / len(samples) if samples else 0
    print("\n" + "="*50)
    print(f" Benchmark Complete")
    print(f" Average CPU Usage: {avg_cpu:.2f}%")
    print("="*50)
    
    if avg_cpu < 5.0:
        print("[SUCCESS] CPU usage well within acceptable limits (< 5%).")
    else:
        print("[WARNING] CPU usage higher than expected.")
    
    # Stop orchestrator
    await orchestrator.stop()

if __name__ == "__main__":
    asyncio.run(run_benchmark())
