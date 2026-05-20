"""
Monitoring and Health Check Module

Tracks system health, performance, and resource usage.
"""

import asyncio
import logging
import psutil
from typing import Dict, Any
from datetime import datetime

logger = logging.getLogger(__name__)


class HealthMonitor:
    """Monitors system health and resource usage"""
    
    def __init__(self):
        self.start_time = datetime.now()
        self.checks: Dict[str, Any] = {}
    
    async def run_checks(self) -> Dict[str, Any]:
        """Run all health checks"""
        return {
            "cpu": self._check_cpu(),
            "memory": self._check_memory(),
            "disk": self._check_disk(),
            "timestamp": datetime.now().isoformat(),
        }
    
    def _check_cpu(self) -> Dict[str, Any]:
        """Check CPU usage"""
        try:
            cpu_percent = psutil.cpu_percent(interval=0.1)
            cpu_count = psutil.cpu_count()
            return {
                "usage_percent": cpu_percent,
                "count": cpu_count,
                "healthy": cpu_percent < 80,
            }
        except Exception as e:
            logger.error(f"CPU check failed: {e}")
            return {"error": str(e)}
    
    def _check_memory(self) -> Dict[str, Any]:
        """Check memory usage"""
        try:
            memory = psutil.virtual_memory()
            return {
                "available_mb": memory.available / (1024 * 1024),
                "percent_used": memory.percent,
                "healthy": memory.percent < 80,
            }
        except Exception as e:
            logger.error(f"Memory check failed: {e}")
            return {"error": str(e)}
    
    def _check_disk(self) -> Dict[str, Any]:
        """Check disk usage"""
        try:
            disk = psutil.disk_usage('/')
            return {
                "free_gb": disk.free / (1024 * 1024 * 1024),
                "percent_used": disk.percent,
                "healthy": disk.percent < 90,
            }
        except Exception as e:
            logger.error(f"Disk check failed: {e}")
            return {"error": str(e)}


class PerformanceTracker:
    """Tracks performance metrics"""
    
    def __init__(self):
        self.metrics: Dict[str, list] = {
            "latency": [],
            "throughput": [],
            "errors": [],
        }
    
    def record_latency(self, operation: str, duration_ms: float):
        """Record operation latency"""
        if "latency" not in self.metrics:
            self.metrics["latency"] = []
        self.metrics["latency"].append({
            "operation": operation,
            "duration_ms": duration_ms,
            "timestamp": datetime.now().isoformat(),
        })
    
    def record_error(self, operation: str, error: str):
        """Record operation error"""
        if "errors" not in self.metrics:
            self.metrics["errors"] = []
        self.metrics["errors"].append({
            "operation": operation,
            "error": error,
            "timestamp": datetime.now().isoformat(),
        })
    
    def get_summary(self) -> Dict[str, Any]:
        """Get performance summary"""
        return {
            "total_operations": len(self.metrics.get("latency", [])),
            "total_errors": len(self.metrics.get("errors", [])),
            "avg_latency_ms": self._avg_latency(),
            "error_rate": self._error_rate(),
        }
    
    def _avg_latency(self) -> float:
        """Calculate average latency"""
        latencies = self.metrics.get("latency", [])
        if not latencies:
            return 0.0
        return sum(m["duration_ms"] for m in latencies) / len(latencies)
    
    def _error_rate(self) -> float:
        """Calculate error rate"""
        total = len(self.metrics.get("latency", []))
        errors = len(self.metrics.get("errors", []))
        if total == 0:
            return 0.0
        return (errors / total) * 100
