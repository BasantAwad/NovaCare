import logging
from typing import Any
from .inference_manager import get_inference_manager
from ..communication.distributed_client import get_distributed_client

logger = logging.getLogger(__name__)

class WorkloadDistributor:
    """
    Decides where to run AI tasks based on complexity and connection status.
    """
    def __init__(self):
        self.inference_manager = get_inference_manager()
        self.remote_client = get_distributed_client()

    async def process(self, task_type: str, data: Any) -> Any:
        """
        Processes a task by routing it to the optimal provider.
        """
        # Tasks that SHOULD be remote if possible
        HEAVY_TASKS = ["llm", "asl_complex", "mental_health_analysis"]
        
        # Tasks that SHOULD be local
        LIGHT_TASKS = ["face_detection", "wake_word", "collision_avoidance"]

        if task_type in HEAVY_TASKS:
            if self.remote_client.connected:
                logger.info(f"Routing {task_type} to REMOTE Laptop")
                return await self.remote_client.call_remote(task_type, data)
            else:
                logger.warning(f"Remote unavailable! Falling back to LOCAL for {task_type}")
                return await self.inference_manager.run_inference(task_type, data)
        
        return await self.inference_manager.run_inference(task_type, data)

_instance = None

def get_workload_distributor() -> WorkloadDistributor:
    global _instance
    if _instance is None:
        _instance = WorkloadDistributor()
    return _instance
