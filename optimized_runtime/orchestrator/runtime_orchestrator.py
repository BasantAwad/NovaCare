"""
Runtime Orchestrator — Central coordinator for all robot operations

Manages:
- Audio pipelines (microphone, STT, speaker, TTS)
- Camera operations and vision processing
- Inference dispatching and queuing
- State synchronization
- Event coordination
- Service lifecycle
"""

import asyncio
import logging
from typing import Optional, Callable, Dict, Any
from dataclasses import dataclass
from datetime import datetime
from enum import Enum

from ..state import get_robot_state, EmotionType, RobotMode
from ..communication import get_websocket_server

logger = logging.getLogger(__name__)


class PipelineTask(Enum):
    """Pipeline task types"""
    AUDIO_CAPTURE = "audio_capture"
    CAMERA_CAPTURE = "camera_capture"
    STT = "stt"  # Speech-To-Text
    LLM_INFERENCE = "llm_inference"
    EMOTION_DETECTION = "emotion_detection"
    ASL_DETECTION = "asl_detection"
    TTS = "tts"  # Text-To-Speech
    ANIMATION = "animation"


@dataclass
class Task:
    """Queued task for async processing"""
    task_type: PipelineTask
    data: Any
    priority: int = 0  # Higher = more important
    created_at: datetime = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()
    
    def __lt__(self, other):
        """For priority queue ordering (higher priority first)"""
        if self.priority != other.priority:
            return self.priority > other.priority
        return self.created_at < other.created_at


class RuntimeOrchestrator:
    """
    Central orchestrator managing all robot runtime operations.
    
    Coordinates:
    - Audio input/output pipelines
    - Camera operations
    - Inference task queuing
    - State management
    - Event broadcasting
    - Service lifecycle
    
    Designed for async operation and optimal resource utilization
    on constrained SERBot hardware.
    """
    
    def __init__(self):
        self.state = get_robot_state()
        self.ws_server = get_websocket_server()
        
        # Pipeline management
        self.task_queue: asyncio.PriorityQueue[Task] = asyncio.PriorityQueue()
        self.max_queue_size = 100
        self.active_tasks: Dict[PipelineTask, bool] = {}
        
        # Pipeline callbacks
        self._audio_capture_callback: Optional[Callable] = None
        self._camera_capture_callback: Optional[Callable] = None
        self._inference_callback: Optional[Callable] = None
        self._animation_callback: Optional[Callable] = None
        
        # Lifecycle
        self._running = False
        self._initialized = False
        self.start_time: Optional[datetime] = None
        
        # Performance metrics
        self.metrics = {
            "tasks_processed": 0,
            "tasks_failed": 0,
            "avg_processing_time": 0.0,
            "queue_depth": 0,
        }
    
    async def initialize(self):
        """Initialize orchestrator and all subsystems"""
        logger.info("Initializing RuntimeOrchestrator...")
        
        try:
            # Start WebSocket server
            await self.ws_server.start()
            
            # Initialize state
            await self.state.set_mode(RobotMode.IDLE)
            await self.state.set_emotion(EmotionType.IDLE)
            
            # Register state change handlers
            self._register_state_handlers()
            
            self._initialized = True
            logger.info("RuntimeOrchestrator initialized successfully")
        
        except Exception as e:
            logger.error(f"Initialization failed: {e}")
            raise
    
    async def start(self):
        """Start the runtime orchestrator"""
        if not self._initialized:
            await self.initialize()
        
        self._running = True
        self.start_time = datetime.now()
        logger.info("RuntimeOrchestrator started")
        
        # Start pipeline workers
        await asyncio.gather(
            self._audio_pipeline(),
            self._camera_pipeline(),
            self._inference_pipeline(),
            self._state_broadcast_loop(),
            return_exceptions=True,
        )
    
    async def stop(self):
        """Stop the runtime orchestrator"""
        self._running = False
        logger.info("RuntimeOrchestrator stopping...")
        await self.ws_server.stop()
        logger.info("RuntimeOrchestrator stopped")
    
    # ========== AUDIO PIPELINE ==========
    
    def set_audio_capture_callback(self, callback: Callable):
        """Set callback for audio capture processing"""
        self._audio_capture_callback = callback
    
    async def _audio_pipeline(self):
        """Main audio processing pipeline"""
        logger.info("Audio pipeline started")
        
        while self._running:
            try:
                # Check for new audio tasks
                if not self.task_queue.empty():
                    try:
                        _, task = self.task_queue.get_nowait()
                        
                        if task.task_type == PipelineTask.AUDIO_CAPTURE:
                            await self._process_audio_capture(task)
                        elif task.task_type == PipelineTask.STT:
                            await self._process_stt(task)
                        elif task.task_type == PipelineTask.TTS:
                            await self._process_tts(task)
                    
                    except asyncio.QueueEmpty:
                        pass
                
                await asyncio.sleep(0.01)  # Prevent busy-waiting
            
            except Exception as e:
                logger.error(f"Audio pipeline error: {e}")
                await self.state.set_mode(RobotMode.ERROR)
    
    async def _process_audio_capture(self, task: Task):
        """Process audio capture task"""
        try:
            if self._audio_capture_callback:
                result = self._audio_capture_callback(task.data)
                if asyncio.iscoroutine(result):
                    await result
            
            self.metrics["tasks_processed"] += 1
        
        except Exception as e:
            logger.error(f"Audio capture error: {e}")
            self.metrics["tasks_failed"] += 1
    
    async def _process_stt(self, task: Task):
        """Process Speech-To-Text task"""
        try:
            # This would dispatch to laptop STT service via adapter
            logger.debug(f"STT task queued: {task.data}")
            self.metrics["tasks_processed"] += 1
        
        except Exception as e:
            logger.error(f"STT error: {e}")
            self.metrics["tasks_failed"] += 1
    
    async def _process_tts(self, task: Task):
        """Process Text-To-Speech task"""
        try:
            logger.debug(f"TTS task queued: {task.data}")
            
            # Play audio
            await self.state.set_speaking(True)
            
            # Simulate playback
            await asyncio.sleep(len(task.data.get("text", "")) * 0.05)
            
            await self.state.set_speaking(False)
            self.metrics["tasks_processed"] += 1
        
        except Exception as e:
            logger.error(f"TTS error: {e}")
            self.metrics["tasks_failed"] += 1
    
    # ========== CAMERA PIPELINE ==========
    
    def set_camera_capture_callback(self, callback: Callable):
        """Set callback for camera frame processing"""
        self._camera_capture_callback = callback
    
    async def _camera_pipeline(self):
        """Main camera processing pipeline"""
        logger.info("Camera pipeline started")
        
        while self._running:
            try:
                # Process camera frames
                if not self.task_queue.empty():
                    try:
                        _, task = self.task_queue.get_nowait()
                        
                        if task.task_type == PipelineTask.CAMERA_CAPTURE:
                            await self._process_camera_capture(task)
                        elif task.task_type == PipelineTask.EMOTION_DETECTION:
                            await self._process_emotion_detection(task)
                        elif task.task_type == PipelineTask.ASL_DETECTION:
                            await self._process_asl_detection(task)
                    
                    except asyncio.QueueEmpty:
                        pass
                
                await asyncio.sleep(0.033)  # ~30 FPS
            
            except Exception as e:
                logger.error(f"Camera pipeline error: {e}")
    
    async def _process_camera_capture(self, task: Task):
        """Process camera frame"""
        try:
            if self._camera_capture_callback:
                result = self._camera_capture_callback(task.data)
                if asyncio.iscoroutine(result):
                    await result
            
            self.metrics["tasks_processed"] += 1
        
        except Exception as e:
            logger.error(f"Camera capture error: {e}")
            self.metrics["tasks_failed"] += 1
    
    async def _process_emotion_detection(self, task: Task):
        """Process emotion detection task"""
        try:
            logger.debug("Emotion detection dispatched to laptop service")
            self.metrics["tasks_processed"] += 1
        
        except Exception as e:
            logger.error(f"Emotion detection error: {e}")
            self.metrics["tasks_failed"] += 1
    
    async def _process_asl_detection(self, task: Task):
        """Process ASL detection task"""
        try:
            logger.debug("ASL detection dispatched to laptop service")
            self.metrics["tasks_processed"] += 1
        
        except Exception as e:
            logger.error(f"ASL detection error: {e}")
            self.metrics["tasks_failed"] += 1
    
    # ========== INFERENCE PIPELINE ==========
    
    def set_inference_callback(self, callback: Callable):
        """Set callback for inference results"""
        self._inference_callback = callback
    
    async def _inference_pipeline(self):
        """Main inference processing pipeline"""
        logger.info("Inference pipeline started")
        
        while self._running:
            try:
                if not self.task_queue.empty():
                    try:
                        _, task = self.task_queue.get_nowait()
                        
                        if task.task_type == PipelineTask.LLM_INFERENCE:
                            await self._process_llm_inference(task)
                    
                    except asyncio.QueueEmpty:
                        pass
                
                await asyncio.sleep(0.1)
            
            except Exception as e:
                logger.error(f"Inference pipeline error: {e}")
    
    async def _process_llm_inference(self, task: Task):
        """Process LLM inference task"""
        try:
            await self.state.set_mode(RobotMode.PROCESSING)
            
            logger.debug(f"LLM inference: {task.data}")
            
            if self._inference_callback:
                result = self._inference_callback(task.data)
                if asyncio.iscoroutine(result):
                    await result
            
            self.metrics["tasks_processed"] += 1
            await self.state.set_mode(RobotMode.IDLE)
        
        except Exception as e:
            logger.error(f"LLM inference error: {e}")
            self.metrics["tasks_failed"] += 1
            await self.state.set_mode(RobotMode.ERROR)
    
    # ========== TASK QUEUEING ==========
    
    async def queue_task(self, task: Task) -> bool:
        """Queue a task for processing"""
        try:
            if self.task_queue.qsize() >= self.max_queue_size:
                logger.warning(f"Task queue full ({self.max_queue_size})")
                return False
            
            await self.task_queue.put((task.priority, task))
            self.metrics["queue_depth"] = self.task_queue.qsize()
            return True
        
        except Exception as e:
            logger.error(f"Queue task error: {e}")
            return False
    
    async def queue_audio_capture(self, data: Any, priority: int = 5) -> bool:
        """Queue audio capture"""
        return await self.queue_task(Task(PipelineTask.AUDIO_CAPTURE, data, priority))
    
    async def queue_camera_capture(self, data: Any, priority: int = 5) -> bool:
        """Queue camera capture"""
        return await self.queue_task(Task(PipelineTask.CAMERA_CAPTURE, data, priority))
    
    async def queue_stt(self, audio_data: Any, priority: int = 10) -> bool:
        """Queue STT task"""
        return await self.queue_task(Task(PipelineTask.STT, audio_data, priority))
    
    async def queue_tts(self, text: str, priority: int = 8) -> bool:
        """Queue TTS task"""
        return await self.queue_task(Task(PipelineTask.TTS, {"text": text}, priority))
    
    async def queue_llm_inference(self, prompt: str, priority: int = 10) -> bool:
        """Queue LLM inference"""
        return await self.queue_task(Task(PipelineTask.LLM_INFERENCE, {"prompt": prompt}, priority))
    
    async def queue_emotion_detection(self, data: Any, priority: int = 3) -> bool:
        """Queue emotion detection"""
        return await self.queue_task(Task(PipelineTask.EMOTION_DETECTION, data, priority))
    
    async def queue_asl_detection(self, frame: Any, priority: int = 2) -> bool:
        """Queue ASL detection"""
        return await self.queue_task(Task(PipelineTask.ASL_DETECTION, frame, priority))
    
    # ========== STATE BROADCAST ==========
    
    async def _state_broadcast_loop(self):
        """Periodically broadcast state to all connected clients"""
        logger.info("State broadcast loop started")
        
        while self._running:
            try:
                state_dict = self.state.get_full_state()
                await self.ws_server.broadcast_state(state_dict)
                await asyncio.sleep(0.1)  # 10 Hz state updates
            
            except Exception as e:
                logger.error(f"Broadcast error: {e}")
                await asyncio.sleep(1)
    
    # ========== STATE HANDLERS ==========
    
    def _register_state_handlers(self):
        """Register callbacks for state changes"""
        self.state.on_state_change("emotion", self._on_emotion_change)
        self.state.on_state_change("audio.speaking", self._on_speaking_change)
        self.state.on_state_change("audio.listening", self._on_listening_change)
    
    async def _on_emotion_change(self, old_value: Any, new_value: Any):
        """Handle emotion change"""
        logger.debug(f"Emotion changed: {old_value} → {new_value}")
    
    async def _on_speaking_change(self, old_value: Any, new_value: Any):
        """Handle speaking state change"""
        logger.debug(f"Speaking changed: {old_value} → {new_value}")
    
    async def _on_listening_change(self, old_value: Any, new_value: Any):
        """Handle listening state change"""
        logger.debug(f"Listening changed: {old_value} → {new_value}")
    
    # ========== METRICS ==========
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current orchestrator metrics"""
        uptime = None
        if self.start_time:
            uptime = (datetime.now() - self.start_time).total_seconds()
        
        return {
            **self.metrics,
            "uptime_seconds": uptime,
            "is_running": self._running,
            "websocket_connections": self.ws_server.connection_count,
        }
