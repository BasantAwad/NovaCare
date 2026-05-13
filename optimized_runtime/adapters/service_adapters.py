"""
Service Adapters — Wrappers for existing NovaCare services

Provides unified interface to:
- Existing LLM backend (Flask port 5000)
- Existing ASL service (FastAPI port 8000)
- Existing TTS service (edge-tts-proxy)
- Existing robot HAL (port 9000)

Adapters handle:
- Network communication (HTTP, WebSocket)
- Request/response marshaling
- Error handling and retries
- Service discovery and health checks
"""

import asyncio
import logging
import httpx
from typing import Optional, Dict, Any
from datetime import datetime

logger = logging.getLogger(__name__)


class ServiceAdapter:
    """Base adapter for external services"""
    
    def __init__(self, service_name: str, base_url: str, timeout: float = 10.0):
        self.service_name = service_name
        self.base_url = base_url
        self.timeout = timeout
        self.is_available = False
        self.last_health_check = None
        self.client: Optional[httpx.AsyncClient] = None
    
    async def initialize(self):
        """Initialize HTTP client"""
        self.client = httpx.AsyncClient(timeout=self.timeout)
    
    async def shutdown(self):
        """Shutdown HTTP client"""
        if self.client:
            await self.client.aclose()
    
    async def health_check(self) -> bool:
        """Check if service is available"""
        try:
            response = await self.client.get(
                f"{self.base_url}/health",
                timeout=self.timeout / 2
            )
            self.is_available = response.status_code == 200
            self.last_health_check = datetime.now()
            return self.is_available
        
        except Exception as e:
            logger.warning(f"{self.service_name} health check failed: {e}")
            self.is_available = False
            return False
    
    async def _make_request(
        self,
        method: str,
        endpoint: str,
        **kwargs
    ) -> Optional[Dict[str, Any]]:
        """Make HTTP request to service"""
        if not self.is_available:
            if not await self.health_check():
                return None
        
        try:
            url = f"{self.base_url}{endpoint}"
            response = await self.client.request(method, url, **kwargs)
            response.raise_for_status()
            return response.json()
        
        except Exception as e:
            logger.error(f"Request error: {e}")
            self.is_available = False
            return None


class LLMServiceAdapter(ServiceAdapter):
    """Adapter for LLM Backend (Flask port 5000)"""
    
    def __init__(self, base_url: str = "http://localhost:5000"):
        super().__init__("LLM Service", base_url)
    
    async def chat(self, message: str) -> Optional[str]:
        """Send message to LLM service"""
        response = await self._make_request(
            "POST",
            "/api/chat",
            json={"message": message}
        )
        
        if response:
            return response.get("response")
        return None
    
    async def clear_history(self) -> bool:
        """Clear conversation history"""
        response = await self._make_request("POST", "/api/clear")
        return response is not None


class ASLServiceAdapter(ServiceAdapter):
    """Adapter for ASL Service (FastAPI port 8000)"""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        super().__init__("ASL Service", base_url)
    
    async def predict_asl(self, frame_data: bytes) -> Optional[Dict[str, Any]]:
        """Predict ASL from video frame"""
        try:
            response = await self._make_request(
                "POST",
                "/predict",
                content=frame_data,
                headers={"Content-Type": "image/jpeg"}
            )
            return response
        
        except Exception as e:
            logger.error(f"ASL prediction error: {e}")
            return None
    
    async def reset_model(self) -> bool:
        """Reset ASL model"""
        response = await self._make_request("POST", "/reset")
        return response is not None


class RobotServiceAdapter(ServiceAdapter):
    """Adapter for Robot HAL (port 9000)"""
    
    def __init__(self, base_url: str = "http://localhost:9000"):
        super().__init__("Robot Service", base_url)
    
    async def move_forward(self, distance: float) -> bool:
        """Move robot forward"""
        response = await self._make_request(
            "POST",
            "/api/move",
            json={"direction": "forward", "distance": distance}
        )
        return response is not None
    
    async def move_backward(self, distance: float) -> bool:
        """Move robot backward"""
        response = await self._make_request(
            "POST",
            "/api/move",
            json={"direction": "backward", "distance": distance}
        )
        return response is not None
    
    async def turn(self, angle: float) -> bool:
        """Turn robot"""
        response = await self._make_request(
            "POST",
            "/api/turn",
            json={"angle": angle}
        )
        return response is not None
    
    async def stop(self) -> bool:
        """Stop all robot movement"""
        response = await self._make_request("POST", "/api/stop")
        return response is not None
    
    async def set_led(self, color: str, brightness: int = 255) -> bool:
        """Set LED color"""
        response = await self._make_request(
            "POST",
            "/api/led",
            json={"color": color, "brightness": brightness}
        )
        return response is not None
    
    async def play_sound(self, frequency: int, duration: float) -> bool:
        """Play sound via speaker"""
        response = await self._make_request(
            "POST",
            "/api/sound",
            json={"frequency": frequency, "duration": duration}
        )
        return response is not None


class TTSServiceAdapter(ServiceAdapter):
    """Adapter for TTS Service (edge-tts-proxy)"""
    
    def __init__(self, base_url: str = "http://localhost:8002"):
        super().__init__("TTS Service", base_url)
    
    async def synthesize(self, text: str, voice: str = "en-US-AvaNeural") -> Optional[bytes]:
        """Synthesize speech from text"""
        response = await self._make_request(
            "POST",
            "/tts",
            json={"text": text, "voice": voice}
        )
        
        if response and "audio" in response:
            # Base64 encoded audio
            import base64
            return base64.b64decode(response["audio"])
        
        return None


class ServiceRegistry:
    """Central registry for all service adapters"""
    
    def __init__(self):
        self.adapters: Dict[str, ServiceAdapter] = {}
        self._lock = asyncio.Lock()
    
    def register(self, name: str, adapter: ServiceAdapter):
        """Register a service adapter"""
        self.adapters[name] = adapter
        logger.info(f"Registered service: {name}")
    
    async def initialize_all(self):
        """Initialize all adapters"""
        for name, adapter in self.adapters.items():
            try:
                await adapter.initialize()
                logger.info(f"Initialized {name}")
            except Exception as e:
                logger.error(f"Failed to initialize {name}: {e}")
    
    async def shutdown_all(self):
        """Shutdown all adapters"""
        for name, adapter in self.adapters.items():
            try:
                await adapter.shutdown()
                logger.info(f"Shutdown {name}")
            except Exception as e:
                logger.error(f"Failed to shutdown {name}: {e}")
    
    async def health_check_all(self) -> Dict[str, bool]:
        """Check health of all services"""
        results = {}
        for name, adapter in self.adapters.items():
            results[name] = await adapter.health_check()
        return results
    
    def get(self, name: str) -> Optional[ServiceAdapter]:
        """Get adapter by name"""
        return self.adapters.get(name)
    
    @property
    def available_services(self) -> list[str]:
        """Get list of available services"""
        return [name for name, adapter in self.adapters.items() if adapter.is_available]


# Global registry
_registry: Optional[ServiceRegistry] = None


def get_service_registry() -> ServiceRegistry:
    """Get or create global service registry"""
    global _registry
    if _registry is None:
        _registry = ServiceRegistry()
        
        # Register default services
        _registry.register("llm", LLMServiceAdapter())
        _registry.register("asl", ASLServiceAdapter())
        _registry.register("robot", RobotServiceAdapter())
        _registry.register("tts", TTSServiceAdapter())
    
    return _registry
