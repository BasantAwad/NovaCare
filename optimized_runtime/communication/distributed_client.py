import asyncio
import json
import logging
from typing import Optional, Any, Callable, Dict

logger = logging.getLogger(__name__)

class DistributedClient:
    """
    WebSocket client for SERBot to communicate with Laptop AI services.
    Supports async request/response and automatic reconnection.
    """
    def __init__(self, uri: str = "ws://localhost:5000/ws/ai"):
        self.uri = uri
        self.websocket = None
        self.connected = False
        self._handlers: Dict[str, Callable] = {}
        self._pending_responses: Dict[str, asyncio.Future] = {}

    async def connect(self):
        """Connect to the remote AI service"""
        import websockets
        while True:
            try:
                logger.info(f"Connecting to {self.uri}...")
                self.websocket = await websockets.connect(self.uri)
                self.connected = True
                logger.info("Connected to remote AI services")
                
                # Start listener loop
                asyncio.ensure_future(self._listener_loop())
                break
            except Exception as e:
                logger.error(f"Connection failed: {e}. Retrying in 5s...")
                await asyncio.sleep(5)

    async def _listener_loop(self):
        """Listen for messages from the remote service"""
        try:
            async for message in self.websocket:
                data = json.loads(message)
                msg_id = data.get("id")
                
                # If it's a response to a pending request
                if msg_id and msg_id in self._pending_responses:
                    self._pending_responses[msg_id].set_result(data.get("payload"))
                    del self._pending_responses[msg_id]
                
                # Handle general events
                msg_type = data.get("type")
                if msg_type in self._handlers:
                    await self._handlers[msg_type](data.get("payload"))
                    
        except Exception as e:
            logger.error(f"Listener loop error: {e}")
            self.connected = False

    async def call_remote(self, method: str, params: Any) -> Any:
        """Call a remote AI method and wait for response"""
        if not self.connected:
            await self.connect()
        
        msg_id = str(hash(method + str(params))) # Simple ID
        future = asyncio.get_event_loop().create_future()
        self._pending_responses[msg_id] = future
        
        request = {
            "id": msg_id,
            "type": "request",
            "method": method,
            "params": params
        }
        
        await self.websocket.send(json.dumps(request))
        return await future

    def register_handler(self, msg_type: str, handler: Callable):
        self._handlers[msg_type] = handler

_instance = None

def get_distributed_client() -> DistributedClient:
    global _instance
    if _instance is None:
        _instance = DistributedClient()
    return _instance
