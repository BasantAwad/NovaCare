"""
WebSocket Communication Layer

Manages bidirectional communication with robot UIs.
Broadcasts robot state updates in real-time.
Handles incoming commands from frontend.
"""

import asyncio
import json
import os
from typing import Set, Callable, Any, Optional
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class WebSocketServer:
    """
    Manages WebSocket connections for real-time robot state updates
    and command handling.
    
    Maintains active connections and broadcasts state changes
    to all connected clients.
    """
    
    def __init__(self, host: str = "0.0.0.0", port: int = 9999):
        # Allow runtime override of websocket port via environment
        env_port = os.getenv("RUNTIME_PORT") or os.getenv("NOVACARE_WS_PORT")
        if env_port:
            try:
                port = int(env_port)
            except Exception:
                pass

        self.host = host
        self.port = port
        self.active_connections: Set[Any] = set()
        self.message_handlers: dict[str, Callable] = {}
        self.server = None
        self._running = False
        
        # Event emitters for different message types
        self._handlers: dict[str, list[Callable]] = {
            "emotion": [],
            "audio": [],
            "hardware": [],
            "animation": [],
            "command": [],
            "error": [],
        }
    
    async def start(self):
        """Start WebSocket server"""
        import websockets
        
        async def handler(websocket, path):
            """Handle new WebSocket connection"""
            await self._handle_connection(websocket)
        
        # Try to bind to the configured port, with graceful fallback
        try:
            self.server = await websockets.serve(
                handler,
                self.host,
                self.port,
                ping_interval=20,
                ping_timeout=20,
            )
        except OSError as e:
            logger.warning(f"Port {self.port} unavailable: {e}; attempting fallback ports")
            bound = False
            # Try a small range of alternative ports
            for candidate in range(self.port + 1, self.port + 11):
                try:
                    self.server = await websockets.serve(
                        handler,
                        self.host,
                        candidate,
                        ping_interval=20,
                        ping_timeout=20,
                    )
                    self.port = candidate
                    bound = True
                    break
                except OSError:
                    continue

            if not bound:
                # Last resort: bind to an ephemeral port
                self.server = await websockets.serve(
                    handler,
                    self.host,
                    0,
                    ping_interval=20,
                    ping_timeout=20,
                )
                # update to the actual bound port
                if getattr(self.server, "sockets", None):
                    try:
                        self.port = self.server.sockets[0].getsockname()[1]
                    except Exception:
                        pass

        # Update port if server bound to an OS-selected port
        if getattr(self.server, "sockets", None) and self.port == 0:
            try:
                self.port = self.server.sockets[0].getsockname()[1]
            except Exception:
                pass

        self._running = True
        logger.info(f"WebSocket server started on ws://{self.host}:{self.port}")
    
    async def stop(self):
        """Stop WebSocket server"""
        if self.server:
            self.server.close()
            await self.server.wait_closed()
        self._running = False
        logger.info("WebSocket server stopped")
    
    def register_handler(self, message_type: str, handler: Callable):
        """
        Register a handler for a specific message type.
        
        This allows external modules (e.g., SummonController) to register
        handlers for custom message types without modifying the server.
        
        Args:
            message_type: The WebSocket message 'type' field to handle
            handler: Async or sync callable that receives the payload dict
        """
        self.message_handlers[message_type] = handler
        logger.info(f"Registered handler for message type: {message_type}")
    
    async def _handle_connection(self, websocket):
        """Handle a new WebSocket connection"""
        self.active_connections.add(websocket)
        logger.info(f"New connection. Total: {len(self.active_connections)}")
        
        try:
            async for message in websocket:
                try:
                    await self._handle_message(websocket, message)
                except Exception as e:
                    logger.error(f"Error handling message: {e}")
                    await self._emit_error(websocket, str(e))
        
        except Exception as e:
            logger.error(f"Connection error: {e}")
        
        finally:
            self.active_connections.remove(websocket)
            logger.info(f"Connection closed. Total: {len(self.active_connections)}")
    
    async def _handle_message(self, websocket, message: str):
        """Handle incoming WebSocket message"""
        try:
            data = json.loads(message)
            message_type = data.get("type", "unknown")
            
            logger.debug(f"Received message: {message_type}")
            
            # Emit to registered handlers
            if message_type in self._handlers:
                for handler in self._handlers[message_type]:
                    try:
                        result = handler(data.get("payload", {}))
                        if asyncio.iscoroutine(result):
                            await result
                    except Exception as e:
                        logger.error(f"Handler error: {e}")
            
            # Call custom handler if registered
            if message_type in self.message_handlers:
                handler = self.message_handlers[message_type]
                result = handler(data.get("payload", {}))
                if asyncio.iscoroutine(result):
                    await result
        
        except json.JSONDecodeError:
            logger.error("Invalid JSON received")
            await self._emit_error(websocket, "Invalid JSON")
    
    async def broadcast_state(self, state_dict: dict):
        """Broadcast full robot state to all connected clients"""
        message = {
            "type": "state_update",
            "data": state_dict,
            "timestamp": datetime.now().isoformat(),
        }
        await self._broadcast(json.dumps(message))
    
    async def broadcast_emotion(self, emotion: str):
        """Broadcast emotion change"""
        message = {
            "type": "emotion",
            "emotion": emotion,
            "timestamp": datetime.now().isoformat(),
        }
        await self._broadcast(json.dumps(message))
    
    async def broadcast_audio(self, audio_state: dict):
        """Broadcast audio state change"""
        message = {
            "type": "audio",
            "data": audio_state,
            "timestamp": datetime.now().isoformat(),
        }
        await self._broadcast(json.dumps(message))
    
    async def broadcast_animation(self, animation: dict):
        """Broadcast animation update"""
        message = {
            "type": "animation",
            "data": animation,
            "timestamp": datetime.now().isoformat(),
        }
        await self._broadcast(json.dumps(message))
    
    async def broadcast_hardware(self, hardware_state: dict):
        """Broadcast hardware state update"""
        message = {
            "type": "hardware",
            "data": hardware_state,
            "timestamp": datetime.now().isoformat(),
        }
        await self._broadcast(json.dumps(message))
    
    async def send_error(self, error_message: str):
        """Send error message to all clients"""
        message = {
            "type": "error",
            "message": error_message,
            "timestamp": datetime.now().isoformat(),
        }
        await self._broadcast(json.dumps(message))
    
    async def _broadcast(self, message: str):
        """Send message to all connected clients"""
        if not self.active_connections:
            return
        
        # Create tasks for all connections
        tasks = []
        for connection in list(self.active_connections):
            try:
                tasks.append(connection.send(message))
            except Exception as e:
                logger.error(f"Send error: {e}")
        
        # Send to all in parallel
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
    
    async def _emit_error(self, websocket, error: str):
        """Send error to specific client"""
        try:
            message = json.dumps({
                "type": "error",
                "message": error,
                "timestamp": datetime.now().isoformat(),
            })
            await websocket.send(message)
        except Exception as e:
            logger.error(f"Error sending error message: {e}")
    
    def register_handler(self, message_type: str, handler: Callable):
        """Register handler for specific message type"""
        if message_type not in self._handlers:
            self._handlers[message_type] = []
        self._handlers[message_type].append(handler)
    
    def unregister_handler(self, message_type: str, handler: Callable):
        """Unregister handler for specific message type"""
        if message_type in self._handlers and handler in self._handlers[message_type]:
            self._handlers[message_type].remove(handler)
    
    @property
    def connection_count(self) -> int:
        """Get number of active connections"""
        return len(self.active_connections)
    
    @property
    def is_running(self) -> bool:
        """Check if server is running"""
        return self._running


# Global singleton
_ws_server: Optional[WebSocketServer] = None


def get_websocket_server() -> WebSocketServer:
    """Get or create global WebSocket server"""
    global _ws_server
    if _ws_server is None:
        _ws_server = WebSocketServer()
    return _ws_server
