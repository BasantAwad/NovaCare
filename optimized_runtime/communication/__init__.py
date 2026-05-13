"""
Communication Layer Module

WebSocket server for real-time bidirectional communication
with robot UIs and monitoring clients.
"""

from .websocket_server import WebSocketServer, get_websocket_server

__all__ = ["WebSocketServer", "get_websocket_server"]
