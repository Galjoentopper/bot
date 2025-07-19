import asyncio
import json
import logging
import socket
from typing import Set

import websockets

class PredictionWebSocketServer:
    """Simple WebSocket server to broadcast prediction updates."""

    def __init__(self, host: str = "0.0.0.0", port: int = 8765) -> None:
        self.host = host
        self.port = port
        self.clients: Set[websockets.WebSocketServerProtocol] = set()
        self.server: websockets.server.Serve = None
        self.logger = logging.getLogger(__name__)

    def _is_port_available(self, port: int) -> bool:
        """Check if a port is available for binding."""
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                sock.bind((self.host, port))
                return True
        except OSError:
            return False

    def _find_available_port(self, start_port: int, max_attempts: int = 100) -> int:
        """Find an available port starting from start_port."""
        for port_offset in range(max_attempts):
            test_port = start_port + port_offset
            if self._is_port_available(test_port):
                return test_port
        raise RuntimeError(f"Could not find an available port after {max_attempts} attempts starting from {start_port}")

    async def _handler(self, websocket: websockets.WebSocketServerProtocol) -> None:
        """Register client connection and keep it open."""
        self.clients.add(websocket)
        self.logger.info("WebSocket client connected")
        try:
            async for _ in websocket:
                pass  # We do not expect messages from clients
        except websockets.exceptions.ConnectionClosed:
            pass
        finally:
            self.clients.discard(websocket)
            self.logger.info("WebSocket client disconnected")

    async def start(self) -> None:
        """Start the WebSocket server."""
        # Check if the preferred port is available
        if not self._is_port_available(self.port):
            self.logger.warning(f"Port {self.port} is already in use, finding alternative port...")
            original_port = self.port
            self.port = self._find_available_port(self.port)
            self.logger.info(f"Using port {self.port} instead of {original_port}")
        
        self.server = await websockets.serve(self._handler, self.host, self.port)
        self.logger.info(f"WebSocket server started on {self.host}:{self.port}")

    async def broadcast(self, data: dict) -> None:
        """Broadcast a JSON-serializable object to all connected clients."""
        if not self.clients:
            return
        message = json.dumps(data, default=str)
        disconnected = []
        for ws in set(self.clients):
            try:
                await ws.send(message)
            except websockets.exceptions.ConnectionClosed:
                disconnected.append(ws)
        for ws in disconnected:
            self.clients.discard(ws)

    async def stop(self) -> None:
        """Stop the WebSocket server."""
        if self.server is not None:
            self.server.close()
            await self.server.wait_closed()
            self.logger.info("WebSocket server stopped")

