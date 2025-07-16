import asyncio
import json
import logging
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

