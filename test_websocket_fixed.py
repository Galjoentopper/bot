import websocket
import json
import time
import threading

class BitvavoWebSocketTest:
    def __init__(self):
        self.ws = None
        self.message_count = 0
        self.connected = False
        
    def on_message(self, ws, message):
        self.message_count += 1
        try:
            data = json.loads(message)
            print(f"Message {self.message_count}: {json.dumps(data, indent=2)}")
        except json.JSONDecodeError:
            print(f"Message {self.message_count}: {message}")
    
    def on_error(self, ws, error):
        print(f"WebSocket error: {error}")
    
    def on_close(self, ws, close_status_code, close_msg):
        print(f"WebSocket closed. Status: {close_status_code}, Message: {close_msg}")
        self.connected = False
    
    def on_open(self, ws):
        print("WebSocket connection opened")
        self.connected = True
        
        # Subscribe to ticker24h and candle data with correct format
        subscribe_message = {
            "action": "subscribe",
            "channels": [
                {
                    "name": "ticker24h",
                    "markets": ["BTC-EUR", "ETH-EUR", "ADA-EUR"]
                },
                {
                    "name": "candles",
                    "interval": ["1h"],
                    "markets": ["BTC-EUR", "ETH-EUR", "ADA-EUR"]
                }
            ]
        }
        
        print(f"Sending subscription: {json.dumps(subscribe_message, indent=2)}")
        ws.send(json.dumps(subscribe_message))
    
    def test_connection(self):
        print("Testing Bitvavo WebSocket connection...")
        
        # Create WebSocket connection
        self.ws = websocket.WebSocketApp(
            "wss://ws.bitvavo.com/v2/",
            on_open=self.on_open,
            on_message=self.on_message,
            on_error=self.on_error,
            on_close=self.on_close
        )
        
        # Start WebSocket in a separate thread
        wst = threading.Thread(target=self.ws.run_forever)
        wst.daemon = True
        wst.start()
        
        # Wait for connection and messages
        print("Waiting for connection and messages...")
        time.sleep(30)
        
        # Close connection
        if self.ws:
            self.ws.close()
        
        print(f"\nTest completed. Received {self.message_count} messages.")
        print(f"Connection status: {'Connected' if self.connected else 'Disconnected'}")

if __name__ == "__main__":
    test = BitvavoWebSocketTest()
    test.test_connection()