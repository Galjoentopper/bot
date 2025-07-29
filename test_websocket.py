#!/usr/bin/env python3
"""
Websocket Test Script for Bitvavo API
====================================

This script tests the websocket connection to Bitvavo to identify any issues
with the current implementation in the paper trader.
"""

import websocket
import json
import threading
import time
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class WebsocketTester:
    def __init__(self):
        self.connection_status = "disconnected"
        self.message_count = 0
        self.last_message_time = None
        self.websocket = None
        self.running = True
        
    def test_websocket_connection(self):
        """Test the websocket connection to Bitvavo"""
        logger.info("Starting websocket connection test...")
        
        def on_message(ws, message):
            try:
                self.message_count += 1
                self.last_message_time = datetime.now()
                
                data = json.loads(message)
                logger.info(f"Message #{self.message_count}: {type(data)} - {str(data)[:200]}...")
                
                # Log specific event types
                if isinstance(data, dict):
                    event_type = data.get('event', 'unknown')
                    market = data.get('market', 'N/A')
                    logger.info(f"Event: {event_type}, Market: {market}")
                elif isinstance(data, list):
                    logger.info(f"Received list with {len(data)} items")
                    
            except Exception as e:
                logger.error(f"Error processing message: {e}")
                
        def on_error(ws, error):
            logger.error(f"Websocket error: {error}")
            self.connection_status = "error"
            
        def on_close(ws, close_status_code, close_msg):
            logger.info(f"Websocket closed: {close_status_code} - {close_msg}")
            self.connection_status = "disconnected"
            
        def on_open(ws):
            logger.info("✅ Websocket connection established successfully!")
            self.connection_status = "connected"
            
            # Test subscription to ticker data
            logger.info("Subscribing to ticker data...")
            subscribe_message = {
                "action": "subscribe",
                "channels": ["ticker"]
            }
            
            # Test with a few popular markets
            test_markets = ["BTC-EUR", "ETH-EUR", "ADA-EUR"]
            subscribe_message["markets"] = test_markets
            
            try:
                ws.send(json.dumps(subscribe_message))
                logger.info(f"Sent subscription for markets: {test_markets}")
            except Exception as e:
                logger.error(f"Error sending subscription: {e}")
                
            # Also test candle subscription
            time.sleep(1)
            logger.info("Subscribing to candle data...")
            for market in test_markets[:2]:  # Test with first 2 markets
                candle_message = {
                    "action": "subscribe",
                    "channels": ["candles"],
                    "markets": [market],
                    "interval": ["15m"]
                }
                try:
                    ws.send(json.dumps(candle_message))
                    logger.info(f"Sent candle subscription for {market}")
                except Exception as e:
                    logger.error(f"Error sending candle subscription for {market}: {e}")
                    
        # Create websocket connection
        self.websocket = websocket.WebSocketApp(
            "wss://ws.bitvavo.com/v2/",
            on_message=on_message,
            on_error=on_error,
            on_close=on_close
        )
        self.websocket.on_open = on_open
        
        # Start websocket in a separate thread
        wst = threading.Thread(target=self.websocket.run_forever)
        wst.daemon = True
        wst.start()
        
        # Monitor connection for 30 seconds
        start_time = datetime.now()
        test_duration = 30  # seconds
        
        logger.info(f"Monitoring connection for {test_duration} seconds...")
        
        while (datetime.now() - start_time).total_seconds() < test_duration:
            time.sleep(5)
            
            # Print status update
            elapsed = (datetime.now() - start_time).total_seconds()
            logger.info(f"Status after {elapsed:.0f}s: {self.connection_status}, Messages: {self.message_count}")
            
            if self.last_message_time:
                time_since_last = (datetime.now() - self.last_message_time).total_seconds()
                logger.info(f"Last message received {time_since_last:.1f} seconds ago")
                
        # Close connection
        logger.info("Test completed. Closing connection...")
        self.running = False
        if self.websocket:
            self.websocket.close()
            
        # Print final results
        logger.info("\n" + "="*50)
        logger.info("WEBSOCKET TEST RESULTS")
        logger.info("="*50)
        logger.info(f"Final Status: {self.connection_status}")
        logger.info(f"Total Messages Received: {self.message_count}")
        logger.info(f"Test Duration: {test_duration} seconds")
        
        if self.message_count > 0:
            logger.info("✅ Websocket connection is working!")
            return True
        else:
            logger.error("❌ No messages received - websocket may have issues")
            return False

def main():
    """Run the websocket test"""
    print("🔌 Bitvavo Websocket Connection Test")
    print("===================================\n")
    
    tester = WebsocketTester()
    success = tester.test_websocket_connection()
    
    if success:
        print("\n🎉 Websocket test completed successfully!")
        print("The websocket connection to Bitvavo is working properly.")
    else:
        print("\n⚠️  Websocket test failed!")
        print("There may be issues with the websocket connection.")
        print("Possible causes:")
        print("- Network connectivity issues")
        print("- Bitvavo API changes")
        print("- Firewall blocking websocket connections")
        print("- Rate limiting or IP restrictions")
    
    return success

if __name__ == "__main__":
    main()