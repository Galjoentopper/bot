"""
Launcher script for the paper trader
"""
import sys
import logging
from paper_trader import PaperTrader
from config import (
    API_KEY, API_SECRET, SYMBOLS, INITIAL_BALANCE, 
    TELEGRAM_TOKEN, TELEGRAM_CHAT_ID, LOG_LEVEL
)

# Configure logging
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("paper_trader.log"),
        logging.StreamHandler()
    ]
)

def main():
    """Main function to start the paper trader"""
    logging.info("Starting paper trader...")
    
    # Validate API credentials
    if API_KEY == "your_api_key_here" or API_SECRET == "your_api_secret_here":
        logging.error("Please set your API credentials in the .env file")
        sys.exit(1)
    
    # Validate Telegram settings
    telegram_configured = True
    if TELEGRAM_TOKEN == "your_telegram_bot_token_here" or TELEGRAM_CHAT_ID == "your_telegram_chat_id_here":
        logging.warning("Telegram notifications disabled - missing token or chat ID")
        telegram_configured = False
        
    # Create and run the paper trader
    trader = PaperTrader(
        api_key=API_KEY,
        api_secret=API_SECRET,
        telegram_token=TELEGRAM_TOKEN if telegram_configured else None,
        telegram_chat_id=TELEGRAM_CHAT_ID if telegram_configured else None,
        symbols=SYMBOLS,
        initial_balance=INITIAL_BALANCE
    )
    
    # Run the trader
    trader.run()

if __name__ == "__main__":
    main()