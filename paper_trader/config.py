"""
Configuration file for the paper trader
"""
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Bitvavo API credentials
API_KEY = os.getenv("BITVAVO_API_KEY", "your_api_key_here")
API_SECRET = os.getenv("BITVAVO_API_SECRET", "your_api_secret_here")

# Telegram settings
TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "your_telegram_bot_token_here")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "your_telegram_chat_id_here")

# Trading parameters
SYMBOLS = ['BTCEUR', 'ETHEUR', 'SOLEUR', 'XRPEUR', 'ADAEUR']
INITIAL_BALANCE = float(os.getenv("INITIAL_CAPITAL", "10000"))  # EUR

# Trade settings
POSITION_SIZE_PCT = float(os.getenv("BASE_POSITION_SIZE", "0.1"))  # 10% of balance per trade
TAKE_PROFIT_PCT = float(os.getenv("TAKE_PROFIT_PCT", "0.005"))  # 0.5%
STOP_LOSS_PCT = float(os.getenv("STOP_LOSS_PCT", "0.005"))    # 0.5%
FEE_PCT = 0.003          # 0.3%

# Prediction settings
PREDICTION_INTERVAL = 60  # seconds

# Connection settings
API_FALLBACK_TIMEOUT = 30  # seconds before falling back to API

# Logging settings
LOG_LEVEL = "INFO"

# Model settings
MODEL_PATH = "models"