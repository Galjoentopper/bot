"""
Helper script to set up and test Telegram notifications
"""
import sys
import logging
import asyncio
from telegram import Bot
from telegram.error import TelegramError

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def setup_telegram_bot(token):
    """
    Set up the Telegram bot and return the bot instance
    
    :param token: Telegram bot token
    :return: Telegram bot instance
    """
    try:
        bot = Bot(token=token)
        async with bot:
            bot_info = await bot.get_me()
            logger.info(f"Bot connected! Bot username: @{bot_info.username}")
            return bot
    except TelegramError as e:
        logger.error(f"Failed to connect to Telegram: {e}")
        return None

async def send_test_message(bot, chat_id):
    """
    Send a test message to verify the bot can message the user
    
    :param bot: Telegram bot instance
    :param chat_id: Chat ID to send the message to
    :return: True if successful, False otherwise
    """
    try:
        message = (
            "🤖 *Paper Trader Bot Test Message*\n\n"
            "If you can see this message, your Telegram notifications are set up correctly!\n\n"
            "You will receive notifications for:\n"
            "✅ Bot startup\n"
            "✅ Trading activity\n"
            "✅ Hourly summaries\n\n"
            "Thank you for setting up the Paper Trader Bot!"
        )
        async with bot:
            await bot.send_message(chat_id=chat_id, text=message, parse_mode='Markdown')
        return True
    except TelegramError as e:
        logger.error(f"Failed to send test message: {e}")
        return False

def main():
    """Main function to test Telegram setup"""
    if len(sys.argv) < 3:
        print("Usage: python setup_telegram.py <bot_token> <chat_id>")
        sys.exit(1)
        
    token = sys.argv[1]
    chat_id = sys.argv[2]
    
    async def test_setup():
        logger.info("Testing Telegram notifications...")
        bot = await setup_telegram_bot(token)
        
        if bot:
            success = await send_test_message(bot, chat_id)
            if success:
                logger.info("Telegram notifications set up successfully!")
                
                # Output for config.py
                print("\n=== Add this to your .env file ===")
                print(f"TELEGRAM_BOT_TOKEN={token}")
                print(f"TELEGRAM_CHAT_ID={chat_id}")
                print("=====================================\n")
            else:
                logger.error("Failed to send test message. Check your chat_id.")
        else:
            logger.error("Failed to set up Telegram bot. Check your bot token.")
    
    # Run the async function
    asyncio.run(test_setup())

if __name__ == "__main__":
    main()