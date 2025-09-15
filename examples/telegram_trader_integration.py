#!/usr/bin/env python3
"""
Example: Integrating New Telegram Bot with Trading System
=========================================================

This example shows how to integrate the new Telegram bot with your trading system.
"""

import asyncio
import os
import signal

# Add project root to path
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.logging_manager import get_system_logger
from src.notifications.telegram_integration import integrate_telegram_with_trader


class MockTrader:
    """Mock trader for demonstration."""

    def __init__(self):
        self.symbols = ["BTCEUR", "ETHEUR", "DOTEUR", "ADAEUR", "LINKEUR"]
        self.balance = 10000.0
        self.initial_balance = 10000.0
        self.is_running = False
        self.logger = get_system_logger("mock_trader")

    def get_model_counts(self):
        """Return model counts."""
        return {"gru": 5, "lightgbm": 5, "ppo": 5, "total": 15}

    async def start_trading(self):
        """Start the trading loop."""
        self.is_running = True
        self.logger.info("🚀 Starting mock trading system...")

        # Simulate some trading activity
        iteration = 0
        while self.is_running:
            try:
                iteration += 1
                await asyncio.sleep(30)  # Trade every 30 seconds for demo

                # Simulate a trade every few iterations
                if iteration % 3 == 0:
                    await self._simulate_trade(iteration)

                # Simulate an error occasionally
                if iteration % 10 == 0:
                    await self._simulate_error()

            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in trading loop: {e}")
                # The telegram integration should catch and notify this error
                if hasattr(self, "telegram_integration"):
                    self.telegram_integration.notify_error(e, "Trading loop")
                await asyncio.sleep(5)

    async def _simulate_trade(self, iteration):
        """Simulate a trade execution."""
        import random

        symbol = random.choice(self.symbols)
        action = random.choice(["BUY", "SELL"])
        quantity = random.uniform(0.001, 0.01)
        price = random.uniform(20000, 100000)
        realized_pnl = random.uniform(-50, 100)
        confidence = random.uniform(0.6, 0.95)

        trade_data = {
            "symbol": symbol,
            "action": action,
            "quantity": quantity,
            "price": price,
            "realized_pnl": realized_pnl,
            "confidence": confidence,
            "reason": f"Mock trade #{iteration}",
            "timestamp": asyncio.get_event_loop().time(),
        }

        self.logger.info(f"🔄 Simulated trade: {action} {quantity:.4f} {symbol} @ €{price:.2f}")

        # Update balance
        self.balance += realized_pnl

        # Notify Telegram (if integrated)
        if hasattr(self, "telegram_integration"):
            self.telegram_integration.notify_trade_execution(trade_data)

    async def _simulate_error(self):
        """Simulate an error for testing."""
        error = ValueError("Mock error: Model prediction failed")
        self.logger.error("🚨 Simulating error for testing")

        # Notify Telegram (if integrated)
        if hasattr(self, "telegram_integration"):
            self.telegram_integration.notify_error(error, "Mock error simulation")

    def stop_trading(self):
        """Stop the trading system."""
        self.is_running = False
        self.logger.info("🛑 Stopping mock trading system...")


async def main():
    """Main function demonstrating Telegram integration."""
    print("🚀 Telegram Bot Integration Example")
    print("=" * 50)

    # Check environment variables using secure env manager
    from src.config.secure_env_manager import get_env_manager

    env_manager = get_env_manager()
    bot_token = env_manager.get("TELEGRAM_BOT_TOKEN")
    chat_id = env_manager.get("TELEGRAM_CHAT_ID")

    if not bot_token or not chat_id:
        print("❌ Missing environment variables:")
        print("export TELEGRAM_BOT_TOKEN=your_bot_token")
        print("export TELEGRAM_CHAT_ID=your_chat_id")
        return False

    # Create mock trader
    trader = MockTrader()

    # Create Telegram integration
    telegram_integration = integrate_telegram_with_trader(trader)
    trader.telegram_integration = telegram_integration  # For error notifications

    # Initialize Telegram bot
    print("🤖 Initializing Telegram bot...")
    if not await telegram_integration.initialize_telegram(bot_token, chat_id):
        print("❌ Failed to initialize Telegram bot")
        return False

    print("✅ Telegram bot initialized successfully")

    # Set up graceful shutdown
    shutdown_event = asyncio.Event()

    def signal_handler():
        print("\n🛑 Shutdown signal received...")
        trader.stop_trading()
        shutdown_event.set()

    # Register signal handlers
    loop = asyncio.get_running_loop()
    loop.add_signal_handler(signal.SIGTERM, signal_handler)
    loop.add_signal_handler(signal.SIGINT, signal_handler)

    try:
        # Start Telegram bot and trading system concurrently
        tasks = [
            asyncio.create_task(telegram_integration.start_telegram_bot()),
            asyncio.create_task(trader.start_trading()),
            asyncio.create_task(shutdown_event.wait()),
        ]

        print("🚀 Starting Telegram bot and trading system...")
        print("💡 You should receive a startup message in Telegram")
        print("📱 Try sending /help, /status, /balance commands to the bot")
        print("⏰ Mock trades will execute every 30 seconds")
        print("🚨 Mock errors will occur every 5 minutes")
        print("📊 Daily performance report scheduled for 14:00 UTC")
        print("\nPress Ctrl+C to stop...")

        # Wait for any task to complete (usually shutdown)
        done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)

        # Cancel remaining tasks
        for task in pending:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

    finally:
        # Clean shutdown
        print("🧹 Cleaning up...")
        await telegram_integration.stop_telegram_bot()
        print("✅ Shutdown complete")

    return True


if __name__ == "__main__":
    try:
        success = asyncio.run(main())
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n🛑 Interrupted by user")
        sys.exit(0)
