#!/usr/bin/env python3
"""
Production Webhook Server Startup Script
=========================================

Starts the webhook server for receiving model notifications from Paperspace
and automatically importing them into the production system.

Usage:
    python start_production_webhook.py [--port PORT] [--debug]

Environment Variables:
    PRODUCTION_API_KEY: API key for webhook authentication
    WEBHOOK_PORT: Port to run webhook server (default: 5000)
    TELEGRAM_BOT_TOKEN: For notifications
    TELEGRAM_CHAT_ID: For notifications
"""

import argparse
import logging
import os
import sys
from pathlib import Path

# Add project root to Python path
sys.path.append(str(Path(__file__).parent))

from paperspace_mlops.production_import_handler import create_webhook_app


def setup_logging(debug: bool = False):
    """Setup logging configuration"""
    level = logging.DEBUG if debug else logging.INFO

    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler("logs/webhook_server.log"), logging.StreamHandler()],
    )


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Production Webhook Server")
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.environ.get("WEBHOOK_PORT", 5000)),
        help="Port to run webhook server (default: 5000)",
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to (default: 0.0.0.0)")

    args = parser.parse_args()

    # Setup logging
    Path("logs").mkdir(exist_ok=True)
    setup_logging(args.debug)

    logger = logging.getLogger(__name__)

    # Check environment
    api_key = os.environ.get("PRODUCTION_API_KEY")
    if not api_key:
        logger.warning("⚠️ PRODUCTION_API_KEY not set - webhook will accept all requests")

    # Create and run app
    logger.info(f"🚀 Starting production webhook server on {args.host}:{args.port}")
    logger.info(f"📥 Webhook endpoint: http://{args.host}:{args.port}/webhook/models")
    logger.info(f"❤️ Health check: http://{args.host}:{args.port}/health")

    if args.debug:
        logger.info("🐛 Debug mode enabled")

    app = create_webhook_app()
    app.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()
