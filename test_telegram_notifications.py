#!/usr/bin/env python3
"""
Test Telegram Notifications for Training Script
===============================================

Simple test to verify Telegram notifications work correctly.
"""

import os
import sys
from pathlib import Path

# Add paths for imports
sys.path.append("/notebooks/bot" if Path("/notebooks").exists() else ".")
sys.path.append("/notebooks/bot/src" if Path("/notebooks").exists() else "./src")

from src.notifier.telegram import TelegramNotifier


def test_telegram_connection():
    """Test basic Telegram connection"""
    print("🧪 Testing Telegram connection...")
    
    # Get credentials from environment
    bot_token = os.environ.get('TELEGRAM_BOT_TOKEN')
    chat_id = os.environ.get('TELEGRAM_CHAT_ID')
    
    if not bot_token or not chat_id:
        print("❌ Missing Telegram credentials")
        print("Please set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID environment variables")
        return False
        
    try:
        notifier = TelegramNotifier(
            bot_token=bot_token,
            chat_id=chat_id,
            enabled=True
        )
        
        # Test connection
        success = notifier.test_connection()
        
        if success:
            print("✅ Telegram connection test successful!")
            return True
        else:
            print("❌ Telegram connection test failed")
            return False
            
    except Exception as e:
        print(f"❌ Error testing Telegram connection: {e}")
        return False


def test_training_success_notification():
    """Test training success notification"""
    print("\n🧪 Testing training success notification...")
    
    # Get credentials from environment
    bot_token = os.environ.get('TELEGRAM_BOT_TOKEN')
    chat_id = os.environ.get('TELEGRAM_CHAT_ID')
    
    if not bot_token or not chat_id:
        print("❌ Missing Telegram credentials")
        return False
        
    try:
        notifier = TelegramNotifier(
            bot_token=bot_token,
            chat_id=chat_id,
            enabled=True
        )
        
        # Mock training result
        mock_result = {
            "success": True,
            "runtime_hours": 2.5,
            "training": {
                "model_results": {
                    "gru": {"BTCEUR": {"success": True}, "ETHEUR": {"success": True}},
                    "lightgbm": {"BTCEUR": {"success": True}, "ETHEUR": {"success": True}},
                    "ppo": {"BTCEUR": {"success": True}}
                }
            },
            "export": {"success": True},
            "s3_upload": {"success": True}
        }
        
        # Create message like the training script would
        training_result = mock_result.get("training", {})
        export_result = mock_result.get("export", {})
        s3_result = mock_result.get("s3_upload", {})
        runtime_hours = mock_result.get("runtime_hours", 0)
        
        models_trained = []
        for model_type, model_results in training_result.get("model_results", {}).items():
            successful_symbols = [s for s, r in model_results.items() if r.get("success")]
            if successful_symbols:
                models_trained.append(f"{model_type.upper()}: {len(successful_symbols)} symbols")
        
        message = f"""
🎉 <b>MODEL TRAINING COMPLETED</b> (TEST)

<b>Runtime:</b> {runtime_hours:.1f} hours
<b>Models Trained:</b>
{chr(10).join(f'• {m}' for m in models_trained) if models_trained else '• No models completed'}

<b>Export Status:</b> {'✅ Success' if export_result.get('success') else '❌ Failed'}
<b>S3 Upload:</b> {'✅ Success' if s3_result.get('success') else '❌ Failed'}

🤖 <b>Ready for deployment!</b>
Type <code>/import</code> to import models in the cryptobot.

<i>Training Server • Test Notification</i>
"""
        
        success = notifier.send_message_sync(message)
        
        if success:
            print("✅ Training success notification test successful!")
            return True
        else:
            print("❌ Training success notification test failed")
            return False
            
    except Exception as e:
        print(f"❌ Error testing training success notification: {e}")
        return False


def test_training_failure_notification():
    """Test training failure notification"""
    print("\n🧪 Testing training failure notification...")
    
    # Get credentials from environment
    bot_token = os.environ.get('TELEGRAM_BOT_TOKEN')
    chat_id = os.environ.get('TELEGRAM_CHAT_ID')
    
    if not bot_token or not chat_id:
        print("❌ Missing Telegram credentials")
        return False
        
    try:
        notifier = TelegramNotifier(
            bot_token=bot_token,
            chat_id=chat_id,
            enabled=True
        )
        
        # Mock failure scenario
        error = "Test error: Model training failed due to insufficient memory"
        pipeline_state = {
            "current_stage": "model_training",
            "errors": ["Memory allocation failed", "CUDA out of memory", "Training interrupted"]
        }
        
        current_stage = pipeline_state.get("current_stage", "unknown")
        runtime_hours = 1.2  # Mock runtime
        errors = pipeline_state.get("errors", [])
        
        message = f"""
🚨 <b>MODEL TRAINING FAILED</b> (TEST)

<b>Stage:</b> {current_stage}
<b>Runtime:</b> {runtime_hours:.1f} hours
<b>Error:</b> {error}

<b>Pipeline Errors:</b>
{chr(10).join(f'• {e}' for e in errors[-3:]) if errors else '• No specific errors logged'}

Please check the training logs for more details.

<i>Training Server • Test Notification</i>
"""
        
        success = notifier.send_message_sync(message)
        
        if success:
            print("✅ Training failure notification test successful!")
            return True
        else:
            print("❌ Training failure notification test failed")
            return False
            
    except Exception as e:
        print(f"❌ Error testing training failure notification: {e}")
        return False


def main():
    """Run all tests"""
    print("🚀 Starting Telegram notification tests...")
    print("=" * 50)
    
    tests_passed = 0
    total_tests = 3
    
    # Test 1: Basic connection
    if test_telegram_connection():
        tests_passed += 1
    
    # Test 2: Success notification
    if test_training_success_notification():
        tests_passed += 1
        
    # Test 3: Failure notification
    if test_training_failure_notification():
        tests_passed += 1
    
    print("\n" + "=" * 50)
    print(f"📊 Test Results: {tests_passed}/{total_tests} passed")
    
    if tests_passed == total_tests:
        print("🎉 All Telegram notification tests passed!")
        return True
    else:
        print("⚠️ Some tests failed. Check your Telegram configuration.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)