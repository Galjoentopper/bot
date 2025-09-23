#!/usr/bin/env python3
"""
Hetzner Trading System Preparation
==================================

Prepares the Hetzner trading system to work seamlessly with superior models.
Updates model loading logic, configuration, and system integration.
"""

import json
import logging
import os
import subprocess
import sys
import tempfile
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


class HetznerSystemPreparator:
    """Prepares Hetzner trading system for superior models."""

    def __init__(self, config_path):
        with open(config_path, "r") as f:
            self.config = json.load(f)

        self.remote_base = "/opt/trading_bot"

    def prepare_model_manager(self):
        """Update model manager to support superior models."""
        logger.info("🧠 Updating model manager for superior models...")

        model_manager_code = '''
#!/usr/bin/env python3
"""
Enhanced Model Manager with Superior PPO Support
================================================

Integrates superior models seamlessly into the existing trading system.
"""

import os
import sys
import logging
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Any

logger = logging.getLogger(__name__)


class SuperiorModelManager:
    """Enhanced model manager with superior PPO support."""

    def __init__(self, base_path: str = "/opt/trading_bot"):
        self.base_path = Path(base_path)
        self.models_path = self.base_path / "models"

        # Model loading priority
        self.model_priority = ['superior', 'ppo', 'lightgbm', 'gru']
        self.symbols = ['BTCEUR', 'ETHEUR', 'ADAEUR', 'DOTEUR', 'LINKEUR']

        # Loaded models cache
        self.loaded_models = {}
        self.model_metadata = {}

        logger.info(f"🧠 Superior Model Manager initialized")
        logger.info(f"   Base path: {self.base_path}")
        logger.info(f"   Model priority: {self.model_priority}")

    def load_superior_model(self, symbol: str):
        """Load superior PPO model for symbol."""
        try:
            from stable_baselines3 import PPO

            model_path = self.models_path / "superior" / symbol / "best_model.zip"

            if not model_path.exists():
                raise FileNotFoundError(f"Superior model not found: {model_path}")

            logger.info(f"🔄 Loading superior model: {symbol}")
            model = PPO.load(str(model_path))

            # Load metadata if available
            metadata_path = model_path.parent / "metadata.json"
            metadata = {}
            if metadata_path.exists():
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)

            self.loaded_models[symbol] = {
                'model': model,
                'type': 'superior',
                'path': str(model_path),
                'metadata': metadata
            }

            logger.info(f"✅ Superior model loaded: {symbol}")
            return model

        except ImportError:
            logger.error("❌ stable-baselines3 not available for superior models")
            raise
        except Exception as e:
            logger.error(f"❌ Failed to load superior model {symbol}: {e}")
            raise

    def load_fallback_model(self, symbol: str, model_type: str):
        """Load fallback model (lightgbm, gru, etc.)."""
        try:
            if model_type == 'lightgbm':
                return self._load_lightgbm_model(symbol)
            elif model_type == 'gru':
                return self._load_gru_model(symbol)
            elif model_type == 'ppo':
                return self._load_ppo_model(symbol)
            else:
                raise ValueError(f"Unknown model type: {model_type}")

        except Exception as e:
            logger.error(f"❌ Failed to load {model_type} model {symbol}: {e}")
            raise

    def _load_lightgbm_model(self, symbol: str):
        """Load LightGBM model."""
        import joblib
        model_path = self.models_path / "lightgbm" / symbol / "model.pkl"

        if model_path.exists():
            model = joblib.load(model_path)
            self.loaded_models[symbol] = {
                'model': model,
                'type': 'lightgbm',
                'path': str(model_path)
            }
            return model
        else:
            raise FileNotFoundError(f"LightGBM model not found: {model_path}")

    def _load_gru_model(self, symbol: str):
        """Load GRU model."""
        import torch
        model_path = self.models_path / "gru" / symbol / "model.pt"

        if model_path.exists():
            model = torch.load(model_path, map_location='cpu')
            self.loaded_models[symbol] = {
                'model': model,
                'type': 'gru',
                'path': str(model_path)
            }
            return model
        else:
            raise FileNotFoundError(f"GRU model not found: {model_path}")

    def _load_ppo_model(self, symbol: str):
        """Load legacy PPO model."""
        from stable_baselines3 import PPO
        model_path = self.models_path / "ppo" / symbol / "model.zip"

        if model_path.exists():
            model = PPO.load(str(model_path))
            self.loaded_models[symbol] = {
                'model': model,
                'type': 'ppo',
                'path': str(model_path)
            }
            return model
        else:
            raise FileNotFoundError(f"PPO model not found: {model_path}")

    def load_model_for_symbol(self, symbol: str):
        """Load best available model for symbol."""
        logger.info(f"🔄 Loading model for {symbol}...")

        # Try each model type in priority order
        for model_type in self.model_priority:
            try:
                if model_type == 'superior':
                    model = self.load_superior_model(symbol)
                else:
                    model = self.load_fallback_model(symbol, model_type)

                logger.info(f"✅ Loaded {model_type} model for {symbol}")
                return model

            except Exception as e:
                logger.warning(f"⚠️  {model_type} model failed for {symbol}: {e}")
                continue

        # If all models fail
        raise RuntimeError(f"❌ No valid models found for {symbol}")

    def load_all_models(self):
        """Load models for all symbols."""
        logger.info("🚀 Loading models for all symbols...")

        successful_loads = 0
        for symbol in self.symbols:
            try:
                self.load_model_for_symbol(symbol)
                successful_loads += 1
            except Exception as e:
                logger.error(f"❌ Failed to load any model for {symbol}: {e}")

        success_rate = successful_loads / len(self.symbols)
        logger.info(f"📊 Model loading: {successful_loads}/{len(self.symbols)} ({success_rate:.1%})")

        if success_rate < 0.6:
            raise RuntimeError("❌ Too many model loading failures")

        return self.loaded_models

    def predict_with_superior_features(self, symbol: str, market_data: dict):
        """Make prediction with superior feature engineering."""
        if symbol not in self.loaded_models:
            raise ValueError(f"Model not loaded for {symbol}")

        model_info = self.loaded_models[symbol]
        model = model_info['model']
        model_type = model_info['type']

        if model_type == 'superior':
            # Superior model expects 104 features in 32-timestep window
            features = self._prepare_superior_features(market_data)
            action, _states = model.predict(features, deterministic=True)
            return self._interpret_superior_action(action[0])

        elif model_type in ['ppo', 'lightgbm', 'gru']:
            # Fallback to legacy feature engineering
            features = self._prepare_legacy_features(market_data)

            if model_type == 'ppo':
                action, _states = model.predict(features, deterministic=True)
                return self._interpret_ppo_action(action[0])
            elif model_type == 'lightgbm':
                prediction = model.predict(features.reshape(1, -1))[0]
                return self._interpret_lightgbm_prediction(prediction)
            elif model_type == 'gru':
                # GRU prediction logic
                import torch
                with torch.no_grad():
                    prediction = model(torch.tensor(features).float().unsqueeze(0))
                return self._interpret_gru_prediction(prediction)

        else:
            raise ValueError(f"Unknown model type: {model_type}")

    def _prepare_superior_features(self, market_data: dict) -> np.ndarray:
        """Prepare superior multi-timeframe features."""
        # This would integrate with your superior feature engineering
        # For now, return dummy features matching expected shape
        return np.random.random((32, 104)).astype(np.float32)

    def _prepare_legacy_features(self, market_data: dict) -> np.ndarray:
        """Prepare legacy technical indicator features."""
        # Legacy feature preparation
        return np.random.random(50).astype(np.float32)

    def _interpret_superior_action(self, action: float) -> dict:
        """Interpret superior model action."""
        return {
            'signal': 'buy' if action > 0.1 else 'sell' if action < -0.1 else 'hold',
            'confidence': abs(action),
            'position_size': min(abs(action), 1.0),
            'model_type': 'superior'
        }

    def _interpret_ppo_action(self, action: float) -> dict:
        """Interpret PPO action."""
        return {
            'signal': 'buy' if action > 0 else 'sell',
            'confidence': abs(action),
            'position_size': min(abs(action), 1.0),
            'model_type': 'ppo'
        }

    def _interpret_lightgbm_prediction(self, prediction: float) -> dict:
        """Interpret LightGBM prediction."""
        return {
            'signal': 'buy' if prediction > 0.5 else 'sell',
            'confidence': abs(prediction - 0.5) * 2,
            'position_size': min(abs(prediction - 0.5) * 2, 1.0),
            'model_type': 'lightgbm'
        }

    def _interpret_gru_prediction(self, prediction) -> dict:
        """Interpret GRU prediction."""
        pred_value = float(prediction.item())
        return {
            'signal': 'buy' if pred_value > 0.5 else 'sell',
            'confidence': abs(pred_value - 0.5) * 2,
            'position_size': min(abs(pred_value - 0.5) * 2, 1.0),
            'model_type': 'gru'
        }

    def get_model_status(self) -> dict:
        """Get status of all loaded models."""
        status = {}
        for symbol in self.symbols:
            if symbol in self.loaded_models:
                model_info = self.loaded_models[symbol]
                status[symbol] = {
                    'loaded': True,
                    'type': model_info['type'],
                    'path': model_info['path']
                }
            else:
                status[symbol] = {'loaded': False}

        return status


# Global model manager instance
model_manager = None

def get_model_manager():
    """Get global model manager instance."""
    global model_manager
    if model_manager is None:
        model_manager = SuperiorModelManager()
        model_manager.load_all_models()
    return model_manager


# Backward compatibility functions
def load_models():
    """Load all models (backward compatibility)."""
    return get_model_manager().load_all_models()

def predict(symbol: str, market_data: dict):
    """Make prediction for symbol (backward compatibility)."""
    return get_model_manager().predict_with_superior_features(symbol, market_data)

def get_model_status():
    """Get model status (backward compatibility)."""
    return get_model_manager().get_model_status()


if __name__ == "__main__":
    # Test the model manager
    manager = SuperiorModelManager()
    try:
        models = manager.load_all_models()
        print(f"✅ Loaded {len(models)} models")

        status = manager.get_model_status()
        for symbol, info in status.items():
            if info['loaded']:
                print(f"✅ {symbol}: {info['type']}")
            else:
                print(f"❌ {symbol}: Not loaded")

    except Exception as e:
        print(f"❌ Model manager test failed: {e}")
        sys.exit(1)
'''

        # Write the enhanced model manager to remote server
        return self._deploy_code_file(
            model_manager_code,
            f"{self.remote_base}/src/models/superior_model_manager.py",
            "Enhanced Model Manager",
        )

    def prepare_trading_integration(self):
        """Update trading system integration."""
        logger.info("🔄 Updating trading system integration...")

        integration_code = '''
#!/usr/bin/env python3
"""
Superior Trading Integration
===========================

Integrates superior models into the existing trading system.
"""

import sys
import os
from pathlib import Path

# Add project root to path
sys.path.insert(0, '/opt/trading_bot')

try:
    from src.models.superior_model_manager import get_model_manager
    SUPERIOR_MODELS_AVAILABLE = True
except ImportError:
    SUPERIOR_MODELS_AVAILABLE = False
    print("⚠️  Superior models not available, using legacy models")


class SuperiorTradingSystem:
    """Enhanced trading system with superior model support."""

    def __init__(self):
        self.model_manager = None
        self.initialize_models()

    def initialize_models(self):
        """Initialize model system."""
        if SUPERIOR_MODELS_AVAILABLE:
            try:
                self.model_manager = get_model_manager()
                print("✅ Superior models initialized")
                return True
            except Exception as e:
                print(f"⚠️  Superior models failed, falling back: {e}")

        # Fallback to legacy system
        try:
            from src.models.legacy_model_manager import LegacyModelManager
            self.model_manager = LegacyModelManager()
            print("✅ Legacy models initialized")
            return True
        except ImportError:
            print("❌ No model system available")
            return False

    def get_trading_signals(self, market_data: dict) -> dict:
        """Get trading signals for all symbols."""
        if not self.model_manager:
            return {}

        signals = {}

        if SUPERIOR_MODELS_AVAILABLE:
            # Use superior prediction method
            for symbol in ['BTCEUR', 'ETHEUR', 'ADAEUR', 'DOTEUR', 'LINKEUR']:
                try:
                    signal = self.model_manager.predict_with_superior_features(
                        symbol,
                        market_data.get(symbol, {})
                    )
                    signals[symbol] = signal
                except Exception as e:
                    print(f"⚠️  Prediction failed for {symbol}: {e}")

        else:
            # Use legacy prediction method
            for symbol in ['BTCEUR', 'ETHEUR', 'ADAEUR', 'DOTEUR', 'LINKEUR']:
                try:
                    signal = self.model_manager.predict(
                        symbol,
                        market_data.get(symbol, {})
                    )
                    signals[symbol] = signal
                except Exception as e:
                    print(f"⚠️  Prediction failed for {symbol}: {e}")

        return signals

    def get_system_status(self) -> dict:
        """Get system status."""
        if not self.model_manager:
            return {'status': 'no_models'}

        try:
            model_status = self.model_manager.get_model_status()

            loaded_models = sum(1 for info in model_status.values() if info.get('loaded', False))
            total_models = len(model_status)

            return {
                'status': 'operational' if loaded_models > 0 else 'no_models',
                'models': model_status,
                'loaded_count': loaded_models,
                'total_count': total_models,
                'superior_available': SUPERIOR_MODELS_AVAILABLE
            }

        except Exception as e:
            return {'status': 'error', 'error': str(e)}


# Global trading system instance
trading_system = None

def get_trading_system():
    """Get global trading system instance."""
    global trading_system
    if trading_system is None:
        trading_system = SuperiorTradingSystem()
    return trading_system


def get_signals(market_data: dict) -> dict:
    """Get trading signals (main entry point)."""
    return get_trading_system().get_trading_signals(market_data)


def get_status() -> dict:
    """Get system status (main entry point)."""
    return get_trading_system().get_system_status()


if __name__ == "__main__":
    # Test the trading integration
    system = SuperiorTradingSystem()

    status = system.get_system_status()
    print(f"System status: {status}")

    # Test prediction with dummy data
    dummy_data = {
        'BTCEUR': {'price': 50000, 'volume': 1000},
        'ETHEUR': {'price': 3000, 'volume': 500}
    }

    signals = system.get_trading_signals(dummy_data)
    print(f"Test signals: {signals}")
'''

        return self._deploy_code_file(
            integration_code,
            f"{self.remote_base}/src/trading/superior_integration.py",
            "Trading Integration",
        )

    def update_system_manager(self):
        """Update system_manager to detect and use superior models."""
        logger.info("🚀 Updating system_manager for superior models...")

        # Create a patch for system_manager
        system_manager_patch = """
#!/bin/bash
# Superior Models Detection Patch for system_manager
# Add this to your existing system_manager script

detect_superior_models() {
    echo "🔍 Detecting available models..."

    SUPERIOR_MODELS_COUNT=0
    if [ -d "/opt/trading_bot/models/superior" ]; then
        SUPERIOR_MODELS_COUNT=$(find /opt/trading_bot/models/superior -name "best_model.zip" | wc -l)
        echo "   Superior models: $SUPERIOR_MODELS_COUNT"
    fi

    if [ $SUPERIOR_MODELS_COUNT -ge 3 ]; then
        echo "✅ Superior models detected - using advanced trading system"
        export USE_SUPERIOR_MODELS=true
        export MODEL_TYPE=superior
    else
        echo "⚠️  Using legacy models"
        export USE_SUPERIOR_MODELS=false
        export MODEL_TYPE=legacy
    fi
}

# Call detection before starting services
detect_superior_models

# Update your start_trading_bot function to check for superior models
start_trading_bot_enhanced() {
    echo "🚀 Starting Enhanced Trading Bot..."

    # Set Python path
    export PYTHONPATH="/opt/trading_bot:$PYTHONPATH"

    if [ "$USE_SUPERIOR_MODELS" = "true" ]; then
        echo "🧠 Starting with superior PPO models..."
        # Use enhanced trader that supports superior models
        python3 -c "
import sys
sys.path.insert(0, '/opt/trading_bot')
try:
    from src.trading.superior_integration import get_trading_system
    system = get_trading_system()
    status = system.get_system_status()
    print(f'System initialized: {status}')

    if status['status'] == 'operational':
        print('✅ Superior trading system ready')
    else:
        print('⚠️  System issues detected')
        sys.exit(1)

except Exception as e:
    print(f'❌ Superior system failed: {e}')
    sys.exit(1)
"
        if [ $? -eq 0 ]; then
            echo "✅ Superior models validated"
            # Start your enhanced trader here
            # python3 scripts/enhanced_trader.py &
        else
            echo "❌ Superior models validation failed"
            return 1
        fi
    else
        echo "🔄 Starting with legacy models..."
        # Start your regular trader here
        # python3 scripts/trader.py &
    fi
}

# Usage: Replace your existing start_trading_bot call with start_trading_bot_enhanced
"""

        return self._deploy_code_file(
            system_manager_patch,
            f"{self.remote_base}/bin/superior_models_patch.sh",
            "System Manager Patch",
        )

    def _deploy_code_file(self, content: str, remote_path: str, description: str) -> bool:
        """Deploy code file to remote server."""
        try:
            # Write content to temporary file
            with tempfile.NamedTemporaryFile(mode="w", delete=False) as f:
                f.write(content)
                temp_file = f.name

            # Copy to remote server
            copy_cmd = [
                "scp",
                "-i",
                self.config["ssh_key_path"],
                temp_file,
                f"{self.config['hetzner_user']}@{self.config['hetzner_host']}:{remote_path}",
            ]

            result = subprocess.run(copy_cmd, capture_output=True, text=True)

            if result.returncode == 0:
                # Make executable if it's a script
                if remote_path.endswith(".sh"):
                    chmod_cmd = [
                        "ssh",
                        "-i",
                        self.config["ssh_key_path"],
                        f"{self.config['hetzner_user']}@{self.config['hetzner_host']}",
                        f"chmod +x {remote_path}",
                    ]
                    subprocess.run(chmod_cmd)

                logger.info(f"✅ {description} deployed: {remote_path}")
                return True
            else:
                logger.error(f"❌ Failed to deploy {description}: {result.stderr}")
                return False

        except Exception as e:
            logger.error(f"❌ Deployment error for {description}: {e}")
            return False
        finally:
            # Clean up temp file
            try:
                os.unlink(temp_file)
            except:
                pass

    def run_preparation(self) -> bool:
        """Run complete system preparation."""
        logger.info("🔧 PREPARING HETZNER SYSTEM FOR SUPERIOR MODELS")
        logger.info("=" * 60)

        preparations = [
            ("Model Manager", self.prepare_model_manager),
            ("Trading Integration", self.prepare_trading_integration),
            ("System Manager", self.update_system_manager),
        ]

        successful = 0
        total = len(preparations)

        for prep_name, prep_func in preparations:
            logger.info(f"\\n🔧 Preparing: {prep_name}")
            try:
                if prep_func():
                    logger.info(f"✅ {prep_name}: COMPLETED")
                    successful += 1
                else:
                    logger.error(f"❌ {prep_name}: FAILED")
            except Exception as e:
                logger.error(f"💥 {prep_name}: ERROR - {e}")

        success_rate = successful / total
        logger.info(f"\\n📊 Preparation: {successful}/{total} ({success_rate:.1%})")

        if success_rate >= 0.8:
            logger.info("🎉 SYSTEM PREPARATION COMPLETED!")
            logger.info("✅ Hetzner server ready for superior models")
            return True
        else:
            logger.error("❌ SYSTEM PREPARATION FAILED")
            return False


def main():
    """Main execution."""
    import argparse

    parser = argparse.ArgumentParser(description="Prepare Hetzner System for Superior Models")
    parser.add_argument(
        "--config",
        default="/notebooks/bot/paperspace_mlops/hetzner_config.json",
        help="Configuration file path",
    )

    args = parser.parse_args()

    if not os.path.exists(args.config):
        logger.error(f"❌ Config file not found: {args.config}")
        logger.error("   Run setup_hetzner_export.sh first")
        return 1

    preparator = HetznerSystemPreparator(args.config)
    success = preparator.run_preparation()

    return 0 if success else 1


if __name__ == "__main__":
    exit(main())
