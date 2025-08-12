#!/usr/bin/env python3
"""
PPO (Proximal Policy Optimization) Trainer using Centralized DatasetBuilder
==========================================================================

This module provides a PPO trainer for reinforcement learning-based trading
that uses the centralized DatasetBuilder for consistent data processing.
"""

import os
import numpy as np
import pandas as pd
import logging
from typing import Dict, Any, Optional, Tuple, List
from pathlib import Path
import json
import pickle

try:
    import tensorflow as tf
    from tensorflow.keras import layers, models, optimizers
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False

from dataset_builder import DatasetBuilder
from model_adapter import ModelAdapter
from time_series_cv import get_time_series_folds
from cost_aware_evaluation import CostAwareEvaluator

logger = logging.getLogger(__name__)


class TradingEnvironment:
    """
    Trading environment for reinforcement learning.
    
    Provides a gym-like interface for trading decisions with
    realistic transaction costs and market dynamics.
    """
    
    def __init__(self, 
                 features: np.ndarray,
                 prices: np.ndarray,
                 cost_model: Dict[str, float] = None):
        """
        Initialize trading environment.
        
        Args:
            features: Feature matrix (time x features)
            prices: Price series
            cost_model: Transaction cost parameters
        """
        self.features = features
        self.prices = prices
        self.cost_model = cost_model or {'fee_bps': 10, 'slippage_bps': 5}
        
        self.current_step = 0
        self.position = 0.0  # -1 (short), 0 (neutral), 1 (long)
        self.cash = 10000.0  # Starting cash
        self.portfolio_value = self.cash
        self.trade_history = []
        
        self.max_steps = len(features) - 1
    
    def reset(self):
        """Reset environment to initial state."""
        self.current_step = 0
        self.position = 0.0
        self.cash = 10000.0
        self.portfolio_value = self.cash
        self.trade_history = []
        return self._get_state()
    
    def step(self, action):
        """
        Execute action and return next state, reward, done, info.
        
        Args:
            action: 0=hold, 1=buy, 2=sell
            
        Returns:
            next_state, reward, done, info
        """
        if self.current_step >= self.max_steps:
            return self._get_state(), 0, True, {}
        
        # Convert action to position change
        if action == 0:  # Hold
            target_position = self.position
        elif action == 1:  # Buy/Long
            target_position = 1.0
        elif action == 2:  # Sell/Short
            target_position = -1.0
        else:
            target_position = self.position  # Invalid action, hold
        
        # Execute trade if position changes
        reward = 0
        if target_position != self.position:
            reward = self._execute_trade(target_position)
        
        # Move to next step
        self.current_step += 1
        
        # Calculate holding reward
        if self.current_step < len(self.prices):
            price_return = (self.prices[self.current_step] - self.prices[self.current_step - 1]) / self.prices[self.current_step - 1]
            holding_reward = self.position * price_return
            reward += holding_reward
        
        # Update portfolio value
        if self.current_step < len(self.prices):
            self.portfolio_value = self.cash + self.position * self.prices[self.current_step] * 1000
        
        done = self.current_step >= self.max_steps
        info = {
            'portfolio_value': self.portfolio_value,
            'position': self.position,
            'step': self.current_step
        }
        
        return self._get_state(), reward, done, info
    
    def _get_state(self):
        """Get current state observation."""
        if self.current_step >= len(self.features):
            # Return zeros if beyond data
            return np.zeros(self.features.shape[1] + 3)
        
        # Current features + position + portfolio info
        features = self.features[self.current_step]
        position_info = np.array([
            self.position,
            self.portfolio_value / 10000.0 - 1.0,  # Normalized portfolio return
            self.current_step / self.max_steps  # Time progress
        ])
        
        return np.concatenate([features, position_info])
    
    def _execute_trade(self, target_position):
        """Execute trade and return immediate reward/cost."""
        position_change = abs(target_position - self.position)
        
        if position_change > 0:
            # Calculate transaction costs
            trade_value = position_change * self.prices[self.current_step] * 1000
            fee = trade_value * self.cost_model['fee_bps'] / 10000
            slippage = trade_value * self.cost_model['slippage_bps'] / 10000
            total_cost = fee + slippage
            
            # Update cash and position
            self.cash -= total_cost
            self.position = target_position
            
            # Record trade
            self.trade_history.append({
                'step': self.current_step,
                'action': target_position,
                'cost': total_cost,
                'price': self.prices[self.current_step]
            })
            
            return -total_cost / 1000  # Normalized cost penalty
        
        return 0


class PPOModelAdapter(ModelAdapter):
    """PPO model adapter for reinforcement learning trading."""
    
    def __init__(self, name: str = "PPO", config: Optional[Dict[str, Any]] = None):
        if not TENSORFLOW_AVAILABLE:
            raise ImportError("TensorFlow is required for PPO models")
        
        super().__init__(name, config)
        self.actor = None
        self.critic = None
        self.env = None
        
    def fit(self, 
            X: np.ndarray, 
            y: np.ndarray,
            train_idx: np.ndarray,
            val_idx: np.ndarray,
            prices: np.ndarray = None,
            **kwargs) -> 'PPOModelAdapter':
        """Train PPO model using environment interaction."""
        
        if prices is None:
            raise ValueError("Prices array is required for PPO training")
        
        # Split data
        X_train = X[train_idx]
        prices_train = prices[train_idx]
        
        # Create environment
        self.env = TradingEnvironment(
            features=X_train,
            prices=prices_train,
            cost_model={'fee_bps': self.config.get('fee_bps', 10), 'slippage_bps': self.config.get('slippage_bps', 5)}
        )
        
        # Build networks
        state_dim = X_train.shape[1] + 3  # features + position info
        self.actor = self._build_actor(state_dim)
        self.critic = self._build_critic(state_dim)
        
        # Train using PPO algorithm
        self._train_ppo()
        
        self.is_fitted = True
        logger.info("PPO training completed")
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Generate PPO predictions (actions)."""
        if not self.is_fitted:
            raise ValueError("Model not fitted")
        
        # Add position info (assume neutral position for prediction)
        batch_size = X.shape[0]
        position_info = np.zeros((batch_size, 3))
        position_info[:, 2] = np.arange(batch_size) / batch_size  # Time progress
        
        states = np.concatenate([X, position_info], axis=1)
        
        # Get action probabilities
        action_probs = self.actor(states)
        actions = tf.argmax(action_probs, axis=1)
        
        return actions.numpy()
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Generate action probabilities."""
        if not self.is_fitted:
            raise ValueError("Model not fitted")
        
        # Add position info
        batch_size = X.shape[0]
        position_info = np.zeros((batch_size, 3))
        position_info[:, 2] = np.arange(batch_size) / batch_size
        
        states = np.concatenate([X, position_info], axis=1)
        
        # Return probability of taking action (1=buy)
        action_probs = self.actor(states)
        return action_probs[:, 1].numpy()  # Buy probability
    
    def get_artifacts(self) -> Dict[str, Any]:
        """Get PPO model artifacts."""
        if not self.is_fitted:
            raise ValueError("Model not fitted")
        
        # Save model weights
        import tempfile
        actor_weights = None
        critic_weights = None
        
        if self.actor:
            with tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as f:
                self.actor.save_weights(f.name)
                with open(f.name, 'rb') as weights_file:
                    actor_weights = weights_file.read()
                os.unlink(f.name)
        
        if self.critic:
            with tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as f:
                self.critic.save_weights(f.name)
                with open(f.name, 'rb') as weights_file:
                    critic_weights = weights_file.read()
                os.unlink(f.name)
        
        return {
            'name': self.name,
            'config': self.config,
            'actor_weights': actor_weights,
            'critic_weights': critic_weights,
            'actor_config': self.actor.get_config() if self.actor else None,
            'critic_config': self.critic.get_config() if self.critic else None,
            'metadata': self.metadata,
            'is_fitted': self.is_fitted
        }
    
    def _restore_from_artifacts(self, artifacts: Dict[str, Any]) -> None:
        """Restore PPO from artifacts."""
        import tempfile
        
        self.config = artifacts['config']
        self.metadata = artifacts['metadata']
        self.is_fitted = artifacts['is_fitted']
        
        # Restore actor
        if artifacts['actor_config'] and artifacts['actor_weights']:
            self.actor = tf.keras.Model.from_config(artifacts['actor_config'])
            with tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as f:
                f.write(artifacts['actor_weights'])
                f.flush()
                self.actor.load_weights(f.name)
                os.unlink(f.name)
        
        # Restore critic
        if artifacts['critic_config'] and artifacts['critic_weights']:
            self.critic = tf.keras.Model.from_config(artifacts['critic_config'])
            with tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as f:
                f.write(artifacts['critic_weights'])
                f.flush()
                self.critic.load_weights(f.name)
                os.unlink(f.name)
    
    def _build_actor(self, state_dim: int) -> tf.keras.Model:
        """Build actor network."""
        inputs = layers.Input(shape=(state_dim,))
        
        x = layers.Dense(self.config.get('hidden_units', 128), activation='relu')(inputs)
        x = layers.Dropout(self.config.get('dropout', 0.1))(x)
        x = layers.Dense(self.config.get('hidden_units', 128) // 2, activation='relu')(x)
        x = layers.Dropout(self.config.get('dropout', 0.1))(x)
        
        # Output layer (3 actions: hold, buy, sell)
        outputs = layers.Dense(3, activation='softmax')(x)
        
        model = models.Model(inputs, outputs)
        model.compile(optimizer=optimizers.Adam(learning_rate=self.config.get('actor_lr', 0.0003)))
        
        return model
    
    def _build_critic(self, state_dim: int) -> tf.keras.Model:
        """Build critic network."""
        inputs = layers.Input(shape=(state_dim,))
        
        x = layers.Dense(self.config.get('hidden_units', 128), activation='relu')(inputs)
        x = layers.Dropout(self.config.get('dropout', 0.1))(x)
        x = layers.Dense(self.config.get('hidden_units', 128) // 2, activation='relu')(x)
        x = layers.Dropout(self.config.get('dropout', 0.1))(x)
        
        # Value output
        outputs = layers.Dense(1)(x)
        
        model = models.Model(inputs, outputs)
        model.compile(optimizer=optimizers.Adam(learning_rate=self.config.get('critic_lr', 0.001)))
        
        return model
    
    def _train_ppo(self):
        """Train using PPO algorithm."""
        episodes = self.config.get('episodes', 100)
        max_steps = self.config.get('max_steps_per_episode', 1000)
        
        for episode in range(episodes):
            states, actions, rewards, values, log_probs = self._collect_trajectory(max_steps)
            
            if len(states) > 0:
                # Calculate advantages
                advantages = self._calculate_advantages(rewards, values)
                
                # Update networks
                self._update_networks(states, actions, advantages, log_probs, values)
            
            if (episode + 1) % 20 == 0:
                logger.debug(f"Episode {episode + 1}/{episodes} completed")
    
    def _collect_trajectory(self, max_steps: int) -> Tuple[List, List, List, List, List]:
        """Collect trajectory from environment."""
        states, actions, rewards, values, log_probs = [], [], [], [], []
        
        state = self.env.reset()
        
        for _ in range(max_steps):
            state_tensor = tf.expand_dims(state, 0)
            
            # Get action probabilities and value
            action_probs = self.actor(state_tensor)
            value = self.critic(state_tensor)
            
            # Sample action
            action = tf.random.categorical(tf.math.log(action_probs), 1)[0, 0]
            log_prob = tf.math.log(action_probs[0, action])
            
            # Take step
            next_state, reward, done, _ = self.env.step(action.numpy())
            
            # Store transition
            states.append(state)
            actions.append(action.numpy())
            rewards.append(reward)
            values.append(value[0, 0].numpy())
            log_probs.append(log_prob.numpy())
            
            state = next_state
            
            if done:
                break
        
        return states, actions, rewards, values, log_probs
    
    def _calculate_advantages(self, rewards: List, values: List) -> np.ndarray:
        """Calculate advantages using GAE."""
        gamma = self.config.get('gamma', 0.99)
        lam = self.config.get('lambda', 0.95)
        
        advantages = np.zeros(len(rewards))
        advantage = 0
        
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = 0
            else:
                next_value = values[t + 1]
            
            delta = rewards[t] + gamma * next_value - values[t]
            advantage = delta + gamma * lam * advantage
            advantages[t] = advantage
        
        return advantages
    
    def _update_networks(self, states: List, actions: List, advantages: np.ndarray, 
                        old_log_probs: List, values: List):
        """Update actor and critic networks."""
        states = tf.convert_to_tensor(states, dtype=tf.float32)
        actions = tf.convert_to_tensor(actions, dtype=tf.int32)
        advantages = tf.convert_to_tensor(advantages, dtype=tf.float32)
        old_log_probs = tf.convert_to_tensor(old_log_probs, dtype=tf.float32)
        returns = advantages + tf.convert_to_tensor(values, dtype=tf.float32)
        
        # Update actor
        with tf.GradientTape() as tape:
            action_probs = self.actor(states)
            new_log_probs = tf.math.log(tf.gather(action_probs, actions[:, None], batch_dims=1)[:, 0])
            
            ratio = tf.exp(new_log_probs - old_log_probs)
            clipped_ratio = tf.clip_by_value(ratio, 0.8, 1.2)  # PPO clip
            
            actor_loss = -tf.reduce_mean(tf.minimum(ratio * advantages, clipped_ratio * advantages))
        
        actor_grads = tape.gradient(actor_loss, self.actor.trainable_variables)
        self.actor.optimizer.apply_gradients(zip(actor_grads, self.actor.trainable_variables))
        
        # Update critic
        with tf.GradientTape() as tape:
            predicted_values = self.critic(states)[:, 0]
            critic_loss = tf.reduce_mean(tf.square(returns - predicted_values))
        
        critic_grads = tape.gradient(critic_loss, self.critic.trainable_variables)
        self.critic.optimizer.apply_gradients(zip(critic_grads, self.critic.trainable_variables))


class PPOTrainer:
    """
    PPO trainer using centralized DatasetBuilder.
    
    Provides reinforcement learning-based trading strategy training.
    """
    
    def __init__(self, 
                 dataset_builder: DatasetBuilder,
                 config: Optional[Dict[str, Any]] = None):
        """
        Initialize PPO trainer.
        
        Args:
            dataset_builder: Centralized dataset builder
            config: PPO model configuration
        """
        self.dataset_builder = dataset_builder
        self.config = config or self._get_default_config()
        self.cost_evaluator = CostAwareEvaluator()
    
    def train_symbol(self, 
                    symbol: str,
                    interval: str = "15m",
                    n_splits: int = 3,  # Fewer splits for RL
                    save_artifacts: bool = True) -> Dict[str, Any]:
        """
        Train PPO model for a single symbol.
        
        Args:
            symbol: Trading symbol
            interval: Data interval
            n_splits: Number of CV splits
            save_artifacts: Whether to save model artifacts
            
        Returns:
            Training results dictionary
        """
        logger.info(f"Training PPO model for {symbol}")
        
        # Get dataset
        features_df, metadata = self.dataset_builder.get_dataset(
            symbol=symbol, 
            interval=interval
        )
        
        # Validate dataset
        validation_report = self.dataset_builder.validate_dataset(features_df, metadata)
        if not validation_report['valid']:
            raise ValueError(f"Dataset validation failed: {validation_report['errors']}")
        
        # Prepare features and prices
        X, prices = self._prepare_data(features_df)
        
        if len(X) == 0:
            raise ValueError("No valid samples found")
        
        # Time-series cross-validation
        cv_folds = get_time_series_folds(
            timestamps=features_df.index,
            n_splits=n_splits,
            embargo_pct=0.05  # Larger embargo for RL
        )
        
        # Train on each fold
        fold_results = []
        models = []
        
        for fold_idx, (train_idx, val_idx) in enumerate(cv_folds):
            logger.info(f"Training fold {fold_idx + 1}/{len(cv_folds)}")
            
            # Create model adapter
            model_adapter = PPOModelAdapter(
                name=f"PPO_{symbol}_fold_{fold_idx}",
                config=self.config
            )
            
            # Train
            model_adapter.fit(X, np.zeros(len(X)), train_idx, val_idx, prices=prices)
            
            # Evaluate
            fold_metrics = self._evaluate_fold(model_adapter, X, val_idx, prices, features_df)
            fold_metrics['fold'] = fold_idx
            fold_results.append(fold_metrics)
            
            models.append(model_adapter)
        
        # Aggregate results
        results = self._aggregate_fold_results(fold_results)
        results['symbol'] = symbol
        results['metadata'] = metadata.__dict__
        results['validation_report'] = validation_report
        
        # Train final model if requested
        if save_artifacts:
            final_model = self._train_final_model(X, prices, symbol)
            self._save_artifacts(final_model, symbol, results)
        
        logger.info(f"PPO training completed for {symbol}")
        return results
    
    def _prepare_data(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare features and prices for PPO."""
        # Select features (exclude target and non-numeric columns)
        feature_columns = [col for col in df.columns 
                          if col != 'target' and df[col].dtype in ['float64', 'int64']]
        
        if len(feature_columns) == 0:
            raise ValueError("No numeric features found")
        
        # Extract features and prices
        X = df[feature_columns].fillna(0).values  # Fill NaN with 0 for RL
        prices = df['close'].values
        
        logger.info(f"Prepared {X.shape[0]} samples with {X.shape[1]} features")
        return X, prices
    
    def _evaluate_fold(self, 
                      model: PPOModelAdapter,
                      X: np.ndarray,
                      val_idx: np.ndarray,
                      prices: np.ndarray,
                      features_df: pd.DataFrame) -> Dict[str, Any]:
        """Evaluate PPO model performance."""
        # Create test environment
        X_val = X[val_idx]
        prices_val = prices[val_idx]
        
        test_env = TradingEnvironment(
            features=X_val,
            prices=prices_val,
            cost_model={'fee_bps': self.config.get('fee_bps', 10), 'slippage_bps': 5}
        )
        
        # Run episode
        state = test_env.reset()
        total_reward = 0
        actions_taken = []
        
        for _ in range(len(X_val) - 1):
            # Get action from model
            state_tensor = tf.expand_dims(state, 0)
            action_probs = model.actor(state_tensor)
            action = tf.argmax(action_probs, axis=1)[0].numpy()
            
            # Take step
            next_state, reward, done, info = test_env.step(action)
            total_reward += reward
            actions_taken.append(action)
            state = next_state
            
            if done:
                break
        
        # Calculate metrics
        final_portfolio_value = test_env.portfolio_value
        total_return = (final_portfolio_value - 10000) / 10000
        num_trades = len(test_env.trade_history)
        
        return {
            'total_reward': total_reward,
            'total_return': total_return,
            'final_portfolio_value': final_portfolio_value,
            'num_trades': num_trades,
            'actions': actions_taken
        }
    
    def _aggregate_fold_results(self, fold_results: list) -> Dict[str, Any]:
        """Aggregate results across folds."""
        if not fold_results:
            return {}
        
        metrics = ['total_reward', 'total_return', 'final_portfolio_value']
        aggregated = {}
        
        for metric in metrics:
            values = [fold[metric] for fold in fold_results if metric in fold]
            if values:
                aggregated[f'avg_{metric}'] = np.mean(values)
                aggregated[f'std_{metric}'] = np.std(values)
        
        # Sum trades
        total_trades = sum([fold['num_trades'] for fold in fold_results])
        aggregated['total_trades'] = total_trades
        
        aggregated['n_folds'] = len(fold_results)
        aggregated['fold_results'] = fold_results
        
        return aggregated
    
    def _train_final_model(self, X: np.ndarray, prices: np.ndarray, symbol: str) -> PPOModelAdapter:
        """Train final model on full dataset."""
        # Use 80/20 split for final training
        split_idx = int(0.8 * len(X))
        train_idx = np.arange(split_idx)
        val_idx = np.arange(split_idx, len(X))
        
        final_model = PPOModelAdapter(
            name=f"PPO_{symbol}_final",
            config=self.config
        )
        
        final_model.fit(X, np.zeros(len(X)), train_idx, val_idx, prices=prices)
        return final_model
    
    def _save_artifacts(self, 
                       model: PPOModelAdapter,
                       symbol: str,
                       results: Dict[str, Any]) -> None:
        """Save model artifacts."""
        # Create artifact directory
        artifacts_dir = Path("models") / "ppo" / symbol.lower()
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        
        # Save model
        model_path = artifacts_dir / "model.pkl"
        model.save(model_path)
        
        # Save training results
        results_path = artifacts_dir / "training_results.json"
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        # Create latest symlink
        latest_dir = artifacts_dir.parent / "latest"
        if latest_dir.is_symlink():
            latest_dir.unlink()
        elif latest_dir.exists():
            import shutil
            shutil.rmtree(latest_dir)
        
        latest_dir.symlink_to(artifacts_dir.name, target_is_directory=True)
        
        logger.info(f"Artifacts saved to {artifacts_dir}")
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default PPO configuration."""
        return {
            'hidden_units': 128,
            'dropout': 0.1,
            'actor_lr': 0.0003,
            'critic_lr': 0.001,
            'episodes': 100,
            'max_steps_per_episode': 1000,
            'gamma': 0.99,
            'lambda': 0.95,
            'fee_bps': 10,
            'slippage_bps': 5
        }