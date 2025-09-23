#!/usr/bin/env python3
"""
PPO Trading Environment for Cryptocurrency Trading

This environment provides a realistic trading simulation for training PPO agents
with proper market dynamics, transaction costs, and risk management.
"""

import logging
import warnings
from typing import Any, Dict, Optional, Tuple

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)


class PPOTradingEnvironment(gym.Env):
    """
    Cryptocurrency trading environment for PPO agent training.

    This environment simulates realistic trading conditions including:
    - 0.25% transaction costs
    - Slippage modeling
    - Portfolio risk management
    - Multi-step episodes with proper termination
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        data: pd.DataFrame,
        features: np.ndarray,
        initial_balance: float = 10000.0,
        transaction_cost: float = 0.0025,  # 0.25%
        slippage: float = 0.0005,  # 0.05%
        max_position_pct: float = 0.15,  # 15% max position size
        sequence_length: int = 32,
        symbol: str = "UNKNOWN",
    ):
        """
        Initialize PPO trading environment.

        Args:
            data: OHLCV market data
            features: Engineered features array (103 features expected)
            initial_balance: Starting portfolio value
            transaction_cost: Transaction cost as decimal (0.0025 = 0.25%)
            slippage: Slippage cost as decimal
            max_position_pct: Maximum position size as percentage of portfolio
            sequence_length: Length of observation sequences
            symbol: Trading symbol for logging
        """
        super().__init__()

        self.data = data.copy()
        self.features = features
        self.initial_balance = initial_balance
        self.transaction_cost = transaction_cost
        self.slippage = slippage
        self.max_position_pct = max_position_pct
        self.sequence_length = sequence_length
        self.symbol = symbol

        # Validate inputs
        if len(self.data) != len(self.features):
            raise ValueError(
                f"Data length ({len(self.data)}) != features length ({len(self.features)})"
            )

        if self.features.shape[1] != 103:
            logger.warning(f"Expected 103 features, got {self.features.shape[1]} for {symbol}")

        # Environment state
        self.current_step = 0
        self.balance = initial_balance
        self.position = 0.0  # Current position (positive = long, negative = short)
        self.position_value = 0.0
        self.total_trades = 0
        self.profitable_trades = 0

        # Performance tracking
        self.portfolio_values = []
        self.returns = []
        self.actions_taken = []
        self.rewards_earned = []

        # Define action and observation spaces
        # Action space: 3 discrete actions [0=HOLD, 1=BUY, 2=SELL]
        self.action_space = spaces.Discrete(3)

        # Observation space: sequence of features + portfolio state
        feature_dim = self.features.shape[1]
        portfolio_state_dim = 4  # [balance_pct, position_pct, portfolio_return, trade_count]
        obs_dim = feature_dim + portfolio_state_dim

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.sequence_length, obs_dim), dtype=np.float32
        )

        # Price columns
        self.close_prices = self.data["close"].values

        logger.info(
            f"PPO Environment initialized for {symbol}: "
            f"Steps={len(self.data)}, Features={feature_dim}, "
            f"Transaction_cost={transaction_cost:.2%}"
        )

    def reset(
        self, seed: Optional[int] = None, options: Optional[Dict] = None
    ) -> Tuple[np.ndarray, Dict]:
        """Reset the environment to initial state."""
        super().reset(seed=seed)

        self.current_step = self.sequence_length
        self.balance = self.initial_balance
        self.position = 0.0
        self.position_value = 0.0
        self.total_trades = 0
        self.profitable_trades = 0

        # Clear performance tracking
        self.portfolio_values = [self.initial_balance]
        self.returns = [0.0]
        self.actions_taken = []
        self.rewards_earned = []

        observation = self._get_observation()
        info = self._get_info()

        return observation, info

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Execute one trading step."""
        if self.current_step >= len(self.data) - 1:
            terminated = True
            truncated = False
            reward = 0.0
            observation = self._get_observation()
            info = self._get_info()
            return observation, reward, terminated, truncated, info

        # Execute action
        reward = self._execute_action(action)

        # Update step
        self.current_step += 1

        # Calculate portfolio value
        current_price = self.close_prices[self.current_step]
        portfolio_value = self.balance + (self.position * current_price)
        self.portfolio_values.append(portfolio_value)

        # Calculate returns
        portfolio_return = (portfolio_value - self.initial_balance) / self.initial_balance
        step_return = (portfolio_value - self.portfolio_values[-2]) / self.portfolio_values[-2]
        self.returns.append(portfolio_return)

        # Store action and reward
        self.actions_taken.append(action)
        self.rewards_earned.append(reward)

        # Check termination conditions
        terminated = (
            self.current_step >= len(self.data) - 1
            or portfolio_value <= self.initial_balance * 0.5  # 50% drawdown limit
        )
        truncated = False

        observation = self._get_observation()
        info = self._get_info()

        return observation, reward, terminated, truncated, info

    def _execute_action(self, action: int) -> float:
        """Execute trading action and return reward."""
        current_price = self.close_prices[self.current_step]
        old_portfolio_value = self.balance + (self.position * current_price)

        # Action mapping: 0=HOLD, 1=BUY, 2=SELL
        if action == 1:  # BUY
            reward = self._execute_buy(current_price)
        elif action == 2:  # SELL
            reward = self._execute_sell(current_price)
        else:  # HOLD
            reward = self._calculate_hold_reward()

        # Calculate new portfolio value
        new_portfolio_value = self.balance + (self.position * current_price)

        # Add portfolio change component to reward
        portfolio_change = (new_portfolio_value - old_portfolio_value) / old_portfolio_value
        reward += portfolio_change * 10  # Scale portfolio performance

        return reward

    def _execute_buy(self, price: float) -> float:
        """Execute buy order with transaction costs."""
        if self.position >= 0:  # Can only buy if not already long
            # Calculate maximum buy amount (percentage of portfolio)
            portfolio_value = self.balance + (self.position * price)
            max_position_value = portfolio_value * self.max_position_pct

            if self.balance > max_position_value * (1 + self.transaction_cost + self.slippage):
                # Apply costs
                effective_price = price * (1 + self.slippage)
                cost_multiplier = 1 + self.transaction_cost

                # Calculate position size
                position_size = max_position_value / (effective_price * cost_multiplier)
                total_cost = position_size * effective_price * cost_multiplier

                if total_cost <= self.balance:
                    self.balance -= total_cost
                    self.position += position_size
                    self.total_trades += 1

                    # Reward for taking action when appropriate
                    return 0.1

        # Penalty for invalid or unnecessary buy
        return -0.05

    def _execute_sell(self, price: float) -> float:
        """Execute sell order with transaction costs."""
        if self.position > 0:  # Can only sell if holding position
            # Apply costs
            effective_price = price * (1 - self.slippage)
            proceeds = self.position * effective_price * (1 - self.transaction_cost)

            # Calculate profit/loss
            purchase_value = (
                self.position_value if hasattr(self, "position_value") else self.position * price
            )
            profit_loss = proceeds - purchase_value

            self.balance += proceeds
            self.position = 0.0
            self.position_value = 0.0
            self.total_trades += 1

            if profit_loss > 0:
                self.profitable_trades += 1
                return 0.2  # Reward for profitable trade
            else:
                return -0.1  # Small penalty for loss

        # Penalty for invalid sell
        return -0.05

    def _calculate_hold_reward(self) -> float:
        """Calculate reward for holding position."""
        if self.current_step < 2:
            return 0.0

        # Reward based on recent price movement and position alignment
        price_change = (
            self.close_prices[self.current_step] - self.close_prices[self.current_step - 1]
        )
        price_return = price_change / self.close_prices[self.current_step - 1]

        if self.position > 0:
            # Long position: reward positive price movements
            return price_return * 5
        elif self.position < 0:
            # Short position: reward negative price movements
            return -price_return * 5
        else:
            # No position: small penalty for inaction in trending markets
            if abs(price_return) > 0.01:  # 1% movement
                return -0.01
            return 0.0

    def _get_observation(self) -> np.ndarray:
        """Get current observation state."""
        if self.current_step < self.sequence_length:
            # Pad with zeros if we don't have enough history
            features_seq = np.zeros((self.sequence_length, self.features.shape[1]))
            available_steps = self.current_step + 1
            features_seq[-available_steps:] = self.features[:available_steps]
        else:
            # Use last sequence_length steps
            start_idx = self.current_step - self.sequence_length + 1
            end_idx = self.current_step + 1
            features_seq = self.features[start_idx:end_idx]

        # Add portfolio state to each step
        current_price = self.close_prices[self.current_step]
        portfolio_value = self.balance + (self.position * current_price)

        portfolio_state = np.array(
            [
                self.balance / self.initial_balance,  # Normalized balance
                (self.position * current_price) / portfolio_value
                if portfolio_value > 0
                else 0,  # Position percentage
                (portfolio_value - self.initial_balance) / self.initial_balance,  # Portfolio return
                min(self.total_trades / 100.0, 1.0),  # Normalized trade count
            ]
        )

        # Broadcast portfolio state to all sequence steps
        portfolio_states = np.tile(portfolio_state, (self.sequence_length, 1))

        # Combine features and portfolio state
        observation = np.concatenate([features_seq, portfolio_states], axis=1)

        return observation.astype(np.float32)

    def _get_info(self) -> Dict[str, Any]:
        """Get environment info."""
        current_price = self.close_prices[self.current_step]
        portfolio_value = self.balance + (self.position * current_price)

        info = {
            "step": self.current_step,
            "balance": self.balance,
            "position": self.position,
            "portfolio_value": portfolio_value,
            "portfolio_return": (portfolio_value - self.initial_balance) / self.initial_balance,
            "total_trades": self.total_trades,
            "profitable_trades": self.profitable_trades,
            "win_rate": self.profitable_trades / max(self.total_trades, 1),
            "current_price": current_price,
        }

        return info

    def get_performance_metrics(self) -> Dict[str, float]:
        """Calculate comprehensive performance metrics."""
        if len(self.portfolio_values) < 2:
            return {"sharpe_ratio": 0.0, "max_drawdown": 0.0, "total_return": 0.0}

        portfolio_values = np.array(self.portfolio_values)
        returns = np.diff(portfolio_values) / portfolio_values[:-1]

        # Calculate metrics
        total_return = (portfolio_values[-1] - self.initial_balance) / self.initial_balance
        sharpe_ratio = np.mean(returns) / (np.std(returns) + 1e-8) * np.sqrt(252)  # Annualized

        # Maximum drawdown
        peak = np.maximum.accumulate(portfolio_values)
        drawdown = (portfolio_values - peak) / peak
        max_drawdown = np.min(drawdown)

        return {
            "total_return": total_return,
            "sharpe_ratio": sharpe_ratio,
            "max_drawdown": max_drawdown,
            "win_rate": self.profitable_trades / max(self.total_trades, 1),
            "total_trades": self.total_trades,
            "profitable_trades": self.profitable_trades,
        }

    def render(self, mode: str = "human") -> None:
        """Render environment state."""
        if mode == "human":
            info = self._get_info()
            print(
                f"Step: {info['step']}, Portfolio: ${info['portfolio_value']:.2f}, "
                f"Return: {info['portfolio_return']:.2%}, Trades: {info['total_trades']}"
            )


def create_ppo_environment(
    data: pd.DataFrame, features: np.ndarray, symbol: str, config: Optional[Dict] = None
) -> PPOTradingEnvironment:
    """
    Factory function to create PPO trading environment with proper configuration.

    Args:
        data: OHLCV market data
        features: Engineered features (should be 103 features)
        symbol: Trading symbol
        config: Optional configuration overrides

    Returns:
        Configured PPOTradingEnvironment
    """
    default_config = {
        "initial_balance": 10000.0,
        "transaction_cost": 0.0025,  # 0.25%
        "slippage": 0.0005,  # 0.05%
        "max_position_pct": 0.15,  # 15%
        "sequence_length": 32,
    }

    if config:
        default_config.update(config)

    env = PPOTradingEnvironment(data=data, features=features, symbol=symbol, **default_config)

    logger.info(f"Created PPO environment for {symbol} with {len(data)} steps")
    return env
