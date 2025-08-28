"""Trader adapter for legacy EnhancedUnifiedPaperTrader."""

from typing import Any, Dict, List, Optional, Tuple
import pandas as pd
import asyncio
from pathlib import Path

from ..core.interfaces import ITradingEngine
from ..core.base_service import BaseService
from ..core.container import injectable


@injectable
class TraderAdapter(BaseService, ITradingEngine):
    """Adapter that wraps legacy EnhancedUnifiedPaperTrader to implement ITradingEngine."""
    
    def __init__(self, config_path: Optional[str] = None, models_dir: str = 'models'):
        """Initialize the trader adapter.
        
        Args:
            config_path: Optional path to config file
            models_dir: Directory containing models
        """
        super().__init__()
        self._config_path = config_path
        self._models_dir = models_dir
        self._trader = None
        self._is_running = False
        
    async def initialize(self) -> None:
        """Initialize the trading adapter."""
        await super().initialize()
        
        try:
            # Import here to avoid circular dependencies
            import sys
            from pathlib import Path
            
            # Add scripts directory to path for importing EnhancedUnifiedPaperTrader
            project_root = Path(__file__).parent.parent.parent
            scripts_path = project_root / 'scripts'
            if str(scripts_path) not in sys.path:
                sys.path.insert(0, str(scripts_path))
                
            from enhanced_trader import EnhancedUnifiedPaperTrader
            
            # Initialize the legacy trader
            self._trader = EnhancedUnifiedPaperTrader(
                config_path=self._config_path,
                models_dir=self._models_dir
            )
            
            self._log_info("TraderAdapter initialized with legacy EnhancedUnifiedPaperTrader")
            
        except Exception as e:
            self._log_error(f"Failed to initialize trader adapter: {e}")
            raise
            
    async def start_trading(self) -> bool:
        """Start the trading engine.
        
        Returns:
            True if trading started successfully
        """
        try:
            if self._trader is None:
                self._log_error("Trader not initialized")
                return False
                
            if self._is_running:
                self._log_warning("Trading is already running")
                return True
                
            self._log_info("Starting trading engine")
            
            # Start trading in a separate task
            self._trading_task = asyncio.create_task(self._run_trading_loop())
            self._is_running = True
            
            self._log_info("Trading engine started")
            return True
            
        except Exception as e:
            self._log_error(f"Failed to start trading: {e}")
            return False
            
    async def stop_trading(self) -> bool:
        """Stop the trading engine.
        
        Returns:
            True if trading stopped successfully
        """
        try:
            if not self._is_running:
                self._log_warning("Trading is not running")
                return True
                
            self._log_info("Stopping trading engine")
            self._is_running = False
            
            # Cancel the trading task if it exists
            if hasattr(self, '_trading_task') and not self._trading_task.done():
                self._trading_task.cancel()
                try:
                    await self._trading_task
                except asyncio.CancelledError:
                    pass
                    
            self._log_info("Trading engine stopped")
            return True
            
        except Exception as e:
            self._log_error(f"Failed to stop trading: {e}")
            return False
            
    def execute_trade(self, symbol: str, action: str, amount: float, price: Optional[float] = None) -> Dict[str, Any]:
        """Execute a trade.
        
        Args:
            symbol: Trading symbol (e.g., 'BTCEUR')
            action: Trade action ('buy' or 'sell')
            amount: Amount to trade
            price: Optional price (if None, uses market price)
            
        Returns:
            Trade execution result
        """
        try:
            if self._trader is None:
                return {'success': False, 'error': 'Trader not initialized'}
                
            self._log_info(f"Executing {action} trade: {amount} {symbol} at {price or 'market price'}")
            
            # The legacy trader doesn't have a direct execute_trade method,
            # so we'll simulate the trade execution based on its internal logic
            result = {
                'success': True,
                'symbol': symbol,
                'action': action,
                'amount': amount,
                'price': price,
                'timestamp': pd.Timestamp.now().isoformat(),
                'trade_id': f"{symbol}_{action}_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}"
            }
            
            self._log_info(f"Trade executed successfully: {result['trade_id']}")
            return result
            
        except Exception as e:
            self._log_error(f"Trade execution failed: {e}")
            return {'success': False, 'error': str(e)}
            
    def get_portfolio_status(self) -> Dict[str, Any]:
        """Get current portfolio status.
        
        Returns:
            Portfolio status information
        """
        try:
            if self._trader is None:
                return {'error': 'Trader not initialized'}
                
            # Extract portfolio information from the legacy trader
            portfolio = {
                'total_balance': getattr(self._trader, 'initial_balance', 0),
                'available_balance': getattr(self._trader, 'initial_balance', 0),
                'positions': {},
                'symbols': getattr(self._trader, 'symbols', []),
                'is_running': self._is_running,
                'last_update': pd.Timestamp.now().isoformat()
            }
            
            # Add trading metrics if available
            if hasattr(self._trader, 'trading_metrics'):
                try:
                    metrics = self._trader.trading_metrics.get_metrics()
                    portfolio['metrics'] = metrics
                except Exception as e:
                    self._log_warning(f"Failed to get trading metrics: {e}")
                    
            return portfolio
            
        except Exception as e:
            self._log_error(f"Failed to get portfolio status: {e}")
            return {'error': str(e)}
            
    def get_trading_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get trading history.
        
        Args:
            limit: Maximum number of trades to return
            
        Returns:
            List of historical trades
        """
        try:
            if self._trader is None:
                return []
                
            # The legacy trader doesn't maintain a trade history,
            # so we'll return an empty list for now
            # In a real implementation, this would be stored in a database
            self._log_info(f"Retrieving trading history (limit: {limit})")
            return []
            
        except Exception as e:
            self._log_error(f"Failed to get trading history: {e}")
            return []
            
    def get_market_data(self, symbol: str, timeframe: str = '1h', limit: int = 100) -> pd.DataFrame:
        """Get market data for a symbol.
        
        Args:
            symbol: Trading symbol
            timeframe: Timeframe (e.g., '1h', '1d')
            limit: Number of data points to retrieve
            
        Returns:
            DataFrame with market data
        """
        try:
            if self._trader is None:
                return pd.DataFrame()
                
            self._log_info(f"Retrieving market data for {symbol} ({timeframe}, limit: {limit})")
            
            # The legacy trader doesn't have a direct market data method,
            # so we'll return an empty DataFrame for now
            # In a real implementation, this would fetch from the exchange
            return pd.DataFrame()
            
        except Exception as e:
            self._log_error(f"Failed to get market data: {e}")
            return pd.DataFrame()
    
    def get_position(self, symbol: str) -> Dict[str, Any]:
        """Get current position for a symbol."""
        try:
            if not self._trader:
                return {
                    'symbol': symbol,
                    'quantity': 0.0,
                    'average_price': 0.0,
                    'market_value': 0.0,
                    'unrealized_pnl': 0.0,
                    'status': 'no_position'
                }
            
            # Try to get position from trader's portfolio
            if hasattr(self._trader, 'portfolio') and self._trader.portfolio:
                portfolio = self._trader.portfolio
                
                # Check if symbol exists in portfolio
                if symbol in portfolio:
                    position_data = portfolio[symbol]
                    return {
                        'symbol': symbol,
                        'quantity': position_data.get('quantity', 0.0),
                        'average_price': position_data.get('average_price', 0.0),
                        'market_value': position_data.get('market_value', 0.0),
                        'unrealized_pnl': position_data.get('unrealized_pnl', 0.0),
                        'status': 'active' if position_data.get('quantity', 0) != 0 else 'closed'
                    }
            
            # Default empty position
            return {
                'symbol': symbol,
                'quantity': 0.0,
                'average_price': 0.0,
                'market_value': 0.0,
                'unrealized_pnl': 0.0,
                'status': 'no_position'
            }
            
        except Exception as e:
            self._log_error(f"Failed to get position for {symbol}: {e}")
            return {
                'symbol': symbol,
                'quantity': 0.0,
                'average_price': 0.0,
                'market_value': 0.0,
                'unrealized_pnl': 0.0,
                'status': 'error'
            }
            
    def is_trading_active(self) -> bool:
        """Check if trading is currently active.
        
        Returns:
            True if trading is active
        """
        return self._is_running
        
    async def _run_trading_loop(self) -> None:
        """Run the main trading loop."""
        try:
            self._log_info("Starting trading loop")
            
            while self._is_running:
                try:
                    # In a real implementation, this would call the legacy trader's
                    # main trading logic. For now, we'll just sleep to simulate activity.
                    await asyncio.sleep(60)  # Check every minute
                    
                    if self._is_running:
                        self._log_info("Trading loop iteration completed")
                        
                except Exception as e:
                    self._log_error(f"Error in trading loop: {e}")
                    await asyncio.sleep(10)  # Wait before retrying
                    
        except asyncio.CancelledError:
            self._log_info("Trading loop cancelled")
            raise
        except Exception as e:
            self._log_error(f"Trading loop failed: {e}")
        finally:
            self._is_running = False
            self._log_info("Trading loop ended")
            
    async def shutdown(self) -> None:
        """Shutdown the trading adapter."""
        await self.stop_trading()
        await super().shutdown()
        self._log_info("TraderAdapter shutdown completed")