"""
Enhanced Telegram Notifier with Interactive Commands
"""
import asyncio
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
import os
import json
from pathlib import Path

from telegram import Bot
from telegram.error import TelegramError

logger = logging.getLogger(__name__)

class EnhancedTelegramNotifier:
    """Enhanced Telegram notifier with interactive commands and performance reports."""

    def __init__(self, bot_token: str, chat_id: str, trading_manager=None, **kwargs):
        self.bot_token = bot_token
        self.chat_id = chat_id
        self.bot = Bot(token=bot_token)
        self.trading_manager = trading_manager
        self.command_handlers = self._register_commands()
        self.performance_history = []

    def _register_commands(self) -> Dict[str, callable]:
        """Register interactive commands."""
        return {
            '/status': self._cmd_status,
            '/start': self._cmd_start,
            '/stop': self._cmd_stop,
            '/restart': self._cmd_restart,
            '/performance': self._cmd_performance,
            '/health': self._cmd_health,
            '/balance': self._cmd_balance,
            '/trades': self._cmd_recent_trades,
            '/logs': self._cmd_logs,
            '/config': self._cmd_config
        }

    async def handle_command(self, command: str, args: List[str] = None) -> str:
        """Handle incoming command."""
        if args is None:
            args = []

        cmd = command.split()[0].lower()
        if cmd in self.command_handlers:
            try:
                return await self.command_handlers[cmd](args)
            except Exception as e:
                logger.error(f"Command error: {e}")
                return f"❌ Error executing command: {str(e)}"
        else:
            return "Unknown command. Available: /status, /start, /stop, /restart, /performance, /health, /balance, /trades, /logs, /config"

    async def _cmd_status(self, args: List[str]) -> str:
        """Get system status."""
        try:
            # Check tmux session
            result = await asyncio.create_subprocess_shell(
                'tmux has-session -t trading_session',
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            await result.wait()

            if result.returncode == 0:
                status = "✅ Running"
            else:
                status = "❌ Stopped"

            # Get system info
            hostname = os.uname().nodename
            uptime = await self._get_system_uptime()

            return f"""🤖 <b>System Status</b>

<b>Trading Bot:</b> {status}
<b>Server:</b> {hostname}
<b>Uptime:</b> {uptime}
<b>Last Check:</b> {datetime.now().strftime('%H:%M:%S')}
"""
        except Exception as e:
            return f"❌ Status check failed: {str(e)}"

    async def _cmd_start(self, args: List[str]) -> str:
        """Start trading system."""
        try:
            result = await asyncio.create_subprocess_shell(
                '/opt/trading_bot/tmux_manager.sh start',
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            await result.wait()

            if result.returncode == 0:
                return "🚀 Trading system started successfully"
            else:
                return "❌ Failed to start trading system"
        except Exception as e:
            return f"❌ Start command failed: {str(e)}"

    async def _cmd_stop(self, args: List[str]) -> str:
        """Stop trading system."""
        try:
            result = await asyncio.create_subprocess_shell(
                '/opt/trading_bot/tmux_manager.sh stop',
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            await result.wait()

            return "🛑 Trading system stopped"
        except Exception as e:
            return f"❌ Stop command failed: {str(e)}"

    async def _cmd_restart(self, args: List[str]) -> str:
        """Restart trading system."""
        try:
            # Stop
            stop_result = await asyncio.create_subprocess_shell(
                '/opt/trading_bot/tmux_manager.sh stop',
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            await stop_result.wait()

            await asyncio.sleep(2)

            # Start
            start_result = await asyncio.create_subprocess_shell(
                '/opt/trading_bot/tmux_manager.sh start',
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            await start_result.wait()

            if start_result.returncode == 0:
                return "🔄 Trading system restarted successfully"
            else:
                return "❌ Failed to restart trading system"
        except Exception as e:
            return f"❌ Restart command failed: {str(e)}"

    async def _cmd_performance(self, args: List[str]) -> str:
        """Get performance metrics."""
        try:
            # Read performance data
            perf_file = Path("/opt/trading_bot/logs/performance_metrics.json")
            if perf_file.exists():
                with open(perf_file, 'r') as f:
                    metrics = json.load(f)

                return f"""📊 <b>Performance Metrics</b>

<b>Portfolio Value:</b> €{metrics.get('portfolio_value', 0):,.2f}
<b>Daily P&L:</b> €{metrics.get('daily_pnl', 0):+,.2f}
<b>Total Return:</b> {metrics.get('total_return', 0):.2%}
<b>Sharpe Ratio:</b> {metrics.get('sharpe_ratio', 0):.2f}
<b>Win Rate:</b> {metrics.get('win_rate', 0):.1%}
<b>Active Positions:</b> {metrics.get('active_positions', 0)}

<i>Last updated: {metrics.get('timestamp', 'N/A')}</i>
"""
            else:
                return "❌ No performance data available"
        except Exception as e:
            return f"❌ Performance check failed: {str(e)}"

    async def _cmd_health(self, args: List[str]) -> str:
        """Get system health."""
        try:
            result = await asyncio.create_subprocess_shell(
                '/opt/trading_bot/health_check.sh',
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await result.communicate()

            if result.returncode == 0:
                return f"```bash\n{stdout.decode().strip()}\n```"
            else:
                return f"❌ Health check failed: {stderr.decode().strip()}"
        except Exception as e:
            return f"❌ Health check error: {str(e)}"

    async def _cmd_balance(self, args: List[str]) -> str:
        """Get current balance."""
        try:
            balance_file = Path("/opt/trading_bot/logs/balance.json")
            if balance_file.exists():
                with open(balance_file, 'r') as f:
                    balance_data = json.load(f)

                return f"""💰 <b>Current Balance</b>

<b>Cash Balance:</b> €{balance_data.get('cash_balance', 0):,.2f}
<b>Portfolio Value:</b> €{balance_data.get('portfolio_value', 0):,.2f}
<b>Total Equity:</b> €{balance_data.get('total_equity', 0):,.2f}

<b>Active Positions:</b>
{chr(10).join([f"• {symbol}: €{value:,.2f}" for symbol, value in balance_data.get('positions', {}).items()])}
"""
            else:
                return "❌ Balance data not available"
        except Exception as e:
            return f"❌ Balance check failed: {str(e)}"

    async def _cmd_recent_trades(self, args: List[str]) -> str:
        """Get recent trades."""
        try:
            trades_file = Path("/opt/trading_bot/logs/trades_report.csv")
            if trades_file.exists():
                # Get last 5 trades
                result = await asyncio.create_subprocess_shell(
                    'tail -5 /opt/trading_bot/logs/trades_report.csv',
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE
                )
                stdout, stderr = await result.communicate()

                if result.returncode == 0:
                    trades = stdout.decode().strip().split('\n')
                    if len(trades) > 1:  # Skip header
                        trade_list = []
                        for trade in trades[1:]:
                            parts = trade.split(',')
                            if len(parts) >= 6:
                                trade_list.append(f"• {parts[1]}: {parts[2]} {parts[3]} @ €{parts[4]}")

                        return f"""📈 <b>Recent Trades</b>

{chr(10).join(trade_list)}
"""
                    else:
                        return "📊 No recent trades found"
                else:
                    return "❌ Failed to read trades"
            else:
                return "❌ Trades log not found"
        except Exception as e:
            return f"❌ Recent trades check failed: {str(e)}"

    async def _cmd_logs(self, args: List[str]) -> str:
        """Get recent logs."""
        try:
            result = await asyncio.create_subprocess_shell(
                'tail -10 /var/log/trading_bot/trading_*.log 2>/dev/null || echo "No logs found"',
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await result.communicate()

            if result.returncode == 0:
                logs = stdout.decode().strip()
                if logs and logs != "No logs found":
                    return f"```bash\n{logs}\n```"
                else:
                    return "📝 No recent logs available"
            else:
                return "❌ Failed to read logs"
        except Exception as e:
            return f"❌ Logs check failed: {str(e)}"

    async def _cmd_config(self, args: List[str]) -> str:
        """Get configuration info."""
        try:
            config_file = Path("/opt/trading_bot/training_config.yaml")
            if config_file.exists():
                # Get basic config info without sensitive data
                result = await asyncio.create_subprocess_shell(
                    'grep -E "^(symbols|interval|initial_balance|max_position_size):" /opt/trading_bot/training_config.yaml',
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE
                )
                stdout, stderr = await result.communicate()

                if result.returncode == 0:
                    config_info = stdout.decode().strip()
                    return f"""⚙️ <b>Configuration</b>

```yaml
{config_info}
```
"""
                else:
                    return "❌ Failed to read configuration"
            else:
                return "❌ Configuration file not found"
        except Exception as e:
            return f"❌ Config check failed: {str(e)}"

    async def _get_system_uptime(self) -> str:
        """Get system uptime."""
        try:
            result = await asyncio.create_subprocess_shell(
                'uptime -p',
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await result.communicate()

            if result.returncode == 0:
                return stdout.decode().strip()
            else:
                return "Unknown"
        except Exception:
            return "Unknown"

    async def send_performance_report(self, metrics: Dict[str, Any]) -> bool:
        """Send comprehensive performance report."""
        try:
            message = f"""📊 <b>Performance Report</b>

<b>Portfolio Value:</b> €{metrics.get('portfolio_value', 0):,.2f}
<b>Daily P&L:</b> €{metrics.get('daily_pnl', 0):+,.2f}
<b>Total Return:</b> {metrics.get('total_return', 0):.2%}
<b>Sharpe Ratio:</b> {metrics.get('sharpe_ratio', 0):.2f}
<b>Win Rate:</b> {metrics.get('win_rate', 0):.1%}
<b>Active Positions:</b> {metrics.get('active_positions', 0)}

<b>System Resources:</b>
• CPU: {metrics.get('cpu_usage', 0):.1f}%
• Memory: {metrics.get('memory_usage', 0):.1f}%
• Disk: {metrics.get('disk_usage', 0):.1f}%

<i>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</i>
"""
            await self.bot.send_message(chat_id=self.chat_id, text=message, parse_mode='HTML')
            return True
        except Exception as e:
            logger.error(f"Failed to send performance report: {e}")
            return False

    async def send_system_health_alert(self, health_data: Dict[str, Any]) -> bool:
        """Send system health alert."""
        try:
            status_emoji = "✅" if health_data.get('overall_status') == 'healthy' else "❌"

            message = f"""{status_emoji} <b>System Health Alert</b>

<b>Status:</b> {health_data.get('overall_status', 'unknown').upper()}
<b>Server:</b> {health_data.get('server', 'unknown')}

<b>Components:</b>
"""
            for component, status in health_data.get('components', {}).items():
                comp_status = "✅" if status.get('status') == 'healthy' else "❌"
                message += f"• {component.title()}: {comp_status} {status.get('status', 'unknown')}\n"

            if health_data.get('warnings'):
                message += f"\n<b>Warnings:</b>\n"
                for warning in health_data['warnings'][:3]:  # Limit to 3 warnings
                    message += f"• {warning}\n"

            message += f"\n<i>Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</i>"

            await self.bot.send_message(chat_id=self.chat_id, text=message, parse_mode='HTML')
            return True
        except Exception as e:
            logger.error(f"Failed to send health alert: {e}")
            return False

    async def send_message(self, message: str) -> bool:
        """Send a simple message."""
        try:
            await self.bot.send_message(chat_id=self.chat_id, text=message, parse_mode='HTML')
            return True
        except Exception as e:
            logger.error(f"Failed to send message: {e}")
            return False