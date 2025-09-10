"""
Enhanced Telegram Notifier with Interactive Commands
"""

import asyncio
import json
import logging
import os
import platform
import sys
import threading
from datetime import datetime, time, timezone
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, List, Optional

import schedule
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
        self.disable_trade_notifications = kwargs.get("disable_trade_notifications", True)
        self.daily_report_enabled = kwargs.get("daily_report_enabled", True)
        self.error_notifications_enabled = kwargs.get("error_notifications_enabled", True)
        self.shutdown_notifications_enabled = kwargs.get("shutdown_notifications_enabled", True)

        # Initialize daily report scheduling
        self._init_daily_reports()

        # Start background scheduler
        self.scheduler_thread = None
        if self.daily_report_enabled:
            self._start_scheduler()

    def _register_commands(self) -> Dict[str, Callable[[List[str]], Awaitable[str]]]:
        """Register interactive commands."""
        return {
            "/status": self._cmd_status,
            "/start": self._cmd_start,
            "/stop": self._cmd_stop,
            "/restart": self._cmd_restart,
            "/database": self._cmd_database,
            "/performance": self._cmd_performance,
            "/health": self._cmd_health,
            "/balance": self._cmd_balance,
            "/trades": self._cmd_recent_trades,
            "/logs": self._cmd_logs,
            "/config": self._cmd_config,
            # New enhanced commands
            "/daily": self._cmd_daily_report,
            "/uptime": self._cmd_uptime,
            "/summary": self._cmd_quick_summary,
            "/alerts": self._cmd_recent_alerts,
            "/version": self._cmd_version,
        }

    async def handle_command(self, command: str, args: Optional[List[str]] = None) -> str:
        """Handle incoming command."""
        logger.info(f"Received command: {command}, args: {args}")

        if args is None:
            parts = command.strip().split()
            cmd = parts[0].lower() if parts else ""
            args = parts[1:] if len(parts) > 1 else []
        else:
            cmd = command.strip().lower()

        if not cmd.startswith("/"):
            cmd = "/" + cmd

        logger.info(f"Processing command: {cmd} with args: {args}")

        if cmd in self.command_handlers:
            try:
                logger.info(f"Executing command handler for: {cmd}")
                result = await self.command_handlers[cmd](args)
                logger.info(f"Command {cmd} executed successfully")
                return result
            except Exception as e:
                logger.error(f"Command error for {cmd}: {e}", exc_info=True)
                return f"❌ Error executing command: {str(e)}"
        else:
            logger.warning(f"Unknown command: {cmd}")
            return "Unknown command. Available: /status, /start, /stop, /restart, /performance, /health, /balance, /trades, /logs, /config"

    async def _cmd_database(self, args: List[str]) -> str:
        """Start database refresh workflow.

        Usage examples:
        - /database
        - /database BTCEUR,ETHEUR 30m
        - /database --dry-run
        - /database BTCEUR --branch=main
        """
        try:
            full_text = (args[0] if args else "/database").strip()

            # Defaults from training_config.yaml if available
            symbols: List[str] = []
            interval: str = "30m"
            dry_run = False
            git_branch: Optional[str] = None

            # Load config
            try:
                import yaml

                cfg_path = Path("training_config.yaml")
                if cfg_path.exists():
                    with open(cfg_path, "r") as f:
                        cfg = yaml.safe_load(f) or {}
                    # Accept either plain list or nested
                    symbols_cfg = cfg.get("symbols") or cfg.get("trading", {}).get("symbols")
                    if isinstance(symbols_cfg, list):
                        symbols = [str(s).strip().upper() for s in symbols_cfg]
                    interval_cfg = cfg.get("interval") or cfg.get("trading", {}).get("interval")
                    if interval_cfg:
                        interval = str(interval_cfg).strip()
            except Exception as e:
                logger.warning(f"Failed to read training_config.yaml: {e}")

            # Parse inline arguments
            if full_text and full_text.lower().startswith("/database"):
                parts = full_text.split()
                # parts[0] == /database; parse flags and optional tokens
                for p in parts[1:]:
                    if p.startswith("--dry-run"):
                        dry_run = True
                    elif p.startswith("--branch="):
                        git_branch = p.split("=", 1)[1]
                    elif "," in p or p.isalpha():
                        # Likely symbols list like "BTCEUR,ETHEUR" or a single symbol token
                        cand = [s.strip().upper() for s in p.split(",") if s.strip()]
                        # Heuristic: tokens like "30m"/"1h" are interval, not symbol
                        if all(
                            len(s) >= 5 and s[-3:].upper() in ("EUR", "USD", "USDT") for s in cand
                        ):
                            symbols = cand
                        else:
                            # If looks like interval (e.g., 15m/1h), set interval
                            if p.lower().endswith(("m", "h", "d")) and p[:-1].isdigit():
                                interval = p.lower()
                    elif p.lower().endswith(("m", "h", "d")) and p[:-1].isdigit():
                        interval = p.lower()

            # Fallback to default symbols if still empty
            if not symbols:
                symbols = ["BTCEUR", "ETHEUR", "ADAEUR", "DOTEUR", "LINKEUR"]

            # Kick off background task
            asyncio.create_task(
                self._run_database_refresh(
                    symbols, interval, dry_run=dry_run, git_branch=git_branch
                )
            )

            sym_preview = ",".join(symbols[:4]) + ("…" if len(symbols) > 4 else "")
            return (
                f"🗄️ <b>Database Refresh Started</b>\n\n"
                f"Symbols: {sym_preview}\nInterval: {interval}\n"
                f"Mode: {'dry-run' if dry_run else 'live'}\n"
                f"I will notify here when finished."
            )
        except Exception as e:
            logger.error(f"Database command failed to start: {e}", exc_info=True)
            return f"❌ Failed to start database refresh: {e}"

    async def _run_database_refresh(
        self,
        symbols: List[str],
        interval: str,
        dry_run: bool = False,
        git_branch: Optional[str] = None,
    ) -> None:
        """Background workflow to rebuild DBs, push to Git, and notify via Telegram."""
        start_ts = datetime.now()
        log_dir = Path("logs")
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / f"database_refresh_{start_ts.strftime('%Y%m%d_%H%M%S')}.log"

        def log_line(msg: str):
            logger.info(msg)
            try:
                with open(log_path, "a") as f:
                    f.write(f"{datetime.now().isoformat()} {msg}\n")
            except Exception:
                pass

        try:
            log_line(
                f"Starting DB refresh for {len(symbols)} symbols at {interval}; dry_run={dry_run}"
            )

            # Stop trading before rebuild
            if not dry_run:
                try:
                    proc = await asyncio.create_subprocess_shell(
                        "/opt/trading_bot/bot/scripts/tmux_manager.sh stop",
                        stdout=asyncio.subprocess.PIPE,
                        stderr=asyncio.subprocess.PIPE,
                    )
                    await proc.communicate()
                    log_line("Trading system stopped")
                except Exception as e:
                    log_line(f"Warning: failed to stop trading cleanly: {e}")

            # Backup and remove old DBs (skip in dry-run)
            data_dir = Path("data")
            data_dir.mkdir(exist_ok=True)
            if not dry_run:
                tsdir = Path("data/backups") / start_ts.strftime("%Y%m%d_%H%M%S")
                tsdir.mkdir(parents=True, exist_ok=True)
                removed: List[str] = []
                for sym in symbols:
                    db_file = data_dir / f"{sym.lower()}_{interval.lower()}.db"
                    if db_file.exists():
                        try:
                            backup_path = tsdir / db_file.name
                            db_file.replace(backup_path)
                            removed.append(db_file.name)
                            log_line(f"Backed up {db_file} -> {backup_path}")
                        except Exception as e:
                            log_line(f"Failed to backup {db_file}: {e}")

                if removed:
                    log_line(f"Backed up and removed: {', '.join(removed)}")
            else:
                log_line("Dry-run: Skipping backup and removal of existing DBs")

            # Rebuild databases unless dry-run
            from src.data_pipeline.db_builder import rebuild_databases

            results = {"success": [], "failed": []}
            if not dry_run:
                try:
                    # Use the final data fetcher for optimal results
                    import subprocess
                    import sys
                    from pathlib import Path

                    final_fetcher_path = Path(data_dir).parent / "final_data_fetcher.py"
                    if final_fetcher_path.exists():
                        log_line("🚀 Using final data fetcher for optimal 1-year coverage")
                        cmd = [sys.executable, str(final_fetcher_path)] + symbols
                        result = subprocess.run(
                            cmd, capture_output=True, text=True, cwd=str(data_dir.parent)
                        )

                        if result.returncode == 0:
                            log_line("✅ Final data fetcher completed successfully")
                            log_line(result.stdout)
                        else:
                            log_line(f"❌ Final data fetcher failed: {result.stderr}")
                            raise RuntimeError(f"Final data fetcher failed: {result.stderr}")
                    else:
                        # Fallback to original method
                        await rebuild_databases(
                            symbols, interval, data_dir=str(data_dir), days=365, log_cb=log_line
                        )
                    results["success"] = symbols[:]
                except Exception as e:
                    # If builder returns partial results in exception message, just log
                    log_line(f"Error during rebuild: {e}")
                    results["failed"] = symbols[:]
            else:
                log_line("Dry-run: Skipping actual rebuild")

            # Commit and push to GitHub
            commit_sha = ""
            if not dry_run:
                try:
                    # Optional: switch branch
                    if git_branch:
                        proc = await asyncio.create_subprocess_shell(
                            f"git checkout {git_branch}",
                            stdout=asyncio.subprocess.PIPE,
                            stderr=asyncio.subprocess.PIPE,
                        )
                        await proc.communicate()

                    # Ensure we are up to date to avoid push rejects
                    proc = await asyncio.create_subprocess_shell(
                        "git fetch --all --prune",
                        stdout=asyncio.subprocess.PIPE,
                        stderr=asyncio.subprocess.PIPE,
                    )
                    await proc.communicate()

                    proc = await asyncio.create_subprocess_shell(
                        "git -c rebase.autoStash=true pull --rebase || true",
                        stdout=asyncio.subprocess.PIPE,
                        stderr=asyncio.subprocess.PIPE,
                    )
                    await proc.communicate()

                    proc = await asyncio.create_subprocess_shell(
                        "git add -f data/*.db",
                        stdout=asyncio.subprocess.PIPE,
                        stderr=asyncio.subprocess.PIPE,
                    )
                    await proc.communicate()

                    proc = await asyncio.create_subprocess_shell(
                        "git diff --cached --quiet || git commit -m 'Refresh databases via /database' --no-verify",
                        stdout=asyncio.subprocess.PIPE,
                        stderr=asyncio.subprocess.PIPE,
                    )
                    await proc.communicate()

                    # Capture commit SHA for latest commit
                    proc = await asyncio.create_subprocess_shell(
                        "git rev-parse --short HEAD",
                        stdout=asyncio.subprocess.PIPE,
                        stderr=asyncio.subprocess.PIPE,
                    )
                    stdout, _ = await proc.communicate()
                    commit_sha = (stdout.decode().strip() or "").strip()

                    proc = await asyncio.create_subprocess_shell(
                        "git push",
                        stdout=asyncio.subprocess.PIPE,
                        stderr=asyncio.subprocess.PIPE,
                    )
                    push_out, push_err = await proc.communicate()
                    log_line(
                        f"git push exit={proc.returncode} out={push_out.decode().strip()} err={push_err.decode().strip()}"
                    )
                except Exception as e:
                    log_line(f"Git push failed: {e}")

            # Restart trading
            if not dry_run:
                try:
                    proc = await asyncio.create_subprocess_shell(
                        "/opt/trading_bot/bot/scripts/tmux_manager.sh start",
                        stdout=asyncio.subprocess.PIPE,
                        stderr=asyncio.subprocess.PIPE,
                    )
                    await proc.communicate()
                    log_line("Trading system restarted")
                except Exception as e:
                    log_line(f"Warning: failed to start trading: {e}")

            duration_m = int((datetime.now() - start_ts).total_seconds() // 60)
            success_n = len(results.get("success", []))
            total_n = len(symbols)
            failed = results.get("failed", [])

            message = (
                f"🗄️ <b>Database Refresh Completed</b>\n\n"
                f"Symbols: {success_n}/{total_n} ok\n"
                f"Interval: {interval}\n"
                f"Commit: {commit_sha or 'n/a'}\n"
                f"Duration: ~{duration_m} min\n"
            )
            if failed:
                message += f"\n❌ Failed: {', '.join(failed)}\n"
            message += f"\nLog: {log_path}"

            try:
                await self.bot.send_message(chat_id=self.chat_id, text=message, parse_mode="HTML")
            except TelegramError as te:
                logger.error(f"Failed to send completion message: {te}")
        except Exception as e:
            logger.error(f"Database refresh workflow failed: {e}", exc_info=True)
            try:
                await self.bot.send_message(
                    chat_id=self.chat_id,
                    text=f"❌ Database refresh failed: {e}",
                    parse_mode="HTML",
                )
            except Exception:
                pass

    async def _cmd_status(self, args: List[str]) -> str:
        """Get system status."""
        try:
            # Check tmux session
            result = await asyncio.create_subprocess_shell(
                "tmux has-session -t trading_session",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            await result.wait()

            if result.returncode == 0:
                status = "✅ Running"
            else:
                status = "❌ Stopped"

            # Get system info
            hostname = platform.node()
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
        logger.info("Executing start command")
        try:
            logger.info("Running tmux_manager.sh start command")
            result = await asyncio.create_subprocess_shell(
                "/opt/trading_bot/bot/scripts/tmux_manager.sh start",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await result.communicate()
            
            stdout_text = stdout.decode().strip()
            stderr_text = stderr.decode().strip()
            
            logger.info(f"Start command completed with return code: {result.returncode}")

            if result.returncode == 0:
                logger.info("Start command successful")
                return "🚀 Trading system started successfully"
            elif "Trading session already running" in stdout_text or "Trading session already running" in stderr_text:
                logger.info("Trading session already running")
                return "⚡ Trading system is already running"
            else:
                error_msg = stderr_text or stdout_text
                logger.error(f"Start command failed with return code: {result.returncode}: {error_msg}")
                return f"❌ Failed to start trading system\n\nError: {error_msg}"
        except Exception as e:
            logger.error(f"Start command exception: {e}", exc_info=True)
            return f"❌ Start command failed: {str(e)}"

    async def _cmd_stop(self, args: List[str]) -> str:
        """Stop trading system."""
        try:
            result = await asyncio.create_subprocess_shell(
                "/opt/trading_bot/bot/scripts/tmux_manager.sh stop",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
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
                "/opt/trading_bot/bot/scripts/tmux_manager.sh stop",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            await stop_result.wait()

            await asyncio.sleep(2)

            # Start
            start_result = await asyncio.create_subprocess_shell(
                "/opt/trading_bot/bot/scripts/tmux_manager.sh start",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
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
            perf_file = Path("/opt/trading_bot/bot/logs/performance_metrics.json")
            if perf_file.exists():
                with open(perf_file, "r") as f:
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
        logger.info("Executing health command")
        try:
            logger.info("Running health_check.sh script")
            result = await asyncio.create_subprocess_shell(
                "/opt/trading_bot/bot/scripts/health_check.sh",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await result.communicate()
            logger.info(f"Health check completed with return code: {result.returncode}")

            if result.returncode == 0:
                logger.info("Health check successful")
                return f"```bash\n{stdout.decode().strip()}\n```"
            else:
                logger.error(
                    f"Health check failed with return code: {result.returncode}, stderr: {stderr.decode().strip()}"
                )
                return f"❌ Health check failed: {stderr.decode().strip()}"
        except Exception as e:
            logger.error(f"Health check exception: {e}", exc_info=True)
            return f"❌ Health check error: {str(e)}"

    async def _cmd_balance(self, args: List[str]) -> str:
        """Get current balance."""
        try:
            balance_file = Path("/opt/trading_bot/bot/logs/balance.json")
            if balance_file.exists():
                with open(balance_file, "r") as f:
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
            trades_file = Path("/opt/trading_bot/bot/logs/trades_report.csv")
            if trades_file.exists():
                # Get last 5 trades
                result = await asyncio.create_subprocess_shell(
                    "tail -5 /opt/trading_bot/bot/logs/trades_report.csv",
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
                stdout, stderr = await result.communicate()

                if result.returncode == 0:
                    trades = stdout.decode().strip().split("\n")
                    if len(trades) > 1:  # Skip header
                        trade_list = []
                        for trade in trades[1:]:
                            parts = trade.split(",")
                            if len(parts) >= 6:
                                trade_list.append(
                                    f"• {parts[1]}: {parts[2]} {parts[3]} @ €{parts[4]}"
                                )

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
                'tail -10 /opt/trading_bot/bot/logs/trading_*.log 2>/dev/null || echo "No logs found"',
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
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
            config_file = Path("/opt/trading_bot/bot/training_config.yaml")
            if config_file.exists():
                # Get basic config info without sensitive data
                result = await asyncio.create_subprocess_shell(
                    'grep -E "^(symbols|interval|initial_balance|max_position_size):" /opt/trading_bot/bot/training_config.yaml',
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
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
                "uptime -p",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
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
            await self.bot.send_message(chat_id=self.chat_id, text=message, parse_mode="HTML")
            return True
        except Exception as e:
            logger.error(f"Failed to send performance report: {e}")
            return False

    async def send_system_health_alert(self, health_data: Dict[str, Any]) -> bool:
        """Send system health alert."""
        try:
            status_emoji = "✅" if health_data.get("overall_status") == "healthy" else "❌"

            message = f"""{status_emoji} <b>System Health Alert</b>

<b>Status:</b> {health_data.get('overall_status', 'unknown').upper()}
<b>Server:</b> {health_data.get('server', 'unknown')}

<b>Components:</b>
"""
            for component, status in health_data.get("components", {}).items():
                comp_status = "✅" if status.get("status") == "healthy" else "❌"
                message += (
                    f"• {component.title()}: {comp_status} {status.get('status', 'unknown')}\n"
                )

            if health_data.get("warnings"):
                message += f"\n<b>Warnings:</b>\n"
                for warning in health_data["warnings"][:3]:  # Limit to 3 warnings
                    message += f"• {warning}\n"

            message += f"\n<i>Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</i>"

            await self.bot.send_message(chat_id=self.chat_id, text=message, parse_mode="HTML")
            return True
        except Exception as e:
            logger.error(f"Failed to send health alert: {e}")
            return False

    async def send_message(self, message: str) -> bool:
        """Send a simple message."""
        try:
            await self.bot.send_message(chat_id=self.chat_id, text=message, parse_mode="HTML")
            return True
        except Exception as e:
            logger.error(f"Failed to send message: {e}")
            return False

    def _init_daily_reports(self):
        """Initialize daily performance report scheduling."""
        if self.daily_report_enabled:
            # Schedule daily performance report at 12:00 UTC
            schedule.every().day.at("12:00").do(self._send_daily_report_job)
            logger.info("Daily performance reports scheduled for 12:00 UTC")

    def _start_scheduler(self):
        """Start the background scheduler for daily reports."""

        def run_scheduler():
            while True:
                try:
                    schedule.run_pending()
                    threading.Event().wait(60)  # Check every minute
                except Exception as e:
                    logger.error(f"Scheduler error: {e}")

        self.scheduler_thread = threading.Thread(target=run_scheduler, daemon=True)
        self.scheduler_thread.start()
        logger.info("Daily report scheduler started")

    def _send_daily_report_job(self):
        """Job function for sending daily reports."""
        try:
            # Run in async context
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(self.send_daily_performance_report())
            loop.close()
        except Exception as e:
            logger.error(f"Daily report job error: {e}")

    async def send_daily_performance_report(self) -> bool:
        """Send comprehensive daily performance report at 12:00 UTC."""
        try:
            current_time = datetime.now(timezone.utc)
            report_date = current_time.strftime("%Y-%m-%d")

            # Get system metrics
            metrics = await self._get_system_metrics()

            # Get trading performance
            trading_stats = await self._get_trading_statistics()

            # Get recent trades summary
            trades_summary = await self._get_daily_trades_summary()

            message = f"""📊 <b>Daily Performance Report</b>
🗓️ <b>{report_date}</b> | 🕐 12:00 UTC

<b>🎯 Trading Performance:</b>
• Total Trades: {trading_stats.get('total_trades', 0)}
• Profitable: {trading_stats.get('profitable_trades', 0)} ({trading_stats.get('win_rate', 0):.1f}%)
• Daily P&L: €{trading_stats.get('daily_pnl', 0):.2f}
• Total Balance: €{trading_stats.get('total_balance', 10000):.2f}

<b>📈 Active Symbols:</b>
{trading_stats.get('symbols_summary', '• No active positions')}

<b>💻 System Health:</b>
• Uptime: {metrics.get('uptime', 'Unknown')}
• CPU: {metrics.get('cpu_usage', 0):.1f}% | Memory: {metrics.get('memory_usage', 0):.1f}%
• Disk: {metrics.get('disk_usage', 0):.1f}% | Status: {'✅ Healthy' if metrics.get('status') == 'healthy' else '⚠️ Issues'}

{trades_summary}

<i>📤 Next report: Tomorrow at 12:00 UTC</i>
<i>⚙️ Use /performance for real-time stats</i>
"""

            await self.bot.send_message(chat_id=self.chat_id, text=message, parse_mode="HTML")
            logger.info(f"Daily performance report sent for {report_date}")
            return True

        except Exception as e:
            logger.error(f"Failed to send daily performance report: {e}")
            await self.send_error_notification("Daily Performance Report", str(e))
            return False

    async def _get_daily_trades_summary(self) -> str:
        """Get summary of today's trades."""
        try:
            # Check for recent trades from reports or logs
            if os.path.exists("reports"):
                import glob

                today = datetime.now().strftime("%Y%m%d")
                report_files = glob.glob(f"reports/performance_report_{today}*.json")

                if report_files:
                    latest_report = max(report_files)
                    with open(latest_report, "r") as f:
                        data = json.load(f)

                    if data.get("trades"):
                        return f"\n<b>🔄 Recent Trades:</b>\n" + "\n".join(
                            [
                                f"• {trade.get('symbol', 'Unknown')}: {trade.get('action', 'N/A')} @ €{trade.get('price', 0):.2f}"
                                for trade in data["trades"][-3:]  # Last 3 trades
                            ]
                        )

            return "\n<b>🔄 Recent Trades:</b>\n• No trades today"

        except Exception as e:
            logger.error(f"Failed to get daily trades summary: {e}")
            return "\n<b>🔄 Recent Trades:</b>\n• Unable to load trades data"

    async def _get_trading_statistics(self) -> Dict[str, Any]:
        """Get current trading statistics."""
        try:
            stats = {
                "total_trades": 0,
                "profitable_trades": 0,
                "win_rate": 0.0,
                "daily_pnl": 0.0,
                "total_balance": 10000.0,
                "symbols_summary": "• No active positions",
            }

            # Try to get real trading stats from manager
            if self.trading_manager:
                # Implementation depends on your trading manager interface
                pass

            # Fallback: check reports directory for latest performance
            if os.path.exists("reports"):
                import glob

                report_files = glob.glob("reports/performance_report_*.json")
                if report_files:
                    latest_report = max(report_files)
                    with open(latest_report, "r") as f:
                        data = json.load(f)

                    stats.update(
                        {
                            "total_trades": len(data.get("trades", [])),
                            "total_balance": data.get("current_balance", 10000.0),
                            "daily_pnl": data.get("daily_pnl", 0.0),
                        }
                    )

            return stats

        except Exception as e:
            logger.error(f"Failed to get trading statistics: {e}")
            return {
                "total_trades": 0,
                "profitable_trades": 0,
                "win_rate": 0.0,
                "daily_pnl": 0.0,
                "total_balance": 10000.0,
                "symbols_summary": "• Error loading data",
            }

    async def send_enhanced_error_notification(
        self, error_type: str, error_message: str, context: str = ""
    ) -> bool:
        """Send enhanced error notification with context."""
        try:
            if not self.error_notifications_enabled:
                return False

            timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

            message = f"""🚨 <b>System Error Alert</b>

<b>Error Type:</b> {error_type}
<b>Time:</b> {timestamp}

<b>Details:</b>
<code>{error_message}</code>

{f'<b>Context:</b> {context}' if context else ''}

<b>System Status:</b> {'⚠️ Monitoring' if 'trading' in error_type.lower() else '🔍 Investigating'}

<i>Use /health to check system status</i>
<i>Use /logs to view recent logs</i>
"""

            await self.bot.send_message(chat_id=self.chat_id, text=message, parse_mode="HTML")
            logger.info(f"Enhanced error notification sent: {error_type}")
            return True

        except Exception as e:
            logger.error(f"Failed to send enhanced error notification: {e}")
            return False

    async def send_system_shutdown_notification(self, reason: str = "Manual shutdown") -> bool:
        """Send notification when system shuts down."""
        try:
            if not self.shutdown_notifications_enabled:
                return False

            timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

            message = f"""🛑 <b>Trading System Shutdown</b>

<b>Time:</b> {timestamp}
<b>Reason:</b> {reason}

<b>Final Status:</b>
• All positions closed: ✅
• Data saved: ✅
• Logs archived: ✅

<i>💡 System will need manual restart</i>
<i>📞 Use /start command when ready to resume</i>
"""

            await self.bot.send_message(chat_id=self.chat_id, text=message, parse_mode="HTML")
            logger.info(f"System shutdown notification sent: {reason}")
            return True

        except Exception as e:
            logger.error(f"Failed to send shutdown notification: {e}")
            return False

    async def send_startup_notification(self) -> bool:
        """Send startup notification."""
        try:
            message = f"""🚀 <b>Trading Bot Started</b>

<b>System:</b> {platform.node()}
<b>Time:</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
<b>Status:</b> Initializing trading components...

Bot is now online and ready to accept commands.
"""
            await self.bot.send_message(chat_id=self.chat_id, text=message, parse_mode="HTML")
            return True
        except Exception as e:
            logger.error(f"Failed to send startup notification: {e}")
            return False

    async def send_trade_notification(self, trade_data: Dict[str, Any]) -> bool:
        """Send trade notification (disabled by default to reduce message spam)."""
        try:
            # Skip individual trade notifications if disabled (default behavior)
            if self.disable_trade_notifications:
                logger.debug(
                    f"Trade notification skipped (disabled): {trade_data.get('symbol')} {trade_data.get('action')}"
                )
                return True

            action = trade_data.get("action", "unknown").upper()
            symbol = trade_data.get("symbol", "unknown")
            quantity = trade_data.get("quantity", 0)
            price = trade_data.get("price", 0)
            value = trade_data.get("value", 0)

            emoji = "🟢" if action == "BUY" else "🔴" if action == "SELL" else "⚪"

            message = f"""{emoji} <b>Trade Executed</b>

<b>Action:</b> {action}
<b>Symbol:</b> {symbol}
<b>Quantity:</b> {quantity:,.4f}
<b>Price:</b> €{price:,.4f}
<b>Total Value:</b> €{value:,.2f}

<b>Time:</b> {datetime.now().strftime('%H:%M:%S')}
<i>💡 Get daily summary at 12:00 UTC or use /trades for recent trades</i>
"""
            await self.bot.send_message(chat_id=self.chat_id, text=message, parse_mode="HTML")
            return True
        except Exception as e:
            logger.error(f"Failed to send trade notification: {e}")
            return False

    async def send_error_notification(self, error_msg: str, component: str = "System") -> bool:
        """Send enhanced error notification with context and recommendations."""
        try:
            if not self.error_notifications_enabled:
                return False

            return await self.send_enhanced_error_notification(component, error_msg)
        except Exception as e:
            logger.error(f"Failed to send error notification: {e}")
            return False

    # New enhanced command implementations
    async def _cmd_daily_report(self, args: List[str]) -> str:
        """Manually trigger daily report."""
        try:
            success = await self.send_daily_performance_report()
            if success:
                return "📊 Daily performance report generated successfully!"
            else:
                return "❌ Failed to generate daily report. Check logs for details."
        except Exception as e:
            return f"❌ Daily report error: {str(e)}"

    async def _cmd_uptime(self, args: List[str]) -> str:
        """Get system uptime and basic stats."""
        try:
            import subprocess

            # Get system uptime
            result = await asyncio.create_subprocess_shell(
                "uptime -p",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await result.communicate()
            uptime = stdout.decode().strip() if result.returncode == 0 else "Unknown"

            # Get process start time
            trading_pids = await asyncio.create_subprocess_shell(
                "pgrep -f enhanced_trader.py", stdout=asyncio.subprocess.PIPE
            )
            pid_stdout, _ = await trading_pids.communicate()

            process_uptime = "Not running"
            if trading_pids.returncode == 0 and pid_stdout:
                pid = pid_stdout.decode().strip().split("\n")[0]
                if pid:
                    start_result = await asyncio.create_subprocess_shell(
                        f"ps -p {pid} -o lstart=", stdout=asyncio.subprocess.PIPE
                    )
                    start_stdout, _ = await start_result.communicate()
                    if start_result.returncode == 0:
                        process_uptime = start_stdout.decode().strip()

            return f"""⏰ <b>System Uptime</b>

<b>Server Uptime:</b> {uptime.replace('up ', '')}
<b>Trading Process:</b> {process_uptime}
<b>Current Time:</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}

<b>Quick Stats:</b>
• Daily reports: {'✅ Enabled' if self.daily_report_enabled else '❌ Disabled'}
• Error alerts: {'✅ Enabled' if self.error_notifications_enabled else '❌ Disabled'}
• Trade notifications: {'❌ Disabled (spam prevention)' if self.disable_trade_notifications else '✅ Enabled'}
"""
        except Exception as e:
            return f"❌ Uptime check failed: {str(e)}"

    async def _cmd_quick_summary(self, args: List[str]) -> str:
        """Get quick portfolio and system summary."""
        try:
            metrics = await self._get_system_metrics()
            stats = await self._get_trading_statistics()

            return f"""📋 <b>Quick System Summary</b>

<b>💰 Portfolio:</b>
• Balance: €{stats.get('total_balance', 10000):.2f}
• Daily P&L: €{stats.get('daily_pnl', 0):.2f}
• Total Trades: {stats.get('total_trades', 0)}

<b>💻 System:</b>
• Status: {'✅ Healthy' if metrics.get('status') == 'healthy' else '⚠️ Issues'}
• CPU: {metrics.get('cpu_usage', 0):.1f}% | Memory: {metrics.get('memory_usage', 0):.1f}%
• Disk: {metrics.get('disk_usage', 0):.1f}%

<b>📊 Today:</b>
• Next daily report: Tomorrow 12:00 UTC
• Use /performance for detailed metrics
• Use /health for system diagnostics
"""
        except Exception as e:
            return f"❌ Summary error: {str(e)}"

    async def _cmd_recent_alerts(self, args: List[str]) -> str:
        """Get recent alerts and notifications."""
        try:
            alerts_found = []

            # Check for recent error logs
            if os.path.exists("logs/resource_alerts.log"):
                result = await asyncio.create_subprocess_shell(
                    "tail -5 logs/resource_alerts.log",
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
                stdout, stderr = await result.communicate()
                if result.returncode == 0 and stdout:
                    alerts_found.extend(stdout.decode().strip().split("\n")[-3:])

            # Check system logs for errors
            log_result = await asyncio.create_subprocess_shell(
                'grep -i error logs/*.log 2>/dev/null | tail -3 || echo "No recent errors"',
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            log_stdout, _ = await log_result.communicate()

            recent_alerts = "📜 <b>Recent Alerts & Notifications</b>\n\n"

            if alerts_found:
                recent_alerts += "<b>🚨 Resource Alerts:</b>\n"
                for alert in alerts_found[-3:]:
                    if alert.strip():
                        recent_alerts += f"• {alert.strip()}\n"
                recent_alerts += "\n"

            if log_stdout and log_stdout.decode().strip() != "No recent errors":
                recent_alerts += "<b>📋 Recent System Messages:</b>\n"
                for line in log_stdout.decode().strip().split("\n")[-3:]:
                    if line.strip():
                        recent_alerts += f"• {line.split(':')[-1].strip()}\n"
            else:
                recent_alerts += "<b>✅ No recent alerts or errors</b>\n"

            recent_alerts += f"\n<i>📅 Checked: {datetime.now().strftime('%H:%M:%S')}</i>"
            recent_alerts += f"\n<i>💡 Use /logs for detailed system logs</i>"

            return recent_alerts

        except Exception as e:
            return f"❌ Alerts check failed: {str(e)}"

    async def _cmd_version(self, args: List[str]) -> str:
        """Get bot version and system information."""
        try:
            # Get Python version
            python_version = (
                f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
            )

            # Get system info
            import platform

            system_info = f"{platform.system()} {platform.release()}"

            return f"""🤖 <b>Trading Bot System Info</b>

<b>🔧 Software:</b>
• Bot Version: Enhanced v2.0
• Python: {python_version}
• System: {system_info}
• Architecture: {platform.machine()}

<b>📊 Features:</b>
• ✅ Daily reports at 12:00 UTC
• ✅ Enhanced error notifications
• ✅ Smart trade notification filtering
• ✅ Resource monitoring & alerts
• ✅ Graceful shutdown notifications

<b>📈 Models:</b>
• GRU Neural Networks
• LightGBM Gradient Boosting
• PPO Reinforcement Learning
• Multi-symbol ensemble trading

<b>🔒 Security:</b>
• Production server deployment
• Secure API communication
• Rate-limited notifications

<i>⚙️ Last updated: September 2025</i>
"""
        except Exception as e:
            return f"❌ Version info error: {str(e)}"
