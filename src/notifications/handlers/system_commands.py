"""
System-related Telegram command handlers.
"""

import asyncio
import json
import logging
import platform
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import psutil
from telegram import Update
from telegram.ext import ContextTypes

from src.core.logging_manager import get_system_logger


class SystemCommandHandler:
    """Handler for system-related Telegram commands."""

    def __init__(self):
        self.logger = get_system_logger(__name__)
        self.start_time = datetime.now(timezone.utc)

    def register_commands(self, registry):
        """Register all system commands with the command registry."""
        registry.register_command(
            name="health",
            handler=self.handle_health,
            description="Show system health status",
            admin_only=False,
            rate_limit=10,
            aliases=["ping"],
        )

        registry.register_command(
            name="uptime",
            handler=self.handle_uptime,
            description="Show system uptime",
            admin_only=False,
            rate_limit=15,
        )

        registry.register_command(
            name="resources",
            handler=self.handle_resources,
            description="Show system resource usage",
            admin_only=False,
            rate_limit=10,
            aliases=["res", "usage"],
        )

        registry.register_command(
            name="logs",
            handler=self.handle_logs,
            description="Show recent system logs",
            admin_only=True,
            rate_limit=5,
        )

        registry.register_command(
            name="config",
            handler=self.handle_config,
            description="Show system configuration",
            admin_only=True,
            rate_limit=3,
        )

        registry.register_command(
            name="version",
            handler=self.handle_version,
            description="Show system version information",
            admin_only=False,
            rate_limit=20,
        )

        # Admin-only: rebuild SQLite databases and optionally push to GitHub
        registry.register_command(
            name="database",
            handler=self.handle_database,
            description="Rebuild 30m databases (1y) and push to GitHub",
            admin_only=True,
            rate_limit=1,
        )

        self.logger.info("System commands registered")

    async def handle_health(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /health command."""
        try:
            health_data = await self._get_system_health()

            # Overall health status
            overall_status = "🟢 HEALTHY" if health_data["overall_healthy"] else "🔴 UNHEALTHY"

            message = f"""
🏥 <b>System Health</b>
━━━━━━━━━━━━━━━━━
<b>Overall:</b> {overall_status}

<b>Services</b>
"""

            # Add service statuses
            for service, status in health_data["services"].items():
                status_emoji = "🟢" if status["healthy"] else "🔴"
                message += f"{status_emoji} {service}: {status['status']}\n"

            # Add system metrics
            message += f"""
<b>Metrics</b>
💾 Memory: {health_data['memory_percent']:.1f}%
🔄 CPU: {health_data['cpu_percent']:.1f}%
💿 Disk: {health_data['disk_percent']:.1f}%
📊 Load: {health_data['load_average']:.2f}

<b>Network</b>
🌐 Internet: {'Connected' if health_data['internet_connected'] else 'Disconnected'}
📡 API: {'Operational' if health_data['api_healthy'] else 'Issues Detected'}
"""

            # Add warnings if any
            if health_data["warnings"]:
                message += "\n⚠️ <b>Warnings:</b>\n"
                for warning in health_data["warnings"]:
                    message += f"• {warning}\n"

            message += f"\n⏰ <i>Checked: {datetime.now(timezone.utc).strftime('%H:%M UTC')}</i>"

            await update.message.reply_text(message, parse_mode="HTML")

        except Exception as e:
            self.logger.error(f"Error handling health command: {e}")
            await update.message.reply_text("❌ Error retrieving health information")

    async def handle_uptime(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /uptime command."""
        try:
            current_time = datetime.now(timezone.utc)
            uptime_delta = current_time - self.start_time

            # Format uptime
            days = uptime_delta.days
            hours, remainder = divmod(uptime_delta.seconds, 3600)
            minutes, seconds = divmod(remainder, 60)

            uptime_str = ""
            if days > 0:
                uptime_str += f"{days}d "
            if hours > 0:
                uptime_str += f"{hours}h "
            if minutes > 0:
                uptime_str += f"{minutes}m "
            uptime_str += f"{seconds}s"

            # Get system boot time
            boot_time = datetime.fromtimestamp(psutil.boot_time(), tz=timezone.utc)
            system_uptime = current_time - boot_time
            sys_days = system_uptime.days
            sys_hours, sys_remainder = divmod(system_uptime.seconds, 3600)
            sys_minutes, _ = divmod(sys_remainder, 60)

            system_uptime_str = ""
            if sys_days > 0:
                system_uptime_str += f"{sys_days}d "
            if sys_hours > 0:
                system_uptime_str += f"{sys_hours}h "
            if sys_minutes > 0:
                system_uptime_str += f"{sys_minutes}m"

            message = f"""
⏱️ <b>Uptime</b>
━━━━━━━━━━━━━━━━━
🤖 <b>Bot:</b> {uptime_str}
🖥️ <b>System:</b> {system_uptime_str}

📅 <b>Bot Started:</b> {self.start_time.strftime('%Y-%m-%d %H:%M:%S UTC')}
🔄 <b>System Boot:</b> {boot_time.strftime('%Y-%m-%d %H:%M:%S UTC')}

💻 <b>Platform:</b> {platform.system()} {platform.release()}
🐍 <b>Python:</b> {platform.python_version()}
"""

            await update.message.reply_text(message, parse_mode="HTML")

        except Exception as e:
            self.logger.error(f"Error handling uptime command: {e}")
            await update.message.reply_text("❌ Error retrieving uptime information")

    async def handle_database(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /database command.

        Usage: /database [interval] [days] [--no-push]
          - interval: e.g., 30m (default), 1h
          - days: number of days history (default 365)
          - --no-push: skip git push after rebuild
        """
        try:
            args = list(context.args or [])
            interval = "30m"
            days = 365
            do_push = True
            # Parse flags
            parsed = []
            for a in args:
                if a == "--no-push":
                    do_push = False
                else:
                    parsed.append(a)
            if parsed:
                interval = parsed[0]
            if len(parsed) > 1 and parsed[1].isdigit():
                days = int(parsed[1])

            await update.message.reply_text(
                f"🗄️ Rebuilding databases: interval={interval}, days={days}, push={'yes' if do_push else 'no'}"
            )

            # Discover symbols from models directory
            symbols = await self._discover_symbols()
            if not symbols:
                symbols = ["BTCEUR", "ETHEUR", "ADAEUR", "DOTEUR", "LINKEUR"]

            # Import builder lazily to avoid heavy imports during normal ops
            from src.data_pipeline.db_builder import rebuild_databases

            async def log_cb(msg: str):
                # Stream important messages selectively to Telegram to avoid spam
                if any(k in msg.lower() for k in ("fetching", "built", "completed")):
                    try:
                        await update.message.reply_text(msg)
                    except Exception:
                        pass

            # Rebuild
            results = await rebuild_databases(symbols, interval, data_dir="data", days=days)

            summary = "\n".join(
                [f"• {r.symbol} {r.interval}: {r.rows} rows -> {r.db_path.name}" for r in results]
            )
            await update.message.reply_text(
                f"✅ Rebuild complete ({len(results)} DBs)\n{summary}", parse_mode="HTML"
            )

            if do_push:
                pushed = await self._git_push_databases([str(r.db_path) for r in results])
                if pushed["success"]:
                    await update.message.reply_text(
                        f"📤 Git push completed: {pushed.get('commit', 'commit created')}"
                    )
                else:
                    await update.message.reply_text(
                        f"⚠️ Git push skipped/failed: {pushed.get('error', 'unknown')}"
                    )

        except Exception as e:
            self.logger.error(f"Error handling database command: {e}")
            await update.message.reply_text(f"❌ Database rebuild failed: {e}", parse_mode="HTML")

    async def _discover_symbols(self) -> list[str]:
        """Discover symbols from installed models directory."""
        try:
            base = Path("models")
            symbols = set()
            for mtype in ("gru", "lightgbm", "ppo"):
                p = base / mtype
                if p.exists():
                    for d in p.iterdir():
                        if d.is_dir() and len(d.name) >= 5:
                            symbols.add(d.name.upper())
            return sorted(symbols)
        except Exception:
            return []

    async def _git_push_databases(self, db_paths: list[str]) -> Dict[str, Any]:
        """Stage, commit, and push rebuilt DBs to the Git remote.

        Returns dict: {success: bool, commit?: str, error?: str}
        """
        import subprocess

        try:
            # Stage files explicitly to avoid shell globbing
            for p in db_paths:
                try:
                    subprocess.run(["git", "add", p], check=True)
                except subprocess.CalledProcessError:
                    # Try relative to repo root
                    rel = str(Path(p).as_posix())
                    subprocess.run(["git", "add", rel], check=False)

            msg = f"chore(data): update SQLite databases ({datetime.now(timezone.utc).isoformat()})"
            # Commit (allow empty false)
            commit = subprocess.run(["git", "commit", "-m", msg], capture_output=True, text=True)
            if commit.returncode != 0 and "nothing to commit" in commit.stderr.lower():
                return {"success": True, "commit": "no changes"}
            if commit.returncode != 0:
                return {"success": False, "error": commit.stderr.strip()}

            push = subprocess.run(["git", "push", "origin", "main"], capture_output=True, text=True)
            if push.returncode != 0:
                return {"success": False, "error": push.stderr.strip()}
            return {"success": True, "commit": msg}
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def handle_resources(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /resources command."""
        try:
            # Get system resource information
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage("/")
            load_avg = psutil.getloadavg()[0] if hasattr(psutil, "getloadavg") else 0.0

            # Network I/O
            net_io = psutil.net_io_counters()

            # Process information
            process = psutil.Process()
            process_memory = process.memory_info()

            message = f"""
🧰 <b>System Resources</b>
━━━━━━━━━━━━━━━━━
<b>CPU</b>
🔄 Usage: {cpu_percent:.1f}%
📈 Load: {load_avg:.2f}
🔥 Cores: {psutil.cpu_count()} ({psutil.cpu_count(logical=False)} phys)

<b>Memory</b>
💾 Used: {memory.used / 1024**3:.1f}GB / {memory.total / 1024**3:.1f}GB ({memory.percent:.1f}%)
🟢 Available: {memory.available / 1024**3:.1f}GB
📈 Bot RSS: {process_memory.rss / 1024**2:.1f}MB

<b>Disk</b>
💿 Used: {disk.used / 1024**3:.1f}GB / {disk.total / 1024**3:.1f}GB ({disk.used/disk.total*100:.1f}%)
📦 Free: {disk.free / 1024**3:.1f}GB

<b>Network</b>
📤 Sent: {net_io.bytes_sent / 1024**2:.1f}MB
📥 Recv: {net_io.bytes_recv / 1024**2:.1f}MB
"""

            # Add resource alerts if needed
            warnings = []
            if memory.percent > 80:
                warnings.append("High memory usage")
            if cpu_percent > 80:
                warnings.append("High CPU usage")
            if disk.used / disk.total > 0.9:
                warnings.append("Low disk space")

            if warnings:
                message += "\n⚠️ <b>Alerts:</b>\n"
                for warning in warnings:
                    message += f"• {warning}\n"

            message += f"\n⏰ <i>Updated: {datetime.now(timezone.utc).strftime('%H:%M UTC')}</i>"

            await update.message.reply_text(message, parse_mode="HTML")

        except Exception as e:
            self.logger.error(f"Error handling resources command: {e}")
            await update.message.reply_text("❌ Error retrieving resource information")

    async def handle_logs(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /logs command (admin only)."""
        try:
            # Parse parameters: [lines] [LEVEL] [trading|telegram|system|health]
            lines = 50
            level = "INFO"
            target = "trading"

            if context.args:
                for arg in context.args:
                    if arg.isdigit():
                        lines = max(1, min(int(arg), 300))
                    elif arg.upper() in ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]:
                        level = arg.upper()
                    elif arg.lower() in ["trading", "telegram", "system", "health"]:
                        target = arg.lower()

            # Resolve log file
            log_file = None
            if target == "trading":
                log_file = "logs/trading.log"
            elif target == "system":
                log_file = "logs/system.log"
            elif target == "health":
                log_file = "logs/health_monitor.log"
            else:  # telegram
                import glob

                files = glob.glob("logs/telegram_*.log")
                files.sort(reverse=True)
                log_file = files[0] if files else None

            if not log_file or not Path(log_file).exists():
                await update.message.reply_text(f"📭 No log file found for '{target}'")
                return

            entries = self._tail_log_filtered(log_file, lines, level)
            if not entries:
                await update.message.reply_text("📭 No matching log entries found")
                return

            header = f"📋 <b>Logs</b> ({target}, {level}+ last {len(entries)})\n\n"
            out = header
            count_msgs = 0
            for ln in entries:
                ln = ln.strip()
                if len(ln) > 220:
                    ln = ln[:217] + "..."
                line_fmt = f"<code>{ln}</code>\n"
                if len(out) + len(line_fmt) > 3800:
                    await update.message.reply_text(out, parse_mode="HTML")
                    out = ""
                    count_msgs += 1
                    if count_msgs >= 5:
                        break
                out += line_fmt
            if out and count_msgs < 5:
                await update.message.reply_text(out, parse_mode="HTML")

        except Exception as e:
            self.logger.error(f"Error handling logs command: {e}")
            await update.message.reply_text("❌ Error retrieving log information")

    def _tail_log_filtered(self, file_path: str, lines: int, min_level: str) -> List[str]:
        """Tail last N lines of a log file and filter by min_level."""
        level_order = {"DEBUG": 10, "INFO": 20, "WARNING": 30, "ERROR": 40, "CRITICAL": 50}
        min_val = level_order.get(min_level.upper(), 20)
        try:
            data = Path(file_path).read_text(encoding="utf-8", errors="ignore").splitlines()
            # Take a buffer to allow filtering
            buf = data[-min(len(data), lines * 5) :]
            res: List[str] = []
            for ln in reversed(buf):
                lvl_val = None
                for k, v in level_order.items():
                    if f" {k} " in ln or ln.startswith(k):
                        lvl_val = v
                        break
                if lvl_val is None:
                    lvl_val = 20
                if lvl_val >= min_val:
                    res.append(ln)
                if len(res) >= lines:
                    break
            return list(reversed(res))
        except Exception:
            return []

    async def handle_config(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /config command (admin only)."""
        try:
            config_data = await self._get_system_config()

            message = f"""
⚙️ <b>Configuration</b>
━━━━━━━━━━━━━━━━━
<b>Environment</b>
🏷️ Mode: {config_data.get('environment', 'Unknown')}
🐍 Python: {sys.version.split()[0]}
💻 Platform: {platform.system()} {platform.release()}
📍 Work Dir: {Path.cwd().name}

<b>Trading</b>
💰 Initial Balance: €{config_data.get('initial_balance', 0):,.2f}
📊 Max Position: {config_data.get('max_position_size', 0):.1%}
💸 Fee: {config_data.get('transaction_fee', 0):.1%}

<b>Logging</b>
📄 Level: {config_data.get('log_level', 'INFO')}
📁 Dir: {config_data.get('log_dir', 'logs/')}

<b>Paths</b>
📊 Data: {config_data.get('data_dir', 'data/')}
🤖 Models: {config_data.get('models_dir', 'models/')}
"""

            # Add API status (without exposing keys)
            apis = config_data.get("apis", {})
            if apis:
                message += "\n<b>API Status:</b>\n"
                for api_name, status in apis.items():
                    status_emoji = "🟢" if status else "🔴"
                    message += f"{status_emoji} {api_name}: {'Configured' if status else 'Not Configured'}\n"

            message += f"\n⏰ <i>Retrieved: {datetime.now(timezone.utc).strftime('%H:%M UTC')}</i>"

            await update.message.reply_text(message, parse_mode="HTML")

        except Exception as e:
            self.logger.error(f"Error handling config command: {e}")
            await update.message.reply_text("❌ Error retrieving configuration information")

    async def handle_version(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /version command."""
        try:
            message = f"""
📦 <b>Version</b>
━━━━━━━━━━━━━━━━━
🤖 Bot: 2.0.0-unified
📅 Build: {datetime.now(timezone.utc).strftime('%Y-%m-%d')}
🏷️ Release: Unified Telegram System

<b>System</b>
🐍 Python: {platform.python_version()}
💻 OS: {platform.system()} {platform.release()}
🏗️ Arch: {platform.machine()}

<b>Components</b>
• Unified Telegram Service
• Message Queue (persistent)
• Secure Credentials
• Command Registry + Auth
• Trading Integration
• Monitoring & Metrics
"""

            await update.message.reply_text(message, parse_mode="HTML")

        except Exception as e:
            self.logger.error(f"Error handling version command: {e}")
            await update.message.reply_text("❌ Error retrieving version information")

    # Helper methods

    async def _get_system_health(self) -> Dict[str, Any]:
        """Get comprehensive system health data."""
        try:
            # System metrics
            cpu_percent = psutil.cpu_percent(interval=0.1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage("/")
            load_avg = psutil.getloadavg()[0] if hasattr(psutil, "getloadavg") else 0.0

            # Service health checks
            services = {
                "Trading Engine": {
                    "healthy": True,
                    "status": "Running",
                },  # TODO: Integrate with actual services
                "Data Pipeline": {"healthy": True, "status": "Active"},
                "Risk Manager": {"healthy": True, "status": "Monitoring"},
                "Model Ensemble": {"healthy": True, "status": "Predicting"},
            }

            # Network connectivity test
            try:
                import socket

                socket.create_connection(("8.8.8.8", 53), timeout=3)
                internet_connected = True
            except:
                internet_connected = False

            # Overall health assessment
            warnings = []
            if memory.percent > 85:
                warnings.append("Memory usage critical")
            if cpu_percent > 90:
                warnings.append("CPU usage critical")
            if disk.used / disk.total > 0.95:
                warnings.append("Disk space critical")
            if not internet_connected:
                warnings.append("Internet connectivity issues")

            overall_healthy = (
                memory.percent < 90
                and cpu_percent < 95
                and disk.used / disk.total < 0.98
                and internet_connected
                and all(s["healthy"] for s in services.values())
            )

            return {
                "overall_healthy": overall_healthy,
                "services": services,
                "memory_percent": memory.percent,
                "cpu_percent": cpu_percent,
                "disk_percent": disk.used / disk.total * 100,
                "load_average": load_avg,
                "internet_connected": internet_connected,
                "api_healthy": True,  # TODO: Integrate with actual API health checks
                "warnings": warnings,
            }

        except Exception as e:
            self.logger.error(f"Error getting system health: {e}")
            return {
                "overall_healthy": False,
                "services": {},
                "memory_percent": 0,
                "cpu_percent": 0,
                "disk_percent": 0,
                "load_average": 0,
                "internet_connected": False,
                "api_healthy": False,
                "warnings": ["Error retrieving health data"],
            }

    async def _get_recent_logs(self, lines: int, level: str) -> List[Dict[str, Any]]:
        """Get recent log entries."""
        try:
            # TODO: Integrate with actual logging system
            # For now, return mock data
            log_entries = []

            levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
            min_level_idx = levels.index(level)

            for i in range(lines):
                entry_level = levels[max(min_level_idx, i % len(levels))]
                timestamp = (datetime.now(timezone.utc) - timedelta(minutes=i * 5)).strftime(
                    "%H:%M:%S"
                )

                messages = {
                    "DEBUG": "Processing market data update for BTCEUR",
                    "INFO": "Trading signal generated for ETHEUR with 78% confidence",
                    "WARNING": "High volatility detected in market conditions",
                    "ERROR": "Failed to connect to exchange API, retrying...",
                    "CRITICAL": "Risk management circuit breaker triggered",
                }

                log_entries.append(
                    {
                        "timestamp": timestamp,
                        "level": entry_level,
                        "message": messages.get(entry_level, "System operation normal"),
                    }
                )

            return log_entries

        except Exception as e:
            self.logger.error(f"Error getting recent logs: {e}")
            return []

    async def _get_system_config(self) -> Dict[str, Any]:
        """Get system configuration (safe/non-sensitive data only)."""
        try:
            import os

            return {
                "environment": os.getenv("ENVIRONMENT", "development"),
                "initial_balance": float(os.getenv("INITIAL_BALANCE", "10000.0")),
                "max_position_size": float(os.getenv("MAX_POSITION_SIZE", "0.1")),
                "transaction_fee": float(os.getenv("TRANSACTION_FEE", "0.001")),
                "log_level": os.getenv("LOG_LEVEL", "INFO"),
                "log_dir": os.getenv("LOG_FILE", "logs/").replace("/trading_bot.log", ""),
                "data_dir": os.getenv("DATA_DIR", "./data"),
                "models_dir": os.getenv("MODELS_DIR", "./models"),
                "apis": {
                    "Telegram": bool(os.getenv("TELEGRAM_BOT_TOKEN")),
                    "Bitvavo": bool(os.getenv("BITVAVO_API_KEY")),
                    "Binance": bool(os.getenv("BINANCE_API_KEY")),
                    "AWS": bool(os.getenv("AWS_ACCESS_KEY_ID")),
                },
            }

        except Exception as e:
            self.logger.error(f"Error getting system config: {e}")
            return {}
