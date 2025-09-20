"""
Trading-specific Telegram command handlers.
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from telegram import Update
from telegram.ext import ContextTypes

from src.core.logging_manager import get_system_logger


class TradingCommandHandler:
    """Handler for trading-related Telegram commands."""

    def __init__(self):
        self.logger = get_system_logger(__name__)

    def register_commands(self, registry):
        """Register all trading commands with the command registry."""
        registry.register_command(
            name="portfolio",
            handler=self.handle_portfolio,
            description="Show current portfolio status",
            admin_only=False,
            rate_limit=10,
        )

        registry.register_command(
            name="positions",
            handler=self.handle_positions,
            description="Show active positions",
            admin_only=False,
            rate_limit=10,
            aliases=["pos"],
        )

        registry.register_command(
            name="performance",
            handler=self.handle_performance,
            description="Show trading performance metrics",
            admin_only=False,
            rate_limit=5,
            aliases=["perf", "pnl"],
        )

        registry.register_command(
            name="trades",
            handler=self.handle_recent_trades,
            description="Show recent trades",
            admin_only=False,
            rate_limit=10,
        )

        registry.register_command(
            name="signals",
            handler=self.handle_signals,
            description="Show current trading signals",
            admin_only=False,
            rate_limit=15,
        )

        registry.register_command(
            name="risk",
            handler=self.handle_risk_status,
            description="Show risk management status",
            admin_only=False,
            rate_limit=5,
        )

        self.logger.info("Trading commands registered")

    async def handle_portfolio(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /portfolio command."""
        try:
            # This would integrate with your actual portfolio manager
            portfolio_data = await self._get_portfolio_data()

            if not portfolio_data:
                await update.message.reply_text("❌ Unable to retrieve portfolio data")
                return

            message = f"""
📊 <b>Portfolio Overview</b>
━━━━━━━━━━━━━━━━━
💰 <b>Total Value:</b> €{portfolio_data.get('total_value', 0):,.2f}
📈 <b>Total P&L:</b> €{portfolio_data.get('total_pnl', 0):+,.2f} ({portfolio_data.get('total_pnl_percent', 0):+.1f}%)
💵 <b>Cash:</b> €{portfolio_data.get('cash', 0):,.2f}

<b>Asset Allocation</b>
"""

            # Add asset breakdown
            for asset, data in portfolio_data.get("assets", {}).items():
                quantity = data.get("quantity", 0)
                value = data.get("value", 0)
                pnl_percent = data.get("pnl_percent", 0)

                status_emoji = "🟢" if pnl_percent >= 0 else "🔴"

                message += f"{status_emoji} <b>{asset}:</b> {quantity:.6f} (€{value:,.2f}) {pnl_percent:+.1f}%\n"

            message += f"\n⏰ <i>Updated: {datetime.now(timezone.utc).strftime('%H:%M UTC')}</i>"

            await update.message.reply_text(message, parse_mode="HTML")

        except Exception as e:
            self.logger.error(f"Error handling portfolio command: {e}")
            await update.message.reply_text("❌ Error retrieving portfolio information")

    async def handle_positions(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /positions command."""
        try:
            positions_data = await self._get_positions_data()

            if not positions_data:
                await update.message.reply_text("📭 No active positions")
                return

            message = f"📍 <b>Active Positions</b> ({len(positions_data)})\n━━━━━━━━━━━━━━━━━\n"

            for position in positions_data:
                symbol = position.get("symbol", "UNKNOWN")
                side = position.get("side", "UNKNOWN")
                size = position.get("size", 0)
                entry_price = position.get("entry_price", 0)
                current_price = position.get("current_price", 0)
                pnl = position.get("unrealized_pnl", 0)
                pnl_percent = position.get("pnl_percent", 0)

                side_emoji = "🟢" if side.upper() == "LONG" else "⚪"
                pnl_emoji = "📈" if pnl >= 0 else "📉"

                message += f"""
{side_emoji} <b>{symbol}</b> • {side.upper()}
📦 Size: {size:.6f}
🎯 Entry: €{entry_price:.4f}
💱 Price: €{current_price:.4f}
{pnl_emoji} P&L: €{pnl:+,.2f} ({pnl_percent:+.1f}%)
"""

            message += f"\n⏰ <i>Updated: {datetime.now(timezone.utc).strftime('%H:%M UTC')}</i>"

            await update.message.reply_text(message, parse_mode="HTML")

        except Exception as e:
            self.logger.error(f"Error handling positions command: {e}")
            await update.message.reply_text("❌ Error retrieving positions information")

    async def handle_performance(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /performance command."""
        try:
            perf_data = await self._get_performance_data()

            if not perf_data:
                await update.message.reply_text("❌ Unable to retrieve performance data")
                return

            # Parse timeframe from command args
            timeframe = "24h"  # default
            if context.args:
                timeframe = context.args[0].lower()

            message = f"""
📈 <b>Performance</b> ({timeframe.upper()})
━━━━━━━━━━━━━━━━━
💰 <b>Total P&L:</b> €{perf_data.get('total_pnl', 0):+,.2f}
📊 <b>ROI:</b> {perf_data.get('roi_percent', 0):+.1f}%
🏆 <b>Win Rate:</b> {perf_data.get('win_rate', 0):.1f}%
⚡ <b>Trades:</b> {perf_data.get('total_trades', 0)}

<b>Risk Metrics</b>
📉 Max Drawdown: {perf_data.get('max_drawdown', 0):.1f}%
📊 Sharpe Ratio: {max(-10.0, min(10.0, perf_data.get('sharpe_ratio', 0))):.2f}
"""

            message += f"\n⏰ <i>Period: {timeframe.upper()} | Updated: {datetime.now(timezone.utc).strftime('%H:%M UTC')}</i>"

            await update.message.reply_text(message, parse_mode="HTML")

        except Exception as e:
            self.logger.error(f"Error handling performance command: {e}")
            await update.message.reply_text("❌ Error retrieving performance information")

    async def handle_recent_trades(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /trades command."""
        try:
            # Parse limit from command args
            limit = 5  # default
            if context.args and context.args[0].isdigit():
                limit = min(int(context.args[0]), 20)  # max 20 trades

            trades_data = await self._get_recent_trades(limit)

            if not trades_data:
                await update.message.reply_text("📭 No recent trades found")
                return

            message = f"📋 <b>Recent Trades</b> (Last {len(trades_data)})\n━━━━━━━━━━━━━━━━━\n"

            for trade in trades_data:
                symbol = trade.get("symbol", "UNKNOWN")
                side = trade.get("side", "UNKNOWN")
                quantity = trade.get("quantity", 0)
                price = trade.get("price", 0)
                pnl = trade.get("realized_pnl", 0)
                timestamp = trade.get("timestamp", datetime.now(timezone.utc))

                if isinstance(timestamp, str):
                    timestamp = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))

                side_emoji = "🟢" if side.upper() == "BUY" else "🔴"
                pnl_emoji = "💰" if pnl > 0 else "💸" if pnl < 0 else "🔄"

                time_str = timestamp.strftime("%m/%d %H:%M")

                message += f"{side_emoji} <b>{symbol}</b> {side.upper()} {abs(quantity):.6f} @ €{price:.4f}\n"
                message += f"   {pnl_emoji} P&L: €{pnl:+,.2f} • {time_str}\n\n"

            await update.message.reply_text(message, parse_mode="HTML")

        except Exception as e:
            self.logger.error(f"Error handling trades command: {e}")
            await update.message.reply_text("❌ Error retrieving trades information")

    async def handle_signals(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /signals command."""
        try:
            signals_data = await self._get_trading_signals()

            if not signals_data:
                await update.message.reply_text("📡 No active trading signals")
                return

            message = f"📡 <b>Active Trading Signals</b> ({len(signals_data)})\n━━━━━━━━━━━━━━━━━\n"

            for signal in signals_data:
                symbol = signal.get("symbol", "UNKNOWN")
                action = signal.get("action", "HOLD")
                confidence = signal.get("confidence", 0)
                price = signal.get("current_price", 0)
                target = signal.get("target_price")
                stop_loss = signal.get("stop_loss")
                timestamp = signal.get("timestamp", datetime.now(timezone.utc))

                if isinstance(timestamp, str):
                    timestamp = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))

                # Determine emoji and color based on action and confidence
                if action.upper() == "BUY":
                    action_emoji = "🟢"
                elif action.upper() == "SELL":
                    action_emoji = "🔴"
                else:
                    action_emoji = "🟡"

                confidence_bar = "🟩" * int(confidence * 10) + "⬜" * (10 - int(confidence * 10))

                time_str = timestamp.strftime("%H:%M")

                message += f"{action_emoji} <b>{symbol}</b> • {action.upper()}\n"
                message += f"📊 Confidence: {confidence:.1%} {confidence_bar}\n"
                message += f"💱 Price: €{price:.4f}"

                # Include thresholds and per-model diagnostics if present
                thresholds = signal.get("thresholds")
                if thresholds:
                    try:
                        message += f"\n⚙️ Thr: buy {float(thresholds.get('buy', 0)):.6f} | sell {float(thresholds.get('sell', 0)):.6f}"
                    except Exception:
                        pass

                per_model = signal.get("per_model") or []
                if per_model:
                    for pm in per_model:
                        try:
                            message += (
                                f"\n• {pm.get('model','?')}: pred {float(pm.get('prediction',0)):.6f} "
                                f"conf {float(pm.get('confidence',0)):.2f} w {float(pm.get('weight',0)):.2f}"
                            )
                        except Exception:
                            continue

                if target:
                    message += f" | 🎯 Target: ${target:.4f}"
                if stop_loss:
                    message += f" | 🛑 Stop: ${stop_loss:.4f}"

                message += f"\n⏰ {time_str}\n\n"

            await update.message.reply_text(message, parse_mode="HTML")

        except Exception as e:
            self.logger.error(f"Error handling signals command: {e}")
            await update.message.reply_text("❌ Error retrieving signals information")

    async def handle_risk_status(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /risk command."""
        try:
            risk_data = await self._get_risk_status()

            if not risk_data:
                await update.message.reply_text("❌ Unable to retrieve risk data")
                return

            # Determine risk level emoji
            risk_level = risk_data.get("risk_level", "UNKNOWN").upper()
            risk_emojis = {"LOW": "🟢", "MEDIUM": "🟡", "HIGH": "🟠", "CRITICAL": "🔴"}
            risk_emoji = risk_emojis.get(risk_level, "⚪")

            message = f"""
🛡️ <b>Risk Status</b>
━━━━━━━━━━━━━━━━━
{risk_emoji} <b>Risk Level:</b> {risk_level}
📊 <b>Portfolio Risk:</b> {risk_data.get('portfolio_risk', 0):.1f}%
📦 <b>Position Size:</b> {risk_data.get('position_size_percent', 0):.1f}%
📉 <b>Drawdown:</b> {risk_data.get('current_drawdown', 0):.1f}%
🎯 <b>Risk Limit:</b> {risk_data.get('risk_limit', 0):.1f}%

<b>Circuit Breakers</b>
"""

            # Add circuit breaker statuses
            breakers = risk_data.get("circuit_breakers", {})
            for breaker_name, status in breakers.items():
                status_emoji = "🔴" if status.get("triggered", False) else "🟢"
                message += f"{status_emoji} {breaker_name}: {'TRIGGERED' if status.get('triggered') else 'NORMAL'}\n"

            # Add warnings if any
            warnings = risk_data.get("warnings", [])
            if warnings:
                message += "\n⚠️ <b>Warnings:</b>\n"
                for warning in warnings:
                    message += f"• {warning}\n"

            message += f"\n⏰ <i>Updated: {datetime.now(timezone.utc).strftime('%H:%M UTC')}</i>"

            await update.message.reply_text(message, parse_mode="HTML")

        except Exception as e:
            self.logger.error(f"Error handling risk command: {e}")
            await update.message.reply_text("❌ Error retrieving risk information")

    # Helper methods to integrate with actual trading system

    async def _get_portfolio_data(self) -> Optional[Dict[str, Any]]:
        """Get current portfolio data from trading system."""
        try:
            # Prefer real-time data from logs/balance.json if present
            balance_path = Path("logs/balance.json")
            if balance_path.exists():
                import json

                with balance_path.open("r", encoding="utf-8") as f:
                    data = json.load(f)

                total_value = float(data.get("portfolio_value", 0.0))
                total_pnl = float(data.get("total_pnl", 0.0))

                # Prefer explicit total_return from metrics when available to avoid
                # drift from repeated derived calculations.
                total_return = data.get("total_return")
                if total_return is not None:
                    try:
                        total_pnl_percent = float(total_return)
                        # Values in JSON are expressed as a ratio (0.01 == 1%).
                        if abs(total_pnl_percent) <= 1.0:
                            total_pnl_percent *= 100.0
                    except Exception:
                        total_pnl_percent = 0.0
                else:
                    initial_estimate = total_value - total_pnl
                    total_pnl_percent = (
                        (total_pnl / initial_estimate) * 100.0 if initial_estimate > 0 else 0.0
                    )

                # Build per-asset snapshots from balance values, then enrich with the
                # reconstructed position quantities used by the /positions command so the
                # portfolio view reports real sizing instead of zeros.
                assets: Dict[str, Dict[str, float]] = {}
                raw_positions = data.get("positions", {}) or {}

                # Reuse the detailed position reconstruction to source quantities/pnl
                # without duplicating CSV parsing logic here.
                detailed_positions = await self._get_positions_data()
                position_lookup = {
                    pos["symbol"]: pos for pos in (detailed_positions or []) if pos.get("symbol")
                }

                for sym, val in raw_positions.items():
                    try:
                        value_f = float(val)
                    except Exception:
                        value_f = 0.0

                    details = position_lookup.get(sym, {})
                    assets[sym] = {
                        "quantity": float(details.get("size", 0.0)) if details else 0.0,
                        "value": value_f,
                        "pnl_percent": (float(details.get("pnl_percent", 0.0)) if details else 0.0),
                    }

                return {
                    "total_value": total_value,
                    "total_pnl": total_pnl,
                    "total_pnl_percent": total_pnl_percent,
                    "cash": float(data.get("cash_balance", 0.0)),
                    "assets": assets,
                }

            # Fallback mock data
            return {
                "total_value": 12350.75,
                "total_pnl": 350.75,
                "total_pnl_percent": 2.9,
                "cash": 2500.00,
                "assets": {},
            }
        except Exception as e:
            self.logger.error(f"Error getting portfolio data: {e}")
            return None

    async def _get_positions_data(self) -> Optional[List[Dict[str, Any]]]:
        """Get active positions from trading system."""
        try:
            # Read balance.json for per-symbol position value
            balance_path = Path("logs/balance.json")
            balance_data: Dict[str, Any] = {}
            if balance_path.exists():
                import json

                with balance_path.open("r", encoding="utf-8") as f:
                    balance_data = json.load(f) or {}

            pos_values: Dict[str, float] = {}
            for sym, val in (balance_data.get("positions", {}) or {}).items():
                try:
                    pos_values[sym] = float(val)
                except Exception:
                    pos_values[sym] = 0.0

            # Reconstruct per-symbol quantity and average cost from trades_report.csv
            quantities: Dict[str, float] = {}
            avg_costs: Dict[str, float] = {}
            trades_path = Path("logs/trades_report.csv")
            if trades_path.exists():
                import csv

                try:
                    # Process in chronological order
                    for row in csv.reader(trades_path.open("r", encoding="utf-8", errors="ignore")):
                        if not row or len(row) < 6:
                            continue
                        symbol = row[2]
                        side = (row[3] or "").strip().upper()
                        try:
                            qty = float(row[4])
                            price = float(row[5])
                        except Exception:
                            qty, price = 0.0, 0.0
                        if not symbol or qty <= 0 or price <= 0:
                            continue

                        cur_qty = quantities.get(symbol, 0.0)
                        cur_avg = avg_costs.get(symbol, 0.0)

                        if side == "BUY":
                            # Moving average cost update
                            new_qty = cur_qty + qty
                            if new_qty > 0:
                                cur_avg = (cur_avg * cur_qty + price * qty) / new_qty
                            cur_qty = new_qty
                        elif side == "SELL":
                            # Reduce quantity; keep avg cost for remaining shares
                            cur_qty = max(0.0, cur_qty - qty)
                            if cur_qty == 0:
                                cur_avg = 0.0

                        quantities[symbol] = cur_qty
                        avg_costs[symbol] = cur_avg
                except Exception:
                    # If CSV parsing fails, fall back silently
                    quantities = {}
                    avg_costs = {}

            # Build a latest price cache to derive actual quantities from portfolio value
            price_cache: Dict[str, float] = {}
            for symbol in pos_values.keys():
                latest_price = await self._get_latest_price(symbol)
                if latest_price and latest_price > 0:
                    price_cache[symbol] = latest_price

            positions: List[Dict[str, Any]] = []
            for symbol, value_f in pos_values.items():
                avg_cost = avg_costs.get(symbol, 0.0)

                market_price = price_cache.get(symbol)

                qty = quantities.get(symbol, 0.0)
                if market_price:
                    inferred_qty = value_f / market_price if market_price else 0.0
                    if inferred_qty > 0:
                        qty = inferred_qty

                current_price = market_price or ((value_f / qty) if qty else 0.0)

                unrealized_pnl = 0.0
                pnl_percent = 0.0
                if qty and avg_cost > 0 and current_price > 0:
                    unrealized_pnl = (current_price - avg_cost) * qty
                    denom = avg_cost * qty
                    if denom > 0:
                        pnl_percent = (unrealized_pnl / denom) * 100.0

                positions.append(
                    {
                        "symbol": symbol,
                        "side": "LONG" if value_f > 0 else "FLAT",
                        "size": abs(qty) if qty else 0.0,
                        "entry_price": avg_cost,
                        "current_price": current_price,
                        "unrealized_pnl": unrealized_pnl,
                        "pnl_percent": pnl_percent,
                    }
                )

            if positions:
                return positions

            # Fallback mock if file not present
            return []
        except Exception as e:
            self.logger.error(f"Error getting positions data: {e}")
            return None

    async def _get_latest_price(self, symbol: str) -> Optional[float]:
        """Fetch the most recent close price for a symbol from local datasets."""
        try:
            import sqlite3

            base = symbol.lower()
            candidates = [
                Path("data") / f"{base}_30m.db",
                Path("data") / f"{base}_1h.db",
                Path("data") / f"{base}_1d.db",
            ]

            for db_path in candidates:
                if not db_path.exists():
                    continue

                conn = sqlite3.connect(str(db_path))
                try:
                    cursor = conn.cursor()
                    cursor.execute("SELECT close FROM market_data ORDER BY timestamp DESC LIMIT 1")
                    row = cursor.fetchone()
                    if row and row[0] is not None:
                        return float(row[0])
                except Exception as e:
                    self.logger.debug(f"Failed to read price from {db_path.name} for {symbol}: {e}")
                finally:
                    conn.close()

            return None
        except Exception as e:
            self.logger.debug(f"Error loading latest price for {symbol}: {e}")
            return None

    async def _get_performance_data(self) -> Optional[Dict[str, Any]]:
        """Get performance metrics from trading system."""
        try:
            # Prefer real-time data from logs/performance_metrics.json if present
            perf_path = Path("logs/performance_metrics.json")
            if perf_path.exists():
                import json

                with perf_path.open("r", encoding="utf-8") as f:
                    data = json.load(f)

                # Map known fields; fill unknowns with safe defaults
                total_pnl = float(data.get("total_pnl", 0.0))
                total_return = float(data.get("total_return", 0.0))
                roi_percent = total_return * 100.0 if abs(total_return) <= 1.0 else total_return

                return {
                    "total_pnl": total_pnl,
                    "roi_percent": roi_percent,
                    "win_rate": float(data.get("win_rate", 0.0)),
                    "avg_win": 0.0,
                    "avg_loss": 0.0,
                    "total_trades": int(data.get("total_trades", 0)),
                    "max_drawdown": float(data.get("max_drawdown", 0.0)),
                    "sharpe_ratio": float(data.get("sharpe_ratio", 0.0)),
                    "profit_factor": 0.0,
                }

            # Fallback mock data
            return {
                "total_pnl": 0.0,
                "roi_percent": 0.0,
                "win_rate": 0.0,
                "avg_win": 0.0,
                "avg_loss": 0.0,
                "total_trades": 0,
                "max_drawdown": 0.0,
                "sharpe_ratio": 0.0,
                "profit_factor": 0.0,
            }
        except Exception as e:
            self.logger.error(f"Error getting performance data: {e}")
            return None

    async def _get_recent_trades(self, limit: int) -> Optional[List[Dict[str, Any]]]:
        """Get recent trades from trading system."""
        try:
            import csv

            trades_path = Path("logs/trades_report.csv")
            if not trades_path.exists():
                return []

            all_rows: List[List[str]] = []
            with trades_path.open("r", encoding="utf-8", errors="ignore") as csv_file:
                reader = csv.reader(csv_file)
                for row in reader:
                    if row:
                        all_rows.append(row)

            if not all_rows:
                return []

            selected_rows = all_rows[-limit:]
            trades: List[Dict[str, Any]] = []

            # Build current price map, favouring on-disk candle data for accuracy
            current_price_map: Dict[str, float] = {}
            try:
                import json

                bal = {}
                bp = Path("logs/balance.json")
                if bp.exists():
                    with bp.open("r", encoding="utf-8") as f:
                        bal = json.load(f) or {}
                pos_vals = bal.get("positions", {}) or {}

                for sym in pos_vals.keys():
                    price = await self._get_latest_price(sym)
                    if price and price > 0:
                        current_price_map[sym] = price
                    else:
                        try:
                            v = float(pos_vals[sym])
                        except Exception:
                            v = 0.0
                        if v > 0:
                            # Fallback: attempt to infer using recent trades quantity
                            qty = 0.0
                            try:
                                for row_all in all_rows:
                                    if len(row_all) < 6 or row_all[2] != sym:
                                        continue
                                    side_all = (row_all[3] or "").strip().upper()
                                    try:
                                        q_all = float(row_all[4])
                                    except Exception:
                                        q_all = 0.0
                                    signed = -abs(q_all) if side_all == "SELL" else abs(q_all)
                                    qty += signed
                                if qty:
                                    current_price_map[sym] = v / qty if qty != 0 else 0.0
                            except Exception:
                                continue
            except Exception:
                current_price_map = {}

            # Ensure we have price coverage for the symbols we'll display even if no
            # active position is recorded for them.
            symbols_in_selected = {row[2] for row in selected_rows if len(row) > 2}
            for sym in symbols_in_selected:
                if sym not in current_price_map:
                    price = await self._get_latest_price(sym)
                    if price and price > 0:
                        current_price_map[sym] = price

            # Track unique trades to prevent duplicates
            seen_trades = set()

            for row in selected_rows:
                # Expected columns:
                # 0 timestamp, 1 trade_id, 2 symbol, 3 trade_type, 4 quantity,
                # 5 price, 6 status, 7 notes, 8 model_used, 9 confidence, 10 balance
                if len(row) < 7:
                    continue

                def _safe_float(x: Any, default: float = 0.0) -> float:
                    try:
                        return float(x)
                    except Exception:
                        return default

                timestamp = row[0]
                symbol = row[2] if len(row) > 2 else "UNKNOWN"
                side = row[3].upper() if len(row) > 3 else "UNKNOWN"
                quantity = _safe_float(row[4] if len(row) > 4 else 0.0)
                price = _safe_float(row[5] if len(row) > 5 else 0.0)
                confidence = _safe_float(row[-2] if len(row) >= 2 else 0.0)

                # Create unique trade identifier to prevent duplicates
                trade_key = (timestamp, symbol, side, quantity, price)
                if trade_key in seen_trades:
                    continue
                seen_trades.add(trade_key)

                # Estimate mark-to-market P&L using current price if known
                cur_px = current_price_map.get(symbol, 0.0)
                est_pnl = 0.0
                if cur_px > 0 and quantity > 0:
                    if side == "BUY":
                        est_pnl = (cur_px - price) * quantity
                    elif side == "SELL":
                        est_pnl = (price - cur_px) * quantity

                trades.append(
                    {
                        "symbol": symbol,
                        "side": side,
                        "quantity": quantity,
                        "price": price,
                        "realized_pnl": est_pnl,
                        "timestamp": timestamp,
                        "confidence": confidence,
                    }
                )

            return trades
        except Exception as e:
            self.logger.error(f"Error getting recent trades: {e}")
            return None

    async def _get_trading_signals(self) -> Optional[List[Dict[str, Any]]]:
        """Get current trading signals, enriched with per-model diagnostics if available."""
        try:
            import json
            from glob import glob
            from pathlib import Path

            ts = datetime.now(timezone.utc)

            # Prefer structured diagnostics JSON from the latest cycle
            diag_dir = Path("logs/diagnostics")
            if diag_dir.exists():
                files = sorted(glob("logs/diagnostics/*.json"))
                if files:
                    latest = files[-1]
                    data = json.loads(Path(latest).read_text(encoding="utf-8"))
                    results: List[Dict[str, Any]] = []
                    for sym, diag in data.items():
                        decision = diag.get("decision", {})
                        action = decision.get("action", "HOLD")
                        confidence = float(decision.get("confidence", 0.0)) if decision else 0.0
                        results.append(
                            {
                                "symbol": sym,
                                "action": action,
                                "confidence": confidence,
                                "thresholds": diag.get("thresholds", {}),
                                "per_model": diag.get("per_model", []),
                                "timestamp": ts,
                            }
                        )
                    return results

            # Fallback: parse last "Enhanced signal generation completed" summary
            import ast
            import re

            candidates = []
            if Path("logs/trading.log").exists():
                candidates.append("logs/trading.log")
            trader_logs = sorted(glob("logs/trader_*.log"), reverse=True)
            if trader_logs:
                candidates.append(trader_logs[0])
            summary = None
            pattern = re.compile(r"Enhanced signal generation completed: (\{.*\})")
            for file_path in candidates:
                try:
                    lines = (
                        Path(file_path).read_text(encoding="utf-8", errors="ignore").splitlines()
                    )
                    for line in reversed(lines[-200:]):
                        m = pattern.search(line)
                        if m:
                            summary = m.group(1)
                            break
                    if summary:
                        break
                except Exception:
                    continue
            if not summary:
                return []
            data = ast.literal_eval(summary)
            results: List[Dict[str, Any]] = []
            for sym, count in data.items():
                try:
                    c = int(count)
                except Exception:
                    c = 0
                if c == 0:
                    continue
                action = "BUY" if c > 0 else "SELL"
                confidence = min(abs(c) / 3.0, 1.0)
                results.append(
                    {
                        "symbol": sym,
                        "action": action,
                        "confidence": confidence,
                        "timestamp": ts,
                    }
                )
            return results
        except Exception as e:
            self.logger.error(f"Error getting trading signals: {e}")
            return None

    async def _get_risk_status(self) -> Optional[Dict[str, Any]]:
        """Get risk management status from logs and derived metrics."""
        try:
            import json
            from pathlib import Path

            # Pull drawdowns and trades/ROI from performance metrics
            perf = {}
            if Path("logs/performance_metrics.json").exists():
                perf = json.loads(Path("logs/performance_metrics.json").read_text(encoding="utf-8"))
            current_drawdown = float(perf.get("current_drawdown", 0.0))
            max_drawdown = float(perf.get("max_drawdown", 0.0))

            # Derive concentration risk from balance.json
            portfolio_value = 0.0
            cash = 0.0
            pos_vals = {}
            if Path("logs/balance.json").exists():
                bal = json.loads(Path("logs/balance.json").read_text(encoding="utf-8"))
                portfolio_value = float(bal.get("portfolio_value", 0.0))
                cash = float(bal.get("cash_balance", 0.0))
                pos_vals = {k: float(v) for k, v in (bal.get("positions", {}) or {}).items()}

            positions_total = sum(v for v in pos_vals.values() if v > 0)
            max_concentration_pct = (
                (max((v for v in pos_vals.values()), default=0.0) / portfolio_value * 100.0)
                if portfolio_value > 0
                else 0.0
            )

            # Heuristic portfolio risk as max of drawdown% and concentration%
            portfolio_risk_pct = max(current_drawdown * 100.0, max_concentration_pct)

            # Simple circuit breakers
            risk_limit_dd = 10.0  # % threshold; could be configured
            risk_limit_conc = 40.0  # % per-symbol concentration
            breakers = {
                "Max Drawdown": {"triggered": (current_drawdown * 100.0) > risk_limit_dd},
                "Position Concentration": {"triggered": max_concentration_pct > risk_limit_conc},
            }

            risk_level = "LOW"
            if (
                breakers["Max Drawdown"]["triggered"]
                or breakers["Position Concentration"]["triggered"]
            ):
                risk_level = "HIGH"
            elif portfolio_risk_pct > 5.0:
                risk_level = "MEDIUM"

            return {
                "risk_level": risk_level,
                "portfolio_risk": portfolio_risk_pct,
                "position_size_percent": (
                    (positions_total / portfolio_value * 100.0) if portfolio_value > 0 else 0.0
                ),
                "current_drawdown": current_drawdown * 100.0,
                "risk_limit": risk_limit_dd,
                "circuit_breakers": breakers,
                "warnings": [],
            }
        except Exception as e:
            self.logger.error(f"Error getting risk status: {e}")
            return None
