"""
Database Builder
================

Utilities to (re)build SQLite databases for symbols and intervals.

This module is invoked by the Telegram `/database` command via
`src.notifier.enhanced_telegram.EnhancedTelegramNotifier`.

The DB schema matches expectations from `src.data_pipeline.loader.DataLoader`:
 - table: market_data
 - columns: datetime (TEXT), timestamp (INTEGER seconds), open, high, low, close, volume
 - optional columns present but nullable: quote_volume, taker_buy_base, taker_buy_quote
"""

from __future__ import annotations

import asyncio
import logging
import sqlite3

# Use the final data fetcher - the ultimate solution
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, Iterable, List, Optional

import pandas as pd

final_fetcher_path = Path(__file__).parent.parent.parent / "final_data_fetcher.py"
if final_fetcher_path.exists():
    sys.path.insert(0, str(final_fetcher_path.parent))
    try:
        from final_data_fetcher import FinalDataFetcher

        SimpleDataFetcher = FinalDataFetcher  # Compatible interface
    except Exception:
        SimpleDataFetcher = None
else:
    SimpleDataFetcher = None

logger = logging.getLogger(__name__)


@dataclass
class BuildResult:
    symbol: str
    interval: str
    db_path: Path
    rows: int


def _ensure_schema(conn: sqlite3.Connection) -> None:
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS market_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            datetime TEXT NOT NULL,
            timestamp INTEGER NOT NULL,
            open REAL NOT NULL,
            high REAL NOT NULL,
            low REAL NOT NULL,
            close REAL NOT NULL,
            volume REAL NOT NULL,
            quote_volume REAL,
            taker_buy_base REAL,
            taker_buy_quote REAL
        )
        """
    )
    cur.execute("CREATE INDEX IF NOT EXISTS idx_market_data_ts ON market_data(timestamp)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_market_data_dt ON market_data(datetime)")
    conn.commit()


def _insert_rows(conn: sqlite3.Connection, df: pd.DataFrame) -> int:
    # Normalize columns to lower-case expected by loader
    col_map = {
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Close": "close",
        "Volume": "volume",
    }
    df2 = df.copy()
    # If index is datetime-like, keep it; otherwise try to locate a datetime column
    if not isinstance(df2.index, pd.DatetimeIndex):
        for cand in ("Datetime", "datetime", "timestamp"):
            if cand in df2.columns:
                try:
                    df2.index = pd.to_datetime(df2[cand])
                    break
                except Exception:
                    continue
    if not isinstance(df2.index, pd.DatetimeIndex):
        raise ValueError("DataFrame must have datetime index or a Datetime/datetime column")

    # Rename OHLCV columns
    for src, dst in col_map.items():
        if src in df2.columns:
            df2.rename(columns={src: dst}, inplace=True)

    # Ensure required columns are present
    for req in ("open", "high", "low", "close", "volume"):
        if req not in df2.columns:
            raise ValueError(f"Missing required column: {req}")

    # Prepare rows
    rows: List[tuple] = []
    for ts, row in df2.iterrows():
        dt = pd.to_datetime(ts).to_pydatetime()
        ts_sec = int(dt.timestamp())
        rows.append(
            (
                dt.isoformat(),
                ts_sec,
                float(row["open"]),
                float(row["high"]),
                float(row["low"]),
                float(row["close"]),
                float(row["volume"]),
                0.0,
                0.0,
                0.0,
            )
        )

    cur = conn.cursor()
    cur.executemany(
        """
        INSERT INTO market_data (
            datetime, timestamp, open, high, low, close, volume, quote_volume, taker_buy_base, taker_buy_quote
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        rows,
    )
    conn.commit()
    return len(rows)


def _build_one_db(symbol: str, interval: str, db_path: Path, df: pd.DataFrame) -> int:
    """Build a single SQLite DB file with provided DataFrame (runs in a worker thread)."""
    # Create connection within this thread to avoid cross-thread issues
    conn = sqlite3.connect(str(db_path))
    try:
        _ensure_schema(conn)
        return _insert_rows(conn, df)
    finally:
        conn.close()


async def rebuild_databases(
    symbols: Iterable[str],
    interval: str,
    data_dir: str = "data",
    days: int = 365,
    log_cb: Optional[Callable[[str], None]] = None,
) -> List[BuildResult]:
    """Rebuild SQLite databases for the given symbols.

    - Fetches OHLCV via `SimpleDataFetcher` with robust fallbacks
    - Creates `data/<symbol>_<interval>.db` files
    - Writes to `market_data` table in expected schema

    Args:
        symbols: List of symbols (e.g., ["BTCEUR", "ETHEUR", ...])
        interval: Candle interval (e.g., "30m", "1h")
        data_dir: Directory to place databases
        days: History depth hint for fetcher
        log_cb: Optional logging callback
    """

    def log(msg: str) -> None:
        if log_cb:
            try:
                log_cb(msg)
                return
            except Exception:
                pass
        logger.info(msg)

    if SimpleDataFetcher is None:
        raise RuntimeError(
            "SimpleDataFetcher not available; ensure paperspace_mlops is in PYTHONPATH"
        )

    out: List[BuildResult] = []
    fetcher = SimpleDataFetcher()
    # Align fetcher configuration to requested interval/days if supported
    try:
        setattr(fetcher, "interval", interval)
        setattr(fetcher, "lookback_days", int(days))
    except Exception:
        pass
    data_path = Path(data_dir)
    data_path.mkdir(parents=True, exist_ok=True)

    for symbol in symbols:
        sym = str(symbol).upper()
        db_path = data_path / f"{sym.lower()}_{interval.lower()}.db"
        log(f"Fetching data for {sym} ({interval}) …")

        # Fetch in thread to avoid blocking loop
        # FinalDataFetcher.fetch_symbol_data expects only (symbol)
        df: Optional[pd.DataFrame] = await asyncio.to_thread(
            fetcher.fetch_symbol_data, sym
        )
        if df is None or len(df) < 20:
            raise RuntimeError(f"No/insufficient data for {sym} at {interval}")

        # Create DB and write rows
        # Remove existing file if present (backup handled by caller)
        try:
            if db_path.exists():
                db_path.unlink()
        except Exception:
            # Ignore unlink errors; we will overwrite
            pass

        rows = await asyncio.to_thread(_build_one_db, sym, interval, db_path, df)
        out.append(BuildResult(symbol=sym, interval=interval, db_path=db_path, rows=rows))
        log(f"Built {db_path.name}: {rows} rows")

    log(f"Completed rebuild for {len(out)} databases")
    return out
