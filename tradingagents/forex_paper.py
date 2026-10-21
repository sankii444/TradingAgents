"""A local-only, USD-quoted spot-FX paper account with bounded position sizing."""

from __future__ import annotations

import math
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from tradingagents.dataflows.config import run_config
from tradingagents.dataflows.symbols import normalize_symbol
from tradingagents.dataflows.vendors.yahoo.ohlcv import load_ohlcv
from tradingagents.portfolio import PortfolioContext, Position

SUPPORTED_PAIRS = frozenset({"EURUSD", "GBPUSD", "AUDUSD", "NZDUSD"})
MAX_RISK_PERCENT = 1.0
STOP_ATR_MULTIPLE = 2.0


@dataclass(frozen=True)
class ForexSnapshot:
    symbol: str
    date: str
    open: float
    high: float
    low: float
    close: float
    atr: float


@dataclass(frozen=True)
class PaperCycleResult:
    action: str
    balance: float
    equity: float
    units: float = 0.0
    stop_price: float | None = None
    realized_pnl: float = 0.0
    duplicate: bool = False


@dataclass(frozen=True)
class PaperAccountSummary:
    initial_capital: float
    balance: float
    equity: float
    realized_pnl: float
    completed_trades: int
    winning_trades: int
    cycles: int


def validate_forex_pair(ticker: str) -> str:
    """Accept only supported USD-quoted spot pairs, returned in broker-neutral form."""
    if not isinstance(ticker, str) or not ticker.strip():
        raise ValueError(
            "Paper trading supports only these USD-quoted forex pairs: "
            "EURUSD, GBPUSD, AUDUSD, or NZDUSD."
        )
    canonical = normalize_symbol(ticker.strip())
    pair = canonical[:-2] if canonical.endswith("=X") else canonical
    if pair not in SUPPORTED_PAIRS:
        choices = ", ".join(sorted(SUPPORTED_PAIRS))
        raise ValueError(f"Paper trading supports only these USD-quoted forex pairs: {choices}.")
    return pair


def snapshot_from_frame(symbol: str, frame: pd.DataFrame) -> ForexSnapshot:
    """Read the newest usable daily bar and calculate a 14-bar ATR."""
    required = {"Date", "Open", "High", "Low", "Close"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Market data is missing required columns: {', '.join(sorted(missing))}.")
    data = frame.loc[:, ["Date", "Open", "High", "Low", "Close"]].copy()
    for column in ("Open", "High", "Low", "Close"):
        data[column] = pd.to_numeric(data[column], errors="coerce")
    data = data.dropna(subset=["Date", "Open", "High", "Low", "Close"])
    if len(data) < 14:
        raise ValueError(f"Not enough daily price history to calculate ATR for {symbol}.")

    previous_close = data["Close"].shift()
    true_range = pd.concat(
        [
            data["High"] - data["Low"],
            (data["High"] - previous_close).abs(),
            (data["Low"] - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    atr = float(true_range.rolling(14, min_periods=14).mean().iloc[-1])
    row = data.iloc[-1]
    prices = [float(row[column]) for column in ("Open", "High", "Low", "Close")]
    if not all(math.isfinite(price) and price > 0 for price in prices):
        raise ValueError(f"Market data contains an invalid price for {symbol}.")
    if not math.isfinite(atr) or atr <= 0:
        raise ValueError(f"Market data does not provide a usable ATR for {symbol}.")
    return ForexSnapshot(
        symbol=symbol,
        date=pd.Timestamp(row["Date"]).date().isoformat(),
        open=prices[0],
        high=prices[1],
        low=prices[2],
        close=prices[3],
        atr=atr,
    )


def get_market_bars(ticker: str, config: dict) -> pd.DataFrame:
    """Load current, non-filled daily bars and refuse stale market data."""
    pair = validate_forex_pair(ticker)
    today = pd.Timestamp.today().date().isoformat()
    with run_config(config):
        frame = load_ohlcv(pair, today, fill_gaps=False)
    if frame.empty:
        raise ValueError(f"No daily market data is available for {pair}.")
    snapshot = snapshot_from_frame(pair, frame)
    if (pd.Timestamp(today) - pd.Timestamp(snapshot.date)).days > 5:
        raise ValueError(f"The latest {pair} daily bar is stale ({snapshot.date}); refusing to trade.")
    return frame


def get_market_snapshot(ticker: str, config: dict) -> ForexSnapshot:
    """Load the newest usable daily bar from the configured Yahoo data cache."""
    pair = validate_forex_pair(ticker)
    return snapshot_from_frame(pair, get_market_bars(pair, config))


def calculate_units(equity: float, used_exposure: float, price: float, atr: float,
                    risk_percent: float) -> float:
    """Size a position to the risk budget, capped at 1x account equity exposure."""
    if not 0 < risk_percent <= MAX_RISK_PERCENT:
        raise ValueError(f"risk_percent must be greater than 0 and at most {MAX_RISK_PERCENT}.")
    if not all(math.isfinite(value) for value in (equity, used_exposure, price, atr)):
        raise ValueError("Position sizing inputs must be finite numbers.")
    if equity <= 0 or price <= 0 or atr <= 0:
        return 0.0
    risk_budget = equity * risk_percent / 100
    stop_distance = STOP_ATR_MULTIPLE * atr
    remaining_exposure = max(0.0, equity - used_exposure)
    return max(0.0, min(risk_budget / stop_distance, remaining_exposure / price))


class ForexPaperAccount:
    """Persistent SQLite ledger; it has no broker or live-order interface."""

    def __init__(self, path: str | Path, initial_capital: float = 10_000.0):
        if not math.isfinite(initial_capital) or initial_capital <= 0:
            raise ValueError("initial_capital must be a positive finite number.")
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS account (
                    id INTEGER PRIMARY KEY CHECK (id = 1),
                    initial_capital REAL NOT NULL,
                    balance REAL NOT NULL
                );
                CREATE TABLE IF NOT EXISTS positions (
                    symbol TEXT PRIMARY KEY,
                    units REAL NOT NULL,
                    entry_price REAL NOT NULL,
                    stop_price REAL NOT NULL,
                    mark_price REAL NOT NULL,
                    opened_on TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS cycles (
                    symbol TEXT NOT NULL,
                    cycle_date TEXT NOT NULL,
                    rating TEXT NOT NULL,
                    price REAL NOT NULL,
                    action TEXT NOT NULL,
                    PRIMARY KEY (symbol, cycle_date)
                );
                CREATE TABLE IF NOT EXISTS trades (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    cycle_date TEXT NOT NULL,
                    event TEXT NOT NULL,
                    units REAL NOT NULL,
                    price REAL NOT NULL,
                    realized_pnl REAL NOT NULL
                );
                """
            )
            connection.execute(
                "INSERT OR IGNORE INTO account (id, initial_capital, balance) VALUES (1, ?, ?)",
                (initial_capital, initial_capital),
            )

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.path, timeout=30)
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    def has_cycle(self, symbol: str, cycle_date: str) -> bool:
        with self._connect() as connection:
            return connection.execute(
                "SELECT 1 FROM cycles WHERE symbol = ? AND cycle_date = ?",
                (symbol, cycle_date),
            ).fetchone() is not None

    def portfolio_context(self) -> PortfolioContext:
        with self._connect() as connection:
            account = connection.execute("SELECT balance FROM account WHERE id = 1").fetchone()
            positions = connection.execute(
                "SELECT symbol, units, entry_price FROM positions ORDER BY symbol"
            ).fetchall()
        return PortfolioContext(
            cash=float(account[0]),
            currency="USD",
            positions=[
                Position(ticker=symbol, quantity=float(units), average_price=float(entry))
                for symbol, units, entry in positions
            ],
        )

    def summary(self) -> PaperAccountSummary:
        with self._connect() as connection:
            initial_capital, balance = connection.execute(
                "SELECT initial_capital, balance FROM account WHERE id = 1"
            ).fetchone()
            equity, _ = self._equity_and_exposure(connection, float(balance))
            realized_pnl, completed_trades, winning_trades = connection.execute(
                """
                SELECT COALESCE(SUM(realized_pnl), 0), COUNT(*),
                       SUM(CASE WHEN realized_pnl > 0 THEN 1 ELSE 0 END)
                FROM trades WHERE event LIKE 'close_%'
                """
            ).fetchone()
            cycles = connection.execute("SELECT COUNT(*) FROM cycles").fetchone()[0]
        return PaperAccountSummary(
            initial_capital=float(initial_capital),
            balance=float(balance),
            equity=float(equity),
            realized_pnl=float(realized_pnl),
            completed_trades=int(completed_trades),
            winning_trades=int(winning_trades or 0),
            cycles=int(cycles),
        )

    def apply_signal(
        self,
        snapshot: ForexSnapshot,
        rating: str,
        *,
        risk_percent: float = 0.5,
    ) -> PaperCycleResult:
        """Apply one daily rating, closing breached stops and recording simulated fills."""
        direction_by_rating = {
            "Buy": 1,
            "Overweight": 1,
            "Hold": 0,
            "Underweight": -1,
            "Sell": -1,
            "REVIEW": 0,
        }
        if rating not in direction_by_rating:
            raise ValueError(f"Unsupported TradingAgents rating: {rating!r}.")
        if not 0 < risk_percent <= MAX_RISK_PERCENT:
            raise ValueError(f"risk_percent must be greater than 0 and at most {MAX_RISK_PERCENT}.")

        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            if connection.execute(
                "SELECT 1 FROM cycles WHERE symbol = ? AND cycle_date = ?",
                (snapshot.symbol, snapshot.date),
            ).fetchone():
                balance, equity = self._account_values(connection)
                return PaperCycleResult("already processed", balance, equity, duplicate=True)

            balance = float(connection.execute("SELECT balance FROM account WHERE id = 1").fetchone()[0])
            row = connection.execute(
                "SELECT units, entry_price, stop_price FROM positions WHERE symbol = ?",
                (snapshot.symbol,),
            ).fetchone()
            realized = 0.0
            action = "no trade"
            units = 0.0
            stop_price = None
            direction = direction_by_rating[rating]

            if row is not None:
                current_units, entry_price, existing_stop = map(float, row)
                stop_hit = (
                    snapshot.low <= existing_stop
                    if current_units > 0
                    else snapshot.high >= existing_stop
                )
                if stop_hit:
                    fill_price = (
                        min(existing_stop, snapshot.open)
                        if current_units > 0
                        else max(existing_stop, snapshot.open)
                    )
                    realized = current_units * (fill_price - entry_price)
                    balance += realized
                    self._record_close(
                        connection, snapshot, current_units, fill_price, realized, "stop"
                    )
                    action = "stop-loss filled; no same-day re-entry"
                    direction = 0
                else:
                    connection.execute(
                        "UPDATE positions SET mark_price = ? WHERE symbol = ?",
                        (snapshot.close, snapshot.symbol),
                    )
                    if direction == 0 or (direction > 0) == (current_units > 0):
                        units = current_units
                        stop_price = existing_stop
                        action = "held long position" if current_units > 0 else "held short position"
                        direction = 0
                    else:
                        realized = current_units * (snapshot.close - entry_price)
                        balance += realized
                        self._record_close(
                            connection, snapshot, current_units, snapshot.close, realized, "signal"
                        )
                        action = "closed position"

            if direction:
                equity, used_exposure = self._equity_and_exposure(connection, balance)
                new_units = calculate_units(
                    equity, used_exposure, snapshot.close, snapshot.atr, risk_percent
                )
                if new_units > 0:
                    units = direction * new_units
                    stop_distance = STOP_ATR_MULTIPLE * snapshot.atr
                    stop_price = snapshot.close - direction * stop_distance
                    connection.execute(
                        """
                        INSERT INTO positions (symbol, units, entry_price, stop_price, mark_price, opened_on)
                        VALUES (?, ?, ?, ?, ?, ?)
                        ON CONFLICT(symbol) DO UPDATE SET
                            units = excluded.units,
                            entry_price = excluded.entry_price,
                            stop_price = excluded.stop_price,
                            mark_price = excluded.mark_price,
                            opened_on = excluded.opened_on
                        """,
                        (snapshot.symbol, units, snapshot.close, stop_price, snapshot.close, snapshot.date),
                    )
                    connection.execute(
                        """
                        INSERT INTO trades (symbol, cycle_date, event, units, price, realized_pnl)
                        VALUES (?, ?, ?, ?, ?, 0)
                        """,
                        (
                            snapshot.symbol,
                            snapshot.date,
                            "open_long" if direction > 0 else "open_short",
                            units,
                            snapshot.close,
                        ),
                    )
                    action = "opened long position" if direction > 0 else "opened short position"
                else:
                    action = "risk/exposure limit prevented entry"

            connection.execute("UPDATE account SET balance = ? WHERE id = 1", (balance,))
            connection.execute(
                "INSERT INTO cycles (symbol, cycle_date, rating, price, action) VALUES (?, ?, ?, ?, ?)",
                (snapshot.symbol, snapshot.date, rating, snapshot.close, action),
            )
            balance, equity = self._account_values(connection)
            if (action == "closed position" and direction == 0) or action.startswith("stop-loss"):
                units = 0.0
                stop_price = None
            return PaperCycleResult(action, balance, equity, units, stop_price, realized)

    @staticmethod
    def _record_close(connection, snapshot, units, price, realized, reason):
        connection.execute(
            "INSERT INTO trades (symbol, cycle_date, event, units, price, realized_pnl) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (
                snapshot.symbol,
                snapshot.date,
                f"close_{reason}",
                units,
                price,
                realized,
            ),
        )
        connection.execute("DELETE FROM positions WHERE symbol = ?", (snapshot.symbol,))

    @staticmethod
    def _equity_and_exposure(connection, balance):
        rows = connection.execute(
            "SELECT units, entry_price, mark_price FROM positions"
        ).fetchall()
        equity = balance + sum(units * (mark - entry) for units, entry, mark in rows)
        exposure = sum(abs(units * mark) for units, _entry, mark in rows)
        return equity, exposure

    @classmethod
    def _account_values(cls, connection):
        balance = float(connection.execute("SELECT balance FROM account WHERE id = 1").fetchone()[0])
        equity, _ = cls._equity_and_exposure(connection, balance)
        return balance, equity
