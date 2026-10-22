"""Strictly demo-only MetaTrader 5 market data and execution."""

from __future__ import annotations

import math
import os
import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from tradingagents.dataflows.symbols import safe_ticker_component
from tradingagents.forex_paper import ForexSnapshot, snapshot_from_frame

MAX_RISK_PERCENT = 1.0
MAX_LOT = 0.1
TAKE_PROFIT_R = 2.0
SERVER_UTC_OFFSET_STEP_SECONDS = 15 * 60
MAX_SERVER_UTC_OFFSET_SECONDS = 14 * 60 * 60
MAX_QUOTE_AGE_SECONDS = 120
MAX_QUOTE_CLOCK_AHEAD_SECONDS = 5
TIMEFRAMES = {
    "M15": ("TIMEFRAME_M15", 15 * 60),
    "H1": ("TIMEFRAME_H1", 60 * 60),
    "H4": ("TIMEFRAME_H4", 4 * 60 * 60),
    "D1": ("TIMEFRAME_D1", 24 * 60 * 60),
}
MT5_STRATEGIES = (
    "agents",
    "trend",
    "mean-reversion",
    "breakout",
    "tradingview",
    "gold-silver",
    "commodity-momentum",
)


SIDES = ("both", "long", "short")


def restrict_signal(signal: str | None, side: str) -> str | None:
    """The signal a strategy limited to one side acts on: the other direction becomes Hold."""
    if side not in SIDES:
        raise ValueError(f"Side must be one of: {', '.join(SIDES)}.")
    if (side == "long" and signal in ("Sell", "Underweight")) or (
        side == "short" and signal in ("Buy", "Overweight")
    ):
        return "Hold"
    return signal


def side_strategy(strategy: str, side: str) -> str:
    """The name a side-limited strategy is logged under, so its cycles never mix with both."""
    return strategy if side == "both" else f"{strategy}/{side}"


def parse_mt5_strategies(value: str) -> list[str]:
    strategies = [item.strip().lower() for item in value.split(",") if item.strip()]
    unknown = [item for item in strategies if item not in MT5_STRATEGIES]
    if not strategies:
        raise ValueError("Choose at least one MT5 demo strategy.")
    if unknown:
        raise ValueError(
            f"Unknown MT5 demo strategy {', '.join(unknown)}; "
            f"choose from {', '.join(MT5_STRATEGIES)}."
        )
    return list(dict.fromkeys(strategies))


@dataclass(frozen=True)
class MT5MarketData:
    symbol: str
    timeframe: str
    bar_time: int
    server_utc_offset_seconds: int
    bars: pd.DataFrame
    snapshot: ForexSnapshot
    equity: float
    balance: float
    currency: str


def _server_utc_offset_seconds(server_time: int, now: float) -> int:
    difference = server_time - now
    offset = round(difference / SERVER_UTC_OFFSET_STEP_SECONDS) * SERVER_UTC_OFFSET_STEP_SECONDS
    if abs(offset) > MAX_SERVER_UTC_OFFSET_SECONDS:
        # No broker clock is that far from UTC, so the last tick itself is old.
        raise RuntimeError(
            f"MT5 quote is stale: its time is {abs(difference) / 3600:.0f} h off the current time, "
            "beyond any broker timezone. If MT5 was just started or woke from sleep, let it "
            "reconnect and retry; otherwise the market may be closed. Refusing to continue."
        )
    quote_age = now - (server_time - offset)
    if quote_age < -MAX_QUOTE_CLOCK_AHEAD_SECONDS or quote_age > MAX_QUOTE_AGE_SECONDS:
        raise RuntimeError(
            "MT5 quote is stale or its server clock offset cannot be verified; refusing to continue."
        )
    return offset


@dataclass(frozen=True)
class MT5TradeResult:
    action: str
    signal: str
    volume: float = 0.0
    stop_price: float | None = None
    take_profit: float | None = None
    risk_amount: float = 0.0
    ticket: int | None = None


class MT5DemoSession:
    """A terminal session that refuses every non-demo account."""

    def __init__(self, terminal_path: str | None = None, mt5_module=None):
        self.terminal_path = terminal_path or os.getenv("TRADINGAGENTS_MT5_TERMINAL")
        self.mt5 = mt5_module
        self.connected = False

    def __enter__(self):
        if self.mt5 is None:
            try:
                import MetaTrader5 as mt5
            except ImportError as exc:
                raise RuntimeError(
                    "MetaTrader5 is not installed. Install with "
                    "`pip install -e .[mt5]` in the Windows environment running MT5."
                ) from exc
            self.mt5 = mt5

        initialized = (
            self.mt5.initialize(path=self.terminal_path)
            if self.terminal_path
            else self.mt5.initialize()
        )
        if not initialized:
            raise RuntimeError(
                f"Could not connect to MetaTrader 5: {self.mt5.last_error()}. "
                "Open the terminal and sign in to a demo account first."
            )
        self.connected = True
        try:
            self._require_demo_account()
        except Exception:
            self.close()
            raise
        return self

    def __exit__(self, _exc_type, _exc_value, _traceback):
        self.close()

    def close(self):
        if self.connected:
            self.mt5.shutdown()
            self.connected = False

    def _require_demo_account(self):
        terminal = self.mt5.terminal_info()
        account = self.mt5.account_info()
        if terminal is None or account is None:
            raise RuntimeError(
                f"MetaTrader 5 did not return terminal/account details: {self.mt5.last_error()}."
            )
        demo_mode = getattr(self.mt5, "ACCOUNT_TRADE_MODE_DEMO", None)
        if demo_mode is None:
            raise RuntimeError(
                "MetaTrader 5 does not expose a demo-account mode; refusing to continue."
            )
        if account.trade_mode != demo_mode:
            raise RuntimeError(
                "This command only permits MT5 demo accounts. The connected account is not demo; "
                "no orders were sent."
            )
        if not terminal.connected:
            raise RuntimeError("MetaTrader 5 terminal is disconnected.")
        return terminal, account

    def account_details(self):
        return self._require_demo_account()[1]

    def trading_enabled(self) -> bool:
        """Whether MT5 currently accepts automated orders (the terminal's Algo Trading button)."""
        terminal, account = self._require_demo_account()
        return bool(terminal.trade_allowed and account.trade_allowed)

    def market_watch_symbols(self, limit: int = 300) -> list[str]:
        """Names of the symbols currently shown in the terminal's Market Watch."""
        symbols = self.mt5.symbols_get() or ()
        return sorted(symbol.name for symbol in symbols if symbol.visible)[:limit]

    def has_symbol(self, symbol: str) -> bool:
        """Whether the broker offers this symbol, shown in Market Watch or not."""
        return self.mt5.symbol_info(symbol) is not None

    def daily_closes(self, symbols: list[str], count: int) -> pd.DataFrame:
        """Closes of the last ``count`` closed daily bars, one column per symbol.

        Rows are the broker's trading days that every symbol has; all symbols share the
        server clock, so their bar times line up without timezone conversion.
        """
        self._require_demo_account()
        columns = {}
        for symbol in symbols:
            rates = self.mt5.copy_rates_from_pos(symbol, self.mt5.TIMEFRAME_D1, 1, count)
            if rates is None or len(rates) == 0:
                raise RuntimeError(
                    f"MT5 returned no daily bars for {symbol}: {self.mt5.last_error()}."
                )
            frame = pd.DataFrame(rates)
            columns[symbol] = pd.Series(frame["close"].to_numpy(), index=frame["time"].to_numpy())
        return pd.DataFrame(columns).dropna().sort_index()

    def history_bars(self, symbol: str, timeframe: str, count: int) -> pd.DataFrame:
        """The last ``count`` closed bars on the server clock, each with its spread in price units.

        Unlike market_data() it needs no fresh quote, so backtests also run while the market is
        closed. Bars the broker stored without a spread get the median recorded spread.
        """
        if timeframe not in TIMEFRAMES:
            raise ValueError(f"Unsupported timeframe {timeframe!r}; use {', '.join(TIMEFRAMES)}.")
        self._require_demo_account()
        info = self.mt5.symbol_info(symbol)
        if info is None:
            raise ValueError(
                f"MT5 symbol {symbol!r} was not found. Use its exact broker symbol from Market Watch."
            )
        timeframe_constant = getattr(self.mt5, TIMEFRAMES[timeframe][0])
        rates = self.mt5.copy_rates_from_pos(symbol, timeframe_constant, 1, count)
        if rates is None or len(rates) == 0:
            raise RuntimeError(
                f"MT5 returned no {timeframe} history for {symbol}: {self.mt5.last_error()}."
            )
        bars = pd.DataFrame(rates).rename(
            columns={"open": "Open", "high": "High", "low": "Low", "close": "Close"}
        )
        points = bars["spread"].astype(float)
        typical = points[points > 0].median() if (points > 0).any() else float(info.spread)
        bars["spread"] = points.where(points > 0, typical) * float(info.point)
        return bars[["time", "Open", "High", "Low", "Close", "spread"]].reset_index(drop=True)

    def market_data(
        self,
        symbol: str,
        timeframe: str = "H1",
        bars_count: int = 250,
    ) -> MT5MarketData:
        if timeframe not in TIMEFRAMES:
            raise ValueError(f"Unsupported timeframe {timeframe!r}; use {', '.join(TIMEFRAMES)}.")
        if not symbol or not symbol.strip():
            raise ValueError("Provide the exact MT5 Market Watch symbol, e.g. XAUUSD or EURUSD.a.")
        try:
            safe_ticker_component(symbol)
        except ValueError as exc:
            raise ValueError(f"Invalid MT5 symbol: {exc}") from exc

        self._require_demo_account()
        info = self.mt5.symbol_info(symbol)
        if info is None:
            raise ValueError(
                f"MT5 symbol {symbol!r} was not found. Use its exact broker symbol from Market Watch."
            )
        if not info.visible and not self.mt5.symbol_select(symbol, True):
            raise RuntimeError(
                f"Could not enable {symbol!r} in Market Watch: {self.mt5.last_error()}."
            )

        tick = self.mt5.symbol_info_tick(symbol)
        if tick is None:
            raise RuntimeError(f"MT5 quote unavailable for {symbol}: {self.mt5.last_error()}.")
        now = datetime.now(UTC).timestamp()
        server_utc_offset = _server_utc_offset_seconds(int(tick.time), now)

        timeframe_constant = getattr(self.mt5, TIMEFRAMES[timeframe][0])
        rates = self.mt5.copy_rates_from_pos(symbol, timeframe_constant, 1, bars_count)
        if rates is None or len(rates) < 50:
            raise RuntimeError(
                f"MT5 returned fewer than 50 closed {timeframe} bars for {symbol}: "
                f"{self.mt5.last_error()}."
            )
        bars = pd.DataFrame(rates)
        bars["time"] = bars["time"] - server_utc_offset
        bars["Date"] = pd.to_datetime(bars["time"], unit="s", utc=True)
        bars = bars.rename(columns={"open": "Open", "high": "High", "low": "Low", "close": "Close"})
        snapshot = snapshot_from_frame(symbol, bars)
        bar_time = int(bars.iloc[-1]["time"])
        age = datetime.now(UTC).timestamp() - bar_time
        if age < 0 or age > TIMEFRAMES[timeframe][1] * 3:
            raise RuntimeError(
                f"The last closed {timeframe} bar for {symbol} is stale; refusing to trade."
            )

        account = self._require_demo_account()[1]
        if not math.isfinite(float(account.equity)) or account.equity <= 0:
            raise RuntimeError("MT5 demo account equity must be positive and finite.")
        return MT5MarketData(
            symbol=symbol,
            timeframe=timeframe,
            bar_time=bar_time,
            server_utc_offset_seconds=server_utc_offset,
            bars=bars,
            snapshot=snapshot,
            equity=float(account.equity),
            balance=float(account.balance),
            currency=str(account.currency),
        )

    def evaluate_signal(
        self,
        market: MT5MarketData,
        signal: str,
        *,
        strategy: str = "agents",
        risk_percent: float = 0.25,
        max_lot: float = 0.1,
        execute: bool = False,
        state_path: str | Path = "mt5-demo.sqlite3",
        magic: int = 20711020,
        side: str = "both",
    ) -> MT5TradeResult:
        terminal, account = self._require_demo_account()
        if execute and (not terminal.trade_allowed or not account.trade_allowed):
            raise RuntimeError("Automated trading is disabled in the terminal or demo account.")
        if signal not in {"Buy", "Overweight", "Hold", "Underweight", "Sell", "REVIEW"}:
            raise ValueError(f"Unsupported strategy signal: {signal!r}.")
        if strategy not in MT5_STRATEGIES:
            raise ValueError(f"Unsupported MT5 demo strategy: {strategy!r}.")
        # From here on a one-sided strategy only sees its own direction and is logged apart.
        signal = restrict_signal(signal, side)
        strategy = side_strategy(strategy, side)
        if not math.isfinite(risk_percent) or not 0 < risk_percent <= MAX_RISK_PERCENT:
            raise ValueError(f"Risk must be greater than 0 and at most {MAX_RISK_PERCENT}%.")
        if not math.isfinite(max_lot) or not 0 < max_lot <= MAX_LOT:
            raise ValueError(f"Maximum volume must be greater than 0 and at most {MAX_LOT} lots.")

        direction = {
            "Buy": 1,
            "Overweight": 1,
            "Hold": 0,
            "Underweight": -1,
            "Sell": -1,
            "REVIEW": 0,
        }[signal]
        info = self.mt5.symbol_info(market.symbol)
        tick = self.mt5.symbol_info_tick(market.symbol)
        if info is None or tick is None:
            raise RuntimeError(f"MT5 quote or symbol metadata unavailable for {market.symbol}.")
        quote_age = datetime.now(UTC).timestamp() - tick.time + market.server_utc_offset_seconds
        if quote_age < -MAX_QUOTE_CLOCK_AHEAD_SECONDS or quote_age > MAX_QUOTE_AGE_SECONDS:
            raise RuntimeError(f"MT5 quote for {market.symbol} is stale; refusing to trade.")

        ledger = MT5DemoLedger(state_path)
        try:
            existing = ledger.get_cycle(market.symbol, market.timeframe, market.bar_time, strategy)
        finally:
            ledger.close()
        if existing is not None:
            if execute and existing["status"] == "signal_only" and existing["signal"] == signal:
                pass
            elif execute and existing["status"] == "signal_only":
                return MT5TradeResult(
                    "signal changed since preview; review it on the next bar", signal
                )
            else:
                return MT5TradeResult("cycle already processed; no duplicate order", signal)

        order_type = self.mt5.ORDER_TYPE_BUY if direction > 0 else self.mt5.ORDER_TYPE_SELL
        entry = float(tick.ask if direction > 0 else tick.bid)
        stop_distance = 2 * market.snapshot.atr
        digits = int(info.digits)
        stop = round(entry - direction * stop_distance, digits) if direction else None
        take_profit = (
            round(entry + direction * stop_distance * TAKE_PROFIT_R, digits) if direction else None
        )
        if direction:
            min_stop_distance = float(info.trade_stops_level) * float(info.point)
            if (
                stop_distance < min_stop_distance
                or (direction > 0 and stop >= tick.bid)
                or (direction < 0 and stop <= tick.ask)
            ):
                raise RuntimeError(
                    f"The 2-ATR stop for {market.symbol} is inside the broker's minimum stop distance."
                )

        positions = self.mt5.positions_get()
        if positions is None:
            raise RuntimeError(f"Could not inspect open positions: {self.mt5.last_error()}.")
        if positions:
            result = MT5TradeResult("an account position is already open; no new order", signal)
            return self._record_cycle(market, strategy, result, execute, state_path)

        pending_orders = self.mt5.orders_get()
        if pending_orders is None:
            raise RuntimeError(f"Could not inspect pending orders: {self.mt5.last_error()}.")
        if pending_orders:
            result = MT5TradeResult("an account order is already pending; no new order", signal)
            return self._record_cycle(market, strategy, result, execute, state_path)

        if not direction:
            result = MT5TradeResult("no order (Hold/REVIEW)", signal)
            return self._record_cycle(market, strategy, result, execute, state_path)

        loss_per_lot = self.mt5.order_calc_profit(order_type, market.symbol, 1.0, entry, stop)
        if loss_per_lot is None or not math.isfinite(loss_per_lot) or loss_per_lot >= 0:
            raise RuntimeError(
                f"MT5 could not estimate stop-loss risk for {market.symbol}: {self.mt5.last_error()}."
            )
        risk_budget = market.equity * risk_percent / 100
        step = float(info.volume_step)
        if step <= 0:
            raise RuntimeError(f"MT5 returned an invalid volume step for {market.symbol}.")
        requested = min(max_lot, float(info.volume_max), risk_budget / abs(loss_per_lot))
        volume = math.floor((requested + step * 1e-9) / step) * step
        volume = round(volume, 8)
        if volume < float(info.volume_min):
            raise RuntimeError(
                f"Minimum broker volume for {market.symbol} exceeds the {risk_percent}% risk budget."
            )
        risk_amount = volume * abs(loss_per_lot)
        result = MT5TradeResult(
            "ready to send demo order" if execute else "signal only; no order sent",
            signal=signal,
            volume=volume,
            stop_price=stop,
            take_profit=take_profit,
            risk_amount=risk_amount,
        )
        if not execute:
            return self._record_cycle(market, strategy, result, False, state_path)

        ledger = MT5DemoLedger(state_path)
        try:
            if ledger.has_trade_today(market.symbol, market.bar_time):
                result = MT5TradeResult("daily entry limit reached; no new order", signal)
                if existing is None:
                    ledger.record(market, strategy, result, "skipped")
                else:
                    ledger.update(market, strategy, "skipped", None)
                return result
            if existing is None:
                ledger.record(market, strategy, result, "pending")
            else:
                ledger.update(market, strategy, "pending", None)
            request = {
                "action": self.mt5.TRADE_ACTION_DEAL,
                "symbol": market.symbol,
                "volume": volume,
                "type": order_type,
                "price": entry,
                "sl": stop,
                "tp": take_profit,
                "deviation": 20,
                "magic": magic,
                "comment": "TradingAgents demo",
                "type_time": self.mt5.ORDER_TIME_GTC,
                "type_filling": self.mt5.ORDER_FILLING_IOC,
            }
            check = self.mt5.order_check(request)
            if check is None or check.retcode != 0:
                ledger.update(market, strategy, "rejected", None)
                detail = check.comment if check is not None else self.mt5.last_error()
                raise RuntimeError(f"MT5 order check rejected the demo order: {detail}.")
            sent = self.mt5.order_send(request)
            accepted = {
                self.mt5.TRADE_RETCODE_DONE,
                self.mt5.TRADE_RETCODE_PLACED,
            }
            if sent is None or sent.retcode not in accepted:
                ledger.update(market, strategy, "uncertain", None)
                detail = sent.comment if sent is not None else self.mt5.last_error()
                raise RuntimeError(
                    "MT5 did not confirm the order result; the bar is locked against retries. "
                    f"Check the demo account history before continuing: {detail}."
                )
            ledger.update(market, strategy, "filled", int(sent.order))
            return MT5TradeResult(
                "demo order accepted",
                signal=signal,
                volume=volume,
                stop_price=stop,
                take_profit=take_profit,
                risk_amount=risk_amount,
                ticket=int(sent.order),
            )
        finally:
            ledger.close()

    @staticmethod
    def _record_cycle(market, strategy, result, execute, state_path):
        ledger = MT5DemoLedger(state_path)
        try:
            existing = ledger.get_cycle(market.symbol, market.timeframe, market.bar_time, strategy)
            if existing is not None:
                return MT5TradeResult("cycle already processed; no duplicate order", result.signal)
            ledger.record(market, strategy, result, "skipped" if execute else "signal_only")
        finally:
            ledger.close()
        return result


class MT5DemoLedger:
    """Persistent idempotency and once-per-day entry guard for demo orders.

    It also keeps the close of every recorded bar, so signals can be checked afterwards.
    """

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(self.path, timeout=30)
        self.connection.execute(
            """
            CREATE TABLE IF NOT EXISTS cycles (
                symbol TEXT NOT NULL,
                timeframe TEXT NOT NULL,
                bar_time INTEGER NOT NULL,
                strategy TEXT NOT NULL,
                signal TEXT NOT NULL,
                status TEXT NOT NULL,
                ticket INTEGER,
                PRIMARY KEY (symbol, timeframe, bar_time, strategy)
            )
            """
        )
        columns = {
            row[1] for row in self.connection.execute("PRAGMA table_info(cycles)").fetchall()
        }
        if "strategy" not in columns:
            self.connection.execute("ALTER TABLE cycles RENAME TO cycles_legacy")
            self.connection.execute(
                """
                CREATE TABLE cycles (
                    symbol TEXT NOT NULL,
                    timeframe TEXT NOT NULL,
                    bar_time INTEGER NOT NULL,
                    strategy TEXT NOT NULL,
                    signal TEXT NOT NULL,
                    status TEXT NOT NULL,
                    ticket INTEGER,
                    PRIMARY KEY (symbol, timeframe, bar_time, strategy)
                )
                """
            )
            self.connection.execute(
                "INSERT INTO cycles "
                "SELECT symbol, timeframe, bar_time, 'agents', signal, status, ticket "
                "FROM cycles_legacy"
            )
            self.connection.execute("DROP TABLE cycles_legacy")
        # A separate table, so processes still writing `cycles` with older code keep working.
        self.connection.execute(
            """
            CREATE TABLE IF NOT EXISTS bars (
                symbol TEXT NOT NULL,
                timeframe TEXT NOT NULL,
                bar_time INTEGER NOT NULL,
                close REAL NOT NULL,
                PRIMARY KEY (symbol, timeframe, bar_time)
            )
            """
        )
        self.connection.commit()

    def get_cycle(self, symbol: str, timeframe: str, bar_time: int, strategy: str):
        row = self.connection.execute(
            "SELECT signal, status FROM cycles "
            "WHERE symbol = ? AND timeframe = ? AND bar_time = ? AND strategy = ?",
            (symbol, timeframe, bar_time, strategy),
        ).fetchone()
        return {"signal": row[0], "status": row[1]} if row else None

    def has_cycle(self, symbol: str, timeframe: str, bar_time: int, strategy: str) -> bool:
        return self.get_cycle(symbol, timeframe, bar_time, strategy) is not None

    def summary(self):
        return self.connection.execute(
            """
            SELECT symbol, timeframe, strategy, signal, status, COUNT(*)
            FROM cycles
            GROUP BY symbol, timeframe, strategy, signal, status
            ORDER BY symbol, timeframe, strategy, signal, status
            """
        ).fetchall()

    def recent_cycles(self, limit: int = 30):
        if not 1 <= limit <= 100:
            raise ValueError("Cycle report limit must be between 1 and 100.")
        return self.connection.execute(
            """
            SELECT symbol, timeframe, strategy, signal, status, bar_time, ticket
            FROM cycles
            ORDER BY bar_time DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()

    def record(self, market: MT5MarketData, strategy: str, result: MT5TradeResult, status: str):
        self.connection.execute(
            "INSERT INTO cycles VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                market.symbol,
                market.timeframe,
                market.bar_time,
                strategy,
                result.signal,
                status,
                result.ticket,
            ),
        )
        self.connection.execute(
            "INSERT OR IGNORE INTO bars VALUES (?, ?, ?, ?)",
            (market.symbol, market.timeframe, market.bar_time, float(market.snapshot.close)),
        )
        self.connection.commit()

    def has_trade_today(self, symbol: str, bar_time: int) -> bool:
        start_of_day = (
            datetime.fromtimestamp(bar_time, UTC)
            .replace(hour=0, minute=0, second=0, microsecond=0)
            .timestamp()
        )
        return (
            self.connection.execute(
                "SELECT 1 FROM cycles WHERE symbol = ? AND bar_time >= ? AND status = 'filled'",
                (symbol, int(start_of_day)),
            ).fetchone()
            is not None
        )

    def update(self, market: MT5MarketData, strategy: str, status: str, ticket: int | None):
        self.connection.execute(
            "UPDATE cycles SET status = ?, ticket = ? "
            "WHERE symbol = ? AND timeframe = ? AND bar_time = ? AND strategy = ?",
            (status, ticket, market.symbol, market.timeframe, market.bar_time, strategy),
        )
        self.connection.commit()

    def close(self):
        self.connection.close()
