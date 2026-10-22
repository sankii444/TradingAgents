"""Replay MT5 history through the bot's strategies with the bot's own position rules.

Each strategy trades alone, as if on its own account: one position at a time and at most one
entry a day. A signal is read from a closed bar, as the live bot reads it, and filled at the
next bar's open; buys pay the ask (bid plus the spread MT5 recorded for that bar). Every
position gets the bot's stop two ATRs away and a target at twice that distance, and only
those close it. Levels are checked against bid prices for longs and ask prices for shorts;
when one bar reaches both, the stop is assumed to come first, and a gap through a level
fills at the bar's open. Each trade risks a fixed share of the current balance, so its
result is its R multiple times that amount; lot rounding, the bot's 0.1-lot cap, swaps and
commissions are left out.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from tradingagents.commodity_strategies import (
    COMMODITY_STRATEGIES,
    EXTRA_DAYS,
    commodity_signal,
    history_days,
    strategy_symbols,
)
from tradingagents.forex_paper import STOP_ATR_MULTIPLE, snapshot_from_frame
from tradingagents.mt5_demo import TAKE_PROFIT_R, TIMEFRAMES
from tradingagents.paper_strategies import signal_for_strategy

BAR_STRATEGIES = ("trend", "mean-reversion", "breakout")
BACKTEST_STRATEGIES = BAR_STRATEGIES + COMMODITY_STRATEGIES
UNSUPPORTED_STRATEGIES = {
    "tradingview": "TradingView does not publish the history of its rating, so it cannot be replayed",
    "agents": "replaying it would take one paid language-model analysis per bar",
}
WARMUP_BARS = 250  # the live bot reads the last 250 closed bars
BARS_PER_YEAR = {"M15": 24_000, "H1": 6_000, "H4": 1_550, "D1": 262}  # gold trades ~23h, 5 days
DAYS_PER_YEAR = 262
DIRECTIONS = {"Buy": 1, "Overweight": 1, "Sell": -1, "Underweight": -1}


@dataclass(frozen=True)
class Trade:
    direction: int
    entry_time: int
    entry: float
    stop: float
    take_profit: float
    exit_time: int
    exit: float
    reason: str  # "SL", "TP", or "end" for a position still open when the history ends
    r_multiple: float
    profit: float


@dataclass(frozen=True)
class BacktestReport:
    symbol: str
    timeframe: str
    first_time: int  # server clock, like MT5 bar times
    last_time: int
    bar_count: int
    capital: float
    currency: str
    risk_percent: float
    buy_and_hold: float
    trades: dict[str, list[Trade]]


def check_strategies(strategies: list[str]) -> None:
    """Reject strategies that cannot be replayed, saying why."""
    problems = [
        f"{name}: {UNSUPPORTED_STRATEGIES.get(name, 'not a replayable strategy')}"
        for name in strategies
        if name not in BACKTEST_STRATEGIES
    ]
    if problems:
        raise ValueError(
            "Cannot backtest "
            + "; ".join(problems)
            + f". Choose from {', '.join(BACKTEST_STRATEGIES)}."
        )


def _exit(direction, stop, target, open_, high, low, spread):
    """The (price, reason) at which a position leaves this bar, or None if it stays open."""
    if direction < 0:  # a short is bought back at the ask
        open_, high, low = open_ + spread, high + spread, low + spread
        if open_ >= stop:
            return open_, "SL"
        if open_ <= target:
            return open_, "TP"
        if high >= stop:
            return stop, "SL"
        return (target, "TP") if low <= target else None
    if open_ <= stop:
        return open_, "SL"
    if open_ >= target:
        return open_, "TP"
    if low <= stop:
        return stop, "SL"
    return (target, "TP") if high >= target else None


def simulate(
    bars: pd.DataFrame,
    signal_at: Callable[[int], str | None],
    atr_at: Callable[[int], float],
    *,
    capital: float,
    risk_percent: float,
    start: int = WARMUP_BARS,
) -> list[Trade]:
    """Trade the signals over ``bars`` (time, Open, High, Low, Close, spread) by the bot's rules.

    ``signal_at(i)`` and ``atr_at(i)`` may only use bars up to and including ``i``.
    """
    times = bars["time"].to_numpy(dtype=np.int64)
    opens, highs, lows, closes, spreads = (
        bars[column].to_numpy(dtype=float) for column in ("Open", "High", "Low", "Close", "spread")
    )
    trades: list[Trade] = []
    equity = capital
    position = None
    last_entry_day = None
    i = start
    while i < len(bars):
        if position is None:
            direction = DIRECTIONS.get(signal_at(i), 0) if i + 1 < len(bars) else 0
            day = int(times[i]) // 86_400
            if not direction or day == last_entry_day:
                i += 1
                continue
            distance = STOP_ATR_MULTIPLE * atr_at(i)
            i += 1
            entry = opens[i] + (spreads[i] if direction > 0 else 0.0)
            position = (direction, int(times[i]), entry, distance, equity * risk_percent / 100)
            last_entry_day = day
        direction, entry_time, entry, distance, risk_amount = position
        stop = entry - direction * distance
        target = entry + direction * distance * TAKE_PROFIT_R
        filled = _exit(direction, stop, target, opens[i], highs[i], lows[i], spreads[i])
        if filled is None and i == len(bars) - 1:
            filled = (closes[i] + (spreads[i] if direction < 0 else 0.0), "end")
        if filled is None:
            i += 1
            continue
        price, reason = filled
        r_multiple = direction * (price - entry) / distance
        trades.append(
            Trade(
                direction,
                entry_time,
                entry,
                stop,
                target,
                int(times[i]),
                price,
                reason,
                r_multiple,
                r_multiple * risk_amount,
            )
        )
        equity += trades[-1].profit
        position = None
        if reason == "end":
            break
        # The bot finds the position closed at the next hour and reads bar i's signal again.
    return trades


def bar_signals(bars: pd.DataFrame, strategy: str) -> Callable[[int], str]:
    """A bar strategy fed the same last 250 closed bars the live bot would read."""
    frame = bars[["Open", "High", "Low", "Close"]]

    def signal_at(i: int) -> str:
        return signal_for_strategy(strategy, frame.iloc[max(0, i - WARMUP_BARS + 1) : i + 1])

    return signal_at


def daily_signals(
    bars: pd.DataFrame,
    timeframe: str,
    closes: pd.DataFrame,
    symbol: str,
    strategy: str,
) -> Callable[[int], str | None]:
    """A daily strategy's signal from the last daily bar already closed when bar i closes."""
    window = history_days(strategy) - EXTRA_DAYS
    available = closes.index.to_numpy(dtype=np.int64) + 86_400  # closes at the next midnight
    bar_close = bars["time"].to_numpy(dtype=np.int64) + TIMEFRAMES[timeframe][1]
    latest = np.searchsorted(available, bar_close, side="right") - 1
    signals: dict[int, str | None] = {}
    for day in sorted(set(latest[latest >= window - 1].tolist())):
        try:
            signals[day] = commodity_signal(
                strategy, closes.iloc[day - window + 1 : day + 1], symbol
            )[0]
        except ValueError:  # e.g. a flat stretch with no correlation to measure
            signals[day] = None

    def signal_at(i: int) -> str | None:
        return signals.get(int(latest[i]))

    return signal_at


def run_mt5_backtest(
    session,
    symbol: str,
    timeframe: str,
    strategies: list[str],
    *,
    years: float,
    risk_percent: float,
    capital: float | None = None,
) -> BacktestReport:
    """Replay about ``years`` of the session's history through each strategy."""
    check_strategies(strategies)
    bars = session.history_bars(
        symbol, timeframe, int(years * BARS_PER_YEAR[timeframe]) + WARMUP_BARS
    )
    if len(bars) < WARMUP_BARS + 2:
        raise ValueError(f"MT5 returned only {len(bars)} {timeframe} bars for {symbol}.")
    account = session.account_details()
    capital = float(capital or account.equity)
    frame = bars.assign(Date=pd.to_datetime(bars["time"], unit="s"))

    def atr_at(i: int) -> float:
        return snapshot_from_frame(symbol, frame.iloc[max(0, i - 30) : i + 1]).atr

    trades = {}
    for strategy in strategies:
        if strategy in COMMODITY_STRATEGIES:
            days = int(years * DAYS_PER_YEAR) + history_days(strategy) + 30
            closes = session.daily_closes(strategy_symbols(session, symbol, strategy), days)
            signal_at = daily_signals(bars, timeframe, closes, symbol, strategy)
        else:
            signal_at = bar_signals(bars, strategy)
        trades[strategy] = simulate(
            bars, signal_at, atr_at, capital=capital, risk_percent=risk_percent
        )
    return BacktestReport(
        symbol=symbol,
        timeframe=timeframe,
        first_time=int(bars["time"].iloc[WARMUP_BARS]),
        last_time=int(bars["time"].iloc[-1]),
        bar_count=len(bars) - WARMUP_BARS,
        capital=capital,
        currency=str(account.currency),
        risk_percent=risk_percent,
        buy_and_hold=float(bars["Close"].iloc[-1] / bars["Open"].iloc[WARMUP_BARS] - 1),
        trades=trades,
    )


def summarize(trades: list[Trade], capital: float) -> dict:
    """Trade count, win rate, R, profit, profit factor, drawdown and profit per exit year."""
    equity, peak, drawdown = capital, capital, 0.0
    by_year: dict[int, float] = {}
    for trade in trades:
        equity += trade.profit
        peak = max(peak, equity)
        drawdown = min(drawdown, equity / peak - 1)
        year = datetime.fromtimestamp(trade.exit_time, UTC).year
        by_year[year] = by_year.get(year, 0.0) + trade.profit
    gains = sum(trade.profit for trade in trades if trade.profit > 0)
    losses = -sum(trade.profit for trade in trades if trade.profit < 0)
    count = len(trades)
    return {
        "trades": count,
        "win_rate": sum(trade.r_multiple > 0 for trade in trades) / count if count else None,
        "average_r": sum(trade.r_multiple for trade in trades) / count if count else None,
        "total_r": sum(trade.r_multiple for trade in trades),
        "profit": equity - capital,
        "return": equity / capital - 1,
        "profit_factor": gains / losses if losses else (math.inf if gains else None),
        "max_drawdown": drawdown,
        "by_year": by_year,
    }


def _amount(value: float, *, signed: bool = True, digits: int = 2) -> str:
    text = f"{value:{'+' if signed else ''},.{digits}f}"
    return text.replace(",", " ").replace(".", ",")


def _percent(fraction: float | None, *, signed: bool = True) -> str:
    if fraction is None:
        return "—"
    return f"{fraction * 100:{'+' if signed else ''}.1f}%".replace(".", ",")


def _server_time(timestamp: int, pattern: str = "%Y-%m-%d %H:%M") -> str:
    return datetime.fromtimestamp(timestamp, UTC).strftime(pattern)


def render_backtest_note(report: BacktestReport, *, now: datetime, summary_note: str) -> str:
    """The Markdown note (Polish, like the rest of the journal) for a backtest report."""
    stats = {name: summarize(trades, report.capital) for name, trades in report.trades.items()}
    ranked = sorted(stats, key=lambda name: stats[name]["profit"], reverse=True)
    currency = report.currency
    risk = f"{report.risk_percent:g}".replace(".", ",")
    rows = []
    for name in ranked:
        result = stats[name]
        factor = result["profit_factor"]
        rows.append(
            "| "
            + " | ".join(
                [
                    name,
                    str(result["trades"]),
                    _percent(result["win_rate"], signed=False),
                    "—" if result["average_r"] is None else _amount(result["average_r"]),
                    _amount(result["total_r"], digits=1),
                    "—"
                    if factor is None
                    else "∞"
                    if math.isinf(factor)
                    else _amount(factor, signed=False),
                    f"{_amount(result['profit'])} {currency}",
                    _percent(result["return"]),
                    _percent(result["max_drawdown"]),
                ]
            )
            + " |"
        )
    years = sorted({year for result in stats.values() for year in result["by_year"]})
    year_rows = [
        f"| {year} | "
        + " | ".join(_amount(stats[name]["by_year"].get(year, 0.0)) for name in ranked)
        + " |"
        for year in years
    ]
    lines = [
        "---",
        "tags:",
        "  - bot-mt5",
        "  - backtest",
        "---",
        f"# Backtest {report.symbol} · {report.timeframe}",
        "",
        f"> Generowane przez `tradingagents mt5-backtest` {now:%d.%m.%Y %H:%M}; "
        "kolejne uruchomienie nadpisze tę notatkę.",
        f"> Wróć do [[{summary_note}]].",
        "",
        f"- **Okres:** {_server_time(report.first_time, '%d.%m.%Y')} – "
        f"{_server_time(report.last_time, '%d.%m.%Y')} "
        f"({report.bar_count} świec {report.timeframe}, czas serwera brokera)",
        f"- **Kapitał startowy:** {_amount(report.capital, signed=False)} {currency} · "
        f"**ryzyko:** {risk}% kapitału na transakcję",
        f"- **Złoto w tym okresie (kup i trzymaj, bez dźwigni):** {_percent(report.buy_and_hold)}",
        "",
        "## Wyniki",
        "",
        "| Strategia | Transakcje | Wygrane | Śr. R | Suma R | Profit factor | Wynik | Wynik % "
        "| Maks. obsunięcie |",
        "|" + " --- |" * 9,
        *rows,
        "",
        "R to wynik transakcji w jednostkach ryzyka: −1 to stop loss, +2 to take profit. "
        "Profit factor to zyski podzielone przez straty; powyżej 1 strategia zarabia.",
        "",
        f"## Wyniki według lat ({currency})",
        "",
        "| Rok | " + " | ".join(ranked) + " |",
        "|" + " --- |" * (len(ranked) + 1),
        *year_rows,
        "",
        "## Zasady symulacji",
        "",
        "- Sygnał z zamkniętej świecy, wejście po cenie otwarcia następnej; kupno po cenie ask "
        "(bid + spread zapisany przez MT5 dla tej świecy).",
        f"- Stop loss {_amount(STOP_ATR_MULTIPLE, signed=False, digits=0)}×ATR(14), take profit "
        f"{_amount(TAKE_PROFIT_R, signed=False, digits=0)}R; pozycję zamyka tylko jeden z nich, "
        "sprawdzany na cenie bid (długie) albo ask (krótkie).",
        "- Gdy jedna świeca dotyka obu poziomów, liczy się stop loss; luka cenowa zamyka po cenie "
        "otwarcia.",
        "- Każda strategia osobno: jedna pozycja naraz i najwyżej jedno wejście dziennie, "
        "jak w bocie.",
        "- Stałe ryzyko od bieżącego kapitału; bez zaokrąglania wolumenu, limitu 0,1 lota, "
        "swapów i prowizji.",
        "- gold-silver i commodity-momentum korzystają tylko z zamkniętych świec dziennych.",
        "",
        "## Pominięte strategie",
        "",
        "- **tradingview**: TradingView nie udostępnia historii swojej oceny, więc nie da się "
        "jej odtworzyć wstecz.",
        "- **agents**: wymagałaby płatnej analizy modelu językowego dla każdej świecy.",
        "",
        "Wyniki z przeszłości nie gwarantują przyszłych; to test strategii, nie porada "
        "inwestycyjna.",
        "",
        "## Ostatnie transakcje",
        "",
    ]
    for name in ranked:
        lines += [
            f"> [!example]- {name}: ostatnie {min(10, len(report.trades[name]))} z "
            f"{len(report.trades[name])}",
            "> | Wejście (czas serwera) | Kierunek | Cena wejścia | Wyjście | Cena wyjścia | Powód "
            "| R | Wynik |",
            "> |" + " --- |" * 8,
        ]
        for trade in report.trades[name][-10:]:
            lines.append(
                "> | "
                + " | ".join(
                    [
                        _server_time(trade.entry_time),
                        "KUPNO" if trade.direction > 0 else "SPRZEDAŻ",
                        _amount(trade.entry, signed=False),
                        _server_time(trade.exit_time),
                        _amount(trade.exit, signed=False),
                        {"end": "koniec danych"}.get(trade.reason, trade.reason),
                        _amount(trade.r_multiple),
                        f"{_amount(trade.profit)} {currency}",
                    ]
                )
                + " |"
            )
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def write_backtest_note(
    report: BacktestReport, journal_dir: str | Path, *, now: datetime | None = None
) -> Path:
    """Write the report as "Backtest <symbol> <timeframe>.md" next to the signal journal."""
    from tradingagents.mt5_journal import SUMMARY_NOTE, write_note

    path = Path(journal_dir) / f"Backtest {report.symbol} {report.timeframe}.md"
    write_note(
        path, render_backtest_note(report, now=now or datetime.now(), summary_note=SUMMARY_NOTE)
    )
    return path
