"""Search strategy settings on older MT5 history, then check the best on the newest years.

Every variant, a strategy with its parameters or a combination of strategies, trades H1, H4
and D1 bars with each stop and target of the grid, by the backtest's rules. Settings are
ranked on the selection period only; the newest years, which the ranking never sees, show
whether the winners hold up. With hundreds of combinations some look good in the selection
period by chance, so only the check period is evidence.

The signals are computed for the whole history at once; for the bot's default parameters
they match the live strategies in paper_strategies and commodity_strategies.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

from tradingagents.commodity_strategies import (
    GOLD_SILVER_WINDOW,
    LONG_CORRELATION_WINDOW,
    SHORT_CORRELATION_WINDOW,
    strategy_symbols,
)
from tradingagents.mt5_backtest import (
    BARS_PER_YEAR,
    DAYS_PER_YEAR,
    WARMUP_BARS,
    average_true_range,
    format_amount,
    format_percent,
    format_server_time,
    simulate,
)
from tradingagents.mt5_demo import SIDES, TIMEFRAMES
from tradingagents.mt5_journal import SIDE_SETTING_LABELS, SUMMARY_NOTE, write_note

OPTIMIZER_TIMEFRAMES = ("H1", "H4", "D1")
STOP_MULTIPLES = (1.5, 2.0, 3.0)
TARGET_MULTIPLES = (1.0, 2.0, 3.0)
MIN_SELECTION_TRADES = 30
MIN_CHECK_TRADES = 10
TOP_VARIANTS = 15
LABELS = np.array(["Hold", "Buy", "Sell"])  # indexed by direction: 0, 1, -1
# The live bot's settings, for the reference rows: (variant, timeframe, stop, target, side).
BOT_DEFAULTS = (
    ("trend 20/50", "H1", 2.0, 2.0, "both"),
    ("mean-reversion RSI14 30/70", "H1", 2.0, 2.0, "both"),
    ("breakout 20", "H1", 2.0, 2.0, "both"),
    ("gold-silver z2", "H1", 2.0, 2.0, "both"),
    ("commodity-momentum 21d", "H1", 2.0, 2.0, "both"),
)
VARIANT_DESCRIPTIONS = {
    "trend 10/30": "EMA 10 powyżej EMA 30 = BUY, poniżej = SELL",
    "trend 20/50": "EMA 20 powyżej EMA 50 = BUY, poniżej = SELL (jak w bocie)",
    "trend 50/200": "EMA 50 powyżej EMA 200 = BUY, poniżej = SELL",
    "mean-reversion RSI14 30/70": "RSI 14 do 30 = BUY, od 70 = SELL (jak w bocie)",
    "mean-reversion RSI14 20/80": "RSI 14 do 20 = BUY, od 80 = SELL",
    "mean-reversion RSI7 30/70": "RSI 7 do 30 = BUY, od 70 = SELL",
    "breakout 20": "zamknięcie ponad maksimum / pod minimum 20 poprzednich świec (jak w bocie)",
    "breakout 55": "to samo dla 55 świec",
    "breakout 100": "to samo dla 100 świec",
    "gold-silver z1.5": "z-score relacji złoto/srebro (5 lat, dzienne) od 1,5",
    "gold-silver z2": "to samo od 2 (jak w bocie)",
    "gold-silver z2.5": "to samo od 2,5",
    "commodity-momentum 21d": "momentum albo odwrócenie złota z 21 dni wg korelacji surowców "
    "(jak w bocie)",
    "commodity-momentum 63d": "to samo z ruchu z 63 dni",
    "consensus 2 of 5": "wejście, gdy co najmniej 2 z 5 strategii bota wskazują ten kierunek, "
    "a przeciwnych jest mniej",
    "consensus 3 of 5": "to samo dla co najmniej 3 z 5",
    "breakout 20 + trend 50/200": "wybicie z 20 świec tylko w kierunku trendu EMA 50/200",
    "mean-reversion RSI14 + trend 50/200": "sygnał RSI 14 30/70 tylko w kierunku trendu "
    "EMA 50/200 (kupno spadków w trendzie wzrostowym)",
    "gold-silver + commodity-momentum": "wejście, gdy obie strategie dzienne wskazują ten sam "
    "kierunek",
}


@dataclass(frozen=True)
class GridResult:
    variant: str
    timeframe: str
    stop_atr: float
    take_profit_r: float
    selection: dict
    check: dict
    side: str = "both"


@dataclass(frozen=True)
class OptimizerReport:
    symbol: str
    timeframes: tuple[str, ...]
    first_time: int  # server clock
    split_time: int
    last_time: int
    risk_percent: float
    results: list[GridResult]
    skipped: list[str]
    sides: tuple[str, ...] = ("both",)


def keep_side(direction: np.ndarray, side: str) -> np.ndarray:
    """The directions a strategy limited to ``side`` acts on, like mt5_demo.restrict_signal."""
    if side == "long":
        return np.where(direction > 0, direction, 0)
    if side == "short":
        return np.where(direction < 0, direction, 0)
    return direction


def _trend(close: pd.Series, fast: int, slow: int) -> np.ndarray:
    fast_ema = close.ewm(span=fast, adjust=False, min_periods=fast).mean()
    slow_ema = close.ewm(span=slow, adjust=False, min_periods=slow).mean()
    return np.nan_to_num(np.sign((fast_ema - slow_ema).to_numpy())).astype(int)


def _mean_reversion(close: pd.Series, period: int, low: float, high: float) -> np.ndarray:
    change = close.diff()
    gain = change.clip(lower=0).rolling(period, min_periods=period).mean().to_numpy()
    loss = -change.clip(upper=0).rolling(period, min_periods=period).mean().to_numpy()
    with np.errstate(divide="ignore", invalid="ignore"):
        rsi = np.where(
            loss == 0,
            np.where(gain > 0, 100.0, 50.0),
            np.where(gain == 0, 0.0, 100 - 100 / (1 + gain / loss)),
        )
    rsi[np.isnan(gain) | np.isnan(loss)] = np.nan
    return np.where(rsi <= low, 1, np.where(rsi >= high, -1, 0))


def _breakout(bars: pd.DataFrame, lookback: int) -> np.ndarray:
    prior_high = bars["High"].shift(1).rolling(lookback, min_periods=lookback).max()
    prior_low = bars["Low"].shift(1).rolling(lookback, min_periods=lookback).min()
    close = bars["Close"]
    return np.where(close > prior_high, 1, np.where(close < prior_low, -1, 0))


def gold_silver_daily(closes: pd.DataFrame, symbol: str, window: int, entry_z: float) -> pd.Series:
    """Daily gold-silver directions, as commodity_strategies.gold_silver_signal decides them."""
    partner = next(column for column in closes.columns if column != symbol)
    ratio = closes[symbol] / closes[partner]
    mean = ratio.rolling(window, min_periods=window).mean()
    spread = ratio.rolling(window, min_periods=window).std()
    z_score = ((ratio - mean) / spread).to_numpy()
    return pd.Series(
        np.where(z_score >= entry_z, -1, np.where(z_score <= -entry_z, 1, 0)), index=closes.index
    )


def commodity_momentum_daily(
    closes: pd.DataFrame,
    symbol: str,
    lookback: int,
    short_window: int = SHORT_CORRELATION_WINDOW,
    long_window: int = LONG_CORRELATION_WINDOW,
) -> pd.Series:
    """Daily commodity-momentum directions, as commodity_momentum_signal decides them."""
    returns = closes.pct_change()

    def mean_correlation(window: int) -> np.ndarray:
        pairs = [
            returns[first].rolling(window, min_periods=window).corr(returns[second])
            for first, second in combinations(closes.columns, 2)
        ]
        frame = pd.concat(pairs, axis=1).replace([np.inf, -np.inf], np.nan)
        return frame.mean(axis=1, skipna=True).to_numpy()

    short, long = mean_correlation(short_window), mean_correlation(long_window)
    move = (closes[symbol] / closes[symbol].shift(lookback) - 1).to_numpy()
    direction = np.sign(move) * np.where(short > long, 1, -1)
    valid = np.isfinite(short) & np.isfinite(long) & np.isfinite(move)
    return pd.Series(np.where(valid, direction, 0).astype(int), index=closes.index)


def on_bars(daily: pd.Series, bars: pd.DataFrame, timeframe: str) -> np.ndarray:
    """Give each bar the daily direction of the last day already closed when the bar closes."""
    available = daily.index.to_numpy(dtype=np.int64) + 86_400
    bar_close = bars["time"].to_numpy(dtype=np.int64) + TIMEFRAMES[timeframe][1]
    latest = np.searchsorted(available, bar_close, side="right") - 1
    values = daily.to_numpy()
    return np.where(latest >= 0, values[np.clip(latest, 0, None)], 0).astype(int)


def consensus(directions: list[np.ndarray], minimum: int) -> np.ndarray:
    """Buy when at least ``minimum`` strategies buy and fewer sell; the same for selling."""
    stacked = np.vstack(directions)
    buys, sells = (stacked > 0).sum(axis=0), (stacked < 0).sum(axis=0)
    return np.where(
        (buys >= minimum) & (buys > sells), 1, np.where((sells >= minimum) & (sells > buys), -1, 0)
    )


def agree(signal: np.ndarray, confirmation: np.ndarray) -> np.ndarray:
    """Keep a signal only where the confirming strategy points the same way."""
    return np.where(signal == confirmation, signal, 0)


def variant_directions(
    bars: pd.DataFrame,
    timeframe: str,
    gold_silver_closes: pd.DataFrame | None,
    basket_closes: pd.DataFrame | None,
    symbol: str,
) -> dict[str, np.ndarray]:
    """Every variant's direction (1 buy, -1 sell, 0 none) at every bar."""
    close = bars["Close"]
    directions = {
        "trend 10/30": _trend(close, 10, 30),
        "trend 20/50": _trend(close, 20, 50),
        "trend 50/200": _trend(close, 50, 200),
        "mean-reversion RSI14 30/70": _mean_reversion(close, 14, 30, 70),
        "mean-reversion RSI14 20/80": _mean_reversion(close, 14, 20, 80),
        "mean-reversion RSI7 30/70": _mean_reversion(close, 7, 30, 70),
        "breakout 20": _breakout(bars, 20),
        "breakout 55": _breakout(bars, 55),
        "breakout 100": _breakout(bars, 100),
    }
    if gold_silver_closes is not None:
        for entry_z, name in ((1.5, "z1.5"), (2.0, "z2"), (2.5, "z2.5")):
            directions[f"gold-silver {name}"] = on_bars(
                gold_silver_daily(gold_silver_closes, symbol, GOLD_SILVER_WINDOW, entry_z),
                bars,
                timeframe,
            )
    if basket_closes is not None:
        for lookback in (21, 63):
            directions[f"commodity-momentum {lookback}d"] = on_bars(
                commodity_momentum_daily(basket_closes, symbol, lookback), bars, timeframe
            )
    bot = [
        directions[name]
        for name in (
            "trend 20/50",
            "mean-reversion RSI14 30/70",
            "breakout 20",
            "gold-silver z2",
            "commodity-momentum 21d",
        )
        if name in directions
    ]
    directions["consensus 2 of 5"] = consensus(bot, 2)
    directions["consensus 3 of 5"] = consensus(bot, 3)
    directions["breakout 20 + trend 50/200"] = agree(
        directions["breakout 20"], directions["trend 50/200"]
    )
    directions["mean-reversion RSI14 + trend 50/200"] = agree(
        directions["mean-reversion RSI14 30/70"], directions["trend 50/200"]
    )
    if "gold-silver z2" in directions and "commodity-momentum 21d" in directions:
        directions["gold-silver + commodity-momentum"] = agree(
            directions["gold-silver z2"], directions["commodity-momentum 21d"]
        )
    return directions


def period_stats(r_values: list[float]) -> dict:
    """Trades, win rate, average and total R, profit factor and deepest R drawdown."""
    count = len(r_values)
    if not count:
        return {
            "trades": 0,
            "win_rate": None,
            "average_r": None,
            "total_r": 0.0,
            "profit_factor": None,
            "max_drawdown_r": 0.0,
        }
    gains = sum(value for value in r_values if value > 0)
    losses = -sum(value for value in r_values if value < 0)
    equity = np.concatenate([[0.0], np.cumsum(r_values)])
    return {
        "trades": count,
        "win_rate": sum(value > 0 for value in r_values) / count,
        "average_r": sum(r_values) / count,
        "total_r": float(equity[-1]),
        "profit_factor": gains / losses if losses else None,
        "max_drawdown_r": float(np.min(equity - np.maximum.accumulate(equity))),
    }


def check_settings(
    timeframes: tuple[str, ...],
    years: float,
    check_years: float,
    sides: tuple[str, ...] = ("both",),
) -> None:
    """Reject timeframes or sides the optimizer cannot use and a check that leaves no selection."""
    unknown = [timeframe for timeframe in timeframes if timeframe not in OPTIMIZER_TIMEFRAMES]
    if unknown or not timeframes:
        raise ValueError(
            f"Optimize on {', '.join(OPTIMIZER_TIMEFRAMES)}; not on {', '.join(unknown) or 'nothing'}."
        )
    unknown = [side for side in sides if side not in SIDES]
    if unknown or not sides:
        raise ValueError(
            f"Sides must be among {', '.join(SIDES)}; not {', '.join(unknown) or 'none'}."
        )
    if not 0 < check_years <= years - 1:
        raise ValueError("Keep at least one year of history for selection before the check years.")


def run_optimizer(
    session,
    symbol: str,
    *,
    years: float,
    check_years: float,
    timeframes: tuple[str, ...] = OPTIMIZER_TIMEFRAMES,
    risk_percent: float = 0.1,
    sides: tuple[str, ...] = SIDES,
) -> OptimizerReport:
    """Trade every variant, timeframe, stop, target and side; split ``check_years`` before the end."""
    check_settings(timeframes, years, check_years, sides)
    skipped = []
    daily = {}
    for strategy, window in (
        ("gold-silver", GOLD_SILVER_WINDOW),
        ("commodity-momentum", LONG_CORRELATION_WINDOW + 1),
    ):
        try:
            symbols = strategy_symbols(session, symbol, strategy)
            daily[strategy] = session.daily_closes(
                symbols, int(years * DAYS_PER_YEAR) + window + 90
            )
        except (ValueError, RuntimeError) as exc:
            daily[strategy] = None
            skipped.append(f"{strategy}: {exc}")

    results: list[GridResult] = []
    first_time = last_time = split_time = None
    for timeframe in timeframes:
        bars = session.history_bars(
            symbol, timeframe, int(years * BARS_PER_YEAR[timeframe]) + WARMUP_BARS
        )
        if len(bars) < WARMUP_BARS + 50:
            skipped.append(f"{timeframe}: MT5 returned only {len(bars)} bars")
            continue
        if split_time is None:
            last_time = int(bars["time"].iloc[-1])
            split_time = last_time - int(check_years * 365.25 * 86_400)
        start_time = int(bars["time"].iloc[WARMUP_BARS])
        first_time = start_time if first_time is None else min(first_time, start_time)
        atr = average_true_range(bars)
        directions = variant_directions(
            bars, timeframe, daily["gold-silver"], daily["commodity-momentum"], symbol
        )
        for name, direction in directions.items():
            for side in sides:
                labels = LABELS[keep_side(direction, side)]
                for stop in STOP_MULTIPLES:
                    for target in TARGET_MULTIPLES:
                        trades = simulate(
                            bars,
                            labels.__getitem__,
                            atr.__getitem__,
                            capital=1.0,
                            risk_percent=1.0,
                            stop_atr=stop,
                            take_profit_r=target,
                        )
                        results.append(
                            GridResult(
                                variant=name,
                                timeframe=timeframe,
                                stop_atr=stop,
                                take_profit_r=target,
                                selection=period_stats(
                                    [t.r_multiple for t in trades if t.entry_time < split_time]
                                ),
                                check=period_stats(
                                    [t.r_multiple for t in trades if t.entry_time >= split_time]
                                ),
                                side=side,
                            )
                        )
    if not results:
        raise ValueError("MT5 returned too little history to optimize; " + "; ".join(skipped))
    return OptimizerReport(
        symbol=symbol,
        timeframes=tuple(timeframes),
        first_time=first_time,
        split_time=split_time,
        last_time=last_time,
        risk_percent=risk_percent,
        results=results,
        skipped=skipped,
        sides=tuple(sides),
    )


def ranked(results: list[GridResult]) -> list[GridResult]:
    """Combinations with enough selection trades, best average R in the selection period first."""
    eligible = [r for r in results if r.selection["trades"] >= MIN_SELECTION_TRADES]
    return sorted(
        eligible,
        key=lambda r: (r.selection["average_r"], r.selection["total_r"]),
        reverse=True,
    )


def verdict(result: GridResult) -> str:
    check = result.check
    if check["trades"] < MIN_CHECK_TRADES:
        return "⚪ za mało transakcji"
    return "✅ zarabia" if check["average_r"] > 0 else "❌ traci"


def _r(value: float | None, digits: int = 2) -> str:
    return "—" if value is None else format_amount(value, digits=digits)


def _cells(result: GridResult, risk_percent: float) -> list[str]:
    selection, check = result.selection, result.check
    return [
        result.variant,
        SIDE_SETTING_LABELS[result.side],
        result.timeframe,
        f"{format_amount(result.stop_atr, signed=False, digits=1)}×ATR / "
        f"{format_amount(result.take_profit_r, signed=False, digits=0)}R",
        f"{_r(selection['average_r'])} ({selection['trades']})",
        f"{_r(check['average_r'])} ({check['trades']})",
        format_percent(check["win_rate"], signed=False),
        _r(check["total_r"], 1),
        format_percent(check["total_r"] * risk_percent / 100),
        verdict(result),
    ]


def subset_numbers(results: list[GridResult]) -> dict:
    """How the selection winners of ``results`` did in the check period, against all of them."""
    eligible = [r for r in results if r.selection["trades"] >= MIN_SELECTION_TRADES]
    best = ranked(results)[:TOP_VARIANTS]
    checked_best = [r for r in best if r.check["trades"] >= MIN_CHECK_TRADES]
    checked_all = [r for r in results if r.check["trades"] >= MIN_CHECK_TRADES]

    def mean_r(chosen: list[GridResult]) -> float | None:
        values = [r.check["average_r"] for r in chosen]
        return sum(values) / len(values) if values else None

    return {
        "best": best,
        "eligible": len(eligible),
        "selection_positive": sum(r.selection["average_r"] > 0 for r in eligible),
        "best_check_r": mean_r(checked_best),
        "all_check_r": mean_r(checked_all),
        "all_positive": sum(r.check["average_r"] > 0 for r in checked_all),
        "all_checked": len(checked_all),
        "best_positive": sum(r.check["average_r"] > 0 for r in checked_best),
        "best_checked": len(checked_best),
    }


def summary_numbers(report: OptimizerReport) -> dict:
    """The selection winners over every side against every combination."""
    return subset_numbers(report.results)


def side_numbers(report: OptimizerReport) -> dict[str, dict]:
    """subset_numbers() for each side the report traded."""
    return {
        side: subset_numbers([r for r in report.results if r.side == side]) for side in report.sides
    }


def _comparison_row(side: str, numbers: dict) -> str:
    best = numbers["best"][0] if numbers["best"] else None
    winner = (
        f"{best.variant}, {best.timeframe}, "
        f"{format_amount(best.stop_atr, signed=False, digits=1)}×ATR / "
        f"{format_amount(best.take_profit_r, signed=False, digits=0)}R: "
        f"{_r(best.selection['average_r'])} → {_r(best.check['average_r'])}"
        if best
        else "—"
    )
    cells = [
        SIDE_SETTING_LABELS[side],
        f"{numbers['selection_positive']} z {numbers['eligible']}",
        _r(numbers["all_check_r"]),
        f"{numbers['all_positive']} z {numbers['all_checked']}",
        f"{_r(numbers['best_check_r'])} ({numbers['best_positive']} z "
        f"{numbers['best_checked']} na plusie)",
        winner,
    ]
    return "| " + " | ".join(cells) + " |"


def render_optimizer_note(
    report: OptimizerReport, *, now: datetime, summary_note: str, backtest_note: str
) -> str:
    """The Markdown note (Polish, like the journal) with the selection winners and their check."""
    numbers = summary_numbers(report)
    variants = sorted({r.variant for r in report.results}, key=list(VARIANT_DESCRIPTIONS).index)
    risk = f"{report.risk_percent:g}".replace(".", ",")
    header = [
        "Wariant",
        "Kierunek",
        "Interwał",
        "SL / TP",
        "Dobór: śr. R (transakcje)",
        "Sprawdzenie: śr. R (transakcje)",
        "Sprawdzenie: trafione",
        "Sprawdzenie: suma R",
        f"Sprawdzenie: wynik przy {risk}%",
        "Ocena",
    ]

    def table(results: list[GridResult], numbered: bool) -> list[str]:
        columns = (["#"] if numbered else []) + header
        rows = [
            "| "
            + " | ".join(([str(rank)] if numbered else []) + _cells(result, report.risk_percent))
            + " |"
            for rank, result in enumerate(results, start=1)
        ]
        return ["| " + " | ".join(columns) + " |", "|" + " --- |" * len(columns), *rows]

    by_key = {
        (r.variant, r.timeframe, r.stop_atr, r.take_profit_r, r.side): r for r in report.results
    }
    defaults = [by_key[key] for key in BOT_DEFAULTS if key in by_key]
    best_r, all_r = numbers["best_check_r"], numbers["all_check_r"]
    comparison, per_side_tables = [], []
    if len(report.sides) > 1:
        per_side = side_numbers(report)
        comparison = [
            "## Porównanie kierunków",
            "",
            "Każdy kierunek to ta sama siatka; „na plusie” to kombinacje ze średnim R powyżej "
            "zera, a średnie R w sprawdzeniu liczę ze wszystkich kombinacji kierunku.",
            "",
            f"| Kierunek | Dobór: na plusie (min. {MIN_SELECTION_TRADES} transakcji) "
            "| Sprawdzenie: śr. R na transakcję "
            f"| Sprawdzenie: na plusie (min. {MIN_CHECK_TRADES} transakcji) "
            f"| {TOP_VARIANTS} najlepszych z doboru w sprawdzeniu "
            "| Najlepsza w doborze (dobór → sprawdzenie) |",
            "|" + " --- |" * 6,
            *(_comparison_row(side, per_side[side]) for side in report.sides),
            "",
        ]
        per_side_tables = ["## Najlepsze dla każdego kierunku", ""]
        for side in report.sides:
            per_side_tables += [
                f"### {SIDE_SETTING_LABELS[side].capitalize()}",
                "",
                *table(per_side[side]["best"][:5], numbered=True),
                "",
            ]
    lines = [
        "---",
        "tags:",
        "  - bot-mt5",
        "  - backtest",
        "  - optymalizacja",
        "---",
        f"# Optymalizacja {report.symbol}",
        "",
        f"> Generowane przez `tradingagents mt5-optimize` {now:%d.%m.%Y %H:%M}; "
        "kolejne uruchomienie nadpisze tę notatkę.",
        f"> Wróć do [[{summary_note}]] · zasady symulacji jak w [[{backtest_note}]].",
        "",
        f"- **Dobór ustawień:** {format_server_time(report.first_time, '%d.%m.%Y')} – "
        f"{format_server_time(report.split_time, '%d.%m.%Y')}: tu wybieram najlepsze kombinacje.",
        f"- **Sprawdzenie:** {format_server_time(report.split_time, '%d.%m.%Y')} – "
        f"{format_server_time(report.last_time, '%d.%m.%Y')}: nowsze dane, których wybór nie "
        "widział.",
        f"- **Siatka:** {len(variants)} wariantów × interwały {', '.join(report.timeframes)} × "
        "stop 1,5 / 2 / 3×ATR × take profit 1R / 2R / 3R × kierunki "
        f"({' / '.join(SIDE_SETTING_LABELS[side] for side in report.sides)}) = "
        f"{len(report.results)} kombinacji.",
        f"- **Wynik w %** to suma R razy ryzyko {risk}% kapitału na transakcję.",
        "",
        *comparison,
        f"## {TOP_VARIANTS} najlepszych w okresie doboru i ich wynik na nowych danych",
        "",
        f"Ranking według średniego R na transakcję w okresie doboru (co najmniej "
        f"{MIN_SELECTION_TRADES} transakcji).",
        "",
        *table(numbers["best"], numbered=True),
        "",
        f"Ocena: ✅ zarabia także w okresie sprawdzenia, ❌ traci, ⚪ mniej niż {MIN_CHECK_TRADES} "
        "transakcji.",
        "",
        *per_side_tables,
        "## Czy dobór coś daje?",
        "",
        f"- Średnie R na transakcję w okresie sprawdzenia: najlepsze z doboru {_r(best_r)}, "
        f"wszystkie kombinacje {_r(all_r)}.",
        f"- {numbers['best_positive']} z {numbers['best_checked']} najlepszych (z wystarczającą "
        "liczbą transakcji) zarabia także na nowych danych.",
        "- Jeśli najlepsze z doboru nie wypadają w sprawdzeniu lepiej niż przeciętna kombinacja, "
        "ich wynik w doborze był głównie przypadkiem.",
        "",
        "## Obecne ustawienia bota (H1, SL 2×ATR, TP 2R, oba kierunki)",
        "",
        *(
            table(defaults, numbered=False)
            if defaults
            else ["Ta siatka ich nie obejmuje (potrzebny interwał H1 i oba kierunki)."]
        ),
        "",
        "## Warianty",
        "",
        *(f"- **{name}**: {VARIANT_DESCRIPTIONS[name]}" for name in variants),
        "",
        "## Zasady i zastrzeżenia",
        "",
        "- Każda kombinacja handluje osobno, jak w backteście: wejście po otwarciu następnej "
        "świecy, spread z MT5, pozycję zamyka tylko SL albo TP, jedna pozycja naraz, najwyżej "
        "jedno wejście dziennie.",
        "- „Tylko kupno” i „tylko sprzedaż” traktują sygnały w drugą stronę jak HOLD, tak jak "
        "opcja --side bota.",
        f"- Z {len(report.results)} kombinacji część wygląda dobrze przypadkiem; dowodem jest "
        "dopiero okres sprawdzenia, a i tam liczba transakcji bywa mała.",
        "- Bez swapów i prowizji; przy pozycjach trzymanych dniami (H4, D1) swap może zjeść wynik, "
        "a dla kupna i sprzedaży jest różny.",
        "- TradingView i agents nie mają historii do odtworzenia, więc ich tu nie ma.",
        "- Wyniki z przeszłości nie gwarantują przyszłych; to test, nie porada inwestycyjna.",
    ]
    if report.skipped:
        lines += ["", "Pominięte: " + "; ".join(report.skipped) + "."]
    return "\n".join(lines).rstrip() + "\n"


def write_optimizer_note(
    report: OptimizerReport, journal_dir: str | Path, *, now: datetime | None = None
) -> Path:
    """Write "Backtest <symbol> optymalizacja.md", which the journal summary links."""
    path = Path(journal_dir) / f"Backtest {report.symbol} optymalizacja.md"
    write_note(
        path,
        render_optimizer_note(
            report,
            now=now or datetime.now(),
            summary_note=SUMMARY_NOTE,
            backtest_note=f"Backtest {report.symbol} H1",
        ),
    )
    return path
