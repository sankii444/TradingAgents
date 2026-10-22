"""Daily commodity strategies from the awesome-systematic-trading list, for one MT5 gold symbol.

* ``gold-silver`` follows "Rolling vs. Expanding Windows in Mean-Reversion Strategies:
  Evidence from Gold-Silver and Cross-Asset Validation" (Gupta, 2026): the gold/silver price
  ratio is scored against a rolling window of daily closes, which the paper finds clearly
  better than an expanding one. A rich ratio sells gold; a cheap one buys it.
* ``commodity-momentum`` follows "How to Improve Commodity Momentum Using Intra-Market
  Correlation" (Pauchlyová and Vojtko, 2024): when commodities moved together more over the
  last month than over the last year, follow gold's one-month move (momentum); otherwise
  fade it (reversal). The paper ranks four commodity sector ETFs monthly; here the same switch
  drives gold against a basket of commodities the broker offers.

Both papers trade daily or monthly, so these read daily bars, and in an hourly test their
signal changes at most once a day. The windows and the z-score threshold are choices within
the papers' ranges, not values the papers prescribe.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

COMMODITY_STRATEGIES = ("gold-silver", "commodity-momentum")
GOLD_SILVER_WINDOW = 1260  # five years of trading days; the paper favours 5-10 year windows
GOLD_SILVER_ENTRY_Z = 2.0
MOMENTUM_LOOKBACK = 21  # one month of trading days
SHORT_CORRELATION_WINDOW = 21
LONG_CORRELATION_WINDOW = 252
EXTRA_DAYS = 60  # spare bars for days on which one market was closed
GOLD_NAMES = ("XAUUSD", "GOLD")
SILVER_NAMES = ("SILVER", "XAGUSD")
# Broker names tried for each basket member, each first with the gold symbol's suffix.
BASKET_NAMES = (
    SILVER_NAMES,
    ("COPPER", "XCUUSD", "HG"),
    ("BRENT", "UKOIL", "XBRUSD", "USOIL", "WTI", "XTIUSD"),
    ("NGAS", "NATGAS", "XNGUSD"),
)


def gold_silver_signal(
    gold: pd.Series,
    silver: pd.Series,
    *,
    window: int = GOLD_SILVER_WINDOW,
    entry_z: float = GOLD_SILVER_ENTRY_Z,
) -> tuple[str, dict]:
    """Sell gold when the gold/silver ratio is ``entry_z`` deviations rich, buy when cheap."""
    ratio = (gold / silver).replace([np.inf, -np.inf], np.nan).dropna()
    if len(ratio) < window:
        raise ValueError(
            f"The gold-silver strategy needs {window} daily closes of both metals; "
            f"MT5 returned {len(ratio)}."
        )
    recent = ratio.iloc[-window:]
    spread = float(recent.std())
    if not math.isfinite(spread) or spread == 0:
        raise ValueError("The gold/silver ratio does not vary over the window.")
    z_score = float((ratio.iloc[-1] - recent.mean()) / spread)
    signal = "Sell" if z_score >= entry_z else "Buy" if z_score <= -entry_z else "Hold"
    details = f"gold/silver ratio {ratio.iloc[-1]:.2f}, z-score {z_score:+.2f} over {window} days"
    return signal, {"strategy_details": details}


def _mean_pairwise_correlation(returns: pd.DataFrame) -> float:
    matrix = returns.corr().to_numpy()
    pairs = matrix[~np.eye(len(matrix), dtype=bool)]
    pairs = pairs[np.isfinite(pairs)]
    return float(pairs.mean()) if pairs.size else math.nan


def commodity_momentum_signal(
    closes: pd.DataFrame,
    target: str,
    *,
    lookback: int = MOMENTUM_LOOKBACK,
    short_window: int = SHORT_CORRELATION_WINDOW,
    long_window: int = LONG_CORRELATION_WINDOW,
) -> tuple[str, dict]:
    """Follow ``target``'s recent move when correlations rise, fade it when they fall."""
    if target not in closes or closes.shape[1] < 3:
        raise ValueError(
            "The commodity-momentum strategy needs gold and at least two other commodities."
        )
    returns = closes.pct_change().iloc[1:]
    if len(returns) < max(long_window, lookback):
        raise ValueError(
            f"The commodity-momentum strategy needs {long_window + 1} daily closes; "
            f"MT5 returned {len(closes)}."
        )
    short = _mean_pairwise_correlation(returns.iloc[-short_window:])
    long = _mean_pairwise_correlation(returns.iloc[-long_window:])
    if not math.isfinite(short) or not math.isfinite(long):
        raise ValueError("Commodity correlations could not be computed from the daily closes.")
    move = float(closes[target].iloc[-1] / closes[target].iloc[-1 - lookback] - 1)
    regime = "momentum" if short > long else "reversal"
    direction = (move > 0) - (move < 0)
    if regime == "reversal":
        direction = -direction
    signal = {1: "Buy", -1: "Sell", 0: "Hold"}[direction]
    details = (
        f"{regime}: correlation {short:.2f} over {short_window} days vs {long:.2f} over "
        f"{long_window}; {target} {move:+.2%} over {lookback} days"
    )
    return signal, {"strategy_details": details}


def _find_symbol(session, names: tuple[str, ...], suffix: str) -> str | None:
    for name in names:
        for candidate in dict.fromkeys((name + suffix, name)):
            if session.has_symbol(candidate):
                return candidate
    return None


def strategy_symbols(session, symbol: str, strategy: str) -> list[str]:
    """The broker symbols whose daily closes ``strategy`` reads, the gold ``symbol`` first."""
    if strategy not in COMMODITY_STRATEGIES:
        raise ValueError(f"Unknown commodity strategy {strategy!r}.")
    base = next((name for name in GOLD_NAMES if symbol.upper().startswith(name)), None)
    if base is None:
        raise ValueError(f"The {strategy} strategy trades gold; {symbol} is not a gold symbol.")
    suffix = symbol[len(base) :]
    if strategy == "gold-silver":
        silver = _find_symbol(session, SILVER_NAMES, suffix)
        if silver is None:
            raise ValueError(
                "The gold-silver strategy needs silver (SILVER or XAGUSD), "
                "which this MT5 broker does not offer."
            )
        return [symbol, silver]
    basket = [symbol]
    for names in BASKET_NAMES:
        found = _find_symbol(session, names, suffix)
        if found is not None and found not in basket:
            basket.append(found)
    return basket


def history_days(strategy: str) -> int:
    """Closed daily bars needed for one signal of ``strategy``."""
    window = GOLD_SILVER_WINDOW if strategy == "gold-silver" else LONG_CORRELATION_WINDOW + 1
    return window + EXTRA_DAYS


def commodity_signal(strategy: str, closes: pd.DataFrame, symbol: str) -> tuple[str, dict]:
    """The signal from daily ``closes`` laid out as strategy_symbols() returns them."""
    if strategy == "gold-silver":
        partner = next(column for column in closes.columns if column != symbol)
        return gold_silver_signal(closes[symbol], closes[partner])
    return commodity_momentum_signal(closes, symbol)


def mt5_commodity_signal(session, symbol: str, strategy: str) -> tuple[str, dict]:
    """Compute a commodity strategy's signal for the gold ``symbol`` from MT5 daily bars."""
    symbols = strategy_symbols(session, symbol, strategy)
    closes = session.daily_closes(symbols, history_days(strategy))
    return commodity_signal(strategy, closes, symbol)
