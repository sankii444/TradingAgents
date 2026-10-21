"""Deterministic signal variants for local paper trading."""

from __future__ import annotations

import math

import pandas as pd

PAPER_STRATEGIES = ("agents", "trend", "mean-reversion", "breakout")


def parse_paper_strategies(value: str) -> list[str]:
    strategies = [item.strip().lower() for item in value.split(",") if item.strip()]
    unknown = [item for item in strategies if item not in PAPER_STRATEGIES]
    if not strategies:
        raise ValueError("Choose at least one strategy.")
    if unknown:
        choices = ", ".join(PAPER_STRATEGIES)
        raise ValueError(f"Unknown strategy {', '.join(unknown)}; choose from {choices}.")
    return list(dict.fromkeys(strategies))


def signal_for_strategy(strategy: str, bars: pd.DataFrame) -> str:
    """Return a daily Buy/Sell/Hold signal using only bars through the latest close."""
    if strategy not in PAPER_STRATEGIES or strategy == "agents":
        raise ValueError(f"{strategy!r} is not a deterministic paper strategy.")
    required = {"High", "Low", "Close"}
    missing = required.difference(bars.columns)
    if missing:
        raise ValueError(f"Market data is missing required columns: {', '.join(sorted(missing))}.")

    close = pd.to_numeric(bars["Close"], errors="coerce")
    if close.empty or not math.isfinite(float(close.iloc[-1])):
        raise ValueError("Market data does not contain a usable latest close.")

    if strategy == "trend":
        if len(close) < 50:
            raise ValueError("The trend strategy requires at least 50 daily closes.")
        fast = close.ewm(span=20, adjust=False, min_periods=20).mean().iloc[-1]
        slow = close.ewm(span=50, adjust=False, min_periods=50).mean().iloc[-1]
        if not math.isfinite(float(fast)) or not math.isfinite(float(slow)):
            raise ValueError("Market data does not contain enough valid closes for EMA.")
        return "Buy" if fast > slow else "Sell" if fast < slow else "Hold"

    if strategy == "mean-reversion":
        if len(close) < 15:
            raise ValueError("The mean-reversion strategy requires at least 15 daily closes.")
        change = close.diff()
        average_gain = change.clip(lower=0).rolling(14, min_periods=14).mean().iloc[-1]
        average_loss = -change.clip(upper=0).rolling(14, min_periods=14).mean().iloc[-1]
        if not math.isfinite(float(average_gain)) or not math.isfinite(float(average_loss)):
            raise ValueError("Market data does not contain enough valid closes for RSI.")
        if average_loss == 0:
            rsi = 100.0 if average_gain > 0 else 50.0
        elif average_gain == 0:
            rsi = 0.0
        else:
            rsi = 100 - (100 / (1 + average_gain / average_loss))
        return "Buy" if rsi <= 30 else "Sell" if rsi >= 70 else "Hold"

    if len(bars) < 21:
        raise ValueError("The breakout strategy requires at least 21 daily bars.")
    high = pd.to_numeric(bars["High"], errors="coerce")
    low = pd.to_numeric(bars["Low"], errors="coerce")
    previous_high = high.iloc[-21:-1].max()
    previous_low = low.iloc[-21:-1].min()
    if not math.isfinite(float(previous_high)) or not math.isfinite(float(previous_low)):
        raise ValueError("Market data does not contain valid prior highs and lows.")
    latest_close = float(close.iloc[-1])
    return (
        "Buy" if latest_close > previous_high else "Sell" if latest_close < previous_low else "Hold"
    )
