"""TradingView technical-analysis signals for MT5 demo cycles."""

from __future__ import annotations

import re

import requests

TRADINGVIEW_INTERVALS = {
    "M15": "INTERVAL_15_MINUTES",
    "H1": "INTERVAL_1_HOUR",
    "H4": "INTERVAL_4_HOURS",
    "D1": "INTERVAL_1_DAY",
}
SUPPORTED_TIMEFRAMES = tuple(TRADINGVIEW_INTERVALS)
SIGNAL_BY_RECOMMENDATION = {
    "STRONG_BUY": "Buy",
    "BUY": "Buy",
    "NEUTRAL": "Hold",
    "SELL": "Sell",
    "STRONG_SELL": "Sell",
}


def default_tradingview_symbol(symbol: str) -> str:
    """Map broker names such as GOLD.a, XAUUSDm or EURUSD.a to TradingView's plain ticker."""
    upper = symbol.strip().upper()
    if upper.startswith("GOLD"):
        return "XAUUSD"
    pair = re.match(r"[A-Z]{6}", upper)
    return pair.group(0) if pair else upper


def get_tradingview_signal(
    symbol: str,
    *,
    timeframe: str,
    exchange: str | None = None,
    handler_class=None,
    interval_class=None,
) -> tuple[str, str]:
    """Return TradingView's indicator rating mapped to a TradingAgents signal."""
    if timeframe not in TRADINGVIEW_INTERVALS:
        raise ValueError(
            f"Unsupported timeframe {timeframe!r}; use {', '.join(SUPPORTED_TIMEFRAMES)}."
        )
    symbol = symbol.strip()
    if not symbol:
        raise ValueError("TradingView analysis requires a symbol.")
    if exchange is not None:
        exchange = exchange.strip()
        if not exchange:
            raise ValueError("TradingView exchange cannot be empty.")
    else:
        # The default OANDA/FX_IDC listings only know plain tickers like XAUUSD and EURUSD.
        symbol = default_tradingview_symbol(symbol)
    is_gold = symbol.upper().startswith(("XAU", "GOLD"))
    screener = "cfd" if is_gold else "forex"
    selected_exchange = exchange or ("OANDA" if is_gold else "FX_IDC")
    if handler_class is None or interval_class is None:
        try:
            from tradingview_ta import Interval, TA_Handler
        except ImportError as exc:
            raise RuntimeError(
                "TradingView analysis requires the optional dependency. "
                'Install it with `pip install -e ".[tradingview]"`.'
            ) from exc
        handler_class = handler_class or TA_Handler
        interval_class = interval_class or Interval

    interval_name = TRADINGVIEW_INTERVALS[timeframe]
    interval = getattr(interval_class, interval_name)
    try:
        result = handler_class(
            symbol=symbol,
            screener=screener,
            exchange=selected_exchange,
            interval=interval,
        ).get_analysis()
    except requests.RequestException as exc:
        raise RuntimeError(
            f"Could not retrieve TradingView analysis for "
            f"{selected_exchange}:{symbol} ({timeframe})."
        ) from exc
    recommendation = result.summary.get("RECOMMENDATION")
    if (
        not isinstance(recommendation, str)
        or recommendation.upper() not in SIGNAL_BY_RECOMMENDATION
    ):
        raise RuntimeError(
            f"TradingView returned no supported recommendation for {selected_exchange}:{symbol} "
            f"({timeframe}): {recommendation!r}."
        )
    return SIGNAL_BY_RECOMMENDATION[recommendation.upper()], recommendation.upper()
