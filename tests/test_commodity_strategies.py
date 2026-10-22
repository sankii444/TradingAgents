from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tradingagents.commodity_strategies import (
    commodity_momentum_signal,
    gold_silver_signal,
    mt5_commodity_signal,
)


def _metals(last_ratio: float, days: int = 100) -> tuple[pd.Series, pd.Series]:
    """Gold and silver closes whose ratio swings around 80 and ends at ``last_ratio``."""
    ratio = 80 + 2 * np.sin(np.arange(days))
    ratio[-1] = last_ratio
    silver = pd.Series(np.full(days, 25.0))
    return silver * ratio, silver


@pytest.mark.unit
@pytest.mark.parametrize(
    ("last_ratio", "expected"), [(90.0, "Sell"), (70.0, "Buy"), (80.0, "Hold")]
)
def test_gold_silver_trades_gold_against_a_stretched_ratio(last_ratio, expected):
    gold, silver = _metals(last_ratio)

    signal, state = gold_silver_signal(gold, silver, window=100)

    assert signal == expected
    assert "z-score" in state["strategy_details"]


@pytest.mark.unit
def test_gold_silver_needs_a_full_rolling_window():
    gold, silver = _metals(80.0, days=50)

    with pytest.raises(ValueError, match="needs 100 daily closes"):
        gold_silver_signal(gold, silver, window=100)


def _commodities(correlated_recently: bool, gold_drift: float) -> pd.DataFrame:
    """Three commodities that move together either in the last month or before it."""
    rng = np.random.default_rng(7)
    days = 300
    independent = rng.normal(0, 0.01, size=(days, 3))
    together = rng.normal(0, 0.01, size=(days, 1)).repeat(3, axis=1)
    recent = np.arange(days) >= days - 21
    joint = recent if correlated_recently else ~recent
    returns = np.where(joint[:, None], together, independent)
    returns[recent, 0] += gold_drift
    closes = 100 * np.cumprod(1 + returns, axis=0)
    return pd.DataFrame(closes, columns=["GOLD", "SILVER", "COPPER"])


@pytest.mark.unit
def test_commodity_momentum_follows_gold_when_correlation_rises():
    signal, state = commodity_momentum_signal(_commodities(True, 0.006), "GOLD")

    assert signal == "Buy"
    assert state["strategy_details"].startswith("momentum")


@pytest.mark.unit
def test_commodity_momentum_fades_gold_when_correlation_falls():
    signal, state = commodity_momentum_signal(_commodities(False, 0.006), "GOLD")

    assert signal == "Sell"
    assert state["strategy_details"].startswith("reversal")


@pytest.mark.unit
def test_commodity_momentum_needs_a_basket_and_a_year_of_closes():
    closes = _commodities(True, 0.003)

    with pytest.raises(ValueError, match="at least two other commodities"):
        commodity_momentum_signal(closes[["GOLD", "SILVER"]], "GOLD")
    with pytest.raises(ValueError, match="needs 253 daily closes"):
        commodity_momentum_signal(closes.iloc[-100:], "GOLD")


class FakeSession:
    def __init__(self, offered, closes):
        self.offered = set(offered)
        self.closes = closes
        self.requests = []

    def has_symbol(self, name):
        return name in self.offered

    def daily_closes(self, symbols, count):
        self.requests.append((list(symbols), count))
        return self.closes[list(symbols)]


@pytest.mark.unit
def test_mt5_signals_find_the_broker_names_of_the_partner_commodities():
    days = 1400
    closes = pd.DataFrame(
        {
            "GOLD.a": 2000 + 50 * np.sin(np.arange(days) / 30),
            "SILVER.a": 25 + np.cos(np.arange(days) / 20),
            "COPPER": 4 + 0.2 * np.sin(np.arange(days) / 15),
            "BRENT": 80 + 5 * np.cos(np.arange(days) / 25),
        }
    )
    session = FakeSession({"GOLD.a", "SILVER.a", "COPPER", "BRENT"}, closes)

    gold_silver, _ = mt5_commodity_signal(session, "GOLD.a", "gold-silver")
    momentum, _ = mt5_commodity_signal(session, "GOLD.a", "commodity-momentum")

    assert gold_silver in {"Buy", "Sell", "Hold"}
    assert momentum in {"Buy", "Sell", "Hold"}
    assert session.requests[0][0] == ["GOLD.a", "SILVER.a"]
    assert session.requests[1][0] == ["GOLD.a", "SILVER.a", "COPPER", "BRENT"]


@pytest.mark.unit
def test_mt5_commodity_signals_refuse_non_gold_symbols_and_missing_silver():
    session = FakeSession({"GOLD"}, pd.DataFrame())

    with pytest.raises(ValueError, match="not a gold symbol"):
        mt5_commodity_signal(session, "EURUSD", "gold-silver")
    with pytest.raises(ValueError, match="needs silver"):
        mt5_commodity_signal(session, "GOLD", "gold-silver")
