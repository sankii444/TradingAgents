from __future__ import annotations

import sqlite3

import pandas as pd
import pytest
from typer.testing import CliRunner

import cli.main as cli_main
from tradingagents.agents.context import build_instrument_context
from tradingagents.forex_paper import (
    ForexPaperAccount,
    ForexSnapshot,
    calculate_units,
    snapshot_from_frame,
    validate_forex_pair,
)
from tradingagents.graph import trading_graph


def _snapshot(day, *, open=1.1, high=1.11, low=1.09, close=1.1, atr=0.01):
    return ForexSnapshot("EURUSD", day, open, high, low, close, atr)


@pytest.mark.unit
@pytest.mark.parametrize(
    "raw, expected",
    [("EURUSD", "EURUSD"), ("eurusd=x", "EURUSD"), ("GBPUSD", "GBPUSD")],
)
def test_accepts_supported_usd_quoted_pairs(raw, expected):
    assert validate_forex_pair(raw) == expected


@pytest.mark.unit
@pytest.mark.parametrize("raw", ["USDJPY", "EURGBP", "XAUUSD", "../EURUSD", ""])
def test_rejects_unsupported_forex_symbols(raw):
    with pytest.raises(ValueError, match="supports only"):
        validate_forex_pair(raw)


@pytest.mark.unit
def test_forex_context_identifies_base_and_quote_currencies():
    context = build_instrument_context("EURUSD", "forex")
    assert "EUR/USD" in context
    assert "first currency is the base currency" in context
    assert "not company earnings" in context


@pytest.mark.unit
def test_forex_context_formats_yahoo_currency_symbol():
    context = build_instrument_context("EURUSD=X", "forex")
    assert "`EUR/USD`" in context
    assert "ticker `EURUSD=X`" in context


@pytest.mark.unit
def test_commodity_context_does_not_describe_gold_as_forex():
    context = build_instrument_context("XAUUSD", "commodity")

    assert "commodity" in context
    assert "not a company or currency pair" in context
    assert "spot foreign exchange" not in context


@pytest.mark.unit
def test_forex_context_skips_company_profile_lookup(monkeypatch):
    monkeypatch.setattr(
        trading_graph,
        "resolve_instrument_identity",
        lambda _ticker: pytest.fail("Forex must not request company metadata"),
    )
    context = trading_graph.TradingAgentsGraph.resolve_instrument_context(
        None, "EURUSD", "forex"
    )
    assert "EUR/USD" in context


@pytest.mark.unit
def test_commodity_context_skips_company_profile_lookup(monkeypatch):
    monkeypatch.setattr(
        trading_graph,
        "resolve_instrument_identity",
        lambda _ticker: pytest.fail("Commodities must not request company metadata"),
    )
    context = trading_graph.TradingAgentsGraph.resolve_instrument_context(
        None, "XAUUSD", "commodity"
    )

    assert "commodity" in context


@pytest.mark.unit
def test_snapshot_uses_daily_bars_and_calculates_atr():
    closes = [1.1 + index * 0.001 for index in range(20)]
    frame = pd.DataFrame(
        {
            "Date": pd.bdate_range("2026-09-01", periods=20),
            "Open": closes,
            "High": [price + 0.01 for price in closes],
            "Low": [price - 0.01 for price in closes],
            "Close": closes,
        }
    )

    snapshot = snapshot_from_frame("EURUSD", frame)

    assert snapshot.date == frame.iloc[-1]["Date"].date().isoformat()
    assert snapshot.close == closes[-1]
    assert snapshot.atr == pytest.approx(0.02)


@pytest.mark.unit
def test_position_sizing_respects_risk_and_one_to_one_exposure_caps():
    units = calculate_units(10_000, 0, 1.25, 0.01, 1.0)
    assert units == pytest.approx(5_000)
    capped_units = calculate_units(10_000, 8_000, 1.25, 0.01, 1.0)
    assert capped_units == pytest.approx(1_600)
    assert units * 1.25 <= 10_000


@pytest.mark.unit
def test_paper_account_opens_once_and_keeps_a_signed_position(tmp_path):
    account = ForexPaperAccount(tmp_path / "forex.sqlite3", initial_capital=10_000)

    opened = account.apply_signal(_snapshot("2026-10-19"), "Buy", risk_percent=0.5)

    assert opened.action == "opened long position"
    assert opened.units > 0
    assert opened.stop_price == pytest.approx(1.08)
    assert opened.balance == 10_000
    assert opened.equity == 10_000
    assert account.portfolio_context().positions[0].quantity == opened.units

    duplicate = account.apply_signal(_snapshot("2026-10-19"), "Sell")
    assert duplicate.duplicate
    assert duplicate.action == "already processed"
    assert account.portfolio_context().positions[0].quantity == opened.units


@pytest.mark.unit
def test_paper_account_reverses_position_and_records_realized_pnl(tmp_path):
    path = tmp_path / "forex.sqlite3"
    account = ForexPaperAccount(path, initial_capital=10_000)
    account.apply_signal(_snapshot("2026-10-19"), "Buy")

    reversed_position = account.apply_signal(
        _snapshot("2026-10-20", open=1.1, high=1.11, low=1.09, close=1.105),
        "Sell",
    )

    assert reversed_position.action == "opened short position"
    assert reversed_position.units < 0
    assert reversed_position.realized_pnl > 0
    with sqlite3.connect(path) as connection:
        events = [row[0] for row in connection.execute("SELECT event FROM trades ORDER BY id")]
    assert events == ["open_long", "close_signal", "open_short"]


@pytest.mark.unit
def test_stop_gap_is_filled_at_open_and_does_not_reenter_same_day(tmp_path):
    account = ForexPaperAccount(tmp_path / "forex.sqlite3", initial_capital=10_000)
    account.apply_signal(_snapshot("2026-10-19"), "Sell")

    stopped = account.apply_signal(
        _snapshot("2026-10-20", open=1.13, high=1.14, low=1.12, close=1.13),
        "Buy",
    )

    assert stopped.action == "stop-loss filled; no same-day re-entry"
    assert stopped.units == 0
    assert stopped.balance < 10_000
    assert account.portfolio_context().positions == []


@pytest.mark.unit
def test_risk_limit_rejects_values_over_one_percent(tmp_path):
    account = ForexPaperAccount(tmp_path / "forex.sqlite3")
    with pytest.raises(ValueError, match="at most 1.0"):
        account.apply_signal(_snapshot("2026-10-19"), "Buy", risk_percent=1.1)


@pytest.mark.unit
def test_paper_account_summary_reports_equity_and_closed_trade_results(tmp_path):
    account = ForexPaperAccount(tmp_path / "forex.sqlite3", initial_capital=10_000)
    account.apply_signal(_snapshot("2026-10-19"), "Buy")
    account.apply_signal(
        _snapshot("2026-10-20", close=1.105),
        "Sell",
    )

    summary = account.summary()

    assert summary.initial_capital == 10_000
    assert summary.realized_pnl > 0
    assert summary.completed_trades == 1
    assert summary.winning_trades == 1
    assert summary.cycles == 2
    assert summary.equity == pytest.approx(summary.balance)


@pytest.mark.unit
def test_paper_trade_command_analyzes_and_records_only_one_daily_cycle(
    monkeypatch, tmp_path
):
    calls = []

    class FakeGraph:
        def __init__(self, analysts, config):
            assert analysts == ["market", "social", "news"]

        def propagate(self, ticker, trade_date, *, asset_type, portfolio):
            calls.append((ticker, trade_date, asset_type, portfolio))
            return {"final_trade_decision": "**Rating**: Buy"}, "Buy"

    monkeypatch.setattr(
        cli_main, "get_market_snapshot", lambda pair, _config: _snapshot("2026-10-19")
    )
    monkeypatch.setattr(cli_main, "TradingAgentsGraph", FakeGraph)
    arguments = [
        "paper-trade",
        "EURUSD",
        "--state",
        str(tmp_path / "forex.sqlite3"),
    ]

    first = CliRunner().invoke(cli_main.app, arguments)
    second = CliRunner().invoke(cli_main.app, arguments)

    assert first.exit_code == 0, first.output
    assert "opened long position" in first.output
    assert second.exit_code == 0, second.output
    assert "already recorded" in second.output
    assert len(calls) == 1
    assert calls[0][2] == "forex"
    assert calls[0][3].currency == "USD"
