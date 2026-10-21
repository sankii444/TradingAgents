import pandas as pd
import pytest
from typer.testing import CliRunner

import cli.main as cli_main
from tradingagents.forex_paper import ForexPaperAccount, ForexSnapshot
from tradingagents.paper_strategies import parse_paper_strategies, signal_for_strategy


@pytest.mark.unit
def test_strategy_parser_deduplicates_and_rejects_unknown_strategies():
    assert parse_paper_strategies("trend, mean-reversion,trend") == [
        "trend",
        "mean-reversion",
    ]
    with pytest.raises(ValueError, match="Unknown strategy"):
        parse_paper_strategies("trend, arbitrage")


@pytest.mark.unit
@pytest.mark.parametrize(
    "values, expected",
    [
        ([1 + index * 0.01 for index in range(60)], "Buy"),
        ([2 - index * 0.01 for index in range(60)], "Sell"),
    ],
)
def test_trend_strategy_follows_ema_alignment(values, expected):
    bars = pd.DataFrame({"High": values, "Low": values, "Close": values})
    assert signal_for_strategy("trend", bars) == expected


@pytest.mark.unit
def test_mean_reversion_strategy_signals_rsi_extremes():
    falling = [2 - index * 0.01 for index in range(20)]
    rising = [1 + index * 0.01 for index in range(20)]

    assert (
        signal_for_strategy(
            "mean-reversion",
            pd.DataFrame({"High": falling, "Low": falling, "Close": falling}),
        )
        == "Buy"
    )
    assert (
        signal_for_strategy(
            "mean-reversion",
            pd.DataFrame({"High": rising, "Low": rising, "Close": rising}),
        )
        == "Sell"
    )


@pytest.mark.unit
@pytest.mark.parametrize(
    "latest_high, latest_low, latest_close, expected",
    [
        (1.2, 1.1, 1.15, "Buy"),
        (1.05, 0.95, 0.99, "Sell"),
        (1.1, 1.0, 1.05, "Hold"),
    ],
)
def test_breakout_compares_close_to_previous_twenty_bars(
    latest_high, latest_low, latest_close, expected
):
    highs = [1.1] * 20 + [latest_high]
    lows = [1.0] * 20 + [latest_low]
    closes = [1.05] * 20 + [latest_close]
    bars = pd.DataFrame({"High": highs, "Low": lows, "Close": closes})

    assert signal_for_strategy("breakout", bars) == expected


@pytest.mark.unit
def test_strategies_reject_insufficient_history():
    bars = pd.DataFrame({"High": [1.1] * 20, "Low": [1.0] * 20, "Close": [1.05] * 20})
    with pytest.raises(ValueError, match="at least 50"):
        signal_for_strategy("trend", bars)


@pytest.mark.unit
def test_paper_trade_runs_technical_signal_in_its_own_account(monkeypatch, tmp_path):
    closes = [1.1 + index * 0.001 for index in range(60)]
    bars = pd.DataFrame(
        {
            "Date": pd.bdate_range(end="2026-10-19", periods=60),
            "Open": closes,
            "High": [price + 0.01 for price in closes],
            "Low": [price - 0.01 for price in closes],
            "Close": closes,
        }
    )
    monkeypatch.setattr(cli_main, "get_market_bars", lambda _pair, _config: bars)
    monkeypatch.setattr(
        cli_main,
        "get_market_snapshot",
        lambda *_args: pytest.fail("Technical strategies do not call the LLM strategy"),
    )
    monkeypatch.setattr(
        cli_main,
        "TradingAgentsGraph",
        lambda *_args, **_kwargs: pytest.fail("Technical strategies do not instantiate agents"),
    )
    state_path = tmp_path / "forex.sqlite3"

    result = CliRunner().invoke(
        cli_main.app,
        ["paper-trade", "EURUSD", "--strategy", "trend", "--state", str(state_path)],
    )

    assert result.exit_code == 0, result.output
    assert "opened long position" in result.output
    assert "Forex paper trading — EURUSD (trend)" in result.output
    assert state_path.with_name("forex-trend.sqlite3").is_file()
    assert not state_path.exists()


@pytest.mark.unit
def test_paper_report_compares_strategy_accounts(tmp_path):
    account_path = tmp_path / "forex-trend.sqlite3"
    account = ForexPaperAccount(account_path, initial_capital=10_000)
    account.apply_signal(
        ForexSnapshot("EURUSD", "2026-10-19", 1.1, 1.11, 1.09, 1.1, 0.01),
        "Buy",
    )

    result = CliRunner().invoke(
        cli_main.app,
        ["paper-report", "--strategies", "trend", "--state-dir", str(tmp_path)],
    )

    assert result.exit_code == 0, result.output
    assert "trend: equity" in result.output
    assert "cycles 1" in result.output
