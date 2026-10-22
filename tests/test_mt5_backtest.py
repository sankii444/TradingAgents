from __future__ import annotations

from datetime import UTC, datetime
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import tradingagents.mt5_backtest as backtest
from tradingagents.mt5_backtest import (
    BacktestReport,
    Trade,
    daily_signals,
    render_backtest_note,
    run_mt5_backtest,
    simulate,
    summarize,
    write_backtest_note,
)

MIDNIGHT = 1_792_540_800  # a server-clock midnight


def _bars(prices, *, spread=0.1, start=MIDNIGHT):
    return pd.DataFrame(
        [
            {
                "time": start + k * 3600,
                "Open": o,
                "High": h,
                "Low": low,
                "Close": c,
                "spread": spread,
            }
            for k, (o, h, low, c) in enumerate(prices)
        ]
    )


def _run(bars, signals, *, atr=1.0):
    return simulate(
        bars,
        lambda i: signals.get(i, "Hold"),
        lambda _i: atr,
        capital=10_000.0,
        risk_percent=1.0,
        start=0,
    )


@pytest.mark.unit
def test_long_buys_at_the_ask_and_takes_profit_at_two_r():
    bars = _bars([(100, 100.5, 99.5, 100), (100, 101, 99.5, 100.5), (101, 104.5, 100.8, 104)])

    [trade] = _run(bars, {0: "Buy"})

    assert trade.entry == pytest.approx(100.1)
    assert (trade.stop, trade.take_profit) == (pytest.approx(98.1), pytest.approx(104.1))
    assert (trade.reason, trade.exit) == ("TP", pytest.approx(104.1))
    assert trade.r_multiple == pytest.approx(2.0)
    assert trade.profit == pytest.approx(200.0)


@pytest.mark.unit
def test_stop_and_target_distances_follow_the_chosen_multiples():
    bars = _bars([(100, 100.5, 99.5, 100), (100, 101, 99.5, 100.5), (101, 104.5, 100.8, 104)])

    [trade] = simulate(
        bars,
        lambda i: "Buy" if i == 0 else "Hold",
        lambda _i: 1.0,
        capital=10_000.0,
        risk_percent=1.0,
        start=0,
        stop_atr=1.5,
        take_profit_r=3.0,
    )

    assert (trade.stop, trade.take_profit) == (pytest.approx(98.6), pytest.approx(104.6))
    assert trade.reason == "end"


@pytest.mark.unit
def test_short_stopped_by_a_gap_fills_at_the_ask_open():
    bars = _bars([(100, 100.5, 99.5, 100), (100, 100.5, 99.6, 100.2), (103, 103.5, 102.8, 103)])

    [trade] = _run(bars, {0: "Sell"})

    assert trade.entry == pytest.approx(100.0)
    assert (trade.reason, trade.exit) == ("SL", pytest.approx(103.1))
    assert trade.r_multiple == pytest.approx(-1.55)


@pytest.mark.unit
def test_a_bar_touching_stop_and_target_counts_as_the_stop():
    bars = _bars([(100, 100.5, 99.5, 100), (100, 105, 98, 101)])

    [trade] = _run(bars, {0: "Buy"})

    assert trade.reason == "SL"
    assert trade.r_multiple == pytest.approx(-1.0)


@pytest.mark.unit
def test_one_entry_a_day_and_an_open_position_closes_when_history_ends():
    day = [
        (100, 100.5, 99.5, 100),
        (100, 100.2, 97, 97.5),
        (97.5, 98, 97, 97.8),
        (98, 98, 97.5, 97.9),
    ]
    next_day = [(98, 98.4, 97.8, 98.2), (98.2, 99, 98, 98.9), (99, 99.5, 98.7, 99.4)]
    bars = pd.concat([_bars(day), _bars(next_day, start=MIDNIGHT + 86_400)], ignore_index=True)

    trades = _run(bars, dict.fromkeys(range(len(bars)), "Buy"))

    assert [trade.reason for trade in trades] == ["SL", "end"]
    assert trades[1].entry_time == MIDNIGHT + 86_400 + 3600
    assert trades[1].exit == pytest.approx(99.4)


@pytest.mark.unit
def test_daily_signals_use_only_days_closed_before_the_bar_closes(monkeypatch):
    monkeypatch.setattr(backtest, "history_days", lambda _strategy: 2 + backtest.EXTRA_DAYS)
    monkeypatch.setattr(
        backtest, "commodity_signal", lambda _strategy, window, _symbol: (str(window.index[-1]), {})
    )
    days = [MIDNIGHT + k * 86_400 for k in range(5)]
    closes = pd.DataFrame({"GOLD": np.arange(5.0), "SILVER": np.arange(5.0)}, index=days)
    bars = _bars([(1, 1, 1, 1)] * 3, start=days[3] - 3600)  # 23:00, 00:00 and 01:00 around day 3

    signal_at = daily_signals(bars, "H1", closes, "GOLD", "gold-silver")

    assert [signal_at(i) for i in range(3)] == [str(days[2]), str(days[2]), str(days[2])]
    late = _bars([(1, 1, 1, 1)], start=days[4] - 3600)
    assert daily_signals(late, "H1", closes, "GOLD", "gold-silver")(0) == str(days[3])


def _trade(profit, r_multiple, exit_time):
    return Trade(1, exit_time - 3600, 1.0, 0.9, 1.2, exit_time, 1.0, "TP", r_multiple, profit)


@pytest.mark.unit
def test_summary_measures_wins_profit_factor_drawdown_and_years():
    in_2025 = int(datetime(2025, 6, 1, tzinfo=UTC).timestamp())
    in_2026 = int(datetime(2026, 6, 1, tzinfo=UTC).timestamp())
    trades = [_trade(200, 2, in_2025), _trade(-100, -1, in_2025), _trade(-100, -1, in_2026)]

    result = summarize(trades, 10_000.0)

    assert result["trades"] == 3
    assert result["win_rate"] == pytest.approx(1 / 3)
    assert result["profit"] == pytest.approx(0.0)
    assert result["profit_factor"] == pytest.approx(1.0)
    assert result["max_drawdown"] == pytest.approx(-200 / 10_200)
    assert result["by_year"] == {2025: pytest.approx(100.0), 2026: pytest.approx(-100.0)}


def _report():
    exit_time = MIDNIGHT + 7200
    return BacktestReport(
        symbol="GOLD",
        timeframe="H1",
        first_time=MIDNIGHT,
        last_time=MIDNIGHT + 86_400,
        bar_count=24,
        capital=100_000.0,
        currency="EUR",
        risk_percent=0.1,
        buy_and_hold=0.125,
        trades={
            "trend": [_trade(200.0, 2.0, exit_time)],
            "breakout": [_trade(-100.0, -1.0, exit_time)],
        },
    )


@pytest.mark.unit
def test_results_note_ranks_strategies_and_lists_their_trades(tmp_path):
    text = render_backtest_note(
        _report(), now=datetime(2026, 10, 22, 14, 0), summary_note="Podsumowanie bota"
    )

    assert "| trend | 1 | 100,0% | +2,00 | +2,0 | ∞ | +200,00 EUR | +0,2% | +0,0% |" in text
    assert text.index("| trend |") < text.index("| breakout |")
    assert "**ryzyko:** 0,1% kapitału" in text
    assert "(kup i trzymaj, bez dźwigni):** +12,5%" in text
    assert "> [!example]- trend: ostatnie 1 z 1" in text
    assert "**tradingview**" in text and "[[Podsumowanie bota]]" in text

    path = write_backtest_note(_report(), tmp_path / "Bot MT5")
    assert path.name == "Backtest GOLD H1.md"
    assert "| Rok | trend | breakout |" in path.read_text(encoding="utf-8")


class FakeSession:
    def __init__(self):
        self.days = [MIDNIGHT - k * 86_400 for k in range(1500, 0, -1)]

    def history_bars(self, _symbol, _timeframe, count):
        hours = np.arange(min(count, 700))
        prices = 2000 + 30 * np.sin(hours / 25)
        return pd.DataFrame(
            {
                "time": MIDNIGHT - 700 * 3600 + hours * 3600,
                "Open": prices,
                "High": prices + 3,
                "Low": prices - 3,
                "Close": prices + 0.5,
                "spread": 0.3,
            }
        )

    def account_details(self):
        return SimpleNamespace(equity=10_000.0, currency="EUR")

    def has_symbol(self, name):
        return name in {"GOLD", "SILVER"}

    def daily_closes(self, symbols, _count):
        steps = np.arange(len(self.days))
        data = {"GOLD": 2000 + 40 * np.sin(steps / 40), "SILVER": 25 + np.cos(steps / 30)}
        return pd.DataFrame({name: data[name] for name in symbols}, index=self.days)


@pytest.mark.unit
def test_backtest_runs_bar_and_daily_strategies_from_one_session():
    report = run_mt5_backtest(
        FakeSession(), "GOLD", "H1", ["trend", "gold-silver"], years=0.1, risk_percent=0.5
    )

    assert set(report.trades) == {"trend", "gold-silver"}
    assert report.trades["trend"], "the sine wave should give the trend strategy trades"
    assert report.bar_count == 700 - backtest.WARMUP_BARS
    assert (report.capital, report.currency) == (10_000.0, "EUR")


@pytest.mark.unit
def test_backtest_command_refuses_strategies_without_history():
    from typer.testing import CliRunner

    import cli.main as cli_main

    result = CliRunner().invoke(
        cli_main.app, ["mt5-backtest", "GOLD", "--strategies", "trend,tradingview"]
    )

    assert result.exit_code == 1
    assert "TradingView does not publish" in result.output
