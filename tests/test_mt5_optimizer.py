from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from tradingagents.commodity_strategies import commodity_momentum_signal, gold_silver_signal
from tradingagents.mt5_optimizer import (
    agree,
    commodity_momentum_daily,
    consensus,
    gold_silver_daily,
    period_stats,
    render_optimizer_note,
    run_optimizer,
    summary_numbers,
    variant_directions,
    write_optimizer_note,
)
from tradingagents.paper_strategies import signal_for_strategy

DIRECTION = {"Buy": 1, "Sell": -1, "Hold": 0}
END = 1_792_540_800  # a server-clock midnight


def _walk(count, *, seed=3, step=3600, start=END - 3600 * 240):
    rng = np.random.default_rng(seed)
    close = 2000 + np.cumsum(rng.normal(0, 4, count))
    return pd.DataFrame(
        {
            "time": start + np.arange(count) * step,
            "Open": close - rng.normal(0, 1, count),
            "High": close + np.abs(rng.normal(3, 1, count)),
            "Low": close - np.abs(rng.normal(3, 1, count)),
            "Close": close,
            "spread": 0.3,
        }
    )


@pytest.mark.unit
@pytest.mark.parametrize(
    ("variant", "strategy", "first"),
    [
        ("trend 20/50", "trend", 50),
        ("mean-reversion RSI14 30/70", "mean-reversion", 15),
        ("breakout 20", "breakout", 21),
    ],
)
def test_bar_variants_decide_like_the_live_strategies(variant, strategy, first):
    bars = _walk(240)  # under the live bot's 250-bar window, so both see the same bars
    directions = variant_directions(bars, "H1", None, None, "GOLD")[variant]

    live = [DIRECTION[signal_for_strategy(strategy, bars.iloc[: i + 1])] for i in range(first, 240)]

    assert directions[first:].tolist() == live


def _daily(days=300, seed=5):
    rng = np.random.default_rng(seed)
    common = rng.normal(0, 0.01, days)
    data = {
        "GOLD": 2000 * np.cumprod(1 + common + rng.normal(0, 0.006, days)),
        "SILVER": 25 * np.cumprod(1 + common + rng.normal(0, 0.012, days)),
        "COPPER": 4 * np.cumprod(1 + rng.normal(0, 0.01, days)),
    }
    return pd.DataFrame(data, index=END - 86_400 * np.arange(days, 0, -1))


@pytest.mark.unit
def test_daily_variants_decide_like_the_live_strategies():
    closes = _daily()
    gold_silver = gold_silver_daily(closes[["GOLD", "SILVER"]], "GOLD", 100, 1.0)
    momentum = commodity_momentum_daily(closes, "GOLD", 21, 21, 100)

    for day in range(100, len(closes)):
        signal, _ = gold_silver_signal(
            closes["GOLD"].iloc[day - 99 : day + 1],
            closes["SILVER"].iloc[day - 99 : day + 1],
            window=100,
            entry_z=1.0,
        )
        assert gold_silver.iloc[day] == DIRECTION[signal]
        signal, _ = commodity_momentum_signal(
            closes.iloc[day - 100 : day + 1], "GOLD", lookback=21, short_window=21, long_window=100
        )
        assert momentum.iloc[day] == DIRECTION[signal]


@pytest.mark.unit
def test_combinations_need_agreement():
    trend = np.array([1, 1, -1, -1, 0])
    breakout = np.array([1, -1, -1, 0, 1])
    rsi = np.array([1, -1, 1, -1, 0])

    assert consensus([trend, breakout, rsi], 2).tolist() == [1, -1, -1, -1, 0]
    assert consensus([trend, breakout, rsi], 3).tolist() == [1, 0, 0, 0, 0]
    assert agree(breakout, trend).tolist() == [1, 0, -1, 0, 0]


@pytest.mark.unit
def test_period_stats_measure_expectancy_and_drawdown():
    stats = period_stats([2.0, -1.0, -1.0, 3.0])

    assert stats["trades"] == 4
    assert stats["win_rate"] == 0.5
    assert stats["average_r"] == pytest.approx(0.75)
    assert stats["total_r"] == pytest.approx(3.0)
    assert stats["profit_factor"] == pytest.approx(2.5)
    assert stats["max_drawdown_r"] == pytest.approx(-2.0)
    assert period_stats([])["average_r"] is None


class FakeSession:
    def history_bars(self, _symbol, timeframe, count):
        step = {"H4": 14_400, "D1": 86_400}[timeframe]
        bars = _walk(count, seed=11, step=step, start=END - step * count)
        swing = 15 * np.sin(np.arange(count) / 40)
        for column in ("Open", "High", "Low", "Close"):
            bars[column] = bars[column] + swing
        return bars

    def has_symbol(self, name):
        return name in {"GOLD", "SILVER", "COPPER"}

    def daily_closes(self, symbols, count):
        return _daily(count, seed=7)[list(symbols)]


@pytest.mark.unit
def test_optimizer_ranks_on_the_selection_period_and_checks_on_the_newest(tmp_path):
    report = run_optimizer(
        FakeSession(), "GOLD", years=3, check_years=1, timeframes=("H4", "D1"), risk_percent=0.1
    )

    variants = {result.variant for result in report.results}
    assert len(variants) == 19
    assert len(report.results) == 19 * 2 * 9
    assert report.first_time < report.split_time < report.last_time
    assert any(result.check["trades"] for result in report.results)
    numbers = summary_numbers(report)
    assert numbers["best"]
    assert all(r.selection["trades"] >= 30 for r in numbers["best"])
    ranks = [r.selection["average_r"] for r in numbers["best"]]
    assert ranks == sorted(ranks, reverse=True)

    text = render_optimizer_note(
        report,
        now=datetime(2026, 10, 22, 16, 0),
        summary_note="Podsumowanie bota",
        backtest_note="Backtest GOLD H1",
    )
    assert "## 15 najlepszych w okresie doboru i ich wynik na nowych danych" in text
    assert "## Obecne ustawienia bota (H1, SL 2×ATR, TP 2R)" in text
    assert "= 342 kombinacji" in text
    path = write_optimizer_note(report, tmp_path / "Bot MT5")
    assert path.name == "Backtest GOLD optymalizacja.md"


@pytest.mark.unit
@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        (["--timeframes", "M15"], "not on M15"),
        (["--years", "3", "--check-years", "2.5"], "at least one year"),
    ],
)
def test_optimize_command_rejects_unusable_settings(arguments, message):
    from typer.testing import CliRunner

    import cli.main as cli_main

    result = CliRunner().invoke(cli_main.app, ["mt5-optimize", "GOLD", *arguments])

    assert result.exit_code == 1
    assert message in result.output
