from __future__ import annotations

from datetime import UTC, datetime
from types import SimpleNamespace

import pytest

from tradingagents.mt5_demo import MT5DemoLedger, MT5TradeResult
from tradingagents.mt5_journal import write_journal

NOW = datetime(2026, 10, 22, 1, 5, tzinfo=UTC)
FIRST_BAR = int(datetime(2026, 10, 21, 22, 0, tzinfo=UTC).timestamp())


def _log(state, bars):
    ledger = MT5DemoLedger(state)
    try:
        for bar_time, close, signals in bars:
            market = SimpleNamespace(
                symbol="GOLD",
                timeframe="H1",
                bar_time=bar_time,
                snapshot=SimpleNamespace(close=close),
            )
            for strategy, signal in signals.items():
                result = MT5TradeResult("signal only; no order sent", signal)
                ledger.record(market, strategy, result, "signal_only")
    finally:
        ledger.close()


@pytest.mark.unit
def test_journal_writes_a_note_per_day_and_a_summary(tmp_path):
    state = tmp_path / "mt5.sqlite3"
    _log(
        state,
        [
            (FIRST_BAR, 4000.0, {"trend": "Sell", "tradingview": "Buy"}),
            (FIRST_BAR + 3600, 3990.0, {"trend": "Sell", "tradingview": "Hold"}),
            (FIRST_BAR + 7200, 3995.5, {"trend": "Buy", "tradingview": "Sell"}),
        ],
    )
    journal = tmp_path / "vault" / "Bot MT5"

    changed = write_journal(state, journal, tz=UTC, now=NOW)

    first_day = journal / "Dziennik" / "Sygnały 2026-10-21.md"
    second_day = journal / "Dziennik" / "Sygnały 2026-10-22.md"
    summary = journal / "Podsumowanie bota.md"
    assert set(changed) == {first_day, second_day, summary}
    day = first_day.read_text(encoding="utf-8")
    assert (
        "| Świeca | Zamknięcie | trend | tradingview | Zgodność | Zmiana do następnej świecy |"
        in day
    )
    assert "| 22:00 | 4000,00 | 🔴 SELL | 🟢 BUY | BUY 1 · SELL 1 | -10,00 |" in day
    assert "| 23:00 | 3990,00 | 🔴 SELL | ⚪ HOLD | SELL 1 · HOLD 1 | +5,50 |" in day
    assert "[[Podsumowanie bota]]" in day
    assert "| 00:00 | 3995,50 | 🟢 BUY | 🔴 SELL | BUY 1 · SELL 1 | — |" in second_day.read_text(
        encoding="utf-8"
    )
    text = summary.read_text(encoding="utf-8")
    assert "| trend | 1 | 2 | 0 | 2 | 1 | 50% | +2,25 |" in text
    assert "| tradingview | 1 | 1 | 1 | 1 | 0 | 0% | -10,00 |" in text
    assert "Zlecenia: brak" in text
    assert text.index("[[Sygnały 2026-10-22]]") < text.index("[[Sygnały 2026-10-21]]")


@pytest.mark.unit
def test_journal_leaves_unchanged_notes_alone(tmp_path):
    state = tmp_path / "mt5.sqlite3"
    _log(state, [(FIRST_BAR, 4000.0, {"trend": "Sell"})])
    journal = tmp_path / "Bot MT5"
    write_journal(state, journal, tz=UTC, now=NOW)

    assert write_journal(state, journal, tz=UTC, now=NOW) == []


@pytest.mark.unit
def test_journal_summary_links_backtest_notes(tmp_path):
    journal = tmp_path / "Bot MT5"
    journal.mkdir()
    (journal / "Backtest GOLD H1.md").write_text("# Backtest", encoding="utf-8")

    write_journal(tmp_path / "missing.sqlite3", journal, tz=UTC, now=NOW)

    assert "- [[Backtest GOLD H1]]" in (journal / "Podsumowanie bota.md").read_text(
        encoding="utf-8"
    )


@pytest.mark.unit
def test_journal_explains_an_empty_log(tmp_path):
    journal = tmp_path / "Bot MT5"

    changed = write_journal(tmp_path / "missing.sqlite3", journal, tz=UTC, now=NOW)

    assert changed == [journal / "Podsumowanie bota.md"]
    assert "Brak zapisanych cykli" in changed[0].read_text(encoding="utf-8")


@pytest.mark.unit
def test_demo_runner_updates_the_journal_and_survives_its_errors(monkeypatch, tmp_path):
    from typer.testing import CliRunner

    import cli.main as cli_main
    import tradingagents.mt5_journal as mt5_journal

    calls = []

    def failing_journal(state_path, journal_dir):
        calls.append((state_path, journal_dir))
        raise OSError("vault is read-only")

    def stop_after_first_round(_seconds):
        raise KeyboardInterrupt

    monkeypatch.setattr(cli_main, "_run_mt5_demo_cycle", lambda *args, **kwargs: None)
    monkeypatch.setattr(cli_main.time, "sleep", stop_after_first_round)
    monkeypatch.setattr(mt5_journal, "write_journal", failing_journal)
    state = tmp_path / "mt5.sqlite3"
    journal = str(tmp_path / "Bot MT5")

    result = CliRunner().invoke(
        cli_main.app,
        [
            "mt5-demo-run",
            "GOLD",
            "--strategies",
            "trend,gold-silver",
            "--every-hour",
            "--state",
            str(state),
            "--journal",
            journal,
        ],
    )

    assert result.exit_code == 0, result.output
    assert calls == [(state, journal)]
    assert "Could not update the journal" in result.output


@pytest.mark.unit
def test_journal_command_writes_the_notes(tmp_path):
    from typer.testing import CliRunner

    import cli.main as cli_main

    state = tmp_path / "mt5.sqlite3"
    _log(state, [(FIRST_BAR, 4000.0, {"trend": "Sell"})])
    journal = tmp_path / "Bot MT5"

    result = CliRunner().invoke(
        cli_main.app, ["mt5-demo-journal", str(journal), "--state", str(state)]
    )

    assert result.exit_code == 0, result.output
    assert (journal / "Podsumowanie bota.md").is_file()
    assert len(list((journal / "Dziennik").glob("*.md"))) == 1
