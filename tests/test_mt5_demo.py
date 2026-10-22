from __future__ import annotations

import sqlite3
import time
from datetime import UTC, datetime
from types import SimpleNamespace

import numpy as np
import pytest

from tradingagents.mt5_demo import MT5DemoLedger, MT5DemoSession
from tradingagents.tradingview_analysis import get_tradingview_signal


class FakeMT5:
    ACCOUNT_TRADE_MODE_DEMO = 0
    TIMEFRAME_H1 = 16385
    ORDER_TYPE_BUY = 0
    ORDER_TYPE_SELL = 1
    TRADE_ACTION_DEAL = 1
    ORDER_TIME_GTC = 0
    ORDER_FILLING_IOC = 1
    TRADE_RETCODE_DONE = 10009
    TRADE_RETCODE_PLACED = 10008

    def __init__(
        self, *, trade_mode=0, trade_allowed=True, server_time_offset_seconds=0, quote_age_seconds=0
    ):
        self.sent_requests = []
        self.shutdown_called = False
        self.open_positions = ()
        self.pending_orders = ()
        self.info = SimpleNamespace(
            visible=True,
            digits=2,
            point=0.01,
            trade_stops_level=2,
            volume_step=0.01,
            volume_min=0.01,
            volume_max=100.0,
        )
        self.tick = SimpleNamespace(
            time=int(time.time() + server_time_offset_seconds - quote_age_seconds),
            ask=2000.1,
            bid=2000.0,
        )
        self.account = SimpleNamespace(
            trade_mode=trade_mode,
            trade_allowed=trade_allowed,
            equity=10_000.0,
            balance=10_000.0,
            currency="USD",
        )
        self.rates = self._rates(server_time_offset_seconds)

    @staticmethod
    def _rates(server_time_offset_seconds=0):
        final_bar = int(time.time() + server_time_offset_seconds) // 3600 * 3600 - 3600
        closes = np.linspace(1990.0, 2000.0, 60)
        return np.array(
            [
                (final_bar - (59 - index) * 3600, close, close + 0.2, close - 0.2, close, 100)
                for index, close in enumerate(closes)
            ],
            dtype=[
                ("time", "i8"),
                ("open", "f8"),
                ("high", "f8"),
                ("low", "f8"),
                ("close", "f8"),
                ("tick_volume", "i8"),
            ],
        )

    def initialize(self, **_kwargs):
        return True

    def shutdown(self):
        self.shutdown_called = True

    def last_error(self):
        return (0, "ok")

    def terminal_info(self):
        return SimpleNamespace(connected=True, trade_allowed=self.account.trade_allowed)

    def account_info(self):
        return self.account

    def symbol_info(self, _symbol):
        return self.info

    def symbol_info_tick(self, _symbol):
        return self.tick

    def symbol_select(self, _symbol, _selected):
        return True

    def copy_rates_from_pos(self, *_args):
        return self.rates

    def positions_get(self, **_kwargs):
        return self.open_positions

    def orders_get(self, **_kwargs):
        return self.pending_orders

    def order_calc_profit(self, _order_type, _symbol, volume, entry, stop):
        return -abs(entry - stop) * 100 * volume

    def order_check(self, _request):
        return SimpleNamespace(retcode=0, comment="ok")

    def order_send(self, request):
        self.sent_requests.append(request)
        return SimpleNamespace(retcode=self.TRADE_RETCODE_DONE, order=12345, comment="done")


@pytest.mark.unit
def test_refuses_real_account_and_shuts_terminal_down():
    mt5 = FakeMT5(trade_mode=1)
    with (
        pytest.raises(RuntimeError, match="only permits MT5 demo"),
        MT5DemoSession(mt5_module=mt5),
    ):
        pytest.fail("real account must never reach trading code")
    assert mt5.shutdown_called
    assert mt5.sent_requests == []


@pytest.mark.unit
def test_refuses_to_continue_if_demo_mode_cannot_be_verified():
    mt5 = FakeMT5()
    mt5.ACCOUNT_TRADE_MODE_DEMO = None

    with (
        pytest.raises(RuntimeError, match="does not expose a demo-account mode"),
        MT5DemoSession(mt5_module=mt5),
    ):
        pytest.fail("an unverifiable account must never reach trading code")

    assert mt5.shutdown_called


@pytest.mark.unit
def test_signal_only_works_when_terminal_order_permission_is_disabled(tmp_path):
    mt5 = FakeMT5(trade_allowed=False)
    with MT5DemoSession(mt5_module=mt5) as session:
        market = session.market_data("XAUUSD")
        preview = session.evaluate_signal(market, "Buy", state_path=tmp_path / "mt5.sqlite3")
        with pytest.raises(RuntimeError, match="Automated trading is disabled"):
            session.evaluate_signal(
                market,
                "Buy",
                execute=True,
                state_path=tmp_path / "mt5.sqlite3",
            )

    assert preview.action == "signal only; no order sent"
    assert mt5.sent_requests == []


@pytest.mark.unit
def test_session_reports_algo_trading_state_and_market_watch_symbols():
    mt5 = FakeMT5(trade_allowed=False)
    mt5.symbols_get = lambda: (
        SimpleNamespace(name="GOLD", visible=True),
        SimpleNamespace(name="AUDCAD", visible=False),
        SimpleNamespace(name="EURUSD", visible=True),
    )
    with MT5DemoSession(mt5_module=mt5) as session:
        assert session.trading_enabled() is False
        assert session.market_watch_symbols() == ["EURUSD", "GOLD"]


@pytest.mark.unit
def test_market_data_normalizes_broker_server_time_to_utc(tmp_path):
    mt5 = FakeMT5(server_time_offset_seconds=3 * 60 * 60)
    with MT5DemoSession(mt5_module=mt5) as session:
        market = session.market_data("XAUUSD")
        result = session.evaluate_signal(market, "Buy", state_path=tmp_path / "mt5.sqlite3")

    now = datetime.now(UTC).timestamp()
    assert market.server_utc_offset_seconds == 3 * 60 * 60
    assert 0 <= now - market.bar_time <= 2 * 60 * 60
    assert market.bars["Date"].iloc[-1].timestamp() == market.bar_time
    assert result.action == "signal only; no order sent"
    assert mt5.sent_requests == []


@pytest.mark.unit
def test_market_data_rejects_stale_quote_after_timezone_normalization():
    mt5 = FakeMT5(server_time_offset_seconds=3 * 60 * 60, quote_age_seconds=180)
    with (
        MT5DemoSession(mt5_module=mt5) as session,
        pytest.raises(RuntimeError, match="quote is stale"),
    ):
        session.market_data("XAUUSD")


@pytest.mark.unit
def test_market_data_reports_a_tick_older_than_any_timezone_as_stale():
    mt5 = FakeMT5(server_time_offset_seconds=3 * 60 * 60, quote_age_seconds=20 * 60 * 60)
    with (
        MT5DemoSession(mt5_module=mt5) as session,
        pytest.raises(RuntimeError, match="quote is stale: its time is 17 h off the current time"),
    ):
        session.market_data("GOLD")


@pytest.mark.unit
def test_demo_cycle_defaults_to_signal_only(tmp_path):
    mt5 = FakeMT5()
    with MT5DemoSession(mt5_module=mt5) as session:
        market = session.market_data("XAUUSD")
        result = session.evaluate_signal(market, "Buy", state_path=tmp_path / "mt5.sqlite3")

    assert result.action == "signal only; no order sent"
    assert result.volume == pytest.approx(0.1)
    assert result.take_profit > mt5.tick.ask
    assert result.risk_amount <= 25.0
    assert mt5.sent_requests == []
    assert mt5.shutdown_called


@pytest.mark.unit
def test_opt_in_sends_demo_order_with_stop_and_prevents_duplicate(tmp_path):
    mt5 = FakeMT5()
    state = tmp_path / "mt5.sqlite3"
    with MT5DemoSession(mt5_module=mt5) as session:
        market = session.market_data("XAUUSD", "H1")
        result = session.evaluate_signal(
            market,
            "Buy",
            risk_percent=0.25,
            execute=True,
            state_path=state,
        )
        duplicate = session.evaluate_signal(
            market,
            "Buy",
            risk_percent=0.25,
            execute=True,
            state_path=state,
        )

    assert result.action == "demo order accepted"
    assert result.ticket == 12345
    assert result.volume <= 0.1
    assert mt5.sent_requests[0]["sl"] < mt5.sent_requests[0]["price"]
    assert mt5.sent_requests[0]["tp"] > mt5.sent_requests[0]["price"]
    assert duplicate.action == "cycle already processed; no duplicate order"
    assert len(mt5.sent_requests) == 1


@pytest.mark.unit
def test_preview_is_logged_and_can_be_promoted_to_demo_order(tmp_path):
    mt5 = FakeMT5()
    state = tmp_path / "mt5.sqlite3"
    with MT5DemoSession(mt5_module=mt5) as session:
        market = session.market_data("XAUUSD")
        preview = session.evaluate_signal(market, "Buy", strategy="trend", state_path=state)
        result = session.evaluate_signal(
            market,
            "Buy",
            strategy="trend",
            execute=True,
            state_path=state,
        )

    assert preview.action == "signal only; no order sent"
    assert result.action == "demo order accepted"
    assert len(mt5.sent_requests) == 1


@pytest.mark.unit
def test_preview_signal_cannot_be_changed_and_executed_on_same_bar(tmp_path):
    mt5 = FakeMT5()
    state = tmp_path / "mt5.sqlite3"
    with MT5DemoSession(mt5_module=mt5) as session:
        market = session.market_data("XAUUSD")
        session.evaluate_signal(market, "Buy", strategy="trend", state_path=state)
        result = session.evaluate_signal(
            market,
            "Sell",
            strategy="trend",
            execute=True,
            state_path=state,
        )

    assert "signal changed since preview" in result.action
    assert mt5.sent_requests == []


@pytest.mark.unit
def test_multiple_strategies_log_independent_signals_for_same_bar(tmp_path):
    mt5 = FakeMT5()
    state = tmp_path / "mt5.sqlite3"
    with MT5DemoSession(mt5_module=mt5) as session:
        market = session.market_data("XAUUSD")
        session.evaluate_signal(market, "Buy", strategy="trend", state_path=state)
        session.evaluate_signal(market, "Sell", strategy="mean-reversion", state_path=state)

    ledger = MT5DemoLedger(state)
    try:
        assert ledger.get_cycle("XAUUSD", "H1", market.bar_time, "trend")["signal"] == "Buy"
        assert (
            ledger.get_cycle("XAUUSD", "H1", market.bar_time, "mean-reversion")["signal"] == "Sell"
        )
    finally:
        ledger.close()


@pytest.mark.unit
def test_daily_entry_limit_applies_after_a_filled_cycle(tmp_path):
    mt5 = FakeMT5()
    state = tmp_path / "mt5.sqlite3"
    with MT5DemoSession(mt5_module=mt5) as session:
        first_market = session.market_data("XAUUSD")
        first = session.evaluate_signal(first_market, "Buy", execute=True, state_path=state)

        mt5.rates["time"] += 3600
        second_market = session.market_data("XAUUSD")
        second = session.evaluate_signal(second_market, "Buy", execute=True, state_path=state)

    assert first.action == "demo order accepted"
    assert second.action == "daily entry limit reached; no new order"
    assert len(mt5.sent_requests) == 1


@pytest.mark.unit
def test_existing_account_positions_block_new_orders(tmp_path):
    mt5 = FakeMT5()
    mt5.open_positions = (SimpleNamespace(ticket=99, symbol="EURUSD", type=mt5.ORDER_TYPE_BUY),)
    with MT5DemoSession(mt5_module=mt5) as session:
        market = session.market_data("XAUUSD")
        result = session.evaluate_signal(
            market, "Buy", execute=True, state_path=tmp_path / "mt5.sqlite3"
        )

    assert result.action == "an account position is already open; no new order"
    assert mt5.sent_requests == []


@pytest.mark.unit
def test_pending_order_blocks_new_demo_entries(tmp_path):
    mt5 = FakeMT5()
    mt5.pending_orders = (SimpleNamespace(ticket=99),)
    with MT5DemoSession(mt5_module=mt5) as session:
        market = session.market_data("XAUUSD")
        result = session.evaluate_signal(
            market, "Buy", execute=True, state_path=tmp_path / "mt5.sqlite3"
        )

    assert result.action == "an account order is already pending; no new order"
    assert mt5.sent_requests == []


@pytest.mark.unit
def test_demo_runner_refuses_multiple_live_demo_strategies():
    from typer.testing import CliRunner

    import cli.main as cli_main

    result = CliRunner().invoke(
        cli_main.app,
        [
            "mt5-demo-run",
            "XAUUSD",
            "--strategies",
            "agents,trend",
            "--execute-demo",
        ],
    )

    assert result.exit_code == 1
    assert "Use one strategy per MT5 account" in result.output


@pytest.mark.unit
def test_tradingview_analysis_maps_rating_and_uses_configured_symbol_exchange():
    requests = []

    class Handler:
        def __init__(self, **kwargs):
            requests.append(kwargs)

        def get_analysis(self):
            return SimpleNamespace(summary={"RECOMMENDATION": "STRONG_BUY"})

    intervals = SimpleNamespace(INTERVAL_1_HOUR="1h")
    signal, rating = get_tradingview_signal(
        "XAUUSD",
        timeframe="H1",
        handler_class=Handler,
        interval_class=intervals,
    )

    assert signal == "Buy"
    assert rating == "STRONG_BUY"
    assert requests == [
        {"symbol": "XAUUSD", "screener": "cfd", "exchange": "OANDA", "interval": "1h"}
    ]


@pytest.mark.unit
def test_tradingview_maps_broker_symbol_names_unless_exchange_is_explicit():
    calls = []

    class Handler:
        def __init__(self, **kwargs):
            calls.append((kwargs["exchange"], kwargs["symbol"]))

        def get_analysis(self):
            return SimpleNamespace(summary={"RECOMMENDATION": "SELL"})

    intervals = SimpleNamespace(INTERVAL_1_HOUR="1h")
    for symbol in ("GOLD", "GOLD.a", "XAUUSDm", "EURUSD.a"):
        get_tradingview_signal(
            symbol, timeframe="H1", handler_class=Handler, interval_class=intervals
        )
    get_tradingview_signal(
        "GOLD", timeframe="H1", exchange="TVC", handler_class=Handler, interval_class=intervals
    )

    assert calls == [
        ("OANDA", "XAUUSD"),
        ("OANDA", "XAUUSD"),
        ("OANDA", "XAUUSD"),
        ("FX_IDC", "EURUSD"),
        ("TVC", "GOLD"),
    ]


@pytest.mark.unit
def test_mt5_ledger_migrates_prior_demo_cycle_database(tmp_path):
    state = tmp_path / "mt5.sqlite3"
    with sqlite3.connect(state) as connection:
        connection.execute(
            """
            CREATE TABLE cycles (
                symbol TEXT NOT NULL,
                timeframe TEXT NOT NULL,
                bar_time INTEGER NOT NULL,
                signal TEXT NOT NULL,
                status TEXT NOT NULL,
                ticket INTEGER,
                PRIMARY KEY (symbol, timeframe, bar_time)
            )
            """
        )
        connection.execute("INSERT INTO cycles VALUES ('XAUUSD', 'H1', 12345, 'Buy', 'filled', 42)")

    ledger = MT5DemoLedger(state)
    try:
        assert ledger.get_cycle("XAUUSD", "H1", 12345, "agents") == {
            "signal": "Buy",
            "status": "filled",
        }
        assert ledger.get_cycle("XAUUSD", "H1", 12345, "trend") is None
    finally:
        ledger.close()


@pytest.mark.unit
def test_mt5_signal_passes_tradingview_exchange(monkeypatch):
    from types import SimpleNamespace

    import cli.main as cli_main
    import tradingagents.tradingview_analysis as tradingview

    calls = {}

    def fake_signal(symbol, *, timeframe, exchange):
        calls.update(symbol=symbol, timeframe=timeframe, exchange=exchange)
        return "Buy", "STRONG_BUY"

    monkeypatch.setattr(tradingview, "get_tradingview_signal", fake_signal)
    market = SimpleNamespace(symbol="GOLD.a", timeframe="H1")

    signal, state = cli_main._mt5_signal(
        market,
        "tradingview",
        "XAUUSD",
        analysis_exchange="FX_IDC",
    )

    assert signal == "Buy"
    assert state == {"tradingview_recommendation": "STRONG_BUY"}
    assert calls == {"symbol": "XAUUSD", "timeframe": "H1", "exchange": "FX_IDC"}


@pytest.mark.unit
def test_mt5_report_displays_persisted_strategy_signals(tmp_path):
    from typer.testing import CliRunner

    import cli.main as cli_main

    mt5 = FakeMT5()
    state = tmp_path / "mt5.sqlite3"
    with MT5DemoSession(mt5_module=mt5) as session:
        market = session.market_data("XAUUSD")
        session.evaluate_signal(market, "Buy", strategy="tradingview", state_path=state)

    result = CliRunner().invoke(
        cli_main.app,
        ["mt5-demo-report", "--state", str(state)],
    )

    assert result.exit_code == 0, result.output
    assert "XAUUSD · H1 · tradingview · Buy · signal_only · 1" in result.output
