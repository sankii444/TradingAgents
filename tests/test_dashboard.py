from __future__ import annotations

from email.message import Message
from pathlib import Path
from types import SimpleNamespace

import pytest

from cli.dashboard import (
    _parse_request,
    _server_matches,
    create_dashboard_server,
    execute_reviewed_signal,
    read_dashboard_status,
)
from tradingagents.mt5_demo import MT5DemoLedger, MT5TradeResult


class FakeDashboardSession:
    """Stands in for MT5DemoSession and records market-data reads and order attempts."""

    def __init__(self, *, server="AdmiralsGroup-Demo", trading_enabled=True):
        self.server = server
        self._trading_enabled = trading_enabled
        self.market_data_called = False
        self.evaluations = []

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return None

    def account_details(self):
        return SimpleNamespace(server=self.server, currency="EUR", equity=10_000.0)

    def trading_enabled(self):
        return self._trading_enabled

    def market_watch_symbols(self):
        return ["EURUSD", "GOLD"]

    def market_data(self, symbol, timeframe):
        self.market_data_called = True
        return SimpleNamespace(
            symbol=symbol,
            timeframe=timeframe,
            bar_time=1_792_573_200,
            snapshot=SimpleNamespace(date="2026-10-21", close=2000.0),
            currency="EUR",
        )

    def evaluate_signal(self, _market, signal, **kwargs):
        self.evaluations.append((signal, kwargs["execute"]))
        return MT5TradeResult("demo order accepted", signal, volume=0.1, ticket=12345)


@pytest.mark.unit
def test_dashboard_binds_only_to_loopback_and_rejects_cross_origin():
    server = create_dashboard_server(port=0)
    try:
        assert server.server_address[0] == "127.0.0.1"
        handler = object.__new__(server.RequestHandlerClass)
        handler.server = server
        handler.client_address = ("127.0.0.1", 12345)
        handler.headers = Message()
        handler.headers["Host"] = f"127.0.0.1:{server.server_port}"
        assert handler._is_local_request()

        handler.headers["Origin"] = "http://attacker.example"
        assert not handler._is_local_request()
    finally:
        server.server_close()


@pytest.mark.unit
def test_dashboard_accepts_valid_analysis_input_and_rejects_injection():
    valid = _parse_request(
        {
            "symbol": "XAUUSD",
            "timeframe": "h1",
            "strategy": "tradingview",
            "analysis_symbol": "XAUUSD",
            "analysis_exchange": "OANDA",
            "risk_percent": 0.25,
            "max_lot": 0.1,
        }
    )
    assert valid == ("XAUUSD", "H1", "tradingview", "XAUUSD", "OANDA", 0.25, 0.1)

    with pytest.raises(ValueError, match="unsupported characters"):
        _parse_request({"symbol": "XAUUSD", "analysis_symbol": "XAUUSD&injected=true"})


@pytest.mark.unit
def test_dashboard_executes_reviewed_signal_on_any_demo_server(monkeypatch, tmp_path):
    session = FakeDashboardSession(server="AdmiralsGroup-Demo")
    monkeypatch.setattr("tradingagents.mt5_demo.MT5DemoSession", lambda: session)
    state = tmp_path / "test.sqlite3"
    ledger = MT5DemoLedger(state)
    try:
        ledger.record(
            session.market_data("GOLD", "H1"),
            "trend",
            MT5TradeResult("signal only; no order sent", "Sell"),
            "signal_only",
        )
    finally:
        ledger.close()

    result = execute_reviewed_signal({"symbol": "GOLD", "strategy": "trend"}, state)

    assert result["ticket"] == 12345
    assert result["signal"] == "Sell"
    assert session.evaluations == [("Sell", True)]


@pytest.mark.unit
def test_dashboard_refuses_execution_without_reviewed_preview(monkeypatch, tmp_path):
    session = FakeDashboardSession()
    monkeypatch.setattr("tradingagents.mt5_demo.MT5DemoSession", lambda: session)

    with pytest.raises(ValueError, match="No reviewed signal"):
        execute_reviewed_signal({"symbol": "GOLD", "strategy": "trend"}, tmp_path / "test.sqlite3")
    assert session.evaluations == []


@pytest.mark.unit
def test_dashboard_server_pin_blocks_orders_on_other_demo_servers(monkeypatch, tmp_path):
    session = FakeDashboardSession(server="AdmiralsGroup-Demo")
    monkeypatch.setattr("tradingagents.mt5_demo.MT5DemoSession", lambda: session)

    with pytest.raises(ValueError, match="does not match the required server"):
        execute_reviewed_signal(
            {"symbol": "GOLD"}, tmp_path / "test.sqlite3", required_server="RoboForex"
        )
    assert not session.market_data_called
    assert session.evaluations == []
    assert _server_matches("AdmiralsGroup-Demo", None)
    assert _server_matches("AdmiralsGroup-Demo", "Admirals Group")
    assert not _server_matches("AdmiralsGroup-Demo", "RoboForex")


@pytest.mark.unit
def test_dashboard_status_reports_order_readiness_and_market_watch_symbols(monkeypatch, tmp_path):
    session = FakeDashboardSession(trading_enabled=False)
    monkeypatch.setattr("tradingagents.mt5_demo.MT5DemoSession", lambda: session)

    status = read_dashboard_status(tmp_path / "missing.sqlite3")
    pinned = read_dashboard_status(tmp_path / "missing.sqlite3", required_server="RoboForex")

    assert status["demo"] is True
    assert status["server"] == "AdmiralsGroup-Demo"
    assert status["server_allowed"] is True
    assert status["trading_enabled"] is False
    assert status["symbols"] == ["EURUSD", "GOLD"]
    assert status["cycles"] == []
    assert pinned["server_allowed"] is False
    assert pinned["required_server"] == "RoboForex"


@pytest.mark.unit
def test_dashboard_rejects_server_pin_without_letters_or_digits():
    with pytest.raises(ValueError, match="letters or digits"):
        create_dashboard_server(port=0, required_server=" - ")


@pytest.mark.unit
def test_second_dashboard_cannot_take_over_a_busy_port():
    first = create_dashboard_server(port=0)
    try:
        with pytest.raises(OSError):
            create_dashboard_server(port=first.server_port)
    finally:
        first.server_close()


@pytest.mark.unit
def test_dashboard_static_assets_use_local_resources_and_fixed_csp():
    repo_root = Path(__file__).parents[1]
    static_dir = repo_root / "cli" / "static"
    html = (static_dir / "dashboard.html").read_text(encoding="utf-8")
    css = (static_dir / "dashboard.css").read_text(encoding="utf-8")
    dashboard_source = (repo_root / "cli" / "dashboard.py").read_text(encoding="utf-8")

    assert 'href="/dashboard.css"' in html
    assert 'src="/dashboard.js"' in html
    assert "fonts.googleapis.com" not in css
    assert "style-src 'self'; script-src 'self'; connect-src 'self'" in dashboard_source
