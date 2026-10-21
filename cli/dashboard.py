"""Loopback-only web dashboard for reviewing and executing MT5 demo signals."""

from __future__ import annotations

import json
import logging
import mimetypes
import os
import re
import threading
import webbrowser
from datetime import UTC, datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit

from tradingagents.default_config import DEFAULT_CONFIG

logger = logging.getLogger(__name__)
_ACTION_LOCK = threading.Lock()
_ASSET_NAMES = {"dashboard.html", "dashboard.css", "dashboard.js"}


def _default_state_path() -> Path:
    return Path(DEFAULT_CONFIG["results_dir"]) / "paper" / "mt5-demo.sqlite3"


def _normalize_server(server: str | None) -> str:
    return re.sub(r"[^a-z0-9]", "", (server or "").lower())


def _server_matches(server: str | None, required_server: str | None) -> bool:
    """Any verified demo server is allowed unless the dashboard was pinned to one."""
    if not required_server:
        return True
    return _normalize_server(required_server) in _normalize_server(server)


def _parse_request(
    payload: dict,
) -> tuple[str, str, str, str | None, str | None, float, float]:
    from tradingagents.mt5_demo import MT5_STRATEGIES, TIMEFRAMES

    symbol = payload.get("symbol")
    timeframe = payload.get("timeframe", "H1")
    strategy = payload.get("strategy", "tradingview")
    if not isinstance(symbol, str) or not symbol.strip():
        raise ValueError("Enter the exact symbol shown in MT5 Market Watch.")
    if not isinstance(timeframe, str) or timeframe.upper() not in TIMEFRAMES:
        raise ValueError(f"Timeframe must be one of: {', '.join(TIMEFRAMES)}.")
    if not isinstance(strategy, str) or strategy not in MT5_STRATEGIES:
        raise ValueError(f"Strategy must be one of: {', '.join(MT5_STRATEGIES)}.")
    analysis_symbol = payload.get("analysis_symbol") or None
    analysis_exchange = payload.get("analysis_exchange") or None
    if analysis_symbol is not None and not isinstance(analysis_symbol, str):
        raise ValueError("Analysis symbol must be text.")
    if analysis_exchange is not None and not isinstance(analysis_exchange, str):
        raise ValueError("Analysis exchange must be text.")
    if analysis_symbol and not re.fullmatch(r"[A-Za-z0-9._=+-]{1,32}", analysis_symbol.strip()):
        raise ValueError("Analysis symbol contains unsupported characters.")
    if analysis_exchange and not re.fullmatch(r"[A-Za-z0-9_-]{1,32}", analysis_exchange.strip()):
        raise ValueError("Analysis exchange contains unsupported characters.")
    try:
        risk_percent = float(payload.get("risk_percent", 0.25))
        max_lot = float(payload.get("max_lot", 0.1))
    except (TypeError, ValueError) as exc:
        raise ValueError("Risk and maximum lot must be numbers.") from exc
    if not 0.1 <= risk_percent <= 1.0:
        raise ValueError("Risk must be between 0.1% and 1.0%.")
    if not 0.01 <= max_lot <= 0.1:
        raise ValueError("Maximum volume must be between 0.01 and 0.1 lots.")
    return (
        symbol.strip(),
        timeframe.upper(),
        strategy,
        analysis_symbol.strip() if analysis_symbol else None,
        analysis_exchange.strip() if analysis_exchange else None,
        risk_percent,
        max_lot,
    )


def _market_payload(market, signal: str, state: dict, result) -> dict:
    return {
        "symbol": market.symbol,
        "timeframe": market.timeframe,
        "bar_time": market.bar_time,
        "bar_date": market.snapshot.date,
        "price": market.snapshot.close,
        "signal": signal,
        "strategy_rating": state.get("tradingview_recommendation"),
        "action": result.action,
        "volume": result.volume,
        "stop": result.stop_price,
        "take_profit": result.take_profit,
        "risk_amount": result.risk_amount,
        "currency": market.currency,
        "ticket": result.ticket,
    }


def analyze_signal(payload: dict, state_path: Path | None = None) -> dict:
    """Generate and persist a signal only; no order can be sent by this endpoint."""
    from cli.main import _mt5_signal
    from tradingagents.mt5_demo import MT5DemoSession

    (
        symbol,
        timeframe,
        strategy,
        analysis_symbol,
        analysis_exchange,
        risk_percent,
        max_lot,
    ) = _parse_request(payload)
    db_path = state_path or _default_state_path()
    with MT5DemoSession() as session:
        market = session.market_data(symbol, timeframe)
        signal, state = _mt5_signal(
            market,
            strategy,
            analysis_symbol,
            analysis_exchange=analysis_exchange,
        )
        result = session.evaluate_signal(
            market,
            signal,
            strategy=strategy,
            risk_percent=risk_percent,
            max_lot=max_lot,
            execute=False,
            state_path=db_path,
        )
    return _market_payload(market, signal, state, result)


def execute_reviewed_signal(
    payload: dict,
    state_path: Path | None = None,
    *,
    required_server: str | None = None,
) -> dict:
    """Execute only the matching signal already reviewed for this closed bar."""
    from tradingagents.mt5_demo import MT5DemoLedger, MT5DemoSession

    (
        symbol,
        timeframe,
        strategy,
        _analysis_symbol,
        _analysis_exchange,
        risk_percent,
        max_lot,
    ) = _parse_request(payload)
    db_path = state_path or _default_state_path()
    with MT5DemoSession() as session:
        account = session.account_details()
        if not _server_matches(str(account.server), required_server):
            raise ValueError(
                f"The connected MT5 demo server {account.server} does not match the required "
                f"server {required_server!r}. Sign in to that demo server in MT5 before "
                "executing an order."
            )
        market = session.market_data(symbol, timeframe)
        ledger = MT5DemoLedger(db_path)
        try:
            preview = ledger.get_cycle(market.symbol, market.timeframe, market.bar_time, strategy)
        finally:
            ledger.close()
        if preview is None or preview["status"] != "signal_only":
            raise ValueError(
                "No reviewed signal exists for the current closed bar. Analyze it again first."
            )
        result = session.evaluate_signal(
            market,
            preview["signal"],
            strategy=strategy,
            risk_percent=risk_percent,
            max_lot=max_lot,
            execute=True,
            state_path=db_path,
        )
    return _market_payload(market, preview["signal"], {}, result)


def read_dashboard_status(
    state_path: Path | None = None,
    *,
    required_server: str | None = None,
) -> dict:
    from tradingagents.mt5_demo import MT5DemoLedger, MT5DemoSession

    db_path = state_path or _default_state_path()
    status = {
        "connected": False,
        "demo": False,
        "server": None,
        "currency": None,
        "required_server": required_server,
        "server_allowed": False,
        "trading_enabled": False,
        "symbols": [],
    }
    try:
        with MT5DemoSession() as session:
            account = session.account_details()
            status.update(
                connected=True,
                demo=True,
                server=str(account.server),
                server_allowed=_server_matches(str(account.server), required_server),
                trading_enabled=session.trading_enabled(),
                currency=str(account.currency),
                equity=float(account.equity),
                symbols=session.market_watch_symbols(),
            )
    except RuntimeError as exc:
        status["error"] = str(exc)

    status["cycles"] = []
    if db_path.is_file():
        ledger = MT5DemoLedger(db_path)
        try:
            status["cycles"] = [
                {
                    "symbol": row[0],
                    "timeframe": row[1],
                    "strategy": row[2],
                    "signal": row[3],
                    "status": row[4],
                    "time": datetime.fromtimestamp(row[5], UTC).isoformat(),
                    "ticket": row[6],
                }
                for row in ledger.recent_cycles(30)
            ]
        finally:
            ledger.close()
    return status


class _DashboardHTTPServer(ThreadingHTTPServer):
    # Windows SO_REUSEADDR lets a second dashboard bind a busy port silently;
    # two processes would then not share the order lock.
    allow_reuse_address = os.name != "nt"


def create_dashboard_server(
    *,
    port: int = 8765,
    required_server: str | None = None,
) -> ThreadingHTTPServer:
    """Build the loopback-only web server."""
    if required_server is not None and not _normalize_server(required_server):
        raise ValueError("The required MT5 server name must contain letters or digits.")
    static_dir = Path(__file__).parent / "static"

    class DashboardHandler(BaseHTTPRequestHandler):
        server_version = "TradingAgentsDashboard"

        def _send(self, status: int, content_type: str, body: bytes):
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header(
                "Content-Security-Policy",
                "default-src 'self'; style-src 'self'; script-src 'self'; connect-src 'self'; base-uri 'none'; frame-ancestors 'none'",
            )
            self.end_headers()
            self.wfile.write(body)

        def _json(self, status: int, value: dict):
            self._send(
                status,
                "application/json; charset=utf-8",
                json.dumps(value, ensure_ascii=True).encode("utf-8"),
            )

        def _is_local_request(self) -> bool:
            if self.client_address[0] not in {"127.0.0.1", "::1"}:
                return False
            active_port = self.server.server_port
            host = self.headers.get("Host", "").lower()
            if host not in {
                f"127.0.0.1:{active_port}",
                f"localhost:{active_port}",
                f"[::1]:{active_port}",
            }:
                return False
            origin = self.headers.get("Origin")
            return origin is None or origin.lower() in {
                f"http://127.0.0.1:{active_port}",
                f"http://localhost:{active_port}",
                f"http://[::1]:{active_port}",
            }

        def do_GET(self):
            if not self._is_local_request():
                self._json(403, {"error": "Local browser access only."})
                return
            route = urlsplit(self.path).path
            if route == "/api/status":
                try:
                    with _ACTION_LOCK:
                        self._json(200, read_dashboard_status(required_server=required_server))
                except Exception as exc:
                    logger.exception("Could not read dashboard status")
                    self._json(500, {"error": str(exc)})
                return
            asset = "dashboard.html" if route == "/" else route.lstrip("/")
            if asset not in _ASSET_NAMES:
                self._json(404, {"error": "Not found."})
                return
            try:
                content = (static_dir / asset).read_bytes()
            except OSError:
                logger.exception("Could not read dashboard asset %s", asset)
                self._json(500, {"error": "Dashboard asset could not be read."})
                return
            content_type = mimetypes.guess_type(asset)[0] or "application/octet-stream"
            charset = (
                "; charset=utf-8"
                if content_type.startswith(("text/", "application/javascript"))
                else ""
            )
            self._send(200, content_type + charset, content)

        def do_POST(self):
            if not self._is_local_request():
                self._json(403, {"error": "Local browser access only."})
                return
            if self.headers.get_content_type() != "application/json":
                self._json(415, {"error": "Send a JSON request."})
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
            except ValueError:
                self._json(400, {"error": "Invalid request length."})
                return
            if not 0 < length <= 8192:
                self._json(413, {"error": "Request body must be between 1 and 8192 bytes."})
                return
            try:
                payload = json.loads(self.rfile.read(length))
                if not isinstance(payload, dict):
                    raise ValueError("The request body must be a JSON object.")
                with _ACTION_LOCK:
                    if self.path == "/api/analyze":
                        result = analyze_signal(payload)
                    elif self.path == "/api/execute":
                        result = execute_reviewed_signal(payload, required_server=required_server)
                    else:
                        self._json(404, {"error": "Not found."})
                        return
                self._json(200, result)
            except (json.JSONDecodeError, UnicodeDecodeError, ValueError) as exc:
                self._json(400, {"error": str(exc)})
            except Exception as exc:
                logger.exception("Dashboard action failed")
                self._json(500, {"error": str(exc)})

        def log_message(self, format_string: str, *args):
            logger.info("dashboard %s - %s", self.address_string(), format_string % args)

    return _DashboardHTTPServer(("127.0.0.1", port), DashboardHandler)


def serve_dashboard(
    *,
    port: int = 8765,
    open_browser: bool = True,
    required_server: str | None = None,
) -> None:
    """Serve the dashboard on loopback; the bind address is intentionally not configurable."""
    server = create_dashboard_server(port=port, required_server=required_server)
    url = f"http://127.0.0.1:{server.server_port}/"
    print(f"TradingAgents dashboard: {url}")
    print("Local access only. Demo orders require an explicit confirmation in the browser.")
    if required_server:
        print(f"Demo orders are limited to MT5 servers matching: {required_server}")
    if open_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nTradingAgents dashboard stopped.")
    finally:
        server.server_close()
