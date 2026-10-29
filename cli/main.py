import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import typer

from cli.display import console
from cli.models import AnalystType, AssetType
from cli.prompts import filter_analysts_for_asset_type, parse_analysts
from cli.run import run_analysis
from tradingagents.backtest import iter_grid, run_backtest, summarize
from tradingagents.default_config import DEFAULT_CONFIG
from tradingagents.forex_paper import (
    ForexPaperAccount,
    get_market_bars,
    get_market_snapshot,
    snapshot_from_frame,
    validate_forex_pair,
)
from tradingagents.graph.trading_graph import TradingAgentsGraph
from tradingagents.paper_strategies import parse_paper_strategies, signal_for_strategy
from tradingagents.portfolio import load_portfolio

# prompt_toolkit's win32 output module is importable only on Windows (it asserts
# the platform at import time), so gate on the platform rather than catching the
# failure — that way a genuinely broken prompt_toolkit on Windows still surfaces
# instead of silently disabling the handler below. Off Windows this stays an
# empty tuple, which `except` accepts and never matches (#1138).
if sys.platform == "win32":  # pragma: no cover - platform dependent
    from prompt_toolkit.output.win32 import NoConsoleScreenBufferError

    _NO_CONSOLE_ERRORS: tuple[type[BaseException], ...] = (NoConsoleScreenBufferError,)
else:
    _NO_CONSOLE_ERRORS = ()

app = typer.Typer(
    name="TradingAgents",
    help="TradingAgents CLI: Multi-Agents LLM Financial Trading Framework",
    add_completion=True,  # Enable shell completion
)


@app.callback(invoke_without_command=True)
def analyze(
    ctx: typer.Context,
    checkpoint: bool | None = typer.Option(
        None,
        "--checkpoint/--no-checkpoint",
        help="Enable/disable checkpoint-resume (save state after each node so a "
        "crashed run can resume). Omit to honor TRADINGAGENTS_CHECKPOINT_ENABLED.",
    ),
    clear_checkpoints: bool = typer.Option(
        False,
        "--clear-checkpoints",
        help="Delete all saved checkpoints before running (force fresh start).",
    ),
    portfolio: str = typer.Option(
        None,
        "--portfolio",
        help="JSON file with current holdings and cash, so the trader, risk and "
        "portfolio agents size against your actual position.",
    ),
    ticker: str = typer.Option(
        None, "--ticker", help="Ticker to analyze, e.g. NVDA or 0700.HK; skips the prompt"
    ),
    date: str = typer.Option(None, "--date", help="Analysis date, YYYY-MM-DD; skips the prompt"),
    analysts: str = typer.Option(
        None, "--analysts", help="Comma-separated analysts, e.g. market,news; skips the prompt"
    ),
    save: bool | None = typer.Option(
        None, "--save/--no-save", help="Save the report under results_dir without asking"
    ),
    show: bool | None = typer.Option(
        None, "--show/--no-show", help="Show the full report at the end without asking"
    ),
    html: bool | None = typer.Option(
        None,
        "--html/--no-html",
        help="Save the report as one HTML page too, complete_report.html, without asking (default: yes)",
    ),
):
    """Run an analysis. This is what a bare `tradingagents` does.

    Flags answer their questions; with provider, models, depth and language also
    set through TRADINGAGENTS_* variables, the run asks nothing.
    """
    if ctx.invoked_subcommand is not None:
        return
    if clear_checkpoints:
        from tradingagents.graph.checkpointer import clear_all_checkpoints

        n = clear_all_checkpoints(DEFAULT_CONFIG["data_cache_dir"])
        console.print(f"[yellow]Cleared {n} checkpoint(s).[/yellow]")
    portfolio_context = None
    if portfolio:
        try:
            portfolio_context = load_portfolio(portfolio)
        except ValueError as exc:
            console.print(f"[red]{exc}[/red]")
            raise typer.Exit(code=1) from None

    try:
        flags = {
            "ticker": ticker,
            "date": date,
            "analysts": analysts,
            "save": save,
            "show": show,
            "html": html,
        }
        run_analysis(checkpoint=checkpoint, portfolio=portfolio_context, flags=flags)
    except _NO_CONSOLE_ERRORS:
        # A terminal with no console buffer cannot host the interactive prompts.
        # Emit one actionable line on stderr instead of a prompt_toolkit
        # traceback; plain text, since rich may not render here either (#1138).
        typer.echo(
            "Error: no Windows console available. The interactive CLI needs a real "
            "console buffer — run it from Windows Terminal, PowerShell, or cmd.exe "
            "rather than a piped or embedded terminal.",
            err=True,
        )
        raise typer.Exit(code=1) from None


@app.command()
def backtest(
    tickers: str = typer.Argument(..., help="Comma-separated tickers, e.g. NVDA,AAPL"),
    start: str = typer.Option(..., "--start", help="First analysis date, YYYY-MM-DD"),
    end: str = typer.Option(..., "--end", help="Last analysis date, YYYY-MM-DD"),
    every: int = typer.Option(7, "--every", help="Days between analysis dates"),
    analysts: str = typer.Option(
        None,
        "--analysts",
        help="Comma-separated analysts to run: market, sentiment, news, fundamentals; omit for all the asset type allows",
    ),
    asset_type: str = typer.Option("stock", "--asset-type", help="stock, crypto, or forex"),
    portfolio: str = typer.Option(
        None, "--portfolio", help="JSON file with holdings and cash, held constant across the grid"
    ),
    run_id: str = typer.Option(
        None, "--run-id", help="Continue an earlier sweep: its cells are skipped and its log reused"
    ),
):
    """Score past decisions over a grid of tickers and dates."""

    try:
        dates = iter_grid(start, end, every)
        book = load_portfolio(portfolio) if portfolio else None
        kind = AssetType(asset_type.strip().lower())
        # The analysts are named and checked as for an analysis; without a
        # choice, every analyst the asset type allows runs.
        chosen = (
            parse_analysts(analysts, kind)
            if analysts
            else filter_analysts_for_asset_type(list(AnalystType), kind)
        )
    except ValueError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1) from None

    names = [t.strip() for t in tickers.split(",") if t.strip()]
    if not names:
        console.print("[red]No ticker to analyze; pass them comma-separated, e.g. NVDA,AAPL[/red]")
        raise typer.Exit(code=1)

    def show_progress(done, total, ticker, date):
        console.print(f"[dim][{done}/{total}] {ticker} {date}[/dim]")

    kwargs = {
        "asset_type": kind.value,
        "portfolio": book,
        "run_id": run_id,
        "progress": show_progress,
        "selected_analysts": [a.value for a in chosen],
    }

    try:
        result = run_backtest(names, dates, DEFAULT_CONFIG, **kwargs)
    except Exception as exc:  # a missing key or an unknown analyst is a setup error
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1) from None
    console.print(summarize(result).render())
    console.print(
        f"\nRan {result.cells_run} cells, skipped {result.skipped}. Log: {result.log_path}"
    )
    console.print(f"Continue or settle this sweep: --run-id {result.run_id}")
    for ticker, date, reason in result.failures:
        console.print(f"[yellow]failed:[/yellow] {ticker} {date}: {reason}")
    for ticker, reason in result.settlement_failures:
        console.print(f"[yellow]unsettled:[/yellow] {ticker}: {reason}")


@app.command("paper-trade")
def paper_trade(
    ticker: str = typer.Argument(
        "EURUSD", help="USD-quoted FX pair: EURUSD, GBPUSD, AUDUSD, or NZDUSD"
    ),
    capital: float = typer.Option(
        10_000.0,
        "--capital",
        min=1.0,
        help="Initial simulated account balance; used only when creating the account",
    ),
    risk_percent: float = typer.Option(
        0.5,
        "--risk-percent",
        min=0.1,
        max=1.0,
        help="Maximum account risk per new position (0.1-1.0%)",
    ),
    strategy: str = typer.Option(
        "agents",
        "--strategy",
        help="Signal strategy: agents, trend, mean-reversion, breakout, or tradingview",
    ),
    state: str = typer.Option(
        None,
        "--state",
        help="SQLite paper-account path (strategy variants use separate databases)",
    ),
):
    """Run one FX paper-trading cycle; never contacts a broker."""
    try:
        pair = validate_forex_pair(ticker)
        strategies = parse_paper_strategies(strategy)
        if len(strategies) != 1:
            raise ValueError("paper-trade accepts exactly one strategy.")
    except ValueError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1) from None

    _run_paper_cycle(
        pair,
        strategies[0],
        capital,
        risk_percent,
        Path(state or Path(DEFAULT_CONFIG["results_dir"]) / "paper" / "forex.sqlite3"),
    )


def _strategy_state_path(base_path: Path, strategy: str) -> Path:
    if strategy == "agents":
        return base_path
    return base_path.with_name(f"{base_path.stem}-{strategy}{base_path.suffix}")


def _run_paper_cycle(
    pair: str,
    strategy: str,
    capital: float,
    risk_percent: float,
    base_path: Path,
) -> None:
    state_path = _strategy_state_path(base_path, strategy)
    account = ForexPaperAccount(state_path, initial_capital=capital)
    bars = None
    if strategy == "agents":
        snapshot = get_market_snapshot(pair, DEFAULT_CONFIG)
    else:
        bars = get_market_bars(pair, DEFAULT_CONFIG)
        snapshot = snapshot_from_frame(pair, bars)

    if account.has_cycle(pair, snapshot.date):
        console.print(
            f"[yellow]Paper cycle already recorded for {pair} ({strategy}) on {snapshot.date}; "
            "no duplicate analysis or trade was made.[/yellow]"
        )
        return

    final_state = {}
    if bars is None:
        graph = TradingAgentsGraph(["market", "social", "news"], config=DEFAULT_CONFIG.copy())
        final_state, rating = graph.propagate(
            pair, snapshot.date, asset_type="forex", portfolio=account.portfolio_context()
        )
    else:
        rating = signal_for_strategy(strategy, bars)
    result = account.apply_signal(snapshot, rating, risk_percent=risk_percent)

    console.print(f"[bold]Forex paper trading — {pair} ({strategy})[/bold]")
    console.print(f"Analysis date: {snapshot.date} · indicative close: {snapshot.close:.5f}")
    console.print(f"Strategy signal: {rating}")
    console.print(f"Paper action: {result.action}")
    if result.units:
        console.print(f"Base-currency units: {result.units:,.0f} · stop: {result.stop_price:.5f}")
    console.print(f"Simulated account balance: ${result.balance:,.2f}")
    console.print(f"Marked equity: ${result.equity:,.2f}")
    if final_state.get("final_trade_decision"):
        console.print(f"\n{final_state['final_trade_decision']}")
    console.print(f"\nPaper-account database: {state_path}")
    console.print(
        "[dim]Simulation only. Daily Yahoo bars, spreads, fees, slippage, and gap risk limit realism.[/dim]"
    )


@app.command("paper-run")
def paper_run(
    tickers: str = typer.Argument(
        "EURUSD",
        help="Comma-separated USD-quoted FX pairs: EURUSD, GBPUSD, AUDUSD, or NZDUSD",
    ),
    strategies: str = typer.Option(
        "agents,trend,mean-reversion,breakout",
        "--strategies",
        help="Comma-separated strategies: agents, trend, mean-reversion, breakout, tradingview",
    ),
    run_at: str = typer.Option(
        "23:30",
        "--run-at",
        help="Daily local time in HH:MM; start the service before this time",
    ),
    capital: float = typer.Option(
        10_000.0,
        "--capital",
        min=1.0,
        help="Starting simulated balance for each strategy account",
    ),
    risk_percent: float = typer.Option(
        0.5,
        "--risk-percent",
        min=0.1,
        max=1.0,
        help="Maximum account risk per new position (0.1-1.0%)",
    ),
    state_dir: str = typer.Option(
        None,
        "--state-dir",
        help="Directory for separate SQLite paper accounts (default: results_dir/paper)",
    ),
):
    """Run selected paper strategies once daily at the configured local time."""
    try:
        pairs = list(dict.fromkeys(validate_forex_pair(ticker) for ticker in tickers.split(",")))
        selected_strategies = parse_paper_strategies(strategies)
        scheduled_time = datetime.strptime(run_at, "%H:%M").time()
    except ValueError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1) from None
    if not pairs:
        console.print("[red]Pass at least one forex pair.[/red]")
        raise typer.Exit(code=1)

    account_dir = Path(state_dir or Path(DEFAULT_CONFIG["results_dir"]) / "paper")
    base_path = account_dir / "forex.sqlite3"
    console.print(
        f"[bold]Paper bot started[/bold] — {', '.join(pairs)} · "
        f"{', '.join(selected_strategies)} · daily at {run_at} local time"
    )
    console.print(
        "[dim]Simulation only; no broker or live orders are used. Stop with Ctrl+C.[/dim]"
    )

    try:
        while True:
            now = datetime.now()
            next_run = now.replace(
                hour=scheduled_time.hour,
                minute=scheduled_time.minute,
                second=0,
                microsecond=0,
            )
            if next_run <= now:
                next_run += timedelta(days=1)
            console.print(f"Next paper cycle: {next_run:%Y-%m-%d %H:%M}")
            time.sleep(max(0, (next_run - now).total_seconds()))
            for pair in pairs:
                for selected_strategy in selected_strategies:
                    try:
                        _run_paper_cycle(
                            pair,
                            selected_strategy,
                            capital,
                            risk_percent,
                            base_path,
                        )
                    except Exception as exc:
                        console.print(
                            f"[red]Paper cycle failed for {pair} ({selected_strategy}): {exc}[/red]"
                        )
    except KeyboardInterrupt:
        console.print("\n[yellow]Paper bot stopped.[/yellow]")


@app.command("paper-report")
def paper_report(
    strategies: str = typer.Option(
        "agents,trend,mean-reversion,breakout",
        "--strategies",
        help="Comma-separated strategies to compare",
    ),
    state_dir: str = typer.Option(
        None,
        "--state-dir",
        help="Directory containing paper accounts (default: results_dir/paper)",
    ),
):
    """Compare recorded results for independent paper-strategy accounts."""
    try:
        selected_strategies = parse_paper_strategies(strategies)
    except ValueError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1) from None

    account_dir = Path(state_dir or Path(DEFAULT_CONFIG["results_dir"]) / "paper")
    base_path = account_dir / "forex.sqlite3"
    console.print("[bold]Forex paper-strategy results[/bold]")
    for selected_strategy in selected_strategies:
        state_path = _strategy_state_path(base_path, selected_strategy)
        if not state_path.is_file():
            console.print(f"{selected_strategy}: no account data yet")
            continue
        summary = ForexPaperAccount(state_path).summary()
        return_percent = (summary.equity / summary.initial_capital - 1) * 100
        win_rate = (
            f"{summary.winning_trades / summary.completed_trades:.1%}"
            if summary.completed_trades
            else "n/a"
        )
        console.print(
            f"{selected_strategy}: equity ${summary.equity:,.2f} "
            f"({return_percent:+.2f}%), realized P/L ${summary.realized_pnl:+,.2f}, "
            f"closed trades {summary.completed_trades}, win rate {win_rate}, "
            f"cycles {summary.cycles}"
        )


def _mt5_signal(
    market,
    strategy: str,
    analysis_symbol: str | None,
    *,
    analysis_exchange: str | None = None,
    session=None,
):
    from tradingagents.commodity_strategies import COMMODITY_STRATEGIES, mt5_commodity_signal

    if strategy == "tradingview":
        from tradingagents.tradingview_analysis import get_tradingview_signal

        signal, recommendation = get_tradingview_signal(
            analysis_symbol or market.symbol,
            timeframe=market.timeframe,
            exchange=analysis_exchange,
        )
        return signal, {"tradingview_recommendation": recommendation}
    if strategy in COMMODITY_STRATEGIES:
        if session is None:
            raise ValueError(f"The {strategy} strategy reads daily bars from an MT5 session.")
        return mt5_commodity_signal(session, market.symbol, strategy)
    if strategy != "agents":
        return signal_for_strategy(strategy, market.bars), {}

    ticker = analysis_symbol or market.symbol
    is_gold = ticker.upper().startswith(("XAU", "GOLD"))
    asset_type = "commodity" if is_gold else "forex"
    from tradingagents.portfolio import PortfolioContext

    portfolio = PortfolioContext(
        cash=market.equity,
        currency=market.currency,
        positions=[],
    )
    graph = TradingAgentsGraph(["market", "social", "news"], config=DEFAULT_CONFIG.copy())
    state, signal = graph.propagate(
        ticker,
        market.snapshot.date,
        asset_type=asset_type,
        portfolio=portfolio,
    )
    return signal, state


def _run_mt5_demo_cycle(
    symbol: str,
    strategy: str,
    timeframe: str,
    risk_percent: float,
    max_lot: float,
    execute_demo: bool,
    state_path: Path,
    analysis_symbol: str | None = None,
    analysis_exchange: str | None = None,
    terminal_path: str | None = None,
    side: str = "both",
):
    from tradingagents.mt5_demo import MT5DemoSession, restrict_signal

    with MT5DemoSession(terminal_path=terminal_path) as session:
        market = session.market_data(symbol, timeframe)
        signal, state = _mt5_signal(
            market,
            strategy,
            analysis_symbol,
            analysis_exchange=analysis_exchange,
            session=session,
        )
        result = session.evaluate_signal(
            market,
            signal,
            strategy=strategy,
            risk_percent=risk_percent,
            max_lot=max_lot,
            execute=execute_demo,
            state_path=state_path,
            side=side,
        )

    console.print(f"[bold]MT5 demo — {symbol} ({strategy}, {timeframe})[/bold]")
    console.print(
        f"Closed bar: {market.snapshot.date} · close: "
        f"{market.snapshot.close:.{5 if market.snapshot.close < 10 else 2}f}"
    )
    console.print(f"Analysis signal: {signal}")
    if side != "both":
        console.print(f"Side {side} only: acting on {restrict_signal(signal, side)}")
    console.print(f"Account equity: {market.equity:,.2f} {market.currency}")
    console.print(f"Action: {result.action}")
    if result.volume:
        console.print(
            f"Volume: {result.volume:g} lots · stop: {result.stop_price} · "
            f"target: {result.take_profit} · "
            f"estimated stop risk: {result.risk_amount:,.2f} {market.currency}"
        )
    if result.ticket is not None:
        console.print(f"Demo order ticket: {result.ticket}")
    if state.get("final_trade_decision"):
        console.print(f"\n{state['final_trade_decision']}")
    if state.get("tradingview_recommendation"):
        console.print(f"TradingView indicator rating: {state['tradingview_recommendation']}")
    if state.get("strategy_details"):
        console.print(f"Strategy details: {state['strategy_details']}")
    console.print("[dim]Orders are restricted to a verified MT5 demo account.[/dim]")


def _update_journal(state_path: Path, journal_dir: str) -> None:
    from tradingagents.mt5_journal import write_journal

    try:
        changed = write_journal(state_path, journal_dir)
    except Exception as exc:  # a notes problem must never stop the trading cycles
        console.print(f"[yellow]Could not update the journal in {journal_dir}: {exc}[/yellow]")
        return
    if changed:
        console.print(f"[dim]Journal: updated {len(changed)} note(s) in {journal_dir}[/dim]")


MT5_RETRY_DELAY_SECONDS = 120


def _sleep_until(moment: datetime) -> None:
    """Wait for a wall-clock time in short naps.

    time.sleep() does not count time the computer spends asleep, so one long sleep would run
    late by however long the computer slept; a nap at most a minute long notices the clock.
    """
    while (remaining := (moment - datetime.now()).total_seconds()) > 0:
        time.sleep(min(remaining, 60))


def _parse_until(value: str) -> datetime:
    """A local end time: "YYYY-MM-DD HH:MM", or "YYYY-MM-DD" for the last minute of that day."""
    text = value.strip()
    for pattern in ("%Y-%m-%d %H:%M", "%Y-%m-%d"):
        try:
            moment = datetime.strptime(text, pattern)
        except ValueError:
            continue
        return moment if pattern.endswith("%M") else moment.replace(hour=23, minute=59)
    raise ValueError(f"--until takes YYYY-MM-DD or YYYY-MM-DD HH:MM, not {value!r}.")


@app.command("mt5-demo-cycle")
def mt5_demo_cycle(
    symbol: str = typer.Argument(
        ..., help="Exact symbol in MT5 Market Watch, e.g. XAUUSD or EURUSD.a"
    ),
    strategy: str = typer.Option(
        "agents",
        "--strategy",
        help="Signal strategy: agents, trend, mean-reversion, breakout, tradingview, "
        "gold-silver, or commodity-momentum",
    ),
    timeframe: str = typer.Option(
        "H1",
        "--timeframe",
        help="Closed MT5 bar timeframe: M15, H1, H4, or D1",
    ),
    risk_percent: float = typer.Option(
        0.25,
        "--risk-percent",
        min=0.1,
        max=1.0,
        help="Maximum account risk at the stop (0.1-1.0%)",
    ),
    max_lot: float = typer.Option(
        0.1,
        "--max-lot",
        min=0.01,
        max=0.1,
        help="Hard maximum MT5 trade volume in lots (0.01-0.1)",
    ),
    execute_demo: bool = typer.Option(
        False,
        "--execute-demo",
        help="Send the order only after MT5 verifies the connected account is demo",
    ),
    analysis_symbol: str = typer.Option(
        None,
        "--analysis-symbol",
        help="TradingView/analysis ticker when the broker's symbol has a suffix, e.g. XAUUSD",
    ),
    analysis_exchange: str = typer.Option(
        None,
        "--analysis-exchange",
        help="Override TradingView exchange (defaults to OANDA for gold and FX_IDC for forex)",
    ),
    terminal: str = typer.Option(
        None,
        "--terminal",
        help="Path to terminal64.exe (or set TRADINGAGENTS_MT5_TERMINAL)",
    ),
    state: str = typer.Option(
        None,
        "--state",
        help="SQLite idempotency log (default: results_dir/paper/mt5-demo.sqlite3)",
    ),
    side: str = typer.Option(
        "both", "--side", help="Trade both directions, or only long or only short signals"
    ),
):
    """Analyze one closed MT5 bar; order sending is opt-in and demo-only."""
    from tradingagents.mt5_demo import SIDES, TIMEFRAMES, parse_mt5_strategies

    try:
        selected = parse_mt5_strategies(strategy)
        if len(selected) != 1:
            raise ValueError("mt5-demo-cycle accepts exactly one strategy.")
        if timeframe.upper() not in TIMEFRAMES:
            raise ValueError(f"Unsupported timeframe {timeframe!r}; use {', '.join(TIMEFRAMES)}.")
        if side not in SIDES:
            raise ValueError(f"Side must be one of: {', '.join(SIDES)}.")
        state_path = Path(
            state or Path(DEFAULT_CONFIG["results_dir"]) / "paper" / "mt5-demo.sqlite3"
        )
        _run_mt5_demo_cycle(
            symbol.strip(),
            selected[0],
            timeframe.upper(),
            risk_percent,
            max_lot,
            execute_demo,
            state_path,
            analysis_symbol=analysis_symbol,
            analysis_exchange=analysis_exchange,
            terminal_path=terminal,
            side=side,
        )
    except (ValueError, RuntimeError) as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1) from None


@app.command("mt5-demo-run")
def mt5_demo_run(
    symbols: str = typer.Argument(
        "XAUUSD,EURUSD",
        help="Comma-separated exact MT5 Market Watch symbols; add broker suffixes if needed",
    ),
    strategies: str = typer.Option(
        "agents,trend,mean-reversion,breakout",
        "--strategies",
        help="Comma-separated strategies to run independently",
    ),
    timeframe: str = typer.Option("H1", "--timeframe", help="Closed MT5 bar: M15, H1, H4, or D1"),
    run_at: str = typer.Option("09:00", "--run-at", help="Daily local time in HH:MM"),
    every_hour: bool = typer.Option(
        False,
        "--every-hour",
        help="Run an immediate cycle, then repeat at the start of every local hour",
    ),
    duration_days: int | None = typer.Option(
        None,
        "--duration-days",
        min=1,
        max=365,
        help="Stop the runner after this many days",
    ),
    until: str | None = typer.Option(
        None,
        "--until",
        help="Stop the runner at this local time (YYYY-MM-DD HH:MM) or at the end of this day "
        "(YYYY-MM-DD); unlike --duration-days, a restarted runner keeps the same end",
    ),
    risk_percent: float = typer.Option(
        0.25,
        "--risk-percent",
        min=0.1,
        max=1.0,
        help="Maximum account risk at the stop (0.1-1.0%)",
    ),
    max_lot: float = typer.Option(0.1, "--max-lot", min=0.01, max=0.1),
    execute_demo: bool = typer.Option(
        False,
        "--execute-demo",
        help="Enable order sending after verifying the MT5 account is demo",
    ),
    analysis_exchange: str = typer.Option(
        None,
        "--analysis-exchange",
        help="Override TradingView exchange (defaults to OANDA for gold and FX_IDC for forex)",
    ),
    analysis_symbols: str = typer.Option(
        None,
        "--analysis-symbols",
        help="Comma-separated TradingView symbols aligned with the MT5 symbols list",
    ),
    terminal: str = typer.Option(None, "--terminal", help="Path to terminal64.exe"),
    state: str = typer.Option(None, "--state", help="SQLite idempotency log path"),
    journal: str = typer.Option(
        None,
        "--journal",
        help="Folder (e.g. in an Obsidian vault) for daily signal notes and a summary, "
        "rewritten after every round",
    ),
    side: str = typer.Option(
        "both",
        "--side",
        help="Trade both directions, or only long or only short signals; one-sided cycles are "
        "logged as e.g. trend/short",
    ),
):
    """Run scheduled MT5 analysis cycles; live accounts are always rejected."""
    from tradingagents.mt5_demo import SIDES, TIMEFRAMES, parse_mt5_strategies

    try:
        if side not in SIDES:
            raise ValueError(f"Side must be one of: {', '.join(SIDES)}.")
        if duration_days is not None and until is not None:
            raise ValueError("Pass --duration-days or --until, not both.")
        end = _parse_until(until) if until is not None else None
        if end is not None and end <= datetime.now():
            raise ValueError(f"The end time {end:%Y-%m-%d %H:%M} has already passed.")
        selected_symbols = list(
            dict.fromkeys(item.strip() for item in symbols.split(",") if item.strip())
        )
        selected_strategies = parse_mt5_strategies(strategies)
        if not selected_symbols:
            raise ValueError("Pass at least one exact MT5 symbol.")
        selected_analysis_symbols = (
            [item.strip() for item in analysis_symbols.split(",")]
            if analysis_symbols is not None
            else [None] * len(selected_symbols)
        )
        if len(selected_analysis_symbols) != len(selected_symbols) or any(
            analysis_symbol == "" for analysis_symbol in selected_analysis_symbols
        ):
            raise ValueError("--analysis-symbols must provide one non-empty item per MT5 symbol.")
        if timeframe.upper() not in TIMEFRAMES:
            raise ValueError(f"Unsupported timeframe {timeframe!r}; use {', '.join(TIMEFRAMES)}.")
        if execute_demo and len(selected_strategies) != 1:
            raise ValueError(
                "Use one strategy per MT5 account when sending demo orders; "
                "multiple strategies may compete for the same positions."
            )
        scheduled_time = datetime.strptime(run_at, "%H:%M").time()
    except ValueError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1) from None

    state_path = Path(state or Path(DEFAULT_CONFIG["results_dir"]) / "paper" / "mt5-demo.sqlite3")
    schedule_description = "every hour" if every_hour else f"daily at {run_at}"
    if duration_days is not None:
        schedule_description += f" for {duration_days} days"
    elif end is not None:
        schedule_description += f" until {end:%Y-%m-%d %H:%M}"
    console.print(
        f"[bold]MT5 demo bot started[/bold] — {', '.join(selected_symbols)} · "
        f"{', '.join(selected_strategies)} · {timeframe.upper()} · {schedule_description}"
        + ("" if side == "both" else f" · {side} only")
    )
    console.print("[dim]No orders without --execute-demo; live accounts are always rejected.[/dim]")
    # Wall-clock times, so a computer that sleeps does not stretch the schedule.
    deadline = datetime.now() + timedelta(days=duration_days) if duration_days is not None else end

    def run_round(cycles):
        """Run the cycles; return those that failed for a reason that may pass (RuntimeError)."""
        transient = []
        for symbol, analysis_symbol, selected_strategy in cycles:
            try:
                _run_mt5_demo_cycle(
                    symbol,
                    selected_strategy,
                    timeframe.upper(),
                    risk_percent,
                    max_lot,
                    execute_demo,
                    state_path,
                    analysis_symbol=analysis_symbol,
                    analysis_exchange=analysis_exchange,
                    terminal_path=terminal,
                    side=side,
                )
            except Exception as exc:
                console.print(
                    f"[red]MT5 demo cycle failed for {symbol} ({selected_strategy}): {exc}[/red]"
                )
                if isinstance(exc, RuntimeError):
                    transient.append((symbol, analysis_symbol, selected_strategy))
        return transient

    def run_cycles():
        failed = run_round(
            [
                (symbol, analysis_symbol, selected_strategy)
                for symbol, analysis_symbol in zip(
                    selected_symbols, selected_analysis_symbols, strict=True
                )
                for selected_strategy in selected_strategies
            ]
        )
        if failed:
            # A terminal that is reconnecting or a stale quote usually recovers within minutes.
            console.print(
                f"[yellow]Retrying {len(failed)} failed cycle(s) in "
                f"{MT5_RETRY_DELAY_SECONDS // 60} minutes.[/yellow]"
            )
            _sleep_until(datetime.now() + timedelta(seconds=MT5_RETRY_DELAY_SECONDS))
            run_round(failed)
        if journal:
            _update_journal(state_path, journal)

    try:
        if every_hour:
            run_cycles()
        while True:
            if deadline is not None and datetime.now() >= deadline:
                console.print("[yellow]MT5 demo test duration completed.[/yellow]")
                break
            now = datetime.now()
            if every_hour:
                next_run = now.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)
            else:
                next_run = now.replace(
                    hour=scheduled_time.hour,
                    minute=scheduled_time.minute,
                    second=0,
                    microsecond=0,
                )
                if next_run <= now:
                    next_run += timedelta(days=1)
            console.print(f"Next MT5 cycle: {next_run:%Y-%m-%d %H:%M}")
            if deadline is not None and deadline <= next_run:
                _sleep_until(deadline)
                console.print("[yellow]MT5 demo test duration completed.[/yellow]")
                break
            _sleep_until(next_run)
            run_cycles()
    except KeyboardInterrupt:
        console.print("\n[yellow]MT5 demo bot stopped.[/yellow]")


@app.command("mt5-demo-journal")
def mt5_demo_journal(
    journal: str = typer.Argument(..., help="Folder for the notes, e.g. inside an Obsidian vault"),
    state: str = typer.Option(
        None,
        "--state",
        help="SQLite signal/order log (default: results_dir/paper/mt5-demo.sqlite3)",
    ),
):
    """Write the MT5 signal log as daily Markdown notes and a summary, e.g. for Obsidian."""
    import sqlite3

    from tradingagents.mt5_journal import write_journal

    state_path = Path(state or Path(DEFAULT_CONFIG["results_dir"]) / "paper" / "mt5-demo.sqlite3")
    try:
        changed = write_journal(state_path, journal)
    except (OSError, sqlite3.Error) as exc:
        console.print(f"[red]Could not write the journal: {exc}[/red]")
        raise typer.Exit(code=1) from None
    console.print(f"Journal in {journal}: {len(changed)} note(s) updated from {state_path}")


@app.command("mt5-backtest")
def mt5_backtest(
    symbol: str = typer.Argument("GOLD", help="Exact gold symbol in MT5 Market Watch"),
    strategies: str = typer.Option(
        "trend,mean-reversion,breakout,gold-silver,commodity-momentum",
        "--strategies",
        help="Comma-separated strategies to replay; tradingview and agents have no history",
    ),
    timeframe: str = typer.Option("H1", "--timeframe", help="Bars to replay: M15, H1, H4, or D1"),
    years: float = typer.Option(
        5.0, "--years", min=0.25, max=10.0, help="Years of MT5 history to replay"
    ),
    risk_percent: float = typer.Option(
        0.25, "--risk-percent", min=0.01, max=1.0, help="Balance risked per trade at the stop (%)"
    ),
    capital: float = typer.Option(
        None, "--capital", min=1.0, help="Starting balance (default: the MT5 demo account's equity)"
    ),
    journal: str = typer.Option(
        None, "--journal", help="Folder for the results note, e.g. the bot's Obsidian journal"
    ),
    terminal: str = typer.Option(None, "--terminal", help="Path to terminal64.exe"),
    side: str = typer.Option(
        "both", "--side", help="Trade both directions, or only long or only short signals"
    ),
):
    """Replay MT5 history through the strategies with the bot's stop, target and spread."""
    from tradingagents.mt5_backtest import (
        check_strategies,
        run_mt5_backtest,
        summarize,
        write_backtest_note,
    )
    from tradingagents.mt5_demo import SIDES, TIMEFRAMES, MT5DemoSession, parse_mt5_strategies

    try:
        selected = parse_mt5_strategies(strategies)
        check_strategies(selected)
        if timeframe.upper() not in TIMEFRAMES:
            raise ValueError(f"Unsupported timeframe {timeframe!r}; use {', '.join(TIMEFRAMES)}.")
        if side not in SIDES:
            raise ValueError(f"Side must be one of: {', '.join(SIDES)}.")
        console.print(
            f"Replaying about {years:g} years of {symbol} {timeframe.upper()} from MT5..."
        )
        with MT5DemoSession(terminal_path=terminal) as session:
            report = run_mt5_backtest(
                session,
                symbol.strip(),
                timeframe.upper(),
                selected,
                years=years,
                risk_percent=risk_percent,
                capital=capital,
                side=side,
            )
    except (ValueError, RuntimeError) as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1) from None

    console.print(
        f"[bold]MT5 backtest — {report.symbol} {report.timeframe}[/bold] · {report.bar_count} bars"
        f" · risk {risk_percent:g}% · start {report.capital:,.2f} {report.currency}"
        + ("" if side == "both" else f" · {side} only")
    )
    for name, trades in report.trades.items():
        result = summarize(trades, report.capital)
        win_rate = "n/a" if result["win_rate"] is None else f"{result['win_rate']:.0%}"
        console.print(
            f"{name}: {result['trades']} trades, win rate {win_rate}, {result['total_r']:+.1f} R, "
            f"{result['profit']:+,.2f} {report.currency} ({result['return']:+.2%}), "
            f"max drawdown {result['max_drawdown']:.2%}"
        )
    console.print(f"{report.symbol} buy and hold over the period: {report.buy_and_hold:+.1%}")
    if journal:
        try:
            path = write_backtest_note(report, journal)
        except OSError as exc:
            console.print(f"[red]Could not write the results note: {exc}[/red]")
            raise typer.Exit(code=1) from None
        console.print(f"Results note: {path}")


@app.command("mt5-optimize")
def mt5_optimize(
    symbol: str = typer.Argument("GOLD", help="Exact gold symbol in MT5 Market Watch"),
    years: float = typer.Option(
        8.0, "--years", min=2.0, max=10.0, help="Years of history: selection plus check"
    ),
    check_years: float = typer.Option(
        2.0, "--check-years", min=0.5, max=5.0, help="Newest years kept to check the selection"
    ),
    timeframes: str = typer.Option(
        "H1,H4,D1", "--timeframes", help="Comma-separated timeframes: H1, H4, D1"
    ),
    risk_percent: float = typer.Option(
        0.1, "--risk-percent", min=0.01, max=1.0, help="Risk per trade used to show results in %"
    ),
    journal: str = typer.Option(
        None, "--journal", help="Folder for the results note, e.g. the bot's Obsidian journal"
    ),
    terminal: str = typer.Option(None, "--terminal", help="Path to terminal64.exe"),
    sides: str = typer.Option(
        "both,long,short",
        "--sides",
        help="Comma-separated directions to compare: both, long (buy only), short (sell only)",
    ),
):
    """Try every strategy setting on older MT5 history and check the best on the newest years."""
    from tradingagents.mt5_demo import MT5DemoSession
    from tradingagents.mt5_optimizer import (
        check_settings,
        run_optimizer,
        side_numbers,
        summary_numbers,
        write_optimizer_note,
    )

    selected = tuple(
        dict.fromkeys(item.strip().upper() for item in timeframes.split(",") if item.strip())
    )
    selected_sides = tuple(
        dict.fromkeys(item.strip().lower() for item in sides.split(",") if item.strip())
    )
    try:
        check_settings(selected, years, check_years, selected_sides)
        console.print(
            f"Optimizing {symbol} on {', '.join(selected)} ({', '.join(selected_sides)}) over "
            f"about {years:g} years of MT5 history, checking on the last {check_years:g}..."
        )
        with MT5DemoSession(terminal_path=terminal) as session:
            report = run_optimizer(
                session,
                symbol.strip(),
                years=years,
                check_years=check_years,
                timeframes=selected,
                risk_percent=risk_percent,
                sides=selected_sides,
            )
    except (ValueError, RuntimeError) as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1) from None

    numbers = summary_numbers(report)

    def r_value(value):
        return "n/a" if value is None else f"{value:+.2f} R"

    console.print(
        f"[bold]{len(report.results)} combinations[/bold] — best in the selection period:"
    )
    for rank, result in enumerate(numbers["best"][:10], start=1):
        console.print(
            f"{rank}. {result.variant} · {result.side} · {result.timeframe} · "
            f"SL {result.stop_atr:g}×ATR · TP {result.take_profit_r:g}R: "
            f"selection {r_value(result.selection['average_r'])} "
            f"({result.selection['trades']}), check {r_value(result.check['average_r'])} "
            f"({result.check['trades']})"
        )
    console.print(
        f"Check-period average per trade: selection winners {r_value(numbers['best_check_r'])}, "
        f"all combinations {r_value(numbers['all_check_r'])}; {numbers['best_positive']} of "
        f"{numbers['best_checked']} winners also profit on the newer data."
    )
    if len(report.sides) > 1:
        for side, side_result in side_numbers(report).items():
            console.print(
                f"{side}: {side_result['selection_positive']} of {side_result['eligible']} "
                f"positive in selection; check average {r_value(side_result['all_check_r'])} "
                f"over all, {r_value(side_result['best_check_r'])} for its "
                f"{len(side_result['best'])} selection winners."
            )
    for problem in report.skipped:
        console.print(f"[yellow]Skipped {problem}[/yellow]")
    if journal:
        try:
            path = write_optimizer_note(report, journal)
        except OSError as exc:
            console.print(f"[red]Could not write the results note: {exc}[/red]")
            raise typer.Exit(code=1) from None
        console.print(f"Results note: {path}")


@app.command("mt5-demo-report")
def mt5_demo_report(
    state: str = typer.Option(
        None,
        "--state",
        help="SQLite signal/order log (default: results_dir/paper/mt5-demo.sqlite3)",
    ),
):
    """Summarize saved MT5 strategy signals and demo-order outcomes."""
    from tradingagents.mt5_demo import MT5DemoLedger

    state_path = Path(state or Path(DEFAULT_CONFIG["results_dir"]) / "paper" / "mt5-demo.sqlite3")
    if not state_path.is_file():
        console.print(f"No MT5 cycles are recorded yet: {state_path}")
        return

    ledger = MT5DemoLedger(state_path)
    try:
        rows = ledger.summary()
    finally:
        ledger.close()
    if not rows:
        console.print(f"No MT5 cycles are recorded yet: {state_path}")
        return

    console.print("[bold]MT5 strategy signal log[/bold]")
    console.print("Symbol · timeframe · strategy · signal · status · cycles")
    for symbol, timeframe, strategy, signal, status, count in rows:
        console.print(f"{symbol} · {timeframe} · {strategy} · {signal} · {status} · {count}")
    console.print(f"Log: {state_path}")


@app.command("dashboard")
def dashboard(
    port: int = typer.Option(8765, "--port", min=1, max=65535, help="Local dashboard port"),
    open_browser: bool = typer.Option(
        True,
        "--open-browser/--no-open-browser",
        help="Open the dashboard in your default browser",
    ),
    require_server: str = typer.Option(
        None,
        "--require-server",
        help="Only send demo orders when the MT5 server name contains this text, e.g. AdmiralsGroup",
    ),
):
    """Open the local web dashboard for MT5 demo analysis."""
    from cli.dashboard import serve_dashboard

    try:
        serve_dashboard(port=port, open_browser=open_browser, required_server=require_server)
    except (OSError, ValueError) as exc:
        console.print(f"[red]Could not start the local dashboard: {exc}[/red]")
        raise typer.Exit(code=1) from None


if __name__ == "__main__":
    app()
