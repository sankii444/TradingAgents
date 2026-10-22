"""Write the MT5 signal log as Markdown notes for a notes app such as Obsidian.

One note per day lists each candle's strategy signals and close; a summary note counts the
signals and checks every BUY/SELL against the close of the following candle. The notes are
regenerated from the log each time, so they never drift from it.
"""

from __future__ import annotations

import os
import sqlite3
from collections import Counter, defaultdict
from datetime import date, datetime, tzinfo
from pathlib import Path

from tradingagents.mt5_demo import MT5_STRATEGIES, TIMEFRAMES

SUMMARY_NOTE = "Podsumowanie bota"
DAY_FOLDER = "Dziennik"
DAY_NOTE_PREFIX = "Sygnały"
NOTICE = (
    "> Generowane automatycznie z dziennika bota TradingAgents (MT5 demo); "
    "ręczne zmiany w tej notatce zostaną nadpisane."
)
SIDES = {
    "Buy": "BUY",
    "Overweight": "BUY",
    "Sell": "SELL",
    "Underweight": "SELL",
    "Hold": "HOLD",
    "REVIEW": "HOLD",
}
SIDE_LABELS = {"BUY": "🟢 BUY", "SELL": "🔴 SELL", "HOLD": "⚪ HOLD"}
DIRECTIONS = {"BUY": 1, "SELL": -1}
STRATEGY_DESCRIPTIONS = {
    "agents": "zespół agentów TradingAgents (model językowy)",
    "trend": "EMA 20 powyżej EMA 50 = BUY, poniżej = SELL",
    "mean-reversion": "RSI 14: do 30 = BUY, od 70 = SELL",
    "breakout": "zamknięcie ponad maksimum / pod minimum 20 poprzednich świec",
    "tradingview": "zbiorcza ocena wskaźników TradingView",
    "gold-silver": "z-score relacji złoto/srebro w 5-letnim oknie kroczącym; świece dzienne",
    "commodity-momentum": "momentum albo odwrócenie złota zależnie od korelacji surowców; "
    "świece dzienne",
}
SUMMARY_HEADER = [
    "Strategia",
    "BUY",
    "SELL",
    "HOLD",
    "Ocenione",
    "Trafione",
    "Trafność",
    "Śr. zmiana w kierunku sygnału",
]


def _read_log(state_path: Path) -> tuple[list[tuple], dict[tuple[str, str, int], float]]:
    """Return the logged cycles and the known bar closes; both empty before the first cycle."""
    if not state_path.is_file():
        return [], {}
    connection = sqlite3.connect(f"{state_path.resolve().as_uri()}?mode=ro", uri=True, timeout=30)
    try:
        cycles = connection.execute(
            "SELECT symbol, timeframe, bar_time, strategy, signal, status, ticket FROM cycles"
        ).fetchall()
        try:
            rows = connection.execute(
                "SELECT symbol, timeframe, bar_time, close FROM bars"
            ).fetchall()
        except sqlite3.OperationalError:  # a log written before bar closes were stored
            rows = []
    finally:
        connection.close()
    return cycles, {
        (symbol, timeframe, bar_time): close for symbol, timeframe, bar_time, close in rows
    }


def _number(value: float | None, digits: int, *, signed: bool = False) -> str:
    if value is None:
        return "—"
    return f"{value:{'+' if signed else ''}.{digits}f}".replace(".", ",")


def _candles(count: int) -> str:
    if count == 1:
        return "1 świeca"
    if count % 10 in (2, 3, 4) and count % 100 not in (12, 13, 14):
        return f"{count} świece"
    return f"{count} świec"


def _table(header: list[str], rows: list[list[str]]) -> list[str]:
    return [
        "| " + " | ".join(header) + " |",
        "|" + " --- |" * len(header),
        *("| " + " | ".join(row) + " |" for row in rows),
    ]


def _write_if_changed(path: Path, text: str) -> bool:
    data = text.encode("utf-8")
    if path.is_file() and path.read_bytes() == data:
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    # Replace atomically so the notes app never reads a half-written note.
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_bytes(data)
    os.replace(temporary, path)
    return True


def _render_market(symbol, timeframe, bars, closes, tz):
    """Return the day-table rows per local date, the table header and the summary lines."""
    strategies = [name for name in MT5_STRATEGIES if any(name in row for row in bars.values())]
    strategies += sorted({name for row in bars.values() for name in row} - set(strategies))
    step = TIMEFRAMES[timeframe][1] if timeframe in TIMEFRAMES else 0
    known = [closes[(symbol, timeframe, bar)] for bar in bars if (symbol, timeframe, bar) in closes]
    digits = 5 if known and min(known) < 10 else 2

    tallies = {name: Counter() for name in strategies}
    moves = {name: [] for name in strategies}
    statuses = Counter()
    rows_by_day: dict[date, list[list[str]]] = defaultdict(list)
    for bar_time in sorted(bars):
        close = closes.get((symbol, timeframe, bar_time))
        following = closes.get((symbol, timeframe, bar_time + step)) if step else None
        change = following - close if close is not None and following is not None else None
        cells, sides = [], Counter()
        for name in strategies:
            if name not in bars[bar_time]:
                cells.append("—")
                continue
            signal, status, ticket = bars[bar_time][name]
            side = SIDES.get(signal, "HOLD")
            sides[side] += 1
            tallies[name][side] += 1
            statuses[status] += 1
            if change is not None and side in DIRECTIONS:
                moves[name].append(DIRECTIONS[side] * change)
            label = SIDE_LABELS[side]
            if status != "signal_only":
                label += f" ({status}{f' #{ticket}' if ticket else ''})"
            cells.append(label)
        moment = datetime.fromtimestamp(bar_time, tz)
        rows_by_day[moment.date()].append(
            [
                f"{moment:%H:%M}",
                _number(close, digits),
                *cells,
                " · ".join(
                    f"{side} {sides[side]}" for side in ("BUY", "SELL", "HOLD") if sides[side]
                ),
                _number(change, digits, signed=True),
            ]
        )

    first = datetime.fromtimestamp(min(bars), tz)
    last = datetime.fromtimestamp(max(bars), tz)
    orders = (
        "- Zlecenia: brak — wszystkie cykle to podgląd sygnału (signal_only)."
        if set(statuses) == {"signal_only"}
        else "- Statusy cykli: " + " · ".join(f"{name} {n}" for name, n in sorted(statuses.items()))
    )
    stats = []
    for name in strategies:
        evaluated = moves[name]
        hits = sum(move > 0 for move in evaluated)
        stats.append(
            [
                name,
                *(str(tallies[name][side]) for side in ("BUY", "SELL", "HOLD")),
                str(len(evaluated)),
                str(hits),
                f"{hits / len(evaluated):.0%}" if evaluated else "—",
                _number(sum(evaluated) / len(evaluated), digits, signed=True) if evaluated else "—",
            ]
        )
    summary = [
        f"## {symbol} · {timeframe}",
        "",
        f"- Ostatnia zapisana świeca: {last:%d.%m.%Y %H:%M}",
        f"- Zapisane: {_candles(len(bars))} od {first:%d.%m.%Y %H:%M}",
        orders,
        "",
        *_table(SUMMARY_HEADER, stats),
        "",
    ]
    header = ["Świeca", "Zamknięcie", *strategies, "Zgodność", "Zmiana do następnej świecy"]
    return rows_by_day, header, summary, strategies


def write_journal(
    state_path: str | Path,
    journal_dir: str | Path,
    *,
    tz: tzinfo | None = None,
    now: datetime | None = None,
) -> list[Path]:
    """Rewrite the day notes and the summary from the signal log; return the notes that changed.

    Times are shown in ``tz``, or in the computer's local time when it is None.
    """
    cycles, closes = _read_log(Path(state_path))
    journal = Path(journal_dir)
    now = now or datetime.now(tz)

    markets: dict[tuple[str, str], dict[int, dict]] = defaultdict(lambda: defaultdict(dict))
    for symbol, timeframe, bar_time, strategy, signal, status, ticket in cycles:
        markets[(symbol, timeframe)][bar_time][strategy] = (signal, status, ticket)

    summary = [
        "---",
        "tags:",
        "  - bot-mt5",
        "---",
        f"# {SUMMARY_NOTE}",
        "",
        NOTICE,
        f"> Ostatnia aktualizacja: {now:%d.%m.%Y %H:%M} (czas lokalny).",
        "",
    ]
    day_sections: dict[date, list[str]] = defaultdict(list)
    day_counts: Counter[date] = Counter()
    used: list[str] = []
    for symbol, timeframe in sorted(markets):
        rows_by_day, header, section, strategies = _render_market(
            symbol, timeframe, markets[(symbol, timeframe)], closes, tz
        )
        summary += section
        used += [name for name in strategies if name not in used]
        for day, rows in rows_by_day.items():
            day_sections[day] += [f"## {symbol} · {timeframe}", "", *_table(header, rows), ""]
            day_counts[day] += len(rows)

    if markets:
        summary += [
            "*Trafność: jak często zamknięcie następnej świecy poszło w kierunku sygnału BUY/SELL. "
            "To uproszczona miara, bez stop lossa, take profitu, spreadu i prowizji.*",
            "",
            "## Strategie",
            "",
            *(
                f"- **{name}**: {STRATEGY_DESCRIPTIONS.get(name, 'opis niedostępny')}"
                for name in used
            ),
            "",
            "## Dni",
            "",
            *(
                f"- [[{DAY_NOTE_PREFIX} {day:%Y-%m-%d}]] — {_candles(day_counts[day])}"
                for day in sorted(day_counts, reverse=True)
            ),
        ]
    else:
        summary.append("Brak zapisanych cykli: notatki pojawią się po pierwszej analizie bota.")

    changed = []
    for day, sections in sorted(day_sections.items()):
        note = [
            "---",
            f"date: {day:%Y-%m-%d}",
            "tags:",
            "  - bot-mt5",
            "---",
            f"# Sygnały bota — {day:%d.%m.%Y}",
            "",
            NOTICE,
            f"> Wróć do [[{SUMMARY_NOTE}]]. Godziny świec w czasie lokalnym; zmiana to różnica "
            "między zamknięciem następnej świecy i tej.",
            "",
            *sections,
        ]
        path = journal / DAY_FOLDER / f"{DAY_NOTE_PREFIX} {day:%Y-%m-%d}.md"
        if _write_if_changed(path, "\n".join(note).rstrip() + "\n"):
            changed.append(path)
    summary_path = journal / f"{SUMMARY_NOTE}.md"
    if _write_if_changed(summary_path, "\n".join(summary).rstrip() + "\n"):
        changed.append(summary_path)
    return changed
