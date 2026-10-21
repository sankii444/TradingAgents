const form = document.querySelector("#analysis-form");
const analyzeButton = document.querySelector("#analyze-button");
const executeButton = document.querySelector("#execute-button");
const emptyState = document.querySelector("#empty-state");
const resultPanel = document.querySelector("#analysis-result");
const errorBox = document.querySelector("#error-box");
const toast = document.querySelector("#toast");

const symbolInput = form.elements.symbol;

let reviewedPayload = null;
let latestResult = null;
let orderBlockReason = "trwa sprawdzanie statusu MT5";
let connectedServer = null;
let symbolTouched = false;

function currentPayload() {
  const values = new FormData(form);
  return {
    symbol: String(values.get("symbol") || "").trim(),
    timeframe: values.get("timeframe"),
    strategy: values.get("strategy"),
    risk_percent: Number(values.get("risk_percent")),
    max_lot: Number(values.get("max_lot")),
    analysis_symbol: String(values.get("analysis_symbol") || "").trim(),
    analysis_exchange: String(values.get("analysis_exchange") || "").trim(),
  };
}

function setBusy(button, busy, label) {
  button.disabled = busy;
  if (busy) {
    button.dataset.originalLabel = button.querySelector("span").textContent;
    button.querySelector("span").textContent = label;
  } else if (button.dataset.originalLabel) {
    button.querySelector("span").textContent = button.dataset.originalLabel;
    delete button.dataset.originalLabel;
  }
}

async function api(path, body) {
  const response = await fetch(path, {
    method: body ? "POST" : "GET",
    headers: body ? { "Content-Type": "application/json" } : {},
    body: body ? JSON.stringify(body) : undefined,
    cache: "no-store",
  });
  const result = await response.json();
  if (!response.ok) throw new Error(result.error || `Błąd serwera (${response.status}).`);
  return result;
}

function showError(message) {
  errorBox.textContent = message;
  errorBox.classList.remove("hidden");
}

function formatPrice(value) {
  if (value === null || value === undefined) return "—";
  return Number(value).toLocaleString("pl-PL", {
    minimumFractionDigits: 2,
    maximumFractionDigits: Number(value) < 10 ? 5 : 2,
  });
}

function renderResult(result) {
  latestResult = result;
  emptyState.classList.add("hidden");
  resultPanel.classList.remove("hidden");
  errorBox.classList.add("hidden");
  document.querySelector("#result-symbol").textContent = result.symbol;
  document.querySelector("#result-meta").textContent =
    `${result.timeframe} · świeca ${new Date(result.bar_time * 1000).toLocaleString("pl-PL")}`;
  document.querySelector("#result-price").textContent = formatPrice(result.price);
  document.querySelector("#result-rating").textContent = result.strategy_rating
    ? `TradingView: ${result.strategy_rating}`
    : "";
  document.querySelector("#result-volume").textContent = result.volume
    ? `${Number(result.volume).toLocaleString("pl-PL")} lot`
    : "brak wejścia";
  document.querySelector("#result-stop").textContent = formatPrice(result.stop);
  document.querySelector("#result-target").textContent = formatPrice(result.take_profit);
  document.querySelector("#result-risk").textContent = result.risk_amount
    ? `${formatPrice(result.risk_amount)} ${result.currency}`
    : "—";
  document.querySelector("#result-action").textContent = result.action;

  const pill = document.querySelector("#signal-pill");
  pill.textContent = result.signal.toUpperCase();
  pill.className = `signal-pill ${["Buy", "Overweight"].includes(result.signal) ? "" : ["Sell", "Underweight"].includes(result.signal) ? "sell" : "hold"}`;
  document.querySelector("#result-time").textContent = "Zapisano lokalnie";

  const actionable = result.action === "signal only; no order sent" && result.volume > 0;
  const canExecute = actionable && !orderBlockReason;
  executeButton.classList.toggle("hidden", !canExecute);
  document.querySelector("#execute-warning").classList.toggle("hidden", !canExecute);
  const blocked = document.querySelector("#execute-blocked");
  blocked.textContent = actionable && orderBlockReason
    ? `Zlecenie DEMO niedostępne: ${orderBlockReason}.`
    : "";
  blocked.classList.toggle("hidden", !blocked.textContent);
}

function renderSymbols(symbols) {
  const options = document.querySelector("#symbol-options");
  options.replaceChildren(...symbols.map((symbol) => new Option(symbol, symbol)));
  // Brokers name gold differently (XAUUSD, GOLD, XAUUSD.a); preselect the one this terminal offers.
  if (!symbolTouched && !reviewedPayload && symbols.length && !symbols.includes(symbolInput.value)) {
    const gold = symbols.find((symbol) => /^(XAU|GOLD)/i.test(symbol));
    if (gold) symbolInput.value = gold;
  }
}

function showToast(message) {
  toast.textContent = message;
  toast.classList.remove("hidden");
  window.setTimeout(() => toast.classList.add("hidden"), 4000);
}

async function refreshStatus() {
  const dot = document.querySelector("#status-dot");
  const label = document.querySelector("#connection-label");
  const serverName = document.querySelector("#server-name");
  const serverDetail = document.querySelector("#server-detail");
  try {
    const status = await api("/api/status");
    connectedServer = status.demo ? status.server : null;
    orderBlockReason = !status.demo
      ? "brak połączenia z kontem demo MT5"
      : !status.server_allowed
        ? `panel wymaga serwera ${status.required_server}, a MT5 jest zalogowany na ${status.server}`
        : !status.trading_enabled
          ? "włącz przycisk „Algo Trading” w terminalu MT5 i odśwież status"
          : null;
    dot.className = `status-dot ${status.demo ? "online" : "offline"}`;
    label.textContent = status.demo ? "MT5 · konto demo" : "MT5 · brak połączenia demo";
    serverName.textContent = status.demo ? status.server : "MT5 NIEPOŁĄCZONE";
    serverDetail.textContent = status.demo
      ? `Konto DEMO · ${status.currency} · ${orderBlockReason ? `zlecenia zablokowane: ${orderBlockReason}` : "zlecenia demo dostępne"}`
      : status.error || "Otwórz terminal MT5 i zaloguj konto demo";
    renderSymbols(status.symbols || []);
    if (latestResult) renderResult(latestResult);
    renderHistory(status.cycles || []);
  } catch (error) {
    connectedServer = null;
    orderBlockReason = "nie udało się odczytać statusu MT5";
    if (latestResult) renderResult(latestResult);
    dot.className = "status-dot offline";
    label.textContent = "Błąd statusu MT5";
    serverName.textContent = "SPRAWDŹ TERMINAL";
    serverDetail.textContent = error.message;
  }
}

function renderHistory(cycles) {
  const body = document.querySelector("#history-body");
  body.replaceChildren();
  if (!cycles.length) {
    const row = document.createElement("tr");
    const cell = document.createElement("td");
    cell.colSpan = 6;
    cell.className = "table-empty";
    cell.textContent = "Brak zapisanych cykli. Historia zostanie zapisana lokalnie.";
    row.append(cell);
    body.append(row);
    return;
  }
  for (const cycle of cycles) {
    const row = document.createElement("tr");
    const values = [
      cycle.symbol,
      cycle.timeframe,
      cycle.strategy,
      cycle.signal,
      cycle.status,
      new Date(cycle.time).toLocaleString("pl-PL", { timeZone: "UTC" }) + " UTC",
    ];
    for (const value of values) {
      const cell = document.createElement("td");
      cell.textContent = value;
      row.append(cell);
    }
    row.children[3].className = `table-signal ${["Buy", "Overweight"].includes(cycle.signal) ? "buy" : ["Sell", "Underweight"].includes(cycle.signal) ? "sell" : "hold"}`;
    row.children[4].className = "table-status";
    body.append(row);
  }
}

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  reviewedPayload = null;
  latestResult = null;
  executeButton.classList.add("hidden");
  document.querySelector("#execute-warning").classList.add("hidden");
  setBusy(analyzeButton, true, "Analizuję…");
  errorBox.classList.add("hidden");
  try {
    const payload = currentPayload();
    const result = await api("/api/analyze", payload);
    reviewedPayload = payload;
    renderResult(result);
    await refreshStatus();
  } catch (error) {
    showError(error.message);
  } finally {
    setBusy(analyzeButton, false);
  }
});

symbolInput.addEventListener("input", () => {
  symbolTouched = true;
});

form.addEventListener("input", () => {
  if (reviewedPayload && JSON.stringify(currentPayload()) !== JSON.stringify(reviewedPayload)) {
    reviewedPayload = null;
    latestResult = null;
    executeButton.classList.add("hidden");
    document.querySelector("#execute-warning").classList.add("hidden");
    document.querySelector("#result-action").textContent =
      "Parametry zmieniono — wykonaj nową analizę przed potwierdzeniem zlecenia.";
  }
});
form.addEventListener("change", () => {
  if (reviewedPayload && JSON.stringify(currentPayload()) !== JSON.stringify(reviewedPayload)) {
    reviewedPayload = null;
    latestResult = null;
    executeButton.classList.add("hidden");
    document.querySelector("#execute-warning").classList.add("hidden");
    document.querySelector("#result-action").textContent =
      "Parametry zmieniono — wykonaj nową analizę przed potwierdzeniem zlecenia.";
  }
});

executeButton.addEventListener("click", async () => {
  if (!reviewedPayload || !latestResult) return;
  const message =
    `Potwierdź zlecenie na koncie DEMO (${connectedServer}):\n\n` +
    `${latestResult.signal.toUpperCase()} ${latestResult.symbol}\n` +
    `Wolumen: ${latestResult.volume} lot\nStop: ${formatPrice(latestResult.stop)}\n` +
    `Ryzyko estymowane: ${formatPrice(latestResult.risk_amount)} ${latestResult.currency}\n\n` +
    "Zlecenie zostanie wysłane tylko po serwerowej weryfikacji konta demo.";
  if (!window.confirm(message)) return;

  setBusy(executeButton, true, "Wysyłam na DEMO…");
  errorBox.classList.add("hidden");
  try {
    const result = await api("/api/execute", reviewedPayload);
    reviewedPayload = null;
    renderResult(result);
    showToast(result.action);
    await refreshStatus();
  } catch (error) {
    showError(error.message);
  } finally {
    setBusy(executeButton, false);
  }
});

document.querySelector("#refresh-status").addEventListener("click", refreshStatus);
document.querySelector("#refresh-history").addEventListener("click", refreshStatus);
refreshStatus();
