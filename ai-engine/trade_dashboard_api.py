from __future__ import annotations

import os
from datetime import datetime, timezone

import MetaTrader5 as mt5
import psycopg
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.responses import HTMLResponse

load_dotenv()

app = FastAPI(title="Sentinel Trade Dashboard", version="1.0.0")

DB_HOST = os.getenv("JOURNAL_DB_HOST", os.getenv("MONITOR_DB_HOST", "localhost"))
DB_PORT = int(os.getenv("JOURNAL_DB_PORT", os.getenv("MONITOR_DB_PORT", "5433")))
DB_NAME = os.getenv("JOURNAL_DB_NAME", os.getenv("MONITOR_DB_NAME", "sentinel_db"))
DB_USER = os.getenv("JOURNAL_DB_USER", os.getenv("MONITOR_DB_USER", "admin"))
DB_PASSWORD = os.getenv("JOURNAL_DB_PASSWORD", os.getenv("MONITOR_DB_PASSWORD", "admin"))
BOT_MAGIC = int(os.getenv("BOT_MAGIC", "123456"))


HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang=\"en\">
<head>
  <meta charset=\"UTF-8\" />
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1.0\" />
  <title>Sentinel Trade Dashboard</title>
  <style>
    :root {
      --bg: #0b1220;
      --panel: #111b2f;
      --card: #1a2742;
      --text: #d9e6ff;
      --muted: #8ea6d9;
      --good: #4dd48a;
      --warn: #f3be4e;
      --bad: #ff6b6b;
      --accent: #4ea1ff;
    }
    body {
      margin: 0;
      font-family: Segoe UI, Tahoma, sans-serif;
      background: radial-gradient(circle at top right, #1b3a73, var(--bg));
      color: var(--text);
    }
    .wrap { max-width: 1100px; margin: 0 auto; padding: 20px; }
    h1 { margin: 0 0 16px; font-size: 28px; }
    .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 12px; }
    .card {
      background: linear-gradient(180deg, var(--card), var(--panel));
      border: 1px solid #233a66;
      border-radius: 14px;
      padding: 14px;
      box-shadow: 0 8px 24px rgba(0, 0, 0, 0.22);
    }
    .label { color: var(--muted); font-size: 13px; margin-bottom: 8px; }
    .value { font-size: 26px; font-weight: 700; }
    .ok { color: var(--good); }
    .warn { color: var(--warn); }
    .bad { color: var(--bad); }
    table { width: 100%; border-collapse: collapse; margin-top: 12px; }
    th, td { text-align: left; padding: 8px; border-bottom: 1px solid #233a66; font-size: 13px; }
    .muted { color: var(--muted); }
  </style>
</head>
<body>
  <div class=\"wrap\">
    <h1>Sentinel Real-Time Trade Dashboard</h1>
    <div class=\"grid\">
      <div class=\"card\"><div class=\"label\">Open Positions</div><div id=\"open_positions\" class=\"value\">-</div></div>
      <div class=\"card\"><div class=\"label\">PnL Today (USD)</div><div id=\"pnl_today\" class=\"value\">-</div></div>
      <div class=\"card\"><div class=\"label\">Win Rate (Last 50)</div><div id=\"win_rate_50\" class=\"value\">-</div></div>
      <div class=\"card\"><div class=\"label\">System Health</div><div id=\"health\" class=\"value\">-</div></div>
      <div class=\"card\"><div class=\"label\">Latency (ms, avg recent)</div><div id=\"latency\" class=\"value\">-</div></div>
    </div>

    <div class=\"card\" style=\"margin-top:14px\">
      <div class=\"label\">Current Open Positions</div>
      <table>
        <thead><tr><th>Ticket</th><th>Symbol</th><th>Type</th><th>Volume</th><th>PnL</th></tr></thead>
        <tbody id=\"positions_body\"></tbody>
      </table>
      <div class=\"muted\" id=\"updated_at\" style=\"margin-top:10px\"></div>
    </div>
  </div>

  <script>
    function clsForValue(v) {
      if (v > 0) return 'ok';
      if (v < 0) return 'bad';
      return 'warn';
    }

    async function refresh() {
      try {
        const r = await fetch('/api/dashboard');
        const data = await r.json();

        document.getElementById('open_positions').textContent = data.open_positions.length;
        const pnlEl = document.getElementById('pnl_today');
        pnlEl.textContent = Number(data.pnl_today_usd).toFixed(2);
        pnlEl.className = 'value ' + clsForValue(Number(data.pnl_today_usd));

        document.getElementById('win_rate_50').textContent = Number(data.win_rate_last_50_pct).toFixed(2) + '%';

        const healthEl = document.getElementById('health');
        healthEl.textContent = data.system_health.mt5_connected ? 'MT5 OK' : 'MT5 DOWN';
        healthEl.className = 'value ' + (data.system_health.mt5_connected ? 'ok' : 'bad');

        const latency = Number(data.system_health.latency_ms_recent_avg);
        const latencyEl = document.getElementById('latency');
        latencyEl.textContent = latency.toFixed(1);
        latencyEl.className = 'value ' + (latency > 500 ? 'bad' : 'ok');

        const body = document.getElementById('positions_body');
        body.innerHTML = '';
        for (const p of data.open_positions) {
          const row = document.createElement('tr');
          row.innerHTML = `<td>${p.ticket}</td><td>${p.symbol}</td><td>${p.type}</td><td>${p.volume}</td><td>${Number(p.profit).toFixed(2)}</td>`;
          body.appendChild(row);
        }
        if (data.open_positions.length === 0) {
          const row = document.createElement('tr');
          row.innerHTML = '<td colspan="5" class="muted">No open positions</td>';
          body.appendChild(row);
        }

        document.getElementById('updated_at').textContent = 'Updated: ' + data.generated_at;
      } catch (e) {
        document.getElementById('updated_at').textContent = 'Dashboard fetch error: ' + e;
      }
    }

    refresh();
    setInterval(refresh, 2000);
  </script>
</body>
</html>
"""


def _connect_db():
    return psycopg.connect(
        host=DB_HOST,
        port=DB_PORT,
        dbname=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD,
    )


def _ensure_mt5_connected() -> bool:
    if mt5.terminal_info() is not None:
        return True
    login = os.getenv("MT5_LOGIN")
    password = os.getenv("MT5_PASSWORD") or os.getenv("MT5_PASS")
    server = os.getenv("MT5_SERVER")
    path = os.getenv("MT5_PATH")
    login_num = int(login) if login and str(login).isdigit() else None
    return bool(mt5.initialize(path=path if path else None, login=login_num, password=password, server=server))


def _open_positions() -> list[dict]:
    connected = _ensure_mt5_connected()
    if not connected:
        return []
    positions = mt5.positions_get(magic=BOT_MAGIC) or []
    out = []
    for p in positions:
        out.append(
            {
                "ticket": int(getattr(p, "ticket", 0)),
                "symbol": getattr(p, "symbol", ""),
                "type": "BUY" if getattr(p, "type", 0) == mt5.POSITION_TYPE_BUY else "SELL",
                "volume": float(getattr(p, "volume", 0.0)),
                "profit": float(getattr(p, "profit", 0.0)),
            }
        )
    return out


def _pnl_today_usd() -> float:
    today_start = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
    query = """
        SELECT COALESCE(SUM(pnl_usd), 0.0)
        FROM signal_journal
        WHERE pnl_usd IS NOT NULL
          AND COALESCE(signal_ts, journal_ts) >= %s
    """
    try:
        with _connect_db() as connection:
            with connection.cursor() as cursor:
                cursor.execute(query, (today_start,))
                row = cursor.fetchone()
        return float(row[0] or 0.0)
    except Exception:
        return 0.0


def _win_rate_last_50() -> float:
    query = """
        SELECT pnl_usd
        FROM signal_journal
        WHERE pnl_usd IS NOT NULL
        ORDER BY COALESCE(signal_ts, journal_ts) DESC
        LIMIT 50
    """
    try:
        with _connect_db() as connection:
            with connection.cursor() as cursor:
                cursor.execute(query)
                rows = cursor.fetchall()
    except Exception:
        return 0.0

    if not rows:
        return 0.0
    wins = sum(1 for (pnl,) in rows if float(pnl) > 0)
    return (wins / len(rows)) * 100.0


def _latency_recent_avg() -> float:
    query = """
        SELECT COALESCE(AVG(send_to_fill_ms), 0.0)
        FROM (
            SELECT send_to_fill_ms
            FROM execution_quality
            WHERE send_to_fill_ms IS NOT NULL
            ORDER BY logged_at DESC
            LIMIT 50
        ) q
    """
    try:
        with _connect_db() as connection:
            with connection.cursor() as cursor:
                cursor.execute(query)
                row = cursor.fetchone()
        return float(row[0] or 0.0)
    except Exception:
        return 0.0


@app.get("/", response_class=HTMLResponse)
def dashboard_html() -> str:
    return HTML_TEMPLATE


@app.get("/api/dashboard")
def dashboard_data() -> dict:
    mt5_connected = _ensure_mt5_connected()
    positions = _open_positions() if mt5_connected else []
    latency = _latency_recent_avg()

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "open_positions": positions,
        "pnl_today_usd": round(_pnl_today_usd(), 2),
        "win_rate_last_50_pct": round(_win_rate_last_50(), 2),
        "system_health": {
            "mt5_connected": bool(mt5_connected),
            "latency_ms_recent_avg": round(latency, 2),
            "latency_alert": bool(latency > 500.0),
        },
    }


@app.get("/api/health")
def health() -> dict:
    mt5_ok = _ensure_mt5_connected()
    latency = _latency_recent_avg()
    return {
        "status": "ok" if mt5_ok else "degraded",
        "mt5_connected": bool(mt5_ok),
        "latency_ms_recent_avg": round(latency, 2),
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("DASHBOARD_PORT", "9100")))
