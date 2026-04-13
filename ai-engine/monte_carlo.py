import argparse
import csv
import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
import numpy as np

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

try:
    import psycopg
    HAS_PSYCOPG = True
except ImportError:
    HAS_PSYCOPG = False

N_SIMULATIONS = int(os.getenv("MC_SIMULATIONS", "1000"))
INITIAL_BALANCE = float(os.getenv("MC_INITIAL_BALANCE", "10000"))
RISK_PER_TRADE = float(os.getenv("MC_RISK_PER_TRADE", "0.01"))


def _load_from_db():
    if not HAS_PSYCOPG:
        raise RuntimeError("psycopg not installed")
    config = {
        "host": os.getenv("JOURNAL_DB_HOST", "localhost"),
        "port": int(os.getenv("JOURNAL_DB_PORT", "5433")),
        "dbname": os.getenv("JOURNAL_DB_NAME", "sentinel_db"),
        "user": os.getenv("JOURNAL_DB_USER", "admin"),
        "password": os.getenv("JOURNAL_DB_PASSWORD", "admin"),
    }
    cutoff = datetime.now(timezone.utc) - timedelta(days=180)
    with psycopg.connect(**config) as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT pnl_r, pnl_usd FROM signal_journal "
                "WHERE decision_status = %s AND is_filled = TRUE "
                "AND (pnl_r IS NOT NULL OR pnl_usd IS NOT NULL) "
                "AND COALESCE(signal_ts, journal_ts) >= %s "
                "ORDER BY COALESCE(signal_ts, journal_ts) ASC",
                ("ACCEPTED", cutoff),
            )
            rows = cur.fetchall()
    returns = []
    for pnl_r, pnl_usd in rows:
        if pnl_r is not None:
            returns.append(float(pnl_r))
        elif pnl_usd is not None:
            returns.append(float(pnl_usd))
    if not returns:
        raise RuntimeError("No closed trades found in last 180 days")
    return returns


def _load_from_walkforward_csv(csv_path):
    returns = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            r = row.get("test_return_pct")
            if r is None:
                continue
            try:
                returns.append(float(r) / 100.0)
            except ValueError:
                continue
    if not returns:
        raise RuntimeError("No test_return_pct values found")
    return returns


def _load_from_trades_csv(csv_path):
    returns = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            val = row.get("pnl_r") or row.get("pnl_usd")
            if val is None:
                continue
            try:
                returns.append(float(val))
            except ValueError:
                continue
    if not returns:
        raise RuntimeError("No pnl_r or pnl_usd values found")
    return returns


def run_monte_carlo(returns, n_sims=N_SIMULATIONS, initial_balance=INITIAL_BALANCE, risk_per_trade=RISK_PER_TRADE):
    rng_returns = np.array(returns, dtype=float)
    rng = np.random.default_rng(seed=42)
    final_returns = []
    max_drawdowns = []

    for _ in range(n_sims):
        shuffled = rng_returns.copy()
        rng.shuffle(shuffled)
        equity = initial_balance
        peak = initial_balance
        max_dd = 0.0
        for r in shuffled:
            pnl = equity * risk_per_trade * r
            equity = max(0.0, equity + pnl)
            if equity > peak:
                peak = equity
            if peak > 0:
                dd = (peak - equity) / peak * 100.0
                if dd > max_dd:
                    max_dd = dd
        final_returns.append((equity - initial_balance) / initial_balance * 100.0)
        max_drawdowns.append(max_dd)

    fr = np.array(final_returns)
    md = np.array(max_drawdowns)
    profitable_pct = float(np.mean(fr > 0) * 100.0)
    var_95 = float(np.percentile(fr, 5))
    cvar_vals = fr[fr <= var_95]
    cvar_95 = float(np.mean(cvar_vals)) if len(cvar_vals) else var_95

    if profitable_pct >= 70 and var_95 > -20 and float(np.median(md)) < 25:
        verdict = "EDGE_CONFIRMED"
        detail = str(round(profitable_pct, 1)) + "% profitable. VaR95=" + str(round(var_95, 2)) + "%. Median DD=" + str(round(float(np.median(md)), 2)) + "%"
    elif profitable_pct >= 55:
        verdict = "EDGE_MARGINAL"
        detail = "Only " + str(round(profitable_pct, 1)) + "% profitable. Improve signal quality before scaling."
    else:
        verdict = "NO_EDGE"
        detail = "Only " + str(round(profitable_pct, 1)) + "% profitable. Do NOT go live."

    return {
        "verdict": verdict,
        "verdict_detail": detail,
        "inputs": {
            "n_trades": len(returns),
            "n_simulations": n_sims,
            "initial_balance": initial_balance,
            "risk_per_trade_pct": risk_per_trade * 100,
        },
        "simulation_final_returns_pct": {
            "mean":   round(float(np.mean(fr)), 4),
            "median": round(float(np.median(fr)), 4),
            "std":    round(float(np.std(fr)), 4),
            "p5":     round(float(np.percentile(fr, 5)), 4),
            "p95":    round(float(np.percentile(fr, 95)), 4),
        },
        "risk_metrics": {
            "profitable_simulations_pct": round(profitable_pct, 2),
            "var_95_pct":                 round(var_95, 4),
            "cvar_95_pct":                round(cvar_95, 4),
            "median_max_drawdown_pct":    round(float(np.median(md)), 4),
            "p95_max_drawdown_pct":       round(float(np.percentile(md, 95)), 4),
            "worst_max_drawdown_pct":     round(float(np.max(md)), 4),
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, choices=["walkforward", "trades", "db"])
    parser.add_argument("--input", default=None)
    parser.add_argument("--output", default="./monte_carlo_results")
    parser.add_argument("--sims", type=int, default=N_SIMULATIONS)
    parser.add_argument("--balance", type=float, default=INITIAL_BALANCE)
    parser.add_argument("--risk", type=float, default=RISK_PER_TRADE)
    args = parser.parse_args()

    print("[MC] Loading returns from source=" + args.source + " ...")
    if args.source == "walkforward":
        returns = _load_from_walkforward_csv(args.input)
    elif args.source == "trades":
        returns = _load_from_trades_csv(args.input)
    else:
        returns = _load_from_db()

    print("[MC] Loaded " + str(len(returns)) + " observations. Running " + str(args.sims) + " simulations ...")
    summary = run_monte_carlo(returns, n_sims=args.sims, initial_balance=args.balance, risk_per_trade=args.risk)

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("=" * 60)
    print("  VERDICT: " + summary["verdict"])
    print("  " + summary["verdict_detail"])
    print("=" * 60)
    print("  Profitable sims : " + str(summary["risk_metrics"]["profitable_simulations_pct"]) + "%")
    print("  Median return   : " + str(summary["simulation_final_returns_pct"]["median"]) + "%")
    print("  VaR 95          : " + str(summary["risk_metrics"]["var_95_pct"]) + "%")
    print("  Median max DD   : " + str(summary["risk_metrics"]["median_max_drawdown_pct"]) + "%")
    print("  Results saved   : " + str(out.resolve()))


if __name__ == "__main__":
    main()