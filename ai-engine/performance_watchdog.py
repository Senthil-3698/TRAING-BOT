from __future__ import annotations

import argparse
import csv
import json
import os
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
import psycopg
import redis
from dotenv import load_dotenv

load_dotenv()


@dataclass
class BaselineProfile:
    backtest_win_rate_pct_90d: float
    backtest_max_drawdown_pct: float
    backtest_slippage_points: float
    backtest_loss_p95_usd: float
    baseline_run_dir: str
    baseline_source: str


def _baseline_account_balance_usd() -> float:
    return float(os.getenv("PERF_BASELINE_ACCOUNT_BALANCE_USD", "10000"))


def _live_account_balance_usd() -> float:
    return float(os.getenv("PERF_LIVE_ACCOUNT_BALANCE_USD", str(_baseline_account_balance_usd())))


def _is_backtest_run_dir(path: Path) -> bool:
    return path.is_dir() and (path / "metrics.json").exists() and (path / "trades.csv").exists()


def _is_walkforward_run_dir(path: Path) -> bool:
    return path.is_dir() and (path / "summary.json").exists() and (path / "window_results.csv").exists()


def _db_config() -> dict[str, Any]:
    return {
        "host": os.getenv("JOURNAL_DB_HOST", os.getenv("MONITOR_DB_HOST", "localhost")),
        "port": int(os.getenv("JOURNAL_DB_PORT", os.getenv("MONITOR_DB_PORT", "5433"))),
        "dbname": os.getenv("JOURNAL_DB_NAME", os.getenv("MONITOR_DB_NAME", "sentinel_db")),
        "user": os.getenv("JOURNAL_DB_USER", os.getenv("MONITOR_DB_USER", "admin")),
        "password": os.getenv("JOURNAL_DB_PASSWORD", os.getenv("MONITOR_DB_PASSWORD", "admin")),
    }


def _redis_client() -> redis.Redis:
    return redis.Redis(
        host=os.getenv("REDIS_HOST", "localhost"),
        port=int(os.getenv("REDIS_PORT", "6380")),
        db=0,
    )


def _find_latest_baseline_run(backtest_root: Path) -> tuple[Path, str]:
    explicit = os.getenv("BACKTEST_BASELINE_RUN_DIR")
    if explicit:
        p = Path(explicit)
        if _is_backtest_run_dir(p):
            return p, "backtest"
        if _is_walkforward_run_dir(p):
            return p, "walkforward"

    candidates = [
        p for p in backtest_root.iterdir()
        if p.is_dir() and (_is_backtest_run_dir(p) or _is_walkforward_run_dir(p))
    ]
    if not candidates:
        raise RuntimeError(
            "No baseline run found. Expected either backtest artifacts (metrics.json + trades.csv) "
            "or walk-forward artifacts (summary.json + window_results.csv)."
        )

    candidates.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    latest = candidates[0]
    if _is_backtest_run_dir(latest):
        return latest, "backtest"
    return latest, "walkforward"


def _load_baseline(backtest_root: Path) -> BaselineProfile:
    run_dir, source = _find_latest_baseline_run(backtest_root)

    if source == "backtest":
        metrics_path = run_dir / "metrics.json"
        trades_path = run_dir / "trades.csv"

        with open(metrics_path, "r", encoding="utf-8") as handle:
            metrics = json.load(handle)

        baseline_win_rate = float(metrics.get("win_rate_pct", 0.0))
        baseline_max_dd = float(metrics.get("max_drawdown_pct", 0.0))

        loss_sizes = []
        with open(trades_path, "r", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                pnl_text = row.get("pnl_usd")
                if pnl_text is None:
                    continue
                try:
                    pnl = float(pnl_text)
                except ValueError:
                    continue
                if pnl < 0:
                    loss_sizes.append(abs(pnl))
    else:
        # Walk-forward baseline compatibility: derive equivalent watchdog stats from window outputs.
        summary_path = run_dir / "summary.json"
        windows_path = run_dir / "window_results.csv"

        with open(summary_path, "r", encoding="utf-8") as handle:
            _ = json.load(handle)

        test_returns_pct: list[float] = []
        with open(windows_path, "r", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                r_txt = row.get("test_return_pct")
                if r_txt is None:
                    continue
                try:
                    test_returns_pct.append(float(r_txt))
                except ValueError:
                    continue

        if test_returns_pct:
            wins = sum(1 for r in test_returns_pct if r > 0)
            baseline_win_rate = (wins / len(test_returns_pct)) * 100.0

            equity = 100.0
            peak = 100.0
            max_dd = 0.0
            for ret_pct in test_returns_pct:
                equity *= (1.0 + (ret_pct / 100.0))
                if equity > peak:
                    peak = equity
                if peak > 0:
                    dd = ((peak - equity) / peak) * 100.0
                    max_dd = max(max_dd, dd)
            baseline_max_dd = max_dd

            neg_returns = [abs(r) for r in test_returns_pct if r < 0]
            account = _baseline_account_balance_usd()
            loss_sizes = [(pct / 100.0) * account for pct in neg_returns]
        else:
            baseline_win_rate = 0.0
            baseline_max_dd = 0.0
            loss_sizes = []

    loss_p95 = float(np.percentile(loss_sizes, 95)) if loss_sizes else float(os.getenv("PERF_FALLBACK_LOSS_P95_USD", "100.0"))
    baseline_slippage = float(os.getenv("PERF_BASELINE_SLIPPAGE_POINTS", "5.0"))

    return BaselineProfile(
        backtest_win_rate_pct_90d=baseline_win_rate,
        backtest_max_drawdown_pct=baseline_max_dd,
        backtest_slippage_points=baseline_slippage,
        backtest_loss_p95_usd=loss_p95,
        baseline_run_dir=str(run_dir),
        baseline_source=source,
    )


def _query_live_closed_trades(limit: int = 500, lookback_days: int = 90) -> list[tuple[datetime, float]]:
    cutoff = datetime.now(timezone.utc) - timedelta(days=lookback_days)
    rows: list[tuple[datetime, float]] = []
    with psycopg.connect(**_db_config()) as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT COALESCE(signal_ts, journal_ts) AS ts, pnl_usd
                FROM signal_journal
                WHERE decision_status = 'ACCEPTED'
                  AND is_filled = TRUE
                  AND pnl_usd IS NOT NULL
                  AND COALESCE(signal_ts, journal_ts) >= %s
                ORDER BY ts DESC
                LIMIT %s
                """,
                (cutoff, limit),
            )
            fetched = cursor.fetchall()

    for ts, pnl in fetched:
        if ts is None or pnl is None:
            continue
        rows.append((ts, float(pnl)))
    return rows


def _query_live_avg_slippage(limit: int = 1000, lookback_days: int = 30) -> float | None:
    cutoff = datetime.now(timezone.utc) - timedelta(days=lookback_days)
    with psycopg.connect(**_db_config()) as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT slippage_points
                FROM execution_quality
                WHERE logged_at >= %s
                  AND slippage_points IS NOT NULL
                ORDER BY logged_at DESC
                LIMIT %s
                """,
                (cutoff, limit),
            )
            rows = cursor.fetchall()

    if not rows:
        return None

    values = [abs(float(r[0])) for r in rows if r[0] is not None]
    if not values:
        return None
    return float(np.mean(values))


def _rolling_30_trade_win_rate_drop(live_pnls_desc: list[float], baseline_win_rate_pct: float) -> tuple[bool, dict[str, Any]]:
    if len(live_pnls_desc) < 30:
        return False, {"reason": "insufficient_live_trades", "required": 30, "available": len(live_pnls_desc)}

    window = live_pnls_desc[:30]
    wins = sum(1 for pnl in window if pnl > 0)
    live_win_rate_pct = (wins / 30.0) * 100.0
    drop = baseline_win_rate_pct - live_win_rate_pct
    breached = drop > 10.0

    return breached, {
        "live_win_rate_pct_30": round(live_win_rate_pct, 4),
        "baseline_win_rate_pct_90d": round(baseline_win_rate_pct, 4),
        "drop_percentage_points": round(drop, 4),
        "threshold_pp": 10.0,
    }


def _slippage_doubled(live_avg_slippage: float | None, baseline_slippage: float) -> tuple[bool, dict[str, Any]]:
    if live_avg_slippage is None:
        return False, {"reason": "insufficient_execution_quality_rows"}
    threshold = baseline_slippage * 2.0
    breached = live_avg_slippage > threshold
    return breached, {
        "live_avg_slippage_points": round(live_avg_slippage, 4),
        "baseline_slippage_points": round(baseline_slippage, 4),
        "threshold_points": round(threshold, 4),
    }


def _drawdown_breach(live_pnls_desc: list[float], baseline_max_dd_pct: float) -> tuple[bool, dict[str, Any]]:
    if not live_pnls_desc:
        return False, {"reason": "no_live_closed_trades"}

    ordered = list(reversed(live_pnls_desc))
    account_balance = _live_account_balance_usd()
    if account_balance <= 0:
        return False, {"reason": "invalid_live_account_balance", "balance": account_balance}

    equity = 100.0
    curve = []
    for pnl in ordered:
        equity += (pnl / account_balance) * 100.0
        curve.append(equity)

    peak = -1e18
    max_dd_pct = 0.0
    for val in curve:
        if val > peak:
            peak = val
        if peak > 0:
            dd_pct = ((peak - val) / peak) * 100.0
            if dd_pct > max_dd_pct:
                max_dd_pct = dd_pct

    threshold = baseline_max_dd_pct * 1.5
    breached = max_dd_pct > threshold
    return breached, {
        "live_max_drawdown_pct": round(max_dd_pct, 4),
        "baseline_max_drawdown_pct": round(baseline_max_dd_pct, 4),
        "threshold_pct": round(threshold, 4),
        "live_account_balance_usd": round(account_balance, 2),
    }


def _effective_poll_interval_seconds(base_interval_seconds: int) -> int:
    """Use faster polling during active UTC trading hours to react quicker in fast drawdowns."""
    fast_interval = int(os.getenv("WATCHDOG_FAST_INTERVAL_SECONDS", "15"))
    active_start = int(os.getenv("WATCHDOG_ACTIVE_START_HOUR_UTC", "6"))
    active_end = int(os.getenv("WATCHDOG_ACTIVE_END_HOUR_UTC", "21"))

    now_hour = datetime.now(timezone.utc).hour
    in_active_window = active_start <= now_hour < active_end
    if in_active_window:
        return max(1, min(base_interval_seconds, fast_interval))
    return max(1, base_interval_seconds)


def _consecutive_extreme_losses_breach(live_pnls_desc: list[float], loss_p95: float) -> tuple[bool, dict[str, Any]]:
    if not live_pnls_desc:
        return False, {"reason": "no_live_closed_trades"}

    streak = 0
    max_streak = 0
    for pnl in live_pnls_desc:
        if pnl < 0 and abs(pnl) >= loss_p95:
            streak += 1
            max_streak = max(max_streak, streak)
        else:
            streak = 0

    breached = max_streak > 3
    return breached, {
        "max_consecutive_extreme_losses": int(max_streak),
        "loss_size_threshold_p95_usd": round(loss_p95, 4),
        "threshold_consecutive": 3,
    }


def _disable_trading(alerts: list[dict[str, Any]], baseline: BaselineProfile) -> None:
    redis_client = _redis_client()
    kill_switch_key = os.getenv("GLOBAL_KILL_SWITCH_KEY", "GLOBAL_KILL_SWITCH")

    payload = {
        "disabled_at": datetime.now(timezone.utc).isoformat(),
        "source": "performance_watchdog",
        "baseline_run_dir": baseline.baseline_run_dir,
        "alerts": alerts,
    }

    redis_client.set(kill_switch_key, "ACTIVE")
    redis_client.set("performance_watchdog:last_disable_reason", json.dumps(payload, default=str))
    redis_client.lpush("performance_watchdog:alerts", json.dumps(payload, default=str))
    redis_client.ltrim("performance_watchdog:alerts", 0, 499)

    print("[WATCHDOG] AUTO-DISABLE TRIGGERED. GLOBAL_KILL_SWITCH set to ACTIVE.")
    print(json.dumps(payload, indent=2, default=str))


def evaluate_once(backtest_root: Path) -> dict[str, Any]:
    baseline = _load_baseline(backtest_root)
    live_rows = _query_live_closed_trades(limit=1000, lookback_days=90)
    live_pnls_desc = [pnl for _, pnl in live_rows]
    live_avg_slippage = _query_live_avg_slippage(limit=1000, lookback_days=30)

    checks: list[dict[str, Any]] = []

    a_breach, a_details = _rolling_30_trade_win_rate_drop(live_pnls_desc, baseline.backtest_win_rate_pct_90d)
    checks.append({"rule": "A_WIN_RATE_DROP_30_TRADES", "breach": a_breach, "details": a_details})

    b_breach, b_details = _slippage_doubled(live_avg_slippage, baseline.backtest_slippage_points)
    checks.append({"rule": "B_SLIPPAGE_DOUBLED", "breach": b_breach, "details": b_details})

    c_breach, c_details = _drawdown_breach(live_pnls_desc, baseline.backtest_max_drawdown_pct)
    checks.append({"rule": "C_DRAWDOWN_1P5X_BACKTEST", "breach": c_breach, "details": c_details})

    d_breach, d_details = _consecutive_extreme_losses_breach(live_pnls_desc, baseline.backtest_loss_p95_usd)
    checks.append({"rule": "D_CONSEC_EXTREME_LOSSES_GT3", "breach": d_breach, "details": d_details})

    breached = [c for c in checks if c["breach"]]
    result = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "baseline": {
            "run_dir": baseline.baseline_run_dir,
            "source": baseline.baseline_source,
            "win_rate_pct_90d": baseline.backtest_win_rate_pct_90d,
            "max_drawdown_pct": baseline.backtest_max_drawdown_pct,
            "baseline_slippage_points": baseline.backtest_slippage_points,
            "loss_p95_usd": baseline.backtest_loss_p95_usd,
        },
        "live_sample": {
            "closed_trades_90d": len(live_pnls_desc),
            "avg_slippage_points_30d": live_avg_slippage,
        },
        "checks": checks,
        "breach_count": len(breached),
    }

    if breached:
        _disable_trading(breached, baseline)

    return result


def run_loop(backtest_root: Path, interval_seconds: int) -> None:
    print("[WATCHDOG] Started.")
    print(f"[WATCHDOG] base_interval_seconds={interval_seconds}")
    while True:
        try:
            report = evaluate_once(backtest_root)
            print(json.dumps(report, default=str))
        except Exception as error:
            print(f"[WATCHDOG] Evaluation error: {error}")
        sleep_seconds = _effective_poll_interval_seconds(interval_seconds)
        time.sleep(sleep_seconds)


def main() -> None:
    parser = argparse.ArgumentParser(description="Live performance watchdog with auto-disable")
    parser.add_argument("--backtest-root", default="../backtest_results", help="Path containing baseline backtest run folders")
    parser.add_argument("--interval-seconds", type=int, default=int(os.getenv("WATCHDOG_INTERVAL_SECONDS", "60")))
    parser.add_argument("--once", action="store_true", help="Run one evaluation and exit")
    args = parser.parse_args()

    root = Path(args.backtest_root)
    if not root.exists():
        raise RuntimeError(f"backtest root not found: {root}")

    if args.once:
        result = evaluate_once(root)
        print(json.dumps(result, indent=2, default=str))
    else:
        run_loop(root, args.interval_seconds)


if __name__ == "__main__":
    main()
