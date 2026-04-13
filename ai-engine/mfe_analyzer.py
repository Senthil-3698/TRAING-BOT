from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any

import MetaTrader5 as mt5
import numpy as np
import pandas as pd
import psycopg
import redis
from dotenv import load_dotenv

load_dotenv()

PARTIAL_R_MIN = 1.2
PARTIAL_R_MAX = 2.0
PARTIAL_R_STEP = 0.1
PARTIAL_R_KEY = os.getenv("EXIT_PARTIAL_R_KEY", "exit:partial_r_multiple")
PARTIAL_TUNE_META_KEY = os.getenv("EXIT_PARTIAL_TUNE_META_KEY", "exit:partial_tune_meta")


@dataclass
class ClosedTrade:
    id: int
    symbol: str
    action: str
    timeframe: str
    entry_ts: datetime
    close_ts: datetime
    entry_price: float
    stop_loss: float
    exit_price: float
    pnl_usd: float
    pnl_r: float | None


@dataclass
class ExcursionMetrics:
    trade_id: int
    symbol: str
    action: str
    risk_price: float
    final_r: float
    mfe_r: float
    mae_r: float
    captured_mfe_pct: float | None


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


def _ensure_mt5() -> bool:
    if mt5.terminal_info() is not None:
        return True

    login_raw = os.getenv("MT5_LOGIN")
    password = os.getenv("MT5_PASS") or os.getenv("MT5_PASSWORD")
    server = os.getenv("MT5_SERVER")
    path = os.getenv("MT5_PATH")
    login = int(login_raw) if login_raw else None

    return bool(
        mt5.initialize(
            path=path if path else None,
            login=login,
            password=password,
            server=server,
        )
    )


def _tf_to_mt5(timeframe: str):
    mapping = {
        "1m": mt5.TIMEFRAME_M1,
        "m1": mt5.TIMEFRAME_M1,
        "5m": mt5.TIMEFRAME_M5,
        "m5": mt5.TIMEFRAME_M5,
        "15m": mt5.TIMEFRAME_M15,
        "m15": mt5.TIMEFRAME_M15,
        "1h": mt5.TIMEFRAME_H1,
        "h1": mt5.TIMEFRAME_H1,
    }
    return mapping.get(str(timeframe).lower(), mt5.TIMEFRAME_M1)


def _fetch_closed_trades(symbol: str | None = None, limit: int = 2000) -> list[ClosedTrade]:
    where_symbol = "AND symbol = %s" if symbol else ""
    params: list[Any] = [limit]
    if symbol:
        params.append(symbol)

    query = f"""
    SELECT id,
           symbol,
           action,
           COALESCE(timeframe, '1m') AS timeframe,
           COALESCE(signal_ts, journal_ts) AS entry_ts,
           journal_ts AS close_ts,
           entry_price,
           stop_loss,
           exit_price,
           pnl_usd,
           pnl_r
    FROM signal_journal
    WHERE decision_status = 'ACCEPTED'
      AND is_filled = TRUE
      AND entry_price IS NOT NULL
      AND stop_loss IS NOT NULL
      AND exit_price IS NOT NULL
      AND pnl_usd IS NOT NULL
      {where_symbol}
    ORDER BY id DESC
    LIMIT %s
    """

    trades: list[ClosedTrade] = []
    with psycopg.connect(**_db_config()) as connection:
        with connection.cursor() as cursor:
            cursor.execute(query, params)
            rows = cursor.fetchall()

    for row in rows:
        (
            row_id,
            row_symbol,
            action,
            timeframe,
            entry_ts,
            close_ts,
            entry_price,
            stop_loss,
            exit_price,
            pnl_usd,
            pnl_r,
        ) = row

        if entry_ts is None or close_ts is None:
            continue

        if not isinstance(entry_ts, datetime):
            continue
        if not isinstance(close_ts, datetime):
            continue

        trades.append(
            ClosedTrade(
                id=int(row_id),
                symbol=str(row_symbol),
                action=str(action).upper(),
                timeframe=str(timeframe),
                entry_ts=entry_ts,
                close_ts=close_ts,
                entry_price=float(entry_price),
                stop_loss=float(stop_loss),
                exit_price=float(exit_price),
                pnl_usd=float(pnl_usd),
                pnl_r=float(pnl_r) if pnl_r is not None else None,
            )
        )

    return trades


def _fetch_bar_path(symbol: str, timeframe: str, start_ts: datetime, end_ts: datetime) -> pd.DataFrame | None:
    tf = _tf_to_mt5(timeframe)

    if end_ts <= start_ts:
        end_ts = start_ts + timedelta(minutes=1)

    rates = mt5.copy_rates_range(symbol, tf, start_ts, end_ts)
    if rates is None or len(rates) == 0:
        # Retry with M1 as a fallback when the original timeframe has sparse bars.
        rates = mt5.copy_rates_range(symbol, mt5.TIMEFRAME_M1, start_ts, end_ts)
        if rates is None or len(rates) == 0:
            return None

    df = pd.DataFrame(rates)
    if "time" in df.columns:
        df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
    return df


def _compute_trade_excursions(trade: ClosedTrade) -> ExcursionMetrics | None:
    df = _fetch_bar_path(trade.symbol, trade.timeframe, trade.entry_ts, trade.close_ts)
    if df is None or df.empty:
        return None

    risk_price = abs(trade.entry_price - trade.stop_loss)
    if risk_price <= 0:
        return None

    highest = float(df["high"].max())
    lowest = float(df["low"].min())

    if trade.action == "BUY":
        favorable_price = max(0.0, highest - trade.entry_price)
        adverse_price = max(0.0, trade.entry_price - lowest)
        final_r = trade.pnl_r if trade.pnl_r is not None else (trade.exit_price - trade.entry_price) / risk_price
    else:
        favorable_price = max(0.0, trade.entry_price - lowest)
        adverse_price = max(0.0, highest - trade.entry_price)
        final_r = trade.pnl_r if trade.pnl_r is not None else (trade.entry_price - trade.exit_price) / risk_price

    mfe_r = favorable_price / risk_price
    mae_r = adverse_price / risk_price

    captured_mfe_pct = None
    if trade.pnl_usd > 0 and mfe_r > 0:
        captured_mfe_pct = max(0.0, min(1.5, final_r / mfe_r))

    return ExcursionMetrics(
        trade_id=trade.id,
        symbol=trade.symbol,
        action=trade.action,
        risk_price=risk_price,
        final_r=float(final_r),
        mfe_r=float(mfe_r),
        mae_r=float(mae_r),
        captured_mfe_pct=float(captured_mfe_pct) if captured_mfe_pct is not None else None,
    )


def _simulate_partial_expectancy(metrics: list[ExcursionMetrics], partial_r: float) -> float:
    if not metrics:
        return 0.0

    total_r = 0.0
    for m in metrics:
        # If price reached the partial threshold at any point, bank half at partial_r.
        if m.mfe_r >= partial_r:
            simulated_r = (0.5 * partial_r) + (0.5 * m.final_r)
        else:
            simulated_r = m.final_r
        total_r += simulated_r

    return total_r / len(metrics)


def _optimize_partial_r(metrics: list[ExcursionMetrics]) -> dict[str, Any]:
    levels = np.arange(PARTIAL_R_MIN, PARTIAL_R_MAX + 1e-9, PARTIAL_R_STEP)
    candidates: list[dict[str, Any]] = []
    best_level = PARTIAL_R_MIN
    best_expectancy = -1e12

    for level in levels:
        expectancy = _simulate_partial_expectancy(metrics, float(level))
        rounded_level = round(float(level), 2)
        candidates.append({"partial_r": rounded_level, "expectancy_r": round(float(expectancy), 6)})
        if expectancy > best_expectancy:
            best_expectancy = expectancy
            best_level = rounded_level

    return {
        "best_partial_r": best_level,
        "best_expectancy_r": round(float(best_expectancy), 6),
        "candidates": candidates,
    }


def _summarize(metrics: list[ExcursionMetrics]) -> dict[str, Any]:
    if not metrics:
        return {
            "sample_size": 0,
            "winner_mfe_capture_pct_mean": None,
            "winner_mfe_capture_pct_median": None,
            "loser_mae_r_median": None,
            "loss_count": 0,
            "win_count": 0,
        }

    winner_capture = [m.captured_mfe_pct for m in metrics if m.captured_mfe_pct is not None]
    losers_mae = [m.mae_r for m in metrics if m.final_r < 0]

    return {
        "sample_size": len(metrics),
        "win_count": sum(1 for m in metrics if m.final_r > 0),
        "loss_count": sum(1 for m in metrics if m.final_r < 0),
        "winner_mfe_capture_pct_mean": round(float(np.mean(winner_capture)), 4) if winner_capture else None,
        "winner_mfe_capture_pct_median": round(float(np.median(winner_capture)), 4) if winner_capture else None,
        "loser_mae_r_median": round(float(np.median(losers_mae)), 4) if losers_mae else None,
    }


def auto_tune_partial_r_if_due(min_new_trades: int = 100, symbol: str | None = None, limit: int = 2000) -> dict[str, Any]:
    trades = _fetch_closed_trades(symbol=symbol, limit=limit)
    if not trades:
        return {"status": "no_closed_trades"}

    cache = _redis_client()
    meta_raw = cache.get(PARTIAL_TUNE_META_KEY)
    last_tuned_count = 0
    if meta_raw:
        try:
            meta = json.loads(meta_raw.decode("utf-8"))
            last_tuned_count = int(meta.get("last_tuned_trade_count", 0))
        except Exception:
            last_tuned_count = 0

    total_closed = len(trades)
    if (total_closed - last_tuned_count) < min_new_trades:
        return {
            "status": "not_due",
            "total_closed": total_closed,
            "last_tuned_trade_count": last_tuned_count,
            "new_closed_since_tune": total_closed - last_tuned_count,
        }

    if not _ensure_mt5():
        return {
            "status": "mt5_unavailable",
            "reason": str(mt5.last_error()),
            "total_closed": total_closed,
        }

    metrics: list[ExcursionMetrics] = []
    for trade in trades:
        m = _compute_trade_excursions(trade)
        if m is not None:
            metrics.append(m)

    if not metrics:
        return {"status": "no_metrics_computed", "total_closed": total_closed}

    optimization = _optimize_partial_r(metrics)
    best_partial_r = float(optimization["best_partial_r"])
    best_partial_r = max(PARTIAL_R_MIN, min(PARTIAL_R_MAX, best_partial_r))

    cache.set(PARTIAL_R_KEY, str(best_partial_r))
    cache.set(
        PARTIAL_TUNE_META_KEY,
        json.dumps(
            {
                "last_tuned_trade_count": total_closed,
                "last_tuned_at": datetime.now(timezone.utc).isoformat(),
                "best_partial_r": best_partial_r,
            }
        ),
    )

    return {
        "status": "tuned",
        "total_closed": total_closed,
        "best_partial_r": best_partial_r,
        "summary": _summarize(metrics),
        "optimization": optimization,
    }


def run_mfe_analysis(symbol: str | None = None, limit: int = 2000, force_tune: bool = False) -> dict[str, Any]:
    trades = _fetch_closed_trades(symbol=symbol, limit=limit)
    if not trades:
        return {"status": "no_closed_trades"}

    if not _ensure_mt5():
        return {
            "status": "mt5_unavailable",
            "reason": str(mt5.last_error()),
            "total_closed": len(trades),
        }

    metrics: list[ExcursionMetrics] = []
    for trade in trades:
        m = _compute_trade_excursions(trade)
        if m is not None:
            metrics.append(m)

    summary = _summarize(metrics)
    optimization = _optimize_partial_r(metrics) if metrics else {}

    tune_result = None
    if force_tune:
        tune_result = auto_tune_partial_r_if_due(min_new_trades=0, symbol=symbol, limit=limit)
    else:
        tune_result = auto_tune_partial_r_if_due(min_new_trades=100, symbol=symbol, limit=limit)

    return {
        "status": "ok",
        "symbol": symbol,
        "total_closed_trades": len(trades),
        "metrics_computed": len(metrics),
        "summary": summary,
        "optimization": optimization,
        "auto_tune": tune_result,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="MFE/MAE analysis and partial-profit auto tuning")
    parser.add_argument("--symbol", type=str, default=None, help="Optional symbol filter, e.g. XAUUSD")
    parser.add_argument("--limit", type=int, default=2000, help="Max closed trades to analyze")
    parser.add_argument("--force-tune", action="store_true", help="Force tuning regardless of the 100-trade threshold")
    parser.add_argument("--out", type=str, default=None, help="Optional path to save JSON report")
    args = parser.parse_args()

    report = run_mfe_analysis(symbol=args.symbol, limit=args.limit, force_tune=args.force_tune)
    text = json.dumps(report, indent=2)
    print(text)

    if args.out:
        with open(args.out, "w", encoding="utf-8") as handle:
            handle.write(text)


if __name__ == "__main__":
    main()
