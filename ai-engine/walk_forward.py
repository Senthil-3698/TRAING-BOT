from __future__ import annotations

import argparse
import json
import itertools
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    import MetaTrader5 as mt5  # type: ignore
except ImportError:
    mt5 = None  # type: ignore


@dataclass
class ParamSet:
    confidence_min: float
    regime_adx_min: float
    cooldown_bars: int
    partial_close_r: float
    atr_sl_mult: float
    kelly_cap: float


MIN_TRADES_PER_WINDOW = 30


@dataclass
class TradeState:
    direction: str
    entry: float
    sl: float
    risk: float
    opened_idx: int
    partial_done: bool = False
    remaining: float = 1.0
    realized_r: float = 0.0


def _ensure_mt5() -> bool:
    if mt5 is None:
        return False
    if mt5.terminal_info() is not None:
        return True
    return bool(mt5.initialize())


def _fetch_m1(symbol: str, from_dt: datetime, to_dt: datetime) -> pd.DataFrame:
    if not _ensure_mt5():
        raise RuntimeError("MT5 unavailable or initialize failed.")

    if not mt5.symbol_select(symbol, True):
        raise RuntimeError(f"Could not select symbol {symbol} in MT5")

    rates = mt5.copy_rates_range(symbol, mt5.TIMEFRAME_M1, from_dt, to_dt)
    if rates is not None and len(rates) > 0:
        df = pd.DataFrame(rates)
        df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
        return df.sort_values("time").reset_index(drop=True)

    # Fallback for brokers/environments where range query is sparse or unsupported.
    total_minutes = int((to_dt - from_dt).total_seconds() / 60)
    bars_needed = max(2_000, total_minutes + 2_000)
    chunks = []
    start_pos = 0
    remaining = bars_needed
    chunk_size = 50_000
    while remaining > 0:
        take = min(chunk_size, remaining)
        chunk = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M1, start_pos, take)
        if chunk is None or len(chunk) == 0:
            break
        chunks.append(pd.DataFrame(chunk))
        start_pos += take
        remaining -= take

    if not chunks:
        raise RuntimeError(f"No M1 data for {symbol} in range {from_dt} -> {to_dt}")

    df = pd.concat(chunks, ignore_index=True)
    df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
    df = df.sort_values("time").drop_duplicates(subset=["time"]).reset_index(drop=True)
    clipped = df[(df["time"] >= pd.Timestamp(from_dt)) & (df["time"] <= pd.Timestamp(to_dt))]
    if clipped.empty:
        raise RuntimeError(
            f"No M1 data for {symbol} after fallback clipping in range {from_dt} -> {to_dt}"
        )
    return clipped.reset_index(drop=True)


def _compute_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out = out.sort_values("time").reset_index(drop=True)
    out["ema20"] = out["close"].ewm(span=20, adjust=False).mean()
    out["ema50"] = out["close"].ewm(span=50, adjust=False).mean()

    delta = out["close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(14, min_periods=14).mean()
    avg_loss = loss.rolling(14, min_periods=14).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    out["rsi14"] = 100 - (100 / (1 + rs))

    hl = out["high"] - out["low"]
    hc = (out["high"] - out["close"].shift(1)).abs()
    lc = (out["low"] - out["close"].shift(1)).abs()
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    out["atr14"] = tr.rolling(14, min_periods=14).mean()
    out["ema_distance"] = (out["close"] - out["ema20"]).abs() / out["close"].replace(0, np.nan)

    # ADX proxy for regime detection (trend vs ranging).
    plus_dm = out["high"].diff()
    minus_dm = -out["low"].diff()
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)
    atr = tr.rolling(14, min_periods=14).mean()
    plus_di = 100 * (plus_dm.rolling(14, min_periods=14).mean() / atr.replace(0, np.nan))
    minus_di = 100 * (minus_dm.rolling(14, min_periods=14).mean() / atr.replace(0, np.nan))
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    out["adx14"] = dx.rolling(14, min_periods=14).mean()

    # MTF confluence proxies: M5/H1/H4 EMA20 vs EMA50 directional bias.
    temp = out.set_index("time")
    for tf_name, rule in [("m5", "5min"), ("h1", "1h"), ("h4", "4h")]:
        close_tf = temp["close"].resample(rule).last().ffill()
        ema20_tf = close_tf.ewm(span=20, adjust=False).mean()
        ema50_tf = close_tf.ewm(span=50, adjust=False).mean()
        bias_tf = np.where(ema20_tf > ema50_tf, 1, -1)
        out[f"{tf_name}_bias"] = pd.Series(bias_tf, index=ema20_tf.index).reindex(out["time"], method="ffill").values

    # Optional live-stack fields from dataset. If absent, apply conservative defaults.
    if "news_block" not in out.columns:
        out["news_block"] = 0
    if "ai_confidence" not in out.columns:
        out["ai_confidence"] = 75.0

    return out.dropna().reset_index(drop=True)


def _build_param_grid() -> list[ParamSet]:
    grid = []
    for confidence_min, regime_adx_min, cooldown, partial_r, atr_sl_mult, kelly_cap in itertools.product(
        [60.0, 65.0, 70.0],
        [18.0, 22.0, 26.0],
        [3, 5, 8],
        [1.2, 1.4, 1.6, 1.8, 2.0],
        [1.2, 1.5, 1.8],
        [0.0125, 0.015, 0.02],
    ):
        grid.append(
            ParamSet(
                confidence_min=confidence_min,
                regime_adx_min=regime_adx_min,
                cooldown_bars=cooldown,
                partial_close_r=partial_r,
                atr_sl_mult=atr_sl_mult,
                kelly_cap=kelly_cap,
            )
        )
    return grid


def _simulate_window(df: pd.DataFrame, params: ParamSet, initial_balance: float = 10_000.0, base_risk_per_trade: float = 0.01) -> dict[str, Any]:
    if df.empty:
        return {"sharpe": 0.0, "trades": 0, "return_pct": 0.0, "final_balance": initial_balance}

    balance = initial_balance
    open_trade: TradeState | None = None
    last_entry_idx = -10_000
    trade_returns: list[float] = []
    trade_r_values: list[float] = []

    def _kelly_fraction() -> float:
        # Fractional Kelly on realized R outcomes. Falls back to base risk when history is thin.
        if len(trade_r_values) < 30:
            return base_risk_per_trade
        wins = [x for x in trade_r_values if x > 0]
        losses = [x for x in trade_r_values if x < 0]
        if not wins or not losses:
            return base_risk_per_trade
        w = len(wins) / len(trade_r_values)
        avg_win = float(np.mean(wins))
        avg_loss = abs(float(np.mean(losses)))
        if avg_loss <= 0:
            return base_risk_per_trade
        r_ratio = avg_win / avg_loss
        f_star = w - ((1 - w) / max(r_ratio, 1e-6))
        frac = max(0.0025, min(0.25 * max(f_star, 0.0), params.kelly_cap))
        return float(frac)

    for i in range(1, len(df)):
        row = df.iloc[i]
        risk_fraction = _kelly_fraction()

        if open_trade is not None:
            # Conservative ordering: stop checks before favorable triggers.
            if open_trade.direction == "BUY":
                if row["low"] <= open_trade.sl:
                    r_out = (open_trade.sl - open_trade.entry) / open_trade.risk
                    open_trade.realized_r += open_trade.remaining * r_out
                    pnl = balance * risk_fraction * open_trade.realized_r
                    trade_returns.append(pnl / max(balance, 1.0))
                    trade_r_values.append(open_trade.realized_r)
                    balance += pnl
                    open_trade = None
                    continue

                if (not open_trade.partial_done) and (row["high"] >= open_trade.entry + params.partial_close_r * open_trade.risk):
                    open_trade.realized_r += 0.5 * params.partial_close_r
                    open_trade.remaining = 0.5
                    open_trade.partial_done = True
                    open_trade.sl = max(open_trade.sl, open_trade.entry)

                if row["high"] >= open_trade.entry + 2.5 * open_trade.risk:
                    open_trade.realized_r += open_trade.remaining * 2.5
                    pnl = balance * risk_fraction * open_trade.realized_r
                    trade_returns.append(pnl / max(balance, 1.0))
                    trade_r_values.append(open_trade.realized_r)
                    balance += pnl
                    open_trade = None
                    continue

            else:
                if row["high"] >= open_trade.sl:
                    r_out = (open_trade.entry - open_trade.sl) / open_trade.risk
                    open_trade.realized_r += open_trade.remaining * r_out
                    pnl = balance * risk_fraction * open_trade.realized_r
                    trade_returns.append(pnl / max(balance, 1.0))
                    trade_r_values.append(open_trade.realized_r)
                    balance += pnl
                    open_trade = None
                    continue

                if (not open_trade.partial_done) and (row["low"] <= open_trade.entry - params.partial_close_r * open_trade.risk):
                    open_trade.realized_r += 0.5 * params.partial_close_r
                    open_trade.remaining = 0.5
                    open_trade.partial_done = True
                    open_trade.sl = min(open_trade.sl, open_trade.entry)

                if row["low"] <= open_trade.entry - 2.5 * open_trade.risk:
                    open_trade.realized_r += open_trade.remaining * 2.5
                    pnl = balance * risk_fraction * open_trade.realized_r
                    trade_returns.append(pnl / max(balance, 1.0))
                    trade_r_values.append(open_trade.realized_r)
                    balance += pnl
                    open_trade = None
                    continue

        if open_trade is not None:
            continue

        if (i - last_entry_idx) < params.cooldown_bars:
            continue

        # Live-stack aligned gates: MTF confluence + regime + news + confidence.
        mtf_buy = (row["m5_bias"] > 0) and (row["h1_bias"] > 0) and (row["h4_bias"] > 0)
        mtf_sell = (row["m5_bias"] < 0) and (row["h1_bias"] < 0) and (row["h4_bias"] < 0)
        regime_ok = float(row["adx14"]) >= params.regime_adx_min
        news_ok = int(row.get("news_block", 0)) == 0
        conf_ok = float(row.get("ai_confidence", 0.0)) >= params.confidence_min

        buy_signal = mtf_buy and regime_ok and news_ok and conf_ok
        sell_signal = mtf_sell and regime_ok and news_ok and conf_ok

        if not (buy_signal or sell_signal):
            continue

        risk = max(float(row["atr14"] * params.atr_sl_mult), float(row["close"] * 0.0002))
        if buy_signal:
            open_trade = TradeState(
                direction="BUY",
                entry=float(row["close"]),
                sl=float(row["close"] - risk),
                risk=risk,
                opened_idx=i,
            )
        else:
            open_trade = TradeState(
                direction="SELL",
                entry=float(row["close"]),
                sl=float(row["close"] + risk),
                risk=risk,
                opened_idx=i,
            )
        last_entry_idx = i

    if open_trade is not None:
        last_price = float(df.iloc[-1]["close"])
        if open_trade.direction == "BUY":
            r_out = (last_price - open_trade.entry) / open_trade.risk
        else:
            r_out = (open_trade.entry - last_price) / open_trade.risk
        open_trade.realized_r += open_trade.remaining * r_out
        pnl = balance * _kelly_fraction() * open_trade.realized_r
        trade_returns.append(pnl / max(balance, 1.0))
        trade_r_values.append(open_trade.realized_r)
        balance += pnl

    if len(trade_returns) < 2:
        sharpe = 0.0
    else:
        arr = np.array(trade_returns, dtype=float)
        std = float(np.std(arr, ddof=1))
        if std > 0:
            duration_days = max(1.0, (df["time"].iloc[-1] - df["time"].iloc[0]).total_seconds() / 86400.0)
            trades_per_year = max(1.0, (len(trade_returns) / duration_days) * 365.0)
            sharpe = float((np.mean(arr) / std) * np.sqrt(trades_per_year))
        else:
            sharpe = 0.0

    return {
        "sharpe": round(sharpe, 6),
        "trades": len(trade_returns),
        "return_pct": round(((balance - initial_balance) / initial_balance) * 100.0, 4),
        "final_balance": round(balance, 2),
    }


def _rolling_windows(from_dt: datetime, to_dt: datetime) -> list[tuple[datetime, datetime, datetime, datetime]]:
    windows = []
    cursor = pd.Timestamp(from_dt)
    end_ts = pd.Timestamp(to_dt)

    while True:
        train_start = cursor
        train_end = train_start + pd.DateOffset(months=3)
        test_end = train_end + pd.DateOffset(months=1)
        if test_end > end_ts:
            break
        windows.append(
            (
                train_start.to_pydatetime(),
                train_end.to_pydatetime(),
                train_end.to_pydatetime(),
                test_end.to_pydatetime(),
            )
        )
        cursor = cursor + pd.DateOffset(months=1)

    return windows


def _stability_report(window_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for param in ["confidence_min", "regime_adx_min", "cooldown_bars", "partial_close_r", "atr_sl_mult", "kelly_cap"]:
        values = window_df[param].astype(float)
        if values.empty:
            continue
        vmin, vmax = float(values.min()), float(values.max())
        vrange = vmax - vmin
        n_unique = int(values.nunique())
        unique_ratio = n_unique / max(len(values), 1)
        normalized_std = float(values.std(ddof=0) / vrange) if vrange > 0 else 0.0
        instability = (0.6 * unique_ratio) + (0.4 * normalized_std)
        unstable = instability > 0.55 and len(values) >= 4
        rows.append(
            {
                "parameter": param,
                "unique_values": n_unique,
                "unique_ratio": round(unique_ratio, 4),
                "normalized_std": round(normalized_std, 4),
                "instability_score": round(instability, 4),
                "status": "UNSTABLE_REMOVE" if unstable else "STABLE_KEEP",
            }
        )
    return pd.DataFrame(rows)


def _render_stability_heatmap(window_df: pd.DataFrame, out_path: Path) -> None:
    if window_df.empty:
        return

    params = ["confidence_min", "regime_adx_min", "cooldown_bars", "partial_close_r", "atr_sl_mult", "kelly_cap"]
    matrix = []
    annotations = []

    for p in params:
        vals = window_df[p].tolist()
        unique = sorted(set(vals))
        encode = {v: i for i, v in enumerate(unique)}
        matrix.append([encode[v] for v in vals])
        annotations.append(vals)

    arr = np.array(matrix, dtype=float)
    fig, ax = plt.subplots(figsize=(max(10, len(window_df) * 1.1), 5.2))
    im = ax.imshow(arr, aspect="auto", cmap="viridis")

    ax.set_yticks(range(len(params)))
    ax.set_yticklabels(params)
    ax.set_xticks(range(len(window_df)))
    ax.set_xticklabels(window_df["window_id"].tolist(), rotation=45, ha="right")
    ax.set_title("Walk-Forward Parameter Stability Heatmap (selected best params by window)")

    for y in range(arr.shape[0]):
        for x in range(arr.shape[1]):
            ax.text(x, y, str(annotations[y][x]), va="center", ha="center", fontsize=7, color="white")

    fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02, label="Ordinal value index")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def _load_m1_csv(csv_path: str, from_dt: datetime, to_dt: datetime) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if "time" not in df.columns:
        raise RuntimeError("CSV must contain a 'time' column.")
    required = {"open", "high", "low", "close"}
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise RuntimeError(f"CSV missing required columns: {missing}")

    df["time"] = pd.to_datetime(df["time"], utc=True)
    df = df.sort_values("time").reset_index(drop=True)
    clipped = df[(df["time"] >= pd.Timestamp(from_dt)) & (df["time"] <= pd.Timestamp(to_dt))]
    if clipped.empty:
        raise RuntimeError("CSV has no rows inside requested date window.")
    return clipped.reset_index(drop=True)


def run_walk_forward(symbol: str, from_dt: datetime, to_dt: datetime, output_dir: str, data_csv: str | None = None) -> dict[str, Any]:
    windows = _rolling_windows(from_dt, to_dt)
    if not windows:
        raise RuntimeError("No valid 3M-train/1M-test windows in the requested date range.")

    # Pull one superset and slice per window for efficiency.
    superset_start = windows[0][0]
    superset_end = windows[-1][3]
    raw = _load_m1_csv(data_csv, superset_start, superset_end) if data_csv else _fetch_m1(symbol, superset_start, superset_end)
    features = _compute_features(raw)

    grid = _build_param_grid()
    window_rows: list[dict[str, Any]] = []

    for idx, (tr_start, tr_end, te_start, te_end) in enumerate(windows, start=1):
        train_df = features[(features["time"] >= pd.Timestamp(tr_start)) & (features["time"] < pd.Timestamp(tr_end))]
        test_df = features[(features["time"] >= pd.Timestamp(te_start)) & (features["time"] < pd.Timestamp(te_end))]

        if train_df.empty or test_df.empty:
            continue

        best_params: ParamSet | None = None
        best_train_sharpe = -1e12

        for p in grid:
            in_sample = _simulate_window(train_df, p)
            if in_sample["trades"] < MIN_TRADES_PER_WINDOW:
                continue
            score = float(in_sample["sharpe"])
            if score > best_train_sharpe:
                best_train_sharpe = score
                best_params = p

        if best_params is None:
            continue

        out_sample = _simulate_window(test_df, best_params)
        window_rows.append(
            {
                "window_id": f"W{idx}",
                "train_start": tr_start.date().isoformat(),
                "train_end": tr_end.date().isoformat(),
                "test_start": te_start.date().isoformat(),
                "test_end": te_end.date().isoformat(),
                "train_sharpe": round(best_train_sharpe, 6),
                "test_sharpe": out_sample["sharpe"],
                "test_trades": out_sample["trades"],
                "test_return_pct": out_sample["return_pct"],
                "confidence_min": best_params.confidence_min,
                "regime_adx_min": best_params.regime_adx_min,
                "cooldown_bars": best_params.cooldown_bars,
                "partial_close_r": best_params.partial_close_r,
                "atr_sl_mult": best_params.atr_sl_mult,
                "kelly_cap": best_params.kelly_cap,
            }
        )

    if not window_rows:
        raise RuntimeError("Walk-forward produced no windows with sufficient train/test trades.")

    out_root = Path(output_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    run_dir = out_root / f"wf_{from_dt.date().isoformat()}__{to_dt.date().isoformat()}"
    run_dir.mkdir(parents=True, exist_ok=True)

    window_df = pd.DataFrame(window_rows)
    window_csv = run_dir / "window_results.csv"
    window_df.to_csv(window_csv, index=False)

    stability_df = _stability_report(window_df)
    stability_csv = run_dir / "stability_report.csv"
    stability_df.to_csv(stability_csv, index=False)

    heatmap_path = run_dir / "parameter_stability_heatmap.png"
    _render_stability_heatmap(window_df, heatmap_path)

    summary = {
        "symbol": symbol,
        "from": from_dt.isoformat(),
        "to": to_dt.isoformat(),
        "strategy_validation_mode": "live_stack_proxy",
        "gates_tested": [
            "mtf_confluence_m5_h1_h4",
            "regime_filter_adx",
            "news_gate",
            "confidence_gate",
            "kelly_position_sizing",
        ],
        "min_trades_per_train_window": MIN_TRADES_PER_WINDOW,
        "windows_total": int(len(window_df)),
        "avg_test_sharpe": round(float(window_df["test_sharpe"].mean()), 4),
        "median_test_sharpe": round(float(window_df["test_sharpe"].median()), 4),
        "unstable_parameters": stability_df[stability_df["status"] == "UNSTABLE_REMOVE"]["parameter"].tolist(),
        "stable_parameters": stability_df[stability_df["status"] == "STABLE_KEEP"]["parameter"].tolist(),
        "artifacts": {
            "window_results_csv": str(window_csv),
            "stability_report_csv": str(stability_csv),
            "heatmap_png": str(heatmap_path),
        },
    }

    summary_path = run_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Walk-forward optimization to defend against overfitting")
    parser.add_argument("--symbol", default="XAUUSD")
    parser.add_argument("--from", dest="from_date", required=True, help="YYYY-MM-DD")
    parser.add_argument("--to", dest="to_date", required=True, help="YYYY-MM-DD")
    parser.add_argument("--output-dir", default="./walkforward_results")
    parser.add_argument("--data-csv", default=None, help="Optional OHLC CSV with time/open/high/low/close for offline run")
    args = parser.parse_args()

    from_dt = datetime.strptime(args.from_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    to_dt = datetime.strptime(args.to_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)

    summary = run_walk_forward(args.symbol, from_dt, to_dt, args.output_dir, data_csv=args.data_csv)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
