from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def _find_run_dir(root: Path, run_dir_arg: str | None) -> Path:
    if run_dir_arg:
        candidate = Path(run_dir_arg)
        if not candidate.is_absolute():
            candidate = root / candidate
        return candidate

    candidates = []
    for d in root.iterdir():
        if d.is_dir() and (d / "metrics.json").exists() and (d / "trades.csv").exists():
            candidates.append(d)
    if not candidates:
        raise RuntimeError(f"No backtest run directory found under {root}")
    return max(candidates, key=lambda p: p.stat().st_mtime)


def _session_from_hour(hour: int) -> str:
    if 13 <= hour < 17:
        return "OVERLAP"
    if 8 <= hour < 13:
        return "LONDON"
    if 17 <= hour < 21:
        return "NEW_YORK"
    return "OFF_SESSION"


def _winrate_table(df: pd.DataFrame, group_col: str, order: list[str] | None = None) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=[group_col, "trades", "wins", "win_rate_pct"])

    agg = (
        df.assign(is_win=df["pnl_usd"] > 0)
        .groupby(group_col, dropna=False)
        .agg(trades=("pnl_usd", "count"), wins=("is_win", "sum"))
        .reset_index()
    )
    agg["win_rate_pct"] = (agg["wins"] / agg["trades"] * 100).round(2)

    if order:
        rank = {k: i for i, k in enumerate(order)}
        agg["_rank"] = agg[group_col].map(rank).fillna(999)
        agg = agg.sort_values(["_rank", group_col]).drop(columns=["_rank"])
    else:
        agg = agg.sort_values(group_col)

    return agg


def _markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "No data."

    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for _, row in df.iterrows():
        vals = [str(row[c]) for c in cols]
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def _equity_sparkline(series: pd.Series, width: int = 60) -> str:
    blocks = "▁▂▃▄▅▆▇█"
    if series.empty:
        return ""
    values = series.astype(float)
    idx = np.linspace(0, len(values) - 1, num=min(width, len(values))).astype(int)
    sampled = values.iloc[idx]
    lo = sampled.min()
    hi = sampled.max()
    if hi == lo:
        return blocks[0] * len(sampled)
    scaled = (sampled - lo) / (hi - lo)
    chars = [blocks[int(round(v * (len(blocks) - 1)))] for v in scaled]
    return "".join(chars)


def _monthly_returns_heatmap(eq_df: pd.DataFrame) -> pd.DataFrame:
    if eq_df.empty:
        return pd.DataFrame()

    daily = eq_df.set_index("time")["equity"].resample("1D").last().ffill().dropna()
    monthly = daily.resample("ME").last().pct_change() * 100
    m = monthly.to_frame("ret")
    m["year"] = m.index.year
    m["month"] = m.index.strftime("%b")

    pivot = m.pivot(index="year", columns="month", values="ret")
    month_order = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    present_cols = [c for c in month_order if c in pivot.columns]
    pivot = pivot[present_cols]
    return pivot.round(2)


def _r_bucket(r: float) -> str:
    if r <= -1.0:
        return "<= -1R"
    if r <= 0.0:
        return "(-1R, 0R]"
    if r <= 1.0:
        return "(0R, 1R]"
    if r <= 2.0:
        return "(1R, 2R]"
    return "> 2R"


def generate_report(run_dir: Path, output_md: Path):
    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    trades = pd.read_csv(run_dir / "trades.csv")
    eq_df = pd.read_csv(run_dir / "equity_curve.csv")

    trades["open_time"] = pd.to_datetime(trades["open_time"], utc=True, errors="coerce")
    trades["close_time"] = pd.to_datetime(trades["close_time"], utc=True, errors="coerce")
    eq_df["time"] = pd.to_datetime(eq_df["time"], utc=True, errors="coerce")

    if "session_label" not in trades.columns:
        trades["session_label"] = trades["open_time"].dt.hour.fillna(0).astype(int).map(_session_from_hour)
    if "day_of_week" not in trades.columns:
        trades["day_of_week"] = trades["open_time"].dt.day_name().fillna("Unknown")

    trades["holding_minutes"] = (trades["close_time"] - trades["open_time"]).dt.total_seconds() / 60.0
    trades["r_bucket"] = trades["r_multiple"].apply(_r_bucket)

    headline = pd.DataFrame(
        [
            ["Total Trades", metrics.get("total_trades", 0)],
            ["Win Rate %", metrics.get("win_rate_pct", 0)],
            ["Profit Factor", metrics.get("profit_factor", 0)],
            ["Total Return %", metrics.get("total_return_pct", 0)],
            ["Max Drawdown %", metrics.get("max_drawdown_pct", 0)],
            ["Sharpe Ratio", metrics.get("sharpe_ratio", 0)],
            ["Sortino Ratio", metrics.get("sortino_ratio", 0)],
            ["Expectancy USD", metrics.get("expectancy_usd", 0)],
            ["Final Balance", metrics.get("final_balance", 0)],
        ],
        columns=["Metric", "Value"],
    )

    session_order = ["LONDON", "OVERLAP", "NEW_YORK", "OFF_SESSION"]
    day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    bias_order = ["ALIGNED", "NO_CONFLUENCE", "COUNTER_TREND"]

    by_session = _winrate_table(trades, "session_label", session_order)
    by_day = _winrate_table(trades, "day_of_week", day_order)

    if "bias_alignment" in trades.columns:
        by_bias = _winrate_table(trades, "bias_alignment", bias_order)
    else:
        by_bias = pd.DataFrame(columns=["bias_alignment", "trades", "wins", "win_rate_pct"])

    mfe_mae = pd.DataFrame(
        [
            ["Avg MFE (R)", round(float(trades.get("mfe_r", pd.Series(dtype=float)).mean()), 3) if "mfe_r" in trades else "N/A"],
            ["Avg MAE (R, adverse)", round(float(trades.get("mae_r", pd.Series(dtype=float)).mean()), 3) if "mae_r" in trades else "N/A"],
        ],
        columns=["Metric", "Value"],
    )

    holding_bins = pd.cut(
        trades["holding_minutes"],
        bins=[-np.inf, 5, 15, 30, 60, 120, np.inf],
        labels=["<=5m", "5-15m", "15-30m", "30-60m", "60-120m", ">120m"],
    )
    hold_dist = (
        holding_bins.value_counts(dropna=False)
        .rename_axis("holding_bin")
        .reset_index(name="trades")
        .sort_values("holding_bin")
    )

    stage_col = "stage_at_close" if "stage_at_close" in trades.columns else "close_reason"
    exit_stage = (
        trades.assign(is_win=trades["pnl_usd"] > 0)
        .groupby(stage_col)
        .agg(trades=("pnl_usd", "count"), wins=("is_win", "sum"), losses=("is_win", lambda s: int((~s).sum())))
        .reset_index()
    )

    exit_reason = (
        trades.assign(is_win=trades["pnl_usd"] > 0)
        .groupby(["close_reason", "r_bucket"])  # exit stage and R-level outcomes
        .agg(trades=("pnl_usd", "count"), wins=("is_win", "sum"))
        .reset_index()
        .sort_values(["close_reason", "r_bucket"])
    )

    month_heatmap = _monthly_returns_heatmap(eq_df)
    month_heatmap_reset = month_heatmap.reset_index() if not month_heatmap.empty else pd.DataFrame()

    equity_rel_png = run_dir.name + "/equity_curve.png"
    sparkline = _equity_sparkline(eq_df["equity"] if "equity" in eq_df else pd.Series(dtype=float))

    report = []
    report.append("# BASELINE Backtest Report (XAUUSD, 12 Months)")
    report.append("")
    report.append(f"Run directory: {run_dir}")
    report.append("")
    report.append("## Headline Stats")
    report.append(_markdown_table(headline))
    report.append("")
    report.append("## Equity Curve")
    report.append(f"![Equity Curve]({equity_rel_png})")
    report.append("")
    report.append("Sparkline:")
    report.append("")
    report.append("```text")
    report.append(sparkline)
    report.append("```")
    report.append("")
    report.append("## Monthly Returns Heatmap (%)")
    report.append(_markdown_table(month_heatmap_reset) if not month_heatmap_reset.empty else "No monthly data.")
    report.append("")
    report.append("## Win Rate by Session")
    report.append(_markdown_table(by_session))
    report.append("")
    report.append("## Win Rate by Day of Week")
    report.append(_markdown_table(by_day))
    report.append("")
    report.append("## Win Rate by H1/H4 Bias Alignment")
    report.append(_markdown_table(by_bias))
    report.append("")
    report.append("## Avg MFE vs MAE")
    report.append(_markdown_table(mfe_mae))
    report.append("")
    report.append("## Holding-Time Distribution")
    report.append(_markdown_table(hold_dist))
    report.append("")
    report.append("## Wins/Losses by Exit Stage")
    report.append(_markdown_table(exit_stage))
    report.append("")
    report.append("## R-Level Outcomes by Exit Reason")
    report.append(_markdown_table(exit_reason))
    report.append("")

    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_md.write_text("\n".join(report), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Generate BASELINE.md from backtest outputs")
    parser.add_argument("--results-root", default="./backtest_results", help="Root backtest results folder")
    parser.add_argument("--run-dir", default=None, help="Specific run directory (name or absolute path)")
    parser.add_argument("--output", default="./backtest_results/BASELINE.md", help="Output markdown path")
    args = parser.parse_args()

    root = Path(args.results_root)
    run_dir = _find_run_dir(root, args.run_dir)
    generate_report(run_dir, Path(args.output))
    print(f"[REPORT] Wrote {args.output} from run {run_dir}")


if __name__ == "__main__":
    main()
