from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Any

import MetaTrader5 as mt5
import numpy as np
import pandas as pd
import redis
from dotenv import load_dotenv

load_dotenv()

REGIME_KEY_PREFIX = os.getenv("REGIME_CACHE_KEY_PREFIX", "market:regime")
REGIME_REFRESH_SECONDS = int(os.getenv("REGIME_REFRESH_SECONDS", "300"))
REGIME_LOOKBACK_BARS = int(os.getenv("REGIME_LOOKBACK_BARS", "240"))
REGIME_HYSTERESIS_CONFIRMATIONS = int(os.getenv("REGIME_HYSTERESIS_CONFIRMATIONS", "2"))
REGIME_STATE_TTL_SECONDS = int(os.getenv("REGIME_STATE_TTL_SECONDS", str(86400)))

ADX_RANGING_MAX = float(os.getenv("ADX_RANGING_MAX", "20"))
ADX_TRENDING_MIN = float(os.getenv("ADX_TRENDING_MIN", "25"))
ADX_TRANSITIONAL_MIN = float(os.getenv("ADX_TRANSITIONAL_MIN", "20"))
ADX_TRANSITIONAL_MAX = float(os.getenv("ADX_TRANSITIONAL_MAX", "25"))

# Slope normalized by ATR (%-move over window divided by ATR%-of-price).
SLOPE_TO_ATR_TREND_THRESHOLD = float(os.getenv("SLOPE_TO_ATR_TREND_THRESHOLD", "0.35"))


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


def _timeframe_value(timeframe: str):
    mapping = {
        "M1": mt5.TIMEFRAME_M1,
        "M5": mt5.TIMEFRAME_M5,
        "M15": mt5.TIMEFRAME_M15,
        "H1": mt5.TIMEFRAME_H1,
    }
    return mapping.get(timeframe.upper(), mt5.TIMEFRAME_M5)


def _to_df(rates) -> pd.DataFrame:
    if rates is None or len(rates) == 0:
        return pd.DataFrame()
    df = pd.DataFrame(rates)
    if "time" in df.columns:
        df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
    return df


def _wilder_smooth(values: np.ndarray, period: int) -> np.ndarray:
    out = np.zeros_like(values, dtype=float)
    if len(values) < period:
        return out
    out[period - 1] = np.sum(values[:period])
    for idx in range(period, len(values)):
        out[idx] = out[idx - 1] - (out[idx - 1] / period) + values[idx]
    return out


def _compute_adx(df: pd.DataFrame, period: int = 14) -> float:
    if len(df) < period + 5:
        return 0.0

    high = df["high"].to_numpy(dtype=float)
    low = df["low"].to_numpy(dtype=float)
    close = df["close"].to_numpy(dtype=float)

    plus_dm = np.zeros_like(high)
    minus_dm = np.zeros_like(high)
    tr = np.zeros_like(high)

    for i in range(1, len(high)):
        up_move = high[i] - high[i - 1]
        down_move = low[i - 1] - low[i]
        plus_dm[i] = up_move if up_move > down_move and up_move > 0 else 0.0
        minus_dm[i] = down_move if down_move > up_move and down_move > 0 else 0.0

        tr1 = high[i] - low[i]
        tr2 = abs(high[i] - close[i - 1])
        tr3 = abs(low[i] - close[i - 1])
        tr[i] = max(tr1, tr2, tr3)

    atr_w = _wilder_smooth(tr, period)
    plus_dm_w = _wilder_smooth(plus_dm, period)
    minus_dm_w = _wilder_smooth(minus_dm, period)

    with np.errstate(divide="ignore", invalid="ignore"):
        plus_di = 100.0 * (plus_dm_w / atr_w)
        minus_di = 100.0 * (minus_dm_w / atr_w)
        dx = 100.0 * np.abs(plus_di - minus_di) / np.where((plus_di + minus_di) == 0, np.nan, (plus_di + minus_di))

    dx = np.nan_to_num(dx, nan=0.0, posinf=0.0, neginf=0.0)
    if len(dx) < period * 2:
        return float(dx[-1]) if len(dx) else 0.0

    adx = np.zeros_like(dx)
    adx[period * 2 - 1] = np.mean(dx[period - 1 : period * 2 - 1])
    for i in range(period * 2, len(dx)):
        adx[i] = ((adx[i - 1] * (period - 1)) + dx[i]) / period

    return float(adx[-1])


def _compute_atr_series(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high_low = df["high"] - df["low"]
    high_close = (df["high"] - df["close"].shift(1)).abs()
    low_close = (df["low"] - df["close"].shift(1)).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return tr.rolling(period, min_periods=period).mean()


def _atr_pct_of_price(df: pd.DataFrame, period: int = 14) -> float:
    atr = _compute_atr_series(df, period=period).dropna()
    if atr.empty:
        return 0.0
    latest_atr = float(atr.iloc[-1])
    latest_close = float(df["close"].iloc[-1]) if not df.empty else 0.0
    if latest_close == 0.0:
        return 0.0
    return latest_atr / latest_close


def _atr_percentile(df: pd.DataFrame, period: int = 14, lookback: int = 120) -> float:
    atr = _compute_atr_series(df, period)
    tail = atr.dropna().tail(lookback)
    if len(tail) < 20:
        return 0.5
    latest = float(tail.iloc[-1])
    pct = float((tail <= latest).sum() / len(tail))
    return max(0.0, min(1.0, pct))


def _bollinger_bandwidth_percentile(df: pd.DataFrame, period: int = 20, lookback: int = 120) -> tuple[float, float]:
    close = df["close"]
    ma = close.rolling(period, min_periods=period).mean()
    std = close.rolling(period, min_periods=period).std(ddof=0)
    upper = ma + (2 * std)
    lower = ma - (2 * std)

    bandwidth = (upper - lower) / ma.replace(0.0, np.nan)
    bandwidth = bandwidth.replace([np.inf, -np.inf], np.nan).dropna()
    tail = bandwidth.tail(lookback)
    if len(tail) < 20:
        return 0.0, 0.5

    latest = float(tail.iloc[-1])
    pct = float((tail <= latest).sum() / len(tail))
    return latest, max(0.0, min(1.0, pct))


def _swing_points(df: pd.DataFrame, lookback: int = 80, neighborhood: int = 2) -> int:
    if len(df) < lookback + (neighborhood * 2) + 1:
        return 0

    sample = df.tail(lookback).reset_index(drop=True)
    highs = sample["high"].to_numpy(dtype=float)
    lows = sample["low"].to_numpy(dtype=float)

    swings = 0
    for i in range(neighborhood, len(sample) - neighborhood):
        if highs[i] == np.max(highs[i - neighborhood : i + neighborhood + 1]):
            swings += 1
        elif lows[i] == np.min(lows[i - neighborhood : i + neighborhood + 1]):
            swings += 1
    return swings


def _close_autocorr(df: pd.DataFrame, lookback: int = 80, lag: int = 1) -> float:
    close = df["close"].tail(lookback)
    if len(close) < lag + 5:
        return 0.0
    returns = close.pct_change().dropna()
    if len(returns) < lag + 5:
        return 0.0
    corr = returns.autocorr(lag=lag)
    if corr is None or np.isnan(corr):
        return 0.0
    return float(corr)


def _close_slope_pct(df: pd.DataFrame, window: int = 30) -> float:
    close = df["close"].tail(window)
    if len(close) < 2:
        return 0.0
    start = float(close.iloc[0])
    end = float(close.iloc[-1])
    if start == 0:
        return 0.0
    return (end - start) / start


def _classify(features: dict[str, float]) -> tuple[str, str]:
    adx = features["adx"]
    atr_pct = features["atr_percentile"]
    bb_pct = features["bb_bandwidth_percentile"]
    swing_points = features["swing_points"]
    autocorr = features["close_autocorr"]
    slope_pct = features["close_slope_pct"]
    atr_pct_of_price = max(float(features.get("atr_pct_of_price", 0.0)), 1e-9)
    slope_to_atr = abs(slope_pct) / atr_pct_of_price

    if atr_pct >= 0.85 and bb_pct >= 0.80 and adx >= 20:
        if slope_pct >= 0:
            return "HIGH_VOL_BREAKOUT", "High volatility expansion with upward pressure"
        return "HIGH_VOL_BREAKOUT", "High volatility expansion with downward pressure"

    if adx >= ADX_TRENDING_MIN and autocorr >= 0.12 and slope_pct > 0 and slope_to_atr >= SLOPE_TO_ATR_TREND_THRESHOLD:
        return "TRENDING_UP", "Strong directional persistence and positive slope"

    if adx >= ADX_TRENDING_MIN and autocorr <= -0.05 and slope_pct < 0 and slope_to_atr >= SLOPE_TO_ATR_TREND_THRESHOLD:
        return "TRENDING_DOWN", "Strong directional persistence and negative slope"

    if atr_pct <= 0.25 and bb_pct <= 0.30 and adx <= 18:
        return "LOW_VOL_DRIFT", "Compressed volatility and weak directional drive"

    if adx <= ADX_RANGING_MAX and bb_pct <= 0.55 and swing_points >= 12:
        return "RANGING", "Frequent swings with weak trend quality"

    # Explicit ADX gray zone handling: 20-25 is transitional unless features strongly confirm trend.
    if ADX_TRANSITIONAL_MIN <= adx < ADX_TRANSITIONAL_MAX:
        return "TRANSITIONAL", "ADX gray zone (20-25): transition regime, avoid hard vetoes"

    if adx >= 22 and slope_pct >= 0 and slope_to_atr >= SLOPE_TO_ATR_TREND_THRESHOLD:
        return "TRENDING_UP", "Fallback trend-up classification"
    if adx >= 22 and slope_pct < 0 and slope_to_atr >= SLOPE_TO_ATR_TREND_THRESHOLD:
        return "TRENDING_DOWN", "Fallback trend-down classification"

    return "RANGING", "Default classification: mixed structure"


def compute_regime(symbol: str, timeframe: str = "M5", bars: int = REGIME_LOOKBACK_BARS) -> dict[str, Any]:
    if not _ensure_mt5():
        return {
            "symbol": symbol,
            "timeframe": timeframe,
            "regime": "UNKNOWN",
            "reason": f"MT5 init failed: {mt5.last_error()}",
            "features": {},
            "computed_at": datetime.now(timezone.utc).isoformat(),
        }

    tf = _timeframe_value(timeframe)
    rates = mt5.copy_rates_from_pos(symbol, tf, 0, bars)
    df = _to_df(rates)
    if df.empty or len(df) < 80:
        return {
            "symbol": symbol,
            "timeframe": timeframe,
            "regime": "UNKNOWN",
            "reason": "Insufficient market data",
            "features": {},
            "computed_at": datetime.now(timezone.utc).isoformat(),
        }

    features: dict[str, float] = {
        "adx": round(_compute_adx(df), 4),
        "atr_percentile": round(_atr_percentile(df), 4),
        "atr_pct_of_price": round(_atr_pct_of_price(df), 6),
        "swing_points": float(_swing_points(df, lookback=80)),
        "close_autocorr": round(_close_autocorr(df, lookback=80), 4),
        "close_slope_pct": round(_close_slope_pct(df, window=30), 6),
    }

    bb_abs, bb_pct = _bollinger_bandwidth_percentile(df)
    features["bb_bandwidth"] = round(bb_abs, 6)
    features["bb_bandwidth_percentile"] = round(bb_pct, 4)

    regime, reason = _classify(features)

    return {
        "symbol": symbol,
        "timeframe": timeframe,
        "regime": regime,
        "reason": reason,
        "features": features,
        "computed_at": datetime.now(timezone.utc).isoformat(),
    }


def _apply_hysteresis(
    cache: redis.Redis,
    key: str,
    computed_payload: dict[str, Any],
    confirmations: int = REGIME_HYSTERESIS_CONFIRMATIONS,
) -> dict[str, Any]:
    """Require N consecutive candidate readings before switching away from active regime."""
    state_key = f"{key}:state"

    default_state = {
        "active_regime": computed_payload.get("regime", "UNKNOWN"),
        "candidate_regime": computed_payload.get("regime", "UNKNOWN"),
        "candidate_count": 1,
        "last_update": datetime.now(timezone.utc).isoformat(),
    }

    try:
        raw_state = cache.get(state_key)
        state = json.loads(raw_state.decode("utf-8")) if raw_state else default_state
    except Exception:
        state = default_state

    active = state.get("active_regime", default_state["active_regime"])
    candidate = computed_payload.get("regime", "UNKNOWN")

    if candidate == active:
        state["candidate_regime"] = candidate
        state["candidate_count"] = 0
        final_regime = active
        final_reason = computed_payload.get("reason", "")
    else:
        if state.get("candidate_regime") == candidate:
            state["candidate_count"] = int(state.get("candidate_count", 0)) + 1
        else:
            state["candidate_regime"] = candidate
            state["candidate_count"] = 1

        if int(state.get("candidate_count", 0)) >= max(1, confirmations):
            state["active_regime"] = candidate
            state["candidate_count"] = 0
            final_regime = candidate
            final_reason = f"{computed_payload.get('reason', '')}; hysteresis switch confirmed"
        else:
            final_regime = active
            final_reason = (
                f"Hysteresis hold: pending switch {active} -> {candidate} "
                f"({state.get('candidate_count')}/{max(1, confirmations)})"
            )

    state["last_update"] = datetime.now(timezone.utc).isoformat()
    try:
        cache.setex(state_key, REGIME_STATE_TTL_SECONDS, json.dumps(state, default=str))
    except Exception:
        pass

    out = dict(computed_payload)
    out["raw_regime"] = computed_payload.get("regime")
    out["regime"] = final_regime
    out["reason"] = final_reason
    out["hysteresis"] = {
        "active_regime": state.get("active_regime"),
        "candidate_regime": state.get("candidate_regime"),
        "candidate_count": int(state.get("candidate_count", 0)),
        "required_confirmations": max(1, confirmations),
    }
    return out


def get_current_regime(symbol: str, timeframe: str = "M5") -> dict[str, Any]:
    key = f"{REGIME_KEY_PREFIX}:{symbol.upper()}:{timeframe.upper()}"
    cache = _redis_client()

    raw = cache.get(key)
    if raw:
        try:
            parsed = json.loads(raw.decode("utf-8"))
            if isinstance(parsed, dict) and parsed.get("regime"):
                return parsed
        except Exception:
            pass

    regime_payload = compute_regime(symbol=symbol, timeframe=timeframe)
    regime_payload = _apply_hysteresis(cache, key, regime_payload)
    cache.setex(key, REGIME_REFRESH_SECONDS, json.dumps(regime_payload, default=str))
    return regime_payload
