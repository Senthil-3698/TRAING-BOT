from __future__ import annotations

import os
import time
from datetime import datetime, timezone
from typing import Any

import httpx

try:
    import MetaTrader5 as mt5  # type: ignore
except ImportError:
    mt5 = None  # type: ignore


_CACHE: dict[str, tuple[float, Any]] = {}
_CACHE_TTL_SECONDS = 60


def _cache_get(key: str):
    record = _CACHE.get(key)
    if not record:
        return None
    ts, value = record
    if time.time() - ts > _CACHE_TTL_SECONDS:
        return None
    return value


def _cache_set(key: str, value: Any) -> Any:
    _CACHE[key] = (time.time(), value)
    return value


def _ensure_mt5() -> bool:
    if mt5 is None:
        return False
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


def _mt5_close_series(symbol_candidates: list[str], bars: int = 24):
    if not _ensure_mt5():
        return None, None

    for candidate in symbol_candidates:
        if not mt5.symbol_select(candidate, True):
            continue
        rates = mt5.copy_rates_from_pos(candidate, mt5.TIMEFRAME_M5, 0, bars)
        if rates is not None and len(rates) >= bars:
            closes = [float(r["close"]) for r in rates]
            return candidate, closes

    return None, None


def _yahoo_close_series(ticker: str, interval: str = "5m", range_str: str = "2d"):
    key = f"yahoo:{ticker}:{interval}:{range_str}"
    cached = _cache_get(key)
    if cached is not None:
        return cached

    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"
    params = {"interval": interval, "range": range_str}
    try:
        with httpx.Client(timeout=8.0) as client:
            response = client.get(url, params=params)
            response.raise_for_status()
            payload = response.json()

        result = payload.get("chart", {}).get("result", [])
        if not result:
            return _cache_set(key, None)

        closes = result[0].get("indicators", {}).get("quote", [{}])[0].get("close", [])
        clean = [float(x) for x in closes if x is not None]
        return _cache_set(key, clean if len(clean) >= 12 else None)
    except Exception:
        return _cache_set(key, None)


def _fred_latest_value(series_id: str):
    api_key = os.getenv("FRED_API_KEY")
    if not api_key:
        return None

    key = f"fred:{series_id}"
    cached = _cache_get(key)
    if cached is not None:
        return cached

    url = "https://api.stlouisfed.org/fred/series/observations"
    params = {
        "series_id": series_id,
        "api_key": api_key,
        "file_type": "json",
        "sort_order": "desc",
        "limit": 5,
    }

    try:
        with httpx.Client(timeout=8.0) as client:
            response = client.get(url, params=params)
            response.raise_for_status()
            payload = response.json()

        observations = payload.get("observations", [])
        for obs in observations:
            value = obs.get("value")
            if value not in (None, "."):
                return _cache_set(key, float(value))
    except Exception:
        pass

    return _cache_set(key, None)


def _safe_pct_change(series: list[float], periods: int) -> float:
    if series is None or len(series) <= periods:
        return 0.0
    prev = series[-periods - 1]
    curr = series[-1]
    if prev == 0:
        return 0.0
    return (curr - prev) / prev


def get_intermarket_context() -> dict[str, Any]:
    """Build intermarket context for XAUUSD using MT5-first, API-fallback data sources."""
    cache_key = "intermarket_context"
    cached = _cache_get(cache_key)
    if cached is not None:
        return cached

    # 1 hour on M5 = 12 bars.
    lookback_bars = 24

    dxy_symbol, dxy_series = _mt5_close_series(["DXY", "USDX", "USDX.i"], bars=lookback_bars)
    if not dxy_series:
        dxy_symbol = "DX-Y.NYB"
        dxy_series = _yahoo_close_series("DX-Y.NYB")

    us10y_symbol, us10y_series = _mt5_close_series(["US10Y", "TNX", "UST10Y"], bars=lookback_bars)
    if not us10y_series:
        us10y_symbol = "^TNX"
        us10y_series = _yahoo_close_series("^TNX")

    spx_symbol, spx_series = _mt5_close_series(["SPX500", "US500", "SPX"], bars=lookback_bars)
    if not spx_series:
        spx_symbol = "^GSPC"
        spx_series = _yahoo_close_series("^GSPC")

    vix_symbol, vix_series = _mt5_close_series(["VIX", "VOLX"], bars=lookback_bars)
    if not vix_series:
        vix_symbol = "^VIX"
        vix_series = _yahoo_close_series("^VIX")

    dxy_change_1h = _safe_pct_change(dxy_series or [], periods=12)
    dxy_breakout_up = False
    dxy_breakout_down = False
    if dxy_series and len(dxy_series) >= 12:
        prior = dxy_series[-12:-1]
        if prior:
            dxy_breakout_up = dxy_series[-1] > max(prior) * 1.0005
            dxy_breakout_down = dxy_series[-1] < min(prior) * 0.9995

    us10y_current = None
    if us10y_series and len(us10y_series) > 0:
        us10y_current = float(us10y_series[-1])
        # Yahoo ^TNX is typically 10x the percentage yield.
        if us10y_symbol == "^TNX":
            us10y_current = us10y_current / 10.0

    cpi_expectation = _fred_latest_value("T5YIE")
    real_yield_proxy = None
    real_yield_bias = "UNKNOWN"
    if us10y_current is not None and cpi_expectation is not None:
        real_yield_proxy = us10y_current - cpi_expectation
        real_yield_bias = "RISING_BEARISH_GOLD" if real_yield_proxy > 0 else "FALLING_BULLISH_GOLD"

    spx_change_1h = _safe_pct_change(spx_series or [], periods=12)
    vix_change_1h = _safe_pct_change(vix_series or [], periods=12)
    if spx_change_1h > 0 and vix_change_1h < 0:
        risk_state = "RISK_ON"
    elif spx_change_1h < 0 and vix_change_1h > 0:
        risk_state = "RISK_OFF"
    else:
        risk_state = "MIXED"

    summary_parts = [
        f"DXY({dxy_symbol}) 1h={dxy_change_1h * 100:.2f}%",
        f"breakout_up={dxy_breakout_up}",
        f"breakout_down={dxy_breakout_down}",
        f"US10Y({us10y_symbol})={us10y_current if us10y_current is not None else 'NA'}",
        f"CPIexp(T5YIE)={cpi_expectation if cpi_expectation is not None else 'NA'}",
        f"realYieldProxy={real_yield_proxy if real_yield_proxy is not None else 'NA'} ({real_yield_bias})",
        f"SPX({spx_symbol}) 1h={spx_change_1h * 100:.2f}%",
        f"VIX({vix_symbol}) 1h={vix_change_1h * 100:.2f}%",
        f"risk_state={risk_state}",
    ]

    context = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "dxy_symbol": dxy_symbol,
        "dxy_change_1h": dxy_change_1h,
        "dxy_breakout_up": dxy_breakout_up,
        "dxy_breakout_down": dxy_breakout_down,
        "us10y_symbol": us10y_symbol,
        "us10y_current": us10y_current,
        "cpi_expectation": cpi_expectation,
        "real_yield_proxy": real_yield_proxy,
        "real_yield_bias": real_yield_bias,
        "spx_symbol": spx_symbol,
        "spx_change_1h": spx_change_1h,
        "vix_symbol": vix_symbol,
        "vix_change_1h": vix_change_1h,
        "risk_state": risk_state,
        "summary": " | ".join(summary_parts),
    }

    return _cache_set(cache_key, context)
