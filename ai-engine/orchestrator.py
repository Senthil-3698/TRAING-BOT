import asyncio
import httpx
import os
import MetaTrader5 as mt5
import numpy as np

from state_manager import get_integrated_bias, track_active_trade, auto_update_bias
from news_aggregator import fetch_macro_news
from strategist import validate_with_ai

EXECUTION_ENGINE_URL = os.getenv("EXECUTION_ENGINE_URL", "http://localhost:8081/execute")
# Minimum AI confidence to execute (0-100 scale)
MIN_CONFIDENCE = 60


def _compute_rsi(closes, period=14):
    if len(closes) < period + 1:
        return 50.0
    deltas = np.diff(closes)
    gains = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)
    avg_gain = np.mean(gains[:period])
    avg_loss = np.mean(losses[:period])
    for i in range(period, len(deltas)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period
    if avg_loss == 0:
        return 100.0
    return 100.0 - (100.0 / (1.0 + avg_gain / avg_loss))


def _enrich_signal(signal):
    """
    Attach live technical context to the signal so the AI has real data.
    Returns signal dict with rsi, ema_distance_pct, atr fields added.
    """
    try:
        symbol = signal.get("symbol", "XAUUSD")
        rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M5, 0, 60)
        if rates is None or len(rates) < 20:
            return signal

        closes = np.array([r["close"] for r in rates], dtype=float)
        highs = np.array([r["high"] for r in rates], dtype=float)
        lows = np.array([r["low"] for r in rates], dtype=float)

        # EMA50 on M5
        k = 2.0 / (50 + 1)
        ema = closes[0]
        for c in closes[1:]:
            ema = c * k + ema * (1 - k)

        current_price = closes[-1]
        ema_distance_pct = round((current_price - ema) / ema * 100, 4)

        # RSI(14)
        rsi = round(_compute_rsi(closes), 1)

        # ATR(14) on M5
        prev_close = closes[:-1]
        h = highs[1:]
        lo = lows[1:]
        tr = np.maximum(h - lo, np.maximum(np.abs(h - prev_close), np.abs(lo - prev_close)))
        atr = round(float(np.mean(tr[-14:])), 5)

        return {**signal, "rsi": rsi, "ema_distance_pct": ema_distance_pct, "atr_m5": atr}
    except Exception:
        return signal


async def on_signal_received(signal):
    """
    The Workflow of an Advanced Scalping Agent
    """
    symbol = signal['symbol']
    tf = signal['timeframe']
    action = signal['action']

    # 0. Keep higher-TF bias fresh before checking it
    auto_update_bias(symbol)

    # 1. Check Technical Confluence (The Filter)
    trend = get_integrated_bias(symbol)
    if trend not in ("NO_CONFLUENCE", action):
        print(f"[FILTER] Counter-trend ignored: {action} vs {trend} bias.")
        return

    # 2. Enrich signal with live technical data
    signal = _enrich_signal(signal)

    # 3. Get Global Context (The News)
    news_context = await fetch_macro_news()

    # 4. AI Final Veto (The Strategist)
    ai_result = await validate_with_ai({
        **signal,
        "context": news_context
    }, macro_bias=trend) or {}

    if isinstance(ai_result, dict):
        decision = ai_result.get("decision", "REJECTED")
        reason = ai_result.get("reason", ai_result.get("reasoning", ""))
        confidence = int(ai_result.get("confidence", 50))
    else:
        decision, reason = ai_result
        confidence = 50

    if isinstance(decision, str):
        decision = decision.upper()

    if reason:
        print(f"[AI] {decision} (conf={confidence}) — {reason}")

    if decision != "APPROVED":
        print(f"[VETO] Signal rejected by AI.")
        return

    if confidence < MIN_CONFIDENCE:
        print(f"[VETO] Confidence {confidence} below minimum {MIN_CONFIDENCE}.")
        return

    print(f"[EXECUTE] {action} {symbol} | conf={confidence} | RSI={signal.get('rsi', '?')}")
    payload = {
        "symbol": symbol,
        "action": action,
        "timeframe": tf,
        "confidence_score": confidence / 100.0
    }

    try:
        async with httpx.AsyncClient(timeout=5.0, follow_redirects=True) as client:
            response = await client.post(EXECUTION_ENGINE_URL, json=payload)

        if response.status_code == 200:
            try:
                result = response.json()
                order = result.get("order")
                entry_price = result.get("price")
                sl = result.get("sl")
                tp = result.get("tp")
                if order and entry_price is not None and sl is not None and tp is not None:
                    track_active_trade(order, entry_price, sl, tp)
            except ValueError:
                pass
            print("[DISPATCHED] Trade sent to execution engine.")
        elif response.status_code == 403:
            print("[BLOCKED] Kill switch or risk manager rejected the trade.")
        else:
            print(f"[ERROR] Execution engine returned {response.status_code}: {response.text}")
    except httpx.HTTPError as error:
        print(f"[ERROR] Dispatch failed: {error}")


if __name__ == "__main__":
    test_signal = {"symbol": "XAUUSD", "timeframe": "5m", "action": "BUY"}
    asyncio.run(on_signal_received(test_signal))
