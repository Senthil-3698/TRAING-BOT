import asyncio
import httpx
from state_manager import get_integrated_bias, track_active_trade
from news_aggregator import fetch_macro_news
from strategist import validate_with_ai

async def on_signal_received(signal):
    """
    The Workflow of an Advanced Scalping Agent
    """
    symbol = signal['symbol']
    tf = signal['timeframe']
    action = signal['action']

    # 1. Check Technical Confluence (The Filter)
    trend = get_integrated_bias(symbol)
    if trend != "NO_CONFLUENCE" and action != trend:
        print(f"Counter-trend signal ignored: {action} against {trend} trend.")
        return

    # 2. Get Global Context (The News)
    news_context = await fetch_macro_news()

    # 3. AI Final Veto (The Strategist)
    ai_result = await validate_with_ai({
        **signal,
        "context": news_context
    }, macro_bias=trend) or {}

    if isinstance(ai_result, dict):
        decision = ai_result.get("decision", "REJECTED")
        reason = ai_result.get("reason", ai_result.get("reasoning", ""))
    else:
        decision, reason = ai_result

    if isinstance(decision, str) and decision.upper() in {"APPROVED", "REJECTED"}:
        decision = decision.upper()

    if reason:
        print(f"AI_REASONING: {reason}")

    if decision == "APPROVED":
        print(f"!!! SIGNAL VALIDATED: Executing {action} on {symbol} !!!")
        payload = {
            "symbol": symbol,
            "action": action,
            "timeframe": tf,
            "confidence_score": signal.get("confidence_score", 0)
        }

        try:
            async with httpx.AsyncClient(timeout=5.0, follow_redirects=True) as client:
                response = await client.post("http://localhost:8080/execute", json=payload)

            if response.status_code == 200:
                try:
                    result = response.json()
                    entry_price = result.get("price")
                    sl = result.get("sl")
                    tp = result.get("tp")
                    if result.get("order") and entry_price is not None and sl is not None and tp is not None:
                        track_active_trade(result.get("order"), entry_price, sl, tp)
                except ValueError:
                    print("Execution engine response was not JSON; skipped trade tracking.")
                print("TRADE_DISPATCHED")
            elif response.status_code == 403:
                print("SENTINEL_BLOCK")
            else:
                print(f"Execution engine returned {response.status_code}: {response.text}")
        except httpx.HTTPError as error:
            print(f"Execution dispatch failed: {error}")
    else:
        print(f"Signal Vetoed by AI: {reason}")

if __name__ == "__main__":
    # Example Test Signal
    test_signal = {"symbol": "XAUUSD", "timeframe": "5m", "action": "BUY"}
    asyncio.run(on_signal_received(test_signal))