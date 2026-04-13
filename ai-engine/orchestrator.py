import asyncio
import httpx
import os
from datetime import datetime, timezone
from state_manager import auto_update_bias, get_integrated_bias, track_active_trade
from news_aggregator import fetch_macro_news
from strategist import validate_with_ai
from trade_journal import TradeJournal
from intermarket import get_intermarket_context
from regime_detector import get_current_regime, is_trade_regime_allowed
from alerts import send_telegram_alert
from zmq_bridge import publish_signal_async

EXECUTION_ENGINE_URL = os.getenv("EXECUTION_ENGINE_URL", "http://localhost:8081/execute")
journal = TradeJournal()


def _signal_ts(signal):
    ts = signal.get("timestamp")
    if isinstance(ts, datetime):
        return ts
    try:
        if ts is not None:
            return datetime.fromtimestamp(float(ts), tz=timezone.utc)
    except (TypeError, ValueError):
        pass
    return datetime.now(timezone.utc)


def _indicators(signal):
    indicators = signal.get("indicators") or {}
    return {
        "rsi_value": indicators.get("rsi"),
        "ema_distance": indicators.get("ema_distance"),
        "atr_value": indicators.get("atr"),
        "m5_trend": indicators.get("m5_trend"),
        "h1_bias": indicators.get("h1_bias"),
        "h4_bias": indicators.get("h4_bias"),
        "integrated_bias": indicators.get("integrated_bias"),
    }

async def on_signal_received(signal):
    """
    The Workflow of an Advanced Scalping Agent
    """
    symbol = signal['symbol']
    tf = signal['timeframe']
    action = signal['action']
    setup_type = signal.get("setup_type")
    intermarket_context = None
    try:
        regime_context = get_current_regime(symbol, timeframe="M5")
    except Exception as error:
        regime_context = {"regime": "UNKNOWN", "reason": f"regime_detector_unavailable: {error}", "features": {}}
        print(f"[ORCHESTRATOR] Regime detector unavailable, continuing fast-path: {error}")
    signal["market_regime"] = regime_context.get("regime", "UNKNOWN")
    signal["regime_context"] = regime_context

    enforce_regime_filter = os.getenv("ENFORCE_EXPANSION_REGIME_FILTER", "1") == "1"
    if enforce_regime_filter:
        allowed, regime_reason = is_trade_regime_allowed(regime_context, action=action)
        if not allowed:
            rejection_reason = f"Regime hard filter veto: {regime_reason}"
            print(f"[ORCHESTRATOR] {rejection_reason}")
            journal.log_signal(
                source=signal.get("source", "orchestrator"),
                symbol=symbol,
                action=action,
                timeframe=tf,
                signal_ts=_signal_ts(signal),
                **_indicators(signal),
                ai_decision="REJECTED",
                ai_reasoning=rejection_reason,
                ai_confidence=signal.get("confidence_score"),
                decision_status="REJECTED",
                rejection_reason=rejection_reason,
                metadata={"stage": "hard_regime_filter", "regime": regime_context},
            )
            return

    bypass_ai_news_gate = os.getenv("BYPASS_AI_NEWS_GATE", "0") == "1"
    ai_structured = None

    if bypass_ai_news_gate:
        decision = "APPROVED"
        reason = "HFT fast-path: AI/news/intermarket gate bypassed."
        try:
            auto_update_bias(symbol)
        except Exception as error:
            print(f"[ORCHESTRATOR] Bias refresh skipped: {error}")
        ai_confidence = signal.get("confidence_score")
        ai_context = "BYPASS_AI_NEWS_GATE=1"
        print("[ORCHESTRATOR] HFT bypass active: dispatching without AI/news gate.")
    else:
        # 1. Check Technical Confluence (The Filter)
        trend = get_integrated_bias(symbol)
        if trend != "NO_CONFLUENCE" and action != trend:
            rejection_reason = f"Counter-trend signal ignored: {action} against {trend} trend."
            print(rejection_reason)
            journal.log_signal(
                source=signal.get("source", "orchestrator"),
                symbol=symbol,
                action=action,
                timeframe=tf,
                signal_ts=_signal_ts(signal),
                **_indicators(signal),
                ai_decision="REJECTED",
                ai_reasoning=rejection_reason,
                ai_confidence=signal.get("confidence_score"),
                decision_status="REJECTED",
                rejection_reason=rejection_reason,
                metadata={"stage": "trend_filter", "regime": regime_context},
            )
            return

        # 2. Get Global Context (The News)
        news_context = await fetch_macro_news()

        # 2b. Intermarket context for gold (DXY, real yield proxy, SPX/VIX risk regime)
        if symbol == "XAUUSD":
            intermarket_context = get_intermarket_context()
            dxy_block_long = action == "BUY" and intermarket_context.get("dxy_breakout_up")
            dxy_block_short = action == "SELL" and intermarket_context.get("dxy_breakout_down")
            if dxy_block_long or dxy_block_short:
                rejection_reason = (
                    "Intermarket veto: DXY breakout up blocks gold longs."
                    if dxy_block_long
                    else "Intermarket veto: DXY breakout down blocks gold shorts."
                )
                print(rejection_reason)
                journal.log_signal(
                    source=signal.get("source", "orchestrator"),
                    symbol=symbol,
                    action=action,
                    timeframe=tf,
                    signal_ts=_signal_ts(signal),
                    **_indicators(signal),
                    news_context=news_context,
                    ai_decision="REJECTED",
                    ai_reasoning=rejection_reason,
                    ai_confidence=signal.get("confidence_score"),
                    decision_status="REJECTED",
                    rejection_reason=rejection_reason,
                    metadata={"stage": "intermarket_dxy", "intermarket": intermarket_context},
                )
                return

        ai_context = news_context
        regime_summary = (
            f"REGIME_CONTEXT: regime={regime_context.get('regime')} reason={regime_context.get('reason')} "
            f"features={regime_context.get('features')}"
        )
        ai_context = f"{ai_context}\n{regime_summary}"
        if intermarket_context:
            ai_context = f"{news_context}\nINTERMARKET_CONTEXT: {intermarket_context.get('summary', '')}"
            ai_context = f"{ai_context}\n{regime_summary}"

        bypass_ai_for_tick_scalper = os.getenv("BYPASS_AI_FOR_TICK_SCALPER", "1") == "1"
        is_tick_fastpath = (str(tf).lower() == "tick") and (setup_type == "TICK_EMA_CROSS_SCALP")

        if bypass_ai_for_tick_scalper and is_tick_fastpath:
            decision = "APPROVED"
            reason = "Tick fast-path approval (AI bypass enabled)."
            ai_confidence = signal.get("confidence_score")
            ai_structured = {
                "primary_thesis": "Tick EMA momentum fast path",
                "top_3_risks": None,
                "invalidation_level": None,
                "expected_hold_time_minutes": 2,
                "suggested_size_multiplier_0_to_1": 1.0,
            }
            print("[ORCHESTRATOR] AI bypass active for tick scalper signal.")
        else:
            # 3. AI Final Veto (The Strategist)
            ai_result = await validate_with_ai({
                **signal,
                "context": ai_context,
                "intermarket_context": intermarket_context,
                "market_regime": regime_context.get("regime"),
                "regime_context": regime_context,
            }, macro_bias=trend) or {}

            if isinstance(ai_result, dict):
                decision = ai_result.get("decision", "REJECTED")
                reason = ai_result.get("reason", ai_result.get("reasoning", ""))
                ai_confidence = ai_result.get("confidence", signal.get("confidence_score"))
                ai_structured = {
                    "primary_thesis": ai_result.get("primary_thesis"),
                    "top_3_risks": ai_result.get("top_3_risks"),
                    "invalidation_level": ai_result.get("invalidation_level"),
                    "expected_hold_time_minutes": ai_result.get("expected_hold_time_minutes"),
                    "suggested_size_multiplier_0_to_1": ai_result.get("suggested_size_multiplier_0_to_1"),
                }
            else:
                decision, reason = ai_result
                ai_confidence = signal.get("confidence_score")

    confidence_score = float(signal.get("confidence_score") or 0.0)
    if confidence_score <= 70.0:
        reason = f"Confidence gate: {confidence_score:.1f} <= 70.0"
        print(reason)
        journal.log_signal(
            source=signal.get("source", "orchestrator"),
            symbol=symbol,
            action=action,
            timeframe=tf,
            signal_ts=_signal_ts(signal),
            **_indicators(signal),
            ai_decision="REJECTED",
            ai_reasoning=reason,
            ai_confidence=confidence_score,
            decision_status="REJECTED",
            rejection_reason=reason,
            metadata={"stage": "confidence_gate", "regime": regime_context},
        )
        return

    trend = get_integrated_bias(symbol)
    if trend == "NO_CONFLUENCE" or action != trend:
        reason = f"Higher-timeframe confluence not aligned: action={action}, bias={trend}"
        print(reason)
        journal.log_signal(
            source=signal.get("source", "orchestrator"),
            symbol=symbol,
            action=action,
            timeframe=tf,
            signal_ts=_signal_ts(signal),
            **_indicators(signal),
            ai_decision="REJECTED",
            ai_reasoning=reason,
            ai_confidence=confidence_score,
            decision_status="REJECTED",
            rejection_reason=reason,
            metadata={"stage": "mtf_confluence", "regime": regime_context, "bias": trend},
        )
        return

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
            "confidenceScore": confidence_score,
            "signal_timestamp": signal.get("timestamp"),
            "signal_bar_time": signal.get("signal_bar_time"),
            "signal_bar_relation": signal.get("signal_bar_relation"),
            "intended_price": signal.get("intended_price"),
        }

        zmq_message_id = None
        try:
            zmq_message_id = await publish_signal_async(payload)
            if zmq_message_id and zmq_message_id != "DISABLED":
                print(f"[ORCHESTRATOR] ZMQ signal published message_id={zmq_message_id}")
        except Exception as zmq_error:
            print(f"[ORCHESTRATOR] ZMQ publish skipped due to error: {zmq_error}")

        try:
            async with httpx.AsyncClient(timeout=5.0, follow_redirects=True) as client:
                response = await client.post(EXECUTION_ENGINE_URL, json=payload)

            if response.status_code == 200:
                entry_price = None
                sl = None
                tp = None
                order_ticket = None
                try:
                    result = response.json()
                    entry_price = result.get("price")
                    sl = result.get("sl")
                    tp = result.get("tp")
                    order_ticket = result.get("order")
                    if result.get("order") and entry_price is not None and sl is not None and tp is not None:
                        try:
                            track_active_trade(
                                result.get("order"),
                                entry_price,
                                sl,
                                tp,
                                symbol=symbol,
                                action=action,
                                timeframe=tf,
                                setup_type=signal.get("setup_type"),
                                opened_at=datetime.now(timezone.utc).isoformat(),
                            )
                        except Exception as track_error:
                            print(f"[ORCHESTRATOR] Track state skipped (Redis unavailable): {track_error}")
                except ValueError:
                    print("Execution engine response was not JSON; skipped trade tracking.")
                    result = {}

                journal.log_signal(
                    source=signal.get("source", "orchestrator"),
                    symbol=symbol,
                    action=action,
                    timeframe=tf,
                    signal_ts=_signal_ts(signal),
                    **_indicators(signal),
                    news_context=ai_context,
                    ai_decision=decision,
                    ai_reasoning=reason,
                    ai_confidence=ai_confidence,
                    decision_status="ACCEPTED",
                    rejection_reason=None,
                    is_filled=bool(order_ticket),
                    order_ticket=str(order_ticket) if order_ticket else None,
                    entry_price=entry_price,
                    stop_loss=sl,
                    take_profit=tp,
                    metadata={"execution_response": result, "intermarket": intermarket_context, "regime": regime_context, "ai_structured": ai_structured, "zmq_message_id": zmq_message_id},
                )
                print("TRADE_DISPATCHED")
            elif response.status_code == 403:
                print("SENTINEL_BLOCK")
                rejection_reason = "Execution engine blocked request (403)."
                send_telegram_alert(
                    "TRADE_ERROR",
                    "Execution engine blocked request.",
                    level="ERROR",
                    extra={"symbol": symbol, "action": action, "status_code": response.status_code},
                )
                journal.log_signal(
                    source=signal.get("source", "orchestrator"),
                    symbol=symbol,
                    action=action,
                    timeframe=tf,
                    signal_ts=_signal_ts(signal),
                    **_indicators(signal),
                    news_context=ai_context,
                    ai_decision=decision,
                    ai_reasoning=reason,
                    ai_confidence=ai_confidence,
                    decision_status="REJECTED",
                    rejection_reason=rejection_reason,
                    metadata={"status_code": response.status_code, "response": response.text, "intermarket": intermarket_context, "regime": regime_context, "ai_structured": ai_structured, "zmq_message_id": zmq_message_id},
                )
            else:
                print(f"Execution engine returned {response.status_code}: {response.text}")
                rejection_reason = f"Execution engine error {response.status_code}"
                send_telegram_alert(
                    "TRADE_ERROR",
                    "Execution engine returned non-success status.",
                    level="ERROR",
                    extra={"symbol": symbol, "action": action, "status_code": response.status_code},
                )
                journal.log_signal(
                    source=signal.get("source", "orchestrator"),
                    symbol=symbol,
                    action=action,
                    timeframe=tf,
                    signal_ts=_signal_ts(signal),
                    **_indicators(signal),
                    news_context=ai_context,
                    ai_decision=decision,
                    ai_reasoning=reason,
                    ai_confidence=ai_confidence,
                    decision_status="REJECTED",
                    rejection_reason=rejection_reason,
                    metadata={"status_code": response.status_code, "response": response.text, "intermarket": intermarket_context, "regime": regime_context, "ai_structured": ai_structured, "zmq_message_id": zmq_message_id},
                )
        except httpx.HTTPError as error:
            print(f"Execution dispatch failed: {error}")
            send_telegram_alert(
                "TRADE_ERROR",
                "Execution dispatch failed with HTTP error.",
                level="ERROR",
                extra={"symbol": symbol, "action": action, "error": str(error)},
            )
            journal.log_signal(
                source=signal.get("source", "orchestrator"),
                symbol=symbol,
                action=action,
                timeframe=tf,
                signal_ts=_signal_ts(signal),
                **_indicators(signal),
                news_context=ai_context,
                ai_decision=decision,
                ai_reasoning=reason,
                ai_confidence=ai_confidence,
                decision_status="REJECTED",
                rejection_reason=f"Execution dispatch failed: {error}",
                metadata={"exception": str(error), "intermarket": intermarket_context, "regime": regime_context, "ai_structured": ai_structured, "zmq_message_id": zmq_message_id},
            )
    else:
        print(f"Signal Vetoed by AI: {reason}")
        journal.log_signal(
            source=signal.get("source", "orchestrator"),
            symbol=symbol,
            action=action,
            timeframe=tf,
            signal_ts=_signal_ts(signal),
            **_indicators(signal),
            news_context=ai_context,
            ai_decision=decision,
            ai_reasoning=reason,
            ai_confidence=ai_confidence,
            decision_status="REJECTED",
            rejection_reason=reason,
            metadata={"stage": "ai_veto", "intermarket": intermarket_context, "regime": regime_context, "ai_structured": ai_structured},
        )

if __name__ == "__main__":
    # Example Test Signal
    test_signal = {"symbol": "XAUUSD", "timeframe": "5m", "action": "BUY"}
    asyncio.run(on_signal_received(test_signal))