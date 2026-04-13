import os
import json

import redis
from dotenv import load_dotenv
from google import genai

load_dotenv()

client = genai.Client(api_key=os.getenv("LLM_API_KEY"))

r = redis.Redis(
    host=os.getenv("REDIS_HOST", "localhost"),
    port=int(os.getenv("REDIS_PORT", "6380")),
    db=0,
)


async def validate_with_ai(signal_data, macro_bias):
    model_id = os.getenv("GEMINI_MODEL_ID", "gemini-3-flash-preview")

    rsi = signal_data.get("rsi", "N/A")
    ema_dist = signal_data.get("ema_distance_pct", "N/A")
    atr = signal_data.get("atr_m5", "N/A")
    action = signal_data.get("action", "")

    # Determine RSI context for the prompt
    if isinstance(rsi, float):
        if action == "BUY" and rsi > 70:
            rsi_note = "WARNING: RSI is overbought — BUY into momentum exhaustion."
        elif action == "SELL" and rsi < 30:
            rsi_note = "WARNING: RSI is oversold — SELL into momentum exhaustion."
        elif action == "BUY" and rsi < 50:
            rsi_note = "RSI has room to the upside — favorable for BUY."
        elif action == "SELL" and rsi > 50:
            rsi_note = "RSI has room to the downside — favorable for SELL."
        else:
            rsi_note = "RSI is neutral."
    else:
        rsi_note = "RSI data unavailable."

    prompt = f"""
You are a senior institutional FX/Gold strategist. Your role is to approve or reject a scalp entry with extreme discipline.

== SIGNAL DETAILS ==
Asset: {signal_data['symbol']}
Timeframe: {signal_data['timeframe']}
Direction: {action}
Higher-TF Bias: {macro_bias}

== TECHNICAL SNAPSHOT ==
RSI(14) on M5: {rsi} — {rsi_note}
EMA50 Distance (M5): {ema_dist}% (positive = price above EMA)
ATR(14) on M5: {atr}
Macro/News Context: {signal_data.get('context', 'No macro context available')}

== YOUR DECISION CRITERIA ==
APPROVE only when ALL of the following hold:
1. Higher-TF bias is aligned or absent (NO_CONFLUENCE is acceptable but reduces confidence).
2. RSI is NOT in an extreme zone against the trade direction (RSI > 70 for BUY = reject, RSI < 30 for SELL = reject).
3. Price is not consolidating against the trend (EMA distance > 0.01%).
4. Macro context does NOT contain a known high-impact event (CPI, NFP, FOMC, rate decision).
5. ATR confirms market is moving (not dead/flat — ATR should be > 0).

REJECT if any criterion fails.

Return ONLY valid JSON (no markdown, no extra text):
{{"decision": "APPROVED" or "REJECTED", "reason": "one concise sentence", "confidence": <integer 0-100>}}
"""

    try:
        response = client.models.generate_content(
            model=model_id,
            contents=prompt,
            config={"response_mime_type": "application/json"},
        )
        raw_text = (getattr(response, "text", "") or "").strip()
        parsed_object = getattr(response, "parsed", None)

        if not raw_text and parsed_object is not None:
            if isinstance(parsed_object, dict):
                return _normalize(parsed_object)
            if isinstance(parsed_object, list) and parsed_object:
                return _normalize(parsed_object[0] if isinstance(parsed_object[0], dict) else {})

        parsed = json.loads(raw_text)
        if isinstance(parsed, list) and parsed:
            parsed = parsed[0] if isinstance(parsed[0], dict) else {}
        if not isinstance(parsed, dict):
            raise ValueError(f"Unexpected type: {type(parsed)}")
        return _normalize(parsed)

    except Exception as error:
        return _fallback(signal_data, macro_bias, error)


def _normalize(parsed):
    return {
        "decision": str(parsed.get("decision", "REJECTED")).upper(),
        "reason": str(parsed.get("reason", parsed.get("reasoning", "No reasoning provided."))).strip(),
        "confidence": int(parsed.get("confidence", 50)),
    }


def _fallback(signal_data, macro_bias, error):
    """
    Rule-based fallback when Gemini is unavailable.
    Conservative: requires trend alignment; blocks high-impact news.
    """
    symbol = str(signal_data.get("symbol", "")).upper()
    action = str(signal_data.get("action", "")).upper()
    news_context = str(signal_data.get("context", "")).upper()
    trend = str(macro_bias).upper()
    rsi = signal_data.get("rsi")

    HIGH_IMPACT = ("CPI", "NFP", "FOMC", "RATE DECISION", "NONFARM")
    if any(kw in news_context for kw in HIGH_IMPACT):
        return {
            "decision": "REJECTED",
            "reason": "High-impact news event is active. Trade blocked until macro outcome is known.",
            "confidence": 10,
        }

    if trend in ("BULLISH", "BEARISH"):
        if (trend == "BULLISH" and action == "SELL") or (trend == "BEARISH" and action == "BUY"):
            return {
                "decision": "REJECTED",
                "reason": "Trade direction conflicts with higher-timeframe trend.",
                "confidence": 20,
            }

    if isinstance(rsi, (int, float)):
        if action == "BUY" and rsi > 70:
            return {
                "decision": "REJECTED",
                "reason": "RSI overbought — momentum exhaustion risk for BUY entry.",
                "confidence": 25,
            }
        if action == "SELL" and rsi < 30:
            return {
                "decision": "REJECTED",
                "reason": "RSI oversold — momentum exhaustion risk for SELL entry.",
                "confidence": 25,
            }

    return {
        "decision": "APPROVED",
        "reason": f"Fallback rules passed. Gemini unavailable: {error}",
        "confidence": 55,
    }


def process_queue():
    while True:
        pass
