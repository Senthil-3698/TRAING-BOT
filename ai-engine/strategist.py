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
    # Advanced 2026 Model with "Thinking" enabled
    model_id = os.getenv("GEMINI_MODEL_ID", "gemini-3-flash-preview")

    prompt = f"""
    You are an Institutional Strategist.
    Current 4H Trend is {macro_bias}.
    An incoming {signal_data['action']} signal has arrived on a lower timeframe.
    Your job is to approve this only if the macro context and the 4H trend provide high-probability confluence.
    If they conflict, be extremely critical of the entry.

    Institutional Trader Analysis:
    Asset: {signal_data['symbol']} | Timeframe: {signal_data['timeframe']} | Action: {signal_data['action']}
    Context: {signal_data.get('context', 'No macro news')}

    Task: Apply 'Sentinel' risk laws. Is this a high-probability institutional entry?
    Return JSON: {{"decision": "APPROVED/REJECTED", "reason": "1-sentence"}}
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
                return {
                    "decision": str(parsed_object.get("decision", "REJECTED")).upper(),
                    "reason": str(parsed_object.get("reason", parsed_object.get("reasoning", "No reasoning provided."))).strip(),
                }

            if isinstance(parsed_object, list) and parsed_object:
                first_item = parsed_object[0]
                if isinstance(first_item, dict):
                    return {
                        "decision": str(first_item.get("decision", "REJECTED")).upper(),
                        "reason": str(first_item.get("reason", first_item.get("reasoning", "No reasoning provided."))).strip(),
                    }
                if len(parsed_object) >= 2:
                    return {
                        "decision": str(parsed_object[0]).upper(),
                        "reason": str(parsed_object[1]).strip(),
                    }

        parsed = json.loads(raw_text)
        if isinstance(parsed, list) and parsed:
            first_item = parsed[0]
            if isinstance(first_item, dict):
                parsed = first_item
            elif len(parsed) >= 2:
                return {
                    "decision": str(parsed[0]).upper(),
                    "reason": str(parsed[1]).strip(),
                }

        if not isinstance(parsed, dict):
            raise ValueError(f"Unexpected JSON response type: {type(parsed).__name__}")

        return {
            "decision": str(parsed.get("decision", "REJECTED")).upper(),
            "reason": str(parsed.get("reason", parsed.get("reasoning", "No reasoning provided."))).strip(),
        }
    except Exception as error:
        symbol = str(signal_data.get("symbol", "")).upper()
        action = str(signal_data.get("action", "")).upper()
        news_context = str(signal_data.get("context", "")).upper()
        trend = str(macro_bias).upper()

        if any(keyword in news_context for keyword in ("CPI", "NFP")):
            decision = "REJECTED"
            reason = "High-impact news is active, so the setup is blocked until the macro surprise is known."
        elif trend in {"BULLISH", "BEARISH"} and ((trend == "BULLISH" and action == "SELL") or (trend == "BEARISH" and action == "BUY")):
            decision = "REJECTED"
            reason = "The trade conflicts with the higher-timeframe trend."
        elif symbol == "XAUUSD" and "LIQUIDITY" in news_context:
            decision = "APPROVED"
            reason = "Liquidity conditions favor the institutional sweep and the signal is aligned."
        else:
            decision = "APPROVED"
            reason = "Macro structure and liquidity do not violate the Sentinel rules."

        return {
            "decision": decision,
            "reason": f"{reason} Fallback used because Gemini request failed: {error}",
        }

def process_queue():
    while True:
        # Pulls the next signal from Redis to process it
        # This ensures we don't miss any alerts
        pass