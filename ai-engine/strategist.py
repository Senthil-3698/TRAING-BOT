import os
import json
from datetime import datetime, timedelta, timezone
from typing import Any

import psycopg
from dotenv import load_dotenv
from google import genai

load_dotenv()

_LLM_API_KEY = os.getenv("LLM_API_KEY")
if not _LLM_API_KEY:
    raise RuntimeError("LLM_API_KEY is required for strategist.py")

client = genai.Client(api_key=_LLM_API_KEY)


def _resolve_model_id() -> str:
    configured = os.getenv("GEMINI_MODEL_ID")
    if configured is None:
        return "gemini-2.0-flash-exp"
    model_id = configured.strip()
    if not model_id:
        raise RuntimeError("GEMINI_MODEL_ID is set but empty. Provide a valid model id (e.g., gemini-2.0-flash-exp).")
    return model_id


def _db_config() -> dict[str, Any]:
    return {
        "host": os.getenv("JOURNAL_DB_HOST", os.getenv("MONITOR_DB_HOST", "localhost")),
        "port": int(os.getenv("JOURNAL_DB_PORT", os.getenv("MONITOR_DB_PORT", "5433"))),
        "dbname": os.getenv("JOURNAL_DB_NAME", os.getenv("MONITOR_DB_NAME", "sentinel_db")),
        "user": os.getenv("JOURNAL_DB_USER", os.getenv("MONITOR_DB_USER", "admin")),
        "password": os.getenv("JOURNAL_DB_PASSWORD", os.getenv("MONITOR_DB_PASSWORD", "admin")),
    }


def _ensure_prompt_log_table() -> None:
    ddl = """
    CREATE TABLE IF NOT EXISTS ai_prompt_log (
        id BIGSERIAL PRIMARY KEY,
        created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
        model_id VARCHAR(128) NOT NULL,
        symbol VARCHAR(20),
        action VARCHAR(10),
        timeframe VARCHAR(10),
        prompt_text TEXT NOT NULL,
        response_text TEXT,
        parsed_response JSONB,
        error_text TEXT
    );
    """
    with psycopg.connect(**_db_config()) as connection:
        with connection.cursor() as cursor:
            cursor.execute(ddl)
        connection.commit()


def _safe_json(value: Any) -> str:
    return json.dumps(value, default=str, ensure_ascii=True)


def _fetch_recent_performance_context(symbol: str) -> dict[str, Any]:
    now = datetime.now(timezone.utc)
    day_start = datetime(now.year, now.month, now.day, tzinfo=timezone.utc)

    context: dict[str, Any] = {
        "last_10_outcomes": [],
        "current_daily_pnl": 0.0,
        "drawdown_state": {
            "max_intraday_drawdown_usd": 0.0,
            "current_drawdown_from_peak_usd": 0.0,
            "state": "FLAT",
        },
        "working_setups_today": [],
    }

    with psycopg.connect(**_db_config()) as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT signal_ts, action, timeframe, integrated_bias, m5_trend, h1_bias, h4_bias,
                       entry_price, exit_price, pnl_usd, pnl_r, exit_reason
                FROM signal_journal
                WHERE symbol = %s
                  AND decision_status = 'ACCEPTED'
                  AND is_filled = TRUE
                  AND pnl_usd IS NOT NULL
                ORDER BY COALESCE(signal_ts, journal_ts) DESC
                LIMIT 10
                """,
                (symbol,),
            )
            rows = cursor.fetchall()

            last_10 = []
            for row in rows:
                ts, action, timeframe, integrated_bias, m5_trend, h1_bias, h4_bias, entry_price, exit_price, pnl_usd, pnl_r, exit_reason = row
                last_10.append(
                    {
                        "ts": ts.isoformat() if ts else None,
                        "action": action,
                        "timeframe": timeframe,
                        "integrated_bias": integrated_bias,
                        "m5_trend": m5_trend,
                        "h1_bias": h1_bias,
                        "h4_bias": h4_bias,
                        "entry_price": float(entry_price) if entry_price is not None else None,
                        "exit_price": float(exit_price) if exit_price is not None else None,
                        "pnl_usd": float(pnl_usd) if pnl_usd is not None else None,
                        "pnl_r": float(pnl_r) if pnl_r is not None else None,
                        "exit_reason": exit_reason,
                    }
                )
            context["last_10_outcomes"] = last_10

            cursor.execute(
                """
                SELECT COALESCE(SUM(pnl_usd), 0)
                FROM signal_journal
                WHERE symbol = %s
                  AND decision_status = 'ACCEPTED'
                  AND is_filled = TRUE
                  AND pnl_usd IS NOT NULL
                  AND COALESCE(signal_ts, journal_ts) >= %s
                """,
                (symbol, day_start),
            )
            daily_row = cursor.fetchone()
            daily_pnl = float(daily_row[0] if daily_row and daily_row[0] is not None else 0.0)
            context["current_daily_pnl"] = daily_pnl

            cursor.execute(
                """
                SELECT COALESCE(signal_ts, journal_ts) AS ts, pnl_usd
                FROM signal_journal
                WHERE symbol = %s
                  AND decision_status = 'ACCEPTED'
                  AND is_filled = TRUE
                  AND pnl_usd IS NOT NULL
                  AND COALESCE(signal_ts, journal_ts) >= %s
                ORDER BY ts ASC
                """,
                (symbol, day_start),
            )
            pnl_rows = cursor.fetchall()

            equity = 0.0
            peak = 0.0
            max_dd = 0.0
            for _, pnl in pnl_rows:
                equity += float(pnl)
                if equity > peak:
                    peak = equity
                dd = peak - equity
                if dd > max_dd:
                    max_dd = dd

            current_dd = max(0.0, peak - equity)
            if current_dd > 150:
                state = "HIGH_DRAWDOWN"
            elif current_dd > 50:
                state = "MODERATE_DRAWDOWN"
            elif equity > 0:
                state = "RECOVERING_OR_GREEN"
            else:
                state = "FLAT"

            context["drawdown_state"] = {
                "max_intraday_drawdown_usd": max_dd,
                "current_drawdown_from_peak_usd": current_dd,
                "state": state,
            }

            cursor.execute(
                """
                SELECT
                    action,
                    COALESCE(timeframe, 'NA') AS timeframe,
                    COALESCE(integrated_bias, 'NA') AS integrated_bias,
                    COUNT(*) AS trades,
                    AVG(CASE WHEN pnl_usd > 0 THEN 1.0 ELSE 0.0 END) AS win_rate,
                    AVG(pnl_usd) AS avg_pnl
                FROM signal_journal
                WHERE symbol = %s
                  AND decision_status = 'ACCEPTED'
                  AND is_filled = TRUE
                  AND pnl_usd IS NOT NULL
                  AND COALESCE(signal_ts, journal_ts) >= %s
                GROUP BY action, COALESCE(timeframe, 'NA'), COALESCE(integrated_bias, 'NA')
                HAVING COUNT(*) >= 2
                ORDER BY avg_pnl DESC, win_rate DESC
                LIMIT 5
                """,
                (symbol, day_start),
            )
            setup_rows = cursor.fetchall()

            setups = []
            for action, timeframe, integrated_bias, trades, win_rate, avg_pnl in setup_rows:
                if float(avg_pnl or 0.0) <= 0:
                    continue
                setups.append(
                    {
                        "setup": f"{action}|{timeframe}|{integrated_bias}",
                        "trades": int(trades),
                        "win_rate": round(float(win_rate or 0.0), 3),
                        "avg_pnl": round(float(avg_pnl or 0.0), 2),
                    }
                )
            context["working_setups_today"] = setups

    return context


def _fetch_few_shot_examples(symbol: str) -> dict[str, list[dict[str, Any]]]:
    since = datetime.now(timezone.utc) - timedelta(days=30)
    wins: list[dict[str, Any]] = []
    losses: list[dict[str, Any]] = []

    with psycopg.connect(**_db_config()) as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT signal_ts, symbol, action, timeframe,
                       integrated_bias, m5_trend, h1_bias, h4_bias,
                       rsi_value, ema_distance, atr_value,
                       news_context, ai_reasoning, pnl_usd, pnl_r, metadata
                FROM signal_journal
                WHERE symbol = %s
                  AND decision_status = 'ACCEPTED'
                  AND is_filled = TRUE
                  AND pnl_usd > 0
                  AND COALESCE(signal_ts, journal_ts) >= %s
                ORDER BY pnl_usd DESC, COALESCE(signal_ts, journal_ts) DESC
                LIMIT 3
                """,
                (symbol, since),
            )
            for row in cursor.fetchall():
                (
                    signal_ts,
                    ex_symbol,
                    action,
                    timeframe,
                    integrated_bias,
                    m5_trend,
                    h1_bias,
                    h4_bias,
                    rsi_value,
                    ema_distance,
                    atr_value,
                    news_context,
                    ai_reasoning,
                    pnl_usd,
                    pnl_r,
                    metadata,
                ) = row
                wins.append(
                    {
                        "signal_ts": signal_ts.isoformat() if signal_ts else None,
                        "symbol": ex_symbol,
                        "action": action,
                        "timeframe": timeframe,
                        "bias": {
                            "integrated": integrated_bias,
                            "m5": m5_trend,
                            "h1": h1_bias,
                            "h4": h4_bias,
                        },
                        "indicators": {
                            "rsi": float(rsi_value) if rsi_value is not None else None,
                            "ema_distance": float(ema_distance) if ema_distance is not None else None,
                            "atr": float(atr_value) if atr_value is not None else None,
                        },
                        "news_context": news_context,
                        "ai_reasoning": ai_reasoning,
                        "metadata": metadata,
                        "outcome": {"pnl_usd": float(pnl_usd), "pnl_r": float(pnl_r) if pnl_r is not None else None, "label": "WIN"},
                    }
                )

            cursor.execute(
                """
                SELECT signal_ts, symbol, action, timeframe,
                       integrated_bias, m5_trend, h1_bias, h4_bias,
                       rsi_value, ema_distance, atr_value,
                       news_context, ai_reasoning, pnl_usd, pnl_r, metadata
                FROM signal_journal
                WHERE symbol = %s
                  AND decision_status = 'ACCEPTED'
                  AND is_filled = TRUE
                  AND pnl_usd < 0
                  AND COALESCE(signal_ts, journal_ts) >= %s
                ORDER BY pnl_usd ASC, COALESCE(signal_ts, journal_ts) DESC
                LIMIT 3
                """,
                (symbol, since),
            )
            for row in cursor.fetchall():
                (
                    signal_ts,
                    ex_symbol,
                    action,
                    timeframe,
                    integrated_bias,
                    m5_trend,
                    h1_bias,
                    h4_bias,
                    rsi_value,
                    ema_distance,
                    atr_value,
                    news_context,
                    ai_reasoning,
                    pnl_usd,
                    pnl_r,
                    metadata,
                ) = row
                losses.append(
                    {
                        "signal_ts": signal_ts.isoformat() if signal_ts else None,
                        "symbol": ex_symbol,
                        "action": action,
                        "timeframe": timeframe,
                        "bias": {
                            "integrated": integrated_bias,
                            "m5": m5_trend,
                            "h1": h1_bias,
                            "h4": h4_bias,
                        },
                        "indicators": {
                            "rsi": float(rsi_value) if rsi_value is not None else None,
                            "ema_distance": float(ema_distance) if ema_distance is not None else None,
                            "atr": float(atr_value) if atr_value is not None else None,
                        },
                        "news_context": news_context,
                        "ai_reasoning": ai_reasoning,
                        "metadata": metadata,
                        "outcome": {"pnl_usd": float(pnl_usd), "pnl_r": float(pnl_r) if pnl_r is not None else None, "label": "LOSS"},
                    }
                )

    return {"winning_examples": wins, "losing_examples": losses}


def _build_prompt(signal_data: dict[str, Any], macro_bias: str, performance: dict[str, Any], examples: dict[str, Any]) -> str:
    return f"""
You are an Institutional Strategist making a deterministic risk decision for a live signal.

CURRENT SIGNAL
{_safe_json(signal_data)}

MACRO/HIGHER TIMEFRAME BIAS
{_safe_json({"macro_bias": macro_bias})}

RECENT PERFORMANCE CONTEXT (from signal journal)
{_safe_json(performance)}

FEW-SHOT HISTORICAL REFERENCE SETUPS
Winning setups (3):
{_safe_json(examples.get("winning_examples", []))}

Losing setups (3):
{_safe_json(examples.get("losing_examples", []))}

REQUIRED OUTPUT JSON ONLY (no markdown, no prose outside JSON):
{{
  "decision": "APPROVED" | "REJECTED",
  "confidence": <float 0..1>,
  "primary_thesis": "<string>",
  "top_3_risks": ["<risk1>", "<risk2>", "<risk3>"],
  "invalidation_level": "<price level or condition>",
  "expected_hold_time_minutes": <integer>,
  "suggested_size_multiplier_0_to_1": <float 0..1>
}}

POLICY CONSTRAINTS
1) Be strict with conflict between direction and bias/performance regime.
2) Use recent outcomes and drawdown state to reduce risk when edge is unstable.
3) If context is contradictory, prefer REJECTED and lower size multiplier.
4) Always populate exactly 3 top risks.
5) Regime policy: REJECT momentum crossover scalps in RANGING regime unless very strong breakout evidence appears in context.
""".strip()


def _extract_setup_type(signal_data: dict[str, Any]) -> str:
    direct = str(signal_data.get("setup_type") or "").strip()
    if direct:
        return direct.upper()

    indicators = signal_data.get("indicators") or {}
    from_indicators = str(indicators.get("setup_type") or "").strip()
    if from_indicators:
        return from_indicators.upper()

    context_blob = json.dumps(signal_data, default=str).upper()
    if "SMA5" in context_blob and "SMA13" in context_blob:
        return "SMA_CROSS_SCALP"
    if "EMA_CROSS" in context_blob:
        return "EMA_CROSS_SCALP"
    return "UNKNOWN"


def _regime_pre_veto(signal_data: dict[str, Any]) -> dict[str, Any] | None:
    regime = str(signal_data.get("market_regime") or "UNKNOWN").upper()
    setup_type = _extract_setup_type(signal_data)

    disallowed_setups = {"SMA_CROSS_SCALP", "EMA_CROSS_SCALP", "MA_CROSS_SCALP"}
    if regime == "RANGING" and setup_type in disallowed_setups:
        veto = {
            "decision": "REJECTED",
            "confidence": 0.95,
            "primary_thesis": f"Regime veto: {setup_type} is disallowed in {regime} due to persistent chop failure profile.",
            "top_3_risks": [
                "Whipsaw risk is elevated in ranging conditions",
                "Momentum edge decays in low directional persistence",
                "False break frequency is high in mean-reverting tape",
            ],
            "invalidation_level": "No trade unless breakout regime is confirmed",
            "expected_hold_time_minutes": 10,
            "suggested_size_multiplier_0_to_1": 0.0,
        }
        veto["reason"] = veto["primary_thesis"]
        return veto

    return None


def _normalize_structured_response(raw: str, parsed_object: Any) -> dict[str, Any]:
    payload: Any = None
    if raw:
        payload = json.loads(raw)
    elif parsed_object is not None:
        payload = parsed_object

    if isinstance(payload, list) and payload:
        payload = payload[0]

    if not isinstance(payload, dict):
        raise ValueError("Model response is not a JSON object.")

    top_risks = payload.get("top_3_risks")
    if not isinstance(top_risks, list):
        top_risks = []
    top_risks = [str(x).strip() for x in top_risks if str(x).strip()][:3]
    while len(top_risks) < 3:
        top_risks.append("Unspecified risk")

    decision = str(payload.get("decision", "REJECTED")).upper()
    if decision not in {"APPROVED", "REJECTED"}:
        decision = "REJECTED"

    confidence = float(payload.get("confidence", 0.0) or 0.0)
    confidence = max(0.0, min(1.0, confidence))

    hold_minutes = int(payload.get("expected_hold_time_minutes", 30) or 30)
    if hold_minutes < 1:
        hold_minutes = 1

    size_multiplier = float(payload.get("suggested_size_multiplier_0_to_1", 0.0) or 0.0)
    size_multiplier = max(0.0, min(1.0, size_multiplier))

    structured = {
        "decision": decision,
        "confidence": confidence,
        "primary_thesis": str(payload.get("primary_thesis", "No thesis provided.")).strip(),
        "top_3_risks": top_risks,
        "invalidation_level": str(payload.get("invalidation_level", "Not specified")).strip(),
        "expected_hold_time_minutes": hold_minutes,
        "suggested_size_multiplier_0_to_1": size_multiplier,
    }
    structured["reason"] = structured["primary_thesis"]
    return structured


def _log_prompt_response(
    *,
    model_id: str,
    signal_data: dict[str, Any],
    prompt_text: str,
    response_text: str | None,
    parsed_response: dict[str, Any] | None,
    error_text: str | None,
) -> None:
    try:
        _ensure_prompt_log_table()
        with psycopg.connect(**_db_config()) as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    INSERT INTO ai_prompt_log (
                        model_id, symbol, action, timeframe, prompt_text, response_text, parsed_response, error_text
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                    """,
                    (
                        model_id,
                        signal_data.get("symbol"),
                        signal_data.get("action"),
                        signal_data.get("timeframe"),
                        prompt_text,
                        response_text,
                        psycopg.types.json.Jsonb(parsed_response) if parsed_response is not None else None,
                        error_text,
                    ),
                )
            connection.commit()
    except Exception as error:
        print(f"[STRATEGIST] Failed to persist prompt log: {error}")


async def validate_with_ai(signal_data, macro_bias):
    model_id = _resolve_model_id()
    performance_context: dict[str, Any] = {}
    few_shot_examples: dict[str, Any] = {"winning_examples": [], "losing_examples": []}

    symbol = str(signal_data.get("symbol", "XAUUSD")).upper()

    try:
        performance_context = _fetch_recent_performance_context(symbol)
        few_shot_examples = _fetch_few_shot_examples(symbol)
    except Exception as context_error:
        performance_context = {
            "last_10_outcomes": [],
            "current_daily_pnl": 0.0,
            "drawdown_state": {"max_intraday_drawdown_usd": 0.0, "current_drawdown_from_peak_usd": 0.0, "state": "UNKNOWN"},
            "working_setups_today": [],
            "context_error": str(context_error),
        }

    enriched_signal = dict(signal_data)
    enriched_signal["performance_context"] = performance_context
    enriched_signal["few_shot_examples"] = few_shot_examples

    regime_veto = _regime_pre_veto(enriched_signal)
    if regime_veto is not None:
        _log_prompt_response(
            model_id=model_id,
            signal_data=signal_data,
            prompt_text="REGIME_PRE_VETO",
            response_text=None,
            parsed_response=regime_veto,
            error_text="regime_pre_veto",
        )
        return regime_veto

    prompt = _build_prompt(enriched_signal, str(macro_bias), performance_context, few_shot_examples)

    try:
        response = client.models.generate_content(
            model=model_id,
            contents=prompt,
            config={
                "response_mime_type": "application/json",
                "temperature": 0,
            },
        )
        raw_text = (getattr(response, "text", "") or "").strip()
        parsed_object = getattr(response, "parsed", None)
        structured = _normalize_structured_response(raw_text, parsed_object)
        _log_prompt_response(
            model_id=model_id,
            signal_data=signal_data,
            prompt_text=prompt,
            response_text=raw_text,
            parsed_response=structured,
            error_text=None,
        )
        return structured
    except Exception as error:
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

        fallback = {
            "decision": decision,
            "confidence": 0.2,
            "primary_thesis": f"{reason} Fallback used because Gemini request failed: {error}",
            "top_3_risks": [
                "LLM unavailable or malformed response",
                "Context may be stale",
                "Decision quality reduced under fallback",
            ],
            "invalidation_level": "Do not execute without manual confirmation",
            "expected_hold_time_minutes": 15,
            "suggested_size_multiplier_0_to_1": 0.1,
        }
        fallback["reason"] = fallback["primary_thesis"]

        _log_prompt_response(
            model_id=model_id,
            signal_data=signal_data,
            prompt_text=prompt,
            response_text=None,
            parsed_response=fallback,
            error_text=str(error),
        )
        return fallback

def process_queue():
    while True:
        # Pulls the next signal from Redis to process it
        # This ensures we don't miss any alerts
        pass