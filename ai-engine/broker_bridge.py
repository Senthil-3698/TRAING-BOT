import os
import time
from datetime import datetime, timezone
from typing import Literal

import MetaTrader5 as mt5
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from risk_engine import RiskEngine
from execution_quality import ExecutionQualityMonitor
from alerts import send_telegram_alert
from execution_stress_wrapper import ExecutionStressWrapper
from position_sizing import calculate_xauusd_lot_size, check_margin_sufficiency

load_dotenv()

app = FastAPI(title="MT5 Broker Bridge", version="1.0.0")
risk_engine = RiskEngine()
quality_monitor = ExecutionQualityMonitor()
stress_wrapper = ExecutionStressWrapper()
DEFAULT_SLIPPAGE_TOLERANCE_PIPS = float(os.getenv("SLIPPAGE_TOLERANCE_PIPS", "2.0"))
MAX_ORDER_RETRIES = int(os.getenv("ORDER_MAX_RETRIES", "3"))
RETRY_BASE_BACKOFF_MS = int(os.getenv("ORDER_RETRY_BASE_BACKOFF_MS", "200"))
LATENCY_ALERT_MS = float(os.getenv("EXECUTION_LATENCY_ALERT_MS", "500"))
STRICT_RISK_PERCENT_DEFAULT = float(os.getenv("STRICT_RISK_PERCENT", "1.0"))
ENABLE_DYNAMIC_XAUUSD_SIZING = os.getenv("ENABLE_DYNAMIC_XAUUSD_SIZING", "1") == "1"
MIN_POST_TRADE_FREE_MARGIN_PCT = float(os.getenv("MIN_POST_TRADE_FREE_MARGIN_PCT", "0.10"))


class LiveTradeRequest(BaseModel):
    symbol: str
    action: Literal["BUY", "SELL"]
    timeframe: str
    volume: float = Field(default=0.01, gt=0)
    stop_loss_pips: float | None = Field(default=None, gt=0)
    take_profit_pips: float | None = Field(default=None, gt=0)
    confidence_score: float | None = None
    signal_timestamp: float | str | None = None
    signal_bar_time: float | str | None = None
    signal_bar_relation: str | None = None
    intended_price: float | None = None
    slippage_tolerance_pips: float | None = Field(default=None, gt=0)
    strict_risk_percent: float | None = Field(default=None, gt=0, le=100)


def _pip_size(symbol: str) -> float:
    normalized = symbol.upper()
    if normalized == "XAUUSD":
        return 0.1
    if normalized.endswith("JPY"):
        return 0.01
    return 0.0001


def _atr_stop_loss_pips(symbol: str, period: int = 14, multiplier: float = 1.5) -> float:
    rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M5, 0, period + 1)
    if rates is None or len(rates) < period + 1:
        return 50.0

    high = rates["high"][1:]
    low = rates["low"][1:]
    prev_close = rates["close"][:-1]
    true_ranges = []
    for h, l, pc in zip(high, low, prev_close):
        true_ranges.append(max(h - l, abs(h - pc), abs(l - pc)))

    atr = sum(true_ranges[-period:]) / period
    pip_size = _pip_size(symbol)
    if pip_size <= 0:
        return 50.0

    atr_pips = atr / pip_size
    return max(1.0, atr_pips * multiplier)


def _connect_mt5() -> None:
    login = os.getenv("MT5_LOGIN")
    password = os.getenv("MT5_PASSWORD") or os.getenv("MT5_PASS")
    server = os.getenv("MT5_SERVER")
    path = os.getenv("MT5_PATH")

    mt5_path = None
    if path and os.path.exists(path):
        mt5_path = path

    init_kwargs = {
        "login": int(login) if login else None,
        "password": password,
        "server": server,
    }
    if mt5_path:
        initialized = mt5.initialize(path=mt5_path, **init_kwargs)
    else:
        initialized = mt5.initialize(**init_kwargs)

    if not initialized:
        raise HTTPException(status_code=503, detail=f"MT5 initialize failed: {mt5.last_error()}")


def _compute_atr_price(symbol: str, period: int = 14, timeframe=mt5.TIMEFRAME_M5) -> float:
    rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, period + 1)
    if rates is None or len(rates) < period + 1:
        return 0.0

    high = rates["high"][1:]
    low = rates["low"][1:]
    prev_close = rates["close"][:-1]
    tr = []
    for h, l, pc in zip(high, low, prev_close):
        tr.append(max(h - l, abs(h - pc), abs(l - pc)))
    return float(sum(tr[-period:]) / period) if tr else 0.0


def _is_retriable_retcode(retcode: int | None) -> bool:
    retriable_codes = {
        getattr(mt5, "TRADE_RETCODE_REQUOTE", -1),
        getattr(mt5, "TRADE_RETCODE_PRICE_CHANGED", -1),
        getattr(mt5, "TRADE_RETCODE_PRICE_OFF", -1),
        getattr(mt5, "TRADE_RETCODE_TIMEOUT", -1),
        getattr(mt5, "TRADE_RETCODE_CONNECTION", -1),
        getattr(mt5, "TRADE_RETCODE_TOO_MANY_REQUESTS", -1),
    }
    return retcode in retriable_codes


def _send_with_retry(request_payload: dict) -> tuple[object | None, int, datetime, datetime]:
    attempts = 0
    final_result = None
    send_ts = datetime.now(timezone.utc)
    done_ts = send_ts

    for attempt in range(1, MAX_ORDER_RETRIES + 1):
        attempts = attempt
        send_ts = datetime.now(timezone.utc)
        result = mt5.order_send(request_payload)
        done_ts = datetime.now(timezone.utc)
        final_result = result

        if result is not None and getattr(result, "retcode", None) == mt5.TRADE_RETCODE_DONE:
            return result, attempts, send_ts, done_ts

        retcode = getattr(result, "retcode", None) if result is not None else None
        if attempt >= MAX_ORDER_RETRIES or not _is_retriable_retcode(retcode):
            return final_result, attempts, send_ts, done_ts

        backoff_ms = RETRY_BASE_BACKOFF_MS * (2 ** (attempt - 1))
        print(f"[EXECUTION_RETRY] attempt={attempt} retcode={retcode} backoff_ms={backoff_ms}")
        time.sleep(backoff_ms / 1000.0)

    return final_result, attempts, send_ts, done_ts


@app.post("/execute")
def execute_trade(request: LiveTradeRequest):
    _connect_mt5()

    decision = risk_engine.pre_trade_check(
        symbol=request.symbol,
        action=request.action,
        timeframe=request.timeframe,
        source="broker_bridge",
        purpose="OPEN",
    )
    if not decision.allowed:
        mt5.shutdown()
        send_telegram_alert(
            "TRADE_ERROR",
            "Risk engine rejected trade request.",
            level="ERROR",
            extra={"symbol": request.symbol, "action": request.action, "risk_code": decision.code},
        )
        raise HTTPException(
            status_code=403,
            detail={
                "risk_code": decision.code,
                "risk_message": decision.message,
                "details": decision.details,
            },
        )

    symbol_info = mt5.symbol_info(request.symbol)
    if symbol_info is None:
        mt5.shutdown()
        send_telegram_alert(
            "TRADE_ERROR",
            "Unknown symbol in broker bridge.",
            level="ERROR",
            extra={"symbol": request.symbol, "action": request.action},
        )
        raise HTTPException(status_code=400, detail=f"Unknown symbol: {request.symbol}")

    if not symbol_info.visible and not mt5.symbol_select(request.symbol, True):
        mt5.shutdown()
        send_telegram_alert(
            "TRADE_ERROR",
            "Unable to select symbol in MT5.",
            level="ERROR",
            extra={"symbol": request.symbol, "action": request.action},
        )
        raise HTTPException(status_code=400, detail=f"Unable to select symbol: {request.symbol}")

    tick = mt5.symbol_info_tick(request.symbol)
    if tick is None:
        mt5.shutdown()
        send_telegram_alert(
            "TRADE_ERROR",
            "No tick data available for symbol.",
            level="ERROR",
            extra={"symbol": request.symbol, "action": request.action},
        )
        raise HTTPException(status_code=503, detail=f"No tick data for {request.symbol}")

    pip_size = _pip_size(request.symbol)
    # For live-fire validation and tight HFT loops, respect upstream requested size.
    min_vol = symbol_info.volume_min or 0.01
    max_vol = symbol_info.volume_max or request.volume
    step = symbol_info.volume_step or 0.01
    clamped = max(min_vol, min(request.volume, max_vol))
    volume = round(round(clamped / step) * step, 2)

    is_buy = request.action.upper() == "BUY"
    price = tick.ask if is_buy else tick.bid
    intended_price = request.intended_price if request.intended_price is not None else price
    slippage_tolerance_pips = (
        request.slippage_tolerance_pips if request.slippage_tolerance_pips is not None else DEFAULT_SLIPPAGE_TOLERANCE_PIPS
    )
    quote_deviation_pips = abs(price - intended_price) / pip_size if pip_size > 0 else 0.0
    if quote_deviation_pips > slippage_tolerance_pips:
        mt5.shutdown()
        send_telegram_alert(
            "TRADE_ERROR",
            "Slippage tolerance exceeded before order send.",
            level="ERROR",
            extra={
                "symbol": request.symbol,
                "action": request.action,
                "deviation_pips": round(quote_deviation_pips, 4),
                "tolerance_pips": slippage_tolerance_pips,
            },
        )
        raise HTTPException(
            status_code=409,
            detail={
                "code": "SLIPPAGE_TOLERANCE_EXCEEDED",
                "message": "Order rejected before send: quote too far from signal price.",
                "deviation_pips": quote_deviation_pips,
                "tolerance_pips": slippage_tolerance_pips,
                "signal_price": intended_price,
                "quote_price": price,
            },
        )

    spread_points = ((tick.ask - tick.bid) / symbol_info.point) if symbol_info.point else 0.0
    stop_loss_pips = request.stop_loss_pips if request.stop_loss_pips is not None else _atr_stop_loss_pips(request.symbol)
    take_profit_pips = request.take_profit_pips if request.take_profit_pips is not None else (stop_loss_pips * 2.0)
    stop_loss = price - (stop_loss_pips * pip_size) if is_buy else price + (stop_loss_pips * pip_size)
    take_profit = price + (take_profit_pips * pip_size) if is_buy else price - (take_profit_pips * pip_size)

    if ENABLE_DYNAMIC_XAUUSD_SIZING and request.symbol.upper() == "XAUUSD":
        account = mt5.account_info()
        if account is None:
            mt5.shutdown()
            raise HTTPException(status_code=503, detail="Unable to read account info for dynamic sizing")

        risk_percent = request.strict_risk_percent if request.strict_risk_percent is not None else STRICT_RISK_PERCENT_DEFAULT
        atr_price = _compute_atr_price(request.symbol, period=14, timeframe=mt5.TIMEFRAME_M5)
        sizing = calculate_xauusd_lot_size(
            equity=float(account.equity),
            risk_percent=float(risk_percent),
            atr_14=float(atr_price),
            entry_price=float(price),
            action=request.action,
            tick_value=float(symbol_info.trade_tick_value or 0.0),
            tick_size=float(symbol_info.trade_tick_size or 0.0),
            volume_step=float(step),
            volume_min=float(min_vol),
            volume_max=float(max_vol),
            point=float(symbol_info.point or 0.0),
            sl_atr_multiplier=1.5,
        )

        if not sizing.allowed:
            mt5.shutdown()
            send_telegram_alert(
                "TRADE_ERROR",
                "Dynamic sizing blocked trade.",
                level="ERROR",
                extra={"symbol": request.symbol, "action": request.action, "reason": sizing.reason},
            )
            raise HTTPException(status_code=422, detail={"code": "DYNAMIC_SIZING_BLOCK", "reason": sizing.reason})

        volume = float(sizing.lot_size)
        stop_loss = float(sizing.stop_loss_price)
        sl_distance = abs(float(price) - stop_loss)
        take_profit = float(price + (2.0 * sl_distance)) if is_buy else float(price - (2.0 * sl_distance))
        stop_loss_pips = sl_distance / pip_size if pip_size > 0 else stop_loss_pips
        take_profit_pips = (2.0 * sl_distance) / pip_size if pip_size > 0 else take_profit_pips

        order_type_for_margin = mt5.ORDER_TYPE_BUY if is_buy else mt5.ORDER_TYPE_SELL
        required_margin = mt5.order_calc_margin(order_type_for_margin, request.symbol, volume, float(price))
        margin_ok, margin_reason = check_margin_sufficiency(
            required_margin=required_margin,
            free_margin=float(account.margin_free),
            equity=float(account.equity),
            min_post_trade_free_margin_pct=MIN_POST_TRADE_FREE_MARGIN_PCT,
        )
        if not margin_ok:
            mt5.shutdown()
            send_telegram_alert(
                "TRADE_ERROR",
                "Margin guard blocked trade.",
                level="ERROR",
                extra={
                    "symbol": request.symbol,
                    "action": request.action,
                    "reason": margin_reason,
                    "required_margin": required_margin,
                    "free_margin": float(account.margin_free),
                },
            )
            raise HTTPException(
                status_code=422,
                detail={
                    "code": "INSUFFICIENT_MARGIN_GUARD",
                    "reason": margin_reason,
                    "required_margin": required_margin,
                    "free_margin": float(account.margin_free),
                },
            )

    deviation_points = 20
    if symbol_info.point and symbol_info.point > 0:
        deviation_points = max(1, int(round((slippage_tolerance_pips * pip_size) / symbol_info.point)))

    order_type = mt5.ORDER_TYPE_BUY if is_buy else mt5.ORDER_TYPE_SELL
    request_payload = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": request.symbol,
        "volume": volume,
        "type": order_type,
        "price": price,
        "sl": stop_loss,
        "tp": take_profit,
        "deviation": deviation_points,
        "magic": 123456,
        "comment": f"TRADING {request.timeframe} live execution",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }

    stress_result = stress_wrapper.apply(
        request_payload=request_payload,
        symbol=request.symbol,
        action=request.action,
        symbol_point=float(symbol_info.point or 0.0),
        volume_step=float(step),
        volume_min=float(min_vol),
    )
    if stress_wrapper.enabled:
        print(
            f"[EXEC_STRESS] symbol={request.symbol} action={request.action} "
            f"slippage_points={stress_result.slippage_points} "
            f"requested_volume={stress_result.requested_volume:.2f} "
            f"effective_volume={stress_result.effective_volume:.2f} "
            f"partial_fill={stress_result.partial_fill_applied}"
        )

    effective_payload = stress_result.payload
    effective_volume = float(stress_result.effective_volume)
    effective_price = float(effective_payload.get("price", price))

    result, attempt_count, order_send_ts, order_send_done_ts = _send_with_retry(effective_payload)
    execution_latency_ms = (order_send_done_ts - order_send_ts).total_seconds() * 1000.0
    print(f"[EXECUTION_LATENCY] symbol={request.symbol} action={request.action} latency_ms={execution_latency_ms:.1f}")
    if execution_latency_ms > LATENCY_ALERT_MS:
        print(
            f"[EXECUTION_LATENCY_ALERT] symbol={request.symbol} action={request.action} "
            f"latency_ms={execution_latency_ms:.1f} threshold_ms={LATENCY_ALERT_MS:.1f}"
        )

    if result is not None and result.retcode == mt5.TRADE_RETCODE_DONE:
        quality_monitor.record_fill(
            source="broker_bridge",
            symbol=request.symbol,
            action=request.action,
            timeframe=request.timeframe,
            order_ticket=int(result.order) if getattr(result, "order", 0) else None,
            deal_ticket=int(result.deal) if getattr(result, "deal", 0) else None,
            signal_ts=request.signal_timestamp,
            signal_bar_time=request.signal_bar_time,
            signal_bar_relation=request.signal_bar_relation,
            order_send_ts=order_send_ts,
            order_send_done_ts=order_send_done_ts,
            intended_price=float(intended_price),
            spread_points=float(spread_points),
        )
        send_telegram_alert(
            "TRADE_ENTRY",
            "Trade entry executed.",
            level="INFO",
            extra={
                "symbol": request.symbol,
                "action": request.action,
                "ticket": int(result.order) if getattr(result, "order", 0) else None,
                "price": round(float(effective_price), 5),
                "latency_ms": round(execution_latency_ms, 2),
            },
        )

    mt5.shutdown()

    if result is None:
        send_telegram_alert(
            "TRADE_ERROR",
            "MT5 order_send returned no result.",
            level="ERROR",
            extra={"symbol": request.symbol, "action": request.action, "attempts": attempt_count},
        )
        raise HTTPException(status_code=502, detail=f"MT5 order_send returned no result after {attempt_count} attempts: {mt5.last_error()}")

    if result.retcode != mt5.TRADE_RETCODE_DONE:
        send_telegram_alert(
            "TRADE_ERROR",
            "MT5 order rejected after retries.",
            level="ERROR",
            extra={
                "symbol": request.symbol,
                "action": request.action,
                "retcode": getattr(result, "retcode", None),
                "attempts": attempt_count,
            },
        )
        raise HTTPException(
            status_code=502,
            detail={
                "retcode": result.retcode,
                "comment": result.comment,
                "attempts": attempt_count,
                "request": request_payload,
                "effective_request": effective_payload,
            },
        )

    status = "PARTIAL_FILLED" if stress_result.partial_fill_applied else "EXECUTED"
    return {
        "status": status,
        "order": result.order,
        "deal": result.deal,
        "volume": effective_volume,
        "requested_volume": float(stress_result.requested_volume),
        "price": effective_price,
        "sl": stop_loss,
        "tp": take_profit,
        "attempts": attempt_count,
        "execution_latency_ms": round(execution_latency_ms, 2),
        "slippage_tolerance_pips": slippage_tolerance_pips,
        "synthetic_slippage_points": int(stress_result.slippage_points),
        "synthetic_slippage_price_delta": float(stress_result.slippage_price_delta),
        "synthetic_partial_fill_applied": bool(stress_result.partial_fill_applied),
        "dynamic_sizing_enabled": bool(ENABLE_DYNAMIC_XAUUSD_SIZING and request.symbol.upper() == "XAUUSD"),
        "strict_risk_percent": float(request.strict_risk_percent if request.strict_risk_percent is not None else STRICT_RISK_PERCENT_DEFAULT),
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("MT5_BRIDGE_PORT", "9000")))
