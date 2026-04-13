import os
from datetime import datetime, timezone
from typing import Literal

import MetaTrader5 as mt5
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from risk_engine import RiskEngine
from execution_quality import ExecutionQualityMonitor

load_dotenv()

app = FastAPI(title="MT5 Broker Bridge", version="1.0.0")
risk_engine = RiskEngine()
quality_monitor = ExecutionQualityMonitor()


class LiveTradeRequest(BaseModel):
    symbol: str
    action: Literal["BUY", "SELL"]
    timeframe: str
    volume: float = Field(default=0.01, gt=0)
    stop_loss_pips: float = Field(default=50.0, gt=0)
    take_profit_pips: float = Field(default=100.0, gt=0)
    confidence_score: float | None = None
    signal_timestamp: float | str | None = None
    signal_bar_time: float | str | None = None
    signal_bar_relation: str | None = None
    intended_price: float | None = None


def _pip_size(symbol: str) -> float:
    normalized = symbol.upper()
    if normalized == "XAUUSD":
        return 0.1
    if normalized.endswith("JPY"):
        return 0.01
    return 0.0001


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
        raise HTTPException(status_code=400, detail=f"Unknown symbol: {request.symbol}")

    if not symbol_info.visible and not mt5.symbol_select(request.symbol, True):
        mt5.shutdown()
        raise HTTPException(status_code=400, detail=f"Unable to select symbol: {request.symbol}")

    tick = mt5.symbol_info_tick(request.symbol)
    if tick is None:
        mt5.shutdown()
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
    spread_points = ((tick.ask - tick.bid) / symbol_info.point) if symbol_info.point else 0.0
    stop_loss = price - (request.stop_loss_pips * pip_size) if is_buy else price + (request.stop_loss_pips * pip_size)
    take_profit = price + (request.take_profit_pips * pip_size) if is_buy else price - (request.take_profit_pips * pip_size)

    order_type = mt5.ORDER_TYPE_BUY if is_buy else mt5.ORDER_TYPE_SELL
    request_payload = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": request.symbol,
        "volume": volume,
        "type": order_type,
        "price": price,
        "sl": stop_loss,
        "tp": take_profit,
        "deviation": 20,
        "magic": 123456,
        "comment": f"TRADING {request.timeframe} live execution",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }

    order_send_ts = datetime.now(timezone.utc)
    result = mt5.order_send(request_payload)
    order_send_done_ts = datetime.now(timezone.utc)

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

    mt5.shutdown()

    if result is None:
        raise HTTPException(status_code=502, detail=f"MT5 order_send returned no result: {mt5.last_error()}")

    if result.retcode != mt5.TRADE_RETCODE_DONE:
        raise HTTPException(
            status_code=502,
            detail={
                "retcode": result.retcode,
                "comment": result.comment,
                "request": request_payload,
            },
        )

    return {
        "status": "EXECUTED",
        "order": result.order,
        "deal": result.deal,
        "volume": volume,
        "price": price,
        "sl": stop_loss,
        "tp": take_profit,
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("MT5_BRIDGE_PORT", "9000")))
