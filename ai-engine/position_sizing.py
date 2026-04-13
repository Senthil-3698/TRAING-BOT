from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional


@dataclass
class PositionSizingResult:
    allowed: bool
    lot_size: float
    stop_loss_distance_price: float
    stop_loss_price: float
    reason: str = ""
    required_margin: float = 0.0


def calculate_xauusd_lot_size(
    *,
    equity: float,
    risk_percent: float,
    atr_14: float,
    entry_price: float,
    action: str,
    tick_value: float,
    tick_size: float,
    volume_step: float,
    volume_min: float,
    volume_max: float,
    point: float,
    sl_atr_multiplier: float = 1.5,
) -> PositionSizingResult:
    """
    Institutional volatility-adjusted sizing for XAUUSD.

    Formula:
      lot = (equity * risk%) / (SL distance * value per price-unit per lot)

    where:
      SL distance = sl_atr_multiplier * ATR(14)
      value per price-unit per lot = tick_value / tick_size
    """
    if equity <= 0:
        return PositionSizingResult(False, 0.0, 0.0, 0.0, "Invalid equity")

    if risk_percent <= 0 or risk_percent > 100:
        return PositionSizingResult(False, 0.0, 0.0, 0.0, "Risk percent must be in (0, 100]")

    if atr_14 <= 0:
        return PositionSizingResult(False, 0.0, 0.0, 0.0, "ATR must be positive")

    if tick_value <= 0 or tick_size <= 0 or point <= 0:
        return PositionSizingResult(False, 0.0, 0.0, 0.0, "Invalid symbol tick/point metadata")

    sl_distance_price = float(atr_14) * float(sl_atr_multiplier)
    if sl_distance_price <= 0:
        return PositionSizingResult(False, 0.0, 0.0, 0.0, "Computed stop-loss distance is invalid")

    risk_amount = float(equity) * (float(risk_percent) / 100.0)
    value_per_price_unit = float(tick_value) / float(tick_size)
    loss_per_lot = sl_distance_price * value_per_price_unit

    if loss_per_lot <= 0:
        return PositionSizingResult(False, 0.0, 0.0, 0.0, "Loss per lot is invalid")

    raw_lot = risk_amount / loss_per_lot

    step = volume_step if volume_step > 0 else 0.01
    lot = round(round(raw_lot / step) * step, 2)

    if lot > volume_max:
        return PositionSizingResult(False, 0.0, sl_distance_price, 0.0, "Lot size exceeds broker max volume")

    if lot < volume_min:
        return PositionSizingResult(False, 0.0, sl_distance_price, 0.0, "Lot size below broker min volume")

    if action.upper() == "BUY":
        sl_price = entry_price - sl_distance_price
    else:
        sl_price = entry_price + sl_distance_price

    # Snap SL to symbol point precision for consistent broker requests.
    point_digits = max(0, int(round(-math.log10(point)))) if point < 1 else 0
    sl_price = round(sl_price, point_digits)

    return PositionSizingResult(
        allowed=True,
        lot_size=lot,
        stop_loss_distance_price=sl_distance_price,
        stop_loss_price=sl_price,
        reason="OK",
    )


def check_margin_sufficiency(
    *,
    required_margin: Optional[float],
    free_margin: float,
    equity: float,
    min_post_trade_free_margin_pct: float,
) -> tuple[bool, str]:
    if required_margin is None:
        return False, "Unable to compute required margin"

    if required_margin > free_margin:
        return False, "Insufficient free margin"

    post_trade_free_margin = free_margin - required_margin
    threshold = max(0.0, equity * min_post_trade_free_margin_pct)
    if post_trade_free_margin < threshold:
        return False, "Post-trade free margin safety threshold breached"

    return True, "OK"
