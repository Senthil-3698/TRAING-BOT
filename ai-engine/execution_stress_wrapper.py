from __future__ import annotations

import os
import random
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any


@dataclass
class StressApplication:
    payload: dict[str, Any]
    requested_volume: float
    effective_volume: float
    slippage_points: int
    slippage_price_delta: float
    partial_fill_applied: bool


class ExecutionStressWrapper:
    """Injects synthetic adverse slippage and partial fills for demo forward-testing."""

    def __init__(self) -> None:
        self.enabled = os.getenv("EXEC_STRESS_ENABLED", "0") == "1"
        self.slippage_enabled = os.getenv("EXEC_STRESS_SLIPPAGE_ENABLED", "1") == "1"
        self.partial_fill_enabled = os.getenv("EXEC_STRESS_PARTIAL_FILL_ENABLED", "1") == "1"

        self.min_slippage_points = int(os.getenv("EXEC_STRESS_MIN_SLIPPAGE_POINTS", "10"))
        self.max_slippage_points = int(os.getenv("EXEC_STRESS_MAX_SLIPPAGE_POINTS", "30"))

        self.partial_trigger_volume = float(os.getenv("EXEC_STRESS_PARTIAL_TRIGGER_VOLUME", "0.05"))
        self.partial_fill_ratio_low = float(os.getenv("EXEC_STRESS_PARTIAL_FILL_RATIO_LOW", "0.40"))
        self.partial_fill_ratio_high = float(os.getenv("EXEC_STRESS_PARTIAL_FILL_RATIO_HIGH", "0.85"))

    def apply(
        self,
        *,
        request_payload: dict[str, Any],
        symbol: str,
        action: str,
        symbol_point: float,
        volume_step: float,
        volume_min: float,
    ) -> StressApplication:
        requested_volume = float(request_payload.get("volume", 0.0) or 0.0)
        stressed = dict(request_payload)

        if not self.enabled or requested_volume <= 0.0 or symbol_point <= 0:
            return StressApplication(
                payload=stressed,
                requested_volume=requested_volume,
                effective_volume=requested_volume,
                slippage_points=0,
                slippage_price_delta=0.0,
                partial_fill_applied=False,
            )

        slippage_points = self._time_of_day_slippage_points(datetime.now(timezone.utc).hour)
        slippage_delta = float(slippage_points) * float(symbol_point)
        if self.slippage_enabled:
            self._apply_adverse_price(
                stressed,
                action=action,
                delta=slippage_delta,
                slippage_points=slippage_points,
            )

        partial_fill_applied = False
        effective_volume = requested_volume
        if self.partial_fill_enabled and requested_volume >= self.partial_trigger_volume:
            ratio = self._partial_fill_ratio_by_volume(requested_volume)
            simulated = requested_volume * ratio
            normalized = self._normalize_volume(simulated, volume_step=volume_step, volume_min=volume_min)
            if 0.0 < normalized < requested_volume:
                effective_volume = normalized
                stressed["volume"] = normalized
                partial_fill_applied = True

        return StressApplication(
            payload=stressed,
            requested_volume=requested_volume,
            effective_volume=effective_volume,
            slippage_points=slippage_points if self.slippage_enabled else 0,
            slippage_price_delta=slippage_delta if self.slippage_enabled else 0.0,
            partial_fill_applied=partial_fill_applied,
        )

    def _time_of_day_slippage_points(self, hour_utc: int) -> int:
        # Session-aware stress profile for XAUUSD:
        # rollover/illiquid windows -> heavier slippage, active sessions -> moderate.
        if 21 <= hour_utc or hour_utc <= 1:
            band = (24, self.max_slippage_points)
        elif 12 <= hour_utc <= 16:
            band = (18, min(self.max_slippage_points, 26))
        elif 6 <= hour_utc <= 11:
            band = (12, min(self.max_slippage_points, 20))
        else:
            band = (self.min_slippage_points, min(self.max_slippage_points, 16))

        low = max(self.min_slippage_points, band[0])
        high = max(low, band[1])
        return random.randint(low, high)

    def _partial_fill_ratio_by_volume(self, volume: float) -> float:
        if volume >= 0.50:
            low = max(0.25, self.partial_fill_ratio_low - 0.10)
            high = min(0.70, self.partial_fill_ratio_high - 0.15)
        elif volume >= 0.20:
            low = max(0.30, self.partial_fill_ratio_low - 0.05)
            high = min(0.80, self.partial_fill_ratio_high - 0.05)
        else:
            low = self.partial_fill_ratio_low
            high = self.partial_fill_ratio_high

        high = max(low, high)
        return random.uniform(low, high)

    @staticmethod
    def _apply_adverse_price(payload: dict[str, Any], action: str, delta: float, slippage_points: int) -> None:
        current_price = float(payload.get("price", 0.0) or 0.0)
        if current_price <= 0.0:
            return

        action_upper = action.upper()
        if action_upper == "BUY":
            payload["price"] = current_price + delta
        else:
            payload["price"] = current_price - delta

        existing_dev = int(payload.get("deviation", 20) or 20)
        payload["deviation"] = max(existing_dev, slippage_points + 5)

    @staticmethod
    def _normalize_volume(volume: float, *, volume_step: float, volume_min: float) -> float:
        step = volume_step or 0.01
        minimum = volume_min or step
        rounded = round(round(volume / step) * step, 2)
        if rounded < minimum:
            return 0.0
        return rounded
