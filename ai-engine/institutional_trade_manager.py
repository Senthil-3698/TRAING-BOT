import math
import os
import time
from dataclasses import dataclass
from datetime import datetime

import MetaTrader5 as mt5
import numpy as np
from dotenv import load_dotenv

load_dotenv()


@dataclass
class ManagedPositionState:
    ticket: int
    symbol: str
    side: int
    entry_price: float
    initial_sl: float
    partial_done: bool = False
    trailing_active: bool = False
    last_sl: float = 0.0


class InstitutionalTradeManager:
    """Tick-by-tick MT5 trade manager for XAUUSD institutional exit orchestration."""

    def __init__(
        self,
        symbol: str = "XAUUSD",
        rr_trigger: float = 1.5,
        partial_close_ratio: float = 0.5,
        atr_period: int = 14,
        atr_timeframe: int = mt5.TIMEFRAME_M1,
        atr_trail_multiplier: float = 1.0,
        poll_interval_seconds: float = 0.2,
        bot_magic_id: int | None = None,
    ) -> None:
        self.symbol = symbol
        self.rr_trigger = rr_trigger
        self.partial_close_ratio = partial_close_ratio
        self.atr_period = atr_period
        self.atr_timeframe = atr_timeframe
        self.atr_trail_multiplier = atr_trail_multiplier
        self.poll_interval_seconds = poll_interval_seconds
        self.bot_magic_id = bot_magic_id if bot_magic_id is not None else int(os.getenv("BOT_MAGIC_IDS", "123456").split(",")[0].strip())
        self._tracked: dict[int, ManagedPositionState] = {}

    def initialize_mt5(self) -> bool:
        login = os.getenv("MT5_LOGIN")
        password = os.getenv("MT5_PASS") or os.getenv("MT5_PASSWORD")
        server = os.getenv("MT5_SERVER")
        if not login or not password or not server:
            print("[TRADE_MANAGER] Missing MT5 credentials in environment.")
            return False
        ok = mt5.initialize(login=int(login), password=password, server=server)
        if not ok:
            print(f"[TRADE_MANAGER] mt5.initialize failed: {mt5.last_error()}")
            return False
        if not mt5.symbol_select(self.symbol, True):
            print(f"[TRADE_MANAGER] Failed to select symbol {self.symbol}: {mt5.last_error()}")
            mt5.shutdown()
            return False
        return True

    def shutdown_mt5(self) -> None:
        mt5.shutdown()

    def run(self) -> None:
        if not self.initialize_mt5():
            return

        print(
            f"[TRADE_MANAGER] Running for {self.symbol} | RR trigger={self.rr_trigger:.2f} "
            f"partial={self.partial_close_ratio:.2f} atr_period={self.atr_period}"
        )

        try:
            while True:
                self.manage_tick()
                time.sleep(self.poll_interval_seconds)
        except KeyboardInterrupt:
            print("[TRADE_MANAGER] Stopped by user.")
        finally:
            self.shutdown_mt5()

    def manage_tick(self) -> None:
        positions = mt5.positions_get(symbol=self.symbol)
        if not positions:
            self._tracked.clear()
            return

        for pos in positions:
            if self.bot_magic_id and int(pos.magic) != int(self.bot_magic_id):
                continue
            if pos.sl is None or float(pos.sl) <= 0.0:
                continue

            state = self._tracked.get(int(pos.ticket))
            if state is None:
                state = ManagedPositionState(
                    ticket=int(pos.ticket),
                    symbol=pos.symbol,
                    side=int(pos.type),
                    entry_price=float(pos.price_open),
                    initial_sl=float(pos.sl),
                    partial_done=False,
                    trailing_active=False,
                    last_sl=float(pos.sl),
                )
                self._tracked[state.ticket] = state

            rr = self._current_rr(pos, state)
            if (not state.partial_done) and rr >= self.rr_trigger:
                self._execute_partial_and_be(state)

            if state.trailing_active:
                self._apply_atr_trailing_stop(state)

        live_tickets = {int(p.ticket) for p in positions}
        for tracked_ticket in list(self._tracked.keys()):
            if tracked_ticket not in live_tickets:
                del self._tracked[tracked_ticket]

    def _current_rr(self, pos, state: ManagedPositionState) -> float:
        tick = mt5.symbol_info_tick(state.symbol)
        if tick is None:
            return 0.0

        current_price = float(tick.bid) if int(pos.type) == mt5.ORDER_TYPE_BUY else float(tick.ask)
        risk_distance = abs(state.entry_price - state.initial_sl)
        if risk_distance <= 0.0:
            return 0.0

        if int(pos.type) == mt5.ORDER_TYPE_BUY:
            reward_distance = current_price - state.entry_price
        else:
            reward_distance = state.entry_price - current_price

        return reward_distance / risk_distance

    def _execute_partial_and_be(self, state: ManagedPositionState) -> None:
        live_position = mt5.positions_get(ticket=state.ticket)
        if not live_position:
            return
        pos = live_position[0]

        partial_result = self._partial_close_position(pos, self.partial_close_ratio)
        if partial_result is None or partial_result.retcode != mt5.TRADE_RETCODE_DONE:
            print(
                f"[TRADE_MANAGER] Partial close failed ticket={state.ticket} "
                f"retcode={getattr(partial_result, 'retcode', None)}"
            )
            return

        time.sleep(0.1)
        remaining = mt5.positions_get(ticket=state.ticket)
        if not remaining:
            state.partial_done = True
            state.trailing_active = False
            print(f"[TRADE_MANAGER] Position {state.ticket} fully closed after partial.")
            return

        updated_pos = remaining[0]
        be_price = self._breakeven_with_cost_cover(updated_pos)
        be_result = self._modify_position_sl(updated_pos, be_price)
        if be_result is None or be_result.retcode != mt5.TRADE_RETCODE_DONE:
            print(
                f"[TRADE_MANAGER] BE move failed ticket={state.ticket} "
                f"retcode={getattr(be_result, 'retcode', None)}"
            )
            return

        state.partial_done = True
        state.trailing_active = True
        state.last_sl = be_price
        print(
            f"[TRADE_MANAGER] ticket={state.ticket} partial=50% complete, "
            f"SL->BE+cost ({be_price:.2f}), trailing active"
        )

    def _partial_close_position(self, pos, ratio: float):
        info = mt5.symbol_info(pos.symbol)
        tick = mt5.symbol_info_tick(pos.symbol)
        if info is None or tick is None:
            return None

        volume_to_close = float(pos.volume) * ratio
        close_volume = self._normalize_volume(pos.symbol, volume_to_close)
        if close_volume <= 0.0:
            return None

        if int(pos.type) == mt5.ORDER_TYPE_BUY:
            close_type = mt5.ORDER_TYPE_SELL
            close_price = float(tick.bid)
        else:
            close_type = mt5.ORDER_TYPE_BUY
            close_price = float(tick.ask)

        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": pos.symbol,
            "position": int(pos.ticket),
            "volume": close_volume,
            "type": close_type,
            "price": close_price,
            "deviation": 20,
            "magic": self.bot_magic_id,
            "comment": "INST_PARTIAL_50",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        return mt5.order_send(request)

    def _breakeven_with_cost_cover(self, pos) -> float:
        info = mt5.symbol_info(pos.symbol)
        if info is None or info.point <= 0 or info.trade_tick_size <= 0:
            return float(pos.price_open)

        commission = float(getattr(pos, "commission", 0.0) or 0.0)
        swap = float(getattr(pos, "swap", 0.0) or 0.0)
        cost_to_cover = max(0.0, -(commission + swap))

        if cost_to_cover <= 0.0:
            points_buffer = 0
        else:
            money_per_point_per_lot = float(info.trade_tick_value) * (float(info.point) / float(info.trade_tick_size))
            money_per_point = max(1e-9, money_per_point_per_lot * float(pos.volume))
            points_buffer = int(math.ceil(cost_to_cover / money_per_point))

        buffer_price = points_buffer * float(info.point)
        if int(pos.type) == mt5.ORDER_TYPE_BUY:
            return float(pos.price_open) + buffer_price
        return float(pos.price_open) - buffer_price

    def _apply_atr_trailing_stop(self, state: ManagedPositionState) -> None:
        live_position = mt5.positions_get(ticket=state.ticket)
        if not live_position:
            return
        pos = live_position[0]

        atr = self._calculate_atr(state.symbol, self.atr_timeframe, self.atr_period)
        if atr is None or atr <= 0.0:
            return

        tick = mt5.symbol_info_tick(state.symbol)
        if tick is None:
            return

        trail_distance = atr * self.atr_trail_multiplier
        if int(pos.type) == mt5.ORDER_TYPE_BUY:
            candidate_sl = float(tick.bid) - trail_distance
            candidate_sl = max(candidate_sl, float(pos.price_open))
            tighten = candidate_sl > float(pos.sl)
        else:
            candidate_sl = float(tick.ask) + trail_distance
            candidate_sl = min(candidate_sl, float(pos.price_open))
            tighten = candidate_sl < float(pos.sl)

        if not tighten:
            return

        result = self._modify_position_sl(pos, candidate_sl)
        if result is not None and result.retcode == mt5.TRADE_RETCODE_DONE:
            state.last_sl = candidate_sl
            print(
                f"[TRADE_MANAGER][TRAIL] ticket={state.ticket} "
                f"atr={atr:.4f} trail_distance={trail_distance:.4f} sl={candidate_sl:.2f}"
            )

    def _modify_position_sl(self, pos, new_sl: float):
        request = {
            "action": mt5.TRADE_ACTION_SLTP,
            "position": int(pos.ticket),
            "symbol": pos.symbol,
            "sl": float(new_sl),
            "tp": float(pos.tp),
            "magic": self.bot_magic_id,
            "comment": "INST_BE_ATR_TRAIL",
        }
        return mt5.order_send(request)

    def _calculate_atr(self, symbol: str, timeframe: int, period: int) -> float | None:
        rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, period + 1)
        if rates is None or len(rates) < period + 1:
            return None

        highs = rates["high"][1:]
        lows = rates["low"][1:]
        prev_closes = rates["close"][:-1]

        tr1 = highs - lows
        tr2 = np.abs(highs - prev_closes)
        tr3 = np.abs(lows - prev_closes)
        true_range = np.maximum(tr1, np.maximum(tr2, tr3))
        return float(np.mean(true_range[-period:]))

    def _normalize_volume(self, symbol: str, volume: float) -> float:
        info = mt5.symbol_info(symbol)
        if info is None:
            return 0.0
        step = float(info.volume_step or 0.01)
        min_lot = float(info.volume_min or 0.01)
        max_lot = float(info.volume_max or volume)

        normalized = round(round(volume / step) * step, 2)
        if normalized < min_lot:
            return 0.0
        return max(min_lot, min(normalized, max_lot))


def main() -> None:
    manager = InstitutionalTradeManager(
        symbol=os.getenv("TRADE_MANAGER_SYMBOL", "XAUUSD"),
        rr_trigger=float(os.getenv("TRADE_MANAGER_RR_TRIGGER", "1.5")),
        partial_close_ratio=float(os.getenv("TRADE_MANAGER_PARTIAL_RATIO", "0.5")),
        atr_period=int(os.getenv("TRADE_MANAGER_ATR_PERIOD", "14")),
        atr_trail_multiplier=float(os.getenv("TRADE_MANAGER_ATR_TRAIL_MULTIPLIER", "1.0")),
        poll_interval_seconds=float(os.getenv("TRADE_MANAGER_POLL_SEC", "0.2")),
    )
    manager.run()


if __name__ == "__main__":
    print(f"[TRADE_MANAGER] start {datetime.utcnow().isoformat()}Z")
    main()
