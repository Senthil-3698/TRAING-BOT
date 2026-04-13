from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any

import MetaTrader5 as mt5
import redis
from dotenv import load_dotenv

from trade_journal import TradeJournal

load_dotenv()


@dataclass
class RiskDecision:
    allowed: bool
    code: str
    message: str
    details: dict[str, Any]


class _SafeRedisClient:
    """Best-effort Redis client that degrades to no-op reads/writes when Redis is down."""

    def __init__(self, client: redis.Redis) -> None:
        self._client = client

    def get(self, key: str):
        try:
            return self._client.get(key)
        except Exception:
            return None

    def set(self, key: str, value: Any):
        try:
            return self._client.set(key, value)
        except Exception:
            return None

    def lpush(self, key: str, *values: Any):
        try:
            return self._client.lpush(key, *values)
        except Exception:
            return None

    def ltrim(self, key: str, start: int, end: int):
        try:
            return self._client.ltrim(key, start, end)
        except Exception:
            return None


class RiskEngine:
    """Single authoritative pre-trade risk gate for all Python execution paths."""

    def __init__(self) -> None:
        self.redis = _SafeRedisClient(redis.Redis(
            host=os.getenv("REDIS_HOST", "localhost"),
            port=int(os.getenv("REDIS_PORT", "6380")),
            db=0,
        ))
        self.journal = TradeJournal()

        self.kill_switch_key = os.getenv("GLOBAL_KILL_SWITCH_KEY", "GLOBAL_KILL_SWITCH")
        self.daily_loss_limit_pct = float(os.getenv("DAILY_LOSS_LIMIT_PCT", "0.03"))
        self.weekly_loss_limit_pct = float(os.getenv("WEEKLY_LOSS_LIMIT_PCT", "0.06"))
        self.daily_drawdown_limit_pct = float(os.getenv("DAILY_DRAWDOWN_CIRCUIT_BREAKER_PCT", "0.03"))
        self.max_consecutive_losses = int(os.getenv("MAX_CONSECUTIVE_LOSSES", "4"))
        self.cooldown_hours = int(os.getenv("LOSS_COOLDOWN_HOURS", "2"))
        self.max_concurrent_positions = int(os.getenv("MAX_CONCURRENT_POSITIONS", "3"))
        self.max_correlated_positions = int(os.getenv("MAX_CORRELATED_POSITIONS", "2"))
        self.news_blackout_minutes = int(os.getenv("NEWS_BLACKOUT_MINUTES", "15"))
        self.spread_lookback = int(os.getenv("SPREAD_LOOKBACK", "20"))
        self.kelly_window = int(os.getenv("KELLY_LOOKBACK_TRADES", "50"))
        self.kelly_min_trades = int(os.getenv("KELLY_MIN_TRADES", "30"))

        magic_csv = os.getenv("BOT_MAGIC_IDS", "123456,20260411")
        self.bot_magic_ids = {
            int(x.strip()) for x in magic_csv.split(",") if x.strip().isdigit()
        }

        self.correlation_groups = self._load_correlation_groups()
        self.blocked_correlated_long_pairs = self._load_blocked_correlated_long_pairs()

    def _load_correlation_groups(self) -> list[set[str]]:
        raw = os.getenv("CORRELATED_SYMBOL_GROUPS", "XAUUSD,GOLD,XAUUSDm,XAUUSD.pro")
        groups: list[set[str]] = []
        for chunk in raw.split(";"):
            symbols = {s.strip().upper() for s in chunk.split(",") if s.strip()}
            if symbols:
                groups.append(symbols)
        return groups

    def _load_blocked_correlated_long_pairs(self) -> list[tuple[str, str]]:
        raw = os.getenv("BLOCKED_CORRELATED_LONG_PAIRS", "EURUSD:GBPUSD")
        pairs: list[tuple[str, str]] = []
        for chunk in raw.split(","):
            part = chunk.strip().upper()
            if not part:
                continue
            if ":" in part:
                left, right = part.split(":", 1)
            elif "-" in part:
                left, right = part.split("-", 1)
            else:
                continue
            left = left.strip()
            right = right.strip()
            if left and right and left != right:
                pairs.append((left, right))
        return pairs

    def _ensure_mt5(self) -> bool:
        if mt5.terminal_info() is not None:
            return True
        login_raw = os.getenv("MT5_LOGIN")
        password = os.getenv("MT5_PASS") or os.getenv("MT5_PASSWORD")
        server = os.getenv("MT5_SERVER")
        path = os.getenv("MT5_PATH")

        login = int(login_raw) if login_raw else None
        return mt5.initialize(
            path=path if path else None,
            login=login,
            password=password,
            server=server,
        )

    def _utc_now(self) -> datetime:
        return datetime.now(timezone.utc)

    def _day_key(self, now: datetime) -> str:
        return now.strftime("%Y-%m-%d")

    def _week_key(self, now: datetime) -> str:
        year, week, _ = now.isocalendar()
        return f"{year}-{week:02d}"

    def _start_balance(self, period: str, now: datetime) -> float:
        if period == "day":
            key = f"risk:start_balance:day:{self._day_key(now)}"
        else:
            key = f"risk:start_balance:week:{self._week_key(now)}"

        cached = self.redis.get(key)
        if cached:
            try:
                return float(cached.decode("utf-8"))
            except ValueError:
                pass

        account = mt5.account_info()
        start_balance = float(account.equity if account else 0.0)
        if start_balance > 0:
            self.redis.set(key, start_balance)
        return start_balance

    def _deals_in_window(self, start: datetime, end: datetime):
        deals = mt5.history_deals_get(start, end)
        if deals is None:
            return []
        return [d for d in deals if getattr(d, "magic", 0) in self.bot_magic_ids]

    def _realized_loss(self, start: datetime, end: datetime) -> float:
        deals = self._deals_in_window(start, end)
        pnl = 0.0
        for d in deals:
            # Closed deal legs are the realized PnL legs.
            if getattr(d, "entry", None) == mt5.DEAL_ENTRY_OUT:
                profit = float(getattr(d, "profit", 0.0))
                commission = float(getattr(d, "commission", 0.0))
                swap = float(getattr(d, "swap", 0.0))
                pnl += profit + commission + swap
        return pnl

    def _consecutive_losses(self, lookback_days: int = 7) -> tuple[int, datetime | None]:
        end = self._utc_now()
        start = end - timedelta(days=lookback_days)
        deals = self._deals_in_window(start, end)
        closed = [d for d in deals if getattr(d, "entry", None) == mt5.DEAL_ENTRY_OUT]
        if not closed:
            return 0, None

        closed.sort(key=lambda d: getattr(d, "time", 0), reverse=True)
        losses = 0
        last_loss_time = None
        for d in closed:
            profit = float(getattr(d, "profit", 0.0)) + float(getattr(d, "commission", 0.0)) + float(getattr(d, "swap", 0.0))
            if profit < 0:
                losses += 1
                if last_loss_time is None:
                    ts = int(getattr(d, "time", 0))
                    last_loss_time = datetime.fromtimestamp(ts, tz=timezone.utc)
            else:
                break
        return losses, last_loss_time

    def _cooldown_active(self, now: datetime) -> tuple[bool, datetime | None]:
        raw = self.redis.get("risk:cooldown_until")
        if not raw:
            return False, None
        try:
            until = datetime.fromisoformat(raw.decode("utf-8"))
        except ValueError:
            return False, None
        return now < until, until

    def _daily_circuit_breaker_active(self, now: datetime) -> tuple[bool, datetime | None]:
        raw = self.redis.get("risk:daily_circuit_breaker_until")
        if not raw:
            return False, None
        try:
            until = datetime.fromisoformat(raw.decode("utf-8"))
        except ValueError:
            return False, None
        return now < until, until

    def _set_daily_circuit_breaker(self, until: datetime, context: dict[str, Any]) -> None:
        self.redis.set("risk:daily_circuit_breaker_until", until.isoformat())
        self.redis.set("risk:daily_circuit_breaker_reason", json.dumps(context, default=str))

    def _set_cooldown(self, until: datetime) -> None:
        self.redis.set("risk:cooldown_until", until.isoformat())

    def _symbol_group(self, symbol: str) -> set[str]:
        normalized = symbol.upper()
        for group in self.correlation_groups:
            if normalized in group:
                return group
        return {normalized}

    def _count_correlated_exposure(self, symbol: str) -> int:
        group = self._symbol_group(symbol)
        positions = mt5.positions_get() or []
        correlated = [
            p
            for p in positions
            if getattr(p, "symbol", "").upper() in group
            and getattr(p, "magic", 0) in self.bot_magic_ids
        ]
        return len(correlated)

    def _open_bot_positions(self):
        positions = mt5.positions_get() or []
        return [p for p in positions if getattr(p, "magic", 0) in self.bot_magic_ids]

    def _count_open_bot_positions(self) -> int:
        return len(self._open_bot_positions())

    def _violates_directional_correlation(self, symbol: str, action: str) -> tuple[bool, dict[str, Any]]:
        normalized_symbol = symbol.upper()
        normalized_action = action.upper()
        if normalized_action != "BUY":
            return False, {}

        positions = self._open_bot_positions()
        open_long_symbols = {
            getattr(p, "symbol", "").upper()
            for p in positions
            if getattr(p, "type", None) == mt5.POSITION_TYPE_BUY
        }

        for a, b in self.blocked_correlated_long_pairs:
            if normalized_symbol == a and b in open_long_symbols:
                return True, {"requested_symbol": a, "blocked_with": b, "action": "BUY"}
            if normalized_symbol == b and a in open_long_symbols:
                return True, {"requested_symbol": b, "blocked_with": a, "action": "BUY"}

        return False, {}

    def _high_impact_events(self) -> list[datetime]:
        events: list[datetime] = []
        raw = self.redis.get("HIGH_IMPACT_EVENTS")
        if raw:
            try:
                parsed = json.loads(raw.decode("utf-8"))
                if isinstance(parsed, list):
                    for item in parsed:
                        if isinstance(item, (int, float)):
                            events.append(datetime.fromtimestamp(float(item), tz=timezone.utc))
                        elif isinstance(item, str):
                            events.append(datetime.fromisoformat(item.replace("Z", "+00:00")))
            except Exception:
                pass
        return events

    def _spread_points(self, symbol: str) -> tuple[float, float]:
        tick = mt5.symbol_info_tick(symbol)
        info = mt5.symbol_info(symbol)
        if tick is None or info is None or info.point <= 0:
            return 0.0, 0.0

        current_spread = (tick.ask - tick.bid) / info.point
        rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M1, 0, self.spread_lookback)
        if rates is None or len(rates) == 0:
            return current_spread, 0.0

        spreads = [float(r["spread"]) for r in rates if "spread" in rates.dtype.names and float(r["spread"]) > 0]
        avg_spread = sum(spreads) / len(spreads) if spreads else 0.0
        return current_spread, avg_spread

    def _reject(
        self,
        *,
        code: str,
        message: str,
        symbol: str,
        action: str,
        timeframe: str,
        source: str,
        details: dict[str, Any] | None = None,
    ) -> RiskDecision:
        payload = {
            "risk_code": code,
            "risk_message": message,
            "source": source,
            **(details or {}),
        }

        self.redis.lpush("risk:rejections", json.dumps(payload, default=str))
        self.redis.ltrim("risk:rejections", 0, 499)

        self.journal.log_signal(
            source=source,
            symbol=symbol,
            action=action,
            timeframe=timeframe,
            ai_decision="REJECTED",
            ai_reasoning=f"RiskEngine:{code}",
            decision_status="REJECTED",
            rejection_reason=message,
            metadata={"risk": payload},
        )

        return RiskDecision(False, code, message, payload)

    def pre_trade_check(
        self,
        *,
        symbol: str,
        action: str,
        timeframe: str = "1m",
        source: str = "unknown",
        purpose: str = "OPEN",
    ) -> RiskDecision:
        purpose = purpose.upper()

        if os.getenv("RISK_ENGINE_BYPASS", "0") == "1":
            return RiskDecision(
                True,
                "ALLOWED_BYPASS",
                "Risk engine bypass enabled via RISK_ENGINE_BYPASS=1.",
                {"source": source, "purpose": purpose},
            )

        if not self._ensure_mt5():
            return self._reject(
                code="MT5_INIT_FAILED",
                message=f"MT5 initialize failed: {mt5.last_error()}",
                symbol=symbol,
                action=action,
                timeframe=timeframe,
                source=source,
            )

        if purpose in {"CLOSE", "MODIFY"}:
            return RiskDecision(True, "ALLOWED_NON_ENTRY", "Non-entry order allowed.", {"purpose": purpose})

        now = self._utc_now()

        kill_switch_status = self.redis.get(self.kill_switch_key)
        if kill_switch_status and kill_switch_status.decode("utf-8").upper() == "ACTIVE":
            return self._reject(
                code="KILL_SWITCH_ACTIVE",
                message="Global kill switch is active.",
                symbol=symbol,
                action=action,
                timeframe=timeframe,
                source=source,
            )

        day_start = datetime(now.year, now.month, now.day, tzinfo=timezone.utc)
        week_start = day_start - timedelta(days=day_start.weekday())

        day_balance = self._start_balance("day", now)
        week_balance = self._start_balance("week", now)

        day_pnl = self._realized_loss(day_start, now)
        week_pnl = self._realized_loss(week_start, now)

        if day_balance > 0 and day_pnl <= -(day_balance * self.daily_loss_limit_pct):
            return self._reject(
                code="DAILY_LOSS_LIMIT",
                message="Daily loss limit reached.",
                symbol=symbol,
                action=action,
                timeframe=timeframe,
                source=source,
                details={"day_pnl": day_pnl, "day_start_balance": day_balance},
            )

        if week_balance > 0 and week_pnl <= -(week_balance * self.weekly_loss_limit_pct):
            return self._reject(
                code="WEEKLY_LOSS_LIMIT",
                message="Weekly loss limit reached.",
                symbol=symbol,
                action=action,
                timeframe=timeframe,
                source=source,
                details={"week_pnl": week_pnl, "week_start_balance": week_balance},
            )

        cooldown_on, cooldown_until = self._cooldown_active(now)
        if cooldown_on:
            return self._reject(
                code="COOLDOWN_ACTIVE",
                message=f"Loss cooldown active until {cooldown_until.isoformat()}",
                symbol=symbol,
                action=action,
                timeframe=timeframe,
                source=source,
            )

        daily_breaker_on, daily_breaker_until = self._daily_circuit_breaker_active(now)
        if daily_breaker_on:
            return self._reject(
                code="DAILY_DRAWDOWN_CIRCUIT_BREAKER",
                message=f"Daily drawdown circuit breaker active until {daily_breaker_until.isoformat()}.",
                symbol=symbol,
                action=action,
                timeframe=timeframe,
                source=source,
            )

        account = mt5.account_info()
        current_equity = float(account.equity) if account else 0.0
        daily_drawdown_pct = 0.0
        if day_balance > 0 and current_equity > 0:
            daily_drawdown_pct = max(0.0, (day_balance - current_equity) / day_balance)

        if day_balance > 0 and daily_drawdown_pct >= self.daily_drawdown_limit_pct:
            midnight_reset = datetime(now.year, now.month, now.day, tzinfo=timezone.utc) + timedelta(days=1)
            context = {
                "triggered_at": now.isoformat(),
                "reset_at": midnight_reset.isoformat(),
                "day_start_balance": day_balance,
                "current_equity": current_equity,
                "drawdown_pct": daily_drawdown_pct,
                "threshold_pct": self.daily_drawdown_limit_pct,
            }
            self._set_daily_circuit_breaker(midnight_reset, context)
            return self._reject(
                code="DAILY_DRAWDOWN_CIRCUIT_BREAKER",
                message="Daily drawdown exceeded limit; trading halted until midnight UTC reset.",
                symbol=symbol,
                action=action,
                timeframe=timeframe,
                source=source,
                details=context,
            )

        open_positions_count = self._count_open_bot_positions()
        if open_positions_count >= self.max_concurrent_positions:
            return self._reject(
                code="MAX_CONCURRENT_POSITIONS",
                message="Max concurrent open positions reached.",
                symbol=symbol,
                action=action,
                timeframe=timeframe,
                source=source,
                details={
                    "open_positions": open_positions_count,
                    "limit": self.max_concurrent_positions,
                },
            )

        corr_violation, corr_details = self._violates_directional_correlation(symbol, action)
        if corr_violation:
            return self._reject(
                code="CORRELATED_LONG_BLOCK",
                message="Blocked correlated long exposure (EURUSD/GBPUSD policy).",
                symbol=symbol,
                action=action,
                timeframe=timeframe,
                source=source,
                details=corr_details,
            )

        consecutive_losses, last_loss_time = self._consecutive_losses()
        if consecutive_losses >= self.max_consecutive_losses and last_loss_time is not None:
            until = last_loss_time + timedelta(hours=self.cooldown_hours)
            if now < until:
                self._set_cooldown(until)
                return self._reject(
                    code="MAX_CONSECUTIVE_LOSSES",
                    message=f"{consecutive_losses} consecutive losses. Cooldown until {until.isoformat()}.",
                    symbol=symbol,
                    action=action,
                    timeframe=timeframe,
                    source=source,
                    details={"consecutive_losses": consecutive_losses},
                )

        correlated_count = self._count_correlated_exposure(symbol)
        if correlated_count >= self.max_correlated_positions:
            return self._reject(
                code="MAX_CORRELATED_EXPOSURE",
                message="Max correlated exposure reached.",
                symbol=symbol,
                action=action,
                timeframe=timeframe,
                source=source,
                details={
                    "correlated_count": correlated_count,
                    "limit": self.max_correlated_positions,
                    "group": list(self._symbol_group(symbol)),
                },
            )

        for event_time in self._high_impact_events():
            delta_min = abs((now - event_time).total_seconds()) / 60.0
            if delta_min <= self.news_blackout_minutes:
                return self._reject(
                    code="NEWS_BLACKOUT",
                    message="Within high-impact news blackout window.",
                    symbol=symbol,
                    action=action,
                    timeframe=timeframe,
                    source=source,
                    details={"event_time": event_time.isoformat(), "delta_minutes": round(delta_min, 2)},
                )

        current_spread, avg_spread = self._spread_points(symbol)
        bypass_spread_sanity = os.getenv("RISK_BYPASS_SPREAD_SANITY", "0") == "1"
        if (not bypass_spread_sanity) and avg_spread > 0 and current_spread > (2.0 * avg_spread):
            return self._reject(
                code="SPREAD_SANITY",
                message="Spread is above 2x average spread.",
                symbol=symbol,
                action=action,
                timeframe=timeframe,
                source=source,
                details={"current_spread": current_spread, "avg_spread": avg_spread},
            )

        return RiskDecision(
            True,
            "ALLOWED",
            "All pre-trade checks passed.",
            {
                "source": source,
                "current_spread": current_spread,
                "avg_spread": avg_spread,
                "consecutive_losses": consecutive_losses,
                "day_pnl": day_pnl,
                "week_pnl": week_pnl,
                "daily_drawdown_pct": daily_drawdown_pct,
                "open_positions": open_positions_count,
            },
        )

    def calculate_position_size(
        self,
        *,
        symbol: str,
        stop_loss_pips: float,
    ) -> float:
        """Volatility-aware position sizing using fractional Kelly with broker limits."""
        if not self._ensure_mt5():
            return 0.01

        account = mt5.account_info()
        if account is None or stop_loss_pips <= 0:
            return 0.01

        info = mt5.symbol_info(symbol)
        if info is None or info.point <= 0:
            return 0.01

        risk_fraction = self._kelly_risk_fraction(symbol)
        equity = float(account.equity)
        risk_dollars = equity * risk_fraction

        pip_value_per_lot = info.point * (info.trade_contract_size or 100)
        sl_dollars_per_lot = stop_loss_pips * pip_value_per_lot
        if sl_dollars_per_lot <= 0:
            return float(info.volume_min or 0.01)

        raw = risk_dollars / sl_dollars_per_lot
        capped = max(info.volume_min or 0.01, min(raw, info.volume_max or raw))
        step = info.volume_step or 0.01
        return round(round(capped / step) * step, 2)

        def calculate_position_size(
                self,
                *,
                symbol: str,
                stop_loss_pips: float,
        ) -> float:
                """
                Position size using a volatility-adjusted fractional Kelly risk fraction.

                Math:
                - Estimate rolling edge from last N=50 closed trades in signal_journal.
                    W = win probability.
                    R = average win / absolute average loss.
                - Kelly-optimal fraction (fixed-odds approximation):
                    f* = W - (1 - W) / R
                - Use quarter-Kelly for robustness:
                    f = 0.25 * max(0, f*)
                - Clamp base risk fraction to [0.25%, 2%].
                - Scale inversely by current ATR percentile so higher realized volatility
                    reduces size.

                If fewer than 30 closed trades exist, use flat 1% risk.

                References:
                - Kelly, J. L. (1956), "A New Interpretation of Information Rate".
                - Thorp, E. O. (1969), "Optimal Gambling Systems for Favorable Games".
                """
                if not self._ensure_mt5():
                        return 0.01
                account = mt5.account_info()
                if account is None or stop_loss_pips <= 0:
                        return 0.01

                info = mt5.symbol_info(symbol)
                if info is None or info.point <= 0:
                        return 0.01

                risk_fraction = self._kelly_risk_fraction(symbol)
                equity = float(account.equity)
                risk_dollars = equity * risk_fraction
                pip_value_per_lot = info.point * (info.trade_contract_size or 100)
                sl_dollars_per_lot = stop_loss_pips * pip_value_per_lot
                if sl_dollars_per_lot <= 0:
                        return 0.01

                raw = risk_dollars / sl_dollars_per_lot
                capped = max(info.volume_min or 0.01, min(raw, info.volume_max or raw))
                step = info.volume_step or 0.01
                return round(round(capped / step) * step, 2)

    def _atr_percentile(self, symbol: str) -> float:
        rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M5, 0, 260)
        if rates is None or len(rates) < 40:
            return 0.5

        highs = [float(r["high"]) for r in rates]
        lows = [float(r["low"]) for r in rates]
        closes = [float(r["close"]) for r in rates]

        true_ranges = []
        for i in range(1, len(rates)):
            h = highs[i]
            l = lows[i]
            prev_c = closes[i - 1]
            tr = max(h - l, abs(h - prev_c), abs(l - prev_c))
            true_ranges.append(tr)

        period = 14
        if len(true_ranges) < period:
            return 0.5

        atr_series = []
        for i in range(period - 1, len(true_ranges)):
            window = true_ranges[i - period + 1 : i + 1]
            atr_series.append(sum(window) / len(window))

        if len(atr_series) < 5:
            return 0.5

        current = atr_series[-1]
        less_or_equal = sum(1 for x in atr_series if x <= current)
        return less_or_equal / len(atr_series)

    def _closed_trade_returns(self, limit: int) -> list[float]:
        if not self.journal.enabled:
            return []

        query = """
            SELECT pnl_r, pnl_usd
            FROM signal_journal
            WHERE is_filled = TRUE
              AND (pnl_r IS NOT NULL OR pnl_usd IS NOT NULL)
            ORDER BY COALESCE(signal_ts, journal_ts) DESC
            LIMIT %s
        """
        try:
            with self.journal._connect() as connection:
                with connection.cursor() as cursor:
                    cursor.execute(query, (limit,))
                    rows = cursor.fetchall()
        except Exception:
            return []

        values: list[float] = []
        for pnl_r, pnl_usd in rows:
            if pnl_r is not None:
                values.append(float(pnl_r))
            elif pnl_usd is not None:
                values.append(float(pnl_usd))
        return values

    def _kelly_risk_fraction(self, symbol: str) -> float:
        returns = self._closed_trade_returns(self.kelly_window)
        if len(returns) < self.kelly_min_trades:
            base_fraction = 0.01
        else:
            wins = [x for x in returns if x > 0]
            losses = [x for x in returns if x < 0]
            if not wins or not losses:
                base_fraction = 0.01
            else:
                w = len(wins) / len(returns)
                avg_win = sum(wins) / len(wins)
                avg_loss_abs = abs(sum(losses) / len(losses))
                if avg_loss_abs <= 0:
                    base_fraction = 0.01
                else:
                    r = avg_win / avg_loss_abs
                    if r <= 0:
                        base_fraction = 0.01
                    else:
                        kelly_full = w - ((1 - w) / r)
                        quarter_kelly = 0.25 * max(0.0, kelly_full)
                        base_fraction = min(0.02, max(0.0025, quarter_kelly))

        atr_pct = self._atr_percentile(symbol)
        vol_scale = max(0.5, min(1.25, 1.25 - atr_pct))
        adjusted = base_fraction * vol_scale
        return min(0.02, max(0.0025, adjusted))
