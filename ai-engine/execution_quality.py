from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone
from typing import Any

import MetaTrader5 as mt5
import psycopg
from dotenv import load_dotenv

load_dotenv()


class ExecutionQualityMonitor:
    def __init__(self) -> None:
        self.db_host = os.getenv("JOURNAL_DB_HOST", os.getenv("MONITOR_DB_HOST", "localhost"))
        self.db_port = int(os.getenv("JOURNAL_DB_PORT", os.getenv("MONITOR_DB_PORT", "5433")))
        self.db_name = os.getenv("JOURNAL_DB_NAME", os.getenv("MONITOR_DB_NAME", "sentinel_db"))
        self.db_user = os.getenv("JOURNAL_DB_USER", os.getenv("MONITOR_DB_USER", "admin"))
        self.db_password = os.getenv("JOURNAL_DB_PASSWORD", os.getenv("MONITOR_DB_PASSWORD", "admin"))

        self.enabled = True
        try:
            self._ensure_table()
        except Exception as error:
            self.enabled = False
            print(f"[EXEC_QUALITY] Disabled, table init failed: {error}")

    def _connect(self):
        return psycopg.connect(
            host=self.db_host,
            port=self.db_port,
            dbname=self.db_name,
            user=self.db_user,
            password=self.db_password,
        )

    def _ensure_table(self) -> None:
        ddl = """
        CREATE TABLE IF NOT EXISTS execution_quality (
            id BIGSERIAL PRIMARY KEY,
            logged_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            source VARCHAR(32) NOT NULL,
            symbol VARCHAR(20) NOT NULL,
            action VARCHAR(10) NOT NULL,
            timeframe VARCHAR(10),
            order_ticket VARCHAR(64),
            deal_ticket VARCHAR(64),
            signal_ts TIMESTAMPTZ,
            order_send_ts TIMESTAMPTZ,
            fill_ts TIMESTAMPTZ,
            signal_to_send_ms DOUBLE PRECISION,
            send_to_fill_ms DOUBLE PRECISION,
            total_latency_ms DOUBLE PRECISION,
            intended_price DOUBLE PRECISION,
            fill_price DOUBLE PRECISION,
            slippage_points DOUBLE PRECISION,
            spread_points DOUBLE PRECISION,
            repaint_flag BOOLEAN,
            signal_bar_time TIMESTAMPTZ,
            signal_bar_relation VARCHAR(16),
            metadata JSONB
        );
        """
        with self._connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(ddl)
            connection.commit()

    def _parse_time(self, value: Any) -> datetime | None:
        if value is None:
            return None
        if isinstance(value, datetime):
            return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
        if isinstance(value, (int, float)):
            return datetime.fromtimestamp(float(value), tz=timezone.utc)
        if isinstance(value, str):
            try:
                return datetime.fromisoformat(value.replace("Z", "+00:00"))
            except ValueError:
                return None
        return None

    def _find_fill_from_history(self, order_ticket: int | None, deal_ticket: int | None, fallback_time: datetime) -> tuple[float | None, datetime | None]:
        start = fallback_time - timedelta(minutes=5)
        end = fallback_time + timedelta(minutes=5)
        deals = mt5.history_deals_get(start, end)
        if deals is None:
            return None, None

        for d in deals:
            if deal_ticket is not None and int(getattr(d, "ticket", -1)) == int(deal_ticket):
                fill_price = float(getattr(d, "price", 0.0))
                time_msc = int(getattr(d, "time_msc", 0))
                if time_msc > 0:
                    fill_ts = datetime.fromtimestamp(time_msc / 1000.0, tz=timezone.utc)
                else:
                    fill_ts = datetime.fromtimestamp(int(getattr(d, "time", 0)), tz=timezone.utc)
                return fill_price, fill_ts
            if order_ticket is not None and int(getattr(d, "order", -1)) == int(order_ticket):
                fill_price = float(getattr(d, "price", 0.0))
                time_msc = int(getattr(d, "time_msc", 0))
                if time_msc > 0:
                    fill_ts = datetime.fromtimestamp(time_msc / 1000.0, tz=timezone.utc)
                else:
                    fill_ts = datetime.fromtimestamp(int(getattr(d, "time", 0)), tz=timezone.utc)
                return fill_price, fill_ts

        return None, None

    def _timeframe_seconds(self, timeframe: str | None) -> int:
        if not timeframe:
            return 60
        tf = timeframe.strip().lower()
        if tf.endswith("m") and tf[:-1].isdigit():
            return int(tf[:-1]) * 60
        if tf.endswith("h") and tf[:-1].isdigit():
            return int(tf[:-1]) * 3600
        if tf.endswith("d") and tf[:-1].isdigit():
            return int(tf[:-1]) * 86400
        return 60

    def _detect_repaint_relation(self, signal_bar_time: datetime | None, timeframe: str | None, now: datetime) -> tuple[bool | None, str | None]:
        if signal_bar_time is None:
            return None, None

        tf_sec = self._timeframe_seconds(timeframe)
        epoch = int(now.timestamp())
        current_bar_open = epoch - (epoch % tf_sec)
        previous_bar_open = current_bar_open - tf_sec

        sig = int(signal_bar_time.timestamp())
        if sig >= current_bar_open:
            return False, "current"
        if sig >= previous_bar_open:
            return True, "previous"
        return None, "older"

    def record_fill(
        self,
        *,
        source: str,
        symbol: str,
        action: str,
        timeframe: str | None,
        order_ticket: int | None,
        deal_ticket: int | None,
        signal_ts: Any,
        signal_bar_time: Any,
        signal_bar_relation: str | None,
        order_send_ts: datetime,
        order_send_done_ts: datetime,
        intended_price: float,
        spread_points: float,
    ) -> None:
        if not self.enabled:
            return

        signal_ts_dt = self._parse_time(signal_ts)
        signal_bar_ts = self._parse_time(signal_bar_time)

        fill_price, fill_ts = self._find_fill_from_history(order_ticket, deal_ticket, order_send_done_ts)
        if fill_price is None:
            fill_price = intended_price
        if fill_ts is None:
            fill_ts = order_send_done_ts

        info = mt5.symbol_info(symbol)
        point = float(info.point) if info and info.point else 0.0
        if point > 0:
            if action.upper() == "BUY":
                slippage_points = (fill_price - intended_price) / point
            else:
                slippage_points = (intended_price - fill_price) / point
        else:
            slippage_points = 0.0

        signal_to_send_ms = None
        if signal_ts_dt is not None:
            signal_to_send_ms = (order_send_ts - signal_ts_dt).total_seconds() * 1000.0

        send_to_fill_ms = (fill_ts - order_send_ts).total_seconds() * 1000.0
        total_latency_ms = None
        if signal_ts_dt is not None:
            total_latency_ms = (fill_ts - signal_ts_dt).total_seconds() * 1000.0

        repaint_flag, detected_relation = self._detect_repaint_relation(signal_bar_ts, timeframe, order_send_ts)
        relation = signal_bar_relation or detected_relation

        warn = abs(slippage_points) > 5 or send_to_fill_ms > 500
        if warn:
            print(
                "[EXEC_QUALITY_WARN]"
                f" symbol={symbol} action={action} slippage_points={slippage_points:.2f}"
                f" send_to_fill_ms={send_to_fill_ms:.1f}"
            )

        stmt = """
            INSERT INTO execution_quality (
                source, symbol, action, timeframe,
                order_ticket, deal_ticket,
                signal_ts, order_send_ts, fill_ts,
                signal_to_send_ms, send_to_fill_ms, total_latency_ms,
                intended_price, fill_price, slippage_points, spread_points,
                repaint_flag, signal_bar_time, signal_bar_relation, metadata
            ) VALUES (
                %s, %s, %s, %s,
                %s, %s,
                %s, %s, %s,
                %s, %s, %s,
                %s, %s, %s, %s,
                %s, %s, %s, %s
            )
        """

        payload = {
            "warning": warn,
            "warning_thresholds": {"slippage_points": 5, "latency_ms": 500},
        }

        try:
            with self._connect() as connection:
                with connection.cursor() as cursor:
                    cursor.execute(
                        stmt,
                        (
                            source,
                            symbol,
                            action,
                            timeframe,
                            str(order_ticket) if order_ticket is not None else None,
                            str(deal_ticket) if deal_ticket is not None else None,
                            signal_ts_dt,
                            order_send_ts,
                            fill_ts,
                            signal_to_send_ms,
                            send_to_fill_ms,
                            total_latency_ms,
                            intended_price,
                            fill_price,
                            slippage_points,
                            spread_points,
                            repaint_flag,
                            signal_bar_ts,
                            relation,
                            psycopg.types.json.Jsonb(payload),
                        ),
                    )
                connection.commit()
        except Exception as error:
            print(f"[EXEC_QUALITY] Failed to record quality row: {error}")
