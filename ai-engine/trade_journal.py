from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Any

import psycopg
from dotenv import load_dotenv

load_dotenv()


class TradeJournal:
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
            print(f"[JOURNAL] Disabled, could not initialize table: {error}")

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
        CREATE TABLE IF NOT EXISTS signal_journal (
            id BIGSERIAL PRIMARY KEY,
            journal_ts TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            signal_ts TIMESTAMPTZ,
            source VARCHAR(32) NOT NULL DEFAULT 'unknown',
            symbol VARCHAR(20) NOT NULL,
            action VARCHAR(10) NOT NULL,
            timeframe VARCHAR(10),

            rsi_value DOUBLE PRECISION,
            ema_distance DOUBLE PRECISION,
            atr_value DOUBLE PRECISION,
            m5_trend VARCHAR(20),
            h1_bias VARCHAR(20),
            h4_bias VARCHAR(20),
            integrated_bias VARCHAR(20),

            news_context TEXT,
            ai_decision VARCHAR(20),
            ai_reasoning TEXT,
            ai_confidence DOUBLE PRECISION,
            decision_status VARCHAR(20) NOT NULL,
            rejection_reason TEXT,

            is_filled BOOLEAN NOT NULL DEFAULT FALSE,
            order_ticket VARCHAR(64),
            entry_price DOUBLE PRECISION,
            stop_loss DOUBLE PRECISION,
            take_profit DOUBLE PRECISION,
            exit_price DOUBLE PRECISION,
            exit_reason TEXT,
            pnl_usd DOUBLE PRECISION,
            pnl_r DOUBLE PRECISION,

            metadata JSONB
        );
        """

        with self._connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(ddl)
            connection.commit()

    def log_signal(
        self,
        *,
        source: str,
        symbol: str,
        action: str,
        timeframe: str | None = None,
        signal_ts: datetime | None = None,
        rsi_value: float | None = None,
        ema_distance: float | None = None,
        atr_value: float | None = None,
        m5_trend: str | None = None,
        h1_bias: str | None = None,
        h4_bias: str | None = None,
        integrated_bias: str | None = None,
        news_context: str | None = None,
        ai_decision: str | None = None,
        ai_reasoning: str | None = None,
        ai_confidence: float | None = None,
        decision_status: str = "REJECTED",
        rejection_reason: str | None = None,
        is_filled: bool = False,
        order_ticket: str | None = None,
        entry_price: float | None = None,
        stop_loss: float | None = None,
        take_profit: float | None = None,
        exit_price: float | None = None,
        exit_reason: str | None = None,
        pnl_usd: float | None = None,
        pnl_r: float | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        if not self.enabled:
            return

        stmt = """
        INSERT INTO signal_journal (
            signal_ts, source, symbol, action, timeframe,
            rsi_value, ema_distance, atr_value, m5_trend, h1_bias, h4_bias, integrated_bias,
            news_context, ai_decision, ai_reasoning, ai_confidence, decision_status, rejection_reason,
            is_filled, order_ticket, entry_price, stop_loss, take_profit, exit_price, exit_reason, pnl_usd, pnl_r, metadata
        ) VALUES (
            %s, %s, %s, %s, %s,
            %s, %s, %s, %s, %s, %s, %s,
            %s, %s, %s, %s, %s, %s,
            %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
        )
        """

        signal_ts_value = signal_ts or datetime.now(timezone.utc)
        metadata_value = metadata if metadata is not None else {}

        try:
            with self._connect() as connection:
                with connection.cursor() as cursor:
                    cursor.execute(
                        stmt,
                        (
                            signal_ts_value,
                            source,
                            symbol,
                            action,
                            timeframe,
                            rsi_value,
                            ema_distance,
                            atr_value,
                            m5_trend,
                            h1_bias,
                            h4_bias,
                            integrated_bias,
                            news_context,
                            ai_decision,
                            ai_reasoning,
                            ai_confidence,
                            decision_status,
                            rejection_reason,
                            is_filled,
                            order_ticket,
                            entry_price,
                            stop_loss,
                            take_profit,
                            exit_price,
                            exit_reason,
                            pnl_usd,
                            pnl_r,
                            psycopg.types.json.Jsonb(metadata_value),
                        ),
                    )
                connection.commit()
        except Exception as error:
            print(f"[JOURNAL] Failed to persist signal: {error}")
