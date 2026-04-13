CREATE TABLE IF NOT EXISTS trades (
    id UUID PRIMARY KEY,
    symbol VARCHAR(20) NOT NULL,
    action VARCHAR(10) NOT NULL,
    timeframe VARCHAR(10) NOT NULL,
    confidence_score NUMERIC(10, 4) NOT NULL,
    entry_price NUMERIC(18, 8) NOT NULL DEFAULT 0,
    stop_loss NUMERIC(18, 8) NOT NULL DEFAULT 0,
    take_profit NUMERIC(18, 8) NOT NULL DEFAULT 0,
    status VARCHAR(20) NOT NULL CHECK (status IN ('PENDING_EXECUTION', 'OPEN', 'CLOSED'))
);

CREATE TABLE IF NOT EXISTS trade_events (
    id UUID PRIMARY KEY,
    ticket_id VARCHAR(64) NOT NULL,
    event_type VARCHAR(20) NOT NULL CHECK (event_type IN ('ENTRY', 'BE', 'PARTIAL_CLOSE')),
    timestamp TIMESTAMP NOT NULL
);

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
