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
