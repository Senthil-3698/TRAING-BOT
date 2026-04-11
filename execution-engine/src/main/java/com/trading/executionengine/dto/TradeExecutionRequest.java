package com.trading.executionengine.dto;

public record TradeExecutionRequest(
        String symbol,
        String action,
        String timeframe,
        Double confidenceScore) {
}