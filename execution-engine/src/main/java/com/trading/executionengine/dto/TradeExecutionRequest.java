package com.trading.executionengine.dto;

import com.fasterxml.jackson.annotation.JsonProperty;

public record TradeExecutionRequest(
        String symbol,
        String action,
        String timeframe,
        Double confidenceScore,
        @JsonProperty("intended_price") Double intendedPrice,
        @JsonProperty("slippage_tolerance_pips") Double slippageTolerancePips) {
}