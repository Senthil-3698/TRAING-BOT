package com.trading.executionengine.dto;

public record TradeEventRequest(
        String ticketId,
        String eventType) {
}