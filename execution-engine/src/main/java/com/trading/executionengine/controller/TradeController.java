package com.trading.executionengine.controller;

import com.trading.executionengine.dto.TradeExecutionRequest;
import com.trading.executionengine.dto.TradeEventRequest;
import com.trading.executionengine.entity.Trade;
import com.trading.executionengine.entity.TradeEvent;
import com.trading.executionengine.repository.TradeRepository;
import com.trading.executionengine.repository.TradeEventRepository;
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RestController;

import java.io.IOException;
import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.math.BigDecimal;
import java.time.Duration;
import java.time.Instant;
import java.util.UUID;

@RestController
public class TradeController {

    private static final double DEFAULT_STOP_LOSS_PIPS = 50.0;

    private final TradeRepository tradeRepository;
    private final TradeEventRepository tradeEventRepository;
    private final String mt5BridgeUrl;
    private final ObjectMapper objectMapper = new ObjectMapper();

    public TradeController(
            TradeRepository tradeRepository,
            TradeEventRepository tradeEventRepository,
            @Value("${mt5.bridge.url:http://localhost:9000/execute}") String mt5BridgeUrl) {
        this.tradeRepository = tradeRepository;
        this.tradeEventRepository = tradeEventRepository;
        this.mt5BridgeUrl = mt5BridgeUrl;
    }

    @PostMapping("/execute")
    public ResponseEntity<String> executeTrade(@RequestBody TradeExecutionRequest request) {
        if (request == null
                || request.symbol() == null || request.symbol().isBlank()
                || request.action() == null || request.action().isBlank()
                || request.timeframe() == null || request.timeframe().isBlank()
                || request.confidenceScore() == null) {
            return ResponseEntity.status(HttpStatus.FORBIDDEN).body("Risk Manager rejected trade.");
        }

        Trade trade = new Trade();
        trade.setId(UUID.randomUUID());
        trade.setSymbol(request.symbol().toUpperCase());
        trade.setAction(request.action().toUpperCase());
        trade.setTimeframe(request.timeframe().toLowerCase());
        trade.setConfidenceScore(BigDecimal.valueOf(request.confidenceScore()));
        trade.setEntryPrice(BigDecimal.ZERO);
        trade.setStopLoss(BigDecimal.ZERO);
        trade.setTakeProfit(BigDecimal.ZERO);
        trade.setStatus("PENDING_EXECUTION");

        tradeRepository.save(trade);

        // Authoritative risk and sizing now run in Python RiskEngine (broker bridge).
        String bridgePayload = String.format(
            "{\"symbol\":\"%s\",\"action\":\"%s\",\"timeframe\":\"%s\",\"volume\":%.2f,\"stop_loss_pips\":%.2f,\"take_profit_pips\":%.2f,\"confidence_score\":%.4f}",
            request.symbol().toUpperCase(),
            request.action().toUpperCase(),
            request.timeframe().toLowerCase(),
            0.01,
            DEFAULT_STOP_LOSS_PIPS,
            DEFAULT_STOP_LOSS_PIPS * 2,
            request.confidenceScore()
        );

        try {
            HttpClient client = HttpClient.newBuilder()
                .connectTimeout(Duration.ofSeconds(5))
                .build();

            HttpRequest bridgeRequest = HttpRequest.newBuilder()
                .uri(URI.create(mt5BridgeUrl))
                .timeout(Duration.ofSeconds(10))
                .header("Content-Type", "application/json")
                .POST(HttpRequest.BodyPublishers.ofString(bridgePayload))
                .build();

            HttpResponse<String> bridgeResponse = client.send(bridgeRequest, HttpResponse.BodyHandlers.ofString());

            if (bridgeResponse.statusCode() != 200) {
            return ResponseEntity.status(HttpStatus.BAD_GATEWAY)
                .body("Trade queued but MT5 bridge rejected execution: " + bridgeResponse.body());
            }

            try {
                JsonNode bridgeResult = objectMapper.readTree(bridgeResponse.body());
                JsonNode orderNode = bridgeResult.get("order");
                if (orderNode != null && !orderNode.isNull()) {
                    TradeEvent event = new TradeEvent();
                    event.setId(UUID.randomUUID());
                    event.setTicketId(orderNode.asText());
                    event.setEventType("ENTRY");
                    event.setTimestamp(Instant.now());
                    tradeEventRepository.save(event);
                }
            } catch (Exception parseError) {
                // Entry event logging is observability only; do not block live execution.
            }

            return ResponseEntity.ok(bridgeResponse.body());
        } catch (InterruptedException error) {
            Thread.currentThread().interrupt();
            return ResponseEntity.status(HttpStatus.BAD_GATEWAY)
                    .body("Trade queued but MT5 bridge call failed: " + error.getMessage());
        } catch (IOException error) {
            return ResponseEntity.status(HttpStatus.BAD_GATEWAY)
                .body("Trade queued but MT5 bridge call failed: " + error.getMessage());
        }
    }

    @PostMapping("/events")
    public ResponseEntity<String> logTradeEvent(@RequestBody TradeEventRequest request) {
        if (request == null
                || request.ticketId() == null || request.ticketId().isBlank()
                || request.eventType() == null || request.eventType().isBlank()) {
            return ResponseEntity.status(HttpStatus.BAD_REQUEST).body("Invalid trade event payload.");
        }

        String eventType = request.eventType().trim().toUpperCase();
        if (!eventType.equals("ENTRY") && !eventType.equals("BE") && !eventType.equals("PARTIAL_CLOSE")) {
            return ResponseEntity.status(HttpStatus.BAD_REQUEST).body("Unsupported event type.");
        }

        TradeEvent event = new TradeEvent();
        event.setId(UUID.randomUUID());
        event.setTicketId(request.ticketId().trim());
        event.setEventType(eventType);
        event.setTimestamp(Instant.now());

        tradeEventRepository.save(event);
        return ResponseEntity.ok("Trade event logged.");
    }
}