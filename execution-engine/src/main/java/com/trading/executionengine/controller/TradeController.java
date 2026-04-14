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

    private static final double MIN_CONFIDENCE_SCORE = 70.0;
    private static final int MAX_BRIDGE_RETRIES = 3;
    private static final long RETRY_BASE_BACKOFF_MS = 200L;
    private static final long LATENCY_ALERT_MS = 500L;

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

        if (request.confidenceScore() <= MIN_CONFIDENCE_SCORE) {
            return ResponseEntity.status(HttpStatus.FORBIDDEN)
                    .body("Signal rejected: confidence score must be greater than 70.");
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

        String optionalFields = "";
        if (request.intendedPrice() != null) {
            optionalFields += String.format(",\"intended_price\":%.10f", request.intendedPrice());
        }
        if (request.slippageTolerancePips() != null) {
            optionalFields += String.format(",\"slippage_tolerance_pips\":%.4f", request.slippageTolerancePips());
        }
        String bridgePayload = String.format(
            "{\"symbol\":\"%s\",\"action\":\"%s\",\"timeframe\":\"%s\",\"volume\":%.2f,\"confidence_score\":%.4f%s}",
            request.symbol().toUpperCase(),
            request.action().toUpperCase(),
            request.timeframe().toLowerCase(),
            0.01,
            request.confidenceScore(),
            optionalFields
        );

        HttpClient client = HttpClient.newBuilder()
            .connectTimeout(Duration.ofSeconds(5))
            .build();

        HttpRequest bridgeRequest = HttpRequest.newBuilder()
            .uri(URI.create(mt5BridgeUrl))
            .timeout(Duration.ofSeconds(10))
            .header("Content-Type", "application/json")
            .POST(HttpRequest.BodyPublishers.ofString(bridgePayload))
            .build();

        HttpResponse<String> bridgeResponse = null;
        int attempts = 0;
        IOException lastIoError = null;
        InterruptedException lastInterrupted = null;

        for (int attempt = 1; attempt <= MAX_BRIDGE_RETRIES; attempt++) {
            attempts = attempt;
            Instant sendStart = Instant.now();
            try {
                bridgeResponse = client.send(bridgeRequest, HttpResponse.BodyHandlers.ofString());
            } catch (IOException ioError) {
                lastIoError = ioError;
            } catch (InterruptedException interruptedError) {
                lastInterrupted = interruptedError;
                Thread.currentThread().interrupt();
                break;
            }

            long latencyMs = Duration.between(sendStart, Instant.now()).toMillis();
            System.out.printf(
                "[EXECUTION_LATENCY] symbol=%s action=%s attempt=%d latency_ms=%d%n",
                request.symbol(),
                request.action(),
                attempt,
                latencyMs
            );
            if (latencyMs > LATENCY_ALERT_MS) {
                System.out.printf(
                    "[EXECUTION_LATENCY_ALERT] symbol=%s action=%s attempt=%d latency_ms=%d threshold_ms=%d%n",
                    request.symbol(),
                    request.action(),
                    attempt,
                    latencyMs,
                    LATENCY_ALERT_MS
                );
            }

            if (bridgeResponse != null && bridgeResponse.statusCode() == 200) {
                break;
            }

            boolean retriableStatus = bridgeResponse != null
                    && (bridgeResponse.statusCode() >= 500 || bridgeResponse.statusCode() == 429);
            if (attempt < MAX_BRIDGE_RETRIES && retriableStatus) {
                long backoffMs = RETRY_BASE_BACKOFF_MS * (1L << (attempt - 1));
                System.out.printf(
                    "[BRIDGE_RETRY] symbol=%s action=%s attempt=%d status=%d backoff_ms=%d%n",
                    request.symbol(),
                    request.action(),
                    attempt,
                    bridgeResponse.statusCode(),
                    backoffMs
                );
                try {
                    Thread.sleep(backoffMs);
                } catch (InterruptedException interruptedSleep) {
                    Thread.currentThread().interrupt();
                    break;
                }
            } else {
                break;
            }
        }

        if (bridgeResponse == null) {
            if (lastInterrupted != null) {
                return ResponseEntity.status(HttpStatus.BAD_GATEWAY)
                        .body("Trade queued but MT5 bridge call interrupted.");
            }
            String message = lastIoError != null ? lastIoError.getMessage() : "No response from bridge.";
            return ResponseEntity.status(HttpStatus.BAD_GATEWAY)
                    .body("Trade queued but MT5 bridge call failed after retries: " + message);
        }

        if (bridgeResponse.statusCode() != 200) {
            return ResponseEntity.status(HttpStatus.BAD_GATEWAY)
                .body("Trade queued but MT5 bridge rejected execution after " + attempts + " attempts: " + bridgeResponse.body());
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