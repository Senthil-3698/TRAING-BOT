package com.trading.executionengine.service;

import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;
import redis.clients.jedis.JedisPooled;

@Service
public class RiskManagerService {

    private static final double RISK_PERCENT = 0.01;
    private static final String GLOBAL_KILL_SWITCH_KEY = "GLOBAL_KILL_SWITCH";

    private final JedisPooled jedis;

    public RiskManagerService(
            @Value("${redis.host:localhost}") String redisHost,
            @Value("${redis.port:6380}") int redisPort) {
        this.jedis = new JedisPooled(redisHost, redisPort);
    }

    public double calculatePositionSize(double accountBalance, double stopLossPips, String symbol) {
        if (accountBalance <= 0 || stopLossPips <= 0 || symbol == null || symbol.isBlank()) {
            return 0;
        }

        // Global kill switch vetoes all trading when activated.
        String killSwitchStatus = jedis.get(GLOBAL_KILL_SWITCH_KEY);
        if ("ACTIVE".equalsIgnoreCase(killSwitchStatus)) {
            return 0;
        }

        double riskAmount = accountBalance * RISK_PERCENT;
        double pipValuePerLot = getPipValuePerLot(symbol);

        if (pipValuePerLot <= 0) {
            return 0;
        }

        // Position sizing formula: lots = max risk / (stop loss in pips * pip value per lot)
        double rawLots = riskAmount / (stopLossPips * pipValuePerLot);
        return roundToTwoDecimals(Math.max(0, rawLots));
    }

    public boolean isSpreadSafe(double currentSpread, double maxAllowed) {
        if (currentSpread < 0 || maxAllowed < 0) {
            return false;
        }
        return currentSpread <= maxAllowed;
    }

    private double getPipValuePerLot(String symbol) {
        String normalizedSymbol = symbol.trim().toUpperCase();

        // XAUUSD usually uses a distinct tick/pip value vs major FX pairs.
        if ("XAUUSD".equals(normalizedSymbol)) {
            return 1.0;
        }

        // Approximation for USD-quoted major FX pairs at 1 standard lot.
        return 10.0;
    }

    private double roundToTwoDecimals(double value) {
        return Math.round(value * 100.0) / 100.0;
    }
}
