package com.trading.executionengine.repository;

import com.trading.executionengine.entity.TradeEvent;
import org.springframework.data.jpa.repository.JpaRepository;

import java.util.UUID;

public interface TradeEventRepository extends JpaRepository<TradeEvent, UUID> {
}