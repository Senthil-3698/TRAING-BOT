from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Any

import httpx
from dotenv import load_dotenv

load_dotenv()


TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "").strip()
TELEGRAM_ENABLED = bool(TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID)


def send_telegram_alert(event_type: str, message: str, level: str = "INFO", extra: dict[str, Any] | None = None) -> bool:
    """Best-effort Telegram alert sender. Returns True if sent, False otherwise."""
    if not TELEGRAM_ENABLED:
        return False

    payload_lines = [
        f"[{level}] {event_type}",
        message,
        f"time={datetime.now(timezone.utc).isoformat()}",
    ]
    if extra:
        for key, value in extra.items():
            payload_lines.append(f"{key}={value}")

    text = "\n".join(payload_lines)
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    try:
        with httpx.Client(timeout=5.0) as client:
            response = client.post(url, json={"chat_id": TELEGRAM_CHAT_ID, "text": text})
        return response.status_code == 200
    except Exception:
        return False
