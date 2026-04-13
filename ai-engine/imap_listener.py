import imaplib
import email
import asyncio
import os
import time
import redis
import re
from pathlib import Path

from dotenv import load_dotenv

from orchestrator import on_signal_received

load_dotenv(Path(__file__).resolve().parent / ".env")

r = redis.Redis(
    host=os.getenv("REDIS_HOST", "localhost"),
    port=int(os.getenv("REDIS_PORT", "6380")),
    db=0,
)

POLL_INTERVAL = 5   # seconds between inbox checks (was 1 — reduce IMAP hammering)


def parse_signal(subject, body):
    """
    Extracts symbol, timeframe, and action from TradingView alert email.
    Body-first parsing is more reliable when TradingView subject formats change.
    """
    subject_upper = subject.upper()
    body_upper = body.upper()

    symbol_match = (
        re.search(r'(XAUUSD|EURUSD|BTCUSD)', subject_upper)
        or re.search(r'(XAUUSD|EURUSD|BTCUSD)', body_upper)
    )
    tf_match = (
        re.search(r'(\d+(?:m|h|d))', subject.lower())
        or re.search(r'(\d+(?:m|h|d))', body.lower())
    )

    if "BUY" in body_upper:
        action = "BUY"
    elif "SELL" in body_upper:
        action = "SELL"
    else:
        m = re.search(r'(BUY|SELL)', subject_upper)
        action = m.group(1) if m else None

    if symbol_match and tf_match and action:
        return {
            "symbol": symbol_match.group(1),
            "timeframe": tf_match.group(1),
            "action": action,
            "timestamp": time.time(),
        }
    return None


def _process_messages(mail):
    """
    Search for UNSEEN TradingView emails, process each, mark as read+deleted.
    Returns number of signals dispatched.
    """
    # Only fetch UNSEEN emails — prevents re-processing on reconnect
    status, message_ids = mail.search(None, '(UNSEEN FROM "noreply@tradingview.com")')
    if status != "OK" or not message_ids[0]:
        return 0

    dispatched = 0
    for msg_id in message_ids[0].split():
        fetch_status, msg_data = mail.fetch(msg_id, "(RFC822)")
        if fetch_status != "OK" or not msg_data or not msg_data[0]:
            continue

        raw = msg_data[0][1]
        message = email.message_from_bytes(raw)
        sender = email.utils.parseaddr(message.get("From", ""))[1].lower()

        # Double-check sender despite the IMAP search filter
        if sender != "noreply@tradingview.com":
            mail.store(msg_id, "+FLAGS", "\\Seen")
            continue

        subject = message.get("Subject", "")
        body_parts = []

        if message.is_multipart():
            for part in message.walk():
                if part.get_content_type() == "text/plain" and "attachment" not in str(
                    part.get("Content-Disposition", "")
                ):
                    payload = part.get_payload(decode=True)
                    if payload:
                        charset = part.get_content_charset() or "utf-8"
                        body_parts.append(payload.decode(charset, errors="ignore"))
        else:
            payload = message.get_payload(decode=True)
            if payload:
                charset = message.get_content_charset() or "utf-8"
                body_parts.append(payload.decode(charset, errors="ignore"))

        body = "\n".join(body_parts)
        signal = parse_signal(subject, body)

        if signal:
            asyncio.run(on_signal_received(signal))
            dispatched += 1

        # Mark read and delete regardless of whether signal was valid
        mail.store(msg_id, "+FLAGS", "\\Seen")
        mail.store(msg_id, "+FLAGS", "\\Deleted")

    mail.expunge()
    return dispatched


def listen_for_alerts():
    print("[IMAP] Agent Active: Listening for TradingView signals...")

    imap_host = os.getenv("IMAP_HOST", "imap.gmail.com")
    imap_user = os.getenv("EMAIL_USER", "")
    imap_password = os.getenv("EMAIL_PASS", "")

    if not imap_user or not imap_password:
        print("[IMAP] ERROR: EMAIL_USER / EMAIL_PASS missing in .env")
        return

    mail = None
    last_noop = 0.0

    while True:
        try:
            # (Re)connect if no live session
            if mail is None:
                mail = imaplib.IMAP4_SSL(imap_host)
                mail.login(imap_user, imap_password)
                mail.select("INBOX")
                print("[IMAP] Connected to Gmail inbox.")

            # Keep connection alive with NOOP every 60 s
            now = time.time()
            if now - last_noop > 60:
                mail.noop()
                last_noop = now

            count = _process_messages(mail)
            if count:
                print(f"[IMAP] Dispatched {count} signal(s).")

        except imaplib.IMAP4.error as error:
            print(f"[IMAP] Error: {error} — reconnecting...")
            try:
                mail.logout()
            except Exception:
                pass
            mail = None

        except Exception as error:
            print(f"[IMAP] Unexpected error: {error}")
            try:
                if mail:
                    mail.logout()
            except Exception:
                pass
            mail = None

        time.sleep(POLL_INTERVAL)


if __name__ == "__main__":
    listen_for_alerts()
