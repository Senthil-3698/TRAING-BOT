import imaplib
import email
import asyncio
import os
import time
import redis
import json
import re

from orchestrator import on_signal_received

# Connect to our Local Redis Cache
r = redis.Redis(
    host=os.getenv("REDIS_HOST", "localhost"),
    port=int(os.getenv("REDIS_PORT", "6380")),
    db=0,
)

def parse_signal(subject, body):
    """
    Advanced Trader Logic: Extracts Asset, Timeframe, and Action.
    Example Subject: "Alert: XAUUSD 5m BUY"
    """
    # Look for common symbols and timeframes in the message
    symbol_match = re.search(r'(XAUUSD|EURUSD|BTCUSD)', subject.upper())
    tf_match = re.search(r'(\d+[m|h|d])', subject.lower())
    action_match = re.search(r'(BUY|SELL)', body.upper())

    if all([symbol_match, tf_match, action_match]):
        return {
            "symbol": symbol_match.group(1),
            "timeframe": tf_match.group(1),
            "action": action_match.group(1),
            "timestamp": time.time()
        }
    return None

def listen_for_alerts():
    # Use credentials from your .env later
    print("Agent Active: Listening for TradingView signals...")

    # Replace these with environment-backed secrets in production.
    imap_host = "imap.gmail.com"
    imap_user = "your_email@example.com"
    imap_password = "your_password"

    # Logic for checking inbox every 1 second
    while True:
        try:
            with imaplib.IMAP4_SSL(imap_host) as mail:
                mail.login(imap_user, imap_password)
                mail.select("INBOX")

                status, message_ids = mail.search(None, 'UNSEEN')
                if status != "OK":
                    time.sleep(1)
                    continue

                for message_id in message_ids[0].split():
                    fetch_status, message_data = mail.fetch(message_id, "(RFC822)")
                    if fetch_status != "OK" or not message_data or not message_data[0]:
                        continue

                    raw_message = message_data[0][1]
                    message = email.message_from_bytes(raw_message)
                    sender = email.utils.parseaddr(message.get("From", ""))[1].lower()

                    if sender != "noreply@tradingview.com":
                        continue

                    subject = message.get("Subject", "")
                    body_parts = []

                    if message.is_multipart():
                        for part in message.walk():
                            content_type = part.get_content_type()
                            disposition = str(part.get("Content-Disposition", ""))
                            if content_type == "text/plain" and "attachment" not in disposition:
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
                        mail.store(message_id, "+FLAGS", "\\Seen")
                        mail.store(message_id, "+FLAGS", "\\Deleted")

                mail.expunge()
        except imaplib.IMAP4.error as error:
            print(f"IMAP error: {error}")
        except Exception as error:
            print(f"Listener error: {error}")

        time.sleep(1) 

if __name__ == "__main__":
    listen_for_alerts()