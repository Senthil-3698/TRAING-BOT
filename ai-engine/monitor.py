import imaplib
import os
from datetime import datetime, timezone

import MetaTrader5 as mt5
import psycopg
import redis
from dotenv import load_dotenv
from rich import box
from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.table import Table

load_dotenv()

console = Console()

REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6380"))
DB_HOST = os.getenv("MONITOR_DB_HOST", "localhost")
DB_PORT = int(os.getenv("MONITOR_DB_PORT", "5433"))
DB_NAME = os.getenv("MONITOR_DB_NAME", "sentinel_db")
DB_USER = os.getenv("MONITOR_DB_USER", "admin")
DB_PASSWORD = os.getenv("MONITOR_DB_PASSWORD", "admin")
MT5_LOGIN = os.getenv("MT5_LOGIN")
MT5_PASS = os.getenv("MT5_PASS")
MT5_SERVER = os.getenv("MT5_SERVER")
EMAIL_USER = os.getenv("EMAIL_USER")
EMAIL_PASS = os.getenv("EMAIL_PASS")

r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=0)


def check_ear_health():
    if not EMAIL_USER or not EMAIL_PASS:
        return False, "Missing email credentials"

    try:
        with imaplib.IMAP4_SSL("imap.gmail.com") as mailbox:
            mailbox.login(EMAIL_USER, EMAIL_PASS)
            mailbox.logout()
        return True, "IMAP connected"
    except Exception as error:
        return False, str(error)


def check_muscle_health():
    if not MT5_LOGIN or not MT5_PASS or not MT5_SERVER:
        return False, "Missing MT5 credentials"

    try:
        connected = mt5.initialize(
            login=int(MT5_LOGIN),
            password=MT5_PASS,
            server=MT5_SERVER,
        )
        if not connected:
            return False, f"MT5 init failed: {mt5.last_error()}"

        account = mt5.account_info()
        terminal = mt5.terminal_info()
        mt5.shutdown()

        if account is None or terminal is None:
            return False, "MT5 terminal/account unavailable"

        return True, f"MT5 connected ({account.balance:.2f})"
    except Exception as error:
        try:
            mt5.shutdown()
        except Exception:
            pass
        return False, str(error)


def fetch_trade_rows():
    rows = []
    mt5_connected = False
    if MT5_LOGIN and MT5_PASS and MT5_SERVER:
        mt5_connected = mt5.initialize(
            login=int(MT5_LOGIN),
            password=MT5_PASS,
            server=MT5_SERVER,
        )

    connection = psycopg.connect(
        host=DB_HOST,
        port=DB_PORT,
        dbname=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD,
    )

    try:
        with connection.cursor() as cursor:
            try:
                cursor.execute(
                    """
                    SELECT DISTINCT ON (ticket_id)
                        ticket_id,
                        event_type,
                        timestamp
                    FROM trade_events
                    ORDER BY ticket_id, timestamp DESC
                    """
                )
                latest_events = cursor.fetchall()
            except Exception:
                latest_events = []

        if mt5_connected:
            positions = mt5.positions_get(magic=123456) or []
        else:
            positions = []
        position_map = {str(position.ticket): position for position in positions}

        for ticket_id, event_type, event_timestamp in latest_events:
            ticket_key = str(ticket_id)
            position = position_map.get(ticket_key)
            trade_state_raw = r.get(f"trade:{ticket_key}")
            stage = "ENTRY"
            if trade_state_raw:
                try:
                    import json

                    stage = json.loads(trade_state_raw.decode("utf-8")).get("stage", "ENTRY")
                except Exception:
                    stage = r.get(f"trade_stage:{ticket_key}")
                    stage = stage.decode("utf-8") if stage else "ENTRY"
            else:
                stage_raw = r.get(f"trade_stage:{ticket_key}")
                stage = stage_raw.decode("utf-8") if stage_raw else "ENTRY"

            symbol = position.symbol if position else "-"
            profit = f"{position.profit:.2f}" if position else "0.00"
            status = stage if stage else event_type

            rows.append(
                {
                    "ticket": ticket_key,
                    "symbol": symbol,
                    "status": status,
                    "profit": profit,
                    "event_type": event_type,
                    "timestamp": event_timestamp,
                }
            )
    finally:
        connection.close()
        if mt5_connected:
            mt5.shutdown()

    return rows


def render_dashboard():
    ear_ok, ear_message = check_ear_health()
    muscle_ok, muscle_message = check_muscle_health()

    health_table = Table.grid(expand=True)
    health_table.add_column(justify="left")
    health_table.add_column(justify="left")
    health_table.add_row(
        f"[bold green]Ear[/bold green]" if ear_ok else f"[bold red]Ear[/bold red]",
        ear_message,
    )
    health_table.add_row(
        f"[bold green]Muscle[/bold green]" if muscle_ok else f"[bold red]Muscle[/bold red]",
        muscle_message,
    )

    trade_table = Table(box=box.SIMPLE_HEAVY, expand=True)
    trade_table.add_column("Ticket ID", style="bold")
    trade_table.add_column("Symbol")
    trade_table.add_column("Status")
    trade_table.add_column("Current Profit", justify="right")

    try:
        trade_rows = fetch_trade_rows()
    except Exception as error:
        trade_rows = []
        trade_table.add_row("-", "-", f"DB error: {error}", "-")

    if not trade_rows:
        trade_table.add_row("-", "-", "No tracked trades yet", "0.00")
    else:
        for row in trade_rows:
            status = row["status"]
            if status == "PARTIAL_CLOSED":
                status_style = "bold green"
            elif status == "BREAKEVEN":
                status_style = "bold yellow"
            elif status == "ENTRY":
                status_style = "bold blue"
            else:
                status_style = "white"

            trade_table.add_row(
                row["ticket"],
                row["symbol"],
                f"[{status_style}]{status}[/{status_style}]",
                row["profit"],
            )

    return Panel(
        health_table,
        title="System Health",
        subtitle="Ear / Muscle / Redis / PostgreSQL",
        border_style="cyan",
    ), trade_table


def main():
    with Live(refresh_per_second=1, screen=True) as live:
        while True:
            health_panel, trade_table = render_dashboard()
            root = Table.grid(expand=True)
            root.add_row(health_panel)
            root.add_row(trade_table)
            live.update(root)
            import time

            time.sleep(2)


if __name__ == "__main__":
    main()