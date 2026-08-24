"""Order Update Service: stream CSV rows and POST valid events."""

from __future__ import annotations

import argparse
import csv
import logging
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Iterator, Optional

import httpx

from src.event import OrderEvent, parse_event

REQUIRED_COLUMNS = ("event_id", "symbol", "transaction_type", "quantity")
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8000
DEFAULT_RATE_LIMIT = 50
DEFAULT_TIMEOUT = 5.0
COMPLETION_MESSAGE = "Input processing complete"

logger = logging.getLogger("order_update_service")


@dataclass(frozen=True)
class Settings:
    input_file: str
    host: str
    port: int
    rate_limit: int
    timeout: float


@dataclass(frozen=True)
class SendResult:
    delivered: bool
    status: Optional[str] = None
    error: Optional[str] = None


def send_interval_seconds(rate_limit: int) -> float:
    """Minimum pause between sent events for the configured max rate."""
    if rate_limit <= 0:
        raise ValueError("rate_limit must be a positive integer")
    return 1.0 / rate_limit


def event_payload(event: OrderEvent) -> dict[str, object]:
    return {
        "event_id": event.event_id,
        "symbol": event.symbol,
        "transaction_type": event.transaction_type,
        "quantity": event.quantity,
    }


def parse_csv_row(row: dict) -> tuple[Optional[OrderEvent], Optional[str]]:
    result = parse_event(
        row.get("event_id"),
        row.get("symbol"),
        row.get("transaction_type"),
        row.get("quantity"),
    )
    if result.ok:
        return result.event, None
    return None, result.error


def missing_columns(fieldnames: Optional[Iterable[str]]) -> list[str]:
    names = set(fieldnames or [])
    return [column for column in REQUIRED_COLUMNS if column not in names]


def iter_csv_rows(path: str | Path) -> Iterator[dict[str, str]]:
    """Yield CSV data rows one at a time. Does not load the file into a list."""
    csv_path = Path(path)
    if not csv_path.is_file():
        raise FileNotFoundError(f"Input file not found: {csv_path}")

    with csv_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        missing = missing_columns(reader.fieldnames)
        if missing:
            raise ValueError(f"CSV is missing required columns: {', '.join(missing)}")
        for row in reader:
            yield row


class EventClient:
    """POST validated events to the Position Maintaining Service."""

    def __init__(
        self,
        host: str,
        port: int,
        timeout: float,
        post: Optional[Callable[..., httpx.Response]] = None,
    ) -> None:
        self.url = f"http://{host}:{port}/events"
        self.timeout = timeout
        self._post = post
        self._client: Optional[httpx.Client] = None
        if post is None:
            # Reuse one connection and ignore HTTP_PROXY for localhost delivery.
            self._client = httpx.Client(timeout=timeout, trust_env=False)

    def close(self) -> None:
        if self._client is not None:
            self._client.close()
            self._client = None

    def __enter__(self) -> EventClient:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    def send(self, event: OrderEvent) -> SendResult:
        try:
            if self._post is not None:
                response = self._post(
                    self.url, json=event_payload(event), timeout=self.timeout
                )
            else:
                assert self._client is not None
                response = self._client.post(self.url, json=event_payload(event))
        except httpx.RequestError as exc:
            return SendResult(delivered=False, error=str(exc))

        if response.status_code >= 400:
            return SendResult(
                delivered=False,
                error=f"HTTP {response.status_code}: {response.text}",
            )

        try:
            body = response.json()
        except Exception:
            return SendResult(delivered=False, error="unexpected HTTP response body")

        if not isinstance(body, dict):
            return SendResult(delivered=False, error="unexpected HTTP response body")

        status = body.get("status")
        if status in ("accepted", "duplicate"):
            return SendResult(delivered=True, status=status)
        return SendResult(delivered=False, error=f"unexpected HTTP response body: {body}")


def process_rows(
    rows: Iterable[dict],
    send: Callable[[OrderEvent], SendResult],
    *,
    interval: float,
    sleeper: Callable[[float], None] = time.sleep,
) -> None:
    """Validate rows in order and send accepted events sequentially."""
    sent_count = 0
    for row_number, row in enumerate(rows, start=1):
        event, error = parse_csv_row(row)
        event_id = _row_event_id(row)
        if event is None:
            logger.warning(
                "row=%s event_id=%s rejected: %s",
                row_number,
                event_id,
                error,
            )
            continue

        logger.info("event_id=%s accepted", event.event_id)

        if sent_count > 0 and interval > 0:
            sleeper(interval)
        sent_count += 1

        try:
            result = send(event)
        except Exception as exc:
            logger.error("event_id=%s delivery failed: %s", event.event_id, exc)
            continue

        if not result.delivered:
            logger.error(
                "event_id=%s delivery failed: %s",
                event.event_id,
                result.error,
            )
            continue

        if result.status == "duplicate":
            logger.info(
                "event_id=%s delivered but ignored as duplicate",
                event.event_id,
            )
        else:
            logger.info("event_id=%s successfully sent", event.event_id)

    logger.info(COMPLETION_MESSAGE)


def run(
    settings: Settings,
    send: Optional[Callable[[OrderEvent], SendResult]] = None,
    sleeper: Callable[[float], None] = time.sleep,
) -> None:
    interval = send_interval_seconds(settings.rate_limit)
    rows = iter_csv_rows(settings.input_file)
    if send is not None:
        process_rows(rows, send, interval=interval, sleeper=sleeper)
        return
    with EventClient(settings.host, settings.port, settings.timeout) as client:
        process_rows(rows, client.send, interval=interval, sleeper=sleeper)


def load_settings(argv: Optional[list[str]] = None) -> Settings:
    parser = argparse.ArgumentParser(description="Order Update Service")
    parser.add_argument(
        "--input-file",
        default=os.environ.get("ORDER_UPDATE_INPUT_FILE"),
        help="Path to the order updates CSV (or ORDER_UPDATE_INPUT_FILE)",
    )
    parser.add_argument(
        "--host",
        default=os.environ.get("POSITION_SERVICE_HOST", DEFAULT_HOST),
        help=f"Position Maintaining Service host (default: {DEFAULT_HOST})",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.environ.get("POSITION_SERVICE_PORT", str(DEFAULT_PORT))),
        help=f"Position Maintaining Service port (default: {DEFAULT_PORT})",
    )
    parser.add_argument(
        "--rate-limit",
        type=int,
        default=int(os.environ.get("ORDER_UPDATE_RATE_LIMIT", str(DEFAULT_RATE_LIMIT))),
        help=f"Max events sent per second (default: {DEFAULT_RATE_LIMIT})",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=float(os.environ.get("ORDER_UPDATE_HTTP_TIMEOUT", str(DEFAULT_TIMEOUT))),
        help=f"HTTP timeout in seconds (default: {DEFAULT_TIMEOUT})",
    )
    args = parser.parse_args(argv)
    if not args.input_file:
        parser.error("--input-file is required (or set ORDER_UPDATE_INPUT_FILE)")
    if args.rate_limit <= 0:
        parser.error("--rate-limit must be a positive integer")
    if args.timeout <= 0:
        parser.error("--timeout must be greater than 0")
    return Settings(
        input_file=args.input_file,
        host=args.host,
        port=args.port,
        rate_limit=args.rate_limit,
        timeout=args.timeout,
    )


def _row_event_id(row: dict) -> str:
    value = row.get("event_id")
    if isinstance(value, str) and value.strip() != "":
        return value
    return "<blank>"


def main(argv: Optional[list[str]] = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    try:
        settings = load_settings(argv)
        run(settings)
    except FileNotFoundError as exc:
        logger.error("%s", exc)
        return 1
    except ValueError as exc:
        logger.error("%s", exc)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
