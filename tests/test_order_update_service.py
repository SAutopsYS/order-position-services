import logging
from pathlib import Path
from typing import Optional

import httpx
import pytest

from src.event import OrderEvent
from src.order_update_service import (
    COMPLETION_MESSAGE,
    EventClient,
    SendResult,
    Settings,
    event_payload,
    iter_csv_rows,
    load_settings,
    parse_csv_row,
    process_rows,
    run,
    send_interval_seconds,
)


def _write_csv(path: Path, body: str) -> Path:
    path.write_text(body, encoding="utf-8")
    return path


def _row(
    event_id: str = "evt-0001",
    symbol: str = "RELIANCE",
    transaction_type: str = "BUY",
    quantity: str = "90",
) -> dict[str, str]:
    return {
        "event_id": event_id,
        "symbol": symbol,
        "transaction_type": transaction_type,
        "quantity": quantity,
    }


class RecordingSender:
    def __init__(self, results: Optional[list[SendResult]] = None) -> None:
        self.events: list[OrderEvent] = []
        self.results = list(results or [])

    def send(self, event: OrderEvent) -> SendResult:
        self.events.append(event)
        if self.results:
            return self.results.pop(0)
        return SendResult(delivered=True, status="accepted")


def test_valid_csv_row_becomes_order_event() -> None:
    event, error = parse_csv_row(_row())

    assert error is None
    assert event == OrderEvent("evt-0001", "RELIANCE", "BUY", 90)


def test_invalid_row_is_rejected() -> None:
    event, error = parse_csv_row(_row(quantity="0"))

    assert event is None
    assert error == "quantity must be a positive integer"


def test_invalid_row_does_not_stop_later_valid_rows() -> None:
    sender = RecordingSender()
    rows = [
        _row("evt-0001", quantity="90"),
        _row("evt-0002", quantity="0"),
        _row("evt-0003", symbol="TCS", transaction_type="SELL", quantity="75"),
    ]

    process_rows(rows, sender.send, interval=0, sleeper=lambda _: None)

    assert [event.event_id for event in sender.events] == ["evt-0001", "evt-0003"]


def test_csv_is_processed_incrementally(tmp_path: Path) -> None:
    csv_path = _write_csv(
        tmp_path / "orders.csv",
        "event_id,symbol,transaction_type,quantity\n"
        "evt-0001,RELIANCE,BUY,90\n"
        "evt-0002,TCS,SELL,75\n",
    )
    sender = RecordingSender()
    sends_before_yield: list[int] = []

    def tracked_rows():
        for row in iter_csv_rows(csv_path):
            sends_before_yield.append(len(sender.events))
            yield row

    rows = iter_csv_rows(csv_path)
    assert not isinstance(rows, list)

    process_rows(tracked_rows(), sender.send, interval=0, sleeper=lambda _: None)

    assert sends_before_yield == [0, 1]
    assert [event.event_id for event in sender.events] == ["evt-0001", "evt-0002"]


def test_events_are_sent_in_csv_order() -> None:
    sender = RecordingSender()
    rows = [
        _row("evt-0001"),
        _row("evt-0002", symbol="TCS", transaction_type="SELL", quantity="75"),
        _row("evt-0003", symbol="HDFCBANK", quantity="60"),
    ]

    process_rows(rows, sender.send, interval=0, sleeper=lambda _: None)

    assert [event.event_id for event in sender.events] == [
        "evt-0001",
        "evt-0002",
        "evt-0003",
    ]
    assert sender.events[0] == OrderEvent("evt-0001", "RELIANCE", "BUY", 90)
    assert sender.events[1] == OrderEvent("evt-0002", "TCS", "SELL", 75)
    assert sender.events[2] == OrderEvent("evt-0003", "HDFCBANK", "BUY", 60)


def test_invalid_rows_are_not_sent() -> None:
    sender = RecordingSender()
    rows = [
        _row("evt-0001", quantity="abc"),
        _row("evt-0002", transaction_type="hold"),
        _row(event_id=""),
    ]

    process_rows(rows, sender.send, interval=0, sleeper=lambda _: None)

    assert sender.events == []


def test_throttle_interval_matches_configured_rate() -> None:
    assert send_interval_seconds(50) == 0.02
    assert send_interval_seconds(10) == 0.1

    sleeps: list[float] = []
    sender = RecordingSender()
    rows = [_row("evt-0001"), _row("evt-0002"), _row("evt-0003", quantity="0")]

    process_rows(rows, sender.send, interval=0.02, sleeper=sleeps.append)

    assert sleeps == [0.02]
    assert [event.event_id for event in sender.events] == ["evt-0001", "evt-0002"]


def test_http_success_response_is_handled(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.INFO)

    def post(url: str, json: dict, timeout: float) -> httpx.Response:
        assert url == "http://127.0.0.1:8000/events"
        assert json == event_payload(OrderEvent("evt-0001", "RELIANCE", "BUY", 90))
        assert timeout == 5.0
        return httpx.Response(200, json={"status": "accepted"})

    client = EventClient("127.0.0.1", 8000, 5.0, post=post)
    result = client.send(OrderEvent("evt-0001", "RELIANCE", "BUY", 90))
    process_rows([_row()], client.send, interval=0, sleeper=lambda _: None)

    assert result == SendResult(delivered=True, status="accepted")
    assert "event_id=evt-0001 successfully sent" in caplog.text


def test_duplicate_response_is_handled(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.INFO)
    sender = RecordingSender([SendResult(delivered=True, status="duplicate")])

    process_rows([_row()], sender.send, interval=0, sleeper=lambda _: None)

    assert "event_id=evt-0001 delivered but ignored as duplicate" in caplog.text


def test_http_error_response_is_handled(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.ERROR)

    def post(url: str, json: dict, timeout: float) -> httpx.Response:
        return httpx.Response(500, text="internal error")

    client = EventClient("127.0.0.1", 8000, 5.0, post=post)
    result = client.send(OrderEvent("evt-0001", "RELIANCE", "BUY", 90))

    process_rows([_row()], client.send, interval=0, sleeper=lambda _: None)

    assert result.delivered is False
    assert "HTTP 500" in (result.error or "")
    assert "event_id=evt-0001 delivery failed" in caplog.text


def test_connection_failure_does_not_stop_later_rows(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.INFO)
    sender = RecordingSender(
        [
            SendResult(delivered=False, error="connection refused"),
            SendResult(delivered=True, status="accepted"),
        ]
    )
    rows = [_row("evt-0001"), _row("evt-0002", symbol="TCS")]

    process_rows(rows, sender.send, interval=0, sleeper=lambda _: None)

    assert [event.event_id for event in sender.events] == ["evt-0001", "evt-0002"]
    assert "event_id=evt-0001 delivery failed: connection refused" in caplog.text
    assert "event_id=evt-0002 successfully sent" in caplog.text


def test_missing_input_file_produces_clear_error(tmp_path: Path) -> None:
    missing = tmp_path / "does-not-exist.csv"

    with pytest.raises(FileNotFoundError, match="Input file not found"):
        list(iter_csv_rows(missing))


def test_required_configuration_can_be_supplied() -> None:
    settings = load_settings(
        [
            "--input-file",
            "orders.csv",
            "--host",
            "10.0.0.2",
            "--port",
            "9000",
            "--rate-limit",
            "25",
            "--timeout",
            "2.5",
        ]
    )

    assert settings == Settings(
        input_file="orders.csv",
        host="10.0.0.2",
        port=9000,
        rate_limit=25,
        timeout=2.5,
    )


def test_processing_completion_is_logged(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.INFO)

    process_rows([_row()], RecordingSender().send, interval=0, sleeper=lambda _: None)

    assert COMPLETION_MESSAGE in caplog.text


def test_run_uses_input_file_and_does_not_require_network(tmp_path: Path) -> None:
    csv_path = _write_csv(
        tmp_path / "orders.csv",
        "event_id,symbol,transaction_type,quantity\n"
        "evt-0001,RELIANCE,BUY,90\n",
    )
    sender = RecordingSender()
    settings = Settings(
        input_file=str(csv_path),
        host="127.0.0.1",
        port=8000,
        rate_limit=50,
        timeout=1.0,
    )

    run(settings, send=sender.send, sleeper=lambda _: None)

    assert sender.events == [OrderEvent("evt-0001", "RELIANCE", "BUY", 90)]


def test_event_client_maps_connection_errors() -> None:
    def post(url: str, json: dict, timeout: float) -> httpx.Response:
        raise httpx.ConnectError("connection refused")

    result = EventClient("127.0.0.1", 8000, 1.0, post=post).send(
        OrderEvent("evt-0003", "INFY", "BUY", 10)
    )

    assert result.delivered is False
    assert "connection refused" in (result.error or "")


def test_missing_csv_columns_raise_clear_error(tmp_path: Path) -> None:
    csv_path = _write_csv(tmp_path / "bad.csv", "event_id,symbol\nevt-0001,RELIANCE\n")

    with pytest.raises(ValueError, match="missing required columns"):
        list(iter_csv_rows(csv_path))
