"""Order event model and validation for the Event Contract."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

BUY = "BUY"
SELL = "SELL"
VALID_TRANSACTION_TYPES = frozenset({BUY, SELL})


@dataclass(frozen=True)
class OrderEvent:
    """One accepted order event from the assignment contract."""

    event_id: str
    symbol: str
    transaction_type: str
    quantity: int


@dataclass(frozen=True)
class ParseResult:
    """Outcome of validating a candidate event.

    If valid, event is set and error is None.
    If invalid, event is None and error is a reason suitable for logging.
    """

    event: Optional[OrderEvent] = None
    error: Optional[str] = None

    @property
    def ok(self) -> bool:
        return self.event is not None


def parse_event(
    event_id: object,
    symbol: object,
    transaction_type: object,
    quantity: object,
) -> ParseResult:
    """Validate raw field values and build an OrderEvent when they are valid."""

    parsed_event_id, event_id_error = _require_non_empty_string(event_id, "event_id")
    if event_id_error is not None:
        return ParseResult(error=event_id_error)

    parsed_symbol, symbol_error = _require_non_empty_string(symbol, "symbol")
    if symbol_error is not None:
        return ParseResult(error=symbol_error)

    parsed_type, type_error = _require_transaction_type(transaction_type)
    if type_error is not None:
        return ParseResult(error=type_error)

    parsed_quantity, quantity_error = _require_positive_integer(quantity)
    if quantity_error is not None:
        return ParseResult(error=quantity_error)

    return ParseResult(
        event=OrderEvent(
            event_id=parsed_event_id,
            symbol=parsed_symbol,
            transaction_type=parsed_type,
            quantity=parsed_quantity,
        )
    )


def _require_non_empty_string(
    value: object, field_name: str
) -> tuple[Optional[str], Optional[str]]:
    if not isinstance(value, str) or value.strip() == "":
        return None, f"{field_name} must be a non-empty string"
    return value, None


def _require_transaction_type(value: object) -> tuple[Optional[str], Optional[str]]:
    if not isinstance(value, str) or value not in VALID_TRANSACTION_TYPES:
        return None, "transaction_type must be exactly BUY or SELL"
    return value, None


def _require_positive_integer(value: object) -> tuple[Optional[int], Optional[str]]:
    if isinstance(value, str):
        if value.strip() == "":
            return None, "quantity must not be blank"
        if not value.strip().isdigit():
            return None, "quantity must be a positive integer"
        number = int(value.strip())
        if number <= 0:
            return None, "quantity must be a positive integer"
        return number, None

    # bool is a subclass of int, so reject it before the int check.
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        return None, "quantity must be a positive integer"

    return value, None
