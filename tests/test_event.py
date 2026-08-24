from src.event import parse_event


def test_valid_buy_event() -> None:
    result = parse_event("evt-0001", "RELIANCE", "BUY", 90)

    assert result.ok
    assert result.error is None
    assert result.event is not None
    assert result.event.event_id == "evt-0001"
    assert result.event.symbol == "RELIANCE"
    assert result.event.transaction_type == "BUY"
    assert result.event.quantity == 90


def test_valid_sell_event() -> None:
    result = parse_event("evt-0002", "TCS", "SELL", 75)

    assert result.ok
    assert result.error is None
    assert result.event is not None
    assert result.event.event_id == "evt-0002"
    assert result.event.symbol == "TCS"
    assert result.event.transaction_type == "SELL"
    assert result.event.quantity == 75


def test_symbol_preserves_supplied_case_and_value() -> None:
    result = parse_event("evt-0003", "Reliance", "BUY", 10)

    assert result.ok
    assert result.event is not None
    assert result.event.symbol == "Reliance"


def test_blank_event_id_is_rejected() -> None:
    result = parse_event("", "RELIANCE", "BUY", 90)

    assert not result.ok
    assert result.event is None
    assert result.error == "event_id must be a non-empty string"


def test_whitespace_event_id_is_rejected() -> None:
    result = parse_event("   ", "RELIANCE", "BUY", 90)

    assert not result.ok
    assert result.event is None
    assert "event_id" in (result.error or "")


def test_blank_symbol_is_rejected() -> None:
    result = parse_event("evt-0001", "", "BUY", 90)

    assert not result.ok
    assert result.event is None
    assert result.error == "symbol must be a non-empty string"


def test_whitespace_symbol_is_rejected() -> None:
    result = parse_event("evt-0001", "   ", "BUY", 90)

    assert not result.ok
    assert result.event is None
    assert "symbol" in (result.error or "")


def test_invalid_transaction_type_is_rejected() -> None:
    result = parse_event("evt-0001", "RELIANCE", "HOLD", 90)

    assert not result.ok
    assert result.event is None
    assert result.error == "transaction_type must be exactly BUY or SELL"


def test_lowercase_transaction_type_is_rejected() -> None:
    result = parse_event("evt-0001", "RELIANCE", "buy", 90)

    assert not result.ok
    assert result.event is None
    assert result.error == "transaction_type must be exactly BUY or SELL"


def test_quantity_zero_is_rejected() -> None:
    result = parse_event("evt-0001", "RELIANCE", "BUY", 0)

    assert not result.ok
    assert result.event is None
    assert result.error == "quantity must be a positive integer"


def test_negative_quantity_is_rejected() -> None:
    result = parse_event("evt-0001", "RELIANCE", "BUY", -5)

    assert not result.ok
    assert result.event is None
    assert result.error == "quantity must be a positive integer"


def test_decimal_quantity_is_rejected() -> None:
    float_result = parse_event("evt-0001", "RELIANCE", "BUY", 1.5)
    string_result = parse_event("evt-0001", "RELIANCE", "BUY", "90.0")

    assert not float_result.ok
    assert float_result.event is None
    assert float_result.error == "quantity must be a positive integer"

    assert not string_result.ok
    assert string_result.event is None
    assert string_result.error == "quantity must be a positive integer"


def test_non_numeric_quantity_is_rejected() -> None:
    result = parse_event("evt-0001", "RELIANCE", "BUY", "abc")

    assert not result.ok
    assert result.event is None
    assert result.error == "quantity must be a positive integer"


def test_blank_quantity_is_rejected() -> None:
    result = parse_event("evt-0001", "RELIANCE", "BUY", "")

    assert not result.ok
    assert result.event is None
    assert result.error == "quantity must not be blank"


def test_numeric_string_quantity_is_accepted() -> None:
    result = parse_event("evt-0001", "RELIANCE", "BUY", "90")

    assert result.ok
    assert result.event is not None
    assert result.event.quantity == 90
