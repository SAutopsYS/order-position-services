from fastapi.testclient import TestClient

from src.position_service import create_app


def _client() -> TestClient:
    return TestClient(create_app())


def _buy(event_id: str = "evt-0001", symbol: str = "RELIANCE", quantity: int = 90) -> dict:
    return {
        "event_id": event_id,
        "symbol": symbol,
        "transaction_type": "BUY",
        "quantity": quantity,
    }


def _sell(event_id: str = "evt-0002", symbol: str = "TCS", quantity: int = 75) -> dict:
    return {
        "event_id": event_id,
        "symbol": symbol,
        "transaction_type": "SELL",
        "quantity": quantity,
    }


def test_get_position_on_empty_service_returns_empty_object() -> None:
    client = _client()

    response = client.get("/position")

    assert response.status_code == 200
    assert response.json() == {}


def test_get_position_after_buy_contains_symbol() -> None:
    client = _client()
    client.post("/events", json=_buy())

    response = client.get("/position")

    assert response.status_code == 200
    assert response.json() == {"RELIANCE": 90}


def test_get_position_after_sell_contains_negative_position() -> None:
    client = _client()
    client.post("/events", json=_sell())

    response = client.get("/position")

    assert response.status_code == 200
    assert response.json() == {"TCS": -75}


def test_get_position_returns_multiple_symbols_independently() -> None:
    client = _client()
    client.post("/events", json=_buy())
    client.post("/events", json=_sell())

    response = client.get("/position")

    assert response.status_code == 200
    assert response.json() == {"RELIANCE": 90, "TCS": -75}


def test_get_position_keeps_zero_position_symbol() -> None:
    client = _client()
    client.post("/events", json=_buy(quantity=100))
    client.post("/events", json=_sell(event_id="evt-0002", symbol="RELIANCE", quantity=100))

    response = client.get("/position")

    assert response.status_code == 200
    assert response.json() == {"RELIANCE": 0}


def test_post_valid_buy_event_is_accepted() -> None:
    client = _client()

    response = client.post("/events", json=_buy())

    assert response.status_code == 200
    assert response.json() == {"status": "accepted"}


def test_post_valid_sell_event_is_accepted() -> None:
    client = _client()

    response = client.post("/events", json=_sell())

    assert response.status_code == 200
    assert response.json() == {"status": "accepted"}


def test_duplicate_event_id_does_not_change_position() -> None:
    client = _client()
    client.post("/events", json=_buy(quantity=100))

    response = client.post(
        "/events",
        json=_sell(event_id="evt-0001", symbol="RELIANCE", quantity=100),
    )

    assert response.status_code == 200
    assert response.json() == {"status": "duplicate"}
    assert client.get("/position").json() == {"RELIANCE": 100}


def test_duplicate_event_id_with_different_fields_is_ignored() -> None:
    client = _client()
    client.post("/events", json=_buy(quantity=100))

    response = client.post(
        "/events",
        json=_sell(event_id="evt-0001", symbol="TCS", quantity=500),
    )

    assert response.status_code == 200
    assert response.json() == {"status": "duplicate"}
    assert client.get("/position").json() == {"RELIANCE": 100}
    assert "TCS" not in client.get("/position").json()


def test_invalid_transaction_type_returns_400() -> None:
    client = _client()
    payload = _buy()
    payload["transaction_type"] = "HOLD"

    response = client.post("/events", json=payload)

    assert response.status_code == 400
    assert "transaction_type" in response.json()["error"]


def test_zero_quantity_returns_400() -> None:
    client = _client()

    response = client.post("/events", json=_buy(quantity=0))

    assert response.status_code == 400
    assert "quantity" in response.json()["error"]


def test_negative_quantity_returns_400() -> None:
    client = _client()
    payload = _buy()
    payload["quantity"] = -5

    response = client.post("/events", json=payload)

    assert response.status_code == 400
    assert "quantity" in response.json()["error"]


def test_decimal_quantity_returns_400() -> None:
    client = _client()
    payload = _buy()
    payload["quantity"] = 1.5

    response = client.post("/events", json=payload)

    assert response.status_code == 400
    assert "quantity" in response.json()["error"]


def test_non_numeric_quantity_returns_400() -> None:
    client = _client()
    payload = _buy()
    payload["quantity"] = "abc"

    response = client.post("/events", json=payload)

    assert response.status_code == 400
    assert "quantity" in response.json()["error"]


def test_blank_event_id_returns_400() -> None:
    client = _client()
    payload = _buy()
    payload["event_id"] = ""

    response = client.post("/events", json=payload)

    assert response.status_code == 400
    assert "event_id" in response.json()["error"]


def test_blank_symbol_returns_400() -> None:
    client = _client()
    payload = _buy()
    payload["symbol"] = ""

    response = client.post("/events", json=payload)

    assert response.status_code == 400
    assert "symbol" in response.json()["error"]


def test_missing_required_fields_return_400() -> None:
    client = _client()

    response = client.post("/events", json={})

    assert response.status_code == 400
    assert "error" in response.json()


def test_invalid_requests_do_not_change_position() -> None:
    client = _client()
    client.post("/events", json=_buy())

    client.post("/events", json=_buy(event_id="", quantity=10))
    bad_type = _buy(event_id="evt-0099")
    bad_type["transaction_type"] = "hold"
    client.post("/events", json=bad_type)
    client.post("/events", content=b"not-json", headers={"Content-Type": "application/json"})

    assert client.get("/position").json() == {"RELIANCE": 90}


def test_post_then_get_returns_expected_position() -> None:
    client = _client()

    post_response = client.post("/events", json=_buy())
    get_response = client.get("/position")

    assert post_response.status_code == 200
    assert post_response.json() == {"status": "accepted"}
    assert get_response.status_code == 200
    assert get_response.json() == {"RELIANCE": 90}


def test_get_position_does_not_expose_internal_store_state() -> None:
    client = _client()
    client.post("/events", json=_buy())

    first = client.get("/position").json()
    first["RELIANCE"] = 999
    first["TCS"] = -1

    second = client.get("/position").json()
    assert second == {"RELIANCE": 90}


def test_invalid_json_returns_400() -> None:
    client = _client()

    response = client.post(
        "/events",
        content=b"not-json",
        headers={"Content-Type": "application/json"},
    )

    assert response.status_code == 400
    assert "error" in response.json()
