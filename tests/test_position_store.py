from concurrent.futures import ThreadPoolExecutor

from src.event import OrderEvent
from src.position_store import PositionStore


def _event(
    event_id: str,
    symbol: str,
    transaction_type: str,
    quantity: int,
) -> OrderEvent:
    return OrderEvent(
        event_id=event_id,
        symbol=symbol,
        transaction_type=transaction_type,
        quantity=quantity,
    )


def test_buy_increases_position() -> None:
    store = PositionStore()

    applied = store.apply(_event("evt-0001", "RELIANCE", "BUY", 90))

    assert applied is True
    assert store.snapshot() == {"RELIANCE": 90}


def test_sell_decreases_position() -> None:
    store = PositionStore()

    applied = store.apply(_event("evt-0002", "TCS", "SELL", 75))

    assert applied is True
    assert store.snapshot() == {"TCS": -75}


def test_multiple_events_for_same_symbol() -> None:
    store = PositionStore()

    store.apply(_event("evt-0001", "INFY", "BUY", 100))
    store.apply(_event("evt-0002", "INFY", "SELL", 40))

    assert store.snapshot() == {"INFY": 60}


def test_multiple_symbols_have_independent_positions() -> None:
    store = PositionStore()

    store.apply(_event("evt-0001", "RELIANCE", "BUY", 90))
    store.apply(_event("evt-0002", "TCS", "SELL", 75))
    store.apply(_event("evt-0003", "HDFCBANK", "BUY", 60))

    assert store.snapshot() == {
        "RELIANCE": 90,
        "TCS": -75,
        "HDFCBANK": 60,
    }


def test_negative_position_is_valid() -> None:
    store = PositionStore()

    store.apply(_event("evt-0001", "TCS", "SELL", 75))

    assert store.snapshot() == {"TCS": -75}


def test_zero_position_is_preserved() -> None:
    store = PositionStore()

    store.apply(_event("evt-0001", "RELIANCE", "BUY", 100))
    store.apply(_event("evt-0002", "RELIANCE", "SELL", 100))

    snapshot = store.snapshot()
    assert "RELIANCE" in snapshot
    assert snapshot["RELIANCE"] == 0


def test_duplicate_event_id_is_ignored() -> None:
    store = PositionStore()

    first = store.apply(_event("evt-0001", "RELIANCE", "BUY", 100))
    second = store.apply(_event("evt-0001", "RELIANCE", "SELL", 100))

    assert first is True
    assert second is False
    assert store.snapshot() == {"RELIANCE": 100}


def test_duplicate_event_id_with_different_fields_is_ignored() -> None:
    store = PositionStore()

    first = store.apply(_event("evt-0001", "RELIANCE", "BUY", 100))
    second = store.apply(_event("evt-0001", "TCS", "SELL", 500))

    assert first is True
    assert second is False
    assert store.snapshot() == {"RELIANCE": 100}
    assert "TCS" not in store.snapshot()


def test_different_event_ids_are_both_applied() -> None:
    store = PositionStore()

    first = store.apply(_event("evt-0001", "RELIANCE", "BUY", 50))
    second = store.apply(_event("evt-0002", "RELIANCE", "BUY", 50))

    assert first is True
    assert second is True
    assert store.snapshot() == {"RELIANCE": 100}


def test_empty_store_snapshot_is_empty() -> None:
    store = PositionStore()

    assert store.snapshot() == {}


def test_snapshot_returns_a_copy() -> None:
    store = PositionStore()
    store.apply(_event("evt-0001", "RELIANCE", "BUY", 90))

    snapshot = store.snapshot()
    snapshot["RELIANCE"] = 999
    snapshot["TCS"] = -1

    assert store.snapshot() == {"RELIANCE": 90}


def test_concurrent_applies_and_snapshots_keep_consistent_state() -> None:
    store = PositionStore()
    events = [
        _event(
            f"evt-{index:04d}",
            "RELIANCE" if index % 2 == 0 else "TCS",
            "BUY" if index % 2 == 0 else "SELL",
            index + 1,
        )
        for index in range(200)
    ]
    expected = {"RELIANCE": 0, "TCS": 0}
    for event in events:
        if event.transaction_type == "BUY":
            expected[event.symbol] += event.quantity
        else:
            expected[event.symbol] -= event.quantity

    errors: list[BaseException] = []

    def apply_event(event: OrderEvent) -> None:
        try:
            store.apply(event)
        except BaseException as exc:
            errors.append(exc)

    def take_snapshots() -> None:
        try:
            for _ in range(100):
                snapshot = store.snapshot()
                assert isinstance(snapshot, dict)
                for symbol, quantity in snapshot.items():
                    assert isinstance(symbol, str)
                    assert isinstance(quantity, int)
        except BaseException as exc:
            errors.append(exc)

    with ThreadPoolExecutor(max_workers=12) as pool:
        apply_futures = [pool.submit(apply_event, event) for event in events]
        snapshot_futures = [pool.submit(take_snapshots) for _ in range(4)]
        for future in apply_futures + snapshot_futures:
            future.result()

    assert errors == []
    assert store.snapshot() == expected
