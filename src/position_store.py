"""In-memory net positions and accepted event IDs."""

from __future__ import annotations

import threading

from src.event import BUY, SELL, OrderEvent


class PositionStore:
    """Thread-safe in-memory positions for accepted order events."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._positions: dict[str, int] = {}
        self._accepted_ids: set[str] = set()

    def apply(self, event: OrderEvent) -> bool:
        """Apply a validated event.

        Returns True if the event was newly applied.
        Returns False if event_id was already accepted and the event was ignored.
        """
        with self._lock:
            if event.event_id in self._accepted_ids:
                return False

            self._accepted_ids.add(event.event_id)

            current = self._positions.get(event.symbol, 0)
            if event.transaction_type == BUY:
                current += event.quantity
            elif event.transaction_type == SELL:
                current -= event.quantity

            self._positions[event.symbol] = current
            return True

    def snapshot(self) -> dict[str, int]:
        """Return a copy of current symbol positions."""
        with self._lock:
            return dict(self._positions)
