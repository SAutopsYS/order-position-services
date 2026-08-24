"""Position Maintaining Service HTTP API."""

from __future__ import annotations

import argparse
import os
from typing import Optional

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from src.event import parse_event
from src.position_store import PositionStore

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8000


def create_app(store: Optional[PositionStore] = None) -> FastAPI:
    """Build the Position Maintaining Service app.

    POST /events is a design choice. The assignment does not name the inbound path.
    """
    position_store = store if store is not None else PositionStore()
    app = FastAPI(title="Position Maintaining Service")

    @app.get("/position")
    def get_position() -> dict[str, int]:
        return position_store.snapshot()

    @app.post("/events")
    async def post_event(request: Request) -> JSONResponse:
        payload, payload_error = await _read_json_object(request)
        if payload_error is not None:
            return JSONResponse({"error": payload_error}, status_code=400)

        result = parse_event(
            payload.get("event_id"),
            payload.get("symbol"),
            payload.get("transaction_type"),
            payload.get("quantity"),
        )
        if not result.ok or result.event is None:
            return JSONResponse({"error": result.error}, status_code=400)

        applied = position_store.apply(result.event)
        status = "accepted" if applied else "duplicate"
        return JSONResponse({"status": status}, status_code=200)

    return app


app = create_app()


async def _read_json_object(request: Request) -> tuple[Optional[dict], Optional[str]]:
    try:
        payload = await request.json()
    except Exception:
        return None, "request body must be valid JSON"
    if not isinstance(payload, dict):
        return None, "request body must be a JSON object"
    return payload, None


def main() -> None:
    parser = argparse.ArgumentParser(description="Position Maintaining Service")
    parser.add_argument(
        "--host",
        default=os.environ.get("POSITION_SERVICE_HOST", DEFAULT_HOST),
        help=f"Bind address (default: {DEFAULT_HOST})",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.environ.get("POSITION_SERVICE_PORT", str(DEFAULT_PORT))),
        help=f"Bind port (default: {DEFAULT_PORT})",
    )
    args = parser.parse_args()
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
