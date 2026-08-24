# Order Position Services

Order Position Services is a pair of independently runnable Python processes that stream order updates, validate them, and keep the current net position for each trading symbol. The **Order Update Service** reads a CSV one row at a time, rejects invalid events, and POSTs valid events over HTTP. The **Position Maintaining Service** applies those events to in-memory state and exposes the result on `GET /position`.

## Overview

A trading system needs a current net position per symbol as buy and sell events arrive. This project solves that with two small services and no extra infrastructure.

| Stage | What happens |
|---|---|
| Input | CSV of `event_id`, `symbol`, `transaction_type`, `quantity` |
| Processing | Incremental read, validation, ordered send |
| Communication | HTTP `POST /events` between two processes |
| State | In-memory positions and accepted event IDs |
| Output | `GET /position` |

The assessment allows HTTP and does not require a database, broker, or cloud deploy. The design stays inside that scope.

## Architecture

```text
order_updates.csv
       |
       v
+---------------------------+
| Order Update Service      |
|                           |
| - Stream CSV              |
| - Validate events         |
| - Throttle sends          |
| - Log processing          |
+-------------+-------------+
              |
              | HTTP POST /events
              v
+---------------------------+
| Position Maintaining      |
| Service                   |
|                           |
| - Validate events         |
| - Deduplicate event IDs   |
| - Update positions        |
| - Maintain in-memory state|
+-------------+-------------+
              |
              | GET /position
              v
        Current positions
```

**Order Update Service** is the producer. It never loads the full CSV into a list. Valid events are sent in CSV order.

**Position Maintaining Service** is the consumer and API. It owns position math and duplicate `event_id` handling.

**PositionStore** is the in-memory store used by the Position Maintaining Service. It is not a separate process.

`POST /events` is an implementation choice. The assessment requires a defined interface. It does not name this path.

## Why HTTP?

HTTP was selected because:

- The two services must run as separate processes over a defined interface
- The assessment explicitly allows HTTP
- Local setup needs no Redis, Kafka, or other infrastructure
- Request and response handling is easy to test
- Connection and delivery failures are visible in logs

HTTP here is a local request/response link. It is not durable delivery.

## Data Flow

Lifecycle of one CSV row:

1. The CSV reader yields one row.
2. The row is validated against the event contract.
3. An invalid row is logged with a reason and skipped.
4. A valid row becomes an `OrderEvent`.
5. The producer waits if needed to stay under the configured send rate, then POSTs `/events`.
6. The Position Maintaining Service validates the JSON payload.
7. `PositionStore.apply()` checks whether `event_id` was already accepted.
8. A duplicate is ignored. The HTTP response is `{"status": "duplicate"}`.
9. A new event updates that symbol: BUY adds, SELL subtracts.
10. `GET /position` returns a copy of the current map.

## Event Contract

```json
{
  "event_id": "evt-0001",
  "symbol": "RELIANCE",
  "transaction_type": "BUY",
  "quantity": 90
}
```

| Field | Meaning |
|---|---|
| `event_id` | Unique ID for one event |
| `symbol` | Trading symbol |
| `transaction_type` | `BUY` or `SELL` |
| `quantity` | Units to add or subtract |

### Validation Rules

- `event_id` must be a non-empty string
- `symbol` must be a non-empty string
- The supplied symbol case and value are preserved
- `transaction_type` must be exactly `BUY` or `SELL` (`buy` is rejected)
- `quantity` must be a positive integer (`0`, negatives, decimals, and non-numeric values are rejected)

Invalid rows are logged and skipped. Later rows still run. The process does not crash on a bad row.

## Position Rules

```text
BUY  -> position += quantity
SELL -> position -= quantity
```

- Negative positions are valid
- Zero positions are valid
- A symbol stays in the map after its net position becomes zero
- Symbols never seen in an accepted event are omitted

## Duplicate Event Handling

The first **valid** event received for an `event_id` wins. Later events with the same ID are ignored even if other fields differ.

```text
evt-0001 RELIANCE BUY 100
evt-0001 RELIANCE SELL 100
```

```json
{
  "RELIANCE": 100
}
```

Accepted IDs live in memory on the Position Maintaining Service. That set is cleared if the process restarts. Persistence is intentionally out of scope.

## Services

### Order Update Service

- Reads the CSV with the standard `csv` module, one row at a time
- Validates each row with the shared event parser
- Sends valid events in CSV order on one thread
- Default maximum send rate is 50 events per second (minimum 0.02 seconds between sends)
- The rate is configurable. Sub-millisecond timing is not claimed
- Invalid rows do not consume a send slot
- Logs accepted events, rejected events with a reason, successful sends, duplicates, delivery failures, and completion
- Each valid event is sent once. On connection failure, timeout, or HTTP error, the failure is logged and the next row still runs
- A missing input file stops the process with a clear error because there is nothing to read

### Position Maintaining Service

- Listens for `POST /events` and `GET /position`
- Re-validates the JSON body so a bad payload cannot crash the service
- Delegates updates and duplicate checks to `PositionStore`
- BUY adds quantity. SELL subtracts quantity
- `GET /position` stays available while events are applied
- Positions and accepted IDs are protected by one `threading.Lock`

## API Reference

### POST /events

Chosen inbound interface between the two services. The exact response body format is an implementation choice.

Request:

```json
{
  "event_id": "evt-0001",
  "symbol": "RELIANCE",
  "transaction_type": "BUY",
  "quantity": 90
}
```

Newly applied (`200`):

```json
{
  "status": "accepted"
}
```

Duplicate `event_id` (`200`):

```json
{
  "status": "duplicate"
}
```

Invalid JSON or contract failure (`400`):

```json
{
  "error": "quantity must be a positive integer"
}
```

### GET /position

Returns every symbol seen in an accepted event. Zero and negative values are included. Key order is not significant. The handler returns `PositionStore.snapshot()`, a copy of the map, not the live dictionary.

```json
{
  "RELIANCE": 90,
  "TCS": -75
}
```

An unused service returns `{}`.

## Concurrency

`PositionStore` guards `_positions` and `_accepted_ids` with the same lock. `apply()` and `snapshot()` both take that lock. `GET /position` therefore sees a consistent copy, not a half-updated map.

This covers concurrent updates and API reads in one process. It is not a distributed locking design.

## Configuration

| Service | CLI option | Environment variable | Default |
|---|---|---|---|
| Order Update | `--input-file` | `ORDER_UPDATE_INPUT_FILE` | required |
| Order Update | `--host` | `POSITION_SERVICE_HOST` | `127.0.0.1` |
| Order Update | `--port` | `POSITION_SERVICE_PORT` | `8000` |
| Order Update | `--rate-limit` | `ORDER_UPDATE_RATE_LIMIT` | `50` |
| Order Update | `--timeout` | `ORDER_UPDATE_HTTP_TIMEOUT` | `5` seconds |
| Position Service | `--host` | `POSITION_SERVICE_HOST` | `127.0.0.1` |
| Position Service | `--port` | `POSITION_SERVICE_PORT` | `8000` |

`--rate-limit` is the maximum number of valid events sent per second.

## Installation

The code uses `str | Path` type syntax, so Python 3.10 or later is required.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

On macOS or Linux:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Dependencies: `fastapi`, `uvicorn`, `httpx`, `pytest`.

## Running the Services

The services are two separate processes. Start the Position Maintaining Service first.

### Terminal 1

```powershell
python -m src.position_service
```

### Terminal 2

```powershell
python -m src.order_update_service --input-file path/to/order_updates.csv --host 127.0.0.1 --port 8000
```

If the supplied assessment CSV is present in this repository:

```powershell
python -m src.order_update_service --input-file "SDE Intern-20260824T180907Z-1-001/SDE Intern/order_updates.csv" --host 127.0.0.1 --port 8000
```

## API Usage

With the Position Maintaining Service running on port 8000:

```powershell
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/events -ContentType "application/json" -Body '{"event_id":"evt-0001","symbol":"RELIANCE","transaction_type":"BUY","quantity":90}'

Invoke-RestMethod -Uri http://127.0.0.1:8000/position
```

## Testing

The suite includes:

- Event validation tests
- PositionStore calculation and concurrency tests
- HTTP API tests via a test client
- Order Update Service tests with a fake HTTP sender
- End-to-end tests that start both services as separate OS processes

```powershell
python -m pytest -q
```

At the time of submission, the full suite contains 67 passing tests.

The E2E tests pick a free local port, poll `GET /position` until the consumer is ready, run a small fixture CSV through the producer, then check the final positions. They do not assert exact send timing.

## Supplied CSV Verification

The supplied `order_updates.csv` was run through both processes as a smoke test (not hard-coded into the application):

- 1000 events processed
- 20 symbols returned
- Positions matched independent BUY/SELL arithmetic
- No rejected rows in that clean file
- Both services ran as separate processes

## Error Handling

### Invalid CSV row

The reason is logged. The row is skipped. Later rows still run.

### Position Service unavailable

The producer logs a delivery failure for that event. It does not log a successful send. The next CSV row still runs. There is no retry queue.

### HTTP non-success response

Treated as a delivery failure. Status and body are included in the log.

### Missing input file

The Order Update Service exits with a clear `Input file not found` error. Processing does not start.

### Missing CSV columns

A header that lacks `event_id`, `symbol`, `transaction_type`, or `quantity` is rejected with a clear error before rows are processed.

## Delivery Guarantees and Limitations

- One delivery attempt per valid event
- No durable queue
- No persistence
- No recovery after a complete process restart
- Duplicate ID state is in memory only
- Exactly-once delivery across restarts is not implemented
- If a send fails, that event is skipped and later rows continue. The event can be lost
- This matches the assessment: durable delivery and exactly-once broker guarantees are out of scope

## Design Trade-offs

| Choice | Why |
|---|---|
| Python | Preferred by the assessment and practical for this scope |
| FastAPI | Small HTTP API for `GET /position` and `POST /events` |
| httpx | HTTP client already used in tests; one reused client, `trust_env=False` so env proxies do not intercept localhost |
| Standard `csv` | Incremental row reads with no extra library |
| In-memory `dict` / `set` | Persistence is out of scope |
| `threading.Lock` | Concurrent apply and snapshot stay consistent |
| HTTP | Allowed interface, no broker required |

Intentionally not used: database, Redis, Kafka, other message brokers, Docker, cloud deploy. Those are assessment non-goals and are not needed for two local processes.

## Project Structure

```text
order-position-services/
├── src/
│   ├── __init__.py
│   ├── event.py
│   ├── position_store.py
│   ├── position_service.py
│   └── order_update_service.py
├── tests/
│   ├── __init__.py
│   ├── test_event.py
│   ├── test_position_store.py
│   ├── test_position_service.py
│   ├── test_order_update_service.py
│   └── test_end_to_end.py
├── README.md
├── requirements.txt
├── pytest.ini
└── .gitignore
```

The supplied assessment files, when present, are left unchanged under:

- `SDE Intern-20260824T180907Z-1-001/SDE Intern/SDE_Intern_Assessment.pdf`
- `SDE Intern-20260824T180907Z-1-001/SDE Intern/order_updates.csv`

## Assessment Alignment

| Assessment area | Implementation |
|---|---|
| Streaming input | `csv.DictReader` yielding one row at a time |
| Validation | Shared `parse_event` used by both services |
| Idempotency | In-memory accepted `event_id` set |
| Service communication | HTTP `POST /events` |
| Position state | In-memory `PositionStore` |
| API | `GET /position` from a snapshot |
| Concurrency | One lock around positions and accepted IDs |
| Configuration | CLI flags and environment variables |
| Testing | pytest unit, API, and process-level E2E tests |
| Documentation | This README |

## AI-Assisted Development

AI-assisted tools were used for planning, implementation support, debugging, and documentation review. The code was reviewed against the assessment, exercised with the automated suite, and run against the supplied CSV. The author can explain the architecture, design choices, and submitted code.

## GitHub

Repository: [https://github.com/SAutopsYS/order-position-services](https://github.com/SAutopsYS/order-position-services)
