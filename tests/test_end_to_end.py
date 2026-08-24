"""End-to-end tests that start both services as separate processes."""

from __future__ import annotations

import os
import socket
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

import httpx

PROJECT_ROOT = Path(__file__).resolve().parents[1]
READY_TIMEOUT_SECONDS = 15.0
PRODUCER_TIMEOUT_SECONDS = 30.0

PIPELINE_CSV = """event_id,symbol,transaction_type,quantity
evt-0001,RELIANCE,BUY,100
evt-0002,TCS,SELL,75
evt-0003,RELIANCE,SELL,40
evt-0004,TCS,BUY,25
evt-0005,INFY,BUY,50
evt-0006,INFY,SELL,50
evt-0007,TCS,HOLD,50
evt-0008,HDFCBANK,BUY,10
evt-0001,RELIANCE,SELL,100
"""

EXPECTED_PIPELINE_POSITIONS = {
    "RELIANCE": 60,
    "TCS": -50,
    "INFY": 0,
    "HDFCBANK": 10,
}


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _stop(proc: subprocess.Popen) -> None:
    if proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=5)


def _wait_ready(url: str, proc: subprocess.Popen, timeout: float = READY_TIMEOUT_SECONDS) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"Position service exited early with code {proc.returncode}")
        try:
            response = httpx.get(url, timeout=0.25)
            if response.status_code == 200:
                return
        except httpx.RequestError:
            pass
        time.sleep(0.05)
    raise TimeoutError(f"Position service did not become ready at {url}")


@contextmanager
def running_position_service() -> Iterator[int]:
    port = _free_port()
    proc = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "src.position_service",
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
        ],
        cwd=PROJECT_ROOT,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        env=os.environ.copy(),
    )
    try:
        _wait_ready(f"http://127.0.0.1:{port}/position", proc)
        yield port
    finally:
        _stop(proc)


def _run_order_update(
    input_file: Path,
    port: int,
    *,
    rate_limit: int = 200,
    timeout: float = 2.0,
) -> subprocess.CompletedProcess[str]:
    proc = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "src.order_update_service",
            "--input-file",
            str(input_file),
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--rate-limit",
            str(rate_limit),
            "--timeout",
            str(timeout),
        ],
        cwd=PROJECT_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=os.environ.copy(),
    )
    try:
        stdout, _ = proc.communicate(timeout=PRODUCER_TIMEOUT_SECONDS)
        return subprocess.CompletedProcess(proc.args, proc.returncode, stdout, None)
    finally:
        _stop(proc)


def test_end_to_end_pipeline_across_two_processes(tmp_path: Path) -> None:
    csv_path = tmp_path / "pipeline.csv"
    csv_path.write_text(PIPELINE_CSV, encoding="utf-8")

    with running_position_service() as port:
        result = _run_order_update(csv_path, port, rate_limit=200, timeout=2.0)
        response = httpx.get(f"http://127.0.0.1:{port}/position", timeout=2.0)

    assert result.returncode == 0, result.stdout
    assert "event_id=evt-0001 accepted" in result.stdout
    assert "event_id=evt-0001 successfully sent" in result.stdout
    assert "event_id=evt-0007 rejected" in result.stdout
    assert "HOLD" in result.stdout or "transaction_type" in result.stdout
    assert "event_id=evt-0001 delivered but ignored as duplicate" in result.stdout
    assert "event_id=evt-0008 successfully sent" in result.stdout
    assert "Input processing complete" in result.stdout
    assert "successfully sent" in result.stdout

    assert response.status_code == 200
    assert response.json() == EXPECTED_PIPELINE_POSITIONS


def test_end_to_end_position_service_unavailable(tmp_path: Path) -> None:
    csv_path = tmp_path / "one-row.csv"
    csv_path.write_text(
        "event_id,symbol,transaction_type,quantity\n"
        "evt-0001,RELIANCE,BUY,90\n"
        "evt-0002,TCS,SELL,75\n",
        encoding="utf-8",
    )
    closed_port = _free_port()

    result = _run_order_update(csv_path, closed_port, rate_limit=200, timeout=1.0)

    assert result.returncode == 0, result.stdout
    assert "event_id=evt-0001 accepted" in result.stdout
    assert "event_id=evt-0001 delivery failed" in result.stdout
    assert "event_id=evt-0002 delivery failed" in result.stdout
    assert "successfully sent" not in result.stdout
    assert "Input processing complete" in result.stdout
