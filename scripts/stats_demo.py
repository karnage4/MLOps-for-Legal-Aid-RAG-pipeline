"""Demonstrate GET /stats changing after a few sample questions.

Uses only the Python standard library. Run it against the local container or
the live service:

    python scripts/stats_demo.py
    python scripts/stats_demo.py --url https://legal-aid-rag-api.whitesand-d7432c32.swedencentral.azurecontainerapps.io

It waits for /health (a scaled-to-zero service may take a minute to wake),
prints /stats, sends three valid questions and one deliberately invalid one
(too short, so the API answers HTTP 422), then prints /stats again and a
before/after summary. It exits with status 1 if anything is unexpected.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request

DEFAULT_URL = "http://127.0.0.1:8000"
SAMPLE_QUESTIONS = (
    ("Who inherits property when a person dies without a will?", "recursive", "hybrid"),
    ("What share does a daughter receive in Islamic inheritance?", "fixed", "hybrid"),
    ("How are debts settled before distributing the estate?", "fixed", "semantic"),
)
INVALID_QUESTION = "hi"  # shorter than the 3-character minimum


def call(base: str, path: str, body: dict | None = None, timeout: float = 180) -> tuple[int, dict]:
    data = json.dumps(body).encode() if body is not None else None
    request = urllib.request.Request(
        base + path,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST" if body is not None else "GET",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.status, json.loads(response.read() or b"{}")
    except urllib.error.HTTPError as error:
        with error:
            try:
                return error.code, json.loads(error.read() or b"{}")
            except json.JSONDecodeError:
                return error.code, {}


def wait_until_healthy(base: str, attempts: int = 24) -> None:
    for attempt in range(1, attempts + 1):
        try:
            status, _ = call(base, "/health", timeout=60)
            if status == 200:
                return
        except (urllib.error.URLError, TimeoutError, ConnectionError):
            pass
        print(f"  waiting for {base}/health (attempt {attempt}/{attempts})...")
        time.sleep(5)
    raise SystemExit(f"{base}/health did not return 200")


def show(title: str, stats: dict) -> None:
    print(f"\n===== /stats {title} =====")
    print(json.dumps(stats, indent=2))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--url", default=DEFAULT_URL, help=f"service base URL (default {DEFAULT_URL})")
    base = parser.parse_args().url.rstrip("/")

    print(f"Service: {base}")
    wait_until_healthy(base)
    status, before = call(base, "/stats")
    if status != 200:
        print(f"/stats returned HTTP {status}")
        return 1
    show("BEFORE", before)

    print("\n===== Sample questions =====")
    problems = []
    for question, strategy, mode in SAMPLE_QUESTIONS:
        body = {"query": question, "chunking_strategy": strategy, "retrieval_mode": mode}
        started = time.perf_counter()
        status, reply = call(base, "/predict", body)
        elapsed = (time.perf_counter() - started) * 1000
        server_ms = reply.get("timings", {}).get("total_ms")
        print(f"  HTTP {status}  {strategy}/{mode}  client {elapsed:,.0f} ms  server {server_ms} ms  {question}")
        if status != 200:
            problems.append(f"valid question returned HTTP {status}")
    status, reply = call(base, "/predict", {"query": INVALID_QUESTION})
    print(f"  HTTP {status}  invalid question {INVALID_QUESTION!r} (expected 422)")
    if status != 422:
        problems.append(f"invalid question returned HTTP {status}, expected 422")

    status, after = call(base, "/stats")
    if status != 200:
        print(f"/stats returned HTTP {status}")
        return 1
    show("AFTER", after)

    old, new = before["queries"], after["queries"]
    print("\n===== Before -> after =====")
    print(f"  questions   {old['total']} -> {new['total']}")
    print(f"  succeeded   {old['succeeded']} -> {new['succeeded']}")
    print(f"  failed      {old['failed']} -> {new['failed']}")
    print(f"  error rate  {old['error_rate']} -> {new['error_rate']}")
    print(f"  p50 / p95   {old['latency_ms']['p50']} / {old['latency_ms']['p95']} ms"
          f" -> {new['latency_ms']['p50']} / {new['latency_ms']['p95']} ms")

    sent = len(SAMPLE_QUESTIONS) + 1
    if after["started_at"] != before["started_at"]:
        problems.append("the service restarted during the demo, so counters were reset")
    elif new["total"] - old["total"] < sent:
        problems.append(f"questions rose by {new['total'] - old['total']}, expected {sent}")
    if problems:
        print("\nProblems:\n  - " + "\n  - ".join(problems))
        return 1
    print("\nAll checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
