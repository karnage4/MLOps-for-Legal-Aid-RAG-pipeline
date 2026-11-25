"""Check the local container without invoking retrieval or generation."""

import json
from pathlib import Path
import time
import urllib.error
import urllib.request

DEST = Path(__file__).resolve().parents[1] / "output" / "reproducibility-2026-11-25"
DEST.mkdir(parents=True, exist_ok=True)
BASE = "http://127.0.0.1:8000"


def request(route, body=None):
    data = json.dumps(body).encode() if body is not None else None
    query = urllib.request.Request(BASE + route, data=data, headers={"Content-Type": "application/json"})
    try:
        response = urllib.request.urlopen(query, timeout=120)
    except urllib.error.HTTPError as exc:
        response = exc
    with response:
        content = response.read()
        result = {"route": route, "http_status": response.status}
        if "application/json" in response.headers.get("Content-Type", ""):
            result["body"] = json.loads(content)
        return result


for attempt in range(24):
    try:
        health = request("/health")
        if health["http_status"] == 200:
            break
    except (urllib.error.URLError, ConnectionError):
        pass
    time.sleep(5)
else:
    raise RuntimeError("Container did not become ready within two minutes")

checks = [health] + [request(route) for route in ("/docs", "/ui/", "/metrics", "/stats")]
checks.append(request("/predict", {"query": ""}))
checks.append(request("/metrics"))
report = {"checks": checks}
(DEST / "container.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
print(json.dumps(report, indent=2, ensure_ascii=False))
assert all(row["http_status"] == ({"/predict": 422}.get(row["route"], 200)) for row in checks)
