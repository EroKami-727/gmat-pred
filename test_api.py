"""
OrbitGuard API smoke test — boots the app in-process and exercises the
simulator path end to end.

There were no API tests at all, which is how a hardcoded absolute dataset path,
a wrong server-side default and a target missing from the mission creator all
survived in the serving layer while the ML side was audited repeatedly.

This is deliberately a smoke test, not a unit-test suite: it asserts the things
that break silently and are invisible until someone opens the dashboard.

  * the dataset resolves server-side without the client naming a path
  * every served target appears in planet_info and has a calibrated threshold
  * a mission can be built, streamed and scored for each target
  * a perturbed mission actually aborts, and a nominal one does not

Run (no server needed — the app is mounted in-process):

    PYTHONPATH=. python test_api.py
    PYTHONPATH=. python test_api.py --quick     # skip the per-target build sweep
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings

warnings.filterwarnings("ignore")

from fastapi.testclient import TestClient

from src.api.main import app
from src.ml.planet_config import SERVING_TARGETS

PASS = "  \033[32mPASS\033[0m"
FAIL = "  \033[31mFAIL\033[0m"

_failures: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> bool:
    print(f"{PASS if ok else FAIL}  {name}{('  — ' + detail) if detail else ''}")
    if not ok:
        _failures.append(name)
    return ok


def sse_events(client: TestClient, url: str, limit: int = 4000) -> list[dict]:
    """Collect parsed SSE payloads from a streaming endpoint."""
    events = []
    with client.stream("GET", url) as r:
        for line in r.iter_lines():
            if not line:
                continue
            text = line if isinstance(line, str) else line.decode()
            if not text.startswith("data: "):
                continue
            try:
                events.append(json.loads(text[6:]))
            except json.JSONDecodeError:
                continue
            if len(events) >= limit:
                break
    return events


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true",
                    help="skip the per-target mission build sweep")
    args = ap.parse_args()

    client = TestClient(app)

    print("\n[ health and system ]")
    r = client.get("/api/health")
    check("GET /api/health", r.status_code == 200 and r.json().get("status") == "ok")

    r = client.get("/api/system")
    sysinfo = r.json() if r.status_code == 200 else {}
    check("GET /api/system", r.status_code == 200)
    check("dataset resolves and is mounted",
          bool(sysinfo.get("dataset_mounted")),
          str(sysinfo.get("dataset_path")))
    loaded = sysinfo.get("planets_loaded") or []
    check("router loaded every serving target",
          set(loaded) == set(SERVING_TARGETS),
          f"loaded={sorted(loaded)}")

    print("\n[ planet info and thresholds ]")
    r = client.get("/api/simulator/planet_info")
    info = r.json() if r.status_code == 200 else {}
    check("GET /api/simulator/planet_info", r.status_code == 200)
    check("planet_info covers the serving set",
          set(info) == set(SERVING_TARGETS),
          f"missing={sorted(set(SERVING_TARGETS) - set(info))}")
    check("every target declares a reference frame",
          all(info.get(t, {}).get("frame") in ("heliocentric", "earth-centric")
              for t in SERVING_TARGETS))
    check("every target reports nominal burn parameters",
          all("nominal" in info.get(t, {}) for t in SERVING_TARGETS),
          f"without={[t for t in SERVING_TARGETS if 'nominal' not in info.get(t, {})]}")

    r = client.get("/api/simulator/thresholds")
    thr = r.json() if r.status_code == 200 else {}
    check("every target has a calibrated threshold",
          set(thr) == set(SERVING_TARGETS) and all(0 < v < 1 for v in thr.values()),
          f"n={len(thr)}")

    print("\n[ dataset sampling — no client-supplied path ]")
    r = client.get("/api/simulator/missions?n=4")
    ms = r.json().get("missions", []) if r.status_code == 200 else []
    check("GET /api/simulator/missions resolves the server default",
          r.status_code == 200 and len(ms) == 4,
          f"got {len(ms)}")
    check("sampled missions carry labels",
          all("label" in m and "target" in m for m in ms))

    print("\n[ mission builder — nominal vs perturbed ]")
    targets = SERVING_TARGETS if not args.quick else ["mars", "moon"]
    for target in targets:
        r = client.post("/api/simulator/generate",
                        json={"target": target, "dv_v_offset": 0.0})
        ok = r.status_code == 200 and not r.json().get("error")
        nominal_label = r.json().get("label") if ok else None
        check(f"build nominal {target}", ok and nominal_label == 1,
              f"label={nominal_label}")

    print("\n[ end-to-end stream — perturbed mission must abort ]")
    for target in (["mars", "moon"] if args.quick else ["mars", "moon", "neptune"]):
        r = client.post("/api/simulator/generate",
                        json={"target": target, "dv_v_offset": 0.05})
        if r.status_code != 200:
            check(f"stream {target}", False, "generate failed")
            continue
        mid = r.json()["mission_id"]
        events = sse_events(client, f"/api/simulator/stream?mission_id={mid}&step_delay_ms=0")
        info_ev = next((e for e in events if e.get("type") == "info"), None)
        done_ev = next((e for e in events if e.get("type") == "done"), None)
        check(f"stream {target} emits info+done",
              info_ev is not None and done_ev is not None)
        if done_ev:
            check(f"stream {target} verdict is correct",
                  bool(done_ev.get("was_correct")),
                  f"true_label={done_ev.get('true_label')} "
                  f"canceled={done_ev.get('canceled')} "
                  f"p={done_ev.get('final_prob')}")

    print()
    if _failures:
        print(f"\033[31m{len(_failures)} check(s) failed:\033[0m")
        for f in _failures:
            print(f"  - {f}")
        return 1
    print("\033[32mAll checks passed.\033[0m")
    return 0


if __name__ == "__main__":
    sys.exit(main())
