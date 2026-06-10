"""
Exact targeting calibration using the production synthetic dynamics.

This intentionally uses the same 900/300/30 second adaptive integration
settings as run_synthetic() for interplanetary production runs. It checkpoints
every candidate to CSV.
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_collection.gmat_runner import MissionConfig
from src.data_collection.mars_targeter import evaluate, make_params


def _existing_keys(path: Path) -> set[tuple[float, float]]:
    if not path.exists():
        return set()
    keys: set[tuple[float, float]] = set()
    with path.open("r", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            keys.add((round(float(row["toi_v"]), 8), round(float(row["aop"]), 8)))
    return keys


def _append_row(path: Path, row: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def _sort_key(row: dict[str, object], target_radper: float) -> tuple[int, float, float]:
    label = int(row["label"])
    return (
        0 if label == 1 else 1,
        abs(float(row["rad_per"]) - target_radper),
        float(row["min_target_rmag"]),
    )


def run_grid(args: argparse.Namespace) -> None:
    cfg = MissionConfig("earth", args.target)
    target_radper = cfg.target_radius + 300.0
    out_csv = Path(args.output_csv)
    done = _existing_keys(out_csv) if args.resume else set()

    v_values = np.linspace(args.v_min, args.v_max, args.v_count)
    aop_values = np.linspace(args.aop_min, args.aop_max, args.aop_count)
    candidates = [(float(v), float(a)) for v in v_values for a in aop_values]
    remaining = [
        (v, a)
        for v, a in candidates
        if (round(v, 8), round(a, 8)) not in done
    ]

    print("=" * 72, flush=True)
    print("Exact production-path targeter", flush=True)
    print(f"target={args.target} time_step={args.time_step}", flush=True)
    print(f"grid={len(candidates)} already_done={len(done)} remaining={len(remaining)}", flush=True)
    print(f"output_csv={out_csv}", flush=True)
    print("=" * 72, flush=True)

    started = time.time()
    for idx, (toi_v, aop) in enumerate(remaining, start=1):
        t0 = time.time()
        result = evaluate(
            make_params(args.target, toi_v, aop),
            far_dt=900.0,
            near_dt=300.0,
            fine_dt=30.0,
        )
        row = {
            "target": args.target,
            "toi_v": round(toi_v, 8),
            "aop": round(aop, 8),
            "label": int(result.label),
            "failure_type": result.failure_type,
            "min_target_rmag": result.min_target_rmag,
            "rad_per": result.rad_per,
            "closest_day": result.closest_day,
            "seconds": round(time.time() - t0, 3),
        }
        _append_row(out_csv, row)
        print(
            f"{idx}/{len(remaining)} V={toi_v:.8f} AOP={aop:.8f} "
            f"label={row['label']} {row['failure_type']} "
            f"min={row['min_target_rmag']:.1f} rad_per={row['rad_per']:.1f} "
            f"dt={row['seconds']}s",
            flush=True,
        )

    rows: list[dict[str, object]] = []
    if out_csv.exists():
        with out_csv.open("r", newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))

    if rows:
        rows.sort(key=lambda r: _sort_key(r, target_radper))
        print("\nTop exact candidates:", flush=True)
        for row in rows[: args.top]:
            print(
                f"label={row['label']} {row['failure_type']:<14} "
                f"V={float(row['toi_v']):.8f} AOP={float(row['aop']):.8f} "
                f"min={float(row['min_target_rmag']):.1f} "
                f"rad_per={float(row['rad_per']):.1f}",
                flush=True,
            )

    print(f"\nElapsed total: {time.time() - started:.1f}s", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Exact production-path targeter.")
    parser.add_argument("--target", default="neptune")
    parser.add_argument("--v-min", type=float, required=True)
    parser.add_argument("--v-max", type=float, required=True)
    parser.add_argument("--v-count", type=int, required=True)
    parser.add_argument("--aop-min", type=float, required=True)
    parser.add_argument("--aop-max", type=float, required=True)
    parser.add_argument("--aop-count", type=int, required=True)
    parser.add_argument("--time-step", type=float, default=54000.0)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--top", type=int, default=20)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    run_grid(args)


if __name__ == "__main__":
    main()
