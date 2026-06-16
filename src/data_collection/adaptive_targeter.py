"""
Adaptive exact targeting calibration using production synthetic dynamics.

This is for expensive outer-planet calibration. It reuses prior exact-targeter
CSV rows, then bisects TOI_V at fixed AOP values to find the success band with
far fewer full-trajectory integrations than a dense grid.
"""

from __future__ import annotations

import argparse
import csv
import glob
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_collection.gmat_runner import MissionConfig
from src.data_collection.mars_targeter import evaluate, make_params


FIELDNAMES = [
    "target",
    "toi_v",
    "aop",
    "label",
    "failure_type",
    "min_target_rmag",
    "rad_per",
    "closest_day",
    "seconds",
    "source",
]


def _key(target: str, toi_v: float, aop: float) -> tuple[str, float, float]:
    return (target, round(float(toi_v), 8), round(float(aop), 8))


def _read_rows(paths: list[Path]) -> dict[tuple[str, float, float], dict[str, object]]:
    rows: dict[tuple[str, float, float], dict[str, object]] = {}
    for path in paths:
        if not path.exists() or path.stat().st_size == 0:
            continue
        with path.open("r", newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if not row:
                    continue
                row.setdefault("source", str(path))
                rows[_key(row["target"], float(row["toi_v"]), float(row["aop"]))] = row
    return rows


def _append_row(path: Path, row: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        if write_header:
            writer.writeheader()
        writer.writerow({name: row.get(name, "") for name in FIELDNAMES})


def _as_float(row: dict[str, object], name: str) -> float:
    return float(row[name])


def _evaluate_cached(
    target: str,
    toi_v: float,
    aop: float,
    out_csv: Path,
    cache: dict[tuple[str, float, float], dict[str, object]],
    source: str,
) -> dict[str, object]:
    k = _key(target, toi_v, aop)
    if k in cache:
        row = dict(cache[k])
        row["source"] = row.get("source") or "cache"
        print(
            f"cache V={toi_v:.8f} AOP={aop:.8f} label={row['label']} "
            f"{row['failure_type']} rad_per={float(row['rad_per']):.1f}",
            flush=True,
        )
        return row

    t0 = time.time()
    result = evaluate(
        make_params(target, toi_v, aop),
        far_dt=900.0,
        near_dt=300.0,
        fine_dt=30.0,
    )
    row = {
        "target": target,
        "toi_v": round(toi_v, 8),
        "aop": round(aop, 8),
        "label": int(result.label),
        "failure_type": result.failure_type,
        "min_target_rmag": result.min_target_rmag,
        "rad_per": result.rad_per,
        "closest_day": result.closest_day,
        "seconds": round(time.time() - t0, 3),
        "source": source,
    }
    cache[k] = row
    _append_row(out_csv, row)
    print(
        f"eval  V={toi_v:.8f} AOP={aop:.8f} label={row['label']} "
        f"{row['failure_type']} min={row['min_target_rmag']:.1f} "
        f"rad_per={row['rad_per']:.1f} dt={row['seconds']}s",
        flush=True,
    )
    return row


def _is_success(row: dict[str, object]) -> bool:
    return int(row["label"]) == 1


def _above_band(row: dict[str, object], cfg: MissionConfig) -> bool:
    return _as_float(row, "rad_per") > cfg.max_radper


def _below_band(row: dict[str, object], cfg: MissionConfig) -> bool:
    return _as_float(row, "rad_per") < cfg.min_radper


def run(args: argparse.Namespace) -> None:
    out_csv = Path(args.output_csv)
    cache_paths = [Path(p) for p in glob.glob(args.cache_glob)]
    cache_paths.append(out_csv)
    cache = _read_rows(cache_paths)
    cfg = MissionConfig("earth", args.target)

    if args.aop_values:
        aop_values = [float(v.strip()) for v in args.aop_values.split(",") if v.strip()]
    else:
        import numpy as np

        aop_values = [float(v) for v in np.linspace(args.aop_min, args.aop_max, args.aop_count)]

    print("=" * 72, flush=True)
    print("Adaptive exact targeter", flush=True)
    print(f"target={args.target}", flush=True)
    print(f"success_radper=[{cfg.min_radper:.1f}, {cfg.max_radper:.1f}]", flush=True)
    print(f"v_bracket=[{args.v_low:.8f}, {args.v_high:.8f}]", flush=True)
    print(f"aop_values={','.join(f'{a:.8f}' for a in aop_values)}", flush=True)
    print(f"cache_rows={len(cache)} output_csv={out_csv}", flush=True)
    print("=" * 72, flush=True)

    successes: list[dict[str, object]] = []
    for aop in aop_values:
        low_v = args.v_low
        high_v = args.v_high
        low = _evaluate_cached(args.target, low_v, aop, out_csv, cache, "low")
        high = _evaluate_cached(args.target, high_v, aop, out_csv, cache, "high")

        if _is_success(low):
            successes.append(low)
            continue
        if _is_success(high):
            successes.append(high)
            continue
        if not (_above_band(low, cfg) and _below_band(high, cfg)):
            print(
                f"skip  AOP={aop:.8f}: no high-to-low bracket "
                f"low={low['failure_type']} rad={float(low['rad_per']):.1f}, "
                f"high={high['failure_type']} rad={float(high['rad_per']):.1f}",
                flush=True,
            )
            continue

        for iteration in range(1, args.max_iter + 1):
            mid_v = (low_v + high_v) / 2.0
            mid = _evaluate_cached(args.target, mid_v, aop, out_csv, cache, f"bisect_{iteration}")
            if _is_success(mid):
                successes.append(mid)
                print(f"SUCCESS AOP={aop:.8f} V={mid_v:.8f}", flush=True)
                if args.stop_on_success:
                    break
                low_v = mid_v
                low = mid
                continue
            if _above_band(mid, cfg):
                low_v = mid_v
                low = mid
            elif _below_band(mid, cfg):
                high_v = mid_v
                high = mid
            else:
                print(
                    f"warning unexpected non-success inside band "
                    f"V={mid_v:.8f} AOP={aop:.8f} type={mid['failure_type']}",
                    flush=True,
                )
                break
        if successes and args.stop_on_success:
            break

    rows = list(cache.values())
    target_mid = (cfg.min_radper + cfg.max_radper) / 2.0
    rows.sort(key=lambda r: (int(r["label"]) == 0, abs(float(r["rad_per"]) - target_mid)))
    print("\nTop adaptive candidates:", flush=True)
    for row in rows[: args.top]:
        print(
            f"label={row['label']} {row['failure_type']:<14} "
            f"V={float(row['toi_v']):.8f} AOP={float(row['aop']):.8f} "
            f"rad_per={float(row['rad_per']):.1f} min={float(row['min_target_rmag']):.1f}",
            flush=True,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Adaptive exact targeter.")
    parser.add_argument("--target", default="neptune")
    parser.add_argument("--v-low", type=float, required=True)
    parser.add_argument("--v-high", type=float, required=True)
    parser.add_argument("--aop-values")
    parser.add_argument("--aop-min", type=float)
    parser.add_argument("--aop-max", type=float)
    parser.add_argument("--aop-count", type=int)
    parser.add_argument("--max-iter", type=int, default=8)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument(
        "--cache-glob",
        default="reports/calibration/neptune_exact*_20260603.csv",
    )
    parser.add_argument("--top", type=int, default=20)
    parser.add_argument("--stop-on-success", action="store_true")
    args = parser.parse_args()
    if not args.aop_values and (
        args.aop_min is None or args.aop_max is None or args.aop_count is None
    ):
        parser.error("provide --aop-values or --aop-min/--aop-max/--aop-count")
    run(args)


if __name__ == "__main__":
    main()
