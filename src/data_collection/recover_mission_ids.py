"""
Attach mission_id to the per-planet .npz extracts.

`extract_per_planet` stored X/y/failure_type/lengths in the order missions
appear in the parquet, but not the mission_ids themselves. The parquet is NOT
sorted by mission_id, so joining an extract to any other table by row position
silently pairs the wrong missions. This recovers the true order by streaming
only the mission_id column (a few hundred MB even for Neptune) and rewrites
each .npz with a `mission_ids` array.

Usage:
    python -m src.data_collection.recover_mission_ids \
        --data $ORBITGUARD_DATA/missions.parquet
"""

from __future__ import annotations

import argparse
import gc
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.dataset as ds_arrow

from src.ml.planet_config import PLANETS


def mission_id_order(dataset, planet: str, batch_size: int = 32_768) -> np.ndarray:
    """Unique mission_ids in the order they appear in the file for `planet`."""
    scanner = dataset.scanner(
        filter=ds_arrow.field("target_body") == planet,
        columns=["mission_id"],
        batch_size=batch_size,
        use_threads=False,
        batch_readahead=1,
        fragment_readahead=1,
    )
    order: list[int] = []
    last = None
    n = 0
    for batch in scanner.to_batches():
        if batch.num_rows == 0:
            continue
        mids = batch.column("mission_id").to_numpy()
        # keep first occurrence of each run; missions are contiguous
        change = np.flatnonzero(np.diff(mids)) + 1
        starts = np.concatenate(([0], change))
        for s in starts:
            mid = int(mids[s])
            if mid != last:
                order.append(mid)
                last = mid
        n += 1
        if n % 200 == 0:
            gc.collect()
            pa.default_memory_pool().release_unused()
    return np.array(order, dtype=np.int64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--dir", default="data/per_planet")
    ap.add_argument("--planets", nargs="+", default=PLANETS)
    args = ap.parse_args()

    dataset = ds_arrow.dataset(args.data, format="parquet")
    d = Path(args.dir)

    for planet in args.planets:
        f = d / f"{planet}.npz"
        if not f.exists():
            print(f"  {planet:9}: no extract — skipped")
            continue
        z = dict(np.load(f))
        if "mission_ids" in z:
            print(f"  {planet:9}: already has mission_ids ({len(z['mission_ids'])})")
            continue
        ids = mission_id_order(dataset, planet)
        if len(ids) != len(z["y"]):
            print(f"  {planet:9}: MISMATCH ids={len(ids)} rows={len(z['y'])} — skipped")
            continue
        z["mission_ids"] = ids
        np.savez_compressed(f, **z)
        print(f"  {planet:9}: attached {len(ids)} mission_ids "
              f"(sorted in file? {bool((np.diff(ids) > 0).all())})")
        del z
        gc.collect()


if __name__ == "__main__":
    main()
