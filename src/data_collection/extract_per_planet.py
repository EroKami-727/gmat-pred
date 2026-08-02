"""
Per-Planet Compact Extraction (memory-safe)
===========================================
Streams the 484M-row merged parquet one small batch at a time and writes a
compact per-planet .npz holding a (N, L, F) float64 array plus labels.

Why float64: the mission-to-mission discriminative signal is ~1e-5 relative
to the feature offsets. float32 storage of raw values (e.g. rel_x ~ 4.1e7)
resolves only ~2.5 km, which quantises that signal. Keep full precision here;
normalisation later brings values into a range where float32 is safe.

Sequence length is normalised across planets: each mission is downsampled to
~TARGET_STEPS rows regardless of native cadence, so Mercury (202 raw rows)
and Neptune (21,471 raw rows) both yield comparable sequences.

Usage:
    python -m src.data_collection.extract_per_planet \
        --data /media/Data/Coding/gmat-pred/data/merged_all_v2/missions.parquet \
        --out-dir data/per_planet
"""

from __future__ import annotations

import argparse
import gc
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.dataset as ds_arrow

from src.ml.dataset import FEATURE_COLS
from src.ml.planet_config import (
    FAILURE_TYPE_MAP, ROWS_PER_MISSION, TARGET_STEPS, downsample_for,
)


def _rss_gb() -> float:
    """Resident set size in GB (Linux), 0.0 if unavailable."""
    try:
        with open("/proc/self/statm") as f:
            return int(f.read().split()[1]) * 4096 / 1e9
    except Exception:
        return 0.0


def extract_planet(dataset, planet: str, ds: int, out_dir: Path, batch_size: int = 32_768):
    """Stream one planet, downsample every ds-th row per mission, save .npz."""
    cols = ["mission_id", "label", "failure_type", "elapsed_secs"] + FEATURE_COLS
    schema_names = dataset.schema.names
    cols = [c for c in cols if c in schema_names]

    # Readahead is the memory risk here, not our own accumulation: with default
    # prefetching, Neptune's 214M rows ballooned to 13.5 GB RSS and took the
    # desktop down. Disable threads/readahead so only one batch is ever live —
    # this stage is disk-bound anyway, so the throughput cost is minimal.
    scanner = dataset.scanner(
        filter=ds_arrow.field("target_body") == planet,
        columns=cols,
        batch_size=batch_size,
        use_threads=False,
        batch_readahead=1,
        fragment_readahead=1,
    )

    seqs: list[np.ndarray] = []
    labels: list[int] = []
    ftypes: list[int] = []

    cur_mid = None
    cur_rows: list[np.ndarray] = []
    cur_label = 0
    cur_ftype = 0
    row_phase = 0          # global row index within the current mission

    def flush():
        nonlocal cur_rows
        if cur_rows:
            seqs.append(np.concatenate(cur_rows, axis=0))
            labels.append(cur_label)
            ftypes.append(cur_ftype)
        cur_rows = []

    n_batches = 0
    for batch in scanner.to_batches():
        if batch.num_rows == 0:
            continue
        n_batches += 1
        mids = batch.column("mission_id").to_numpy()
        labs = batch.column("label").to_numpy()
        feats = np.column_stack([
            batch.column(c).to_numpy(zero_copy_only=False).astype(np.float64)
            for c in FEATURE_COLS
        ])
        # Only the first row of each mission carries the failure_type we need.
        # to_pylist() here would build ~32k throwaway Python strings per batch.
        ft_arr = (batch.column("failure_type")
                  if "failure_type" in batch.schema.names else None)

        # Mission boundaries inside this batch (data is sorted by mission_id)
        change = np.flatnonzero(np.diff(mids)) + 1
        starts = np.concatenate(([0], change))
        ends = np.concatenate((change, [len(mids)]))

        for s, e in zip(starts, ends):
            mid = mids[s]
            if mid != cur_mid:
                flush()
                cur_mid = mid
                cur_label = int(labs[s])
                ft_val = ft_arr[s].as_py() if ft_arr is not None else "unknown"
                cur_ftype = FAILURE_TYPE_MAP.get(str(ft_val), 7)
                row_phase = 0
            # keep rows whose global index within the mission is divisible by ds
            n = e - s
            first = (-row_phase) % ds
            if first < n:
                idx = np.arange(first, n, ds)
                cur_rows.append(feats[s + idx])
            row_phase = (row_phase + n) % ds

        del batch, mids, labs, feats, ft_arr
        # Arrow keeps freed blocks in its pool; without an explicit release the
        # RSS climbs ~2.5 MB/batch and OOMs on Neptune's 6.5k batches.
        if n_batches % 50 == 0:
            gc.collect()
            pa.default_memory_pool().release_unused()
        if n_batches % 500 == 0:
            print(f"      {planet}: {n_batches} batches, {len(seqs)} missions, "
                  f"RSS={_rss_gb():.2f} GB", flush=True)

    flush()

    if not seqs:
        print(f"  {planet}: no rows found — skipped")
        return

    # Keep true per-mission lengths: failed missions terminate early, so
    # truncating everything to min() would discard both data and that signal.
    lengths = np.array([len(s) for s in seqs], dtype=np.int64)
    L = int(lengths.max())
    F = len(FEATURE_COLS)
    X = np.zeros((len(seqs), L, F), dtype=np.float64)
    for i, s in enumerate(seqs):
        X[i, :len(s)] = s

    y = np.array(labels, dtype=np.int64)
    ft = np.array(ftypes, dtype=np.int64)

    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{planet}.npz"
    np.savez_compressed(out, X=X, y=y, failure_type=ft, lengths=lengths)
    mb = out.stat().st_size / 1e6
    print(f"  {planet:9s}: X={X.shape} len[min={lengths.min()} med={int(np.median(lengths))} "
          f"max={L}] fail={int((y==0).sum())} pass={int((y==1).sum())} → {out.name} ({mb:.0f} MB)")

    del X, y, ft, seqs, lengths
    gc.collect()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out-dir", default="data/per_planet")
    ap.add_argument("--planets", nargs="+", default=list(ROWS_PER_MISSION))
    ap.add_argument("--target-steps", type=int, default=TARGET_STEPS)
    args = ap.parse_args()

    dataset = ds_arrow.dataset(args.data, format="parquet")
    out_dir = Path(args.out_dir)

    print(f"\n[ Per-planet extraction → ~{args.target_steps} steps/mission ]\n")
    for planet in args.planets:
        rpm = ROWS_PER_MISSION.get(planet, 500)
        ds = downsample_for(planet, args.target_steps)
        print(f"  {planet:9s}: {rpm} rows/mission → ds={ds}", flush=True)
        extract_planet(dataset, planet, ds, out_dir)
        gc.collect()

    print("\nDone.\n")


if __name__ == "__main__":
    main()
