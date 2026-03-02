"""
OrbitGuard ML Dataset — Physics-Invariant Trajectory Sequences
================================================================
PyTorch Dataset and DataLoader for loading pre-generated Parquet
mission databases into tensors suitable for LSTM / Transformer training.

Key Design Decisions
--------------------
1. **Physics-Invariant Features Only**: Raw Cartesian pos_x/y/z are dropped.
   The model sees only: Synodic-frame coords, Specific Orbital Energy,
   Flight Path Angle, Normalized Target Distance, Radial Velocity, and
   Velocity Magnitude.

2. **Temporal Downsampling**: The raw 60-second timestep produces ~8640
   steps per mission. We downsample to a configurable interval (default
   15 minutes → ~576 steps) to keep sequences within LSTM memory limits.

3. **Early-Exit Slicing**: For ablation studies, the dataset can serve
   only the first N% of each trajectory, forcing the model to predict
   the outcome from truncated telemetry.

4. **Multiple Target Modes**:
   - 'binary'     → label (0/1)
   - 'multiclass' → failure_type encoded as int
   - 'regression' → min_target_rmag (closest approach in km)

5. **Scaling**: Fits a RobustScaler on training data only. The scaler
   is saved alongside the dataset for inference-time use.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Literal, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import RobustScaler


# ═══════════════════════════════════════════════════════════════════════════
# Feature Configuration
# ═══════════════════════════════════════════════════════════════════════════

# Physics-invariant features the model will see
FEATURE_COLS = [
    "rel_x", "rel_y", "rel_z",       # Synodic frame position
    "spec_energy",                     # Specific orbital energy
    "fpa_deg",                         # Flight path angle
    "norm_target_dist",                # Distance / SOI
    "radial_vel",                      # dr/dt toward target
    "vel_mag",                         # Speed magnitude
    "earth_rmag",                      # Distance from source body
    "ecc",                             # Eccentricity
    # ── Context features (constant per mission, enable cross-planet learning) ──
    "mu_ratio",                        # target μ / Sun μ
    "soi_ratio",                       # target SOI / transfer distance
    "dist_ratio",                      # transfer distance / 1 AU
]

# Failure type encoding for multi-class mode
FAILURE_TYPE_MAP = {
    "success":          0,
    "surface_impact":   1,
    "orbit_too_high":   2,
    "missed_target":    3,   # generalized from "missed_moon"
    "missed_moon":      3,   # backward compat alias
    "source_impact":    4,   # generalized from "earth_impact"
    "earth_impact":     4,   # backward compat alias
    "hyperbolic_flyby": 5,
    "degenerate_orbit": 6,
    "unknown":          7,
}


# ═══════════════════════════════════════════════════════════════════════════
# Dataset Class
# ═══════════════════════════════════════════════════════════════════════════

class TrajectoryDataset(Dataset):
    """
    PyTorch Dataset that loads trajectories from a Parquet database
    and returns fixed-length, downsampled, optionally truncated sequences.

    Parameters
    ----------
    parquet_path : str or Path
        Path to `missions.parquet`.
    target_mode : str
        'binary', 'multiclass', or 'regression'.
    downsample_factor : int
        Keep every N-th row (e.g., 15 for 60s→15min).
    max_seq_len : int or None
        Pad/truncate all sequences to this length. If None, use the
        longest sequence in the dataset.
    early_exit_frac : float
        Fraction of each trajectory to keep (0.0–1.0). Default 1.0 = full.
    scaler : RobustScaler or None
        Pre-fitted scaler. If None, one will be fitted on this data.
    """

    def __init__(
        self,
        parquet_path: str | Path,
        target_mode: Literal["binary", "multiclass", "regression"] = "binary",
        downsample_factor: int = 15,
        max_seq_len: Optional[int] = None,
        early_exit_frac: float = 1.0,
        scaler: Optional[RobustScaler] = None,
    ):
        self.target_mode = target_mode
        self.downsample_factor = downsample_factor
        self.early_exit_frac = early_exit_frac

        # ── Streaming Load (Memory-Safe) ──
        import pyarrow.parquet as pq
        pf = pq.ParquetFile(parquet_path)
        
        # Columns needed for features + labels
        needed_cols = list(set(FEATURE_COLS + ["mission_id", "label", "failure_type", "min_target_rmag", "elapsed_secs"]))
        needed_cols = [c for c in needed_cols if c in pf.schema_arrow.names]
        
        self.sequences = []
        self.labels = []
        self.mission_ids_ordered = []
        
        # Buffer for current mission building
        current_mission_id = None
        current_mission_rows = []
        
        print(f"  [Loading {pf.metadata.num_rows:,} rows via streaming batches...]")
        
        # Iterate in batches to save RAM
        for batch in pf.iter_batches(batch_size=500000, columns=needed_cols):
            # Using categories for string columns to save RAM
            df = batch.to_pandas(categories=["failure_type"])
            
            for mid, group in df.groupby("mission_id", sort=False):
                if mid != current_mission_id:
                    # Process completed mission
                    if current_mission_rows:
                        self._process_mission(pd.concat(current_mission_rows), current_mission_id)
                    
                    current_mission_id = mid
                    current_mission_rows = [group]
                else:
                    current_mission_rows.append(group)
                    
        # Process final mission
        if current_mission_rows:
            self._process_mission(pd.concat(current_mission_rows), current_mission_id)

        # ── Determine max sequence length ──
        lengths = [s.shape[0] for s in self.sequences]
        self.max_seq_len = max_seq_len or (max(lengths) if lengths else 0)

        # ── Fit or Apply Scaler ──
        if scaler is None and self.sequences:
            all_data = np.vstack(self.sequences)
            self.scaler = RobustScaler()
            self.scaler.fit(all_data)
        else:
            self.scaler = scaler

        # Scale and pad
        self._padded_sequences = []
        self._lengths = []
        for seq in self.sequences:
            scaled = self.scaler.transform(seq) if self.scaler else seq
            seq_len = scaled.shape[0]
            if seq_len >= self.max_seq_len:
                padded = scaled[:self.max_seq_len]
                self._lengths.append(self.max_seq_len)
            else:
                pad_width = self.max_seq_len - seq_len
                padded = np.pad(scaled, ((0, pad_width), (0, 0)), mode='constant')
                self._lengths.append(seq_len)
            self._padded_sequences.append(padded)

        # Convert to tensor storage
        if self._padded_sequences:
            self._X = torch.tensor(np.stack(self._padded_sequences), dtype=torch.float32)
            self._lengths_tensor = torch.tensor(self._lengths, dtype=torch.long)
            dtype = torch.float32 if target_mode == "regression" else torch.long
            self._y = torch.tensor(self.labels, dtype=dtype)
        else:
            # Empty dataset fallback
            self._X = torch.empty((0, self.max_seq_len or 1, len(FEATURE_COLS)))
            self._y = torch.empty((0,))
            self._lengths_tensor = torch.empty((0,))

    def _process_mission(self, mission_df: pd.DataFrame, mid: int):
        """Helper to downsample, slice, and store a completed mission."""
        mission_df = mission_df.sort_values("elapsed_secs")
        
        # 1. Extract Target
        row = mission_df.iloc[0]
        if self.target_mode == "binary":
            target = int(row["label"])
        elif self.target_mode == "multiclass":
            target = FAILURE_TYPE_MAP.get(row["failure_type"], 7)
        else: # regression
            target = float(row["min_target_rmag"])
            
        # 2. Downsample
        mission_df = mission_df.iloc[::self.downsample_factor].reset_index(drop=True)

        # 3. Early-exit slicing
        if self.early_exit_frac < 1.0:
            keep_n = max(1, int(len(mission_df) * self.early_exit_frac))
            mission_df = mission_df.iloc[:keep_n]

        # 4. Store
        features = mission_df[FEATURE_COLS].values.astype(np.float32)
        self.sequences.append(features)
        self.labels.append(target)
        self.mission_ids_ordered.append(mid)


    def __len__(self) -> int:
        return self._X.shape[0]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns (sequence, target, actual_length)."""
        return self._X[idx], self._y[idx], self._lengths_tensor[idx]

    @property
    def num_features(self) -> int:
        return len(FEATURE_COLS)

    @property
    def feature_names(self) -> list[str]:
        return list(FEATURE_COLS)

    def save_scaler(self, path: str | Path):
        """Save the fitted scaler for inference-time use."""
        with open(path, "wb") as f:
            pickle.dump(self.scaler, f)

    @staticmethod
    def load_scaler(path: str | Path) -> RobustScaler:
        """Load a previously saved scaler."""
        with open(path, "rb") as f:
            return pickle.load(f)


# ═══════════════════════════════════════════════════════════════════════════
# DataLoader Factory
# ═══════════════════════════════════════════════════════════════════════════

def create_dataloaders(
    parquet_path: str | Path,
    target_mode: Literal["binary", "multiclass", "regression"] = "binary",
    downsample_factor: int = 15,
    max_seq_len: Optional[int] = None,
    early_exit_frac: float = 1.0,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    batch_size: int = 32,
    seed: int = 42,
    num_workers: int = 0,
) -> Tuple[DataLoader, DataLoader, DataLoader, RobustScaler]:
    """
    Create train / val / test DataLoaders from a Parquet database.

    The scaler is fitted ONLY on the training set to prevent data leakage.

    Returns
    -------
    train_loader, val_loader, test_loader, scaler
    """
    # ── Memory-Safe ID Extraction ──
    import pyarrow.parquet as pq
    pf = pq.ParquetFile(parquet_path)
    # Just read mission_id column
    mission_ids = pf.read(columns=["mission_id"])["mission_id"].unique().to_numpy()

    rng = np.random.default_rng(seed)
    rng.shuffle(mission_ids)

    n_total = len(mission_ids)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)

    train_ids = set(mission_ids[:n_train])
    val_ids = set(mission_ids[n_train:n_train + n_val])
    test_ids = set(mission_ids[n_train + n_val:])

    # ── Memory-Safe Splitting ──
    # Instead of creating large DFs in RAM, we write batches
    parquet_dir = Path(parquet_path).parent
    train_path = parquet_dir / "_train_split.parquet"
    val_path = parquet_dir / "_val_split.parquet"
    test_path = parquet_dir / "_test_split.parquet"

    # Use ParquetWriter to split files row-by-row / batch-by-batch
    schema = pf.schema_arrow
    with pq.ParquetWriter(train_path, schema) as w_train, \
         pq.ParquetWriter(val_path, schema) as w_val, \
         pq.ParquetWriter(test_path, schema) as w_test:
        
        for batch in pf.iter_batches(batch_size=200000):
            # We can use pyarrow masks for speed and memory efficiency
            batch_mids = batch["mission_id"].to_numpy()
            
            # This is still a bit heavy but much better than loading the whole 42M rows
            w_train.write_batch(batch.filter(np.isin(batch_mids, list(train_ids))))
            w_val.write_batch(batch.filter(np.isin(batch_mids, list(val_ids))))
            w_test.write_batch(batch.filter(np.isin(batch_mids, list(test_ids))))

    # Build datasets — scaler fitted on train only
    train_ds = TrajectoryDataset(
        train_path, target_mode, downsample_factor,
        max_seq_len, early_exit_frac, scaler=None,
    )
    scaler = train_ds.scaler  # Fitted on training data

    # Determine max_seq_len from training set
    resolved_max_len = train_ds.max_seq_len

    val_ds = TrajectoryDataset(
        val_path, target_mode, downsample_factor,
        resolved_max_len, early_exit_frac, scaler=scaler,
    )
    test_ds = TrajectoryDataset(
        test_path, target_mode, downsample_factor,
        resolved_max_len, early_exit_frac, scaler=scaler,
    )

    # Clean up temp files
    train_path.unlink(missing_ok=True)
    val_path.unlink(missing_ok=True)
    test_path.unlink(missing_ok=True)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    print(f"  ─── Dataset Summary ───")
    print(f"  Target mode      : {target_mode}")
    print(f"  Downsample factor: {downsample_factor}x (60s → {downsample_factor}min)")
    print(f"  Early-exit frac  : {early_exit_frac:.0%}")
    print(f"  Max seq length   : {resolved_max_len}")
    print(f"  Features ({train_ds.num_features})   : {train_ds.feature_names}")
    print(f"  Train missions   : {len(train_ids)}")
    print(f"  Val missions     : {len(val_ids)}")
    print(f"  Test missions    : {len(test_ids)}")
    print(f"  Batch size       : {batch_size}")
    print(f"  X shape          : (batch, {resolved_max_len}, {train_ds.num_features})")

    return train_loader, val_loader, test_loader, scaler


# ═══════════════════════════════════════════════════════════════════════════
# CLI Smoke Test
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys
    data_path = sys.argv[1] if len(sys.argv) > 1 else "data/missions.parquet"

    print("=" * 60)
    print("  OrbitGuard ML Dataset — Smoke Test")
    print("=" * 60)

    train_loader, val_loader, test_loader, scaler = create_dataloaders(
        data_path,
        target_mode="binary",
        downsample_factor=15,
        early_exit_frac=0.5,  # First 50% of flight
        batch_size=16,
    )

    # Grab one batch
    for X, y, lengths in train_loader:
        print(f"\n  ─── Sample Batch ───")
        print(f"  X shape   : {X.shape}")     # (batch, seq_len, features)
        print(f"  y shape   : {y.shape}")     # (batch,)
        print(f"  lengths   : {lengths[:5]}")  # Actual (unpadded) lengths
        print(f"  y values  : {y[:8]}")
        print(f"  X[0, 0]   : {X[0, 0]}")     # First timestep, first mission
        break

    print("\n  ✓ Smoke test passed!")
    print("=" * 60)
