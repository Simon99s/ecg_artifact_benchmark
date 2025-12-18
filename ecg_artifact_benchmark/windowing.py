"""
ecg_artifact_benchmark/windowing.py

Step 0 — Windowing for MIT-BIH (and other WFDB datasets)

Design goals:
- Put all reusable logic here (NO plotting, NO experimentation).
- Notebooks/scripts should import these functions.
- Store window indices (start/end samples) rather than duplicating signal arrays.

Dependencies:
  pip install wfdb numpy pandas scipy

Typical MIT-BIH:
- fs ~ 360 Hz
- 2 leads (channels)
"""

from __future__ import annotations

from dataclasses import dataclass
from math import gcd
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import numpy as np
import pandas as pd
import wfdb

try:
    from scipy.signal import resample_poly
except ImportError:  # pragma: no cover
    resample_poly = None


@dataclass(frozen=True)
class WindowConfig:
    """Configuration for fixed-length sliding windows."""
    window_sec: float = 5.0
    overlap: float = 0.5              # 0.0.. <1.0
    target_fs: Optional[int] = None   # None = keep original
    lead: int = 0                     # channel index
    drop_last_partial: bool = True    # drop trailing partial window


def _require(condition: bool, msg: str) -> None:
    if not condition:
        raise ValueError(msg)


def _safe_resample_1d(x: np.ndarray, fs_in: int, fs_out: int) -> np.ndarray:
    """Resample a 1D signal using polyphase filtering (scipy)."""
    if fs_in == fs_out:
        return x
    if resample_poly is None:
        raise ImportError("scipy is required for resampling. Install with: pip install scipy")

    g = gcd(fs_in, fs_out)
    up = fs_out // g
    down = fs_in // g
    y = resample_poly(x, up=up, down=down)
    return y.astype(np.float32)


def load_wfdb_record(
    record: str,
    *,
    pn_dir: Optional[str] = "mitdb",
    local_dir: Optional[Union[str, Path]] = None,
    lead: int = 0,
    target_fs: Optional[int] = None,
) -> Tuple[np.ndarray, int, Dict]:
    """
    Load a WFDB record and return a single lead as float32 plus fs and metadata.

    Choose ONE of:
      - pn_dir="mitdb" (download from PhysioNet via wfdb)
      - local_dir="data/raw/mitdb" (read from local record files)

    Args:
        record: record name e.g. "100"
        pn_dir: PhysioNet database directory, e.g. "mitdb". If None, use local_dir.
        local_dir: path to local folder containing WFDB record files.
        lead: channel index to extract.
        target_fs: resample to this sampling rate (optional).

    Returns:
        x: (N,) float32 signal
        fs: int sampling rate
        meta: dict with basic record metadata
    """
    _require(record and isinstance(record, str), "record must be a non-empty string.")
    _require((pn_dir is None) ^ (local_dir is None) or (pn_dir is not None and local_dir is None),
             "Specify either pn_dir (PhysioNet) OR local_dir (local files).")

    if local_dir is not None:
        local_dir = Path(local_dir)
        rec = wfdb.rdrecord(str(local_dir / record))
    else:
        rec = wfdb.rdrecord(record_name=record, pn_dir=pn_dir)

    fs = int(rec.fs)
    sig = rec.p_signal
    if sig is None:
        raise RuntimeError("WFDB returned no p_signal. Check the record or WFDB settings.")

    _require(0 <= lead < sig.shape[1], f"lead={lead} out of range (n_channels={sig.shape[1]}).")

    x = sig[:, lead].astype(np.float32)

    if target_fs is not None:
        x = _safe_resample_1d(x, fs_in=fs, fs_out=int(target_fs))
        fs = int(target_fs)

    meta = {
        "record": record,
        "fs": fs,
        "n_samples": int(x.shape[0]),
        "n_channels": int(sig.shape[1]),
        "sig_name": getattr(rec, "sig_name", None),
        "units": getattr(rec, "units", None),
    }
    return x, fs, meta


def window_indices(
    n_samples: int,
    fs: int,
    cfg: WindowConfig,
) -> np.ndarray:
    """
    Create fixed-length window indices.

    Returns:
        idx: array shape (n_windows, 2) where each row is (start, end) samples, end exclusive.
    """
    _require(fs > 0, "fs must be > 0.")
    _require(n_samples > 0, "n_samples must be > 0.")
    _require(cfg.window_sec > 0, "window_sec must be > 0.")
    _require(0.0 <= cfg.overlap < 1.0, "overlap must be in [0, 1).")

    win_len = int(round(cfg.window_sec * fs))
    _require(win_len >= 2, "window length too small; increase window_sec or fs.")

    hop = int(round(win_len * (1.0 - cfg.overlap)))
    hop = max(1, hop)

    starts = np.arange(0, n_samples, hop, dtype=np.int64)
    ends = starts + win_len

    if cfg.drop_last_partial:
        mask = ends <= n_samples
        starts, ends = starts[mask], ends[mask]
    else:
        ends = np.minimum(ends, n_samples)

    idx = np.stack([starts, ends], axis=1)
    return idx


def build_window_table(
    record: str,
    idx: np.ndarray,
    fs: int,
) -> pd.DataFrame:
    """
    Build a window metadata table from indices.

    Columns:
      record, fs, win_id, start, end, start_sec, end_sec, duration_sec
    """
    _require(idx.ndim == 2 and idx.shape[1] == 2, "idx must have shape (n_windows, 2).")
    df = pd.DataFrame({
        "record": record,
        "fs": int(fs),
        "win_id": np.arange(idx.shape[0], dtype=np.int64),
        "start": idx[:, 0].astype(np.int64),
        "end": idx[:, 1].astype(np.int64),
    })
    df["start_sec"] = df["start"] / fs
    df["end_sec"] = df["end"] / fs
    df["duration_sec"] = (df["end"] - df["start"]) / fs
    return df


def window_record(
    record: str,
    cfg: WindowConfig,
    *,
    pn_dir: Optional[str] = "mitdb",
    local_dir: Optional[Union[str, Path]] = None,
) -> Tuple[np.ndarray, int, Dict, pd.DataFrame]:
    """
    Convenience wrapper: load record, compute windows, return x/fs/meta/window_table.

    Returns:
        x, fs, meta, windows_df
    """
    x, fs, meta = load_wfdb_record(
        record,
        pn_dir=pn_dir,
        local_dir=local_dir,
        lead=cfg.lead,
        target_fs=cfg.target_fs,
    )
    idx = window_indices(len(x), fs, cfg)
    win_df = build_window_table(record=record, idx=idx, fs=fs)
    return x, fs, meta, win_df
