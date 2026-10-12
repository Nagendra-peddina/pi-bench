#!/usr/bin/env python3
"""
prepare_dataset.py - fixed capture window, features and a capacitor-level
train/test split for the Raspberry Pi inference benchmark.

Input   data/processed/waveforms.csv  (made by extract_waveforms2.py, --min-valid 190)
Output  data/processed/dataset.npz    X_train, y_train, X_test, y_test (+ capacitor ids)
                                       -> NOT in Git; this script regenerates it
        data/split/split.json         train/test capacitors, window, normalisation
                                       -> IN Git; this is what makes the split "fixed"

Run from the repo root:
    python prep/prepare_dataset.py

Re-running reuses the split saved in split.json. Use --resplit only if you
deliberately want a new split (and then re-train every model).
"""
import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SAMPLE_COL = re.compile(r"^v(\d+)$")      # v000, v001, ... from extract_waveforms2.py
META_COLS = ["stress_v", "capacitor", "label"]


def sha256(path: Path) -> str:
    """Fingerprint of the input CSV, so a changed file is noticed later."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def pick_test_caps(cap_class: pd.Series, per_class: int, seed: int) -> list:
    """Hold out `per_class` whole capacitors from each class (0 healthy, 1 degraded)."""
    rng = np.random.default_rng(seed)
    test = []
    for cls in (0, 1):
        caps = sorted(cap_class[cap_class == cls].index)
        if len(caps) <= per_class:
            sys.exit(f"Only {len(caps)} capacitor(s) in class {cls}; cannot hold "
                     f"out {per_class} and still have one left to train on.")
        test += [str(c) for c in rng.choice(caps, size=per_class, replace=False)]
    return sorted(test)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", default="data/processed/waveforms.csv")
    ap.add_argument("--stress-v", type=float, default=14.0,
                    help="stress group to use (default 14 V = ES14)")
    ap.add_argument("--window", type=int, default=190,
                    help="samples kept per waveform = the fixed capture window")
    ap.add_argument("--test-per-class", type=int, default=1,
                    help="capacitors held out for testing, per class")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--split", default="data/split/split.json")
    ap.add_argument("--out", default="data/processed/dataset.npz")
    ap.add_argument("--resplit", action="store_true",
                    help="ignore an existing split.json and draw a new split")
    args = ap.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        sys.exit(f"{csv_path} not found - copy waveforms.csv into data/processed/ first.")
    csv_hash = sha256(csv_path)

    # 1. Load and check the columns --------------------------------------------
    df = pd.read_csv(csv_path)
    missing = [c for c in META_COLS if c not in df.columns]
    if missing:
        sys.exit(f"Missing columns {missing}. First columns found: {list(df.columns)[:12]}")
    df["capacitor"] = df["capacitor"].astype(str)
    sample_cols = sorted((c for c in df.columns if SAMPLE_COL.match(c)),
                         key=lambda c: int(SAMPLE_COL.match(c).group(1)))
    if len(sample_cols) < args.window:
        sys.exit(f"Only {len(sample_cols)} sample columns, but window is {args.window}.")
    print(f"loaded {len(df)} waveforms with {len(sample_cols)} sample columns")

    # 2. Keep one stress group ---------------------------------------------------
    df = df[np.isclose(df["stress_v"].astype(float), args.stress_v)].copy()
    if df.empty:
        sys.exit(f"No rows with stress_v = {args.stress_v}.")
    print(f"{len(df)} waveforms at {args.stress_v:g} V")

    # 3. Fixed capture window: first N samples, all of them finite ---------------
    win_cols = sample_cols[: args.window]
    X_all = df[win_cols].to_numpy(dtype=np.float32)
    ok = np.isfinite(X_all).all(axis=1)
    if (~ok).any():
        print(f"dropped {int((~ok).sum())} waveforms with NaN inside the window")
    df, X_all = df[ok], X_all[ok]
    y_all = df["label"].to_numpy(dtype=np.int64)
    caps_all = df["capacitor"].to_numpy()

    # 4. Class per capacitor = majority label (only used to balance the split) ---
    stats = df.groupby("capacitor")["label"].agg(["mean", "count"])
    cap_class = (stats["mean"] >= 0.5).astype(int)
    print("\nper capacitor (mean = fraction of waveforms labelled degraded):")
    print(stats.to_string())

    # 5. Split by capacitor; reuse the saved split so it stays fixed -------------
    split_path = Path(args.split)
    if split_path.exists() and not args.resplit:
        saved = json.loads(split_path.read_text())
        test_caps = sorted(saved["test_capacitors"])
        print(f"\nusing saved split from {split_path}")
        if saved.get("source_sha256") != csv_hash:
            print("WARNING: waveforms.csv changed since this split was made.")
    else:
        test_caps = pick_test_caps(cap_class, args.test_per_class, args.seed)
        print(f"\nnew split drawn with seed {args.seed}")
    unknown = set(test_caps) - set(cap_class.index)
    if unknown:
        sys.exit(f"Test capacitors {sorted(unknown)} are not in the data. "
                 "Check the CSV, or run with --resplit.")
    train_caps = sorted(set(cap_class.index) - set(test_caps))
    assert not set(train_caps) & set(test_caps), "a capacitor is in both splits"
    is_test = np.isin(caps_all, test_caps)

    X_train, y_train = X_all[~is_test], y_all[~is_test]
    X_test, y_test = X_all[is_test], y_all[is_test]

    # 6. Features: z-score using constants from the TRAINING set only ------------
    mean = float(X_train.mean())
    std = float(X_train.std())
    X_train = ((X_train - mean) / std).astype(np.float32)
    X_test = ((X_test - mean) / std).astype(np.float32)

    # 7. Save ---------------------------------------------------------------------
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, X_train=X_train, y_train=y_train,
                        X_test=X_test, y_test=y_test,
                        caps_train=caps_all[~is_test], caps_test=caps_all[is_test])

    counts = {
        "train": {"healthy": int((y_train == 0).sum()), "degraded": int((y_train == 1).sum())},
        "test": {"healthy": int((y_test == 0).sum()), "degraded": int((y_test == 1).sum())},
    }
    split = {
        "source_csv": str(csv_path),
        "source_sha256": csv_hash,
        "stress_v": args.stress_v,
        "window_samples": args.window,
        "seed": args.seed,
        "train_capacitors": train_caps,
        "test_capacitors": test_caps,
        "normalisation": {"mean": mean, "std": std},
        "counts": counts,
    }
    split_path.parent.mkdir(parents=True, exist_ok=True)
    split_path.write_text(json.dumps(split, indent=2) + "\n")

    # 8. Report ---------------------------------------------------------------------
    print(f"\ntrain capacitors: {train_caps}")
    print(f"test capacitors:  {test_caps}")
    print(f"X_train {X_train.shape}   healthy {counts['train']['healthy']:>5}   "
          f"degraded {counts['train']['degraded']:>5}")
    print(f"X_test  {X_test.shape}   healthy {counts['test']['healthy']:>5}   "
          f"degraded {counts['test']['degraded']:>5}")
    print(f"normalisation: mean={mean:.6g}  std={std:.6g}")
    print(f"\nwrote {out}  (not in Git)")
    print(f"wrote {split_path}  (commit this file)")


if __name__ == "__main__":
    main()
