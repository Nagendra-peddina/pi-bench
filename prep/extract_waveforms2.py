#!/usr/bin/env python3
"""
extract_waveforms2.py - as before, but NaN-aware and it explains itself.

Usage:
    # diagnose first: where are the NaNs?
    python extract_waveforms2.py "data/12. Capacitor Electrical Stress" --diagnose

    # then extract
    python extract_waveforms2.py "data/12. Capacitor Electrical Stress" \
        --labels esr_capacitance_labelled.csv --per-cap 300 -o waveforms.csv

Changes from v1:
  * --diagnose samples waveforms and reports the NaN pattern (leading,
    trailing, scattered) instead of silently dropping everything
  * keeps a waveform if the finite part is long enough (--min-valid),
    trims trailing/leading NaN padding, interpolates isolated gaps
  * prints a breakdown of WHY rows were dropped
  * saves figs/example_waveforms.png so you can see the actual signal
"""

import argparse
import re
import sys
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

try:
    import h5py
    import pandas as pd
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    sys.exit("pip install h5py pandas matplotlib")


def to_str(ds):
    a = np.asarray(ds[()])
    if a.ndim <= 1 or 1 in a.shape:
        return "".join(chr(int(c)) for c in a.ravel() if int(c)).strip()
    if a.shape[0] > a.shape[1]:
        a = a.T
    return "".join(chr(int(c)) for c in a[0] if int(c)).strip()


def dt_to_datenum(dt):
    return dt.toordinal() + 366


def datenum_to_dt(d):
    return datetime.fromordinal(int(d) - 366) + timedelta(days=float(d) % 1)


def fit_time_origin(f, eis):
    if "EIS_Reference_Table" not in eis:
        return None
    refs = np.asarray(eis["EIS_Reference_Table"][()])
    if refs.shape[0] < 4:
        return None
    origins = []
    for c in range(refs.shape[1]):
        try:
            date_s = to_str(f[refs[0, c]])
            h = float(re.sub(r"[^0-9.\-]", "", to_str(f[refs[3, c]])))
            dt = None
            for fmt in ("%m/%d/%Y", "%m/%d/%y"):
                try:
                    dt = datetime.strptime(date_s, fmt)
                    break
                except ValueError:
                    continue
            if dt is not None:
                origins.append(dt_to_datenum(dt) - h / 24.0)
        except Exception:
            continue
    return float(np.median(origins)) if origins else None


def nan_profile(wf):
    """Describe where the NaNs are in one waveform."""
    bad = ~np.isfinite(wf)
    n = bad.sum()
    if n == 0:
        return "clean", 0
    lead = int(np.argmax(~bad)) if (~bad).any() else len(wf)
    trail = int(np.argmax(~bad[::-1])) if (~bad).any() else len(wf)
    inner = n - lead - trail
    if inner == 0 and trail > 0 and lead == 0:
        return "trailing", n
    if inner == 0 and lead > 0 and trail == 0:
        return "leading", n
    if inner == 0:
        return "both ends", n
    return "scattered", n


def clean_waveform(wf, min_valid):
    """Trim NaN padding, interpolate small gaps. Returns None if unusable."""
    finite = np.isfinite(wf)
    if not finite.any():
        return None
    first, last = int(np.argmax(finite)), len(wf) - int(np.argmax(finite[::-1]))
    core = wf[first:last]
    if core.size < min_valid:
        return None
    bad = ~np.isfinite(core)
    if bad.any():
        if bad.mean() > 0.2:
            return None
        idx = np.arange(core.size)
        core[bad] = np.interp(idx[bad], idx[~bad], core[~bad])
    return core


def diagnose(path, n=60):
    print(f"\n{path.name}")
    with h5py.File(path, "r") as f:
        root = f[[k for k in f.keys() if k != "#refs#"][0]]
        if "Transient_Data" not in root:
            return
        td = root["Transient_Data"]
        caps = [k for k in td.keys() if k != "Serial_Date"]
        for cap in caps[:2]:
            for sig in ("VO", "VL"):
                if sig not in td[cap]:
                    continue
                ds = td[cap][sig]
                idx = np.unique(np.linspace(0, ds.shape[0] - 1, n).astype(int))
                pats, counts, lens = Counter(), [], []
                for i in idx:
                    wf = np.asarray(ds[i, :], dtype=np.float64)
                    p, c = nan_profile(wf)
                    pats[p] += 1
                    counts.append(c)
                    lens.append(int(np.isfinite(wf).sum()))
                print(f"  {cap}/{sig}: patterns={dict(pats)}  "
                      f"NaNs/waveform median={int(np.median(counts))}  "
                      f"finite samples min={min(lens)} max={max(lens)}")


def plot_examples(path, out="figs/example_waveforms.png"):
    Path("figs").mkdir(exist_ok=True)
    with h5py.File(path, "r") as f:
        root = f[[k for k in f.keys() if k != "#refs#"][0]]
        td = root["Transient_Data"]
        caps = [k for k in td.keys() if k != "Serial_Date"]
        fig, ax = plt.subplots(figsize=(9, 4))
        for cap in caps[:4]:
            ds = td[cap]["VO"]
            wf = np.asarray(ds[ds.shape[0] // 2, :], dtype=np.float64)
            ax.plot(wf, lw=1, label=f"{cap} (raw, NaNs gapped)")
        ax.set_xlabel("sample index (0-399)")
        ax.set_ylabel("VO / V")
        ax.set_title("One charge/discharge waveform per capacitor")
        ax.grid(alpha=.3)
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(out, dpi=130)
        print(f"\nwrote {out}")


def extract(path, labels_df, per_cap, signal, min_valid, rows, reasons):
    stress_v = int(re.search(r"ES(\d+)", path.stem).group(1))
    with h5py.File(path, "r") as f:
        root = f[[k for k in f.keys() if k != "#refs#"][0]]
        if "Transient_Data" not in root:
            return
        td = root["Transient_Data"]
        t0 = fit_time_origin(f, root["EIS_Data"]) if "EIS_Data" in root else None
        if t0 is None:
            print(f"  ! {path.name}: no time origin")
            return
        sd = np.asarray(td["Serial_Date"][()]).ravel()
        for cap in [k for k in td.keys() if k != "Serial_Date"]:
            grp = td[cap]
            if signal not in grp:
                continue
            ds = grp[signal]
            lab = labels_df[(labels_df.stress_v == stress_v)
                            & (labels_df.capacitor == cap)]
            if lab.empty:
                reasons["no labels"] += 1
                continue
            lab_h = lab.stress_hours.to_numpy()
            lab_y = lab.degraded.to_numpy()
            lab_r = lab.cap_ratio.to_numpy()
            n = min(ds.shape[0], sd.size)
            idx = np.unique(np.linspace(0, n - 1, per_cap).astype(int))
            kept = 0
            for i in idx:
                wf = np.asarray(ds[i, :], dtype=np.float64)
                core = clean_waveform(wf, min_valid)
                if core is None:
                    reasons["too few finite samples"] += 1
                    continue
                # pad/trim to a fixed length so every row has same width
                out = np.full(min_valid, np.nan)
                out[:min(min_valid, core.size)] = core[:min_valid]
                if not np.all(np.isfinite(out)):
                    reasons["short after trim"] += 1
                    continue
                hours = (sd[i] - t0) * 24.0
                j = int(np.argmin(np.abs(lab_h - hours)))
                row = {"stress_v": stress_v, "capacitor": cap,
                       "wf_index": int(i), "stress_hours": round(hours, 2),
                       "label": int(lab_y[j]),
                       "cap_ratio": round(float(lab_r[j]), 4),
                       "label_gap_h": round(abs(float(lab_h[j] - hours)), 2)}
                row.update({f"v{k:03d}": round(float(v), 6)
                            for k, v in enumerate(out)})
                rows.append(row)
                kept += 1
            print(f"  {cap}: kept {kept}/{len(idx)}  "
                  f"({int(lab_y.sum())}/{len(lab_y)} EIS sessions degraded)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folder")
    ap.add_argument("--labels", default="esr_capacitance_labelled.csv")
    ap.add_argument("--per-cap", type=int, default=300)
    ap.add_argument("--signal", default="VO", choices=["VO", "VL"])
    ap.add_argument("--min-valid", type=int, default=300,
                    help="required finite samples per waveform")
    ap.add_argument("-o", "--out", default="waveforms.csv")
    ap.add_argument("--diagnose", action="store_true")
    args = ap.parse_args()

    files = sorted(Path(args.folder).glob("*.mat"))
    if not files:
        sys.exit(f"No .mat files in {args.folder}")

    if args.diagnose:
        print("=" * 66)
        print("NaN DIAGNOSIS (sampling 60 waveforms per capacitor)")
        print("=" * 66)
        for p in files:
            diagnose(p)
        plot_examples(files[-1])
        print("\nIf the pattern is 'trailing', these are padded buffers - "
              "normal.\nRe-run without --diagnose, setting --min-valid to "
              "just under the\nsmallest 'finite samples' value above.")
        return

    labels_df = pd.read_csv(args.labels)
    rows, reasons = [], Counter()
    print("=" * 66)
    print(f"EXTRACT {args.per_cap} '{args.signal}' waveforms/capacitor, "
          f"min {args.min_valid} finite samples")
    print("=" * 66)
    for p in files:
        print(f"\n{p.name}")
        extract(p, labels_df, args.per_cap, args.signal,
                args.min_valid, rows, reasons)

    print("\ndropped:", dict(reasons) or "nothing")
    if not rows:
        sys.exit("Nothing extracted - run with --diagnose.")
    df = pd.DataFrame(rows)
    df.to_csv(args.out, index=False)
    print(f"\nwrote {len(df)} rows x {df.shape[1]} cols -> {args.out}")
    print(f"  healthy {int((df.label == 0).sum())}, "
          f"degraded {int((df.label == 1).sum())}")
    print("\nper capacitor (mean label = fraction degraded):")
    print(df.groupby(["stress_v", "capacitor"]).label
            .agg(["mean", "count"]).to_string())


if __name__ == "__main__":
    main()
