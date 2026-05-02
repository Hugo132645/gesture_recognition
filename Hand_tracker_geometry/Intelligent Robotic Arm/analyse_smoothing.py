# ===========================
# analyze_smoothing.py
# Professional analysis script
# ===========================
from __future__ import annotations

import os
import glob
import json
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


FINGERS = ["THUMB", "INDEX", "MIDDLE", "RING", "PINKY"]


@dataclass
class RunData:
    name: str
    raw: pd.DataFrame
    quant: pd.DataFrame
    calib: Optional[dict] = None


def load_csv_pair(base_no_suffix: str) -> RunData:
    """
    base_no_suffix = path without _raw.csv/_quant.csv suffix
    Example: runs/off_trial1  -> runs/off_trial1_raw.csv and runs/off_trial1_quant.csv
    """
    raw_path = base_no_suffix + "_raw.csv"
    quant_path = base_no_suffix + "_quant.csv"
    if not os.path.exists(raw_path) or not os.path.exists(quant_path):
        raise FileNotFoundError(f"Missing raw/quant for base: {base_no_suffix}")

    raw = pd.read_csv(raw_path)
    quant = pd.read_csv(quant_path)

    # Normalize time to seconds from start (helps alignment)
    for df in (raw, quant):
        if "t" not in df.columns:
            raise ValueError(f"Missing 't' column in {base_no_suffix}")
        df["t"] = df["t"] - df["t"].iloc[0]

    # Optional: load calibration if present in working dir
    calib = None
    if os.path.exists("servo_calib.txt"):
        try:
            with open("servo_calib.txt", "r", encoding="utf-8") as f:
                calib = json.load(f)
        except Exception:
            calib = None

    return RunData(name=os.path.basename(base_no_suffix), raw=raw, quant=quant, calib=calib)


def align_on_time(dfs: Dict[str, pd.DataFrame], dt: float = 1/60.0) -> Tuple[np.ndarray, Dict[str, pd.DataFrame]]:
    """
    Resample all trajectories onto a common time grid using linear interpolation.
    Returns (t_grid, aligned_dfs).
    """
    t_max = min(df["t"].iloc[-1] for df in dfs.values())
    t_grid = np.arange(0.0, t_max, dt)

    aligned = {}
    for k, df in dfs.items():
        out = pd.DataFrame({"t": t_grid})
        for col in df.columns:
            if col == "t":
                continue
            # interpolate
            out[col] = np.interp(t_grid, df["t"].to_numpy(), df[col].to_numpy())
        aligned[k] = out
    return t_grid, aligned


def normalized_detected(raw: pd.DataFrame, calib: Optional[dict]) -> pd.DataFrame:
    """
    Convert detected angles (deg) into a normalized [0,1] fraction using calibration open/fist angles.
    If calib is missing, returns z-scored-ish normalization per finger (still usable for comparisons).
    """
    out = pd.DataFrame({"t": raw["t"].to_numpy()})
    if calib and isinstance(calib, dict) and "calib" in calib:
        c = calib["calib"]
        for f in FINGERS:
            a = raw[f"{f}_angle"].to_numpy()
            a_open = float(c[f]["open_angle"])
            a_fist = float(c[f]["fist_angle"])
            denom = (a_fist - a_open) if abs(a_fist - a_open) > 1e-6 else 1.0
            tdet = (a - a_open) / denom
            out[f"{f}_t_det"] = np.clip(tdet, 0.0, 1.0)
    else:
        for f in FINGERS:
            a = raw[f"{f}_angle"].to_numpy()
            mu = float(np.nanmean(a))
            sd = float(np.nanstd(a) + 1e-6)
            out[f"{f}_t_det"] = (a - mu) / sd
    return out


def normalized_commanded(quant: pd.DataFrame, calib: Optional[dict]) -> pd.DataFrame:
    """
    Convert commanded servo degrees into a normalized [0,1] fraction using servo open/close degrees.
    If calib missing: normalize per finger min/max in-run.
    """
    out = pd.DataFrame({"t": quant["t"].to_numpy()})
    if calib and isinstance(calib, dict) and "calib" in calib:
        c = calib["calib"]
        for f in FINGERS:
            s = quant[f"{f}_servo"].to_numpy()
            s_open = float(c[f]["servo_open"])
            s_close = float(c[f]["servo_close"])
            denom = (s_close - s_open) if abs(s_close - s_open) > 1e-6 else 1.0
            tcmd = (s - s_open) / denom
            out[f"{f}_t_cmd"] = np.clip(tcmd, 0.0, 1.0)
    else:
        for f in FINGERS:
            s = quant[f"{f}_servo"].to_numpy()
            lo, hi = float(np.nanmin(s)), float(np.nanmax(s))
            denom = (hi - lo) if abs(hi - lo) > 1e-6 else 1.0
            out[f"{f}_t_cmd"] = (s - lo) / denom
    return out


def hf_energy_ratio(x: np.ndarray, fs: float, split_hz: float = 3.0) -> float:
    """
    High-frequency energy ratio using FFT:
    HF ratio = energy(f > split_hz) / energy(all f > 0)
    """
    x = np.asarray(x, dtype=float)
    x = x - np.nanmean(x)
    n = len(x)
    if n < 8:
        return float("nan")

    freqs = np.fft.rfftfreq(n, d=1.0/fs)
    X = np.fft.rfft(x)
    power = (np.abs(X) ** 2)

    mask_all = freqs > 0.0
    mask_hf = freqs >= split_hz
    denom = float(np.sum(power[mask_all]) + 1e-12)
    numer = float(np.sum(power[mask_hf]))
    return numer / denom


def detect_fast_segments(x: np.ndarray, t: np.ndarray, speed_thresh: float) -> np.ndarray:
    """
    Boolean mask of 'fast' samples where |dx/dt| > speed_thresh.
    """
    dt = np.gradient(t)
    dx = np.gradient(x)
    v = np.abs(dx / np.maximum(dt, 1e-6))
    return v > speed_thresh


def step_metrics(t: np.ndarray, x: np.ndarray, min_step: float = 0.35) -> Dict[str, float]:
    """
    Very lightweight step-response proxy:
    - Find largest change between start and end quartiles as a 'step amplitude'.
    - Compute time to reach 90% of final value from initial.
    - Overshoot relative to final.
    Works best for runs that contain at least one strong step.
    """
    n = len(x)
    if n < 20:
        return {"step_amp": np.nan, "t90": np.nan, "overshoot": np.nan}

    q = n // 4
    x0 = float(np.nanmedian(x[:q]))
    x1 = float(np.nanmedian(x[-q:]))
    amp = x1 - x0
    if abs(amp) < min_step:
        return {"step_amp": amp, "t90": np.nan, "overshoot": np.nan}

    target = x0 + 0.9 * amp
    if amp > 0:
        idx = np.where(x >= target)[0]
    else:
        idx = np.where(x <= target)[0]

    t90 = float(t[idx[0]]) if len(idx) else np.nan
    peak = float(np.nanmax(x) if amp > 0 else np.nanmin(x))
    overshoot = (peak - x1) / (abs(amp) + 1e-9)
    return {"step_amp": amp, "t90": t90, "overshoot": overshoot}


def summarize_run(run: RunData, resample_dt: float = 1/60.0) -> Dict[str, float]:
    """
    Compute per-run metrics aggregated over fingers (mean).
    """
    # Build normalized trajectories
    det = normalized_detected(run.raw, run.calib)
    cmd = normalized_commanded(run.quant, run.calib)

    # Align on common time
    t_grid, aligned = align_on_time({"det": det, "cmd": cmd}, dt=resample_dt)
    det_a, cmd_a = aligned["det"], aligned["cmd"]

    fs = 1.0 / resample_dt

    metrics = {}
    # Choose a "rest window": first 3 seconds (or first 20% if short)
    rest_end = min(3.0, t_grid[-1] * 0.2)
    rest_mask = t_grid <= rest_end

    jitters = []
    rmses = []
    hf_ratios = []
    t90s = []
    overs = []

    for f in FINGERS:
        xd = det_a[f"{f}_t_det"].to_numpy()
        xc = cmd_a[f"{f}_t_cmd"].to_numpy()

        # jitter at rest (detected)
        jitters.append(float(np.nanstd(xd[rest_mask])))

        # tracking RMSE between detected vs commanded
        rmses.append(float(np.sqrt(np.nanmean((xd - xc) ** 2))))

        # high-frequency ratio on detected (proxy jitter)
        hf_ratios.append(hf_energy_ratio(xd, fs=fs, split_hz=3.0))

        # step metrics on commanded (how aggressively we changed)
        sm = step_metrics(t_grid, xc, min_step=0.35)
        t90s.append(sm["t90"])
        overs.append(sm["overshoot"])

    metrics["jitter_rest_mean"] = float(np.nanmean(jitters))
    metrics["rmse_track_mean"] = float(np.nanmean(rmses))
    metrics["hf_ratio_mean"] = float(np.nanmean(hf_ratios))
    metrics["t90_mean"] = float(np.nanmean(t90s))
    metrics["overshoot_mean"] = float(np.nanmean(overs))
    return metrics


def plot_overlay_same_motion(runs: List[RunData], finger: str = "THUMB", resample_dt: float = 1/60.0) -> None:
    """
    One figure: detected normalized trajectory overlay for the same finger across runs.
    """
    dets = {}
    for r in runs:
        dets[r.name] = normalized_detected(r.raw, r.calib)
    t_grid, aligned = align_on_time(dets, dt=resample_dt)

    plt.figure()
    for name, df in aligned.items():
        plt.plot(df["t"], df[f"{finger}_t_det"], label=name)
    plt.xlabel("Time (s)")
    plt.ylabel(f"{finger} normalized detected (t_det)")
    plt.title(f"Overlay: {finger} detected trajectory across smoothing configs")
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_detected_vs_commanded(run: RunData, finger: str = "THUMB", resample_dt: float = 1/60.0) -> None:
    det = normalized_detected(run.raw, run.calib)
    cmd = normalized_commanded(run.quant, run.calib)
    t_grid, aligned = align_on_time({"det": det, "cmd": cmd}, dt=resample_dt)

    d = aligned["det"][f"{finger}_t_det"].to_numpy()
    c = aligned["cmd"][f"{finger}_t_cmd"].to_numpy()

    plt.figure()
    plt.plot(t_grid, d, label="detected (t_det)")
    plt.plot(t_grid, c, label="commanded (t_cmd)")
    plt.xlabel("Time (s)")
    plt.ylabel("Normalized fraction")
    plt.title(f"{run.name}: {finger} detected vs commanded (normalized)")
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_error_vs_speed(run: RunData, finger: str = "THUMB", speed_thresh: float = 0.6, resample_dt: float = 1/60.0) -> None:
    """
    Compare tracking error in slow vs fast segments based on detected speed.
    speed_thresh is in normalized-units per second.
    """
    det = normalized_detected(run.raw, run.calib)
    cmd = normalized_commanded(run.quant, run.calib)
    t_grid, aligned = align_on_time({"det": det, "cmd": cmd}, dt=resample_dt)

    d = aligned["det"][f"{finger}_t_det"].to_numpy()
    c = aligned["cmd"][f"{finger}_t_cmd"].to_numpy()
    e = d - c

    fast = detect_fast_segments(d, t_grid, speed_thresh=speed_thresh)

    plt.figure()
    plt.hist(np.abs(e[~fast]), bins=40, alpha=0.7, label="slow segments")
    plt.hist(np.abs(e[fast]), bins=40, alpha=0.7, label="fast segments")
    plt.xlabel("|tracking error| (normalized)")
    plt.ylabel("Count")
    plt.title(f"{run.name}: {finger} |error| distribution (slow vs fast)")
    plt.legend()
    plt.tight_layout()
    plt.show()


def pca_runs(metrics_table: pd.DataFrame) -> None:
    """
    PCA scatter plot of runs based on metric columns.
    """
    cols = [c for c in metrics_table.columns if c != "run"]
    X = metrics_table[cols].to_numpy(dtype=float)

    Xs = StandardScaler().fit_transform(X)
    Z = PCA(n_components=2).fit_transform(Xs)

    plt.figure()
    plt.scatter(Z[:, 0], Z[:, 1])
    for i, name in enumerate(metrics_table["run"].tolist()):
        plt.text(Z[i, 0], Z[i, 1], name, fontsize=9)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title("PCA of smoothing runs (based on metrics)")
    plt.tight_layout()
    plt.show()


def main():
    # 1) Discover runs by finding *_raw.csv files and pairing them with *_quant.csv
    raw_files = sorted(glob.glob("**/*_raw.csv", recursive=True))
    bases = [f[:-8] for f in raw_files]  # strip "_raw.csv"
    bases = [b for b in bases if os.path.exists(b + "_quant.csv")]

    if not bases:
        raise SystemExit("No runs found. Put logs like runs/name_raw.csv and runs/name_quant.csv")

    runs = [load_csv_pair(b) for b in bases]
    print(f"Loaded {len(runs)} runs")

    # 2) Compute metrics per run
    rows = []
    for r in runs:
        m = summarize_run(r)
        m["run"] = r.name
        rows.append(m)

    metrics_table = pd.DataFrame(rows).sort_values("run")
    print("\n=== Metrics summary ===")
    print(metrics_table.to_string(index=False))

    # 3) Key plots (edit which ones you want)
    # Overlay thumb across all configs (great for presentations)
    plot_overlay_same_motion(runs, finger="THUMB")

    # Detected vs commanded for each run (thumb)
    for r in runs:
        plot_detected_vs_commanded(r, finger="THUMB")

    # Error distribution slow vs fast (thumb)
    for r in runs:
        plot_error_vs_speed(r, finger="THUMB", speed_thresh=0.6)

    # PCA across runs
    pca_runs(metrics_table)


if __name__ == "__main__":
    main()
