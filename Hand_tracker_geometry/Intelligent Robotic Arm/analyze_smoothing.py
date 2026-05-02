from __future__ import annotations

import os
import glob
from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# =========================
# CONFIG
# =========================
FINGERS = ["THUMB", "INDEX", "MIDDLE", "RING", "PINKY"]
FS = 60.0                # assumed frame rate
VIZ_SMOOTH_K = 7         # rolling window for visualization only
FAST_SPEED_THRESH = 0.4  # normalized units / second


# =========================
# HELPERS
# =========================
def viz_smooth(x, k=7):
    return pd.Series(x).rolling(k, center=True, min_periods=1).mean().to_numpy()


def detect_fast_segments(x, t, thresh):
    dx = np.gradient(x)
    dt = np.gradient(t)
    speed = np.abs(dx / np.maximum(dt, 1e-6))
    return speed > thresh


def hf_energy_ratio(x, fs, split_hz=3.0):
    x = x - np.nanmean(x)
    freqs = np.fft.rfftfreq(len(x), d=1 / fs)
    P = np.abs(np.fft.rfft(x)) ** 2
    return P[freqs >= split_hz].sum() / (P[freqs > 0].sum() + 1e-9)


def resample(df, t_grid):
    out = {"t": t_grid}
    for c in df.columns:
        if c != "t":
            out[c] = np.interp(t_grid, df["t"], df[c])
    return pd.DataFrame(out)


# =========================
# DATA LOADING
# =========================
@dataclass
class Run:
    name: str
    raw: pd.DataFrame
    quant: pd.DataFrame


def load_runs():
    runs = []
    for raw_path in glob.glob("**/*_raw.csv", recursive=True):
        base = raw_path[:-8]
        quant_path = base + "_quant.csv"
        if not os.path.exists(quant_path):
            continue
        raw = pd.read_csv(raw_path)
        quant = pd.read_csv(quant_path)
        raw["t"] -= raw["t"].iloc[0]
        quant["t"] -= quant["t"].iloc[0]
        runs.append(Run(os.path.basename(base), raw, quant))
    return runs


# =========================
# NORMALIZATION
# =========================
def normalize(raw, quant, finger):
    a = raw[f"{finger}_angle"].to_numpy()
    s = quant[f"{finger}_servo"].to_numpy()

    a_n = (a - np.nanmin(a)) / (np.nanmax(a) - np.nanmin(a) + 1e-6)
    s_n = (s - np.nanmin(s)) / (np.nanmax(s) - np.nanmin(s) + 1e-6)

    return a_n, s_n


# =========================
# METRICS
# =========================
def compute_metrics(a, s, t):
    rest = t < 3.0
    jitter = np.nanstd(a[rest])
    rmse = np.sqrt(np.nanmean((a - s) ** 2))
    hf = hf_energy_ratio(a, FS)

    fast = detect_fast_segments(s, t, FAST_SPEED_THRESH)
    err_fast = np.abs(a[fast] - s[fast])
    err_slow = np.abs(a[~fast] - s[~fast])

    return {
        "jitter": jitter,
        "rmse": rmse,
        "hf_ratio": hf,
        "err_fast_mean": np.nanmean(err_fast),
        "err_slow_mean": np.nanmean(err_slow),
    }


# =========================
# OVERLAY PLOT (FIXED TIME)
# =========================
def plot_overlay_same_time(runs, finger):
    # 1) determine shared duration
    T_common = min(r.raw["t"].iloc[-1] for r in runs)
    t_grid = np.arange(0, T_common, 1 / FS)

    plt.figure()
    for r in runs:
        a, _ = normalize(r.raw, r.quant, finger)
        df = pd.DataFrame({"t": r.raw["t"], "a": a})
        df_r = resample(df, t_grid)
        plt.plot(
            t_grid,
            viz_smooth(df_r["a"].to_numpy()),
            label=r.name
        )

    plt.title(f"{finger} – detected (overlay)")
    plt.xlabel("Time (s)")
    plt.ylabel("Normalized")
    plt.legend()
    plt.tight_layout()
    plt.show()


# =========================
# OTHER PLOTS
# =========================
def plot_detected_vs_commanded(run, finger):
    a, s = normalize(run.raw, run.quant, finger)
    T = min(run.raw["t"].iloc[-1], run.quant["t"].iloc[-1])
    t = np.arange(0, T, 1 / FS)

    a_r = resample(pd.DataFrame({"t": run.raw["t"], "a": a}), t)["a"]
    s_r = resample(pd.DataFrame({"t": run.quant["t"], "s": s}), t)["s"]

    plt.figure()
    plt.plot(t, viz_smooth(a_r), label="detected")
    plt.plot(t, viz_smooth(s_r), label="commanded")
    plt.title(f"{run.name} – {finger}")
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_error_hist(run, finger):
    a, s = normalize(run.raw, run.quant, finger)
    T = min(run.raw["t"].iloc[-1], run.quant["t"].iloc[-1])
    t = np.arange(0, T, 1 / FS)

    a_r = resample(pd.DataFrame({"t": run.raw["t"], "a": a}), t)["a"]
    s_r = resample(pd.DataFrame({"t": run.quant["t"], "s": s}), t)["s"]

    fast = detect_fast_segments(s_r, t, FAST_SPEED_THRESH)
    err = np.abs(a_r - s_r)

    plt.figure()
    plt.hist(err[~fast], bins=40, alpha=0.7, label="slow")
    plt.hist(err[fast], bins=40, alpha=0.7, label="fast")
    plt.title(f"{run.name} – {finger} |error|")
    plt.legend()
    plt.tight_layout()
    plt.show()


# =========================
# PCA PER FINGER
# =========================
def pca_per_finger(df, finger):
    cols = ["jitter", "rmse", "hf_ratio", "err_fast_mean", "err_slow_mean"]
    X = df[cols].to_numpy()
    X = np.nan_to_num(X, nan=np.nanmedian(X))
    Xs = StandardScaler().fit_transform(X)
    Z = PCA(n_components=2).fit_transform(Xs)

    plt.figure()
    plt.scatter(Z[:, 0], Z[:, 1])
    for i, name in enumerate(df["run"]):
        plt.text(Z[i, 0], Z[i, 1], name)
    plt.title(f"PCA – {finger}")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.tight_layout()
    plt.show()


# =========================
# MAIN
# =========================
def main():
    runs = load_runs()
    print(f"Loaded {len(runs)} runs")

    for finger in FINGERS:
        print(f"\n=== {finger} ===")
        rows = []

        # FIXED overlay
        plot_overlay_same_time(runs, finger)

        for r in runs:
            plot_detected_vs_commanded(r, finger)
            plot_error_hist(r, finger)

            a, s = normalize(r.raw, r.quant, finger)
            T = min(r.raw["t"].iloc[-1], r.quant["t"].iloc[-1])
            t = np.arange(0, T, 1 / FS)

            a_r = resample(pd.DataFrame({"t": r.raw["t"], "a": a}), t)["a"]
            s_r = resample(pd.DataFrame({"t": r.quant["t"], "s": s}), t)["s"]

            m = compute_metrics(a_r.to_numpy(), s_r.to_numpy(), t)
            m["run"] = r.name
            rows.append(m)

        df = pd.DataFrame(rows)
        print(df)
        pca_per_finger(df, finger)


if __name__ == "__main__":
    main()