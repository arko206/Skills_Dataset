#!/usr/bin/env python3
"""
Stand -> HandStand transition: HandStand-only vs HandStand+Move datasets
=========================================================================

Reads the per-trial transition tables of two INDEPENDENT sets of experiments

    HS  : HandStand-only experiments
    HSM : HandStand+Move experiments

and draws three separate figures (delta_x_body, delta_y_body, delta_theta versus
trial number), each with both datasets overlaid and one dashed mean line per
dataset. Displacements are read directly from the tables (initial-body-frame
SE(2) displacement, already computed); nothing is recomputed, smoothed, filtered
or interpolated.

The shared trial-number axis only shows trial-to-trial variation WITHIN each
dataset. Trial i of one dataset is not related to trial i of the other, so no
paired differences are computed; the comparison is at dataset level
(means, spread, systematic offset).

Input CSVs are only read. Figures are written to OUTPUT_DIR.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# =============================================================================
# CONFIGURATION
# =============================================================================
DATA_ROOT = Path("/home/unitree-arka/Go2_Skill_Base_Data_Sensor")

##--- Loading the "Stand -> HandStand" transition tables for HandStand trajectory --------###
HS_CSV = DATA_ROOT / "HSStage_Stand_HS" / "HS_Stand_to_HandStand_transition_table.csv"

##--- Loading the "Stand -> HandStand" transition tables for HandStand+Move trajectory ---###
HSM_CSV = DATA_ROOT / "Stage_Stand_HS" / "Stand_to_HandStand_transition_table.csv"

OUTPUT_DIR = DATA_ROOT / "Stand_HS_Dataset_Comparison"

REQUIRED_COLS = ["trial_number", "delta_x_body", "delta_y_body", "delta_theta"]

# (column, y-axis label, unit, figure title, output file)
FIGURES = [
    ("delta_x_body", r"$\Delta x_{body}$ [m]", "m",
     "Stand-to-HandStand Transition: Longitudinal Displacement", "Stand_HS_compare_delta_x.png"),
    ("delta_y_body", r"$\Delta y_{body}$ [m]", "m",
     "Stand-to-HandStand Transition: Lateral Displacement", "Stand_HS_compare_delta_y.png"),
    ("delta_theta", r"$\Delta\theta$ [rad]", "rad",
     "Stand-to-HandStand Transition: Yaw Change", "Stand_HS_compare_delta_theta.png"),
]

# Visual conventions shared by all three figures
STYLE = {
    "HS":  {"name": "HandStand only", "mean_name": "HandStand-only mean",
            "color": "tab:blue", "marker": "o"},
    "HSM": {"name": "HandStand+Move", "mean_name": "HandStand+Move mean",
            "color": "tab:orange", "marker": "s"},
}
FIGSIZE = (10, 6)
DPI = 300
SHOW_FIGURES = True            # also open the figures on screen after saving

plt.rcParams.update({"font.size": 12, "axes.titlesize": 14, "axes.labelsize": 13,
                     "legend.fontsize": 11, "xtick.labelsize": 11, "ytick.labelsize": 11})


# =============================================================================
# LOADING / VALIDATION
# =============================================================================
def load_table(path, label):
    """Read one transition table, validate it, and return it sorted by trial_number."""
    if not path.is_file():
        raise SystemExit(f"[ERROR] {label} CSV not found: {path}")
    df = pd.read_csv(path)

    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise SystemExit(f"[ERROR] {label} CSV is missing column(s) {missing}: {path}")

    values = df[REQUIRED_COLS].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    bad = ~np.isfinite(values).all(axis=1)
    if bad.any():
        raise SystemExit(f"[ERROR] {label} CSV has {int(bad.sum())} row(s) with NaN/non-finite "
                         f"values in {REQUIRED_COLS}: {path}")
    if df["trial_number"].duplicated().any():
        dup = sorted(df.loc[df["trial_number"].duplicated(), "trial_number"].unique().tolist())
        raise SystemExit(f"[ERROR] {label} CSV has duplicated trial_number values {dup}: {path}")

    df = df.sort_values("trial_number").reset_index(drop=True)
    print(f"{label:<16s}: {len(df)} valid rows, trials {int(df.trial_number.min())}"
          f"-{int(df.trial_number.max())}   ({path})")
    return df


def with_gaps_broken(trials, values):
    """
    Insert NaN wherever trial numbers are not consecutive, so the connecting line
    stops at a missing trial instead of drawing an interpolating segment across it.
    """
    trials = np.asarray(trials, dtype=float)
    values = np.asarray(values, dtype=float)
    x, y = [trials[0]], [values[0]]
    for t_prev, t, v in zip(trials[:-1], trials[1:], values[1:]):
        if t - t_prev > 1:
            x.append(np.nan)
            y.append(np.nan)
        x.append(t)
        y.append(v)
    return np.array(x), np.array(y)


# =============================================================================
# PLOTTING
# =============================================================================
def plot_comparison(data, column, ylabel, unit, title, out_path, x_range):
    """One separate figure: both datasets trial-by-trial + one dashed mean line each."""
    fig, ax = plt.subplots(figsize=FIGSIZE)

    for key in ("HS", "HSM"):
        df, st = data[key], STYLE[key]
        x, y = with_gaps_broken(df["trial_number"], df[column])
        ax.plot(x, y, color=st["color"], marker=st["marker"], markersize=5, linewidth=1.2,
                linestyle="-", label=f"{st['name']} (N={len(df)})")

    for key in ("HS", "HSM"):
        df, st = data[key], STYLE[key]
        mean = float(df[column].mean())
        ax.axhline(mean, color=st["color"], linestyle="--", linewidth=1.8,
                   label=f"{st['mean_name']} ({mean:+.4f} {unit})")

    ax.set_xlim(x_range[0] - 1, x_range[1] + 1)
    ax.set_xlabel("Trial Number")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.35)
    # Legend below the axes so it can never cover a data point
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=2, framealpha=0.9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight")
    print(f"Saved: {out_path}")
    return fig


# =============================================================================
# TERMINAL SUMMARY
# =============================================================================
def print_summary(data):
    hs, hsm = data["HS"], data["HSM"]
    print("\n" + "=" * 60)
    print("STAND -> HANDSTAND DATASET COMPARISON")
    print("=" * 60)
    print(f"\nHandStand-only trials     : {len(hs)}")
    print(f"HandStand+Move trials     : {len(hsm)}")

    for column, _, unit, _, _ in FIGURES:
        m_hs, m_hsm = float(hs[column].mean()), float(hsm[column].mean())
        s_hs, s_hsm = float(np.std(hs[column], ddof=1)), float(np.std(hsm[column], ddof=1))
        diff = m_hsm - m_hs
        print(f"\n{column}:")
        print(f"    HandStand-only mean   = {m_hs:+.6f} {unit}   (sample std {s_hs:.6f} {unit})")
        print(f"    HandStand+Move mean   = {m_hsm:+.6f} {unit}   (sample std {s_hsm:.6f} {unit})")
        print(f"    difference (HSM - HS) = {diff:+.6f} {unit}")
        print(f"    |mean_HS - mean_HSM|  = {abs(diff):.6f} {unit}")
        if column == "delta_theta":
            print(f"    HandStand-only mean   = {np.degrees(m_hs):+.3f} deg")
            print(f"    HandStand+Move mean   = {np.degrees(m_hsm):+.3f} deg")
            print(f"    difference (HSM - HS) = {np.degrees(diff):+.3f} deg")
            print(f"    |mean_HS - mean_HSM|  = {np.degrees(abs(diff)):.3f} deg")


# =============================================================================
# MAIN
# =============================================================================
def main():
    data = {"HS": load_table(HS_CSV, "HandStand-only"),
            "HSM": load_table(HSM_CSV, "HandStand+Move")}

    x_range = (min(int(d.trial_number.min()) for d in data.values()),
               max(int(d.trial_number.max()) for d in data.values()))

    print_summary(data)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print()
    for column, ylabel, unit, title, filename in FIGURES:
        plot_comparison(data, column, ylabel, unit, title, OUTPUT_DIR / filename, x_range)

    if SHOW_FIGURES:
        plt.show()
    plt.close("all")


if __name__ == "__main__":
    main()
