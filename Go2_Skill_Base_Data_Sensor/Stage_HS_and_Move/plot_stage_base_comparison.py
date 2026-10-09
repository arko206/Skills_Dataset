#!/usr/bin/env python3
"""
HandStand+Move stage: ORIGINAL Robot Base trajectories in the floor frame
=========================================================================

Reads ONLY the filtered Robot Base CSV of each stage:

    <STAGE_ROOT>/Stage_N_HS&Move/Tf_handstand_stage_HandStandMove_Trial_N.csv

(leg-frame CSVs are not loaded) and draws four figures:

    1. Stages 1-5   : X-Y-Z
    2. Stages 61-65 : X-Y-Z
    3. Stages 1-5   : X-Y-Yaw
    4. Stages 61-65 : X-Y-Yaw

COORDINATE CONVENTION
---------------------
The recorded floor-frame values are plotted directly:

    X   = x_floor_m
    Y   = y_floor_m
    Z   = z_floor_m
    Yaw = np.unwrap(yaw_floor_rad)        (NO subtraction of the initial yaw)

No translation to a common origin, no rotation into the initial body frame,
no inv(F_T_B0) @ F_T_Bt. Each trajectory starts at its own recorded location,
and the absolute Base height is visible directly on the Z axis.

np.unwrap only removes artificial +/-pi discontinuities: it leaves the FIRST
sample unchanged and adds a multiple of 2*pi to later samples wherever the
stored angle jumped across +/-pi. So Yaw[0] is the actual recorded initial
heading. For a stage that crosses +/-pi, later values continue smoothly past
+/-pi (e.g. -3.38 rad instead of +2.91 rad); these are the same physical
heading, just not folded back into [-pi, pi]. Such stages are flagged in the
console. Angle processing for the regression target delta_theta is handled
separately in the regression script.

The console also prints initial-to-final deltas for INSPECTION ONLY; they are
not used for plotting and are NOT body-frame displacements.

Source CSVs are only read, never written.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers the 3D projection)


# =============================================================================
# CONFIGURATION
# =============================================================================
STAGE_ROOT = Path("/home/unitree-arka/Go2_Skill_Base_Data_Sensor/Stage_HS_and_Move")
OUTPUT_DIR = STAGE_ROOT / "Plots_Base_Comparison"

# Explicit stage numbers (NOT "first five folders" in alphabetical order,
# which would give Stage_1, Stage_10, Stage_11, ...)
FIRST_GROUP = [6, 7, 8, 9, 10]
LAST_GROUP = [11, 12, 13, 14, 15]

REQUIRED_COLS = ["common_time_ns", "x_floor_m", "y_floor_m", "z_floor_m", "yaw_floor_rad"]

# Yaw = np.unwrap(yaw_floor_rad) (True) or the raw stored yaw (False).
# The initial yaw is never subtracted in either case.
UNWRAP_YAW = True

# Same colour for position 1..5 inside each group, so "Stage 1" and "Stage 61"
# share a colour and the two groups read the same way.
STAGE_COLORS = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple"]
# All trajectories solid. To distinguish stages by dash pattern as well, use e.g.
# ["-", "--", "-.", ":", (0, (5, 1, 1, 1))]
STAGE_LINESTYLES = ["-", "-", "-", "-", "-"]

# X-Y-Z: one equal metric scale on all three axes (True), or independent axis
# ranges (False). With equal scaling a small height change can look flat when
# the trajectories are spread out horizontally; the Z tick values still show
# the absolute Base height.
EQUAL_XYZ_SCALE = True

LINE_WIDTH = 1.8
SAMPLE_MARKER_SIZE = 2.5   # small dots at every recorded sample
ENDPOINT_SIZE = 90
ENDPOINT_EDGE_WIDTH = 1.6
VIEW = {"elev": 25, "azim": -60}   # identical camera for every figure
DPI = 300

CAPTION = ("Trajectories are shown using their original recorded floor-frame poses; "
           "no initial-pose translation or body-frame rotation is applied.\n"
           "Yaw is np.unwrap()-ed only to remove artificial \u00b1\u03c0 jumps; "
           "the initial yaw is not subtracted.")


# =============================================================================
# FILE PATHS
# =============================================================================
def stage_dir(n):
    return STAGE_ROOT / f"Stage_{n}_HS&Move"


def stage_base_csv(n):
    return stage_dir(n) / f"Tf_handstand_stage_HandStandMove_Trial_{n}.csv"


# =============================================================================
# CSV LOADING / VALIDATION
# =============================================================================
def load_stage_base(n):
    """
    Load one stage's Base CSV. Returns a time-sorted DataFrame, or None after
    printing a [WARNING] explaining why the stage cannot be used.
    """
    folder, path = stage_dir(n), stage_base_csv(n)
    if not folder.is_dir():
        print(f"[WARNING] Stage {n}: folder not found: {folder}")
        return None
    if not path.is_file():
        print(f"[WARNING] Stage {n}: Base CSV not found: {path}")
        return None

    try:
        df = pd.read_csv(path)
    except Exception as exc:
        print(f"[WARNING] Stage {n}: could not read {path.name}: {exc}")
        return None

    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        print(f"[WARNING] Stage {n}: missing column(s) {missing} in {path.name}")
        return None

    df = df[REQUIRED_COLS].copy()
    for c in REQUIRED_COLS[1:]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    n_bad = int(df[REQUIRED_COLS].isna().any(axis=1).sum())
    if n_bad:
        # Not dropped silently: removing rows would change the trajectory.
        print(f"[WARNING] Stage {n}: {n_bad} row(s) with missing/non-numeric values in "
              f"{REQUIRED_COLS}; stage skipped")
        return None
    if len(df) < 2:
        print(f"[WARNING] Stage {n}: only {len(df)} sample(s); need at least 2")
        return None

    if not df["common_time_ns"].is_monotonic_increasing:
        print(f"[WARNING] Stage {n}: rows not in time order; sorted by common_time_ns")
        df = df.sort_values("common_time_ns", kind="stable")

    return df.reset_index(drop=True)


# =============================================================================
# ORIGINAL FLOOR-FRAME VALUES (no shift, no rotation; yaw only unwrapped)
# =============================================================================
def original_values(df):
    """
    X, Y, Z: the recorded columns, unchanged.
    Yaw: np.unwrap(yaw_floor_rad) -- removes artificial +/-pi jumps only.
         The initial yaw is NOT subtracted, so Yaw[0] equals the recorded
         first yaw. "Yaw_raw" keeps the stored values for the console report.
    """
    yaw_raw = df["yaw_floor_rad"].to_numpy(dtype=float)
    yaw = np.unwrap(yaw_raw) if UNWRAP_YAW else yaw_raw.copy()
    return {
        "X": df["x_floor_m"].to_numpy(dtype=float),
        "Y": df["y_floor_m"].to_numpy(dtype=float),
        "Z": df["z_floor_m"].to_numpy(dtype=float),
        "Yaw": yaw,
        "Yaw_raw": yaw_raw,
    }


def wrap_angle(a):
    """Map an angle into [-pi, pi]. Used ONLY for the printed delta_Yaw."""
    return float(np.arctan2(np.sin(a), np.cos(a)))


def report_stage(n, df, v):
    """Console inspection for one stage (printed values only; plots are unaffected)."""
    t0, t1 = df["common_time_ns"].iloc[0], df["common_time_ns"].iloc[-1]
    print(f"\nStage {n}")
    print(f"  CSV path          : {stage_base_csv(n)}")
    print(f"  Number of samples : {len(df)}   (duration {(t1 - t0) / 1e9:.3f} s)")
    yaw_kind = "unwrapped" if UNWRAP_YAW else "recorded"
    print(f"  Initial X,Y,Z,Yaw : ({v['X'][0]:.4f} m, {v['Y'][0]:.4f} m, "
          f"{v['Z'][0]:.4f} m, {v['Yaw'][0]:.4f} rad)   [yaw: {yaw_kind}]")
    print(f"  Final   X,Y,Z,Yaw : ({v['X'][-1]:.4f} m, {v['Y'][-1]:.4f} m, "
          f"{v['Z'][-1]:.4f} m, {v['Yaw'][-1]:.4f} rad)   [yaw: {yaw_kind}]")
    if not np.isclose(v["Yaw"][-1], v["Yaw_raw"][-1]):
        print(f"  Final recorded yaw: {v['Yaw_raw'][-1]:.4f} rad "
              f"(same heading; unwrapped value differs by "
              f"{(v['Yaw'][-1] - v['Yaw_raw'][-1]) / (2 * np.pi):+.0f} x 2pi)")
    if UNWRAP_YAW and not np.isclose(v["Yaw"][0], v["Yaw_raw"][0]):
        print("  [WARNING] unwrapped initial yaw differs from the recorded initial yaw")
    print(f"  Z range           : {v['Z'].min():.4f} .. {v['Z'].max():.4f} m")

    # Inspection only: floor-frame differences, NOT body-frame displacements.
    dyaw = v["Yaw"][-1] - v["Yaw"][0]
    print(f"  delta_X / delta_Y / delta_Z : {v['X'][-1] - v['X'][0]:+.4f} / "
          f"{v['Y'][-1] - v['Y'][0]:+.4f} / {v['Z'][-1] - v['Z'][0]:+.4f} m (floor frame)")
    print(f"  delta_Yaw         : {yaw_kind} final-initial = {dyaw:+.4f} rad, "
          f"wrapped to [-pi, pi] = {wrap_angle(dyaw):+.4f} rad")

    raw_max_step = float(np.max(np.abs(np.diff(v["Yaw_raw"]))))
    if raw_max_step > np.pi:
        if UNWRAP_YAW:
            print(f"  [NOTE] recorded yaw crosses +/-pi (largest raw step {raw_max_step:.3f} rad); "
                  f"unwrapping removed it, so plotted yaw spans "
                  f"{v['Yaw'].min():.3f} .. {v['Yaw'].max():.3f} rad (may extend beyond +/-pi)")
        else:
            print(f"  [NOTE] recorded yaw crosses +/-pi (largest step {raw_max_step:.3f} rad); the "
                  "X-Y-Yaw plot will show a vertical jump there that is not robot motion")


def load_group(stage_numbers, group_label):
    """Load every stage of a group. Returns {stage: original-value dict}."""
    print("\n" + "=" * 70)
    print(f"{group_label}: stages {stage_numbers}")
    print("=" * 70)
    loaded = {}
    for n in stage_numbers:
        df = load_stage_base(n)
        if df is None:
            continue
        v = original_values(df)
        report_stage(n, df, v)
        loaded[n] = v
    missing = [n for n in stage_numbers if n not in loaded]
    if missing:
        print(f"\n[WARNING] {group_label}: stages not plotted: {missing}")
    return loaded


# =============================================================================
# COMMON AXIS-LIMIT CALCULATION
# =============================================================================
def padded_limits(values, margin=0.08):
    """(min, max) of the values with a small margin; guards a zero range."""
    lo, hi = float(np.min(values)), float(np.max(values))
    span = hi - lo if hi > lo else 1e-3
    return lo - margin * span, hi + margin * span


def xyz_limits(all_loaded):
    """
    Limits shared by BOTH X-Y-Z figures, computed from both groups.
    EQUAL_XYZ_SCALE=True gives one equal metric scale on all three axes.
    """
    pts = np.vstack([np.column_stack([v["X"], v["Y"], v["Z"]]) for v in all_loaded])
    if not EQUAL_XYZ_SCALE:
        return [padded_limits(pts[:, k]) for k in range(3)]
    mins, maxs = pts.min(axis=0), pts.max(axis=0)
    mids = (mins + maxs) / 2
    half = max((maxs - mins).max() / 2 * 1.1, 1e-3)
    return [(m - half, m + half) for m in mids]


def xyyaw_limits(all_loaded):
    """
    Independent limits per axis (metres, metres, radians) -- no forced equal
    scale -- computed from BOTH groups so the two X-Y-Yaw figures match.
    """
    return [padded_limits(np.concatenate([v[k] for v in all_loaded]))
            for k in ("X", "Y", "Yaw")]


# =============================================================================
# SHARED PLOTTING HELPER
# =============================================================================
def set_equal_box(fig, ax):
    """
    Draw the 3D box as a true cube, so equal axis ranges give equal on-screen
    lengths. matplotlib >= 3.3 has set_box_aspect (default box is 4:4:3).
    Older versions (e.g. Ubuntu 20.04's 3.1.2) always use a cube in data space
    but stretch it to fill the axes rectangle, so make that rectangle square.
    """
    if hasattr(ax, "set_box_aspect"):
        ax.set_box_aspect([1, 1, 1])
        return
    fig_w, fig_h = fig.get_size_inches()
    pos = ax.get_position()
    side = min(pos.width * fig_w, pos.height * fig_h)   # inches
    ax.set_position([pos.x0, pos.y0, side / fig_w, side / fig_h])


def plot_group(group_loaded, group_numbers, third_key, limits, equal_box,
               zlabel, title, out_path):
    """
    One figure, one 3D axes, one INDEPENDENT ax.plot() call per stage
    (stages are never concatenated, so no line joins two stages).
    Colour is chosen by the stage's position in its group definition.
    """
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    legend_handles = []

    for pos, n in enumerate(group_numbers):
        if n not in group_loaded:
            continue
        v = group_loaded[n]
        x, y, w = v["X"], v["Y"], v[third_key]
        color, ls = STAGE_COLORS[pos % 5], STAGE_LINESTYLES[pos % 5]

        ax.plot(x, y, w, color=color, linestyle=ls, linewidth=LINE_WIDTH,
                marker="o", markersize=SAMPLE_MARKER_SIZE)
        ax.scatter(x[0], y[0], w[0], color=color, marker="o", s=ENDPOINT_SIZE,
                   edgecolor="black", linewidths=ENDPOINT_EDGE_WIDTH, zorder=10)   # first sample
        ax.scatter(x[-1], y[-1], w[-1], color=color, marker="s", s=ENDPOINT_SIZE,
                   edgecolor="black", linewidths=ENDPOINT_EDGE_WIDTH, zorder=10)  # last sample
        legend_handles.append(Line2D([0], [0], color=color, linestyle=ls,
                                     linewidth=2.2, label=f"Stage {n}"))

    legend_handles += [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="gray",
               markeredgecolor="black", markersize=9, label="Start"),
        Line2D([0], [0], marker="s", color="w", markerfacecolor="gray",
               markeredgecolor="black", markersize=9, label="End"),
    ]

    ax.set_xlim(*limits[0])
    ax.set_ylim(*limits[1])
    ax.set_zlim(*limits[2])
    if equal_box:
        set_equal_box(fig, ax)
    ax.view_init(**VIEW)
    ax.set_xlabel("X position in floor frame (m)")
    ax.set_ylabel("Y position in floor frame (m)")
    ax.set_zlabel(zlabel)
    ax.set_title(title)
    ax.legend(handles=legend_handles, loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize=9)
    fig.text(0.5, 0.01, CAPTION, ha="center", fontsize=9, style="italic", color="#444444")

    # fig.savefig(out_path, dpi=DPI, bbox_inches="tight")
    # print(f"  Saved: {out_path}")
    return fig


# =============================================================================
# X-Y-Z PLOTTING
# =============================================================================
def plot_xyz(group_loaded, group_numbers, limits, label, filename):
    return plot_group(group_loaded, group_numbers, "Z", limits, equal_box=EQUAL_XYZ_SCALE,
                      zlabel="Z position in floor frame (m)",
                      title=f"HandStandMove Stage: Original Base X-Y-Z Trajectories \u2014 {label}",
                      out_path=OUTPUT_DIR / filename)


# =============================================================================
# X-Y-YAW PLOTTING
# =============================================================================
def plot_xyyaw(group_loaded, group_numbers, limits, label, filename):
    return plot_group(group_loaded, group_numbers, "Yaw", limits, equal_box=False,
                      zlabel=("Yaw in floor frame, unwrapped (rad)" if UNWRAP_YAW
                              else "Yaw in floor frame (rad)"),
                      title=f"HandStandMove Stage: Original Base X-Y-Yaw Trajectories \u2014 {label}",
                      out_path=OUTPUT_DIR / filename)


# =============================================================================
# MAIN EXECUTION
# =============================================================================
def main():
    first = load_group(FIRST_GROUP, "FIRST GROUP")
    last = load_group(LAST_GROUP, "LAST GROUP")

    all_loaded = list(first.values()) + list(last.values())
    if not all_loaded:
        raise SystemExit("[ERROR] no stage could be loaded; nothing to plot")

    # Output/saving: only the plot folder is created; source CSVs are untouched
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    lim_xyz = xyz_limits(all_loaded)        # shared by both X-Y-Z figures
    lim_yaw = xyyaw_limits(all_loaded)      # shared by both X-Y-Yaw figures

    first_label = f"Stages {FIRST_GROUP[0]}\u2013{FIRST_GROUP[-1]}"
    last_label = f"Stages {LAST_GROUP[0]}\u2013{LAST_GROUP[-1]}"

    print("\n" + "=" * 70)
    print("SAVING FIGURES")
    print("=" * 70)
    # New file names ("Original"), so the earlier shifted-origin figures are not overwritten
    for loaded, numbers, label, tag in [(first, FIRST_GROUP, first_label, "Stages_01_05"),
                                        (last, LAST_GROUP, last_label, "Stages_61_65")]:
        if not loaded:
            print(f"[WARNING] {label}: no stages loaded; figures for this group skipped")
            continue
        plot_xyz(loaded, numbers, lim_xyz, label, f"HandStandMove_Original_Base_XYZ_{tag}.png")
        plot_xyyaw(loaded, numbers, lim_yaw, label, f"HandStandMove_Original_Base_XY_Yaw_{tag}.png")

    fmt = lambda lim: f"[{lim[0]:.3f}, {lim[1]:.3f}]"
    scale = "equal scale" if EQUAL_XYZ_SCALE else "per axis"
    print(f"\nShared X-Y-Z limits ({scale})  : X {fmt(lim_xyz[0])} m, Y {fmt(lim_xyz[1])} m, "
          f"Z {fmt(lim_xyz[2])} m")
    print(f"Shared X-Y-Yaw limits (per axis) : X {fmt(lim_yaw[0])} m, Y {fmt(lim_yaw[1])} m, "
          f"Yaw {fmt(lim_yaw[2])} rad")

    plt.show()   # all figures open together; rotate freely, re-save from the window if desired


if __name__ == "__main__":
    main()
