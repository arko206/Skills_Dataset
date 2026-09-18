"""
HandStand Selected-Trial Publication-Quality Analysis
=======================================================
DEFAULT MODE ("selected"): ONE FIGURE PER INDIVIDUAL TRIAL.

The previous behaviour -- 25 trials x 5 trajectories overlaid on a single
3D axes (125 overlapping curves) -- produced figures that were far too
dense to interpret. That pipeline is still available via `--mode overlay`,
but it is no longer the default.

The new default plots twelve individually selected trials, one figure at a
time, in the visual style of plot_skill_trajectories.py:

    Group 1 (trials 1-25) : first three = 1, 2, 3    last three = 23, 24, 25
    Group 2 (trials 26-50): first three = 26, 27, 28 last three = 48, 49, 50

For each selected trial, one 3D figure is produced per frame category
(Hip, Thigh, Calf, Calflower, Calflower1, Foot) containing

    Robot Base + FL_<frame> + FR_<frame> + RL_<frame> + RR_<frame>

plus a Base-only figure, each saved from two fixed camera viewpoints.
Trials are NEVER overlaid on shared axes in this mode.

IMPORTANT: the cross-trial mean-offset alignment used by the overlay mode
(align_leg_trajectory_to_mean_offset) is deliberately NOT applied here.
Each selected trial shows its own raw measured leg-frame offsets.

COORDINATE-FRAME NORMALIZATION (see accompanying explanation)
---------------------------------------------------------------
For every trial independently:

    F_T_B_0 = floor -> base transform at the trial's first sample
    B0_T_B(t) = inv(F_T_B_0) @ F_T_B(t)      (base, relative to its own start)
    B0_T_L(t) = inv(F_T_B_0) @ F_T_L(t)      (any leg frame, SAME initial-base
                                                frame -- physical offsets from
                                                the base are preserved)

So the base always starts at the origin (0,0,0) with identity orientation,
while every leg frame keeps its true physical offset from the base at t=0.

Because small sensor/calibration variations exist between trials, each leg
frame's initial offset is *averaged across all 50 trials* to get one common
nominal starting position per frame. Individual trials are then shifted
(not reshaped) so they all start from this shared nominal offset, which is
what makes 25 overlaid trials read as a single coherent physical cluster.

No smoothing, resampling, or interpolation is performed anywhere in this
script -- only rigid coordinate transforms, translation, and simple
per-trial time rebasing.
"""

import argparse
import os
import re
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------
# Command-line options are parsed BEFORE pyplot is imported, so that a
# pure batch export can select the non-interactive Agg backend and never
# attempt to open a GUI window.
# ---------------------------------------------------------------------
_parser = argparse.ArgumentParser(
    description="HandStand trajectory figures (per-trial by default).",
    formatter_class=argparse.RawDescriptionHelpFormatter,
)
_parser.add_argument(
    "--mode", choices=["selected", "overlay"], default="selected",
    help="'selected' (default) = one figure per individual trial; "
         "'overlay' = the legacy 25-trials-per-axes figures.",
)
_parser.add_argument(
    "--show", action="store_true",
    help="after saving, also open the figures with plt.show() so they can "
         "be rotated to a preferred angle and saved by hand. Combine with "
         "--trials/--frames -- showing all 84 figures at once is unusable.",
)
_parser.add_argument(
    "--trials", type=int, nargs="+", default=None,
    help="restrict 'selected' mode to these trial numbers.",
)
_parser.add_argument(
    "--frames", type=str, nargs="+", default=None,
    help="restrict 'selected' mode to these categories, e.g. Base Hip Calf.",
)
ARGS = _parser.parse_args()

import matplotlib
if not ARGS.show:
    matplotlib.use("Agg")          # headless batch export
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (3D projection)

# =========================================================================
# CONFIGURATION
# =========================================================================
DATASET_ROOT = os.path.expanduser("~/Go2_Skill_Base_Data_Sensor/HandStand")
PLOTS_ROOT = os.path.join(DATASET_ROOT, "Plots")
N_TRIALS = 50
GROUP_1 = list(range(1, 26))    # Trials 1-25
GROUP_2 = list(range(26, 51))   # Trials 26-50

FRAME_TYPES = ["hip", "thigh", "calf", "calflower", "calflower1", "foot"]
LEGS = ["FL", "FR", "RL", "RR"]
ALL_FRAME_NAMES = [f"{leg}_{ftype}" for ftype in FRAME_TYPES for leg in LEGS]

# Fixed color convention, used identically across every figure
LEG_COLORS = {
    "FL": "tab:orange",
    "FR": "tab:green",
    "RL": "tab:red",
    "RR": "tab:purple",
}
BASE_COLOR = "#222222"  # neutral/dark

TRIAL_LINE_ALPHA = 0.35
TRIAL_LINE_WIDTH = 0.9
MEAN_LINE_WIDTH = 0.0  # (no per-group mean line requested; kept for clarity)

DPI = 300
BBOX = "tight"

# =========================================================================
# CONFIGURATION -- "selected" (per-trial) MODE
# =========================================================================
SELECTED_PLOTS_ROOT = os.path.join(DATASET_ROOT, "Plots_Selected_Trials")

# First three and last three trials of each 25-trial group.
SELECTED_GROUP_1 = [1, 2, 3, 23, 24, 25]      # group 1 spans trials 1-25
SELECTED_GROUP_2 = [26, 27, 28, 48, 49, 50]   # group 2 spans trials 26-50
SELECTED_TRIALS = SELECTED_GROUP_1 + SELECTED_GROUP_2

# Display-name -> CSV frame_name suffix. Drives both the figure content
# and the output sub-directory names.
FRAME_CATEGORIES = {
    "Hip": "hip",
    "Thigh": "thigh",
    "Calf": "calf",
    "Calflower": "calflower",
    "Calflower1": "calflower1",
    "Foot": "foot",
}

# Fixed colours, identical in every per-trial figure. The base is blue here
# (not the neutral BASE_COLOR used by the dense overlay figures), because a
# single trial's base curve must read as one of five distinct trajectories.
SELECTED_BASE_COLOR = "tab:blue"
SELECTED_COLORS = {
    "Base": SELECTED_BASE_COLOR,
    "FL": LEG_COLORS["FL"],   # orange
    "FR": LEG_COLORS["FR"],   # green
    "RL": LEG_COLORS["RL"],   # red
    "RR": LEG_COLORS["RR"],   # purple
}

# Trajectory-line style, adapted from plot_skill_trajectories.py.
# Fully opaque: no transparent trajectory clouds.
SEL_LINE_WIDTH = 2.0     # requested 1.8 - 2.2
SEL_MARKER_SIZE = 3.0    # requested 2.5 - 3.5
SEL_LINE_ALPHA = 1.0

# Start / End marker style.
SEL_ENDPOINT_SIZE = 140          # requested 120 - 150
SEL_ENDPOINT_EDGE_WIDTH = 2.2    # requested 2.0 - 2.5
SEL_ENDPOINT_ZORDER = 10

# The two camera viewpoints, applied identically to every trial and every
# frame category so figures stay directly comparable.
SEL_VIEWS = {
    "View1": dict(elev=25, azim=-60),   # standard oblique perspective
    "View2": dict(elev=25, azim=30),    # rotated to reveal hidden motion
}

SEL_CAPTION = "Coordinates expressed relative to the initial base pose"

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "grid.alpha": 0.25,
})


# =========================================================================
# LOW-LEVEL LOADERS
# =========================================================================
def load_base_trial(trial_num):
    """
    Load the base 6-DoF trajectory CSV for one trial.

    Expects (verified against the provided reference CSV header):
        common_time_ns, T00..T33

    Returns the raw DataFrame sorted by common_time_ns, or None (with a
    printed warning) if the file is missing or malformed.
    """
    path = os.path.join(DATASET_ROOT, f"Trial_{trial_num}",
                         f"Tf_stationary_HandStand_Trial_{trial_num}.csv")
    if not os.path.isfile(path):
        print(f"  [MISSING] Base CSV not found for Trial {trial_num}: {path}")
        return None

    df = pd.read_csv(path)

    required_cols = ["common_time_ns"] + [f"T{i}{j}" for i in range(4) for j in range(4)]
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        print(f"  [MALFORMED] Trial {trial_num} base CSV missing columns: "
              f"{missing_cols}")
        return None

    return df.sort_values("common_time_ns").reset_index(drop=True)


def load_leg_trial(trial_num):
    """
    Load the synchronized leg-frame CSV for one trial.

    Expects (verified against the provided reference CSV header):
        base_common_time_ns, frame_name, F_T00..F_T33

    Returns the raw DataFrame, or None (with a printed warning) if the file
    is missing or malformed.
    """
    path = os.path.join(DATASET_ROOT, f"Trial_{trial_num}",
                         f"Tf_leg_stationary_HandStand_Trial_{trial_num}.csv")
    if not os.path.isfile(path):
        print(f"  [MISSING] Leg CSV not found for Trial {trial_num}: {path}")
        return None

    df = pd.read_csv(path)

    required_cols = ["base_common_time_ns", "frame_name"] + \
        [f"F_T{i}{j}" for i in range(4) for j in range(4)]
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        print(f"  [MALFORMED] Trial {trial_num} leg CSV missing columns: "
              f"{missing_cols}")
        return None

    missing_frames = [f for f in ALL_FRAME_NAMES if f not in df["frame_name"].unique()]
    if missing_frames:
        print(f"  [WARNING] Trial {trial_num} leg CSV is missing frames: "
              f"{missing_frames}")

    return df


def matrix_from_csv_row(row, prefix):
    """
    Build a 4x4 homogeneous transform matrix (row-major) from a CSV row,
    given a column prefix ('T' for base, 'F_T' for leg frames).

    Expects columns <prefix>00 ... <prefix>33 encoding:
        [T00 T01 T02 T03]
        [T10 T11 T12 T13]
        [T20 T21 T22 T23]
        [T30 T31 T32 T33]
    """
    M = np.eye(4)
    for i in range(4):
        for j in range(4):
            M[i, j] = row[f"{prefix}{i}{j}"]
    return M


# =========================================================================
# COORDINATE TRANSFORM
# =========================================================================
def rotation_to_euler(R):
    """
    Extract (roll, pitch, yaw) from a 3x3 rotation matrix using the
    standard XYZ intrinsic (roll-pitch-yaw) convention:
        R = Rz(yaw) @ Ry(pitch) @ Rx(roll)
    """
    pitch = np.arcsin(np.clip(-R[2, 0], -1.0, 1.0))
    roll = np.arctan2(R[2, 1], R[2, 2])
    yaw = np.arctan2(R[1, 0], R[0, 0])
    return roll, pitch, yaw


def transform_trial_to_initial_base_frame(base_df, leg_df):
    """
    Given one trial's raw base_df and leg_df, express EVERYTHING in the
    frame of the base's own first sample:

        F_T_B_0   = transform at base_df row 0
        B0_T_B(t) = inv(F_T_B_0) @ F_T_B(t)     for every base sample
        B0_T_L(t) = inv(F_T_B_0) @ F_T_L(t)     for every leg-frame sample

    Returns:
        base_result : DataFrame with columns
            t_rel, x, y, z, roll, pitch, yaw
        leg_result : dict[frame_name] -> DataFrame with columns
            t_rel, x, y, z
    """
    # --- Base ---
    F_T_B0 = matrix_from_csv_row(base_df.iloc[0], "T")
    B0_T_F = np.linalg.inv(F_T_B0)

    t_ns = base_df["common_time_ns"].to_numpy()
    t_rel = (t_ns - t_ns[0]) / 1e9

    xs, ys, zs, rolls, pitches, yaws = [], [], [], [], [], []
    for _, row in base_df.iterrows():
        F_T_Bt = matrix_from_csv_row(row, "T")
        B0_T_Bt = B0_T_F @ F_T_Bt
        xs.append(B0_T_Bt[0, 3])
        ys.append(B0_T_Bt[1, 3])
        zs.append(B0_T_Bt[2, 3])
        r, p, y = rotation_to_euler(B0_T_Bt[:3, :3])
        rolls.append(r)
        pitches.append(p)
        yaws.append(y)

    # Unwrap to avoid artificial +pi/-pi jumps
    rolls = np.unwrap(np.array(rolls))
    pitches = np.unwrap(np.array(pitches))
    yaws = np.unwrap(np.array(yaws))

    base_result = pd.DataFrame({
        "t_rel": t_rel,
        "x": xs, "y": ys, "z": zs,
        "roll": rolls, "pitch": pitches, "yaw": yaws,
    })

    # --- Leg frames (all present frame_names), using the SAME B0_T_F ---
    leg_result = {}
    for frame_name in leg_df["frame_name"].unique():
        frame_df = leg_df[leg_df["frame_name"] == frame_name].sort_values(
            "base_common_time_ns"
        ).reset_index(drop=True)

        t_ns_leg = frame_df["base_common_time_ns"].to_numpy()
        t_rel_leg = (t_ns_leg - t_ns_leg[0]) / 1e9

        xs_l, ys_l, zs_l = [], [], []
        for _, row in frame_df.iterrows():
            F_T_Lt = matrix_from_csv_row(row, "F_T")
            B0_T_Lt = B0_T_F @ F_T_Lt
            xs_l.append(B0_T_Lt[0, 3])
            ys_l.append(B0_T_Lt[1, 3])
            zs_l.append(B0_T_Lt[2, 3])

        leg_result[frame_name] = pd.DataFrame({
            "t_rel": t_rel_leg, "x": xs_l, "y": ys_l, "z": zs_l,
        })

    return base_result, leg_result


def normalize_time(timestamps_ns):
    """t_rel = (t - t[0]) / 1e9, as a standalone reusable utility."""
    t_ns = np.asarray(timestamps_ns)
    return (t_ns - t_ns[0]) / 1e9


# =========================================================================
# LOADING A FULL GROUP OF TRIALS
# =========================================================================
def load_trial_group(trial_numbers):
    """
    Load + transform every trial in `trial_numbers` into the initial-base
    frame. Reports missing/malformed trials explicitly rather than
    silently skipping them.

    Returns:
        dict[trial_num] -> (base_result_df, leg_result_dict)
        (only successfully-loaded trials are included)
    """
    results = {}
    print(f"Loading trials: {trial_numbers[0]}-{trial_numbers[-1]}")
    for trial_num in trial_numbers:
        base_df = load_base_trial(trial_num)
        leg_df = load_leg_trial(trial_num)
        if base_df is None or leg_df is None:
            print(f"  -> Skipping Trial {trial_num} (see warning above)")
            continue

        base_result, leg_result = transform_trial_to_initial_base_frame(
            base_df, leg_df
        )
        results[trial_num] = (base_result, leg_result)

    n_ok = len(results)
    n_expected = len(trial_numbers)
    print(f"  Loaded {n_ok}/{n_expected} trials successfully.\n")
    return results


# =========================================================================
# MEAN INITIAL LEG-FRAME OFFSETS (computed ONCE, across ALL 50 trials)
# =========================================================================
def compute_mean_initial_frame_offsets(all_trials_dict):
    """
    all_trials_dict: dict[trial_num] -> (base_result_df, leg_result_dict)
    covering ALL loaded trials (both groups combined).

    For each of the 24 frame names, average that frame's FIRST sample
    (x, y, z) in the initial-base frame across every trial where it is
    present. This is a single global average -- it must NOT be computed
    separately per group, so that Group 1 and Group 2 figures share
    identical nominal offsets.

    Returns:
        dict[frame_name] -> np.array([mean_x, mean_y, mean_z])
    """
    per_frame_initials = {name: [] for name in ALL_FRAME_NAMES}

    for trial_num, (base_result, leg_result) in all_trials_dict.items():
        for frame_name in ALL_FRAME_NAMES:
            if frame_name not in leg_result:
                continue
            first_row = leg_result[frame_name].iloc[0]
            per_frame_initials[frame_name].append(
                np.array([first_row["x"], first_row["y"], first_row["z"]])
            )

    mean_offsets = {}
    for frame_name, samples in per_frame_initials.items():
        if len(samples) == 0:
            print(f"  [WARNING] No trials contained frame '{frame_name}' -- "
                  "cannot compute its mean initial offset.")
            continue
        mean_offsets[frame_name] = np.mean(np.vstack(samples), axis=0)

    return mean_offsets


def align_leg_trajectory_to_mean_offset(frame_traj_df, mean_offset):
    """
    frame_traj_df : DataFrame with columns t_rel, x, y, z (one trial, one
                     frame, already in the initial-base frame)
    mean_offset   : np.array([mean_x, mean_y, mean_z]) for this frame,
                     shared across all trials

    Returns a NEW DataFrame with x, y, z shifted so the trial's own first
    sample coincides with mean_offset, while the trial's recorded motion
    shape (x(t)-x(0), etc.) is fully preserved:

        p_plot(t) = mean_offset + (p(t) - p(0))
    """
    p0 = frame_traj_df[["x", "y", "z"]].iloc[0].to_numpy()
    shifted = frame_traj_df.copy()
    shifted[["x", "y", "z"]] = (
        frame_traj_df[["x", "y", "z"]].to_numpy() - p0 + mean_offset
    )
    return shifted


# =========================================================================
# AXIS-SCALING HELPERS
# =========================================================================
def calculate_common_axis_limits(point_clouds, margin_factor=1.1):
    """
    point_clouds: list of (N,3) arrays (already gathered from BOTH groups)
    Returns (xlim, ylim, zlim, mids, max_range) for identical, equal-scaled
    3D axes across Group 1 and Group 2 figures.
    """
    all_points = np.vstack(point_clouds)
    mins = all_points.min(axis=0)
    maxs = all_points.max(axis=0)
    mids = (mins + maxs) / 2
    max_range = (maxs - mins).max() / 2 * margin_factor
    xlim = (mids[0] - max_range, mids[0] + max_range)
    ylim = (mids[1] - max_range, mids[1] + max_range)
    zlim = (mids[2] - max_range, mids[2] + max_range)
    return xlim, ylim, zlim


def set_equal_3d_axes(ax, xlim, ylim, zlim, elev=20, azim=-60):
    """Apply identical limits, box aspect, and viewing angle to a 3D axis."""
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_zlim(*zlim)
    try:
        ax.set_box_aspect([1, 1, 1])
    except AttributeError:
        pass
    ax.view_init(elev=elev, azim=azim)


def calculate_per_axis_limits(value_arrays, margin_factor=1.1):
    """
    value_arrays: list of 1D arrays, ALL drawn from the same axis/column
                  (e.g. every trial's 'yaw' column from BOTH groups).

    Unlike calculate_common_axis_limits(), this does NOT force equal range
    across x/y/z -- appropriate for pose-space plots that mix meters and
    radians on different axes, where forcing a cube aspect would be
    meaningless. Still guarantees Group 1 and Group 2 share identical
    limits on each axis, as required.
    """
    all_vals = np.concatenate(value_arrays)
    vmin, vmax = all_vals.min(), all_vals.max()
    mid = (vmin + vmax) / 2
    half_range = (vmax - vmin) / 2 * margin_factor
    if half_range == 0:  # guard against a perfectly constant column
        half_range = 1e-3
    return mid - half_range, mid + half_range


# =========================================================================
# FIGURE FAMILY A: Base XYZ only
# =========================================================================
def plot_base_xyz(group_trials, xlim, ylim, zlim, title, out_path):
    """
    group_trials: dict[trial_num] -> (base_result_df, leg_result_dict)
    Plots all base trajectories (thin, semi-transparent) in one 3D figure.
    """
    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection="3d")

    for trial_num, (base_result, _) in group_trials.items():
        ax.plot(base_result["x"], base_result["y"], base_result["z"],
                color=BASE_COLOR, alpha=TRIAL_LINE_ALPHA,
                linewidth=TRIAL_LINE_WIDTH)

    # Mark the common starting point once (all trials begin at ~(0,0,0))
    ax.scatter([0], [0], [0], color="red", marker="o", s=60,
               edgecolor="white", linewidths=1.0, zorder=5, label="Start")

    set_equal_3d_axes(ax, xlim, ylim, zlim)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.set_title(f"{title}\n(Trajectories expressed relative to the initial "
                 f"base pose)")
    ax.grid(True)
    ax.legend(loc="upper left")

    plt.tight_layout()
    plt.savefig(out_path, dpi=DPI, bbox_inches=BBOX)
    plt.close(fig)
    print(f"  Saved: {out_path}")


# =========================================================================
# FIGURE FAMILY B: Base + four leg frames of one type (hip/thigh/.../foot)
# =========================================================================
def plot_base_and_leg_frame(group_trials, frame_type, mean_offsets,
                             xlim, ylim, zlim, title, out_path):
    """
    group_trials : dict[trial_num] -> (base_result_df, leg_result_dict)
    frame_type   : one of 'hip','thigh','calf','calflower','calflower1','foot'
    mean_offsets : dict[frame_name] -> mean (x,y,z), from
                   compute_mean_initial_frame_offsets() over ALL 50 trials

    Plots Base + FL_<type>/FR_<type>/RL_<type>/RR_<type> for every trial in
    the group, each leg's 25 lines sharing one fixed color.
    """
    fig = plt.figure(figsize=(9, 8))
    ax = fig.add_subplot(111, projection="3d")

    # Base (neutral/dark)
    for trial_num, (base_result, _) in group_trials.items():
        ax.plot(base_result["x"], base_result["y"], base_result["z"],
                color=BASE_COLOR, alpha=TRIAL_LINE_ALPHA,
                linewidth=TRIAL_LINE_WIDTH)

    # Four leg-frame clusters
    for leg in LEGS:
        frame_name = f"{leg}_{frame_type}"
        if frame_name not in mean_offsets:
            continue  # already warned about during offset computation
        mean_offset = mean_offsets[frame_name]
        color = LEG_COLORS[leg]

        for trial_num, (_, leg_result) in group_trials.items():
            if frame_name not in leg_result:
                continue
            aligned = align_leg_trajectory_to_mean_offset(
                leg_result[frame_name], mean_offset
            )
            ax.plot(aligned["x"], aligned["y"], aligned["z"],
                    color=color, alpha=TRIAL_LINE_ALPHA,
                    linewidth=TRIAL_LINE_WIDTH)

    # Mark base common start
    ax.scatter([0], [0], [0], color="red", marker="o", s=50,
               edgecolor="white", linewidths=1.0, zorder=5)

    set_equal_3d_axes(ax, xlim, ylim, zlim)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.set_title(f"{title}\n(Trajectories expressed relative to the initial "
                 f"base pose)")
    ax.grid(True)

    # Clean legend: exactly Base + FL/FR/RL/RR, no 25-entry legend
    from matplotlib.lines import Line2D
    legend_handles = [
        Line2D([0], [0], color=BASE_COLOR, lw=2, label="Base"),
        Line2D([0], [0], color=LEG_COLORS["FL"], lw=2, label="FL " + frame_type.capitalize()),
        Line2D([0], [0], color=LEG_COLORS["FR"], lw=2, label="FR " + frame_type.capitalize()),
        Line2D([0], [0], color=LEG_COLORS["RL"], lw=2, label="RL " + frame_type.capitalize()),
        Line2D([0], [0], color=LEG_COLORS["RR"], lw=2, label="RR " + frame_type.capitalize()),
    ]
    ax.legend(handles=legend_handles, loc="upper left", fontsize=9)

    plt.tight_layout()
    plt.savefig(out_path, dpi=DPI, bbox_inches=BBOX)
    plt.close(fig)
    print(f"  Saved: {out_path}")


# =========================================================================
# FIGURE FAMILY: Base pose-space plots (X,Y,Yaw / X,Z,Pitch / Y,Z,Roll)
# =========================================================================
def plot_base_pose_space(group_trials, cols, labels, xlim, ylim, zlim,
                          title, out_path):
    """
    cols   : tuple of 3 column names from base_result_df, e.g. ('x','y','yaw')
    labels : tuple of 3 axis label strings matching cols
    xlim/ylim/zlim : shared axis limits (identical across both groups),
                      from calculate_per_axis_limits()
    """
    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection="3d")

    for trial_num, (base_result, _) in group_trials.items():
        ax.plot(base_result[cols[0]], base_result[cols[1]], base_result[cols[2]],
                color=BASE_COLOR, alpha=TRIAL_LINE_ALPHA,
                linewidth=TRIAL_LINE_WIDTH)

    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_zlim(*zlim)
    ax.set_xlabel(labels[0])
    ax.set_ylabel(labels[1])
    ax.set_zlabel(labels[2])
    ax.set_title(f"{title}\n(Poses expressed relative to the initial base "
                 f"pose)")
    ax.grid(True)
    # NOTE: no forced cube aspect here -- these axes mix meters and
    # radians, so an equal box aspect would be physically meaningless.
    # Shared xlim/ylim/zlim across groups is what makes Group 1 vs
    # Group 2 directly comparable instead.
    ax.view_init(elev=20, azim=-60)

    plt.tight_layout()
    plt.savefig(out_path, dpi=DPI, bbox_inches=BBOX)
    plt.close(fig)
    print(f"  Saved: {out_path}")


# =========================================================================
# SELECTED-TRIAL MODE: SYNCHRONIZATION VERIFICATION
# (same check as plot_skill_trajectories.py -- raises, never interpolates)
# =========================================================================
def verify_trial_synchronization(base_df, leg_df, trial_num):
    """
    Verify that the base and leg CSVs of one trial are already sample-for-
    sample synchronized. Raises ValueError with a clear message otherwise;
    this script must never interpolate, resample or silently proceed.

    Returns the sorted array of unique base timestamps (ns).
    """
    base_timestamps = np.sort(base_df["common_time_ns"].unique())
    leg_timestamps = np.sort(leg_df["base_common_time_ns"].unique())

    if not np.array_equal(base_timestamps, leg_timestamps):
        raise ValueError(
            f"Synchronization check FAILED for Trial {trial_num}: unique "
            f"'base_common_time_ns' values in the leg CSV do not exactly "
            f"match 'common_time_ns' values in the base CSV. Refusing to "
            f"interpolate or silently proceed. Base has "
            f"{len(base_timestamps)} unique timestamps, leg has "
            f"{len(leg_timestamps)} unique timestamps."
        )

    # Every individual frame must have exactly one sample per base timestamp
    # (i.e. no missing or duplicated samples for any single frame).
    for frame_name in ALL_FRAME_NAMES:
        subset = leg_df[leg_df["frame_name"] == frame_name]
        if len(subset) == 0:
            raise ValueError(
                f"Synchronization check FAILED for Trial {trial_num}: frame "
                f"'{frame_name}' is completely absent from the leg CSV."
            )
        frame_ts_sorted = np.sort(subset["base_common_time_ns"].values)
        if not np.array_equal(frame_ts_sorted, base_timestamps):
            raise ValueError(
                f"Synchronization check FAILED for Trial {trial_num}, frame "
                f"'{frame_name}': its 'base_common_time_ns' values do not "
                f"exactly match the base CSV timestamps "
                f"({len(frame_ts_sorted)} vs {len(base_timestamps)} samples)."
            )

    return base_timestamps


def load_selected_trial(trial_num):
    """
    Load one selected trial, verify synchronization (hard error on failure),
    and express base + every leg frame in that trial's INITIAL BASE FRAME
    via the existing transform_trial_to_initial_base_frame():

        B0_T_B(t) = inv(F_T_B_0) @ F_T_B(t)
        B0_T_L(t) = inv(F_T_B_0) @ F_T_L(t)

    One single inv(F_T_B_0) is applied to the base and to all leg frames, so
    the base starts at (0, 0, 0) while the leg frames keep their real
    measured offsets around it. No per-frame origin subtraction, and NO
    mean-offset alignment.

    Returns (base_result_df, leg_result_dict, base_timestamps_ns).
    """
    base_df = load_base_trial(trial_num)
    leg_df = load_leg_trial(trial_num)
    if base_df is None or leg_df is None:
        raise FileNotFoundError(
            f"Trial {trial_num} could not be loaded (see warning above). "
            f"A selected trial is mandatory -- refusing to skip it silently."
        )

    base_timestamps = verify_trial_synchronization(base_df, leg_df, trial_num)
    base_result, leg_result = transform_trial_to_initial_base_frame(
        base_df, leg_df
    )
    return base_result, leg_result, base_timestamps


# =========================================================================
# SELECTED-TRIAL MODE: SINGLE-TRIAL FIGURE
# (trajectory style, start/end markers, equal-axis maths and legend
#  construction all adapted from plot_skill_trajectories.py)
# =========================================================================
def plot_single_trial_figure(trajectories, title):
    """
    trajectories : ordered dict {legend_label: (color, (N,3) xyz array)}
    title        : concise one-line figure title

    Returns (fig, ax). The caller applies the viewpoints and saves.
    """
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    all_points = []
    for label, (color, xyz) in trajectories.items():
        x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]

        # Continuous line + small markers, so temporal motion is visible.
        ax.plot(x, y, z, color=color, linewidth=SEL_LINE_WIDTH,
                marker="o", markersize=SEL_MARKER_SIZE,
                alpha=SEL_LINE_ALPHA, label=label)

        # START = large filled circle, thick black edge, high z-order.
        ax.scatter(x[0], y[0], z[0], color=color, edgecolor="black",
                   marker="o", s=SEL_ENDPOINT_SIZE,
                   linewidths=SEL_ENDPOINT_EDGE_WIDTH,
                   zorder=SEL_ENDPOINT_ZORDER)
        # END = large filled square, thick black edge, high z-order.
        ax.scatter(x[-1], y[-1], z[-1], color=color, edgecolor="black",
                   marker="s", s=SEL_ENDPOINT_SIZE,
                   linewidths=SEL_ENDPOINT_EDGE_WIDTH,
                   zorder=SEL_ENDPOINT_ZORDER)

        all_points.append(xyz)

    # ---- Equal X/Y/Z scaling over this trial's Base + four leg frames ----
    all_points = np.vstack(all_points)
    x_min, y_min, z_min = all_points.min(axis=0)
    x_max, y_max, z_max = all_points.max(axis=0)

    x_mid = (x_min + x_max) / 2
    y_mid = (y_min + y_max) / 2
    z_mid = (z_min + z_max) / 2

    # One shared half-range applied to all three axes -> no distortion.
    max_range = max(x_max - x_min, y_max - y_min, z_max - z_min) / 2
    if max_range == 0:
        max_range = 1e-3          # fully stationary safeguard
    max_range *= 1.1              # margin so the big markers are not clipped

    ax.set_xlim(x_mid - max_range, x_mid + max_range)
    ax.set_ylim(y_mid - max_range, y_mid + max_range)
    ax.set_zlim(z_mid - max_range, z_mid + max_range)

    try:
        ax.set_box_aspect([1, 1, 1])  # matplotlib >= 3.3
    except (AttributeError, TypeError):
        # matplotlib < 3.3 has no Axes3D.set_box_aspect. It is not needed
        # for correctness there: older Axes3D normalises each axis range
        # into the same cube, so equal limits already give equal visual
        # scaling. The call is kept for newer matplotlib.
        pass

    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.set_title(title)
    ax.grid(True)

    # ---- Legend outside the axes: series + exactly two marker entries ----
    marker_legend = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="gray",
               markeredgecolor="black", markersize=9, label="Start"),
        Line2D([0], [0], marker="s", color="w", markerfacecolor="gray",
               markeredgecolor="black", markersize=9, label="End"),
    ]
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles=handles + marker_legend,
              labels=labels + ["Start", "End"],
              loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize=9)

    fig.text(0.01, 0.01, SEL_CAPTION, fontsize=8, color="dimgray")

    return fig, ax


def save_two_views(fig, ax, out_dir, stem):
    """Save the same figure from View1 and View2, then restore View1."""
    saved = []
    for view_name, angles in SEL_VIEWS.items():
        ax.view_init(**angles)
        out_path = os.path.join(out_dir, f"{stem}_{view_name}.png")
        fig.savefig(out_path, dpi=DPI, bbox_inches=BBOX)
        saved.append(out_path)
    # Leave the figure on View1 so an interactive window opens on the
    # standard oblique perspective.
    ax.view_init(**SEL_VIEWS["View1"])
    return saved


def xyz_array(df):
    """(N,3) float array of the x/y/z columns of a transformed trajectory."""
    return df[["x", "y", "z"]].to_numpy(dtype=float)


# =========================================================================
# MAIN PIPELINE -- "selected" (per-trial) MODE  [DEFAULT]
# =========================================================================
def main_selected():
    trials = ARGS.trials if ARGS.trials else SELECTED_TRIALS
    unknown = [t for t in trials if t not in SELECTED_TRIALS]
    if unknown:
        print(f"[NOTE] Trials {unknown} are outside the selected set; "
              f"plotting them anyway.\n")

    categories = ["Base"] + list(FRAME_CATEGORIES)
    if ARGS.frames:
        requested = {f.lower() for f in ARGS.frames}
        categories = [c for c in categories if c.lower() in requested]
        if not categories:
            raise SystemExit(
                f"No known frame categories in {ARGS.frames}. Choose from: "
                f"Base {' '.join(FRAME_CATEGORIES)}"
            )

    for category in categories:
        os.makedirs(os.path.join(SELECTED_PLOTS_ROOT, category), exist_ok=True)

    print("=" * 70)
    print("HANDSTAND SELECTED-TRIAL FIGURES (one figure per trial)")
    print("=" * 70)
    print(f"Group 1 (trials 1-25)  first/last three : {SELECTED_GROUP_1}")
    print(f"Group 2 (trials 26-50) first/last three : {SELECTED_GROUP_2}")
    print(f"Trials to plot                          : {trials}")
    print(f"Frame categories                        : {categories}")
    for name, angles in SEL_VIEWS.items():
        print(f"{name:<40}: elev={angles['elev']}, azim={angles['azim']}")
    print(f"Output root                             : {SELECTED_PLOTS_ROOT}")
    print("=" * 70)

    total_files = 0
    open_figs = []

    for trial_num in trials:
        base_result, leg_result, base_timestamps = load_selected_trial(trial_num)

        base_xyz = xyz_array(base_result)
        duration_s = (base_timestamps[-1] - base_timestamps[0]) / 1e9

        print(f"\nTrial {trial_num:2d}: {len(base_timestamps)} samples, "
              f"{duration_s:.3f} s, base/leg timestamps match exactly: True")
        print(f"          normalized base start = "
              f"({base_xyz[0, 0]:+.6f}, {base_xyz[0, 1]:+.6f}, "
              f"{base_xyz[0, 2]:+.6f})")

        for category in categories:
            if category == "Base":
                trajectories = {
                    "Robot Base": (SELECTED_COLORS["Base"], base_xyz),
                }
                title = f"HandStand Trial {trial_num}: Base Trajectory"
            else:
                suffix = FRAME_CATEGORIES[category]
                trajectories = {
                    "Robot Base": (SELECTED_COLORS["Base"], base_xyz),
                }
                for leg in LEGS:
                    frame_name = f"{leg}_{suffix}"
                    trajectories[f"{leg} {category}"] = (
                        SELECTED_COLORS[leg], xyz_array(leg_result[frame_name])
                    )
                title = (f"HandStand Trial {trial_num}: "
                         f"Base and {category}-Frame Trajectories")

            fig, ax = plot_single_trial_figure(trajectories, title)
            stem = f"HandStand_Trial_{trial_num:02d}_{category}"
            saved = save_two_views(
                fig, ax, os.path.join(SELECTED_PLOTS_ROOT, category), stem
            )
            total_files += len(saved)
            print(f"          Saved: {os.path.basename(saved[0])} | "
                  f"{os.path.basename(saved[1])}")

            if ARGS.show:
                open_figs.append(fig)
            else:
                plt.close(fig)

    print("\n" + "=" * 70)
    print(f"ALL FIGURES GENERATED SUCCESSFULLY ({total_files} PNG files)")
    print(f"Output directory: {SELECTED_PLOTS_ROOT}")
    print("=" * 70)

    if ARGS.show:
        if len(open_figs) > 12:
            print(f"\n[WARNING] {len(open_figs)} interactive windows are "
                  f"about to open at once. Narrow the selection with "
                  f"--trials / --frames, e.g.\n"
                  f"    python3 handstand_50trial_analysis.py "
                  f"--show --trials 1 --frames Hip")
        print("\nInteractive windows open on View1. Rotate to any angle and "
              "use the toolbar save button to export your own version.")
        plt.show()


# =========================================================================
# MAIN PIPELINE -- "overlay" MODE  [LEGACY, opt-in via --mode overlay]
# =========================================================================
def main_overlay():
    # --- 0. Create output directory structure ---
    subdirs = ["Base_XYZ", "Base_Hip", "Base_Thigh", "Base_Calf",
               "Base_Calflower", "Base_Calflower1", "Base_Foot", "Base_Pose"]
    for sd in subdirs:
        os.makedirs(os.path.join(PLOTS_ROOT, sd), exist_ok=True)

    # --- 1. Load and transform both groups ---
    print("=" * 70)
    print("LOADING AND TRANSFORMING TRIALS")
    print("=" * 70)
    group1 = load_trial_group(GROUP_1)
    group2 = load_trial_group(GROUP_2)

    missing_trials = (len(GROUP_1) - len(group1)) + (len(GROUP_2) - len(group2))
    if missing_trials > 0:
        print(f"[SUMMARY] {missing_trials} of {N_TRIALS} expected trials "
              f"could not be loaded (see warnings above).\n")
    else:
        print(f"[SUMMARY] All {N_TRIALS} expected trials loaded successfully.\n")

    all_trials = {**group1, **group2}

    # --- 2. Compute mean initial leg-frame offsets ONCE, across all trials ---
    print("=" * 70)
    print("COMPUTING MEAN INITIAL LEG-FRAME OFFSETS (all loaded trials)")
    print("=" * 70)
    mean_offsets = compute_mean_initial_frame_offsets(all_trials)

    print(f"\n{'frame_name':<15}{'mean_X_m':>12}{'mean_Y_m':>12}{'mean_Z_m':>12}")
    for frame_name in ALL_FRAME_NAMES:
        if frame_name in mean_offsets:
            mx, my, mz = mean_offsets[frame_name]
            print(f"{frame_name:<15}{mx:>12.4f}{my:>12.4f}{mz:>12.4f}")
        else:
            print(f"{frame_name:<15}{'N/A':>12}{'N/A':>12}{'N/A':>12}")
    print()

    # --- 3. Common axis limits (identical for both groups) ---
    # Base XYZ limits (Figure family A)
    base_points = [
        base_result[["x", "y", "z"]].to_numpy()
        for base_result, _ in all_trials.values()
    ]
    base_xlim, base_ylim, base_zlim = calculate_common_axis_limits(base_points)

    # Base + all-leg-frame limits (Figure family B, shared across ALL frame
    # types so that, e.g., Base_Hip and Base_Foot are still each internally
    # consistent between their own two groups)
    leg_family_limits = {}
    for frame_type in FRAME_TYPES:
        points = list(base_points)  # include base extent too
        for leg in LEGS:
            frame_name = f"{leg}_{frame_type}"
            if frame_name not in mean_offsets:
                continue
            mean_offset = mean_offsets[frame_name]
            for trial_num, (_, leg_result) in all_trials.items():
                if frame_name not in leg_result:
                    continue
                aligned = align_leg_trajectory_to_mean_offset(
                    leg_result[frame_name], mean_offset
                )
                points.append(aligned[["x", "y", "z"]].to_numpy())
        leg_family_limits[frame_type] = calculate_common_axis_limits(points)

    # --- 4. Figure family A: Base XYZ ---
    print("=" * 70)
    print("GENERATING FIGURE FAMILY A: Base XYZ")
    print("=" * 70)
    plot_base_xyz(
        group1, base_xlim, base_ylim, base_zlim,
        "HandStand Base Trajectories (Trials 1-25)",
        os.path.join(PLOTS_ROOT, "Base_XYZ", "HandStand_Base_XYZ_Trials_01_25.png"),
    )
    plot_base_xyz(
        group2, base_xlim, base_ylim, base_zlim,
        "HandStand Base Trajectories (Trials 26-50)",
        os.path.join(PLOTS_ROOT, "Base_XYZ", "HandStand_Base_XYZ_Trials_26_50.png"),
    )

    # --- 5. Figure family B: Base + each leg-frame type ---
    frame_type_to_subdir = {
        "hip": "Base_Hip", "thigh": "Base_Thigh", "calf": "Base_Calf",
        "calflower": "Base_Calflower", "calflower1": "Base_Calflower1",
        "foot": "Base_Foot",
    }
    print("=" * 70)
    print("GENERATING FIGURE FAMILY B: Base + Leg Frames (per type)")
    print("=" * 70)
    for frame_type in FRAME_TYPES:
        subdir = frame_type_to_subdir[frame_type]
        xlim, ylim, zlim = leg_family_limits[frame_type]
        label = frame_type.capitalize()

        plot_base_and_leg_frame(
            group1, frame_type, mean_offsets, xlim, ylim, zlim,
            f"HandStand Base + {label} Trajectories (Trials 1-25)",
            os.path.join(PLOTS_ROOT, subdir,
                         f"HandStand_Base_{label}_Trials_01_25.png"),
        )
        plot_base_and_leg_frame(
            group2, frame_type, mean_offsets, xlim, ylim, zlim,
            f"HandStand Base + {label} Trajectories (Trials 26-50)",
            os.path.join(PLOTS_ROOT, subdir,
                         f"HandStand_Base_{label}_Trials_26_50.png"),
        )

    # --- 6. Base pose-space plots ---
    print("=" * 70)
    print("GENERATING BASE POSE-SPACE PLOTS")
    print("=" * 70)
    pose_space_configs = [
        (("x", "y", "yaw"), ("X (m)", "Y (m)", "Yaw (rad)"),
         "HandStand X, Y, Yaw", "HandStand_X_Y_Yaw"),
        (("x", "z", "pitch"), ("X (m)", "Z (m)", "Pitch (rad)"),
         "HandStand X, Z, Pitch", "HandStand_X_Z_Pitch"),
        (("y", "z", "roll"), ("Y (m)", "Z (m)", "Roll (rad)"),
         "HandStand Y, Z, Roll", "HandStand_Y_Z_Roll"),
    ]
    for cols, labels, title_base, filename_base in pose_space_configs:
        # Shared axis limits across BOTH groups for this pose-space config
        col_a_vals = [g[t][0][cols[0]].to_numpy() for g in (group1, group2)
                      for t in g]
        col_b_vals = [g[t][0][cols[1]].to_numpy() for g in (group1, group2)
                      for t in g]
        col_c_vals = [g[t][0][cols[2]].to_numpy() for g in (group1, group2)
                      for t in g]
        pxlim = calculate_per_axis_limits(col_a_vals)
        pylim = calculate_per_axis_limits(col_b_vals)
        pzlim = calculate_per_axis_limits(col_c_vals)

        plot_base_pose_space(
            group1, cols, labels, pxlim, pylim, pzlim,
            f"{title_base} (Trials 1-25)",
            os.path.join(PLOTS_ROOT, "Base_Pose",
                         f"{filename_base}_Trials_01_25.png"),
        )
        plot_base_pose_space(
            group2, cols, labels, pxlim, pylim, pzlim,
            f"{title_base} (Trials 26-50)",
            os.path.join(PLOTS_ROOT, "Base_Pose",
                         f"{filename_base}_Trials_26_50.png"),
        )

    print("=" * 70)
    print("ALL FIGURES GENERATED SUCCESSFULLY")
    print(f"Output directory: {PLOTS_ROOT}")
    print("=" * 70)


if __name__ == "__main__":
    if ARGS.mode == "overlay":
        main_overlay()
    else:
        main_selected()
