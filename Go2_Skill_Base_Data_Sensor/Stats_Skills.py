"""
HandStandMove - Grouped Selected-Trial Base (SE(2)) Trajectory Visualization
=============================================================================

*** Currently running in BASE-ONLY mode ***
This version only loads/transforms/plots the Robot Base trajectory, since
the present planning representation is SE(2): (x, y, yaw). All leg-frame
(hip/thigh/calf/calflower/calflower1/foot) loading, synchronization
checking and plotting is disabled and kept in commented form below so it
can be restored later -- it has not been deleted.

Plots THREE independent trials together in each figure.

Trial groups (current):
    Trials 2, 3, 4    = Forward only
    Trials 8, 9, 10   = Lateral Left only
    Trials 11, 12, 13 = Lateral Right only

For each group, only the Robot Base is plotted:
    - Base X-Y-Z Cartesian trajectory
    - Base X-Y-Yaw   (the SE(2) planning representation: x, y, theta=yaw)
    - Base X-Z-Pitch
    - Base Y-Z-Roll

Color encodes the trial within the group, for easy visual distinction:
    Red, Green, Blue

All poses are expressed relative to each trial's OWN initial base pose.
The three trials remain independent and are never concatenated.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (3D projection)
from scipy.spatial.transform import Rotation as Rotation

# =========================================================================
# CONFIGURATION
# =========================================================================
DATASET_ROOT = os.path.expanduser("~/Go2_Skill_Base_Data_Sensor/HandStandMove")
PLOTS_ROOT = os.path.join(DATASET_ROOT, "Plots_Selected_Trials")

# Trials are now plotted in groups of three, three trials overlaid on the
# SAME axes (but never joined into one connected line -- see
# plot_trial_group_base_and_leg_frame / plot_trial_group_base_only).
TRIAL_GROUPS = {
    "Trials_first": [56, 57, 58, 59, 60],
    "Trials_second": [61, 62, 63, 64, 65],
    #"Trials_third": [40, 45, 46]
}
SELECTED_TRIALS = sorted({t for trials in TRIAL_GROUPS.values() for t in trials})

# Color encodes WHICH trial within a group (index 0/1/2 in that group's
# trial list), for the currently-active BASE-ONLY plots (see TRIAL_COLORS
# below).
TRIAL_COLORS = {
    0: "red",
    1: "green",
    2: "blue",
    3: "orange",
    4: "purple",
}

# --- Preserved for leg-frame mode restore -----------------------------
# When leg-frame plots are re-enabled, color goes back to encoding the
# physical frame/leg (BASE_COLOR / LEG_COLORS below) and linestyle goes
# back to encoding the trial within the group.
TRIAL_LINESTYLES = {
    0: "-",     # first trial in the group
    1: "--",    # second trial
    2: ":",     # third trial
}

FRAME_TYPES = ["hip", "thigh", "calf", "calflower", "calflower1", "foot"]
LEGS = ["FL", "FR", "RL", "RR"]

# Base-only pose-space plot families: (X,Y,Yaw), (X,Z,Pitch), (Y,Z,Roll).
# "cols" selects, per plotted axis, which array ("xyz" or "rpy") and which
# column of it to use. Units are intentionally mixed (metres + radians),
# which is why these get their own independent per-axis limit handling
# instead of the equal-XYZ-scaling used for the Cartesian plots.
POSE_TYPES = {
    "xy_yaw": {
        "cols": [("xyz", 0), ("xyz", 1), ("rpy", 2)],
        "labels": ("X (m)", "Y (m)", "Yaw (rad)"),
        "name": "X_Y_Yaw",
        "title": "X-Y-Yaw",
    },
    "xz_pitch": {
        "cols": [("xyz", 0), ("xyz", 2), ("rpy", 1)],
        "labels": ("X (m)", "Z (m)", "Pitch (rad)"),
        "name": "X_Z_Pitch",
        "title": "X-Z-Pitch",
    },
    "yz_roll": {
        "cols": [("xyz", 1), ("xyz", 2), ("rpy", 0)],
        "labels": ("Y (m)", "Z (m)", "Roll (rad)"),
        "name": "Y_Z_Roll",
        "title": "Y-Z-Roll",
    },
}
POSE_KEYS = ["xy_yaw", "xz_pitch", "yz_roll"]

# Fixed colors, identical across every figure
BASE_COLOR = "tab:blue"
LEG_COLORS = {
    "FL": "tab:orange",
    "FR": "tab:green",
    "RL": "tab:red",
    "RR": "tab:purple",
}

# Trajectory line style for GROUPED plots (three trials sharing one axes,
# so slightly lighter than the old single-trial style to stay readable)
LINE_WIDTH = 1.8        # spec: 1.8
MARKER_SIZE = 2.5       # spec: 2.5
LINE_ALPHA = 0.85       # spec: 0.85 -- not extremely transparent

# Start / End marker style (slightly reduced vs. single-trial plots,
# since up to 3 start + 3 end markers per color now share the same axes)
ENDPOINT_SIZE = 100         # spec: 90 - 110
ENDPOINT_EDGE_WIDTH = 2.0   # spec: 1.8 - 2.2
ENDPOINT_ZORDER = 10

# Two fixed camera views, identical for every group and every frame type
VIEW_1 = {"elev": 25, "azim": -60}   # standard oblique (reference script)
VIEW_2 = {"elev": 25, "azim": 30}    # rotated 90 deg in azimuth

DPI = 300
BBOX = "tight"

# --- Interactive display control -----------------------------------------
# NOTE: grouping trials cut the figure count from 168 down to
# 4 groups x 7 categories x 2 views = 56 PNGs. Every figure is saved to
# disk; plt.show() is now called for EVERY figure (unconditionally) so you
# can rotate and save each one at your preferred angle. Each call blocks
# until you close that window, so the run will pause 56 times in total --
# close a window to move on to the next figure.
SHOW_PLOTS = True

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "grid.alpha": 0.3,
})


# =========================================================================
# LOADING
# =========================================================================
def trial_paths(trial_num):
    """Return (base_csv_path, leg_csv_path) for a trial number."""
    folder = os.path.join(DATASET_ROOT, f"Trial_{trial_num}")
    base_csv = os.path.join(folder, f"Tf_stationary_HandStandMove_Trial_{trial_num}.csv")
    leg_csv = os.path.join(folder, f"Tf_leg_stationary_HandStandMove_Trial_{trial_num}.csv")
    return base_csv, leg_csv


def load_trial_csvs(trial_num):
    """
    Load one trial's base CSV.

    *** BASE-ONLY mode ***: the leg CSV (Tf_leg_stationary_..._Trial_N.csv)
    is NOT required or loaded for the current execution -- only
    Tf_stationary_..._Trial_N.csv (the base trajectory) is needed. The
    original leg-loading/validation code is preserved below in commented
    form for when leg-frame analysis is restored.

    Returns (base_df, leg_df) with leg_df always None for now, or
    (None, None) with a printed report if the base file is missing or
    malformed. Missing data is never silently dropped.
    """
    base_csv, leg_csv = trial_paths(trial_num)

    if not os.path.isfile(base_csv):
        print(f"  [MISSING] Base CSV not found: {base_csv}")
        return None, None
    # --- Leg CSV existence check disabled for BASE-ONLY mode ---
    # if not os.path.isfile(leg_csv):
    #     print(f"  [MISSING] Leg CSV not found: {leg_csv}")
    #     return None, None

    base_df = pd.read_csv(base_csv)
    # leg_df = pd.read_csv(leg_csv)
    leg_df = None

    base_required = ["common_time_ns"] + [f"T{i}{j}" for i in range(4) for j in range(4)]
    missing = [c for c in base_required if c not in base_df.columns]
    if missing:
        print(f"  [MALFORMED] Trial {trial_num} base CSV missing columns: {missing}")
        return None, None

    # --- Leg CSV column checks disabled for BASE-ONLY mode ---
    # leg_required = ["base_common_time_ns", "frame_name"] + \
    #     [f"F_T{i}{j}" for i in range(4) for j in range(4)]
    # missing = [c for c in leg_required if c not in leg_df.columns]
    # if missing:
    #     print(f"  [MALFORMED] Trial {trial_num} leg CSV missing columns: {missing}")
    #     return None, None

    return base_df, leg_df


def matrix_from_csv_row(row, prefix):
    """
    Build a 4x4 homogeneous transform from CSV columns <prefix>00..<prefix>33
    (row-major): 'T' for the base CSV, 'F_T' for the leg CSV.
    """
    M = np.eye(4)
    for i in range(4):
        for j in range(4):
            M[i, j] = row[f"{prefix}{i}{j}"]
    return M


# =========================================================================
# TIMESTAMP VERIFICATION (same approach as plot_skill_trajectories.py)
# =========================================================================
def verify_trial_synchronization(trial_num, base_df, leg_df, frame_names):
    """
    Verify that, for each requested frame, the leg CSV's
    'base_common_time_ns' values match the base CSV's 'common_time_ns'
    values EXACTLY. Raises ValueError on any mismatch -- the script never
    interpolates to paper over a synchronization problem.

    Returns (base_timestamps, frame_sample_counts).
    """
    base_timestamps = np.sort(base_df["common_time_ns"].unique())

    subset = leg_df[leg_df["frame_name"].isin(frame_names)]
    leg_timestamps = np.sort(subset["base_common_time_ns"].unique())

    if not np.array_equal(base_timestamps, leg_timestamps):
        raise ValueError(
            f"Trial {trial_num}: synchronization check FAILED. Unique "
            "'base_common_time_ns' values in the leg CSV do not exactly "
            "match 'common_time_ns' in the base CSV. Refusing to "
            f"interpolate. Base has {len(base_timestamps)} unique "
            f"timestamps, leg has {len(leg_timestamps)}."
        )

    frame_sample_counts = {}
    for frame in frame_names:
        frame_subset = subset[subset["frame_name"] == frame]
        if len(frame_subset) == 0:
            raise ValueError(
                f"Trial {trial_num}: frame '{frame}' is absent from the leg CSV."
            )
        frame_sample_counts[frame] = len(frame_subset)
        frame_ts = np.sort(frame_subset["base_common_time_ns"].values)
        if not np.array_equal(frame_ts, base_timestamps):
            raise ValueError(
                f"Trial {trial_num}: synchronization check FAILED for frame "
                f"'{frame}': its 'base_common_time_ns' values do not exactly "
                "match the base CSV timestamps."
            )

    return base_timestamps, frame_sample_counts


# =========================================================================
# COORDINATE TRANSFORM
# =========================================================================
def transform_trial_to_initial_base_frame(base_df, leg_df):
    """
    Express the base in the frame of the base's FIRST sample:
        B0_T_F    = inv(F_T_B_0)
        base(t)   = B0_T_F @ F_T_B(t)

    *** BASE-ONLY mode ***: only the base trajectory is computed. The
    original leg-frame transform (leg(t) = B0_T_F @ F_T_L(t), using the
    SAME B0_T_F so leg frames keep their physical offset from the base) is
    preserved below in commented form -- `leg_df` is accepted but unused
    for now, kept only so the function signature is unchanged when
    leg-frame analysis is restored. The base-frame maths itself is
    unchanged.

    Base orientation is extracted from the relative transform B0_T_Bt
    (never from the original floor-frame Euler columns), so translation
    and orientation are expressed in the same initial-base frame. Euler
    convention is "xyz" (roll about X, pitch about Y, yaw about Z),
    radians, unwrapped to remove +-pi discontinuities only.

    Returns:
        base_xyz  : (N, 3) array [X, Y, Z] (m), first row ~(0, 0, 0)
        base_rpy  : (N, 3) array [Roll, Pitch, Yaw] (rad), first row ~(0, 0, 0)
    """
    base_sorted = base_df.sort_values("common_time_ns").reset_index(drop=True)

    F_T_B0 = matrix_from_csv_row(base_sorted.iloc[0], "T")
    B0_T_F = np.linalg.inv(F_T_B0)

    base_points = []
    base_rpy_points = []
    for _, row in base_sorted.iterrows():
        B0_T_Bt = B0_T_F @ matrix_from_csv_row(row, "T")
        xyz = B0_T_Bt[:3, 3]
        R_relative = B0_T_Bt[:3, :3]
        roll, pitch, yaw = Rotation.from_matrix(R_relative).as_euler(
            "xyz", degrees=False
        )
        base_points.append(xyz)
        base_rpy_points.append((roll, pitch, yaw))
    base_xyz = np.vstack(base_points)
    base_rpy = np.vstack(base_rpy_points)

    # Remove artificial +-pi wraparound only -- no smoothing/filtering.
    base_rpy[:, 0] = np.unwrap(base_rpy[:, 0])
    base_rpy[:, 1] = np.unwrap(base_rpy[:, 1])
    base_rpy[:, 2] = np.unwrap(base_rpy[:, 2])

    # --- Leg-frame transformation disabled for BASE-ONLY mode ---
    # Restore this block (and the leg_xyz return value) to compute leg
    # frames again when leg-frame plotting is re-enabled.
    # leg_xyz = {}
    # for frame_name in leg_df["frame_name"].unique():
    #     frame_sorted = leg_df[leg_df["frame_name"] == frame_name].sort_values(
    #         "base_common_time_ns"
    #     ).reset_index(drop=True)
    #
    #     pts = []
    #     for _, row in frame_sorted.iterrows():
    #         B0_T_Lt = B0_T_F @ matrix_from_csv_row(row, "F_T")
    #         pts.append(B0_T_Lt[:3, 3])
    #     leg_xyz[frame_name] = np.vstack(pts)

    return base_xyz, base_rpy


# =========================================================================
# PLOTTING HELPERS (style adapted from plot_skill_trajectories.py)
# =========================================================================
def draw_trajectory(ax, xyz, color, linestyle="-", label=None):
    """
    Draw ONE trial's trajectory as its own, independent line: continuous
    line + small temporal markers, plus a large Start circle and a large
    End square with thick black edges.

    Called once per (trial, frame) pair with THAT trial's own (N,3) array
    only -- never with multiple trials concatenated -- so trials are never
    visually joined to one another (see plot_trial_group_* below).

    color     : encodes the physical frame/leg (fixed across all trials)
    linestyle : encodes WHICH trial within the group ('-', '--', ':')
    """
    x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]

    ax.plot(x, y, z, color=color, linestyle=linestyle, linewidth=LINE_WIDTH,
            marker="o", markersize=MARKER_SIZE, alpha=LINE_ALPHA, label=label)

    # Start = large filled circle
    ax.scatter(x[0], y[0], z[0], color=color, edgecolor="black",
               marker="o", s=ENDPOINT_SIZE, linewidths=ENDPOINT_EDGE_WIDTH,
               alpha=LINE_ALPHA, zorder=ENDPOINT_ZORDER)
    # End = large filled square
    ax.scatter(x[-1], y[-1], z[-1], color=color, edgecolor="black",
               marker="s", s=ENDPOINT_SIZE, linewidths=ENDPOINT_EDGE_WIDTH,
               alpha=LINE_ALPHA, zorder=ENDPOINT_ZORDER)


def calculate_equal_3d_limits(all_points, margin_factor=1.1):
    """
    Calculate one equal X/Y/Z plotting volume.

    Returns limits that can be reused by multiple figures.
    """

    x_min, y_min, z_min = all_points.min(axis=0)
    x_max, y_max, z_max = all_points.max(axis=0)

    x_mid = (x_min + x_max) / 2.0
    y_mid = (y_min + y_max) / 2.0
    z_mid = (z_min + z_max) / 2.0

    max_range = max(
        x_max - x_min,
        y_max - y_min,
        z_max - z_min,
    ) / 2.0

    max_range *= margin_factor

    if max_range == 0.0:
        max_range = 1e-3

    return {
        "xlim": (
            x_mid - max_range,
            x_mid + max_range,
        ),
        "ylim": (
            y_mid - max_range,
            y_mid + max_range,
        ),
        "zlim": (
            z_mid - max_range,
            z_mid + max_range,
        ),
    }


def apply_equal_3d_limits(ax, limits):
    """
    Apply previously calculated common XYZ limits.
    """

    ax.set_xlim(*limits["xlim"])
    ax.set_ylim(*limits["ylim"])
    ax.set_zlim(*limits["zlim"])

    try:
        ax.set_box_aspect([1, 1, 1])
    except AttributeError:
        pass



def calculate_common_category_limits(
    trial_data,
    frame_type=None,
):
    """
    Calculate common XYZ limits across ALL selected trials.

    frame_type=None:
        base-only plots (the only case currently used -- BASE-ONLY mode)

    frame_type='hip', 'thigh', etc.:
        Disabled for now -- see commented block below. Would calculate
        Base + FL/FR/RL/RR of that frame type once leg_xyz is available
        again in trial_data.
    """

    all_points = []

    for trial_num in SELECTED_TRIALS:

        if trial_num not in trial_data:
            continue

        base_xyz, base_rpy = trial_data[trial_num]

        # Base is always included.
        all_points.append(base_xyz)

        # --- Leg-frame limit calculation disabled for BASE-ONLY mode ---
        # if frame_type is None:
        #     continue
        # for leg in LEGS:
        #     frame_name = f"{leg}_{frame_type}"
        #     if frame_name not in leg_xyz:
        #         continue
        #     all_points.append(leg_xyz[frame_name])

    if not all_points:
        raise RuntimeError(
            f"No points available for frame_type={frame_type}"
        )

    all_points = np.vstack(all_points)

    return calculate_equal_3d_limits(
        all_points
    )


def extract_pose_array(base_xyz, base_rpy, pose_key):
    """
    Stack the 3 columns (from base_xyz and/or base_rpy) that a given
    pose_key (see POSE_TYPES) plots on its 3 axes, e.g. for "xy_yaw":
    [X, Y, Yaw].
    """
    source_arrays = {"xyz": base_xyz, "rpy": base_rpy}
    columns = [
        source_arrays[source][:, idx]
        for source, idx in POSE_TYPES[pose_key]["cols"]
    ]
    return np.column_stack(columns)


def calculate_independent_axis_limits(all_points, margin_factor=1.1):
    """
    Calculate plotting limits independently per axis/column of all_points
    (N, 3), WITHOUT forcing a shared numeric range across axes.

    Used for the mixed-unit (metres + radians) base-pose plots, where the
    equal-range/equal-box-aspect treatment used for Cartesian XYZ plots
    (see calculate_equal_3d_limits/apply_equal_3d_limits) would not be
    physically meaningful.
    """
    mins = all_points.min(axis=0)
    maxs = all_points.max(axis=0)

    limits = []
    for lo, hi in zip(mins, maxs):
        mid = (lo + hi) / 2.0
        half_range = (hi - lo) / 2.0 * margin_factor
        if half_range == 0.0:
            half_range = 1e-3
        limits.append((mid - half_range, mid + half_range))

    return {"xlim": limits[0], "ylim": limits[1], "zlim": limits[2]}


def apply_independent_3d_limits(ax, limits):
    """Apply previously calculated per-axis-independent pose limits."""
    ax.set_xlim(*limits["xlim"])
    ax.set_ylim(*limits["ylim"])
    ax.set_zlim(*limits["zlim"])


def calculate_common_pose_limits(trial_data, pose_key):
    """
    Calculate common (per-axis-independent) limits for one pose type
    across ALL selected trials (1,2,3,48,49,50), so the two group figures
    for that pose type share identical axis ranges.
    """
    all_points = []

    for trial_num in SELECTED_TRIALS:
        if trial_num not in trial_data:
            continue

        base_xyz, base_rpy = trial_data[trial_num]
        all_points.append(extract_pose_array(base_xyz, base_rpy, pose_key))

    if not all_points:
        raise RuntimeError(f"No points available for pose_key={pose_key}")

    all_points = np.vstack(all_points)

    return calculate_independent_axis_limits(all_points)


def build_legend(ax, trajectory_labels, colors):
    """
    Clean scientific legend placed OUTSIDE the axes: one entry per
    trajectory, plus exactly two generic marker explanations
    (Start = circle, End = square).
    """
    handles = [
        Line2D([0], [0], color=colors[lbl], lw=2.0, marker="o",
               markersize=4, label=lbl)
        for lbl in trajectory_labels
    ]
    handles += [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="gray",
               markeredgecolor="black", markeredgewidth=1.5,
               markersize=10, label="Start"),
        Line2D([0], [0], marker="s", color="w", markerfacecolor="gray",
               markeredgecolor="black", markeredgewidth=1.5,
               markersize=10, label="End"),
    ]
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.02, 1.0),
              fontsize=9, frameon=True)


def build_group_legend(ax, frame_labels, frame_colors, trial_numbers):
    """
    Compact TWO-PART legend for a grouped (3-trial) figure -- NOT one
    entry per (trial x frame) combination, which would be 15+ entries.

    Part 1 (color = physical frame): one line per frame_label, drawn
        solid, using that frame's fixed color.
    Part 2 (linestyle = trial): one line per trial in the group, drawn in
        neutral gray, using that trial's linestyle, labelled with the
        trial's REAL number (so e.g. "Trial 23 = solid" is correct even
        though 23 is index 0 within its own group).
    Plus the usual generic Start/End marker explanation.
    """
    frame_handles = [
        Line2D([0], [0], color=frame_colors[lbl], lw=2.2, linestyle="-",
               label=lbl)
        for lbl in frame_labels
    ]

    trial_handles = [
        Line2D([0], [0], color="dimgray", lw=2.0,
               linestyle=TRIAL_LINESTYLES[i], label=f"Trial {trial_num}")
        for i, trial_num in enumerate(trial_numbers)
    ]

    marker_handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="gray",
               markeredgecolor="black", markeredgewidth=1.5,
               markersize=9, label="Start"),
        Line2D([0], [0], marker="s", color="w", markerfacecolor="gray",
               markeredgecolor="black", markeredgewidth=1.5,
               markersize=9, label="End"),
    ]

    all_handles = frame_handles + trial_handles + marker_handles
    ax.legend(handles=all_handles, loc="upper left", bbox_to_anchor=(1.02, 1.0),
              fontsize=9, frameon=True)


def build_trial_color_legend(ax, trial_numbers, trial_colors):
    """
    Legend for a grouped (3-trial) BASE-ONLY figure: one colored line per
    trial (Red/Green/Blue, per TRIAL_COLORS), labelled with the trial's
    REAL number, plus the usual generic Start/End marker explanation.
    """
    trial_handles = [
        Line2D([0], [0], color=trial_colors[i], lw=2.2, linestyle="-",
               label=f"Trial {trial_num}")
        for i, trial_num in enumerate(trial_numbers)
    ]

    marker_handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="gray",
               markeredgecolor="black", markeredgewidth=1.5,
               markersize=9, label="Start"),
        Line2D([0], [0], marker="s", color="w", markerfacecolor="gray",
               markeredgecolor="black", markeredgewidth=1.5,
               markersize=9, label="End"),
    ]

    ax.legend(handles=trial_handles + marker_handles, loc="upper left",
              bbox_to_anchor=(1.02, 1.0), fontsize=9, frameon=True)


def finalize_and_output(fig, ax, title, out_path, view_angles,
                         xlabel="X (m)", ylabel="Y (m)", zlabel="Z (m)"):
    """
    Apply the camera view and caption, then display the figure once so it
    can be rotated manually. No automatic saving -- use the plot window's
    own save/export control if you want to keep a figure.

    xlabel/ylabel/zlabel default to the original Cartesian XYZ labels so
    every existing call site is unaffected; the new base-pose plots pass
    their own (e.g. "Yaw (rad)").
    """
    ax.view_init(elev=view_angles["elev"], azim=view_angles["azim"])
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_zlabel(zlabel)
    ax.set_title(title)
    ax.grid(True)

    # Concise caption instead of a long explanatory title line
    fig.text(0.5, 0.02, "Coordinates expressed relative to the initial base pose",
             ha="center", fontsize=9, style="italic", color="#444444")

    plt.tight_layout()

    plt.show()
    plt.close(fig)


# =========================================================================
# FIGURE BUILDER (LEG-FRAME): disabled for BASE-ONLY mode.
# Depends on leg_xyz, which trial_data no longer carries while leg-frame
# loading/transforming is disabled (see load_trial_csvs and
# transform_trial_to_initial_base_frame above). Preserved here, fully
# commented out, to restore together with those when leg analysis returns.
# =========================================================================
# # =========================================================================
# # FIGURE BUILDERS -- now GROUPED: 3 trials overlaid on shared axes,
# # plotted as 3 independent (unjoined) lines per trajectory.
# # =========================================================================
# def plot_trial_group_base_and_leg_frame(
#     trial_numbers,
#     trial_data,
#     frame_type,
#     out_dir,
#     group_name,
#     axis_limits,
# ):
#     """
#     One figure (x2 views) containing, for EVERY trial in trial_numbers:
#         Robot Base, FL_<frame_type>, FR_<frame_type>, RL_<frame_type>, RR_<frame_type>

#     trial_numbers : e.g. [1, 2, 3] -- the REAL trial numbers, in the order
#                     that maps to TRIAL_LINESTYLES (index 0 -> '-', etc.)
#     trial_data    : dict[trial_num] -> (base_xyz, leg_xyz), already
#                     transformed into each trial's own initial-base frame

#     Each trial's trajectory is drawn with its OWN separate ax.plot() call
#     (see draw_trajectory) -- trials are laid on the same axes but are
#     never concatenated into one array, so nothing connects the end of one
#     trial's line to the start of the next.
#     """
#     label_type = frame_type.capitalize()

#     # Frame-color legend info (color only, independent of trial)
#     frame_labels = ["Robot Base"] + [f"{leg} {label_type}" for leg in LEGS]
#     frame_colors = {"Robot Base": BASE_COLOR}
#     frame_colors.update({f"{leg} {label_type}": LEG_COLORS[leg] for leg in LEGS})

#     # Gather every trial's arrays first (for equal-axis limits) and to
#     # detect/report any missing frame without silently dropping it
#     per_trial_trajectories = {}   # trial_num -> {label: xyz}
#     for trial_num in trial_numbers:
#         base_xyz, base_rpy, leg_xyz = trial_data[trial_num]
#         traj = {"Robot Base": base_xyz}
#         for leg in LEGS:
#             frame_name = f"{leg}_{frame_type}"
#             if frame_name not in leg_xyz:
#                 print(f"    [WARNING] Trial {trial_num}: frame '{frame_name}' "
#                       "absent; skipped for this trial.")
#                 continue
#             traj[f"{leg} {label_type}"] = leg_xyz[frame_name]
#         per_trial_trajectories[trial_num] = traj

#     for view_index, view_angles in [(1, VIEW_1), (2, VIEW_2)]:
#         fig = plt.figure(figsize=(10, 8))
#         ax = fig.add_subplot(111, projection="3d")

#         # Independent trial loop: one separate ax.plot() call per
#         # (trial, frame) pair -- this is what keeps the 3 trials unjoined.
#         for i, trial_num in enumerate(trial_numbers):
#             linestyle = TRIAL_LINESTYLES[i]
#             for label, xyz in per_trial_trajectories[trial_num].items():
#                 draw_trajectory(ax, xyz, frame_colors[label], linestyle=linestyle)

#         apply_equal_3d_limits(
#             ax,
#             axis_limits,
#         )
#         build_group_legend(ax, frame_labels, frame_colors, trial_numbers)

#         trial_range_text = f"Trials {trial_numbers[0]}-{trial_numbers[-1]}"
#         title = (f"HandStandMove {trial_range_text}: Base and "
#                  f"{label_type}-Frame Trajectories")
#         filename = f"HandStandMove_{group_name}_{label_type}_View{view_index}.png"
#         out_path = os.path.join(out_dir, filename)

#         finalize_and_output(fig, ax, title, out_path, view_angles)


def plot_trial_group_base_only(
    trial_numbers,
    trial_data,
    out_dir,
    group_name,
    axis_limits,
):
    """Base-only figure (x2 views), 3 trials overlaid in Red/Green/Blue."""
    for view_index, view_angles in [(1, VIEW_1), (2, VIEW_2)]:
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection="3d")

        for i, trial_num in enumerate(trial_numbers):
            base_xyz, _ = trial_data[trial_num]
            draw_trajectory(ax, base_xyz, TRIAL_COLORS[i])

        apply_equal_3d_limits(
            ax,
            axis_limits,
        )
        build_trial_color_legend(ax, trial_numbers, TRIAL_COLORS)

        trial_range_text = f"Trials {trial_numbers[0]}-{trial_numbers[-1]}"
        title = f"HandStandMove {trial_range_text}: Base Trajectory"
        filename = f"HandStandMove_{group_name}_Base_View{view_index}.png"
        out_path = os.path.join(out_dir, filename)

        finalize_and_output(fig, ax, title, out_path, view_angles)


def plot_trial_group_base_pose(
    trial_numbers,
    trial_data,
    pose_key,
    out_dir,
    group_name,
    axis_limits,
):
    """
    Base-only pose-space figure (x2 views), 3 trials overlaid, same group
    style as plot_trial_group_base_only -- but plotting one of the
    POSE_TYPES families (e.g. X vs Y vs Yaw) instead of Cartesian XYZ.

    Trial within the group is distinguished by color (Red/Green/Blue, per
    TRIAL_COLORS), same scheme as the Cartesian base plots.
    """
    spec = POSE_TYPES[pose_key]

    for view_index, view_angles in [(1, VIEW_1), (2, VIEW_2)]:
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection="3d")

        for i, trial_num in enumerate(trial_numbers):
            base_xyz, base_rpy = trial_data[trial_num]
            pose_pts = extract_pose_array(base_xyz, base_rpy, pose_key)
            draw_trajectory(ax, pose_pts, TRIAL_COLORS[i])

        apply_independent_3d_limits(
            ax,
            axis_limits,
        )
        build_trial_color_legend(ax, trial_numbers, TRIAL_COLORS)

        trial_range_text = f"Trials {trial_numbers[0]}-{trial_numbers[-1]}"
        title = f"HandStandMove {trial_range_text}: Base {spec['title']} Trajectory"
        filename = f"HandStandMove_{group_name}_{spec['name']}_View{view_index}.png"
        out_path = os.path.join(out_dir, filename)

        finalize_and_output(
            fig, ax, title, out_path, view_angles,
            xlabel=spec["labels"][0], ylabel=spec["labels"][1], zlabel=spec["labels"][2],
        )


# =========================================================================
# MAIN
# =========================================================================
def main():
    # Output directories (BASE-ONLY mode: no per-leg-frame-type subdirs)
    subdirs = {"base": "Base", "base_pose": "Base_Pose"}
    # for ftype in FRAME_TYPES:
    #     subdirs[ftype] = ftype.capitalize()
    for sd in subdirs.values():
        os.makedirs(os.path.join(PLOTS_ROOT, sd), exist_ok=True)

    # --- Leg-frame dependency disabled for BASE-ONLY mode ---
    # all_frame_names was only needed by verify_trial_synchronization below.
    # all_frame_names = [f"{leg}_{ftype}" for ftype in FRAME_TYPES for leg in LEGS]

    # --- Pass 1: load and transform every selected trial ONCE ---
    # (independently per trial -- each trial gets its own B0_T_F, per the
    # coordinate-normalization requirement)
    trial_data = {}   # trial_num -> (base_xyz, base_rpy)  -- BASE-ONLY mode
    processed, skipped = [], []

    print("=" * 70)
    print("LOADING AND TRANSFORMING SELECTED TRIALS")
    print("=" * 70)
    for trial_num in SELECTED_TRIALS:
        print(f"--- Trial {trial_num} ---")

        base_df, leg_df = load_trial_csvs(trial_num)
        if base_df is None:
            skipped.append(trial_num)
            print(f"  -> Skipping Trial {trial_num}\n")
            continue

        # --- Base/leg synchronization check disabled for BASE-ONLY mode ---
        # Unnecessary when only the base trajectory is being analysed;
        # restore this call (and all_frame_names above) if leg-frame
        # plotting is re-enabled.
        # base_timestamps, frame_counts = verify_trial_synchronization(
        #     trial_num, base_df, leg_df, all_frame_names
        # )
        base_timestamps = np.sort(base_df["common_time_ns"].unique())
        duration = (base_timestamps[-1] - base_timestamps[0]) / 1e9
        print(f"  Base samples                : {len(base_df)}")
        print(f"  Unique base timestamps       : {len(base_timestamps)}")
        print(f"  Trajectory duration          : {duration:.3f} s")

        base_xyz, base_rpy = transform_trial_to_initial_base_frame(base_df, leg_df)
        print(f"  Base start point             : "
              f"({base_xyz[0,0]:.6f}, {base_xyz[0,1]:.6f}, {base_xyz[0,2]:.6f})")

        # Verify B0_T_B(0) ~= I: first relative sample should be ~0 in
        # both translation and orientation (small float tolerance).
        xyz0_ok = np.allclose(base_xyz[0], 0.0, atol=1e-6)
        rpy0_ok = np.allclose(base_rpy[0], 0.0, atol=1e-6)
        print(f"  Base start XYZ               : "
              f"({base_xyz[0,0]:.6e}, {base_xyz[0,1]:.6e}, {base_xyz[0,2]:.6e})"
              f"  [{'OK' if xyz0_ok else 'WARNING: not ~0'}]")
        print(f"  Base start RPY               : "
              f"({base_rpy[0,0]:.6e}, {base_rpy[0,1]:.6e}, {base_rpy[0,2]:.6e})"
              f"  [{'OK' if rpy0_ok else 'WARNING: not ~0'}]")

        trial_data[trial_num] = (base_xyz, base_rpy)
        processed.append(trial_num)
        print()

    # --- Pass 2: build grouped figures ---
    # ============================================================
    # COMMON AXIS LIMITS
    # ============================================================

    print("=" * 70)
    print("CALCULATING COMMON AXIS LIMITS")
    print("=" * 70)

    common_axis_limits = {}

    # Base-only figures
    common_axis_limits["base"] = (
        calculate_common_category_limits(
            trial_data,
            frame_type=None,
        )
    )

    # --- Leg-frame axis limits disabled for BASE-ONLY mode ---
    # Base + each leg-frame category
    # for ftype in FRAME_TYPES:
    #     common_axis_limits[ftype] = (
    #         calculate_common_category_limits(
    #             trial_data,
    #             frame_type=ftype,
    #         )
    #     )

    # Base-pose figures (X-Y-Yaw, X-Z-Pitch, Y-Z-Roll): mixed metre/radian
    # axes, so limits are computed per-axis independently (see
    # calculate_independent_axis_limits), never with the equal-XYZ scaling
    # used above.
    for pose_key in POSE_KEYS:
        common_axis_limits[pose_key] = calculate_common_pose_limits(
            trial_data,
            pose_key,
        )

    print("Common axis limits calculated.")
    print()
    print("GENERATING GROUPED FIGURES")
    print("=" * 70)
    groups_plotted = 0
    for group_name, trial_numbers in TRIAL_GROUPS.items():
        available = [t for t in trial_numbers if t in trial_data]
        missing = [t for t in trial_numbers if t not in trial_data]
        if missing:
            print(f"[WARNING] Group '{group_name}' is missing trials "
                  f"{missing} (see [MISSING]/[MALFORMED] above); plotting "
                  f"only {available}.")
        if len(available) == 0:
            print(f"[WARNING] Group '{group_name}' has NO usable trials -- "
                  "skipping this group entirely.\n")
            continue

        print(f"--- Group '{group_name}': trials {available} ---")

        plot_trial_group_base_only(
            available,
            trial_data,
            os.path.join(PLOTS_ROOT, "Base"),
            group_name,
            common_axis_limits["base"],
        )
        # for ftype in FRAME_TYPES:
        #     out_dir = os.path.join(PLOTS_ROOT, subdirs[ftype])
        #     plot_trial_group_base_and_leg_frame(
        #         available,
        #         trial_data,
        #         ftype,
        #         out_dir,
        #         group_name,
        #         common_axis_limits[ftype],
        #     )

        for pose_key in POSE_KEYS:
            plot_trial_group_base_pose(
                available,
                trial_data,
                pose_key,
                os.path.join(PLOTS_ROOT, subdirs["base_pose"]),
                group_name,
                common_axis_limits[pose_key],
            )

        groups_plotted += 1
        print()

    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Trials loaded     : {processed}")
    if skipped:
        print(f"Trials skipped    : {skipped}  (see [MISSING]/[MALFORMED] above)")
    else:
        print("Trials skipped    : none")
    categories_per_group = 1 + len(POSE_KEYS)  # base + pose (leg-frame plots disabled)
    print(f"Groups plotted    : {groups_plotted} / {len(TRIAL_GROUPS)}")
    print(f"Figures shown     : {groups_plotted * categories_per_group * 2}")
    print(f"Output directory  : {PLOTS_ROOT}")
    print("=" * 70)


if __name__ == "__main__":
    main()