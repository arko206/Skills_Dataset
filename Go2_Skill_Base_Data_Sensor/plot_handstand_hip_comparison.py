"""
Multi-Trial Hip-Frame Trajectory Comparison
=======================================================
Compares the floor-relative 3D trajectories of the four hip frames
(FL_hip, FR_hip, RL_hip, RR_hip) across all trials of a given Go2 skill
mode (FrontJump, HandStand, HandStand_and_Move).

For each trial and each frame:
  - time is rebased independently:      t_rel = (t_i - t_0) / 1e9
  - position is rebased independently:  (dx, dy, dz) = (x,y,z) - (x0,y0,z0)

Cross-trial statistics (mean +/- std) are computed on a common relative-time
grid spanning [0, T_common], where T_common = min(duration) across all
trials. Interpolation is used ONLY to build this common grid for the
statistical comparison; the original recorded (rebased) trajectories are
preserved and plotted separately, unmodified.

No smoothing, no row-index averaging, no extrapolation beyond a trial's
recorded duration, and no modification of the original CSVs.
"""

import argparse
import glob
import os
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (3D projection)

# -----------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------
MODES = ["FrontJump", "HandStand", "HandStand_and_Move"]
HIP_FRAMES = ["FL_hip", "FR_hip", "RL_hip", "RR_hip"]
N_COMMON_SAMPLES = 100  # number of points on the common interpolation grid

TRIAL_COLOR_CYCLE = [
    "tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple",
    "tab:brown", "tab:pink", "tab:gray", "tab:olive", "tab:cyan",
]

parser = argparse.ArgumentParser(
    description="Compare hip-frame trajectories across trials for a Go2 skill mode."
)
parser.add_argument("--mode", required=True, choices=MODES,
                     help="Skill mode to process (selects the mode's folder).")
args = parser.parse_args()

MODE = args.mode
MODE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), MODE)
OUTPUT_DIR = os.path.join(MODE_DIR, f"Statistics_of_{MODE}")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Auto-detect trial folders (Trial_1, Trial_2, ...) inside the mode directory
trial_dirs = sorted(
    glob.glob(os.path.join(MODE_DIR, "Trial_*")),
    key=lambda p: int(re.search(r"Trial_(\d+)", p).group(1)),
)
TRIAL_NUMBERS = [int(re.search(r"Trial_(\d+)", p).group(1)) for p in trial_dirs]
if not TRIAL_NUMBERS:
    raise FileNotFoundError(f"No Trial_* folders found under {MODE_DIR}")

TRIAL_COLORS = {
    trial: TRIAL_COLOR_CYCLE[i % len(TRIAL_COLOR_CYCLE)]
    for i, trial in enumerate(TRIAL_NUMBERS)
}

INPUT_TEMPLATE = os.path.join(
    MODE_DIR, "Trial_{trial}", f"Tf_leg_stationary_{MODE}_Trial_{{trial}}.csv"
)

# -----------------------------------------------------------------------
# Step 1: Load every trial's synchronized leg CSV
# -----------------------------------------------------------------------
raw_trial_dfs = {}
for trial in TRIAL_NUMBERS:
    path = INPUT_TEMPLATE.format(trial=trial)
    raw_trial_dfs[trial] = pd.read_csv(path)

# -----------------------------------------------------------------------
# Step 2 & 3: For each trial and each hip frame:
#   - extract + sort by timestamp
#   - verify no duplicated timestamps
#   - rebase time independently: t_rel = (t - t0) / 1e9
#   - rebase position independently: subtract that frame's own first waypoint
# Results are stored as a nested dict:
#   trajectories[frame][trial] = DataFrame with columns
#       ['t_rel', 'dx', 'dy', 'dz']
# -----------------------------------------------------------------------
trajectories = {frame: {} for frame in HIP_FRAMES}

print("=" * 70)
print("VERIFICATION SUMMARY")
print("=" * 70)

for trial in TRIAL_NUMBERS:
    df = raw_trial_dfs[trial]
    print(f"\n--- Trial {trial} ---")

    for frame in HIP_FRAMES:
        frame_df = df[df["frame_name"] == frame].copy()

        if len(frame_df) == 0:
            # Do NOT silently discard missing data -- report it explicitly.
            print(f"  {frame}: MISSING -- no rows found for this frame "
                  f"in Trial {trial}!")
            continue

        # Sort by synchronized timestamp (do not assume row order)
        frame_df = frame_df.sort_values("base_common_time_ns").reset_index(drop=True)

        # Verify no duplicated synchronized timestamps for this frame
        n_duplicates = frame_df["base_common_time_ns"].duplicated().sum()
        if n_duplicates > 0:
            raise ValueError(
                f"Trial {trial}, frame {frame}: found {n_duplicates} "
                "duplicated 'base_common_time_ns' values. Refusing to "
                "silently proceed."
            )

        # --- Rebase time independently for this trial ---
        # (original 'base_common_time_ns' column is NOT modified; we only
        # add a new derived column)
        t_ns = frame_df["base_common_time_ns"].to_numpy()
        t_rel = (t_ns - t_ns[0]) / 1e9

        # --- Rebase Cartesian position independently for this frame ---
        x = frame_df["F_T03"].to_numpy()
        y = frame_df["F_T13"].to_numpy()
        z = frame_df["F_T23"].to_numpy()
        dx = x - x[0]
        dy = y - y[0]
        dz = z - z[0]

        rebased_df = pd.DataFrame({
            "t_rel": t_rel,
            "dx": dx,
            "dy": dy,
            "dz": dz,
        })
        trajectories[frame][trial] = rebased_df

        n_samples = len(rebased_df)
        duration = t_rel[-1] - t_rel[0]
        print(f"  {frame}: n_samples={n_samples:3d}   "
              f"duration={duration:.4f} s   duplicated_timestamps=0")

print("\n" + "=" * 70)

# -----------------------------------------------------------------------
# Step 4: Determine the common time interval [0, T_common] and build the
#          shared interpolation grid.
#   T_common = min over all trials AND all frames of that trial's duration
# -----------------------------------------------------------------------
all_durations = []
for frame in HIP_FRAMES:
    for trial in TRIAL_NUMBERS:
        if trial in trajectories[frame]:
            df_ft = trajectories[frame][trial]
            all_durations.append(df_ft["t_rel"].iloc[-1])

T_common = min(all_durations)
common_time_grid = np.linspace(0.0, T_common, N_COMMON_SAMPLES)

print(f"Common trajectory interval : 0.0 s to {T_common:.4f} s")
print(f"Common time grid           : {N_COMMON_SAMPLES} evenly spaced samples")
print("=" * 70)

# -----------------------------------------------------------------------
# Step 4 (cont'd): Interpolate each trial's rebased trajectory onto the
#          common time grid, per frame. This is used ONLY for cross-trial
#          statistics -- the original rebased samples in `trajectories`
#          remain untouched.
# Result: interpolated[frame] -> dict with keys 'dx','dy','dz', each an
#         array of shape (n_trials, N_COMMON_SAMPLES)
# -----------------------------------------------------------------------
interpolated = {frame: {"dx": [], "dy": [], "dz": []} for frame in HIP_FRAMES}

for frame in HIP_FRAMES:
    for trial in TRIAL_NUMBERS:
        if trial not in trajectories[frame]:
            continue  # already reported as missing above
        df_ft = trajectories[frame][trial]
        t_src = df_ft["t_rel"].to_numpy()

        # np.interp performs linear interpolation only within [t_src[0],
        # t_src[-1]]; since common_time_grid <= T_common <= t_src[-1] for
        # every trial, this never extrapolates.
        dx_interp = np.interp(common_time_grid, t_src, df_ft["dx"].to_numpy())
        dy_interp = np.interp(common_time_grid, t_src, df_ft["dy"].to_numpy())
        dz_interp = np.interp(common_time_grid, t_src, df_ft["dz"].to_numpy())

        interpolated[frame]["dx"].append(dx_interp)
        interpolated[frame]["dy"].append(dy_interp)
        interpolated[frame]["dz"].append(dz_interp)

# -----------------------------------------------------------------------
# Step 5: Compute mean and standard deviation across the 5 trials at every
#          point on the common time grid, separately for X, Y, Z.
# -----------------------------------------------------------------------
stats = {}
for frame in HIP_FRAMES:
    dx_stack = np.vstack(interpolated[frame]["dx"])  # shape (n_trials, N)
    dy_stack = np.vstack(interpolated[frame]["dy"])
    dz_stack = np.vstack(interpolated[frame]["dz"])

    stats[frame] = {
        "mean_dx": dx_stack.mean(axis=0), "std_dx": dx_stack.std(axis=0),
        "mean_dy": dy_stack.mean(axis=0), "std_dy": dy_stack.std(axis=0),
        "mean_dz": dz_stack.mean(axis=0), "std_dz": dz_stack.std(axis=0),
    }

# -----------------------------------------------------------------------
# Step 8: Save per-frame mean/std statistics to CSV
# -----------------------------------------------------------------------
for frame in HIP_FRAMES:
    s = stats[frame]
    out_df = pd.DataFrame({
        "relative_time_sec": common_time_grid,
        "mean_dx_m": s["mean_dx"], "std_dx_m": s["std_dx"],
        "mean_dy_m": s["mean_dy"], "std_dy_m": s["std_dy"],
        "mean_dz_m": s["mean_dz"], "std_dz_m": s["std_dz"],
    })
    out_path = f"{OUTPUT_DIR}/{MODE}_{frame}_mean_std.csv"
    out_df.to_csv(out_path, index=False)
    print(f"Saved statistics CSV: {out_path}")

print()

# -----------------------------------------------------------------------
# Step 6: One 3D figure per hip frame -- 5 individual rebased trajectories
#          (thin, semi-transparent) + mean trajectory (thick) + start marker
# -----------------------------------------------------------------------
for frame in HIP_FRAMES:
    fig = plt.figure(figsize=(9, 8))
    ax = fig.add_subplot(111, projection="3d")

    all_points_for_scaling = []

    # Individual trial trajectories (thin, semi-transparent)
    for trial in TRIAL_NUMBERS:
        if trial not in trajectories[frame]:
            continue
        df_ft = trajectories[frame][trial]
        ax.plot(df_ft["dx"], df_ft["dy"], df_ft["dz"],
                 color=TRIAL_COLORS[trial], alpha=0.4, linewidth=1.2,
                 label=f"Trial {trial}")
        all_points_for_scaling.append(
            df_ft[["dx", "dy", "dz"]].to_numpy()
        )

    # Mean trajectory (thick, clearly visible)
    s = stats[frame]
    ax.plot(s["mean_dx"], s["mean_dy"], s["mean_dz"],
             color="black", linewidth=3.0, label="Mean")
    all_points_for_scaling.append(
        np.column_stack([s["mean_dx"], s["mean_dy"], s["mean_dz"]])
    )

    # Mark the common starting point (0, 0, 0)
    ax.scatter([0], [0], [0], color="black", marker="o", s=120,
               edgecolor="white", linewidths=1.5, zorder=5,
               label="Start (0,0,0)")

    # Equal 3D axis scaling
    all_points_for_scaling = np.vstack(all_points_for_scaling)
    mins = all_points_for_scaling.min(axis=0)
    maxs = all_points_for_scaling.max(axis=0)
    mids = (mins + maxs) / 2
    max_range = (maxs - mins).max() / 2 * 1.1
    ax.set_xlim(mids[0] - max_range, mids[0] + max_range)
    ax.set_ylim(mids[1] - max_range, mids[1] + max_range)
    ax.set_zlim(mids[2] - max_range, mids[2] + max_range)
    try:
        ax.set_box_aspect([1, 1, 1])
    except AttributeError:
        pass

    ax.set_xlabel("\u0394X (m)")
    ax.set_ylabel("\u0394Y (m)")
    ax.set_zlabel("\u0394Z (m)")
    ax.set_title(f"{MODE}: {frame.replace('_hip', '')} Hip Trajectories "
                 f"Across {len(TRIAL_NUMBERS)} Trials")
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize=8)

    plt.tight_layout()
    out_path = f"{OUTPUT_DIR}/{MODE}_{frame}_3D_trajectories.png"
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figure: {out_path}")

print()

# -----------------------------------------------------------------------
# Step 7: Research-paper-style mean +/- SD plots (3 stacked subplots per
#          frame: dx, dy, dz vs relative time)
# -----------------------------------------------------------------------
axis_info = [
    ("dx", "\u0394X (m)"),
    ("dy", "\u0394Y (m)"),
    ("dz", "\u0394Z (m)"),
]

for frame in HIP_FRAMES:
    fig, axes = plt.subplots(3, 1, figsize=(9, 11), sharex=True)
    s = stats[frame]

    for ax_row, (axis_key, axis_label) in zip(axes, axis_info):
        # Individual trials (thin, semi-transparent), using original
        # recorded rebased samples (not the interpolated grid)
        for trial in TRIAL_NUMBERS:
            if trial not in trajectories[frame]:
                continue
            df_ft = trajectories[frame][trial]
            ax_row.plot(df_ft["t_rel"], df_ft[axis_key],
                        color=TRIAL_COLORS[trial], alpha=0.4, linewidth=1.0,
                        label=f"Trial {trial}")

        mean_vals = s[f"mean_{axis_key}"]
        std_vals = s[f"std_{axis_key}"]

        # Mean trajectory (thick)
        ax_row.plot(common_time_grid, mean_vals, color="black",
                    linewidth=2.5, label="Mean")

        # Shaded mean +/- 1 SD band
        ax_row.fill_between(common_time_grid,
                             mean_vals - std_vals, mean_vals + std_vals,
                             color="black", alpha=0.15,
                             label="Mean \u00b1 1 SD")

        ax_row.set_ylabel(axis_label)
        ax_row.grid(True, alpha=0.5)

    axes[-1].set_xlabel("Relative time (s)")
    axes[0].set_title(f"{MODE}: {frame.replace('_hip', '')} Hip "
                       f"\u0394X/\u0394Y/\u0394Z Across {len(TRIAL_NUMBERS)} Trials "
                       f"(Mean \u00b1 1 SD)")

    # Single shared legend (avoid repeating 5 trial entries x3 subplots)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper left",
               bbox_to_anchor=(1.0, 1.0), fontsize=8)

    plt.tight_layout()
    out_path = f"{OUTPUT_DIR}/{MODE}_{frame}_mean_std_2D.png"
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figure: {out_path}")

print("\nAll figures and statistics CSVs generated successfully.")
