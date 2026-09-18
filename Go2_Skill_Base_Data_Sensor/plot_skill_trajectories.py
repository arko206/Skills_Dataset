"""
FrontJump Trial 3 - 3D Cartesian Trajectory Visualization
===========================================================
Plots the floor-relative 3D trajectories of the robot base and the four
hip frames (FL_hip, FR_hip, RL_hip, RR_hip) recorded during a Unitree Go2
FrontJump experiment.

Inputs:
  - Tf_stationary_FrontJump_Trial_3.csv       (robot base pose)
  - Tf_leg_stationary_FrontJump_Trial_3.csv   (synchronized leg/frame poses)

No interpolation, resampling, or smoothing is performed. The two CSVs were
already synchronized offline; this script only verifies that the shared
timestamp columns match exactly.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (needed for 3D projection)

# -----------------------------------------------------------------------
# 1. Load the CSV files
# -----------------------------------------------------------------------
BASE_CSV = "/home/unitree-arka/Go2_Skill_Base_Data_Sensor/HandStand/Trial_10/Tf_stationary_HandStand_Trial_10.csv"
LEG_CSV = "/home/unitree-arka/Go2_Skill_Base_Data_Sensor/HandStand/Trial_10/Tf_leg_stationary_HandStand_Trial_10.csv"

base_df = pd.read_csv(BASE_CSV)
leg_df = pd.read_csv(LEG_CSV)

# -----------------------------------------------------------------------
# 2. Extract only the four hip frames we care about for this analysis
# -----------------------------------------------------------------------
HIP_FRAMES = ["FL_hip", "FR_hip", "RL_hip", "RR_hip"]
hip_df = leg_df[leg_df["frame_name"].isin(HIP_FRAMES)].copy()

# -----------------------------------------------------------------------
# 3. Verify that base and leg synchronized timestamps match EXACTLY
#    (no interpolation/resampling is allowed -- raise an error if they
#    do not line up one-to-one)
# -----------------------------------------------------------------------
base_timestamps = np.sort(base_df["common_time_ns"].unique())
leg_timestamps = np.sort(hip_df["base_common_time_ns"].unique())

timestamps_match = np.array_equal(base_timestamps, leg_timestamps)

if not timestamps_match:
    raise ValueError(
        "Synchronization check FAILED: unique 'base_common_time_ns' values in "
        "the leg CSV do not exactly match 'common_time_ns' values in the base "
        "CSV. Refusing to interpolate or silently proceed. "
        f"Base has {len(base_timestamps)} unique timestamps, "
        f"leg has {len(leg_timestamps)} unique timestamps."
    )

# Also verify each hip frame individually has one sample per timestamp
# (i.e. no missing/duplicated samples for any single frame)
frame_sample_counts = {}
for frame in HIP_FRAMES:
    frame_subset = hip_df[hip_df["frame_name"] == frame]
    frame_sample_counts[frame] = len(frame_subset)
    frame_ts_sorted = np.sort(frame_subset["base_common_time_ns"].values)
    if not np.array_equal(frame_ts_sorted, base_timestamps):
        raise ValueError(
            f"Synchronization check FAILED for frame '{frame}': its "
            "'base_common_time_ns' values do not exactly match the base "
            "CSV timestamps."
        )

# -----------------------------------------------------------------------
# 4. Build a relative time variable (t_relative = (t_i - t0) / 1e9)
#    This is ONLY for reference/printing -- original timestamp columns
#    in the dataframes are left untouched.
# -----------------------------------------------------------------------
t0 = base_timestamps[0]
relative_time_s = (base_timestamps - t0) / 1e9
total_duration_s = relative_time_s[-1] - relative_time_s[0]

# -----------------------------------------------------------------------
# 5. Print verification summary (as requested, before plotting)
# -----------------------------------------------------------------------
print("=" * 60)
print("VERIFICATION SUMMARY")
print("=" * 60)
print(f"Number of base samples                : {len(base_df)}")
print(f"Number of unique base timestamps       : {len(base_timestamps)}")
print(f"Number of unique leg (hip) timestamps  : {len(leg_timestamps)}")
for frame in HIP_FRAMES:
    print(f"  Samples for {frame:8s}             : {frame_sample_counts[frame]}")
print(f"Base/leg timestamps match exactly      : {timestamps_match}")
print(f"Total trajectory duration              : {total_duration_s:.3f} s")
print("=" * 60)

# -----------------------------------------------------------------------
# 6. Build ordered (x, y, z) trajectory arrays for each of the 5 entities,
#    sorted by timestamp to guarantee correct temporal ordering.
# -----------------------------------------------------------------------
base_sorted = base_df.sort_values("common_time_ns")
base_xyz = base_sorted[["x_floor_m", "y_floor_m", "z_floor_m"]].to_numpy()

trajectories = {"Robot Base": base_xyz}

for frame in HIP_FRAMES:
    frame_sorted = hip_df[hip_df["frame_name"] == frame].sort_values(
        "base_common_time_ns"
    )
    trajectories[frame] = frame_sorted[["F_T03", "F_T13", "F_T23"]].to_numpy()

# -----------------------------------------------------------------------
# 7. Plot all 5 trajectories in a single 3D Matplotlib figure
# -----------------------------------------------------------------------
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection="3d")

colors = {
    "Robot Base": "tab:blue",
    "FL_hip": "tab:orange",
    "FR_hip": "tab:green",
    "RL_hip": "tab:red",
    "RR_hip": "tab:purple",
}

all_points = []  # collect all points for equal-axis scaling later

for label, xyz in trajectories.items():
    x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    color = colors[label]

    # Trajectory line + markers
    ax.plot(x, y, z, color=color, linewidth=1.8, marker="o", markersize=3,
             label=label)

    # Mark start point (green-edged circle) and end point (black-edged square)
    ax.scatter(x[0], y[0], z[0], color=color, edgecolor="black",
               marker="o", s=110, linewidths=1.5, zorder=5)
    ax.scatter(x[-1], y[-1], z[-1], color=color, edgecolor="black",
               marker="s", s=110, linewidths=1.5, zorder=5)

    all_points.append(xyz)

all_points = np.vstack(all_points)

# -----------------------------------------------------------------------
# 8. Enforce equal/visually comparable scaling on all three spatial axes
#    so the geometry of the motion is not visually distorted.
# -----------------------------------------------------------------------
x_min, y_min, z_min = all_points.min(axis=0)
x_max, y_max, z_max = all_points.max(axis=0)

x_mid, y_mid, z_mid = (x_min + x_max) / 2, (y_min + y_max) / 2, (z_min + z_max) / 2
max_range = max(x_max - x_min, y_max - y_min, z_max - z_min) / 2
# Add a small margin so markers aren't clipped at the plot edges
max_range *= 1.1

ax.set_xlim(x_mid - max_range, x_mid + max_range)
ax.set_ylim(y_mid - max_range, y_mid + max_range)
ax.set_zlim(z_mid - max_range, z_mid + max_range)

try:
    ax.set_box_aspect([1, 1, 1])  # matplotlib >= 3.3
except AttributeError:
    pass

# -----------------------------------------------------------------------
# 9. Labels, title, legend
# -----------------------------------------------------------------------
ax.set_xlabel("X position (m)")
ax.set_ylabel("Y position (m)")
ax.set_zlabel("Z position (m)")
ax.set_title("FrontJump Trial 3: Base and Hip-Frame 3D Trajectories")

# Custom legend entries to explain the start/end markers, in addition to
# the automatically generated per-trajectory color legend.
from matplotlib.lines import Line2D
marker_legend = [
    Line2D([0], [0], marker="o", color="w", markerfacecolor="gray",
           markeredgecolor="black", markersize=9, label="Start"),
    Line2D([0], [0], marker="s", color="w", markerfacecolor="gray",
           markeredgecolor="black", markersize=9, label="End"),
]
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles=handles + marker_legend, labels=labels + ["Start", "End"],
          loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize=9)

plt.tight_layout()
plt.savefig("/home/unitree-arka/Go2_Skill_Base_Data_Sensor/Verifyfrontjump_trial3_3d_trajectories.png",
            dpi=200, bbox_inches="tight")
plt.show()

print("\nPlot saved to: /home/unitree-arka/Go2_Skill_Base_Data_Sensor/Verifyfrontjump_trial3_3d_trajectories.png")
