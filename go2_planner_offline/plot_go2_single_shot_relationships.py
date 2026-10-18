#!/usr/bin/env python3

"""
Analyze Unitree Go2 single-shot Move() experiments.

For every file named:
    Tf_stationary_<vx>_<vy>_<vyaw>.csv

the script:
1. Reads the commanded (vx, vy, vyaw) from the filename.
2. Uses the first recorded pose as the start pose.
3. Uses the last row with stationary_candidate == 1 as the final settled pose
   (or the last row if no stationary candidate exists).
4. Computes world-frame displacement.
5. Rotates translational displacement into the START robot/body frame.
6. Wraps yaw difference to [-pi, pi].
7. Saves one summary row per experiment.
8. Plots the 9 requested input-output relationships.
9. Prints Pearson correlations and fitted straight-line equations.

Model variables:
    input  = [vx, vy, vyaw]
    output = [delta_x_body, delta_y_body, delta_theta]
"""

from pathlib import Path
import math
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# USER SETTINGS
# ============================================================

# DATA_DIR = Path(
#     "/home/unitree-arka/Go2_Walk_Base_Data_Sensor/"
#     "Old_Data_Collection/Parameters_RS/"
#     "Stationary_CSV_req"
# )

DATA_DIR = Path(
    "/home/unitree-arka/Go2_Walk_Base_Data_Sensor/New_Data_collection/"
   
)

FILE_PATTERN = "Tf_stationary_*.csv"

SUMMARY_CSV = DATA_DIR / "new_single_shot_body_displacement_summary.csv"
PLOT_PNG = DATA_DIR / "new_single_shot_9_relationships.png"
PLOT_PDF = DATA_DIR / "new_single_shot_9_relationships.pdf"


# ============================================================
# HELPERS
# ============================================================

def wrap_angle(angle_rad: float) -> float:
    """Wrap angle to [-pi, pi]."""
    return math.atan2(math.sin(angle_rad), math.cos(angle_rad))


def parse_command_from_filename(path: Path):
    """
    Parse:
        Tf_stationary_0.5_0.0_-0.5.csv
    into:
        vx=0.5, vy=0.0, vyaw=-0.5
    """
    pattern = (
        r"^Tf_stationary_"
        r"(-?\d+(?:\.\d+)?)_"
        r"(-?\d+(?:\.\d+)?)_"
        r"(-?\d+(?:\.\d+)?)\.csv$"
    )

    match = re.match(pattern, path.name)
    if match is None:
        raise ValueError(
            f"Filename does not match expected format: {path.name}"
        )

    vx = float(match.group(1))
    vy = float(match.group(2))
    vyaw = float(match.group(3))

    return vx, vy, vyaw


def choose_start_and_final_pose(df: pd.DataFrame):
    """
    Start:
        first recorded row.

    Final:
        last row where stationary_candidate == 1.
        If no such row exists, use the final row in the CSV.
    """
    required = [
        "x_floor_m",
        "y_floor_m",
        "yaw_floor_rad",
    ]

    missing = [column for column in required if column not in df.columns]
    if missing:
        raise KeyError(f"Missing required CSV columns: {missing}")

    if len(df) < 2:
        raise ValueError("CSV must contain at least two pose rows.")

    start_row = df.iloc[0]

    if "stationary_candidate" in df.columns:
        stationary_rows = df[df["stationary_candidate"] == 1]

        if not stationary_rows.empty:
            final_row = stationary_rows.iloc[-1]
        else:
            final_row = df.iloc[-1]
    else:
        final_row = df.iloc[-1]

    return start_row, final_row


def compute_body_displacement(start_row, final_row):
    """
    Compute final-minus-start SE(2) displacement and express translation
    in the robot body frame attached to the START pose.

    World displacement:
        dx_g = x_f - x_s
        dy_g = y_f - y_s

    World -> start-body rotation:
        dx_b =  cos(theta_s) dx_g + sin(theta_s) dy_g
        dy_b = -sin(theta_s) dx_g + cos(theta_s) dy_g

    Yaw:
        dtheta = wrap(theta_f - theta_s)
    """

    x_start = float(start_row["x_floor_m"])
    y_start = float(start_row["y_floor_m"])
    theta_start = float(start_row["yaw_floor_rad"])

    x_final = float(final_row["x_floor_m"])
    y_final = float(final_row["y_floor_m"])
    theta_final = float(final_row["yaw_floor_rad"])

    # World/floor-frame displacement
    delta_x_global = x_final - x_start
    delta_y_global = y_final - y_start

    # Rotate world displacement into robot body frame at START pose
    cos_theta = math.cos(theta_start)
    sin_theta = math.sin(theta_start)

    delta_x_body = (
        cos_theta * delta_x_global
        + sin_theta * delta_y_global
    )

    delta_y_body = (
        -sin_theta * delta_x_global
        + cos_theta * delta_y_global
    )

    # Wrapped yaw displacement
    delta_theta = wrap_angle(theta_final - theta_start)

    return {
        "x_start": x_start,
        "y_start": y_start,
        "theta_start": theta_start,
        "x_final": x_final,
        "y_final": y_final,
        "theta_final": theta_final,
        "delta_x_global": delta_x_global,
        "delta_y_global": delta_y_global,
        "delta_x_body": delta_x_body,
        "delta_y_body": delta_y_body,
        "delta_theta_rad": delta_theta,
        "delta_theta_deg": math.degrees(delta_theta),
    }


def safe_correlation(x, y):
    """
    Pearson correlation.
    Returns NaN if x or y has zero variance.
    """
    # Convert inputs to numeric arrays so the correlation calculation is robust.
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    # A Pearson correlation is undefined for fewer than two samples.
    if len(x) < 2:
        return np.nan

    # If either variable is constant, the correlation is undefined.
    if np.isclose(np.std(x), 0.0) or np.isclose(np.std(y), 0.0):
        return np.nan

    # Pearson r between the two variables; result is NaN-safe when variance is zero.
    return float(np.corrcoef(x, y)[0, 1])


def add_scatter_and_fit(ax, x, y, xlabel, ylabel):
    """Scatter plot plus least-squares straight-line fit."""
    # Convert to arrays to ensure consistent numeric plotting behavior.
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    # Plot the raw experimental points as a scatter cloud.
    ax.scatter(x, y, s=45)

    # Compute correlation strength to summarize the relationship.
    r = safe_correlation(x, y)

    # A line fit only makes sense when x varies across at least two distinct values.
    if len(x) >= 2 and len(np.unique(x)) >= 2:
        # Fit the best straight line y = m*x + b with least squares.
        slope, intercept = np.polyfit(x, y, 1)

        # Build a line across the range of x values for plotting.
        x_line = np.linspace(np.min(x), np.max(x), 200)
        y_line = slope * x_line + intercept

        # Draw the fitted trend line on top of the scatter points.
        ax.plot(x_line, y_line, linewidth=1.5)

        # Show either the line equation alone or the equation plus Pearson r.
        if np.isnan(r):
            title = f"y = {slope:.4f}x + {intercept:.4f}"
        else:
            title = (
                f"y = {slope:.4f}x + {intercept:.4f}\n"
                f"Pearson r = {r:.3f}"
            )
    else:
        # Not enough variation in x to justify a trend line.
        title = "Insufficient variation for linear fit"

    # Label the axes and add the summary title to the subplot.
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=9)
    ax.grid(True, alpha=0.3)


# ============================================================
# READ ALL EXPERIMENTS
# ============================================================

files = sorted(DATA_DIR.glob(FILE_PATTERN))

if not files:
    raise FileNotFoundError(
        f"No files matching {FILE_PATTERN} were found in:\n{DATA_DIR}"
    )

records = []

for csv_path in files:
    try:
        vx_cmd, vy_cmd, vyaw_cmd = parse_command_from_filename(csv_path)

        df = pd.read_csv(csv_path)

        start_row, final_row = choose_start_and_final_pose(df)
        displacement = compute_body_displacement(start_row, final_row)

        record = {
            "file": csv_path.name,
            "vx_cmd": vx_cmd,
            "vy_cmd": vy_cmd,
            "vyaw_cmd": vyaw_cmd,
            **displacement,
        }

        records.append(record)

    except Exception as exc:
        print(f"[WARNING] Skipping {csv_path.name}: {exc}")


summary = pd.DataFrame(records)

if summary.empty:
    raise RuntimeError("No valid experiments could be processed.")


# ============================================================
# SAVE EXTRACTED TRAINING TABLE
# ============================================================

summary.to_csv(SUMMARY_CSV, index=False)

print("\n============================================================")
print("Single-shot dataset summary")
print("============================================================")
print(
    summary[
        [
            "file",
            "vx_cmd",
            "vy_cmd",
            "vyaw_cmd",
            "delta_x_body",
            "delta_y_body",
            "delta_theta_rad",
        ]
    ].to_string(index=False)
)

print(f"\nSaved summary CSV:\n{SUMMARY_CSV}")


# ============================================================
# 9 INPUT-OUTPUT RELATIONSHIPS
# ============================================================

relationships = [
    ("vx_cmd",   "delta_x_body",    r"$v_x$ (m/s)",             r"$\Delta x_b$ (m)"),
    ("vx_cmd",   "delta_y_body",    r"$v_x$ (m/s)",             r"$\Delta y_b$ (m)"),
    ("vx_cmd",   "delta_theta_rad", r"$v_x$ (m/s)",             r"$\Delta \theta$ (rad)"),

    ("vy_cmd",   "delta_x_body",    r"$v_y$ (m/s)",             r"$\Delta x_b$ (m)"),
    ("vy_cmd",   "delta_y_body",    r"$v_y$ (m/s)",             r"$\Delta y_b$ (m)"),
    ("vy_cmd",   "delta_theta_rad", r"$v_y$ (m/s)",             r"$\Delta \theta$ (rad)"),

    ("vyaw_cmd", "delta_x_body",    r"$v_{\mathrm{yaw}}$ (rad/s)", r"$\Delta x_b$ (m)"),
    ("vyaw_cmd", "delta_y_body",    r"$v_{\mathrm{yaw}}$ (rad/s)", r"$\Delta y_b$ (m)"),
    ("vyaw_cmd", "delta_theta_rad", r"$v_{\mathrm{yaw}}$ (rad/s)", r"$\Delta \theta$ (rad)"),
]

fig, axes = plt.subplots(3, 3, figsize=(15, 12))

for ax, (x_col, y_col, x_label, y_label) in zip(
    axes.flatten(),
    relationships
):
    add_scatter_and_fit(
        ax,
        summary[x_col],
        summary[y_col],
        x_label,
        y_label,
    )

fig.suptitle(
    "Unitree Go2 Single-Shot Command vs Body-Frame Landing Displacement",
    fontsize=14,
)

fig.tight_layout(rect=[0, 0, 1, 0.96])

fig.savefig(PLOT_PNG, dpi=300, bbox_inches="tight")
fig.savefig(PLOT_PDF, bbox_inches="tight")

print(f"\nSaved 9-panel PNG:\n{PLOT_PNG}")
print(f"Saved 9-panel PDF:\n{PLOT_PDF}")


# ============================================================
# PRINT PEARSON CORRELATION TABLE
# ============================================================

print("\n============================================================")
print("Pairwise Pearson correlations")
print("============================================================")

for x_col, y_col, _, _ in relationships:
    r = safe_correlation(summary[x_col], summary[y_col])

    if np.isnan(r):
        print(f"{x_col:10s} vs {y_col:16s}: r = undefined")
    else:
        print(f"{x_col:10s} vs {y_col:16s}: r = {r:+.4f}")


plt.show()
