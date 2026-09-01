#!/usr/bin/env python3

"""
Analyse repeated Unitree Go2 stationary-trajectory CSV files.

For every Tf_stationary*.csv file in DATA_DIR:

1. Read the first and last valid pose rows.
2. Compute the 3D Euclidean start-to-stationary displacement:

       d = sqrt((x_end-x_start)^2
              + (y_end-y_start)^2
              + (z_end-z_start)^2)

   and convert metres -> centimetres.

3. Compute the AprilTag sensor-time duration:

       T = (common_time_ns_end - common_time_ns_start) / 1e9

4. Plot:
       X-axis = AprilTag sensor duration [s]
       Y-axis = Euclidean displacement [cm]

5. Across all trials, calculate:
       mean distance
       sample standard deviation (ddof=1)

   The plot shows:
       - one scatter point per trial
       - a horizontal mean line
       - a shaded mean +/- 1 standard-deviation band

6. Save:
       go2_distance_vs_sensor_duration.png
       go2_trial_distance_summary.csv
"""

from pathlib import Path
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# USER SETTINGS
# ============================================================

DATA_DIR = Path(
    "/home/unitree-arka/"
    "Go2_Walk_Base_Data_Sensor/"
    "Trials_0.2_0.15_0.0/"
)

FILE_PATTERN = "Tf_stationary_0.2_0.15_0.0.csv"

PLOT_OUTPUT = DATA_DIR / "go2_distance_vs_sensor_duration.png"
SUMMARY_OUTPUT = DATA_DIR / "go2_trial_distance_summary.csv"


# ============================================================
# HELPERS
# ============================================================

def natural_sort_key(path: Path):
    """
    Natural filename sorting:
        file2.csv comes before file10.csv
    """
    return [
        int(part) if part.isdigit() else part.lower()
        for part in re.split(r"(\d+)", path.name)
    ]


def analyse_stationary_csv(csv_path: Path):
    """
    Return metrics for one stationary trajectory CSV.
    """

    required_columns = [
        "common_time_ns",
        "x_floor_m",
        "y_floor_m",
        "z_floor_m",
    ]

    df = pd.read_csv(csv_path)

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"{csv_path.name}: missing columns {missing_columns}"
        )

    # Keep only rows that contain all values needed for this analysis.
    valid = df[required_columns].dropna().copy()

    if len(valid) < 2:
        raise ValueError(
            f"{csv_path.name}: fewer than two valid pose rows."
        )

    # Preserve sensor-time order.
    valid = valid.sort_values("common_time_ns").reset_index(drop=True)

    first = valid.iloc[0]
    last = valid.iloc[-1]

    # --------------------------------------------------------
    # Start -> stationary 3D Euclidean displacement
    # --------------------------------------------------------
    dx_m = float(last["x_floor_m"] - first["x_floor_m"])
    dy_m = float(last["y_floor_m"] - first["y_floor_m"])
    dz_m = float(last["z_floor_m"] - first["z_floor_m"])

    euclidean_distance_m = np.sqrt(
        dx_m**2
        + dy_m**2
        + dz_m**2
    )

    euclidean_distance_cm = euclidean_distance_m * 100.0

    # --------------------------------------------------------
    # AprilTag sensor-time duration
    # --------------------------------------------------------
    start_sensor_time_ns = int(first["common_time_ns"])
    stationary_sensor_time_ns = int(last["common_time_ns"])

    sensor_duration_sec = (
        stationary_sensor_time_ns
        - start_sensor_time_ns
    ) / 1_000_000_000.0

    if sensor_duration_sec <= 0.0:
        raise ValueError(
            f"{csv_path.name}: non-positive sensor duration."
        )

    return {
        "filename": csv_path.name,
        "start_sensor_time_ns": start_sensor_time_ns,
        "stationary_sensor_time_ns": stationary_sensor_time_ns,
        "sensor_duration_sec": sensor_duration_sec,
        "x_start_m": float(first["x_floor_m"]),
        "y_start_m": float(first["y_floor_m"]),
        "z_start_m": float(first["z_floor_m"]),
        "x_stationary_m": float(last["x_floor_m"]),
        "y_stationary_m": float(last["y_floor_m"]),
        "z_stationary_m": float(last["z_floor_m"]),
        "dx_m": dx_m,
        "dy_m": dy_m,
        "dz_m": dz_m,
        "euclidean_distance_cm": euclidean_distance_cm,
    }


# ============================================================
# MAIN
# ============================================================

def main():

    if not DATA_DIR.is_dir():
        raise FileNotFoundError(
            f"Data directory does not exist:\n{DATA_DIR}"
        )

    csv_files = sorted(
        DATA_DIR.glob(
            f"Trial_*/{FILE_PATTERN}"
        ),
        key=lambda path: natural_sort_key(path.parent),
    )

    if not csv_files:
        raise FileNotFoundError(
            f"No files matching '{FILE_PATTERN}' found in:\n"
            f"{DATA_DIR}"
        )

    print(
        f"[INFO] Found {len(csv_files)} stationary CSV file(s)."
    )

    results = []

    for trial_index, csv_path in enumerate(csv_files, start=1):

        try:
            result = analyse_stationary_csv(csv_path)

        except Exception as error:
            print(
                f"[WARNING] Skipping {csv_path.name}: {error}"
            )
            continue

        result["trial"] = trial_index
        results.append(result)

        print(
            f"[TRIAL {trial_index:02d}] "
            f"T_sensor = {result['sensor_duration_sec']:.6f} s, "
            f"d_3D = {result['euclidean_distance_cm']:.3f} cm, "
            f"file = {csv_path.name}"
        )

    if not results:
        raise RuntimeError(
            "No valid stationary CSV files were available for analysis."
        )

    summary = pd.DataFrame(results)

    # Reorder the most useful columns first.
    preferred_columns = [
        "trial",
        "filename",
        "sensor_duration_sec",
        "euclidean_distance_cm",
        "start_sensor_time_ns",
        "stationary_sensor_time_ns",
        "x_start_m",
        "y_start_m",
        "z_start_m",
        "x_stationary_m",
        "y_stationary_m",
        "z_stationary_m",
        "dx_m",
        "dy_m",
        "dz_m",
    ]

    summary = summary[preferred_columns]

    distances_cm = summary[
        "euclidean_distance_cm"
    ].to_numpy(dtype=float)

    durations_sec = summary[
        "sensor_duration_sec"
    ].to_numpy(dtype=float)

    # Arithmetic mean across repeated trials.
    mean_distance_cm = np.mean(distances_cm)

    # Sample standard deviation is normally the appropriate quantity
    # when the repeated trials are treated as a sample of robot behaviour.
    if len(distances_cm) >= 2:
        std_distance_cm = np.std(
            distances_cm,
            ddof=1,
        )
    else:
        std_distance_cm = 0.0

    print()
    print("==============================================")
    print("DISTANCE REPEATABILITY SUMMARY")
    print("==============================================")
    print(f"Number of valid trials : {len(distances_cm)}")
    print(
        f"Mean distance          : "
        f"{mean_distance_cm:.3f} cm"
    )
    print(
        f"Sample std. deviation  : "
        f"{std_distance_cm:.3f} cm"
    )
    print(
        f"Mean +/- std           : "
        f"{mean_distance_cm:.3f} +/- "
        f"{std_distance_cm:.3f} cm"
    )
    print("==============================================")

    # Save numerical results.
    summary.to_csv(
        SUMMARY_OUTPUT,
        index=False,
    )

    # ========================================================
    # PLOT
    # ========================================================

    fig, ax = plt.subplots(
        figsize=(10, 6)
    )

    # One point per trial.
    ax.scatter(
        durations_sec,
        distances_cm,
        s=75,
        label="Individual trials",
        zorder=3,
    )

    # Label each point T1, T2, ...
    for _, row in summary.iterrows():
        ax.annotate(
            f"T{int(row['trial'])}",
            (
                row["sensor_duration_sec"],
                row["euclidean_distance_cm"],
            ),
            xytext=(6, 6),
            textcoords="offset points",
            fontsize=9,
        )

    # Mean line.
    ax.axhline(
        mean_distance_cm,
        linestyle="--",
        linewidth=2,
        label=(
            f"Mean distance = "
            f"{mean_distance_cm:.2f} cm"
        ),
    )

    # Mean +/- one sample standard deviation.
    x_min = float(np.min(durations_sec))
    x_max = float(np.max(durations_sec))

    # Avoid a zero-width fill region if only one duration exists.
    if np.isclose(x_min, x_max):
        x_min -= 0.05
        x_max += 0.05

    ax.fill_between(
        [x_min, x_max],
        [
            mean_distance_cm - std_distance_cm,
            mean_distance_cm - std_distance_cm,
        ],
        [
            mean_distance_cm + std_distance_cm,
            mean_distance_cm + std_distance_cm,
        ],
        alpha=0.18,
        label=(
            f"Mean +/- 1 SD "
            f"({std_distance_cm:.2f} cm)"
        ),
    )

    # Summary box.
    summary_text = (
        f"N = {len(distances_cm)}\n"
        f"Mean = {mean_distance_cm:.2f} cm\n"
        f"SD = {std_distance_cm:.2f} cm"
    )

    ax.text(
        0.02,
        0.98,
        summary_text,
        transform=ax.transAxes,
        va="top",
        ha="left",
        bbox=dict(
            boxstyle="round",
            alpha=0.15,
        ),
    )

    ax.set_xlabel(
        "AprilTag sensor-time duration: "
        "(stationary common_time_ns - start common_time_ns) [s]"
    )

    ax.set_ylabel(
        "Start-to-stationary 3D Euclidean displacement [cm]"
    )

    ax.set_title(
        "Unitree Go2 Repeatability: "
        "Distance vs AprilTag Sensor-Time Duration\n"
        "Command = (Vx, Vy, Vyaw) = (0.15, 0.0, 0.0)"
    )

    ax.grid(
        True,
        alpha=0.3,
    )

    ax.legend()

    fig.tight_layout()

    fig.savefig(
        PLOT_OUTPUT,
        dpi=300,
        bbox_inches="tight",
    )

    print()
    print(f"[SAVED] Summary CSV: {SUMMARY_OUTPUT}")
    print(f"[SAVED] Plot       : {PLOT_OUTPUT}")

    plt.show()


if __name__ == "__main__":
    main()
