#!/usr/bin/env python3
"""
Plot AprilTag elapsed sensor time versus retained sample number for repeated trials.

For each stationary-filtered trajectory:
    elapsed_time_i =
        (common_time_ns_i - common_time_ns_0) / 1e9

X-axis:
    Sample number from the start of the stationary-filtered trajectory.

Y-axis:
    Elapsed AprilTag sensor time in seconds.

No mean or standard deviation is calculated.

Each circular marker corresponds to one recorded AprilTag sample.
No reference-frequency lines, mean, variance, or standard deviation are plotted.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def trial_number_from_path(path: Path) -> int:
    """
    Infer trial number from any parent directory such as:
        Trial_1_0.15_0.0_0.0
        Trail_1_0.15_0.0_0.0

    Both Trial_ and Trail_ are accepted.
    """
    pattern = re.compile(r"Tr(?:ial|ail)_(\d+)", re.IGNORECASE)

    for candidate in [path.parent.name] + [p.name for p in path.parents]:
        match = pattern.search(candidate)
        if match:
            return int(match.group(1))

    # If no trial folder is available, try a timestamp suffix or simply
    # place the file after numbered trials.
    return 10**9


def find_stationary_csvs(
    trials_dir: Path,
    stationary_filename: str,
) -> List[Path]:
    """Recursively find stationary CSV files for the selected command."""
    if not trials_dir.is_dir():
        raise NotADirectoryError(f"Trials directory not found: {trials_dir}")

    exact_matches = list(trials_dir.rglob(stationary_filename))

    if exact_matches:
        files = exact_matches
    else:
        # Fallback supports files with suffixes added by copying/downloading,
        # e.g. Tf_stationary_0.15_0.0_0.0(1).csv.
        stem = Path(stationary_filename).stem
        files = list(trials_dir.rglob(f"{stem}*.csv"))

    if not files:
        raise FileNotFoundError(
            f"No files matching {stationary_filename!r} were found under "
            f"{trials_dir}"
        )

    files.sort(key=lambda p: (trial_number_from_path(p), str(p)))
    return files


def load_elapsed_sensor_time(csv_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Return 1-based sample number and elapsed AprilTag sensor time."""
    df = pd.read_csv(csv_path)

    if "common_time_ns" not in df.columns:
        raise ValueError(
            f"{csv_path} does not contain the required column 'common_time_ns'."
        )

    sensor_time_ns = pd.to_numeric(
        df["common_time_ns"],
        errors="coerce",
    )

    if sensor_time_ns.isna().any():
        bad_rows = sensor_time_ns.index[sensor_time_ns.isna()].tolist()
        raise ValueError(
            f"{csv_path} contains invalid common_time_ns values at rows "
            f"{bad_rows}."
        )

    sensor_time_ns = sensor_time_ns.astype("int64")

    if not sensor_time_ns.is_monotonic_increasing:
        raise ValueError(
            f"common_time_ns is not monotonically increasing in {csv_path}."
        )

    elapsed_time_sec = (
        sensor_time_ns - sensor_time_ns.iloc[0]
    ).astype("float64") / 1.0e9

    # 1-based sample numbering:
    # sample 1 corresponds to elapsed sensor time t = 0 s.
    sample_number = np.arange(1, len(df) + 1)

    return sample_number, elapsed_time_sec.to_numpy(dtype=float)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compare AprilTag elapsed sensor time versus sample number "
            "across repeated stationary-filtered trials."
        )
    )

    parser.add_argument(
        "--trials-dir",
        type=Path,
        required=True,
        help=(
            "Directory containing Trial_1..., Trial_2..., etc. "
            "The search is recursive."
        ),
    )

    parser.add_argument(
        "--stationary-filename",
        default="Tf_stationary_0.15_0.0_0.0.csv",
        help=(
            "Stationary-filtered CSV filename expected inside each trial "
            "folder."
        ),
    )

    parser.add_argument(
        "--max-trials",
        type=int,
        default=11,
        help="Maximum number of trials to plot (default: 11).",
    )

    parser.add_argument(
        "--y-max",
        type=float,
        default=5.0,
        help="Upper limit of elapsed sensor-time axis in seconds (default: 5).",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path("apriltag_elapsed_time_vs_sample_number.png"),
        help="Output PNG path.",
    )

    parser.add_argument(
        "--show",
        action="store_true",
        help="Display the figure interactively in addition to saving it.",
    )

    args = parser.parse_args()

    csv_files = find_stationary_csvs(
        args.trials_dir,
        args.stationary_filename,
    )[: args.max_trials]

    if len(csv_files) < args.max_trials:
        print(
            f"Warning: found {len(csv_files)} trial file(s); "
            f"requested up to {args.max_trials}."
        )

    fig, ax = plt.subplots(figsize=(11.5, 7.0))

    maximum_samples = 1

    for fallback_index, csv_path in enumerate(csv_files, start=1):
        sample_number, elapsed_time_sec = load_elapsed_sensor_time(csv_path)

        trial_number = trial_number_from_path(csv_path)
        if trial_number == 10**9:
            trial_number = fallback_index

        maximum_samples = max(maximum_samples, int(sample_number[-1]))

        line, = ax.plot(
            sample_number,
            elapsed_time_sec,
            linewidth=1.4,
            label=f"Trial {trial_number}",
        )

        # Each black circular marker is one recorded AprilTag sample.
        ax.scatter(
            sample_number,
            elapsed_time_sec,
            s=22,
            c="black",
            edgecolors="black",
            linewidths=0.3,
            zorder=3,
        )

        print(
            f"Trial {trial_number}: "
            f"{len(sample_number)} samples, "
            f"elapsed sensor time = {elapsed_time_sec[-1]:.6f} s, "
            f"file = {csv_path}"
        )

    ax.set_xlim(1, maximum_samples)
    ax.set_ylim(0.0, args.y_max)

    ax.set_xlabel("Stationary-filtered trajectory sample number")
    ax.set_ylabel("Elapsed AprilTag sensor time (s)")
    ax.set_title(
        "AprilTag Elapsed Sensor Time Across Repeated Trials\n"
        r"Command $(V_x,V_y,V_{\mathrm{yaw}})=(0.20,0.15,0.0)$"
    )

    ax.grid(True, linestyle=":", linewidth=0.8)

    fig.text(
        0.5,
        0.015,
        "Black circular markers denote individual recorded AprilTag samples.",
        ha="center",
        fontsize=9,
    )

    ax.legend(
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        borderaxespad=0.0,
    )

    fig.tight_layout()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=300, bbox_inches="tight")

    print(f"\nSaved plot: {args.output.resolve()}")

    if args.show:
        plt.show()
    else:
        plt.close(fig)


if __name__ == "__main__":
    main()
