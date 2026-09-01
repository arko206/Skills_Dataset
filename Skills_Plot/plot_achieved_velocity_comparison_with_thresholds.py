#!/usr/bin/env python3
"""Compare achieved Unitree Go2 body-frame velocities for two commands.

Features
--------
1. Constructs relative time from common_time_ns for each trial.
2. Connects every finite velocity estimate chronologically.
3. Highlights threshold-exceeding samples in green.
4. Prints exceedance counts and percentages for vx, vy, and yaw rate.

By default, threshold tests use absolute magnitude:
    |vx| > 0.08 m/s
    |vy| > 0.08 m/s
    |yaw rate| > 0.10 rad/s

Use --threshold-mode positive to test positive values only.
"""

from __future__ import annotations

import argparse
import math
import re
from pathlib import Path
from typing import Dict, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REQUIRED_COLUMNS = {
    "common_time_ns",
    "vx_body_mps",
    "vy_body_mps",
    "vyaw_body_radps",
}

DEFAULT_LINEAR_THRESHOLD_MPS = 0.08
DEFAULT_YAW_THRESHOLD_RADPS = 0.10

Trial = Dict[str, object]


def parse_command_values_from_path(path: Path) -> Optional[Tuple[float, float, float]]:
    """Extract up to three numeric command values from a file or folder name."""
    candidates = [path.name, path.stem, path.parent.name]
    pattern = re.compile(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)")

    for candidate in candidates:
        values = [float(value) for value in pattern.findall(candidate)]
        if len(values) >= 3:
            return tuple(values[:3])

    return None


def resolve_command_csv(
    input_path: Path,
    vx: Optional[float] = None,
    vy: Optional[float] = None,
    vyaw: Optional[float] = None,
    filename_template: Optional[str] = None,
) -> Path:
    """Resolve a CSV path from a file, a directory, or a custom filename pattern."""
    if input_path.is_file():
        return input_path

    if not input_path.exists():
        raise FileNotFoundError(f"Input path does not exist: {input_path}")

    if not input_path.is_dir():
        raise ValueError(f"Input path must be a file or directory: {input_path}")

    if vx is None or vy is None or vyaw is None:
        raise ValueError(
            "When passing a directory, provide --vx, --vy, and --vyaw values "
            "to locate the matching CSV."
        )

    if filename_template is None:
        filename_template = "Tf_stationary_{Vx}_{Vy}_{Vyaw}.csv"

    template = filename_template.strip()
    template_regex = re.escape(template)
    template_regex = template_regex.replace(r"\{Vx\}", r"([-+]?(?:\d+(?:\.\d*)?|\.\d+))")
    template_regex = template_regex.replace(r"\{Vy\}", r"([-+]?(?:\d+(?:\.\d*)?|\.\d+))")
    template_regex = template_regex.replace(r"\{Vyaw\}", r"([-+]?(?:\d+(?:\.\d*)?|\.\d+))")
    pattern = re.compile(r"^" + template_regex + r"$")

    matches = []
    for csv_path in input_path.rglob("*.csv"):
        match = pattern.match(csv_path.name)
        if not match:
            continue

        actual_vx = float(match.group(1))
        actual_vy = float(match.group(2))
        actual_vyaw = float(match.group(3))

        if (
            math.isclose(actual_vx, vx, rel_tol=1e-9, abs_tol=1e-9)
            and math.isclose(actual_vy, vy, rel_tol=1e-9, abs_tol=1e-9)
            and math.isclose(actual_vyaw, vyaw, rel_tol=1e-9, abs_tol=1e-9)
        ):
            matches.append(csv_path)

    if not matches:
        raise FileNotFoundError(
            f"No CSV matching template {filename_template!r} with values {vx}, {vy}, {vyaw} found under {input_path}"
        )

    if len(matches) > 1:
        raise ValueError(
            f"Multiple matching CSV files found for {vx}, {vy}, {vyaw}: {matches}"
        )

    return matches[0]


def make_command_label(vx: float, vy: float, vyaw: float) -> str:
    """Create a compact label for the command tuple."""
    return (
        rf"Command $(V_x,V_y,V_{{\mathrm{{yaw}}}})=({vx:.6g},{vy:.6g},{vyaw:.6g})$"
    )


def format_component_value(value: float) -> str:
    """Format a shared numeric component without unnecessary trailing zeros."""
    if value == 0.0:
        return "0"
    if float(value).is_integer():
        return str(int(value))
    return format(value, ".15g")


def format_component_value_for_pair(value: float) -> str:
    """Format a numeric component for a differing pair, preserving 0.0."""
    if value == 0.0:
        return "0.0"
    return format_component_value(value)


def make_output_subdir_name(
    command_a_values: Tuple[float, float, float],
    command_b_values: Tuple[float, float, float],
) -> str:
    """Create a compact output-folder name from the two command tuples."""
    parts = []
    for value_a, value_b in zip(command_a_values, command_b_values):
        if math.isclose(value_a, value_b, rel_tol=1e-9, abs_tol=1e-9):
            parts.append(format_component_value(value_a))
        else:
            parts.append(
                f"{format_component_value_for_pair(value_a)}∕{format_component_value_for_pair(value_b)}"
            )

    return f"Comp_plots_{{{parts[0]}}}_{{{parts[1]}}}_{{{parts[2]}}}"


def load_trial(csv_path: Path, label: str) -> Trial:
    """Load one CSV, validate timestamps, and construct relative time."""
    if not csv_path.is_file():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    df = pd.read_csv(csv_path)

    missing = REQUIRED_COLUMNS.difference(df.columns)
    if missing:
        raise ValueError(
            f"{csv_path.name} is missing required columns: {sorted(missing)}"
        )

    for column in REQUIRED_COLUMNS:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    if df["common_time_ns"].isna().any():
        bad_rows = df.index[df["common_time_ns"].isna()].tolist()
        raise ValueError(
            f"{csv_path.name} contains invalid common_time_ns values "
            f"at rows {bad_rows}."
        )

    if len(df) < 2:
        raise ValueError(f"{csv_path.name} must contain at least two rows.")

    common_time_ns = df["common_time_ns"].astype("int64")

    if not common_time_ns.is_monotonic_increasing:
        raise ValueError(
            f"common_time_ns is not monotonically increasing in {csv_path.name}."
        )

    if common_time_ns.duplicated().any():
        duplicate_rows = df.index[common_time_ns.duplicated()].tolist()
        raise ValueError(
            f"Duplicate common_time_ns values in {csv_path.name} "
            f"at rows {duplicate_rows}."
        )
    ##-- estimating the relative time difference --------------------###
    ##-- Converting the start time to 0 -------------------###
    df["relative_time_sec"] = (
        common_time_ns - common_time_ns.iloc[0]
    ).astype("float64") / 1.0e9

    return {
        "path": csv_path,
        "label": label,
        "df": df,
        "duration_sec": float(df["relative_time_sec"].iloc[-1]),
    }


def shared_y_limits(
    trials: Tuple[Trial, Trial],
    column: str,
    threshold: float,
    padding_fraction: float = 0.10,
) -> Tuple[float, float]:
    """Return common y-limits that include the plotted data and key reference lines."""
    # Gather all finite values from both trials so the same y-axis can be used.
    arrays = []

    for trial in trials:
        df = trial["df"]
        # Keep only rows where the selected column has a real numeric value.
        values = df.loc[df[column].notna(), column].to_numpy(dtype=float)
        if values.size:
            arrays.append(values)

    # If nothing is usable, the plot cannot define sensible limits.
    if not arrays:
        raise ValueError(f"No valid values available for {column}.")

    # Combine the data from both trials into one array for a shared range.
    values = np.concatenate(arrays)

    # Ensure the limits include the data range, the zero line, and the threshold line.
    lower = min(float(np.min(values)), -threshold, 0.0)
    upper = max(float(np.max(values)), threshold, 0.0)

    # Compute the overall span of the visible range.
    span = upper - lower

    # Guard against a degenerate range where all values are identical.
    if span == 0.0:
        span = max(abs(lower), 1.0)

    # Add a small padding margin so the data does not touch the plot borders.
    padding = padding_fraction * span
    return lower - padding, upper + padding


def threshold_exceedance_mask(
    df: pd.DataFrame,
    column: str,
    threshold: float,
    threshold_mode: str,
) -> pd.Series:
    """Identify samples that exceed the selected threshold rule."""
    finite = df[column].notna()

    if threshold_mode == "magnitude":
        return finite & (df[column].abs() > threshold)

    if threshold_mode == "positive":
        return finite & (df[column] > threshold)

    raise ValueError(
        f"Unsupported threshold mode {threshold_mode!r}; "
        "expected 'magnitude' or 'positive'."
    )


def plot_velocity_component(
    trials: Tuple[Trial, Trial],
    column: str,
    ylabel: str,
    title: str,
    output_path: Path,
    x_max_sec: float,
    threshold: float,
    threshold_unit: str,
    threshold_mode: str,
    show: bool,
) -> None:
    """Plot one velocity component and highlight threshold exceedances."""
    fig, ax = plt.subplots(figsize=(12.0, 5.8))
    count_lines = []

    for trial_index, trial in enumerate(trials):
        df = trial["df"]
        ##-- used to know whether the sample is valid or not, if it is valid then we can plot it. --------------------###
        valid = df[column].notna()

        # Connect every available finite sample in time order.
        ax.plot(
            df.loc[valid, "relative_time_sec"],
            df.loc[valid, column],
            marker="o",
            markersize=4,
            linewidth=1.6,
            label=str(trial["label"]),
            zorder=2,
        )

        # Determine which samples cross the selected threshold rule.
        # This returns a boolean mask with True for rows that should be highlighted.
        exceeds = threshold_exceedance_mask(
            df=df,
            column=column,
            threshold=threshold,
            threshold_mode=threshold_mode,
        )

        # The user requested green markers for threshold-exceeding samples.
        ax.scatter(
            df.loc[exceeds, "relative_time_sec"],
            df.loc[exceeds, column],
            s=62,
            color="green",
            edgecolors="black",
            linewidths=0.6,
            zorder=4,
            label="Threshold exceeded" if trial_index == 0 else "_nolegend_",
        )

        valid_count = int(valid.sum())
        exceed_count = int(exceeds.sum())
        percentage = (
            100.0 * exceed_count / valid_count
            if valid_count > 0
            else float("nan")
        )
        count_lines.append(
            f"{trial['label']}: {exceed_count}/{valid_count} "
            f"({percentage:.1f}%)"
        )

    ax.axhline(0.0, linewidth=1.0, linestyle="--")

    if threshold_mode == "magnitude":
        ax.axhline(
            threshold,
            linewidth=1.0,
            linestyle=":",
            color="green",
            alpha=0.75,
        )
        ax.axhline(
            -threshold,
            linewidth=1.0,
            linestyle=":",
            color="green",
            alpha=0.75,
        )
        threshold_text = rf"$|v|>{threshold:.2f}$ {threshold_unit}"
    else:
        ax.axhline(
            threshold,
            linewidth=1.0,
            linestyle=":",
            color="green",
            alpha=0.75,
        )
        threshold_text = rf"$v>{threshold:.2f}$ {threshold_unit}"

    y_min, y_max = shared_y_limits(trials, column, threshold)
    ax.set_xlim(0.0, x_max_sec)
    ax.set_ylim(y_min, y_max)
    ax.set_xlabel("Relative AprilTag time, $t$ (s)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, linestyle=":", linewidth=0.8)
    ax.legend(
        loc="upper left",
        bbox_to_anchor=(1.01, 1.0),
        borderaxespad=0.0,
        fontsize=9,
    )

    ax.text(
        1.01,
        0.61,
        "Threshold: " + threshold_text + "\n" + "\n".join(count_lines),
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=9,
        bbox={
            "boxstyle": "round,pad=0.35",
            "facecolor": "white",
            "edgecolor": "0.6",
            "alpha": 0.90,
        },
    )

    fig.text(
        0.5,
        0.012,
        (
            "Green markers denote threshold exceedances. All finite estimates "
            "are connected in time order."
        ),
        ha="center",
        fontsize=9,
    )

    fig.subplots_adjust(
        left=0.09,
        right=0.72,
        bottom=0.17,
        top=0.90,
    )
    fig.savefig(output_path, dpi=300, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)


def print_trial_report(
    trial: Trial,
    linear_threshold_mps: float,
    yaw_threshold_radps: float,
    threshold_mode: str,
) -> None:
    """Print duration, ranges, and threshold-exceedance counts."""
    df = trial["df"]

    print(f"\n{trial['label']}")
    print(f"  File: {trial['path']}")
    print(f"  Rows: {len(df)}")
    print(f"  Relative duration: {trial['duration_sec']:.6f} s")

    specs = {
        "vx_body_mps": (linear_threshold_mps, "m/s"),
        "vy_body_mps": (linear_threshold_mps, "m/s"),
        "vyaw_body_radps": (yaw_threshold_radps, "rad/s"),
    }

    for column, (threshold, unit) in specs.items():
        finite = df[column].dropna()
        exceeds = threshold_exceedance_mask(
            df, column, threshold, threshold_mode
        )

        count = int(exceeds.sum())
        total = int(finite.shape[0])
        percentage = (
            100.0 * count / total if total > 0 else float("nan")
        )

        rule = "|value| >" if threshold_mode == "magnitude" else "value >"

        print(
            f"  {column}: min={finite.min(): .6f}, "
            f"max={finite.max(): .6f}"
        )
        print(
            f"    {rule} {threshold:.2f} {unit}: "
            f"{count}/{total} ({percentage:.1f}%) "
        )



def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare achieved vx, vy, and yaw rate for two command trajectories."
        )
    )
    parser.add_argument(
        "--command-a",
        "--vy0",
        dest="command_a",
        type=Path,
        required=True,
        help=(
            "CSV file or directory for the first command trajectory. "
            "If a directory is passed, also provide --vx-a, --vy-a, and --vyaw-a."
        ),
    )
    parser.add_argument(
        "--command-b",
        "--vy04",
        dest="command_b",
        type=Path,
        required=True,
        help=(
            "CSV file or directory for the second command trajectory. "
            "If a directory is passed, also provide --vx-b, --vy-b, and --vyaw-b."
        ),
    )
    parser.add_argument("--vx-a", type=float, default=None, help="Vx value for the first command.")
    parser.add_argument("--vy-a", type=float, default=None, help="Vy value for the first command.")
    parser.add_argument("--vyaw-a", type=float, default=None, help="Vyaw value for the first command.")
    parser.add_argument("--vx-b", type=float, default=None, help="Vx value for the second command.")
    parser.add_argument("--vy-b", type=float, default=None, help="Vy value for the second command.")
    parser.add_argument("--vyaw-b", type=float, default=None, help="Vyaw value for the second command.")
    parser.add_argument(
        "--filename-template-a",
        default="Tf_stationary_{Vx}_{Vy}_{Vyaw}.csv",
        help="Filename template for the first command CSV, using {Vx}, {Vy}, and {Vyaw} placeholders.",
    )
    parser.add_argument(
        "--filename-template-b",
        default="Tf_stationary_{Vx}_{Vy}_{Vyaw}.csv",
        help="Filename template for the second command CSV, using {Vx}, {Vy}, and {Vyaw} placeholders.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("Velocity_Comparison_Plots"),
        help=(
            "Base output directory for PNG figures; a subfolder will be created "
            "from the two command tuples."
        ),
    )
    parser.add_argument(
        "--linear-threshold",
        type=float,
        default=DEFAULT_LINEAR_THRESHOLD_MPS,
        help=(
            "Threshold for vx and vy "
            f"(default: {DEFAULT_LINEAR_THRESHOLD_MPS:.2f} m/s)."
        ),
    )
    parser.add_argument(
        "--yaw-threshold",
        type=float,
        default=DEFAULT_YAW_THRESHOLD_RADPS,
        help=(
            "Threshold for yaw rate "
            f"(default: {DEFAULT_YAW_THRESHOLD_RADPS:.2f} rad/s)."
        ),
    )
    parser.add_argument(
        "--threshold-mode",
        choices=("magnitude", "positive"),
        default="magnitude",
        help=(
            "Threshold absolute magnitude or positive values only "
            "(default: magnitude)."
        ),
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display figures interactively as well as saving them.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()

    if args.linear_threshold <= 0.0:
        raise ValueError("--linear-threshold must be positive.")
    if args.yaw_threshold <= 0.0:
        raise ValueError("--yaw-threshold must be positive.")

    output_root_dir = args.output_dir
    output_root_dir.mkdir(parents=True, exist_ok=True)

    command_a_path = resolve_command_csv(
        args.command_a,
        vx=args.vx_a,
        vy=args.vy_a,
        vyaw=args.vyaw_a,
        filename_template=args.filename_template_a,
    )
    command_b_path = resolve_command_csv(
        args.command_b,
        vx=args.vx_b,
        vy=args.vy_b,
        vyaw=args.vyaw_b,
        filename_template=args.filename_template_b,
    )

    command_a_values = parse_command_values_from_path(command_a_path)
    command_b_values = parse_command_values_from_path(command_b_path)

    if command_a_values is None:
        raise ValueError(f"Could not infer command values from {command_a_path}")
    if command_b_values is None:
        raise ValueError(f"Could not infer command values from {command_b_path}")

    output_subdir = output_root_dir / make_output_subdir_name(
        command_a_values,
        command_b_values,
    )
    output_subdir.mkdir(parents=True, exist_ok=True)

    trial_reference = load_trial(
        command_a_path,
        label=make_command_label(*command_a_values),
    )
    trial_comparison = load_trial(
        command_b_path,
        label=make_command_label(*command_b_values),
    )
    trials = (trial_reference, trial_comparison)

    print_trial_report(
        trial_reference,
        args.linear_threshold,
        args.yaw_threshold,
        args.threshold_mode,
    )
    print_trial_report(
        trial_comparison,
        args.linear_threshold,
        args.yaw_threshold,
        args.threshold_mode,
    )

    maximum_duration = max(float(t["duration_sec"]) for t in trials)
    x_max_sec = math.ceil(maximum_duration * 10.0) / 10.0
    print(f"\nShared x-axis: 0.0 to {x_max_sec:.1f} s")

    plot_specs = (
        (
            "vx_body_mps",
            r"Achieved longitudinal velocity, $v_x^{\mathrm{body}}$ (m/s)",
            "Achieved Body-Frame Longitudinal Velocity",
            "achieved_vx_vs_time_thresholds.png",
            args.linear_threshold,
            "m/s",
        ),
        (
            "vy_body_mps",
            r"Achieved lateral velocity, $v_y^{\mathrm{body}}$ (m/s)",
            "Achieved Body-Frame Lateral Velocity",
            "achieved_vy_vs_time_thresholds.png",
            args.linear_threshold,
            "m/s",
        ),
        (
            "vyaw_body_radps",
            r"Achieved yaw rate, $\omega_{\mathrm{yaw}}$ (rad/s)",
            "Achieved Body-Frame Yaw Rate",
            "achieved_yaw_rate_vs_time_thresholds.png",
            args.yaw_threshold,
            "rad/s",
        ),
    )

    for (
        column,
        ylabel,
        title,
        filename,
        threshold,
        threshold_unit,
    ) in plot_specs:
        output_path = output_subdir / filename

        plot_velocity_component(
            trials=trials,
            column=column,
            ylabel=ylabel,
            title=title,
            output_path=output_path,
            x_max_sec=x_max_sec,
            threshold=threshold,
            threshold_unit=threshold_unit,
            threshold_mode=args.threshold_mode,
            show=args.show,
        )
        print(f"Saved: {output_path.resolve()}")


if __name__ == "__main__":
    main()
