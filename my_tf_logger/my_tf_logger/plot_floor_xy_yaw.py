#!/usr/bin/env python3
"""
Plot a Unitree Go2 floor-relative SE(2) trajectory from a filtered CSV.

The plot uses:
    x_floor_m      -> floor-frame x position
    y_floor_m      -> floor-frame y position
    yaw_floor_rad  -> robot heading

For every retained pose, the script draws:
    1. the connected x-y trajectory,
    2. the rectangular robot footprint rotated by yaw,
    3. a waypoint dot,
    4. selected heading arrows using the same movement-based rule as the
       supplied SE(2) planner plotting code.

No obstacles, goal features, or RRT tree are plotted.

Example:
    python3 plot_floor_xy_yaw_modified.py \
        Fourth_floor_to_base_motion_filtered.csv

Using ros2 run:
    ros2 run my_tf_logger plot_floor_xy_yaw \
        ~/Go2_Walk_Base_Data_Sensor/Fourth_floor_to_base_motion_filtered.csv
"""

import argparse
from pathlib import Path

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REQUIRED_COLUMNS = {
    "x_floor_m",
    "y_floor_m",
    "yaw_floor_rad",
}


def load_trajectory(csv_file: Path) -> pd.DataFrame:
    """Load and validate the filtered floor-to-base trajectory."""
    if not csv_file.is_file():
        raise FileNotFoundError(
            f"CSV file does not exist: {csv_file}"
        )

    dataframe = pd.read_csv(csv_file)

    missing_columns = REQUIRED_COLUMNS - set(dataframe.columns)
    if missing_columns:
        raise ValueError(
            "CSV is missing required columns: "
            + ", ".join(sorted(missing_columns))
        )

    numeric_columns = list(REQUIRED_COLUMNS)

    if "wall_time_ns" in dataframe.columns:
        numeric_columns.append("wall_time_ns")

    for column in numeric_columns:
        dataframe[column] = pd.to_numeric(
            dataframe[column],
            errors="coerce",
        )

    dataframe = dataframe.dropna(
        subset=sorted(REQUIRED_COLUMNS)
    ).copy()

    if dataframe.empty:
        raise ValueError(
            "No valid x, y, and yaw rows were found in the CSV."
        )

    if "wall_time_ns" in dataframe.columns:
        dataframe = dataframe.sort_values(
            "wall_time_ns",
            kind="stable",
        )

    return dataframe.reset_index(drop=True)


def rotated_robot_corners(
    x: float,
    y: float,
    yaw: float,
    robot_length: float,
    robot_width: float,
) -> np.ndarray:
    """Return the four floor-frame corners of the rotated robot rectangle."""
    half_length = robot_length / 2.0
    half_width = robot_width / 2.0

    corners_robot = np.array(
        [
            [-half_length, -half_width],
            [ half_length, -half_width],
            [ half_length,  half_width],
            [-half_length,  half_width],
        ],
        dtype=float,
    )

    cosine = np.cos(yaw)
    sine = np.sin(yaw)

    rotation = np.array(
        [
            [cosine, -sine],
            [sine, cosine],
        ],
        dtype=float,
    )

    corners_floor = corners_robot @ rotation.T
    corners_floor[:, 0] += x
    corners_floor[:, 1] += y

    return corners_floor


def should_show_heading_arrow(
    index: int,
    x: np.ndarray,
    y: np.ndarray,
    minimum_motion_distance: float,
) -> bool:
    """
    Apply the same arrow-selection rule used in the supplied planner plot.

    The first and last poses always receive arrows. For an interior pose,
    show an arrow when either its next or previous translational separation
    exceeds minimum_motion_distance. This suppresses excessive arrows during
    repeated rotation-in-place samples at the same x-y location.
    """
    sample_count = len(x)

    if index == 0 or index == sample_count - 1:
        return True

    distance_to_next = np.hypot(
        x[index] - x[index + 1],
        y[index] - y[index + 1],
    )

    if distance_to_next > minimum_motion_distance:
        return True

    distance_to_previous = np.hypot(
        x[index] - x[index - 1],
        y[index] - y[index - 1],
    )

    return distance_to_previous > minimum_motion_distance


def plot_xy_yaw(
    dataframe: pd.DataFrame,
    output_file: Path,
    robot_length: float,
    robot_width: float,
    arrow_length: float,
    arrow_head_width: float,
    arrow_head_length: float,
    minimum_motion_distance: float,
    rectangle_stride: int,
    relative: bool,
    show_plot: bool,
) -> None:
    """Plot the measured floor-frame trajectory, robot rectangles, and yaw."""
    x = dataframe["x_floor_m"].to_numpy(dtype=float)
    y = dataframe["y_floor_m"].to_numpy(dtype=float)
    yaw = dataframe["yaw_floor_rad"].to_numpy(dtype=float)

    if relative:
        x = x - x[0]
        y = y - y[0]

    figure, axis = plt.subplots(figsize=(8, 8))

    # Connected measured trajectory, matching the planner-path presentation.
    axis.plot(
        x,
        y,
        "b-",
        linewidth=2.5,
        label="Measured base trajectory",
        zorder=4,
    )

    heading_arrow_count = 0
    rectangle_count = 0

    for index in range(len(dataframe)):
        # Draw the robot footprint at every selected stride.
        if (
            index % rectangle_stride == 0
            or index == len(dataframe) - 1
        ):
            rectangle_points = rotated_robot_corners(
                x=x[index],
                y=y[index],
                yaw=yaw[index],
                robot_length=robot_length,
                robot_width=robot_width,
            )

            robot_polygon = patches.Polygon(
                rectangle_points,
                closed=True,
                color="blue",
                alpha=0.15,
                zorder=3,
                label="Robot footprint" if rectangle_count == 0 else "",
            )
            axis.add_patch(robot_polygon)
            rectangle_count += 1

        # Centre point of each measured pose.
        axis.plot(
            x[index],
            y[index],
            "k.",
            markersize=4,
            zorder=5,
        )

        # Copy the planner plot's conditional heading-arrow logic.
        if should_show_heading_arrow(
            index=index,
            x=x,
            y=y,
            minimum_motion_distance=minimum_motion_distance,
        ):
            delta_x = arrow_length * np.cos(yaw[index])
            delta_y = arrow_length * np.sin(yaw[index])

            axis.arrow(
                x[index],
                y[index],
                delta_x,
                delta_y,
                head_width=arrow_head_width,
                head_length=arrow_head_length,
                fc="black",
                ec="black",
                length_includes_head=True,
                zorder=6,
                label="Yaw heading" if heading_arrow_count == 0 else "",
            )
            heading_arrow_count += 1

    axis.plot(
        x[0],
        y[0],
        "go",
        markersize=8,
        label="Start pose",
        zorder=7,
    )
    axis.plot(
        x[-1],
        y[-1],
        "rx",
        markersize=9,
        markeredgewidth=2,
        label="End pose",
        zorder=7,
    )

    coordinate_description = (
        "relative to first pose"
        if relative
        else "with respect to floor frame"
    )

    axis.set_title(
        f"Unitree Go2 SE(2) Base Trajectory ({coordinate_description})"
    )
    axis.set_xlabel("Floor-frame x position (m)")
    axis.set_ylabel("Floor-frame y position (m)")
    axis.set_aspect("equal", adjustable="datalim")
    axis.grid(True, alpha=0.3)
    axis.legend()
    axis.margins(0.15)

    figure.tight_layout()

    output_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    figure.savefig(
        output_file,
        dpi=300,
        bbox_inches="tight",
    )

    print(f"Valid trajectory samples: {len(dataframe)}")
    print(f"Robot rectangles plotted: {rectangle_count}")
    print(f"Heading arrows plotted: {heading_arrow_count}")
    print(f"Saved plot: {output_file}")

    if show_plot:
        plt.show()
    else:
        plt.close(figure)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot a filtered floor-to-base SE(2) trajectory using "
            "rotated robot rectangles and yaw arrows."
        )
    )

    parser.add_argument(
        "csv_file",
        type=Path,
        help="Path to the filtered floor-to-base CSV file.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Output image path. By default, save beside the CSV with "
            "the suffix '_xy_yaw_robot_frames.png'."
        ),
    )
    parser.add_argument(
        "--robot-length",
        type=float,
        default=0.40,
        help="Displayed rectangular robot length in metres (default: 0.40).",
    )
    parser.add_argument(
        "--robot-width",
        type=float,
        default=0.35,
        help="Displayed rectangular robot width in metres (default: 0.35).",
    )
    parser.add_argument(
        "--arrow-length",
        type=float,
        default=0.08,
        help="Displayed yaw-arrow shaft length in metres (default: 0.08).",
    )
    parser.add_argument(
        "--arrow-head-width",
        type=float,
        default=0.04,
        help="Displayed yaw-arrow head width in metres (default: 0.04).",
    )
    parser.add_argument(
        "--arrow-head-length",
        type=float,
        default=0.04,
        help="Displayed yaw-arrow head length in metres (default: 0.04).",
    )
    parser.add_argument(
        "--minimum-motion-distance",
        type=float,
        default=0.01,
        help=(
            "Minimum adjacent x-y displacement for an interior heading "
            "arrow (default: 0.01 m)."
        ),
    )
    parser.add_argument(
        "--rectangle-stride",
        type=int,
        default=1,
        help=(
            "Draw one robot rectangle every N samples. Use 1 for every "
            "sample (default: 1)."
        ),
    )
    parser.add_argument(
        "--relative",
        action="store_true",
        help="Translate the first pose to x=0, y=0 before plotting.",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Save the plot without opening a Matplotlib window.",
    )

    return parser.parse_args()


def validate_arguments(arguments: argparse.Namespace) -> None:
    positive_values = {
        "--robot-length": arguments.robot_length,
        "--robot-width": arguments.robot_width,
        "--arrow-length": arguments.arrow_length,
        "--arrow-head-width": arguments.arrow_head_width,
        "--arrow-head-length": arguments.arrow_head_length,
    }

    for argument_name, value in positive_values.items():
        if value <= 0.0:
            raise ValueError(
                f"{argument_name} must be greater than zero."
            )

    if arguments.minimum_motion_distance < 0.0:
        raise ValueError(
            "--minimum-motion-distance cannot be negative."
        )

    if arguments.rectangle_stride <= 0:
        raise ValueError(
            "--rectangle-stride must be greater than zero."
        )


def main() -> None:
    arguments = parse_arguments()
    validate_arguments(arguments)

    output_file = arguments.output
    if output_file is None:
        output_file = arguments.csv_file.with_name(
            f"{arguments.csv_file.stem}_xy_yaw_robot_frames.png"
        )

    trajectory = load_trajectory(arguments.csv_file)

    plot_xy_yaw(
        dataframe=trajectory,
        output_file=output_file,
        robot_length=arguments.robot_length,
        robot_width=arguments.robot_width,
        arrow_length=arguments.arrow_length,
        arrow_head_width=arguments.arrow_head_width,
        arrow_head_length=arguments.arrow_head_length,
        minimum_motion_distance=arguments.minimum_motion_distance,
        rectangle_stride=arguments.rectangle_stride,
        relative=arguments.relative,
        show_plot=not arguments.no_show,
    )


if __name__ == "__main__":
    main()
