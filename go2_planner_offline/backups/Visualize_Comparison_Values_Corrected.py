#!/usr/bin/env python3
"""
Visualize successful fixed-seed comparison paths.

Expected directory layout:

~/Navigation_Dataset/
└── Comparison_Values/
    └── Comparison_<comparison_idx>/
        ├── query_<comparison_idx>.txt
        ├── RRT_<comparison_idx>/
        │   ├── rrt_run_<run_idx>/
        │   │   ├── se2_waypoints_<run_idx>_rrt.txt
        │   │   ├── local_obs_<run_idx>_rrt.txt
        │   │   └── local_goal_<run_idx>_rrt.txt
        │   └── ...
        ├── Diffusion_RRT_<comparison_idx>/
        │   ├── diff_rrt_<run_idx>/
        │   │   ├── se2_waypoints_<run_idx>_diff_rrt.txt
        │   │   ├── local_obs_<run_idx>_diff_rrt.txt
        │   │   └── local_goal_<run_idx>_diff_rrt.txt
        │   └── ...
        └── Lateral_Diff_RRT_<comparison_idx>/
            ├── lateral_diff_rrt_<run_idx>/
            │   ├── se2_waypoints_<run_idx>_lateral_diff_rrt.txt
            │   ├── local_obs_<run_idx>_lateral_diff_rrt.txt
            │   └── local_goal_<run_idx>_lateral_diff_rrt.txt
            └── ...

A plot is generated only when a non-empty waypoint file exists. The PNG is
saved inside the same run directory as the waypoint file.
"""

from __future__ import annotations

import argparse
import ast
import re
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot all successful classical-RRT, diffusion-RRT, and "
            "lateral-diffusion-RRT paths from one "
            "Comparison_Values/Comparison_<idx> folder."
        )
    )
    parser.add_argument(
        "--comparison_idx",
        type=int,
        required=True,
        help="Comparison index, for example 1, 2, or 3.",
    )
    parser.add_argument(
        "--dataset_root",
        type=Path,
        default=(
            Path.home()
            / "Single_Step_Position_Change"
            / "Navigation_Dataset"
        ),
        help="Navigation_Dataset root directory.",
    )
    parser.add_argument(
        "--planner_cfg",
        type=Path,
        default=Path.home() / "go2_planner_offline" / "planner.cfg",
        help=(
            "Optional planner.cfg used only to supplement plotting settings "
            "such as robot dimensions and map bounds. The archived "
            "query_<idx>.txt always overrides it."
        ),
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=160,
        help="Output PNG resolution.",
    )
    return parser.parse_args()


def load_numeric_config(filepath: Path) -> Dict[str, float]:
    """Load numeric key=value entries while ignoring comments and strings."""
    config: Dict[str, float] = {}

    if not filepath.is_file():
        return config

    with filepath.open("r", encoding="utf-8") as stream:
        for raw_line in stream:
            line = raw_line.split("#", 1)[0].strip()
            if not line or "=" not in line:
                continue

            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().rstrip(";")

            try:
                config[key] = float(value)
            except ValueError:
                # String-valued settings are irrelevant for plotting.
                continue

    return config


def resolve_query_file(comparison_dir: Path, comparison_idx: int) -> Path:
    expected = comparison_dir / f"query_{comparison_idx}.cfg"
    if expected.is_file():
        return expected

    fallback = comparison_dir / "query.cfg"
    if fallback.is_file():
        return fallback

    candidates = sorted(comparison_dir.glob("query*.cfg"))
    if len(candidates) == 1:
        return candidates[0]

    if not candidates:
        raise FileNotFoundError(
            f"No archived query file was found in {comparison_dir}. "
            f"Expected {expected.name}."
        )

    raise RuntimeError(
        f"Multiple query files were found in {comparison_dir}: "
        + ", ".join(path.name for path in candidates)
    )


def load_waypoints(filepath: Path) -> np.ndarray:
    if not filepath.is_file() or filepath.stat().st_size == 0:
        return np.empty((0, 3), dtype=float)

    try:
        data = np.loadtxt(filepath, delimiter=",", dtype=float)
    except ValueError:
        data = np.loadtxt(filepath, dtype=float)

    data = np.atleast_2d(data)

    if data.shape[1] < 3:
        raise ValueError(
            f"{filepath} must contain at least x, y, theta columns."
        )

    return data[:, :3]


def load_local_obstacle_info(filepath: Path) -> List[List[List[float]]]:
    """
    Each non-empty row is expected to look like:
    [[rel_x, rel_y, distance], [rel_x, rel_y, distance], ...]
    """
    rows: List[List[List[float]]] = []

    if not filepath.is_file() or filepath.stat().st_size == 0:
        return rows

    with filepath.open("r", encoding="utf-8") as stream:
        for line_number, raw_line in enumerate(stream, start=1):
            line = raw_line.strip()
            if not line:
                continue

            try:
                parsed = ast.literal_eval(line)
            except (ValueError, SyntaxError) as error:
                raise ValueError(
                    f"Could not parse {filepath}:{line_number}: {line}"
                ) from error

            if not isinstance(parsed, list):
                raise ValueError(
                    f"Expected a list in {filepath}:{line_number}."
                )

            rows.append(parsed)

    return rows


def load_local_goal_info(filepath: Path) -> np.ndarray:
    """
    Each row is expected to contain:
    goal_rel_x, goal_rel_y, goal_rel_distance
    """
    if not filepath.is_file() or filepath.stat().st_size == 0:
        return np.empty((0, 3), dtype=float)

    try:
        data = np.loadtxt(filepath, delimiter=",", dtype=float)
    except ValueError:
        data = np.loadtxt(filepath, dtype=float)

    data = np.atleast_2d(data)

    if data.shape[1] < 3:
        raise ValueError(
            f"{filepath} must contain rel_x, rel_y, rel_distance columns."
        )

    return data[:, :3]


def robot_frame_to_world(
    robot_pose: Sequence[float],
    rel_x: float,
    rel_y: float,
) -> Tuple[float, float]:
    x, y, theta = robot_pose[:3]
    c = np.cos(theta)
    s = np.sin(theta)

    world_x = x + rel_x * c - rel_y * s
    world_y = y + rel_x * s + rel_y * c
    return float(world_x), float(world_y)


def rounded_rectangle_points(
    length: float,
    width: float,
    radius: float,
    num_points: int = 12,
) -> np.ndarray:
    half_length = length / 2.0
    half_width = width / 2.0
    points: List[List[float]] = []

    corners = [
        (half_length, half_width, 0.0, np.pi / 2.0),
        (-half_length, half_width, np.pi / 2.0, np.pi),
        (-half_length, -half_width, np.pi, 3.0 * np.pi / 2.0),
        (half_length, -half_width, 3.0 * np.pi / 2.0, 2.0 * np.pi),
    ]

    for corner_x, corner_y, start_angle, end_angle in corners:
        for angle in np.linspace(start_angle, end_angle, num_points):
            points.append(
                [
                    corner_x + radius * np.cos(angle),
                    corner_y + radius * np.sin(angle),
                ]
            )

    return np.asarray(points, dtype=float)


def plot_environment(ax: plt.Axes, cfg: Dict[str, float]) -> None:
    ax.set_xlim(cfg.get("x_min", -2.0), cfg.get("x_max", 2.0))
    ax.set_ylim(cfg.get("y_min", -2.0), cfg.get("y_max", 3.0))

    safety_margin = cfg.get("safety_margin", 0.05)
    obstacle_idx = 0

    while f"obs.{obstacle_idx}.x" in cfg:
        x = cfg[f"obs.{obstacle_idx}.x"]
        y = cfg[f"obs.{obstacle_idx}.y"]
        theta = cfg.get(f"obs.{obstacle_idx}.theta", 0.0)
        length = cfg[f"obs.{obstacle_idx}.length"]
        width = cfg[f"obs.{obstacle_idx}.width"]

        c = np.cos(theta)
        s = np.sin(theta)
        rotation = np.asarray(((c, -s), (s, c)), dtype=float)

        inflated = rounded_rectangle_points(
            length,
            width,
            safety_margin,
        ).dot(rotation.T)
        inflated[:, 0] += x
        inflated[:, 1] += y

        ax.add_patch(
            patches.Polygon(
                inflated,
                closed=True,
                color="red",
                alpha=0.10,
                label="Inflated obstacle" if obstacle_idx == 0 else "",
                zorder=1,
            )
        )

        corners = np.asarray(
            [
                [-length / 2.0, -width / 2.0],
                [length / 2.0, -width / 2.0],
                [length / 2.0, width / 2.0],
                [-length / 2.0, width / 2.0],
            ],
            dtype=float,
        ).dot(rotation.T)

        corners[:, 0] += x
        corners[:, 1] += y

        ax.add_patch(
            patches.Polygon(
                corners,
                closed=True,
                color="red",
                alpha=0.50,
                label="Obstacle" if obstacle_idx == 0 else "",
                zorder=2,
            )
        )

        obstacle_idx += 1

    start_x = cfg.get("start_x", 0.0)
    start_y = cfg.get("start_y", 0.0)
    goal_x = cfg.get("goal_x", 0.0)
    goal_y = cfg.get("goal_y", 0.0)
    goal_tol = cfg.get("goal_tol", 0.1)

    ax.plot(
        start_x,
        start_y,
        marker="o",
        linestyle="None",
        markersize=8,
        label="Start",
        zorder=10,
    )

    ax.add_patch(
        patches.Circle(
            (goal_x, goal_y),
            goal_tol,
            color="green",
            alpha=0.30,
            label="Goal region",
            zorder=2,
        )
    )
    ax.plot(
        goal_x,
        goal_y,
        marker="*",
        linestyle="None",
        markersize=10,
        color="green",
        zorder=10,
    )


def plot_robot_path(
    ax: plt.Axes,
    waypoints: np.ndarray,
    cfg: Dict[str, float],
    method_label: str,
) -> None:
    ax.plot(
        waypoints[:, 0],
        waypoints[:, 1],
        linewidth=2.5,
        label=f"{method_label} path",
        zorder=5,
    )

    robot_length = cfg.get("robot_length", 0.4)
    robot_width = cfg.get("robot_width", 0.35)

    for waypoint_idx, waypoint in enumerate(waypoints):
        x, y, theta = waypoint
        c = np.cos(theta)
        s = np.sin(theta)
        rotation = np.asarray(((c, -s), (s, c)), dtype=float)

        rectangle = np.asarray(
            [
                [-robot_length / 2.0, -robot_width / 2.0],
                [robot_length / 2.0, -robot_width / 2.0],
                [robot_length / 2.0, robot_width / 2.0],
                [-robot_length / 2.0, robot_width / 2.0],
            ],
            dtype=float,
        ).dot(rotation.T)

        rectangle[:, 0] += x
        rectangle[:, 1] += y

        ax.add_patch(
            patches.Polygon(
                rectangle,
                closed=True,
                alpha=0.15,
                zorder=4,
                label="Robot footprint" if waypoint_idx == 0 else "",
            )
        )

        ax.plot(x, y, "k.", markersize=4, zorder=6)

        is_first = waypoint_idx == 0
        is_last = waypoint_idx == len(waypoints) - 1
        show_arrow = is_first or is_last

        if not show_arrow and waypoint_idx + 1 < len(waypoints):
            next_waypoint = waypoints[waypoint_idx + 1]
            distance_next = np.hypot(
                waypoint[0] - next_waypoint[0],
                waypoint[1] - next_waypoint[1],
            )

            if distance_next > 0.01:
                show_arrow = True
            elif waypoint_idx > 0:
                previous_waypoint = waypoints[waypoint_idx - 1]
                distance_previous = np.hypot(
                    waypoint[0] - previous_waypoint[0],
                    waypoint[1] - previous_waypoint[1],
                )
                show_arrow = distance_previous > 0.01

        if show_arrow:
            ax.arrow(
                x,
                y,
                0.08 * np.cos(theta),
                0.08 * np.sin(theta),
                head_width=0.04,
                head_length=0.04,
                fc="black",
                ec="black",
                zorder=7,
            )


def plot_local_obstacle_information(
    ax: plt.Axes,
    waypoints: np.ndarray,
    obstacle_rows: List[List[List[float]]],
) -> None:
    number_of_rows = min(len(obstacle_rows), max(len(waypoints) - 1, 0))

    for row_idx in range(number_of_rows):
        robot_pose = waypoints[row_idx]
        robot_x, robot_y = robot_pose[:2]

        for obstacle_idx, feature in enumerate(obstacle_rows[row_idx]):
            if not isinstance(feature, (list, tuple)) or len(feature) < 3:
                continue

            rel_x = float(feature[0])
            rel_y = float(feature[1])
            rel_distance = float(feature[2])

            obstacle_world_x, obstacle_world_y = robot_frame_to_world(
                robot_pose,
                rel_x,
                rel_y,
            )

            ax.plot(
                [robot_x, obstacle_world_x],
                [robot_y, obstacle_world_y],
                linestyle="--",
                linewidth=1.0,
                alpha=0.65,
                color="orange",
                zorder=8,
                label=(
                    "Local obstacle vector"
                    if row_idx == 0 and obstacle_idx == 0
                    else ""
                ),
            )
            ax.plot(
                obstacle_world_x,
                obstacle_world_y,
                marker="o",
                linestyle="None",
                markersize=3,
                color="orange",
                zorder=9,
            )
            ax.text(
                (robot_x + obstacle_world_x) / 2.0,
                (robot_y + obstacle_world_y) / 2.0,
                f"{rel_distance:.2f}",
                fontsize=6,
                color="orange",
                zorder=9,
            )


def plot_local_goal_information(
    ax: plt.Axes,
    waypoints: np.ndarray,
    goal_rows: np.ndarray,
) -> None:
    number_of_rows = min(goal_rows.shape[0], max(len(waypoints) - 1, 0))

    for row_idx in range(number_of_rows):
        robot_pose = waypoints[row_idx]
        robot_x, robot_y = robot_pose[:2]

        rel_x = float(goal_rows[row_idx, 0])
        rel_y = float(goal_rows[row_idx, 1])
        rel_distance = float(goal_rows[row_idx, 2])

        goal_world_x, goal_world_y = robot_frame_to_world(
            robot_pose,
            rel_x,
            rel_y,
        )

        ax.plot(
            [robot_x, goal_world_x],
            [robot_y, goal_world_y],
            linestyle="-",
            linewidth=1.2,
            alpha=0.70,
            color="green",
            zorder=8,
            label="Local goal vector" if row_idx == 0 else "",
        )
        ax.plot(
            goal_world_x,
            goal_world_y,
            marker="o",
            linestyle="None",
            markersize=3,
            color="green",
            zorder=9,
        )
        ax.text(
            (robot_x + goal_world_x) / 2.0,
            (robot_y + goal_world_y) / 2.0,
            f"{rel_distance:.2f}",
            fontsize=6,
            color="green",
            zorder=9,
        )


def extract_run_index(run_directory: Path) -> int:
    match = re.search(r"(\d+)$", run_directory.name)
    if match is None:
        raise ValueError(
            f"Could not extract a run index from {run_directory.name}."
        )
    return int(match.group(1))


def sorted_run_directories(
    method_directory: Path,
    directory_pattern: str,
) -> List[Path]:
    directories = [
        path
        for path in method_directory.glob(directory_pattern)
        if path.is_dir()
    ]
    return sorted(directories, key=extract_run_index)


def visualize_run(
    run_directory: Path,
    run_idx: int,
    suffix: str,
    method_label: str,
    comparison_idx: int,
    cfg: Dict[str, float],
    dpi: int,
) -> bool:
    waypoint_file = (
        run_directory
        / f"se2_waypoints_{run_idx}_{suffix}.txt"
    )

    # Failed planner runs do not contain a waypoint file.
    if not waypoint_file.is_file() or waypoint_file.stat().st_size == 0:
        print(
            f"[SKIP] {method_label} run {run_idx}: "
            "no non-empty waypoint file."
        )
        return False

    obstacle_file = (
        run_directory
        / f"local_obs_{run_idx}_{suffix}.txt"
    )
    goal_file = (
        run_directory
        / f"local_goal_{run_idx}_{suffix}.txt"
    )

    waypoints = load_waypoints(waypoint_file)
    if waypoints.shape[0] == 0:
        print(
            f"[SKIP] {method_label} run {run_idx}: "
            "waypoint file contains no rows."
        )
        return False

    obstacle_rows = load_local_obstacle_info(obstacle_file)
    goal_rows = load_local_goal_info(goal_file)

    fig, ax = plt.subplots(figsize=(8, 8))

    plot_environment(ax, cfg)
    plot_robot_path(ax, waypoints, cfg, method_label)

    if obstacle_rows:
        plot_local_obstacle_information(
            ax,
            waypoints,
            obstacle_rows,
        )
    else:
        print(
            f"[WARN] {method_label} run {run_idx}: "
            f"local obstacle file missing or empty: {obstacle_file.name}"
        )

    if goal_rows.shape[0] > 0:
        plot_local_goal_information(
            ax,
            waypoints,
            goal_rows,
        )
    else:
        print(
            f"[WARN] {method_label} run {run_idx}: "
            f"local goal file missing or empty: {goal_file.name}"
        )

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("World x [m]")
    ax.set_ylabel("World y [m]")
    ax.set_title(
        f"Comparison {comparison_idx}: {method_label} run {run_idx}"
    )
    ax.grid(True, linewidth=0.4, alpha=0.30)
    ax.legend(loc="best", fontsize=7)

    output_file = (
        run_directory
        / f"planned_path_plot_{run_idx}_{suffix}.png"
    )

    fig.savefig(
        output_file,
        dpi=dpi,
        bbox_inches="tight",
    )
    plt.close(fig)

    print(f"[SAVED] {output_file}")
    return True


def main() -> int:
    args = parse_args()

    if args.comparison_idx <= 0:
        print(
            "ERROR: --comparison_idx must be a positive integer.",
            file=sys.stderr,
        )
        return 2

    dataset_root = args.dataset_root.expanduser().resolve()
    comparison_dir = (
        dataset_root
        / "Comparison_Values"
        / f"Comparison_{args.comparison_idx}"
    )

    if not comparison_dir.is_dir():
        print(
            f"ERROR: comparison directory does not exist: "
            f"{comparison_dir}",
            file=sys.stderr,
        )
        return 1

    try:
        query_file = resolve_query_file(
            comparison_dir,
            args.comparison_idx,
        )
    except (FileNotFoundError, RuntimeError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1

    # planner.cfg supplies generic map/robot plotting parameters when present.
    # The archived query always has priority for start, goal and obstacles.
    cfg: Dict[str, float] = {}
    planner_cfg = args.planner_cfg.expanduser()

    if planner_cfg.is_file():
        cfg.update(load_numeric_config(planner_cfg))
        print(f"Loaded supplemental planner config: {planner_cfg}")
    else:
        print(
            f"[WARN] Supplemental planner config not found: "
            f"{planner_cfg}; using defaults where necessary."
        )

    cfg.update(load_numeric_config(query_file))
    print(f"Loaded archived comparison query: {query_file}")

    method_specs = [
        {
            "label": "Classical RRT",
            "directory": comparison_dir / f"RRT_{args.comparison_idx}",
            "glob": "rrt_run_*",
            "suffix": "rrt",
        },
        {
            "label": "Diffusion RRT",
            "directory": (
                comparison_dir
                / f"Diffusion_RRT_{args.comparison_idx}"
            ),
            "glob": "diff_rrt_*",
            "suffix": "diff_rrt",
        },
        {
            "label": "Lateral Diffusion RRT",
            "directory": (
                comparison_dir
                / f"Lateral_Diff_RRT_{args.comparison_idx}"
            ),
            "glob": "lateral_diff_rrt_*",
            "suffix": "lateral_diff_rrt",
        },
    ]

    total_runs = 0
    total_plots = 0
    total_skipped = 0

    for specification in method_specs:
        method_directory: Path = specification["directory"]

        if not method_directory.is_dir():
            print(
                f"[WARN] Method directory not found: {method_directory}"
            )
            continue

        run_directories = sorted_run_directories(
            method_directory,
            specification["glob"],
        )

        print(
            f"\nProcessing {specification['label']}: "
            f"{len(run_directories)} run folders"
        )

        for run_directory in run_directories:
            total_runs += 1
            run_idx = extract_run_index(run_directory)

            try:
                plotted = visualize_run(
                    run_directory=run_directory,
                    run_idx=run_idx,
                    suffix=specification["suffix"],
                    method_label=specification["label"],
                    comparison_idx=args.comparison_idx,
                    cfg=cfg,
                    dpi=args.dpi,
                )
            except Exception as error:
                total_skipped += 1
                print(
                    f"[ERROR] Could not plot "
                    f"{specification['label']} run {run_idx}: {error}"
                )
                continue

            if plotted:
                total_plots += 1
            else:
                total_skipped += 1

    print("\nVisualization complete")
    print(f"Comparison index : {args.comparison_idx}")
    print(f"Run folders seen : {total_runs}")
    print(f"Plots generated  : {total_plots}")
    print(f"Runs skipped     : {total_skipped}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
