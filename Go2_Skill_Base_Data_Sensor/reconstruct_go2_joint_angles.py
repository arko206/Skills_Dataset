#!/usr/bin/env python3
"""
Recover the 12 actuated Unitree Go2 joint angles from synchronized
FrontJump recordings and save them as CSV files.

Inputs
------
1. Tf_stationary_FrontJump_Trial_3.csv
   Contains floor -> base pose at each synchronized base timestamp.

2. Tf_leg_stationary_FrontJump_Trial_3.csv
   Contains floor -> leg-frame poses, already synchronized to the same
   base timestamps.

Reconstruction
--------------
For each leg L in {FL, FR, RL, RR}:

    base_T_hip
        = inv(floor_T_base) @ floor_T_hip

    hip_T_thigh
        = inv(floor_T_hip) @ floor_T_thigh

    thigh_T_calf
        = inv(floor_T_thigh) @ floor_T_calf

The Go2 URDF joint axes used here are:

    hip joint   : X axis
    thigh joint : Y axis
    calf joint  : Y axis

Therefore:

    q_hip   = atan2(R_hip[2,1],   R_hip[1,1])
    q_thigh = atan2(R_thigh[0,2], R_thigh[0,0])
    q_calf  = atan2(R_calf[0,2],  R_calf[0,0])

No smoothing, interpolation, or resampling is performed in this script.
The synchronized timestamps already stored in the input CSVs are used directly.
"""

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd


LEG_PREFIXES = ("FL", "FR", "RL", "RR")
REQUIRED_FRAME_SUFFIXES = ("hip", "thigh", "calf")


def matrix_from_row(row, prefix):
    """Build one 4x4 homogeneous transform from CSV matrix columns."""
    return np.array(
        [
            [
                float(row[f"{prefix}00"]),
                float(row[f"{prefix}01"]),
                float(row[f"{prefix}02"]),
                float(row[f"{prefix}03"]),
            ],
            [
                float(row[f"{prefix}10"]),
                float(row[f"{prefix}11"]),
                float(row[f"{prefix}12"]),
                float(row[f"{prefix}13"]),
            ],
            [
                float(row[f"{prefix}20"]),
                float(row[f"{prefix}21"]),
                float(row[f"{prefix}22"]),
                float(row[f"{prefix}23"]),
            ],
            [
                float(row[f"{prefix}30"]),
                float(row[f"{prefix}31"]),
                float(row[f"{prefix}32"]),
                float(row[f"{prefix}33"]),
            ],
        ],
        dtype=float,
    )


def recover_joint_angles(base_csv, leg_csv):
    """
    Recover all 12 actuated joint angles at every synchronized timestamp.

    Returns
    -------
    pandas.DataFrame
        One row per synchronized base sample.
    """

    base_df = pd.read_csv(base_csv)
    leg_df = pd.read_csv(leg_csv)

    # ------------------------------------------------------------
    # 1. Validate required columns
    # ------------------------------------------------------------
    base_required = {
        "timestamp_iso",
        "common_time_ns",
        "x_floor_m",
        "y_floor_m",
        "z_floor_m",
        *[f"T{r}{c}" for r in range(4) for c in range(4)],
    }

    leg_required = {
        "base_common_time_ns",
        "frame_name",
        *[f"F_T{r}{c}" for r in range(4) for c in range(4)],
    }

    missing_base = base_required - set(base_df.columns)
    missing_leg = leg_required - set(leg_df.columns)

    if missing_base:
        raise ValueError(
            f"Base CSV is missing required columns: {sorted(missing_base)}"
        )

    if missing_leg:
        raise ValueError(
            f"Leg CSV is missing required columns: {sorted(missing_leg)}"
        )

    # ------------------------------------------------------------
    # 2. Sort base samples and verify timestamp synchronization
    # ------------------------------------------------------------
    base_df = (
        base_df
        .sort_values("common_time_ns")
        .reset_index(drop=True)
    )

    base_timestamps = base_df["common_time_ns"].astype(np.int64).to_numpy()

    if len(np.unique(base_timestamps)) != len(base_timestamps):
        raise ValueError(
            "Base CSV contains duplicated common_time_ns values."
        )

    required_frames = [
        f"{leg}_{suffix}"
        for leg in LEG_PREFIXES
        for suffix in REQUIRED_FRAME_SUFFIXES
    ]

    required_leg_df = leg_df[
        leg_df["frame_name"].isin(required_frames)
    ].copy()

    leg_unique_timestamps = np.sort(
        required_leg_df["base_common_time_ns"]
        .astype(np.int64)
        .unique()
    )

    if not np.array_equal(base_timestamps, leg_unique_timestamps):
        raise ValueError(
            "Base and synchronized leg timestamps do not match exactly. "
            "This script will not interpolate or resample them."
        )

    # Every required frame must have exactly one row per base timestamp.
    for frame_name in required_frames:
        frame_df = required_leg_df[
            required_leg_df["frame_name"] == frame_name
        ]

        frame_timestamps = np.sort(
            frame_df["base_common_time_ns"]
            .astype(np.int64)
            .to_numpy()
        )

        if not np.array_equal(frame_timestamps, base_timestamps):
            raise ValueError(
                f"Frame '{frame_name}' does not contain exactly one "
                "sample for every synchronized base timestamp."
            )

    # ------------------------------------------------------------
    # 3. Build fast lookup:
    #       timestamp -> frame_name -> CSV row
    # ------------------------------------------------------------
    grouped_leg_rows = {}

    for timestamp_ns, group in required_leg_df.groupby(
        "base_common_time_ns",
        sort=False,
    ):
        grouped_leg_rows[int(timestamp_ns)] = (
            group.set_index("frame_name")
        )

    # ------------------------------------------------------------
    # 4. Reconstruct 12 joint angles
    # ------------------------------------------------------------
    t0_ns = int(base_timestamps[0])
    output_rows = []

    for sample_index, base_row in base_df.iterrows():

        timestamp_ns = int(base_row["common_time_ns"])
        relative_time_sec = (
            timestamp_ns - t0_ns
        ) / 1_000_000_000.0

        floor_to_base = matrix_from_row(
            base_row,
            "T",
        )

        frame_group = grouped_leg_rows[timestamp_ns]

        output_row = {
            "sample_index": int(sample_index),
            "timestamp_iso": base_row["timestamp_iso"],
            "common_time_ns": timestamp_ns,
            "relative_time_sec": relative_time_sec,

            # Keep the base trajectory beside the joint angles so
            # later plots can directly compare leg articulation
            # with the robot's global FrontJump motion.
            "base_x_m": float(base_row["x_floor_m"]),
            "base_y_m": float(base_row["y_floor_m"]),
            "base_z_m": float(base_row["z_floor_m"]),
        }

        for leg_prefix in LEG_PREFIXES:

            floor_to_hip = matrix_from_row(
                frame_group.loc[f"{leg_prefix}_hip"],
                "F_T",
            )

            floor_to_thigh = matrix_from_row(
                frame_group.loc[f"{leg_prefix}_thigh"],
                "F_T",
            )

            floor_to_calf = matrix_from_row(
                frame_group.loc[f"{leg_prefix}_calf"],
                "F_T",
            )

            # ----------------------------------------------------
            # Convert absolute floor-relative poses into the
            # URDF parent -> child relative transforms.
            # ----------------------------------------------------
            base_to_hip = (
                np.linalg.inv(floor_to_base)
                @ floor_to_hip
            )

            hip_to_thigh = (
                np.linalg.inv(floor_to_hip)
                @ floor_to_thigh
            )

            thigh_to_calf = (
                np.linalg.inv(floor_to_thigh)
                @ floor_to_calf
            )

            R_hip = base_to_hip[:3, :3]
            R_thigh = hip_to_thigh[:3, :3]
            R_calf = thigh_to_calf[:3, :3]

            # ----------------------------------------------------
            # Recover the revolute joint angles using the Go2
            # URDF joint axes.
            # ----------------------------------------------------

            # Hip: X-axis revolute joint.
            q_hip_rad = np.arctan2(
                R_hip[2, 1],
                R_hip[1, 1],
            )

            # Thigh: Y-axis revolute joint.
            q_thigh_rad = np.arctan2(
                R_thigh[0, 2],
                R_thigh[0, 0],
            )

            # Calf: Y-axis revolute joint.
            q_calf_rad = np.arctan2(
                R_calf[0, 2],
                R_calf[0, 0],
            )

            output_row[
                f"{leg_prefix}_hip_rad"
            ] = float(q_hip_rad)

            output_row[
                f"{leg_prefix}_thigh_rad"
            ] = float(q_thigh_rad)

            output_row[
                f"{leg_prefix}_calf_rad"
            ] = float(q_calf_rad)

            output_row[
                f"{leg_prefix}_hip_deg"
            ] = float(np.degrees(q_hip_rad))

            output_row[
                f"{leg_prefix}_thigh_deg"
            ] = float(np.degrees(q_thigh_rad))

            output_row[
                f"{leg_prefix}_calf_deg"
            ] = float(np.degrees(q_calf_rad))

        output_rows.append(output_row)

    return pd.DataFrame(output_rows)


def save_joint_csvs(joint_df, output_dir, label):
    """
    Save one complete CSV plus one compact CSV for each leg.
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------
    # 1. Complete file: all 12 joint trajectories
    # ------------------------------------------------------------
    all_file = output_dir / f"JointAngles_{label}_All.csv"

    joint_df.to_csv(
        all_file,
        index=False,
        float_format="%.9f",
    )

    # ------------------------------------------------------------
    # 2. Per-leg files
    # ------------------------------------------------------------
    leg_files = {}

    common_columns = [
        "sample_index",
        "timestamp_iso",
        "common_time_ns",
        "relative_time_sec",
        "base_x_m",
        "base_y_m",
        "base_z_m",
    ]

    for leg_prefix in LEG_PREFIXES:

        leg_columns = common_columns + [
            f"{leg_prefix}_hip_rad",
            f"{leg_prefix}_thigh_rad",
            f"{leg_prefix}_calf_rad",
            f"{leg_prefix}_hip_deg",
            f"{leg_prefix}_thigh_deg",
            f"{leg_prefix}_calf_deg",
        ]

        leg_file = (
            output_dir
            / f"JointAngles_{label}_{leg_prefix}.csv"
        )

        joint_df[leg_columns].to_csv(
            leg_file,
            index=False,
            float_format="%.9f",
        )

        leg_files[leg_prefix] = leg_file

    return all_file, leg_files


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Recover Unitree Go2 FL/FR/RL/RR hip, thigh, and calf "
            "joint angles from synchronized spatial-pose recordings."
        )
    )

    parser.add_argument(
        "--base-csv",
        default=(
            "/home/unitree-arka/Go2_Skill_Base_Data_Sensor/"
            "FrontJump/Trial_3/Tf_stationary_FrontJump_Trial_3.csv"
        ),
        help="Path to Tf_stationary_*.csv",
    )

    parser.add_argument(
        "--leg-csv",
        default=(
            "/home/unitree-arka/Go2_Skill_Base_Data_Sensor/"
            "FrontJump/Trial_3/Tf_leg_stationary_FrontJump_Trial_3.csv"
        ),
        help="Path to Tf_leg_stationary_*.csv",
    )

    parser.add_argument(
        "--output-dir",
        default=(
            "/home/unitree-arka/Go2_Skill_Base_Data_Sensor/"
            "FrontJump/Trial_3"
        ),
        help="Directory where reconstructed joint-angle CSVs are saved",
    )

    parser.add_argument(
        "--label",
        default="FrontJump_Trial_3",
        help="Filename label for the generated CSV files",
    )

    args = parser.parse_args()

    joint_df = recover_joint_angles(
        args.base_csv,
        args.leg_csv,
    )

    all_file, leg_files = save_joint_csvs(
        joint_df,
        args.output_dir,
        args.label,
    )

    print("=" * 72)
    print("JOINT-ANGLE RECONSTRUCTION COMPLETE")
    print("=" * 72)
    print(f"Samples reconstructed : {len(joint_df)}")
    print(
        "Trajectory duration   : "
        f"{joint_df['relative_time_sec'].iloc[-1]:.6f} s"
    )
    print(f"Combined CSV          : {all_file}")

    for leg_prefix, leg_file in leg_files.items():
        print(
            f"{leg_prefix} CSV"
            f"{' ' * (15 - len(leg_prefix))}: {leg_file}"
        )

    print("=" * 72)

    # Simple numerical range summary for quick sanity checking.
    for leg_prefix in LEG_PREFIXES:
        print(f"\n[{leg_prefix}]")
        for joint_name in ("hip", "thigh", "calf"):
            column = f"{leg_prefix}_{joint_name}_deg"
            print(
                f"  {joint_name:6s}: "
                f"min={joint_df[column].min():9.3f} deg, "
                f"max={joint_df[column].max():9.3f} deg"
            )


if __name__ == "__main__":
    main()
