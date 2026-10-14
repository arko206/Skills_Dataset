#!/usr/bin/env python3
"""
Unitree Go2: Stand -> HandStand transition model (fixed skill, no input parameters)
==================================================================================

The Stand -> HandStand skill has no command vector V, so there is no regression.
The learned model is the MEAN net SE(2) displacement over repeated executions:

    Delta_p_body = [delta_x_body, delta_y_body, delta_theta]^T     (m, m, rad)

For each trial (already-filtered Stand -> HandStand Base trajectory):

    initial pose (x0, y0, theta0) = first row,   final pose (x1, y1, theta1) = last row

    dx_world = x1 - x0,   dy_world = y1 - y0
    delta_x_body =  cos(theta0) * dx_world + sin(theta0) * dy_world
    delta_y_body = -sin(theta0) * dx_world + cos(theta0) * dy_world
    delta_theta  = atan2(sin(theta1 - theta0), cos(theta1 - theta0))

(the same SE(2) convention as the HandStandMove affine model). Over all valid
trials D (N x 3):

    mean_transition = mean(D, axis=0)
    std_transition  = std(D, axis=0, ddof=1)            (sample standard deviation)
    covariance      = cov(D, rowvar=False, ddof=1)      (3 x 3 sample covariance)

Variable ordering everywhere: delta_x_body, delta_y_body, delta_theta.
Input CSVs are only read. Outputs: a per-trial table and a text model file.
"""

from pathlib import Path

import numpy as np
import pandas as pd

# =============================================================================
# CONFIGURATION
# =============================================================================
STAGE_ROOT = Path("/home/unitree-arka/Go2_Skill_Base_Data_Sensor/Stage_Stand_HS")
TRIAL_NUMBERS = list(range(1, 66))                       # Trials 1-65

TIME_COL = "common_time_ns"
POSE_COLS = ["x_floor_m", "y_floor_m", "yaw_floor_rad"]

TABLE_PATH = STAGE_ROOT / "Stand_to_HandStand_transition_table.csv"
MODEL_PATH = STAGE_ROOT / "Stand_to_HandStand_transition_model.txt"

OUTPUT_NAMES = ["delta_x_body", "delta_y_body", "delta_theta"]
DECIMALS = 12                                            # precision in the model file


class TrialError(Exception):
    """A trial that cannot be used; it is reported and skipped."""


def base_csv(n):
    return STAGE_ROOT / f"Stage_Stand_HS_trial_{n}" / f"Tf_Stand_to_HS_HandStandMove_Trial_{n}.csv"


# =============================================================================
# INITIAL / FINAL POSE
# =============================================================================
def read_initial_and_final_pose(n):
    """First and last row of the filtered trajectory, after basic validation."""
    path = base_csv(n)
    if not path.is_file():
        raise TrialError(f"Base CSV not found: {path}")
    df = pd.read_csv(path)

    missing = [c for c in POSE_COLS + [TIME_COL] if c not in df.columns]
    if missing:
        raise TrialError(f"missing column(s) {missing}")
    if len(df) < 2:
        raise TrialError(f"only {len(df)} row(s); need at least 2")
    t = pd.to_numeric(df[TIME_COL], errors="coerce").to_numpy()
    if not np.all(np.diff(t) > 0):
        raise TrialError(f"{TIME_COL} is not strictly increasing")

    pose_initial = pd.to_numeric(df[POSE_COLS].iloc[0], errors="coerce").to_numpy(dtype=float)
    pose_final = pd.to_numeric(df[POSE_COLS].iloc[-1], errors="coerce").to_numpy(dtype=float)
    if not (np.isfinite(pose_initial).all() and np.isfinite(pose_final).all()):
        raise TrialError("initial or final pose contains a missing/non-finite value")
    return pose_initial, pose_final


# =============================================================================
# SE(2) DISPLACEMENT IN THE INITIAL BODY FRAME
# =============================================================================
def wrap_angle(angle):
    """Map an angle (radians) into [-pi, pi]."""
    return np.arctan2(np.sin(angle), np.cos(angle))


def compute_body_frame_displacement(pose_initial, pose_final):
    """
    [dx_body, dy_body]^T = R(theta0)^T [dx_world, dy_world]^T,  delta_theta wrapped.
    """
    x0, y0, th0 = pose_initial
    x1, y1, th1 = pose_final

    dx_world = x1 - x0
    dy_world = y1 - y0

    c, s = np.cos(th0), np.sin(th0)
    return {
        "delta_x_world": float(dx_world),
        "delta_y_world": float(dy_world),
        "delta_x_body": float(c * dx_world + s * dy_world),
        "delta_y_body": float(-s * dx_world + c * dy_world),
        "delta_theta": float(wrap_angle(th1 - th0)),
    }


# =============================================================================
# PER-TRIAL TABLE
# =============================================================================
def build_table():
    rows, skipped = [], []
    for n in TRIAL_NUMBERS:
        try:
            p0, p1 = read_initial_and_final_pose(n)
        except TrialError as exc:
            print(f"[WARNING] Trial {n} skipped: {exc}")
            skipped.append(n)
            continue
        except Exception as exc:                     # unreadable file etc.
            print(f"[WARNING] Trial {n} skipped ({type(exc).__name__}): {exc}")
            skipped.append(n)
            continue

        d = compute_body_frame_displacement(p0, p1)
        print(f"Trial {n}")
        print(f"  initial pose = [{p0[0]:.6f}, {p0[1]:.6f}, {p0[2]:.6f}]")
        print(f"  final pose   = [{p1[0]:.6f}, {p1[1]:.6f}, {p1[2]:.6f}]")
        print(f"  Delta_world  = [{d['delta_x_world']:+.6f}, {d['delta_y_world']:+.6f}]")
        print(f"  Delta_body   = [{d['delta_x_body']:+.6f}, {d['delta_y_body']:+.6f}, {d['delta_theta']:+.6f}]")

        rows.append({
            "trial_number": n,
            "x_initial": p0[0], "y_initial": p0[1], "yaw_initial": p0[2],
            "x_final": p1[0], "y_final": p1[1], "yaw_final": p1[2],
            **d,
        })
    return pd.DataFrame(rows), skipped


# =============================================================================
# MODEL FILE
# =============================================================================
def f(value):
    return f"{value:+.{DECIMALS}f}"


def write_model_file(n_valid, mean, std, cov):
    lines = [
        "Unitree Go2 Stand -> HandStand transition model",
        "",
        "Number of valid trials:",
        f"{n_valid}",
        "",
        "Representation:",
        "Delta_p_body = [delta_x_body, delta_y_body, delta_theta]^T",
        "Units: delta_x_body [m], delta_y_body [m], delta_theta [rad]",
        "Translation expressed in the robot's INITIAL body frame; delta_theta wrapped to [-pi, pi].",
        "",
        "Mean transition:",
        f"mean_delta_x_body   = {f(mean[0])}",
        f"mean_delta_y_body   = {f(mean[1])}",
        f"mean_delta_theta    = {f(mean[2])}",
        "",
        "Mean transition vector:",
        f"[ {f(mean[0])}  {f(mean[1])}  {f(mean[2])} ]",
        "",
        "Standard deviation (sample, ddof = 1):",
        f"std_delta_x_body    = {f(std[0])}",
        f"std_delta_y_body    = {f(std[1])}",
        f"std_delta_theta     = {f(std[2])}",
        "",
        "Sample covariance matrix (ddof = 1):",
        *[f"[ {f(cov[i, 0])}  {f(cov[i, 1])}  {f(cov[i, 2])} ]" for i in range(3)],
        "",
        "Variable ordering:",
        "delta_x_body, delta_y_body, delta_theta",
        "",
    ]
    MODEL_PATH.write_text("\n".join(lines))


# =============================================================================
# MAIN
# =============================================================================
def main():
    table, skipped = build_table()

    print(f"\nTrials requested : {len(TRIAL_NUMBERS)}")
    print(f"Valid trials     : {len(table)}")
    print(f"Skipped trials   : {len(skipped)}" + (f"  {skipped}" if skipped else ""))
    if len(table) < 2:
        raise SystemExit("[ERROR] fewer than 2 valid trials; the sample covariance cannot be computed")

    D = table[OUTPUT_NAMES].to_numpy(dtype=float)          # N x 3
    mean = np.mean(D, axis=0)
    std = np.std(D, axis=0, ddof=1)
    cov = np.cov(D, rowvar=False, ddof=1)

    # The arithmetic mean of delta_theta is only meaningful if the values do not
    # straddle +/-pi (where wrapping would split them into two far-apart groups).
    if D[:, 2].max() - D[:, 2].min() > np.pi:
        print("[WARNING] delta_theta values span more than pi rad; their arithmetic mean may be "
              "misleading (values near +/-pi wrap). Inspect the table before using the mean.")

    TABLE_PATH.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(TABLE_PATH, index=False)
    write_model_file(len(table), mean, std, cov)

    print("\n" + "=" * 64)
    print("MEAN STAND -> HANDSTAND TRANSITION")
    print("=" * 64)
    print(f"mean Delta_x_body = {f(mean[0])} m")
    print(f"mean Delta_y_body = {f(mean[1])} m")
    print(f"mean Delta_theta  = {f(mean[2])} rad")
    print("\nSTANDARD DEVIATION (sample, ddof = 1)")
    print(f"std  Delta_x_body = {f(std[0])} m")
    print(f"std  Delta_y_body = {f(std[1])} m")
    print(f"std  Delta_theta  = {f(std[2])} rad")
    print("\nSAMPLE COVARIANCE MATRIX (ordering: delta_x_body, delta_y_body, delta_theta)")
    for i in range(3):
        print(f"[ {f(cov[i, 0])}  {f(cov[i, 1])}  {f(cov[i, 2])} ]")
    print(f"\nPer-trial table : {TABLE_PATH}")
    print(f"Model file      : {MODEL_PATH}")


if __name__ == "__main__":
    main()
