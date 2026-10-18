#!/usr/bin/env python3

"""
Fit the Unitree Go2 single-shot affine dynamics model

    Delta_p = B V + C

using BATCH GRADIENT DESCENT.

Inputs per experiment:
    V = [vx, vy, vyaw]^T

Outputs per experiment:
    Delta_p = [delta_x_body, delta_y_body, delta_theta]^T

For every file:
    Tf_stationary_<vx>_<vy>_<vyaw>.csv

the script:
1. Parses (vx, vy, vyaw) from the filename.
2. Takes the first pose as the initial pose.
3. Takes the last row with stationary_candidate == 1 as the settled pose
   (falls back to the final CSV row if no stationary candidate exists).
4. Computes world-frame displacement.
5. Rotates translation into the robot body frame at the INITIAL pose.
6. Fits B and C using gradient descent.
7. Prints learned B, C, B^{-1}, MSE, RMSE, and R^2.
8. Compares the learned values against the matrices in go2_navg_exec.cpp.
9. Saves training loss and measured-vs-predicted plots.
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

DATA_DIR = Path(
    "/home/unitree-arka/Go2_Walk_Base_Data_Sensor/"
    "Old_Data_Collection/Parameters_RS/"
    "Stationary_CSV_req"
)

FILE_PATTERN = "Tf_stationary_*.csv"

# Gradient-descent hyperparameters
LEARNING_RATE = 0.10
MAX_ITERATIONS = 20000
TOLERANCE = 1e-14

# Outputs
TRAINING_TABLE_CSV = DATA_DIR / "affine_dynamics_training_table.csv"
LOSS_PLOT_PNG = DATA_DIR / "affine_dynamics_gradient_descent_loss.png"
PREDICTION_PLOT_PNG = DATA_DIR / "affine_dynamics_measured_vs_predicted.png"


# ============================================================
# SUPERVISOR'S VALUES FROM go2_navg_exec.cpp
# Used only for comparison AFTER training.
# They are NOT used during optimization.
# ============================================================

B_SUPERVISOR = np.array([
    [ 0.49006,  0.01829,  0.01400],
    [-0.01689,  0.52700,  0.03446],
    [-0.06484,  0.07759,  0.53092],
], dtype=float)

C_SUPERVISOR = np.array([
     0.00060,
    -0.00547,
    -0.08325,
], dtype=float)

B_INV_SUPERVISOR = np.array([
    [ 2.03186, -0.06325, -0.04946],
    [ 0.04937,  1.91432, -0.12556],
    [ 0.24092, -0.28750,  1.89582],
], dtype=float)


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def wrap_angle(angle_rad: float) -> float:
    """Wrap angle into [-pi, pi]."""
    return math.atan2(math.sin(angle_rad), math.cos(angle_rad))


def parse_command_from_filename(path: Path):
    """
    Example:
        Tf_stationary_0.5_0.0_-0.5.csv
    gives:
        vx = 0.5
        vy = 0.0
        vyaw = -0.5
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
            f"Filename does not match expected pattern: {path.name}"
        )

    return (
        float(match.group(1)),
        float(match.group(2)),
        float(match.group(3)),
    )


def choose_start_and_final_pose(df: pd.DataFrame):
    """
    Initial pose:
        first row of the CSV.

    Final settled pose:
        last row where stationary_candidate == 1.

    If stationary_candidate is unavailable or contains no 1,
    use the final CSV row.
    """
    required = [
        "x_floor_m",
        "y_floor_m",
        "yaw_floor_rad",
    ]

    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns: {missing}")

    if len(df) < 2:
        raise ValueError("CSV contains fewer than two pose samples.")

    start = df.iloc[0]

    if "stationary_candidate" in df.columns:
        stationary = df[df["stationary_candidate"] == 1]

        if not stationary.empty:
            final = stationary.iloc[-1]
        else:
            final = df.iloc[-1]
    else:
        final = df.iloc[-1]

    return start, final


def compute_body_frame_displacement(start, final):
    """
    Compute net SE(2) displacement from the initial pose to the final pose.

    World/floor frame:
        dx_g = x_f - x_i
        dy_g = y_f - y_i

    Convert translation into the INITIAL robot body frame:

        dx_b =  cos(theta_i) dx_g + sin(theta_i) dy_g
        dy_b = -sin(theta_i) dx_g + cos(theta_i) dy_g

    Angular displacement:
        dtheta = wrap(theta_f - theta_i)
    """

    x_i = float(start["x_floor_m"])
    y_i = float(start["y_floor_m"])
    theta_i = float(start["yaw_floor_rad"])

    x_f = float(final["x_floor_m"])
    y_f = float(final["y_floor_m"])
    theta_f = float(final["yaw_floor_rad"])

    dx_g = x_f - x_i
    dy_g = y_f - y_i

    c = math.cos(theta_i)
    s = math.sin(theta_i)

    dx_b = c * dx_g + s * dy_g
    dy_b = -s * dx_g + c * dy_g

    dtheta = wrap_angle(theta_f - theta_i)

    return dx_b, dy_b, dtheta


# ============================================================
# BUILD TRAINING DATASET
# ============================================================

files = sorted(DATA_DIR.glob(FILE_PATTERN))

if not files:
    raise FileNotFoundError(
        f"No files matching {FILE_PATTERN} found in:\n{DATA_DIR}"
    )

records = []

for csv_path in files:
    try:
        vx, vy, vyaw = parse_command_from_filename(csv_path)

        df = pd.read_csv(csv_path)

        start, final = choose_start_and_final_pose(df)

        dx_b, dy_b, dtheta = compute_body_frame_displacement(
            start,
            final,
        )

        records.append({
            "file": csv_path.name,
            "vx": vx,
            "vy": vy,
            "vyaw": vyaw,
            "delta_x_body": dx_b,
            "delta_y_body": dy_b,
            "delta_theta": dtheta,
        })

    except Exception as exc:
        print(f"[WARNING] Skipping {csv_path.name}: {exc}")


data = pd.DataFrame(records)

if data.empty:
    raise RuntimeError("No valid experiments were found.")

data.to_csv(TRAINING_TABLE_CSV, index=False)

print("\n============================================================")
print("TRAINING DATA")
print("============================================================")
print(data.to_string(index=False))
print(f"\nNumber of experiments N = {len(data)}")


# ============================================================
# MATRIX FORM OF THE REGRESSION
# ============================================================
#
# For experiment i:
#
#       y_i = B v_i + C
#
# where
#
#       v_i = [vx, vy, vyaw]^T
#
#       y_i = [delta_x_body, delta_y_body, delta_theta]^T
#
#
# In NumPy we store experiments by ROW:
#
#       V : N x 3
#       Y : N x 3
#
# therefore prediction for all experiments is:
#
#       Y_hat = V B^T + C
#
# C is automatically broadcast over every row.
# ============================================================

V = data[
    ["vx", "vy", "vyaw"]
].to_numpy(dtype=float)

Y = data[
    ["delta_x_body", "delta_y_body", "delta_theta"]
].to_numpy(dtype=float)

N = V.shape[0]


# ============================================================
# INITIALIZE MODEL PARAMETERS
# ============================================================

# B has 9 trainable coefficients.
B = np.zeros((3, 3), dtype=float)

# C has 3 trainable intercepts.
C = np.zeros(3, dtype=float)


# ============================================================
# BATCH GRADIENT DESCENT
# ============================================================
#
# Model:
#
#       Y_hat = V B^T + C
#
# Error:
#
#       E = Y_hat - Y
#
# Loss:
#
#                    1
#       J(B,C) = --------- sum_i || y_hat_i - y_i ||_2^2
#                    N
#
#
# Gradients:
#
#       dJ/dB = (2/N) E^T V
#
#       dJ/dC = (2/N) sum_i E_i
#
#
# Gradient descent:
#
#       B <- B - alpha * dJ/dB
#
#       C <- C - alpha * dJ/dC
#
# ============================================================

loss_history = []

previous_loss = np.inf

for iteration in range(MAX_ITERATIONS):

    # --------------------------------------------------------
    # 1. FORWARD PREDICTION
    # --------------------------------------------------------
    Y_hat = V @ B.T + C

    # --------------------------------------------------------
    # 2. RESIDUAL / ERROR
    # --------------------------------------------------------
    error = Y_hat - Y

    # --------------------------------------------------------
    # 3. MEAN SQUARED VECTOR ERROR
    # --------------------------------------------------------
    loss = np.sum(error ** 2) / N
    loss_history.append(loss)

    # --------------------------------------------------------
    # 4. GRADIENTS
    # --------------------------------------------------------
    grad_B = (2.0 / N) * (error.T @ V)

    grad_C = (2.0 / N) * np.sum(
        error,
        axis=0,
    )

    # --------------------------------------------------------
    # 5. GRADIENT-DESCENT UPDATE
    # --------------------------------------------------------
    B = B - LEARNING_RATE * grad_B
    C = C - LEARNING_RATE * grad_C

    # --------------------------------------------------------
    # PRINT PROGRESS
    # --------------------------------------------------------
    if iteration % 1000 == 0:
        print(
            f"Iteration {iteration:6d} | "
            f"Loss = {loss:.12f}"
        )

    # --------------------------------------------------------
    # EARLY STOPPING
    # --------------------------------------------------------
    if abs(previous_loss - loss) < TOLERANCE:
        print(
            f"\nConverged at iteration {iteration} "
            f"with loss {loss:.12f}"
        )
        break

    previous_loss = loss


# ============================================================
# FINAL PREDICTIONS
# ============================================================

Y_pred = V @ B.T + C

residual = Y_pred - Y

mse_each_output = np.mean(residual ** 2, axis=0)
rmse_each_output = np.sqrt(mse_each_output)


# R^2 for each output
ss_res = np.sum((Y - Y_pred) ** 2, axis=0)
ss_tot = np.sum(
    (Y - np.mean(Y, axis=0)) ** 2,
    axis=0,
)

r2 = 1.0 - ss_res / ss_tot


# ============================================================
# PRINT LEARNED MODEL
# ============================================================

np.set_printoptions(
    precision=8,
    suppress=True,
)

print("\n============================================================")
print("LEARNED FORWARD MODEL")
print("Delta_p = B V + C")
print("============================================================")

print("\nLearned B:")
print(B)

print("\nLearned C:")
print(C)


# ============================================================
# INVERSE MATRIX
# ============================================================

det_B = np.linalg.det(B)

print("\ndet(B) =", det_B)

if abs(det_B) > 1e-12:
    B_inv = np.linalg.inv(B)

    print("\nComputed inverse B^{-1}:")
    print(B_inv)
else:
    B_inv = None
    print("\nB is singular or nearly singular; inverse not computed.")


# ============================================================
# MODEL QUALITY
# ============================================================

print("\n============================================================")
print("TRAINING FIT QUALITY")
print("============================================================")

names = [
    "delta_x_body",
    "delta_y_body",
    "delta_theta",
]

for j, name in enumerate(names):
    print(
        f"{name:16s} | "
        f"MSE = {mse_each_output[j]:.8f} | "
        f"RMSE = {rmse_each_output[j]:.8f} | "
        f"R^2 = {r2[j]:.6f}"
    )


# ============================================================
# COMPARE WITH SUPERVISOR'S MODEL
# ============================================================

print("\n============================================================")
print("SUPERVISOR MODEL")
print("============================================================")

print("\nSupervisor B:")
print(B_SUPERVISOR)

print("\nSupervisor C:")
print(C_SUPERVISOR)


print("\n============================================================")
print("DIFFERENCE: learned - supervisor")
print("============================================================")

print("\nB difference:")
print(B - B_SUPERVISOR)

print("\nC difference:")
print(C - C_SUPERVISOR)


if B_inv is not None:
    print("\nSupervisor B_INV:")
    print(B_INV_SUPERVISOR)

    print("\nLearned B_INV - Supervisor B_INV:")
    print(B_inv - B_INV_SUPERVISOR)


# ============================================================
# PRINT THE THREE LEARNED REGRESSION EQUATIONS
# ============================================================

print("\n============================================================")
print("LEARNED EQUATIONS")
print("============================================================")

print(
    "\ndelta_x_body = "
    f"{B[0,0]:+.6f} vx "
    f"{B[0,1]:+.6f} vy "
    f"{B[0,2]:+.6f} vyaw "
    f"{C[0]:+.6f}"
)

print(
    "delta_y_body = "
    f"{B[1,0]:+.6f} vx "
    f"{B[1,1]:+.6f} vy "
    f"{B[1,2]:+.6f} vyaw "
    f"{C[1]:+.6f}"
)

print(
    "delta_theta  = "
    f"{B[2,0]:+.6f} vx "
    f"{B[2,1]:+.6f} vy "
    f"{B[2,2]:+.6f} vyaw "
    f"{C[2]:+.6f}"
)


# ============================================================
# SAVE LOSS CURVE
# ============================================================

plt.figure(figsize=(8, 5))

plt.plot(loss_history)

plt.xlabel("Gradient-descent iteration")
plt.ylabel("Training loss J(B,C)")
plt.title("Affine Dynamics Model: Gradient-Descent Convergence")
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(
    LOSS_PLOT_PNG,
    dpi=300,
    bbox_inches="tight",
)

print(f"\nSaved loss plot:\n{LOSS_PLOT_PNG}")


# ============================================================
# MEASURED VS PREDICTED PLOTS
# ============================================================

fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

labels = [
    r"$\Delta x_b$ (m)",
    r"$\Delta y_b$ (m)",
    r"$\Delta \theta$ (rad)",
]

for j, ax in enumerate(axes):

    measured = Y[:, j]
    predicted = Y_pred[:, j]

    ax.scatter(measured, predicted, s=45)

    min_value = min(
        np.min(measured),
        np.min(predicted),
    )

    max_value = max(
        np.max(measured),
        np.max(predicted),
    )

    ax.plot(
        [min_value, max_value],
        [min_value, max_value],
        linewidth=1.5,
    )

    ax.set_xlabel("Measured " + labels[j])
    ax.set_ylabel("Predicted " + labels[j])
    ax.set_title(
        f"{labels[j]}\n"
        f"$R^2$ = {r2[j]:.4f}"
    )

    ax.grid(True, alpha=0.3)

fig.suptitle(
    "Measured vs Predicted Single-Shot Body-Frame Displacement",
    fontsize=14,
)

fig.tight_layout(rect=[0, 0, 1, 0.94])

fig.savefig(
    PREDICTION_PLOT_PNG,
    dpi=300,
    bbox_inches="tight",
)

print(
    f"Saved measured-vs-predicted plot:\n"
    f"{PREDICTION_PLOT_PNG}"
)

plt.show()
