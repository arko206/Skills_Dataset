#!/usr/bin/env python3
"""
Unitree Go2 HandStandMove: affine single-shot transition model fitted by batch gradient descent
===============================================================================================

Model
-----
For one HandStandMove command, the Base's net displacement during the HandStand+Move
stage, expressed in the robot's INITIAL body frame, is modelled as

    Delta_p_body = B V + C

    V            = [vx, vy, vyaw]^T                               (m/s, m/s, rad/s)
    Delta_p_body = [delta_x_body, delta_y_body, delta_theta]^T    (m, m, rad)
    B : 3 x 3,  C : 3 x 1

Data (one training sample per stage)
------------------------------------
    Stage_N_HS&Move/Tf_handstand_stage_HandStandMove_Trial_N.csv     (filtered Base trajectory)
        -> first row = initial HandStand pose,  last row = final HandStand pose
           (the files were already selected and checked in RViz; no further
            motion-state filtering is done here)
    HandStandMove/Trial_N/cmd_HandStandMove_Trial_N.csv              (commanded vx, vy, vyaw)

Stage_N is always paired with Trial_N. The command values are read from the
command CSV only; they are never inferred from the trajectory, and the measured
velocity columns of the Base CSV (vx_body_mps, ...) are NOT used.

Displacement (unchanged from the previous Move() script)
--------------------------------------------------------
    dx_world = x1 - x0,  dy_world = y1 - y0
    delta_x_body =  cos(theta0) * dx_world + sin(theta0) * dy_world
    delta_y_body = -sin(theta0) * dx_world + cos(theta0) * dy_world
    delta_theta  = wrap_angle(theta1 - theta0),  wrap_angle(a) = atan2(sin a, cos a)

Fitting
-------
Batch gradient descent with a FIXED learning rate (0.1) on standardized inputs
Vs = (V - mean) / std. The learned standardized parameters are converted back so
that the printed B and C act directly on vx [m/s], vy [m/s], vyaw [rad/s].

Source CSVs are only read. Outputs: the training table, a plain-text file with the
physical-unit B, C and B_inverse, the loss-vs-iteration plot, and two diagnostic
plots of the learned model on the training stages (measured vs predicted, and
prediction error = measured - predicted vs predicted).
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# =============================================================================
# CONFIGURATION
# =============================================================================
STAGE_ROOT = Path("/home/unitree-arka/Go2_Skill_Base_Data_Sensor/Stage_HS_and_Move")
TRIAL_ROOT = Path("/home/unitree-arka/Go2_Skill_Base_Data_Sensor/HandStandMove")
##--- there are 65 collected Trajectories for HandStand Move ------------------------######
STAGE_NUMBERS = list(range(1, 66))              # Stage_1_HS&Move ... Stage_65_HS&Move

POSE_COLS = ["x_floor_m", "y_floor_m", "yaw_floor_rad"]
TIME_COL = "common_time_ns"

# --- Command CSV --------------------------------------------------------------
# Expected file name inside HandStandMove/Trial_N/. .
CMD_FILE_TEMPLATE = "cmd_HandStandMove_Trial_{n}.csv"

# Accepted column names for each commanded velocity, checked in this order.
# If your command CSV uses different names, ADD THEM HERE. The script stops with
# the list of actual columns rather than guessing. Never add the Base CSV's
# measured columns (vx_body_mps, vy_body_mps, vyaw_body_radps).
CMD_COLUMN_CANDIDATES = {
    "vx":   ["vx"],
    "vy":   ["vy"],
    "vyaw": ["vyaw"],
}

# --- Gradient descent ---------------------------------------------------------
LEARNING_RATE = 0.1
MAX_ITERS = 100_000
LOSS_TOL = 1e-10   # stop when the change in loss between consecutive iterations is below this value
PROGRESS_EVERY = 10

SHOW_PLOTS = True           # use --no-show on a machine without a display

# --- Outputs (written under STAGE_ROOT) ----------------------------------------
TRAINING_TABLE_NAME = "HandStandMove_linear_training_table.csv"
MODEL_PARAMETERS_NAME = "HandStandMove_affine_model_parameters.txt"
PARAM_DECIMALS = 12             # decimal places written for B, C, B_inverse
SINGULAR_DET_TOL = 1e-12        # |det(B)| below this: B is treated as singular, no inverse
LOSS_PLOT_NAME = "HandStandMove_GD_loss.png"
MEASURED_VS_PREDICTED_PLOT_NAME = "HandStandMove_measured_vs_predicted.png"
ERROR_VS_PREDICTED_PLOT_NAME = "HandStandMove_error_vs_predicted.png"

POINT_COLOR = "#2a78d6"         # scatter points (one point per stage)
REF_LINE_COLOR = "#52514e"      # y = x and error = 0 reference lines

INPUT_NAMES = ["vx", "vy", "vyaw"]
INPUT_UNITS = ["m/s", "m/s", "rad/s"]
OUTPUT_NAMES = ["delta_x_body", "delta_y_body", "delta_theta"]
OUTPUT_UNITS = ["m", "m", "rad"]


class StageError(Exception):
    """A stage that cannot be used; it is reported and skipped."""


# =============================================================================
# FILE PATHS
# =============================================================================
def stage_base_csv(n):
    return STAGE_ROOT / f"Stage_{n}_HS&Move" / f"Tf_handstand_stage_HandStandMove_Trial_{n}.csv"


def trial_dir(n):
    return TRIAL_ROOT / f"Trial_{n}"


# =============================================================================
# COMMAND INPUT  [vx, vy, vyaw]  (from HandStandMove/Trial_N)
# =============================================================================
def find_command_csv(n):
    exact = trial_dir(n) / CMD_FILE_TEMPLATE.format(n=n)

    if not exact.is_file():
        raise StageError(f"command CSV not found: {exact}")

    print(f"  [NOTE] using {exact.name}")
    return exact


def read_command(n):
    """
    Return (vx, vy, vyaw) for Trial_N from its command CSV.

    The command CSV may be an event log with several rows, so each velocity
    column must contain exactly ONE distinct numeric value (blank rows ignored).
    Missing columns or conflicting values stop this stage with an explicit error.
    """
    path = find_command_csv(n)
    cmd = pd.read_csv(path)
    print(f"  [NOTE] read {len(cmd)} row(s) from {path.name}")

    ###---- printing the trial_number column if it exists, and check that it matches n ----###
    if "trial_number" in cmd.columns:
        nums = set(pd.to_numeric(cmd["trial_number"], errors="coerce").dropna().astype(int))
        print(f"  [NOTE] trial_number column has values {sorted(nums)}")
        if nums and nums != {n}:
            raise StageError(f"{path.name} has trial_number {sorted(nums)}, expected {n}")

    values = []
    for name in INPUT_NAMES:
        ##--It goes through the candidates one at a time and only hands back (yields) a candidate c if that exact string is one of the CSV's column 
        # headers (cmd.columns).------###

        col = next((c for c in CMD_COLUMN_CANDIDATES[name] if c in cmd.columns), None)

        ##--ext(generator, None) asks the generator for its first item:----###
        ##--candidate matched, next returns that string. The generator stops there and doesn't check the remaining candidates.----#####
        if col is None:
            raise StageError(f"{path.name}: no column for '{name}' (accepted names "
                             f"{CMD_COLUMN_CANDIDATES[name]}); actual columns are {list(cmd.columns)}. "
                             "Add the correct name to CMD_COLUMN_CANDIDATES.")
        vals = pd.to_numeric(cmd[col], errors="coerce").dropna().unique()
        if len(vals) == 0:
            raise StageError(f"{path.name}: column '{col}' has no numeric value")
        if len(vals) > 1:
            raise StageError(f"{path.name}: column '{col}' has several different values "
                             f"{sorted(vals.tolist())}; which one was commanded is ambiguous")
        values.append(float(vals[0]))
    print(f"  [NOTE] command = [{values[0]:.4f}, {values[1]:.4f}, {values[2]:.4f}]")
    return tuple(values), path


# =============================================================================
# TARGET  [delta_x_body, delta_y_body, delta_theta]  (from the filtered Base CSV)
# =============================================================================
def wrap_angle(angle):
    """Map an angle (scalar or array, radians) into [-pi, pi]."""
    return np.arctan2(np.sin(angle), np.cos(angle))


def read_initial_and_final_pose(n):
    """
    pose_initial = first row, pose_final = last row of the filtered Base CSV.
    No searching inside the trajectory; the rows are only checked to be usable.
    """
    path = stage_base_csv(n)
    if not path.is_file():
        raise StageError(f"Base CSV not found: {path}")
    df = pd.read_csv(path)

    missing = [c for c in POSE_COLS + [TIME_COL] if c not in df.columns]
    if missing:
        raise StageError(f"{path.name}: missing column(s) {missing}")
    if len(df) < 2:
        raise StageError(f"{path.name}: only {len(df)} row(s)")
    ##---- still a sanity check to find whether the recorded time in the "common_time_ns" column is in increasing order. If not, the first and last rows would not correspond to the initial and final poses.----###
    if not df[TIME_COL].is_monotonic_increasing:
        raise StageError(f"{path.name}: rows are not in time order, so first/last row "
                         "would not be the initial/final pose")

    pose_initial = df.iloc[0][POSE_COLS].to_numpy(dtype=float)
    pose_final = df.iloc[-1][POSE_COLS].to_numpy(dtype=float)
    if not (np.isfinite(pose_initial).all() and np.isfinite(pose_final).all()):
        raise StageError(f"{path.name}: first or last row has a missing pose value")
    return pose_initial, pose_final


def compute_body_frame_displacement(pose_initial, pose_final):
    """
    Relative SE(2) displacement from the initial to the final pose, with the
    translation rotated into the INITIAL robot body frame from world frame.

        [dx_body]   [ cos(th0)  sin(th0)] [dx_world]
        [dy_body] = [-sin(th0)  cos(th0)] [dy_world]      (= R(th0)^T * d_world)

    The yaw change is wrapped into [-pi, pi] so a crossing of +/-pi in the
    floor-frame yaw does not appear as a ~2*pi jump.
    """
    x0, y0, th0 = pose_initial
    x1, y1, th1 = pose_final

    dx_world = x1 - x0
    dy_world = y1 - y0

    c, s = np.cos(th0), np.sin(th0)
    dx_body = c * dx_world + s * dy_world
    dy_body = -s * dx_world + c * dy_world
    dtheta = float(wrap_angle(th1 - th0))

    return {
        "delta_x_world": float(dx_world),
        "delta_y_world": float(dy_world),
        "delta_x_body": float(dx_body),
        "delta_y_body": float(dy_body),
        "delta_theta": dtheta,
    }


# =============================================================================
# TRAINING TABLE
# =============================================================================
def build_dataset():
    """One row per usable stage. Returns (table, skipped list of (stage, reason))."""
    rows, skipped = [], []
    for n in STAGE_NUMBERS:
        try:
            (vx, vy, vyaw), _ = read_command(n)
            p0, p1 = read_initial_and_final_pose(n)
        except StageError as exc:
            print(f"[WARNING] Stage {n} skipped: {exc}")
            skipped.append((n, str(exc)))
            continue
        except Exception as exc:          # unreadable file etc. -- report, do not hide
            print(f"[WARNING] Stage {n} skipped ({type(exc).__name__}): {exc}")
            skipped.append((n, f"{type(exc).__name__}: {exc}"))
            continue

        d = compute_body_frame_displacement(p0, p1)
        print(f"Stage {n:2d} | V = [{vx:.4f}, {vy:.4f}, {vyaw:.4f}]"
              f" | initial pose = [{p0[0]:.4f}, {p0[1]:.4f}, {p0[2]:.4f}]"
              f" | final pose = [{p1[0]:.4f}, {p1[1]:.4f}, {p1[2]:.4f}]"
              f" | Delta_world = [{d['delta_x_world']:+.4f}, {d['delta_y_world']:+.4f}]"
              f" | Delta_body = [{d['delta_x_body']:+.4f}, {d['delta_y_body']:+.4f}, "
              f"{d['delta_theta']:+.4f}]")

        rows.append({
            "trial_number": n, "vx": vx, "vy": vy, "vyaw": vyaw,
            "x_initial": p0[0], "y_initial": p0[1], "yaw_initial": p0[2],
            "x_final": p1[0], "y_final": p1[1], "yaw_final": p1[2],
            "delta_x_world": d["delta_x_world"], "delta_y_world": d["delta_y_world"],
            "delta_x_body": d["delta_x_body"], "delta_y_body": d["delta_y_body"],
            "delta_theta": d["delta_theta"],
        })
    return pd.DataFrame(rows), skipped


# =============================================================================
# BATCH GRADIENT DESCENT
# =============================================================================
def fit_gradient_descent(V, Y):
    """
    Fit  Y_hat = V @ B.T + C  by batch gradient descent.

    The three command channels have different ranges, so descent runs on
    standardized inputs  Vs = (V - mean) / std  with parameters W (3x3), b (3):

        Y_hat = Vs @ W.T + b
        loss  = (1/N) * sum_n || y_hat_n - y_n ||^2

    With standardized inputs a fixed step of 0.1 is always stable: the loss
    curvature is at most 6 here (2 x the largest eigenvalue of [Vs, 1]^T [Vs, 1] / N,
    which is at most 3), and any step below 2/6 = 0.33 converges.

    Converting back to physical units (so B, C act on vx, vy, vyaw directly):
        Y_hat = ((V - mean)/std) @ W.T + b  =  V @ (W/std).T + (b - (W/std) @ mean)
        =>  B = W / std  (column j divided by std_j),   C = b - B @ mean
    """
    N = V.shape[0]
    mean = V.mean(axis=0)
    std = V.std(axis=0)
    ###---- check for any command column that never varies in this data, and warn that its coefficient 
    # cannot be learned (set to 0) ----###
    if np.any(std < 1e-12):
        const = [INPUT_NAMES[j] for j in np.flatnonzero(std < 1e-12)]
        print(f"[WARNING] command column(s) {const} never vary in this data; "
              "their coefficients cannot be learned (set to 0)")

    ###--- replace any std < 1e-12 with 1.0 to avoid divide-by-zero, so the corresponding W column is learned as 0.---###
    std = np.where(std < 1e-12, 1.0, std)

    ##---normalization of the input data to have zero mean and unit variance. This is important for gradient descent to converge properly, especially when 
    # the input features have different scales.---###
    Vs = (V - mean) / std

    W = np.zeros((3, 3))
    b = np.zeros(3)
    loss_history = []

    previous_loss = None

    for iteration in range(1, MAX_ITERS + 1):
        Y_hat = Vs @ W.T + b                       # prediction            (N x 3)
        error = Y_hat - Y                          # prediction error      (N x 3)
        loss = float(np.sum(error ** 2) / N)       # mean summed squared error
        loss_history.append(loss)

        grad_W = (2.0 / N) * (error.T @ Vs)        # d loss / d W          (3 x 3)
        grad_b = (2.0 / N) * error.sum(axis=0)     # d loss / d b          (3,)

        ##--- reporting the loss every PROGRESS_EVERY iterations, and also on the first iteration. This is useful for monitoring convergence.---###
        if iteration == 1 or iteration % PROGRESS_EVERY == 0:
            print(f"  iteration {iteration:6d} | loss = {loss:.10f}")

        if previous_loss is not None and abs(previous_loss - loss) < LOSS_TOL:
            print(f"  iteration {iteration:6d} | loss = {loss:.10f} (converged)")
            break                                   # successive loss change is below LOSS_TOL
        previous_loss = loss

        W = W - LEARNING_RATE * grad_W
        b = b - LEARNING_RATE * grad_b

    converged = iteration < MAX_ITERS
    B = W / std[None, :]                           # back to physical units
    C = b - B @ mean
    return B, C, np.array(loss_history), iteration, converged


# =============================================================================
# OUTPUT HELPERS
# =============================================================================
def fmt_matrix(M, decimals=5):
    """Readable matrix/vector text."""
    with np.printoptions(precision=decimals, suppress=True, linewidth=120,
                         formatter={"float_kind": lambda x: f"{x: .{decimals}f}"}):
        return np.array2string(np.atleast_1d(M), separator=", ")


def fmt_equation(coeffs, const):
    """'a vx + b vy + c vyaw + d' with proper +/- signs."""
    text = f"{coeffs[0]:.5f} {INPUT_NAMES[0]}"
    for c, n in zip(coeffs[1:], INPUT_NAMES[1:]):
        text += f" {'+' if c >= 0 else '-'} {abs(c):.5f} {n}"
    text += f" {'+' if const >= 0 else '-'} {abs(const):.5f}"
    return text


def fmt_row(values, decimals=PARAM_DECIMALS):
    """Space-separated fixed-point numbers, e.g. ' 0.181790000000 -0.030280000000'."""
    return " ".join(f"{x: .{decimals}f}" for x in values)


def save_model_parameters(B, C, B_inv, path):
    """
    Write the physical-unit B, C (after converting back from the standardized
    W, b) and B_inverse as plain text, one matrix row per line.
    B_inv is None when B is numerically singular; then no inverse is written.
    """
    lines = [
        "HandStandMove affine transition model",
        "",
        "Model:",
        "Delta_p_body = B V + C",
        "",
        "Input order:",
        ", ".join(INPUT_NAMES),
        "",
        "Output order:",
        ", ".join(OUTPUT_NAMES),
        "",
        "B:",
        *[fmt_row(r) for r in B],
        "",
        "C:",
        fmt_row(C),
        "",
        "B_inverse:",
    ]
    if B_inv is None:
        lines.append(f"not saved: B is numerically singular (|det(B)| < {SINGULAR_DET_TOL:g})")
    else:
        lines += [fmt_row(r) for r in B_inv]
    path.write_text("\n".join(lines) + "\n")


def save_loss_plot(loss_history, path):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.semilogy(np.arange(1, len(loss_history) + 1), loss_history, lw=1.5)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Training loss (1/N) \u03a3 ||error||\u00b2")
    ax.set_title(f"HandStandMove affine model: gradient-descent loss (learning rate {LEARNING_RATE})")
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    return fig


##--- Additional Functions additions for Evaluating the Model -----#######
def evaluate_model(V, Y, B, C):
    Y_hat = V @ B.T + C
    error = Y - Y_hat

    mse = np.mean(error ** 2, axis=0)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(error), axis=0)

    ss_res = np.sum(error ** 2, axis=0)
    ss_tot = np.sum((Y - Y.mean(axis=0)) ** 2, axis=0)

    with np.errstate(divide="ignore", invalid="ignore"):
        r2 = np.where(
            ss_tot > 0,
            1.0 - ss_res / ss_tot,
            np.nan
        )

    max_abs_error = np.max(np.abs(error), axis=0)

    return {
        "Y_hat": Y_hat,
        "residual": error,              # measured - predicted   (N x 3)
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "max_abs_error": max_abs_error,
    }


# =============================================================================
# DIAGNOSTIC PLOTS
# =============================================================================
def save_measured_vs_predicted_plot(Y, Y_hat, r2, path):
    """One subplot per output: measured (x) vs predicted (y), with the ideal line y = x."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for k, ax in enumerate(axes):
        name, unit = OUTPUT_NAMES[k], OUTPUT_UNITS[k]
        measured, predicted = Y[:, k], Y_hat[:, k]

        ##--- identical X and Y limits, so y = x is the 45-degree diagonal ---###
        lo = min(measured.min(), predicted.min())
        hi = max(measured.max(), predicted.max())
        pad = 0.05 * (hi - lo) if hi > lo else 1e-3
        lo, hi = lo - pad, hi + pad

        ax.plot([lo, hi], [lo, hi], ls="--", lw=1.2, color=REF_LINE_COLOR,
                label="y = x (ideal)", zorder=2)
        ax.scatter(measured, predicted, s=36, color=POINT_COLOR,
                   edgecolors="white", linewidths=0.6, zorder=3)
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel(f"Measured {name} [{unit}]")
        ax.set_ylabel(f"Predicted {name} [{unit}]")
        ax.set_title(f"{name}   (R² = {r2[k]:.4f})")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper left", fontsize=9)
    fig.suptitle(f"HandStandMove affine model: measured vs predicted ({len(Y)} stages)", y=1.0)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(path, dpi=300, bbox_inches="tight")
    return fig


def save_error_vs_predicted_plot(Y_hat, residual, path):
    """One subplot per output: predicted value (x) vs error = measured - predicted (y)."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for k, ax in enumerate(axes):
        name, unit = OUTPUT_NAMES[k], OUTPUT_UNITS[k]

        ax.axhline(0.0, ls="--", lw=1.2, color=REF_LINE_COLOR, zorder=2)
        ax.scatter(Y_hat[:, k], residual[:, k], s=36, color=POINT_COLOR,
                   edgecolors="white", linewidths=0.6, zorder=3)

        ##--- symmetric about 0, so an off-centre cloud is easy to see ---###
        lim = 1.1 * np.max(np.abs(residual[:, k]))
        if lim > 0:
            ax.set_ylim(-lim, lim)
        ax.set_xlabel(f"Predicted {name} [{unit}]")
        ax.set_ylabel(f"Error = measured − predicted [{unit}]")
        ax.set_title(name)
        ax.grid(True, alpha=0.3)
    fig.suptitle(f"HandStandMove affine model: prediction error vs predicted "
                 f"({len(Y_hat)} stages)")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(path, dpi=300, bbox_inches="tight")
    return fig


def make_plots(Y, metrics, loss_history, out_dir):
    """
    Save the loss curve and the two diagnostic figures of the affine model
    already learned by gradient descent (B, C); return the saved paths.

    Nothing is refitted or recomputed here: the predictions and errors are the
    ones from metrics = evaluate_model(V, Y, B, C), i.e.
        Y_hat = V @ B.T + C,     error = Y - Y_hat
    """
    out_dir = Path(out_dir)
    loss_path = out_dir / LOSS_PLOT_NAME
    measured_path = out_dir / MEASURED_VS_PREDICTED_PLOT_NAME
    error_path = out_dir / ERROR_VS_PREDICTED_PLOT_NAME

    save_loss_plot(loss_history, loss_path)
    save_measured_vs_predicted_plot(Y, metrics["Y_hat"], metrics["r2"], measured_path)
    save_error_vs_predicted_plot(metrics["Y_hat"], metrics["residual"], error_path)
    return [loss_path, measured_path, error_path]


# =============================================================================
# MAIN
# =============================================================================
def main():
    parser = argparse.ArgumentParser(description="Fit the Go2 HandStandMove affine transition model.")
    parser.add_argument("--no-show", action="store_true", help="save the plots without opening windows")
    args = parser.parse_args()

    print("=" * 70)
    print("HandStandMove training data (first row -> last row of each filtered stage)")
    print("=" * 70)

    ##---- Step 1: build the training dataset, and report any skipped stages ----###
    table, skipped = build_dataset()

    print(f"\nStages requested : {len(STAGE_NUMBERS)}")
    print(f"Valid stages     : {len(table)}")
    print(f"Skipped          : {len(skipped)}")
    for n, reason in skipped:
        print(f"    Stage {n}: {reason}")
    if len(table) < 4:
        raise SystemExit("[ERROR] need at least 4 valid stages to fit 3 slopes + 1 intercept per output")
    ##------Step 2: save the training table locally----###
    table_path = STAGE_ROOT / TRAINING_TABLE_NAME
    table.to_csv(table_path, index=False)
    print(f"Training table   : {table_path}")

    ##--- Step 3: Collecting the input and output data from the training table into numpy arrays for fitting the model----###
    V = table[INPUT_NAMES].to_numpy(dtype=float)       # N x 3
    Y = table[OUTPUT_NAMES].to_numpy(dtype=float)      # N x 3

    print("\n" + "=" * 70)
    print("Batch gradient descent")
    print("=" * 70)

    ###--- Step 4: Fit the model using gradient descent, and get the learned parameters B and C, along with the loss history and convergence information.----###
    B, C, loss_history, iterations, converged = fit_gradient_descent(V, Y)

    ##--- Step 5: invert B only if it is numerically non-singular, then save B, C, B_inverse ----###
    det_B = np.linalg.det(B)
    B_inv = None if abs(det_B) < SINGULAR_DET_TOL else np.linalg.inv(B)
    params_path = STAGE_ROOT / MODEL_PARAMETERS_NAME
    save_model_parameters(B, C, B_inv, params_path)

    print("\n" + "=" * 70)
    print("HANDSTANDMOVE AFFINE TRANSITION MODEL   Delta_p_body = B V + C")
    print("=" * 70)
    print(f"Number of valid stages = {len(table)}")
    print(f"Learning rate          = {LEARNING_RATE}")
    print(f"Iterations             = {iterations}"
          + ("" if converged else f"   [WARNING] reached MAX_ITERS={MAX_ITERS} before convergence"))
    print(f"Final loss             = {loss_history[-1]:.10f}")
    print("\nB =\n" + fmt_matrix(B))
    print("\nC =\n" + fmt_matrix(C))
    print(f"\ndet(B) = {det_B:.10g}")
    if B_inv is None:
        print(f"\n[WARNING] B is numerically singular (|det(B)| < {SINGULAR_DET_TOL:g}); "
              "B_inverse is NOT computed and NOT saved")
    else:
        print("\nB_inverse =\n" + fmt_matrix(B_inv))
    print("\nRegression equations (vx, vy in m/s; vyaw in rad/s):")
    for k, name in enumerate(OUTPUT_NAMES):
        print(f"  {name:<12s} = {fmt_equation(B[k], C[k])}")
    print("=" * 70)

    ##--- Step 6: Evaluate the model on the training data ----###
    metrics = evaluate_model(V, Y, B, C)
    print("\nModel evaluation on training data:")
    for k, name in enumerate(OUTPUT_NAMES):
        unit = OUTPUT_UNITS[k]
        print(f"  {name} [{unit}]:")
        print(f"    MSE            : {metrics['mse'][k]:.6f} {unit}^2")
        print(f"    RMSE           : {metrics['rmse'][k]:.6f} {unit}")
        print(f"    MAE            : {metrics['mae'][k]:.6f} {unit}")
        print(f"    Max Abs. Error : {metrics['max_abs_error'][k]:.6f} {unit}")
        print(f"    R^2            : {metrics['r2'][k]:.4f}")

    ##--- Step 7: Loss curve + diagnostic plots of the learned model ----###
    plot_paths = make_plots(Y, metrics, loss_history, STAGE_ROOT)
    print("\nSaved figures:")
    for path in plot_paths:
        print(f"  {path}")
    print(f"\nModel parameters (B, C, B_inverse): {params_path}")

    if SHOW_PLOTS and not args.no_show:
        plt.show()
    plt.close("all")


if __name__ == "__main__":
    main()
