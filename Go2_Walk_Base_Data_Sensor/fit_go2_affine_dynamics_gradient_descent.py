#!/usr/bin/env python3
"""
Unitree Go2 Move() affine single-shot transition model -- refit on the NEW dataset
===================================================================================

Model
-----
For one Move(vx, vy, vyaw) command, the robot's net displacement, expressed in
the robot's INITIAL body frame, is modelled as

    Delta_p_body = B V + C

    V            = [vx, vy, vyaw]^T                      (m/s, m/s, rad/s)
    Delta_p_body = [delta_x_body, delta_y_body, delta_theta]^T   (m, m, rad)
    B            : 3 x 3 matrix
    C            : 3 x 1 vector

Data
----
Every experiment lives in its own directory

    <ROOT>/Fd_{vx}_{vy}_{vyaw}          e.g.  Fd_0.1500_-0.3125_-0.5625

The command [vx, vy, vyaw] is parsed from the directory name (nothing is
hard-coded).  The trajectory is read from the Tf_stationary_*.csv inside that
directory.  Only the columns x_floor_m, y_floor_m, yaw_floor_rad (and, if
present, stationary_candidate) are used.

Relative displacement (NOT absolute floor coordinates)
------------------------------------------------------
    dx_world = x_final - x_initial
    dy_world = y_final - y_initial
    theta_0  = yaw_initial

    delta_x_body =  cos(theta_0) * dx_world + sin(theta_0) * dy_world
    delta_y_body = -sin(theta_0) * dx_world + cos(theta_0) * dy_world
    delta_theta  = wrap_angle(yaw_final - yaw_initial)

i.e. (delta_x_body, delta_y_body) = R(theta_0)^T (dx_world, dy_world).

Fitting
-------
* Batch gradient descent on the MSE loss (as in the previous script).
  Inputs are standardized DURING optimization only (the new command range is
  much larger than before, so an un-scaled learning rate is not trustworthy).
  The learned parameters are converted back so the printed B and C act
  directly on vx [m/s], vy [m/s], vyaw [rad/s].
* A closed-form ordinary-least-squares solution is computed independently and
  used ONLY as a numerical verification. It never replaces the GD result.

Relationship to the previous script (Tf_stationary_<vx>_<vy>_<vyaw>.csv in one flat folder)
--------------------------------------------------------------------------------------------
Kept: the (V, Delta_p_body) convention, the body-frame rotation formulas, the wrap_angle
definition, the loss J = (1/N) sum ||y_hat - y||^2 with gradients (2/N) E^T V and (2/N) sum E,
the forward/residual/loss/gradient/update loop and the trailing plt.show().
Changed: experiments come from Fd_* folders; the fixed learning rate is replaced by
standardized inputs + a curvature-based step size; the stopping test is on the gradient norm
rather than on |loss change| (a tiny loss change does not imply convergence when some
directions in parameter space are flat); NaN rows are dropped before pose selection.
Removed: the previously learned B/C/B_inv constants and every comparison with them.

Three models
------------
Exactly three models are fitted, each with the same fit_gradient_descent (checked against
fit_least_squares) and each evaluated ONLY on its own fitting experiments. There is no
cross-evaluation between the models.

  1. All current experiments (B_N, C_N).
  2. The current experiments without the delta_x over-predictions: the experiments that model 1
     over-predicts in delta_x_body by more than --dx-threshold are left out and the fit is
     repeated on the rest. No folder is moved or deleted (the exclusion is a filter on the
     in-memory table).
  3. Combined: the retained experiments of model 2 plus the old experiments, i.e. every Fd_*
     directory found recursively under <root>/Old_Data_Collection. The old RAW trajectories are
     processed by the same functions as the current ones (no previously learned B/C is used), and
     the delta_x outlier rule is NOT re-applied to them. Every row keeps its dataset_source and
     experiment_path. Repeated commands are kept; only the same recording reached twice (same
     directory, or a byte-identical stationary CSV in a copied folder) is counted once.

This script only READS the experiment CSVs. It writes new analysis outputs
(training tables + plots + delta_x over-prediction CSVs) into the root directory.

Usage
-----
    python3 fit_go2_affine_dynamics_gradient_descent.py
    python3 fit_go2_affine_dynamics_gradient_descent.py --root /some/other/dir
    python3 fit_go2_affine_dynamics_gradient_descent.py --dx-threshold 0.05
    python3 fit_go2_affine_dynamics_gradient_descent.py --old-root /some/dir/with/old/Fd_folders
"""

from __future__ import annotations

import argparse
import hashlib
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# =============================================================================
# CONFIGURATION
# =============================================================================
DEFAULT_ROOT = Path("/home/unitree-arka/Go2_Walk_Base_Data_Sensor")

FOLDER_GLOB = "Fd_*"
STATIONARY_CSV_GLOB = "Tf_stationary_*.csv"   # note: does NOT match Tf_leg_stationary_*

POSE_COLS = ["x_floor_m", "y_floor_m", "yaw_floor_rad"]
STATIONARY_FLAG_COL = "stationary_candidate"
TIME_COL = "common_time_ns"                   # only used to make sure rows are in time order
MIN_VALID_SAMPLES = 2

# The final settled pose is "the last row with stationary_candidate == 1". If the ONLY such
# rows are the pre-motion idle block at the start of the file, that literal rule would return
# a pose identical to the initial one (zero displacement) for a robot that actually moved.
#   True  -> in that situation fall back to the final valid row and print a warning
#   False -> use the literal "last stationary_candidate == 1 row" rule unconditionally
REQUIRE_SETTLE_AFTER_MOTION = True

# What to do if a folder contains MORE than one Tf_stationary_*.csv:
#   "skip"   -> skip that experiment and list it in the skipped report (default, safest)
#   "newest" -> use the most recently modified file, with an explicit warning
MULTIPLE_CSV_POLICY = "skip"

# Gradient descent settings
GD_MAX_ITERS = 200_000
GD_GRAD_TOL = 1e-10          # stop when ||gradient|| (in standardized-parameter space) < tol
GD_PROGRESS_EVERY = 1000     # print the loss every this many iterations
LS_AGREEMENT_TOL = 1e-6      # combined model: GD and least squares "agree" if max |difference| < this

# Show the figures on screen at the end (as in the previous script). All figures are saved to
# disk either way. Use --no-show on a machine without a display (e.g. plain SSH).
SHOW_PLOTS = True

# Inverse-model safety limits
INVERSE_MIN_ABS_DET = 1e-12
INVERSE_MAX_COND = 1e8

# Output names (all written inside the root directory)
TRAINING_TABLE_NAME = "affine_dynamics_training_table_new.csv"
PLOT_LOSS = "affine_new_gd_loss_vs_iteration.png"
PLOT_MEAS_PRED = "affine_new_measured_vs_predicted.png"
PLOT_RESIDUAL = "affine_new_residual_vs_predicted.png"
PLOT_CMD_RESP = "affine_new_command_vs_response.png"
DX_OUTLIERS_NAME = "delta_x_prediction_outliers.csv"

# Outputs of the diagnostic refit without the delta_x over-predictions. The plot names get the
# number of retained experiments, e.g. affine_57_gd_loss.png, so the full-dataset plots are kept.
FILTERED_TABLE_NAME = "affine_dynamics_training_table_without_dx_outliers.csv"
EXCLUDED_DX_NAME = "affine_dynamics_excluded_dx_outliers.csv"
DIAG_PLOT_TEMPLATES = ("affine_{n}_gd_loss.png", "affine_{n}_measured_vs_predicted.png",
                       "affine_{n}_residual_vs_predicted.png", "affine_{n}_command_vs_response.png")

# Combined model = the retained (delta_x-filtered) current experiments + the old experiments.
# The old Fd_* folders are searched recursively under <root>/OLD_DATA_DIRNAME (override with
# --old-root). The table name gets the number of combined experiments, e.g. ..._combined_82.csv.
OLD_DATA_DIRNAME = "Old_Data_Collection"
COMBINED_TABLE_TEMPLATE = "affine_dynamics_training_table_combined_{n}.csv"
COMBINED_PLOT_TEMPLATES = ("affine_{n}_combined_gd_loss.png", "affine_{n}_combined_measured_vs_predicted.png",
                           "affine_{n}_combined_residual_vs_predicted.png",
                           "affine_{n}_combined_command_vs_response.png")

# delta_x_body over-prediction report: list experiments with predicted_dx - measured_dx above
# this value (m). Override with --dx-threshold; 0 lists every over-prediction.
DX_OVERPREDICTION_THRESHOLD = 0.10

INPUT_NAMES = ["vx", "vy", "vyaw"]
OUTPUT_NAMES = ["delta_x_body", "delta_y_body", "delta_theta"]
OUTPUT_UNITS = ["m", "m", "rad"]
INPUT_UNITS = ["m/s", "m/s", "rad/s"]

# Folder-name pattern. Each number: optional sign, digits, optional decimals.
# The three numbers are separated by underscores, so a leading minus sign is
# never confused with a separator.
_NUM = r"[-+]?\d+(?:\.\d+)?"
FOLDER_PATTERN = re.compile(rf"^Fd_(?P<vx>{_NUM})_(?P<vy>{_NUM})_(?P<vyaw>{_NUM})$")


# =============================================================================
# 1. PARSING / FILE DISCOVERY
# =============================================================================
def parse_command_from_folder(folder_name: str) -> Optional[Tuple[float, float, float]]:
    """
    Extract (vx, vy, vyaw) as floats from a folder name like
    'Fd_0.1500_-0.3125_-0.5625'. Returns None if the name does not match.
    """
    match = FOLDER_PATTERN.match(folder_name)
    if match is None:
        return None
    return float(match["vx"]), float(match["vy"]), float(match["vyaw"])


def find_stationary_csv(folder: Path) -> Tuple[Optional[Path], Optional[str]]:
    """
    Locate the stationary trajectory CSV of one experiment folder.

    Returns (csv_path, problem):
        csv_path is None when the experiment must be skipped, and `problem`
        then says why. When `problem` is not None but csv_path is set, it is a
        warning that was resolved explicitly by MULTIPLE_CSV_POLICY.
    """
    candidates = sorted(folder.glob(STATIONARY_CSV_GLOB))

    if len(candidates) == 0:
        return None, f"no file matching {STATIONARY_CSV_GLOB}"

    if len(candidates) == 1:
        return candidates[0], None

    names = [c.name for c in candidates]
    if MULTIPLE_CSV_POLICY == "newest":
        chosen = max(candidates, key=lambda p: p.stat().st_mtime)
        return chosen, (f"{len(candidates)} stationary CSVs {names}; "
                        f"using most recently modified: {chosen.name}")
    return None, f"{len(candidates)} stationary CSVs found {names} (policy='skip')"


# =============================================================================
# 2. POSE SELECTION AND RELATIVE DISPLACEMENT
# =============================================================================
def wrap_angle(angle):
    """Map an angle (scalar or array, radians) into [-pi, pi]."""
    return np.arctan2(np.sin(angle), np.cos(angle))


def choose_start_and_final_pose(df: pd.DataFrame):
    """
    Select the initial pose and the final settled pose of one trajectory.

    1. Rows with NaN / non-finite values in the required pose columns are removed
       BEFORE anything is selected. At least MIN_VALID_SAMPLES rows must remain.
    2. Initial pose  = first valid row.
    3. Final pose    = last valid row with stationary_candidate == 1.
       If there is no such row (or the column is missing), fall back to the last
       valid row and report it in `notes`.
       If the only stationary rows are the initial idle block (robot moved but was
       never flagged stationary again), that is also treated as "no settled row"
       (see REQUIRE_SETTLE_AFTER_MOTION); using it would silently give ~zero
       displacement.

    Returns (pose_initial, pose_final, final_source, notes), where each pose is a
    numpy array [x, y, yaw]. Raises ValueError if the trajectory is unusable.
    """
    missing_cols = [c for c in POSE_COLS if c not in df.columns]
    if missing_cols:
        raise ValueError(f"missing required column(s): {missing_cols}")

    work = df.copy()
    for col in POSE_COLS:
        work[col] = pd.to_numeric(work[col], errors="coerce")

    # Keep rows in time order if a time column is available.
    if TIME_COL in work.columns:
        work[TIME_COL] = pd.to_numeric(work[TIME_COL], errors="coerce")
        work = work.sort_values(TIME_COL, kind="stable", na_position="last")

    valid_mask = np.isfinite(work[POSE_COLS].to_numpy(dtype=float)).all(axis=1)
    work = work.loc[valid_mask].reset_index(drop=True)

    if len(work) < MIN_VALID_SAMPLES:
        raise ValueError(f"only {len(work)} valid pose sample(s); need at least {MIN_VALID_SAMPLES}")

    notes: List[str] = []
    initial_idx = 0
    final_idx = len(work) - 1
    final_source = "last_valid_row"

    if STATIONARY_FLAG_COL in work.columns:
        flags = pd.to_numeric(work[STATIONARY_FLAG_COL], errors="coerce").fillna(0).to_numpy()
        stationary_idx = np.flatnonzero(flags == 1)
        last_row = len(work) - 1

        if stationary_idx.size == 0:
            notes.append("no row with stationary_candidate == 1; fell back to final valid row")
        else:
            candidate = int(stationary_idx[-1])

            # End of the INITIAL stationary block = the contiguous run of flag==1 rows
            # that starts at the first row (robot idling before the command begins).
            initial_block_end = -1
            if flags[0] == 1:
                initial_block_end = 0
                while initial_block_end + 1 <= last_row and flags[initial_block_end + 1] == 1:
                    initial_block_end += 1

            if (REQUIRE_SETTLE_AFTER_MOTION and candidate <= initial_block_end
                    and candidate < last_row):
                # The only stationary rows are the pre-motion idle block. Taking the
                # "last stationary row" would give ~zero displacement for an experiment
                # in which the robot moved and never re-settled, so use the final row.
                notes.append("stationary_candidate == 1 only during the initial idle block "
                             "(never re-settled after moving); fell back to final valid row")
            else:
                final_idx = candidate
                final_source = "stationary_candidate"
    else:
        notes.append(f"column '{STATIONARY_FLAG_COL}' not present; used final valid row")

    pose_initial = work.loc[initial_idx, POSE_COLS].to_numpy(dtype=float)
    pose_final = work.loc[final_idx, POSE_COLS].to_numpy(dtype=float)
    return pose_initial, pose_final, final_source, notes


def compute_body_frame_displacement(pose_initial: np.ndarray, pose_final: np.ndarray) -> Dict[str, float]:
    """
    Relative SE(2) displacement from the initial to the final pose, with the
    translation rotated into the INITIAL robot body frame.

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
# 3. BUILD THE TRAINING TABLE
# =============================================================================
def process_experiment_folder(folder: Path, label: Optional[str] = None) -> Tuple[Optional[dict], Optional[str]]:
    """
    Turn one Fd_{vx}_{vy}_{vyaw} directory into one training row. Shared by the current dataset
    (build_dataset) and the old one (build_old_dataset), so both are processed identically.

    Returns (row, None), or (None, reason) when the experiment must be skipped. Warnings are
    printed with `label` (default: the folder name).
    """
    name = folder.name
    label = label or name

    # --- command from folder name ---
    command = parse_command_from_folder(name)
    if command is None:
        print(f"[WARN] {label}: skipped -- folder name could not be parsed")
        return None, "folder name does not match Fd_{vx}_{vy}_{vyaw}"
    vx, vy, vyaw = command

    # --- CSV selection ---
    csv_path, problem = find_stationary_csv(folder)
    if csv_path is None:
        print(f"[WARN] {label}: skipped -- {problem}")
        return None, problem
    if problem is not None:
        print(f"[WARN] {label}: {problem}")

    # --- read + pose selection ---
    try:
        df = pd.read_csv(csv_path)
        pose_i, pose_f, final_source, notes = choose_start_and_final_pose(df)
    except Exception as exc:  # unreadable / malformed CSV, too few samples, ...
        print(f"[WARN] {label}: skipped -- {csv_path.name}: {exc}")
        return None, f"{csv_path.name}: {exc}"
    for note in notes:
        print(f"[WARN] {label}: {note}")

    disp = compute_body_frame_displacement(pose_i, pose_f)

    return {
        "folder_name": name,
        "csv_file": csv_path.name,
        "vx": vx, "vy": vy, "vyaw": vyaw,
        "x_initial": pose_i[0], "y_initial": pose_i[1], "theta_initial": pose_i[2],
        "x_final": pose_f[0], "y_final": pose_f[1], "theta_final": pose_f[2],
        "delta_x_world": disp["delta_x_world"],
        "delta_y_world": disp["delta_y_world"],
        "delta_x_body": disp["delta_x_body"],
        "delta_y_body": disp["delta_y_body"],
        "delta_theta": disp["delta_theta"],
        "displacement_xy_body": float(np.hypot(disp["delta_x_body"], disp["delta_y_body"])),
        "final_pose_source": final_source,
    }, None


def build_dataset(root: Path):
    """
    Walk every <root>/Fd_* directory and build one training row per valid experiment.

    Returns (table_df, skipped, n_dirs_found) where `skipped` is a list of
    (folder_name, reason). Nothing is dropped silently.
    """
    fd_dirs = sorted(p for p in root.glob(FOLDER_GLOB) if p.is_dir())
    rows: List[dict] = []
    skipped: List[Tuple[str, str]] = []

    for folder in fd_dirs:
        row, problem = process_experiment_folder(folder)
        if row is None:
            skipped.append((folder.name, problem))
        else:
            rows.append(row)

    return pd.DataFrame(rows), skipped, len(fd_dirs)


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_old_dataset(old_root: Path, current_root: Path, current_table: pd.DataFrame):
    """
    Old experiments for the combined model: every directory found RECURSIVELY under `old_root`
    whose name passes parse_command_from_folder, turned into a row by the same
    process_experiment_folder as the current dataset. Each row also gets `experiment_path`,
    the directory it was read from.

    Repeated commands are kept: two runs of the same (vx, vy, vyaw) are separate observations.
    Only the SAME recording reached twice is dropped:
      * the same directory (after resolving symlinks), or a directory of the current dataset;
      * a directory whose stationary CSV is byte-identical to one already accepted, i.e. a copied
        folder (e.g. Old_Data_Collection/Parameters_RS/Fd_x next to Old_Data_Collection/Fd_x).
        The copy nearest to `old_root` is kept. Copies of a current experiment are dropped too.

    Returns (old_table, skipped, duplicates, n_dirs_found); `skipped` and `duplicates` are lists
    of (path relative to old_root, reason). Nothing is dropped silently.
    """
    current_root_resolved = current_root.resolve()

    # The current experiments, by directory and by CSV content, so that an old entry which is
    # really one of them is not counted twice.
    seen_dirs: Dict[Path, str] = {}
    seen_csvs: Dict[str, str] = {}
    for name, csv_file in zip(current_table["folder_name"], current_table["csv_file"]):
        folder = current_root / name
        seen_dirs[folder.resolve()] = str(folder)
        seen_csvs[file_sha256(folder / csv_file)] = str(folder)

    # Shallowest first, so that of two copies the one nearer old_root is the one kept.
    fd_dirs = sorted((p for p in old_root.rglob(FOLDER_GLOB) if p.is_dir()),
                     key=lambda p: (len(p.relative_to(old_root).parts), str(p)))
    rows: List[dict] = []
    skipped: List[Tuple[str, str]] = []
    duplicates: List[Tuple[str, str]] = []

    for folder in fd_dirs:
        rel = str(folder.relative_to(old_root))
        resolved = folder.resolve()
        if resolved in seen_dirs or resolved.parent == current_root_resolved:
            reason = (f"same directory as {seen_dirs[resolved]}" if resolved in seen_dirs
                      else f"{resolved} is a current-dataset folder")
            duplicates.append((rel, reason))
            print(f"[WARN] {rel}: skipped -- {reason}")
            continue

        row, problem = process_experiment_folder(folder, label=rel)
        if row is None:
            skipped.append((rel, problem))
            continue

        digest = file_sha256(folder / row["csv_file"])
        if digest in seen_csvs:
            reason = f"{row['csv_file']} is byte-identical to the one in {seen_csvs[digest]} (copy)"
            duplicates.append((rel, reason))
            print(f"[WARN] {rel}: skipped -- {reason}")
            continue

        seen_dirs[resolved] = str(folder)
        seen_csvs[digest] = str(folder)
        row["experiment_path"] = str(folder)
        rows.append(row)

    return pd.DataFrame(rows), skipped, duplicates, len(fd_dirs)


# =============================================================================
# 4. FITTING
# =============================================================================
def fit_gradient_descent(V: np.ndarray, Y: np.ndarray,
                         max_iters: int = GD_MAX_ITERS, grad_tol: float = GD_GRAD_TOL):
    """
    Batch gradient descent for   Y_hat = V @ B.T + C     (V: N x 3, Y: N x 3).

    Loss (same definition as the previous script: squared VECTOR error, averaged over N):

        J = 1/N * sum_n || y_hat_n - y_n ||_2^2  =  1/N * sum_{n,k} (Y_hat[n,k] - Y[n,k])^2

    Conditioning: the command columns have very different ranges (vx spans
    metres per second, vyaw is small), so gradient descent is run on
    STANDARDIZED inputs  Vs = (V - mu) / sd  with parameters (W, b):

        Y_hat = Vs @ W.T + b

    Learning rate: for a quadratic loss the gradient is Lipschitz with constant
    lambda_max(H), H = 2/N * X^T X, X = [Vs, 1]. Choosing  lr = 1 / lambda_max
    is guaranteed to decrease the loss monotonically (any lr < 2/lambda_max is
    stable), so nothing here depends on the old script's learning rate.

    Conversion back to PHYSICAL units (so B, C act directly on vx, vy, vyaw):

        Y_hat = ((V - mu)/sd) @ W.T + b
              = V @ (W / sd).T + (b - (W / sd) @ mu)
        =>  B = W / sd   (each column j divided by sd_j),   C = b - B @ mu
    """
    N, n_in = V.shape
    n_out = Y.shape[1]

    # ---- standardization statistics (used only inside the optimizer) ----
    mu = V.mean(axis=0)
    sd = V.std(axis=0)
    degenerate = sd < 1e-12
    if degenerate.any():
        print(f"[WARN] constant command column(s) {[INPUT_NAMES[i] for i in np.flatnonzero(degenerate)]}; "
              "their coefficients are not identifiable from this data")
    sd_safe = np.where(degenerate, 1.0, sd)
    Vs = (V - mu) / sd_safe

    # ---- stable learning rate from the Hessian's largest eigenvalue ----
    X = np.hstack([Vs, np.ones((N, 1))])
    hessian = (2.0 / N) * (X.T @ X)
    lr = 1.0 / np.linalg.eigvalsh(hessian).max()

    W = np.zeros((n_out, n_in))
    b = np.zeros(n_out)
    loss_history: List[float] = []
    converged = False
    grad_norm = np.inf
    iterations = 0

    for it in range(1, max_iters + 1):
        # --- forward prediction ---
        Y_hat = Vs @ W.T + b                      # N x 3

        # --- residual calculation (prediction - measurement) ---
        R = Y_hat - Y                             # N x 3

        # --- loss J = (1/N) * sum of squared residuals ---
        loss = float(np.sum(R ** 2) / N)
        loss_history.append(loss)
        if it == 1 or it % GD_PROGRESS_EVERY == 0:
            print(f"  iteration {it:7d} | loss J = {loss:.12f}")

        # --- gradient of W (dJ/dW_kj = 2/N * sum_n R[n,k] * Vs[n,j]) ---
        grad_W = (2.0 / N) * (R.T @ Vs)            # 3 x 3

        # --- gradient of b (dJ/db_k = 2/N * sum_n R[n,k]) ---
        grad_b = (2.0 / N) * R.sum(axis=0)

        # --- stopping condition: gradient (nearly) zero ---
        grad_norm = float(np.sqrt(np.sum(grad_W ** 2) + np.sum(grad_b ** 2)))
        iterations = it
        if grad_norm < grad_tol:
            converged = True
            break

        # --- parameter update ---
        W -= lr * grad_W
        b -= lr * grad_b

    print(f"  stopped at iteration {iterations} | loss J = {loss_history[-1]:.12f}")

    # ---- convert back to physical units ----
    B = W / sd_safe[None, :]
    C = b - B @ mu

    info = {"loss_history": np.array(loss_history), "iterations": iterations,
            "converged": converged, "learning_rate": lr, "final_grad_norm": grad_norm,
            "input_mean": mu, "input_std": sd}
    return B, C, info


def fit_least_squares(V: np.ndarray, Y: np.ndarray):
    """
    Closed-form OLS check: augment V with a column of ones and solve
        [V 1] @ Theta = Y     (Theta is 4 x 3)
    Then B = Theta[:3].T and C = Theta[3]. Verification only.
    """
    X = np.hstack([V, np.ones((V.shape[0], 1))])
    theta, *_ = np.linalg.lstsq(X, Y, rcond=None)
    return theta[:3, :].T, theta[3, :]


# =============================================================================
# 5. EVALUATION
# =============================================================================
def evaluate_model(V: np.ndarray, Y: np.ndarray, B: np.ndarray, C: np.ndarray) -> dict:
    """
    Evaluate Y_hat = V @ B.T + C against measurements Y, per output dimension.

    Residual convention here: residual = measured - predicted.
    R^2 = 1 - SS_res / SS_tot, with SS_tot about the mean of the MEASURED values
    of that output. R^2 can be negative for a model that fits worse than simply
    predicting the mean (this can happen when an old model is evaluated on new data).
    """
    Y_hat = V @ B.T + C
    resid = Y - Y_hat

    mse = np.mean(resid ** 2, axis=0)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(resid), axis=0)

    ss_res = np.sum(resid ** 2, axis=0)
    ss_tot = np.sum((Y - Y.mean(axis=0)) ** 2, axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        r2 = np.where(ss_tot > 0, 1.0 - ss_res / ss_tot, np.nan)

    return {
        "Y_hat": Y_hat, "residual": resid,
        "mse": mse, "rmse": rmse, "mae": mae, "r2": r2,
        "overall_loss": float(np.sum(resid ** 2) / len(resid)),   # J, same definition as the GD loss
        "resid_mean": resid.mean(axis=0),
        "resid_std": resid.std(axis=0, ddof=1) if len(resid) > 1 else np.full(3, np.nan),
        "resid_min": resid.min(axis=0),
        "resid_max": resid.max(axis=0),
    }


def compute_inverse(B: np.ndarray):
    """
    det(B), cond(B) and, if B is safely non-singular, B_inv.
    Returns (det, cond, B_inv_or_None, message).
    """
    det = float(np.linalg.det(B))
    cond = float(np.linalg.cond(B))
    if abs(det) < INVERSE_MIN_ABS_DET or cond > INVERSE_MAX_COND:
        return det, cond, None, (f"B is (numerically) singular: |det|={abs(det):.3e}, cond={cond:.3e}; "
                                 "inverse not computed")
    return det, cond, np.linalg.inv(B), ""


def find_delta_x_prediction_errors(table: pd.DataFrame, V: np.ndarray, Y: np.ndarray,
                                   Y_hat: np.ndarray, residual: np.ndarray, out_dir: Path,
                                   threshold: float = DX_OVERPREDICTION_THRESHOLD) -> pd.DataFrame:
    """
    List the experiments where the model OVER-predicts the forward displacement delta_x_body.

    Residual convention (same as evaluate_model): residual = measured - predicted.
        residual_dx > 0  -> robot went FURTHER than predicted: the model UNDER-predicts
        residual_dx < 0  -> robot fell SHORT of the prediction: the model OVER-predicts

    An experiment is reported when  predicted_dx - measured_dx > threshold  (m), i.e.
    residual_dx < -threshold. threshold = 0 reports every over-prediction (residual_dx < 0).
    Rows are sorted by absolute_error = |measured_dx - predicted_dx|, largest first.
    The result is printed, saved to <out_dir>/DX_OUTLIERS_NAME and returned.
    """
    if threshold < 0:
        raise ValueError(f"threshold must be >= 0 (got {threshold}); a negative value would "
                         "also include under-predicted experiments")

    k = OUTPUT_NAMES.index("delta_x_body")
    measured_dx = Y[:, k]
    predicted_dx = Y_hat[:, k]
    residual_dx = measured_dx - predicted_dx

    # `residual` comes from evaluate_model with the same convention, so the two must agree.
    if not np.allclose(residual_dx, residual[:, k]):
        raise ValueError("residual[:, delta_x_body] != Y - Y_hat; inputs are from different fits")

    # Folder names are taken from the dataset table; its rows are in the same order as V and Y.
    errors = pd.DataFrame({
        "folder_name": table["folder_name"].to_numpy(),
        "vx": V[:, 0], "vy": V[:, 1], "vyaw": V[:, 2],
        "measured_dx": measured_dx,
        "predicted_dx": predicted_dx,
        "residual_dx": residual_dx,
        "absolute_error": np.abs(residual_dx),
    })

    over = errors[predicted_dx - measured_dx > threshold]
    over = over.sort_values("absolute_error", ascending=False, kind="stable").reset_index(drop=True)

    n_over_all = int(np.sum(residual_dx < 0))
    print(f"Over-predicted (residual_dx < 0)               : {n_over_all} of {len(errors)}")
    print(f"Over-predicted by more than {threshold:.3f} m            : {len(over)}")
    if len(over) > 0:
        print(over.to_string(index=False, float_format=lambda x: f"{x: .4f}"))

    path = out_dir / DX_OUTLIERS_NAME
    over.to_csv(path, index=False)
    print(f"delta_x outliers saved : {path}")
    return over


# =============================================================================
# 6. DIAGNOSTIC REFIT WITHOUT THE DELTA_X OVER-PREDICTIONS
# =============================================================================
def fit_model_excluding_dx_outliers(table: pd.DataFrame, dx_outliers: pd.DataFrame) -> Optional[dict]:
    """
    Second model: refit Delta_p_body = B V + C after leaving out the experiments listed in
    `dx_outliers` (the DataFrame returned by find_delta_x_prediction_errors).

    The exclusion list is dx_outliers["folder_name"]; the filter works on a copy of `table`, so
    neither the table nor any folder on disk is changed. The refit uses the same
    fit_gradient_descent as the full-dataset model, checked against fit_least_squares, and is
    evaluated with evaluate_model on the retained experiments only.

    Returns a dict with the filtered/excluded tables and the fit results, or None when there is
    nothing to exclude or too few experiments remain to fit 4 parameters per output.
    """
    excluded_folders = dx_outliers["folder_name"].tolist()
    filtered_table = table[~table["folder_name"].isin(excluded_folders)].copy()
    filtered_table = filtered_table.reset_index(drop=True)

    n_original = len(table)
    n_remaining = len(filtered_table)
    n_excluded = n_original - n_remaining
    if n_excluded != len(excluded_folders):
        raise ValueError(f"{len(excluded_folders)} folder(s) listed for exclusion but {n_excluded} row(s) "
                         "removed; folder names are missing from the table or duplicated")

    print(f"Original experiments : {n_original}")
    print(f"Excluded experiments : {n_excluded}")
    print(f"Remaining experiments: {n_remaining}")

    if n_excluded == 0:
        print("No experiment exceeds the delta_x threshold; the refit would equal the original fit -- skipped")
        return None
    if n_remaining < 4:
        print(f"[WARN] only {n_remaining} experiment(s) left; need at least 4 to fit -- refit skipped")
        return None

    print("\nExcluded experiments (prediction and residual from the original model):")
    shown_cols = ["folder_name", "vx", "vy", "vyaw",
                  "measured_dx", "predicted_dx", "residual_dx", "absolute_error"]
    print(dx_outliers[shown_cols].to_string(index=False, float_format=lambda x: f"{x: .4f}"))

    # Full training rows of the excluded experiments + the original model's delta_x error, in the
    # same order as dx_outliers (largest absolute error first).
    excluded_table = (table[table["folder_name"].isin(excluded_folders)]
                      .merge(dx_outliers[["folder_name", "predicted_dx", "residual_dx", "absolute_error"]],
                             on="folder_name")
                      .sort_values("absolute_error", ascending=False, kind="stable")
                      .reset_index(drop=True))

    V_filtered = filtered_table[INPUT_NAMES].to_numpy(dtype=float)
    Y_filtered = filtered_table[OUTPUT_NAMES].to_numpy(dtype=float)

    # ------------------------------------------------------- gradient descent
    print_section(f"GRADIENT DESCENT ON THE {n_remaining} RETAINED EXPERIMENTS (same fit_gradient_descent)")
    B_filtered, C_filtered, gd_info_filtered = fit_gradient_descent(V_filtered, Y_filtered)
    print(f"learning rate (from Hessian)  : {gd_info_filtered['learning_rate']:.6g}")
    print(f"iterations                    : {gd_info_filtered['iterations']}")
    print(f"converged (grad-norm < {GD_GRAD_TOL:g}) : {gd_info_filtered['converged']}")
    print(f"final gradient norm           : {gd_info_filtered['final_grad_norm']:.3e}")
    if not gd_info_filtered["converged"]:
        print("[WARN] gradient descent hit the iteration limit before meeting the tolerance")

    # ---------------------------------------------- least-squares sanity check
    B_filtered_ls, C_filtered_ls = fit_least_squares(V_filtered, Y_filtered)
    print_section(f"LEAST-SQUARES SANITY CHECK, {n_remaining} EXPERIMENTS (verification only)")
    print("B_filtered_GD =\n" + fmt_matrix(B_filtered, 6))
    print("C_filtered_GD =\n" + fmt_matrix(C_filtered, 6))
    print("B_filtered_LS =\n" + fmt_matrix(B_filtered_ls, 6))
    print("C_filtered_LS =\n" + fmt_matrix(C_filtered_ls, 6))
    print("B_filtered_GD - B_filtered_LS =\n" + fmt_matrix(B_filtered - B_filtered_ls, 3))
    print(f"    max |dB| = {np.max(np.abs(B_filtered - B_filtered_ls)):.3e}")
    print("C_filtered_GD - C_filtered_LS =\n" + fmt_matrix(C_filtered - C_filtered_ls, 3))
    print(f"    max |dC| = {np.max(np.abs(C_filtered - C_filtered_ls)):.3e}")

    # ------------------------------------------------------------- evaluation
    m_filtered = evaluate_model(V_filtered, Y_filtered, B_filtered, C_filtered)
    print_section(f"FIT QUALITY OF THE {n_remaining}-SAMPLE MODEL ON ITS {n_remaining} FITTING EXPERIMENTS")
    print_metrics_block("", m_filtered)

    # ---------------------------------------------------------------- inverse
    det_filtered, cond_filtered, B_inv_filtered, inv_msg_filtered = compute_inverse(B_filtered)
    print_section(f"INVERSE OF THE {n_remaining}-SAMPLE MODEL")
    print(f"det(B_filtered)              = {det_filtered:.6f}")
    print(f"condition_number(B_filtered) = {cond_filtered:.4f}")
    if B_inv_filtered is None:
        print(f"[WARN] {inv_msg_filtered}")
    else:
        print("B_inv_filtered =\n" + fmt_matrix(B_inv_filtered))

    return {
        "filtered_table": filtered_table, "excluded_table": excluded_table,
        "excluded_folders": excluded_folders,
        "n_original": n_original, "n_excluded": n_excluded, "n_remaining": n_remaining,
        "V": V_filtered, "Y": Y_filtered,
        "B": B_filtered, "C": C_filtered, "gd_info": gd_info_filtered,
        "metrics": m_filtered,
        "det": det_filtered, "cond": cond_filtered, "B_inv": B_inv_filtered, "inv_msg": inv_msg_filtered,
    }


# =============================================================================
# 7. PLOTS
# =============================================================================
def make_plots(table: pd.DataFrame, V: np.ndarray, Y: np.ndarray, B: np.ndarray, C: np.ndarray,
               metrics: dict, gd_info: dict, out_dir: Path,
               file_names: Tuple[str, str, str, str] = (PLOT_LOSS, PLOT_MEAS_PRED, PLOT_RESIDUAL, PLOT_CMD_RESP),
               suptitle: Optional[str] = None) -> List[Path]:
    """
    Save the four diagnostic figures and return their paths. `file_names` are the loss,
    measured-vs-predicted, residual and command-vs-response file names; `suptitle`, if given,
    labels every figure (used to tell the diagnostic refit's figures apart).
    """
    saved: List[Path] = []
    Y_hat, resid, r2 = metrics["Y_hat"], metrics["residual"], metrics["r2"]
    loss_name, meas_pred_name, resid_name, cmd_resp_name = file_names

    def save(fig, name: str) -> None:
        if suptitle:
            fig.suptitle(suptitle)
            fig.tight_layout(rect=(0, 0, 1, 0.93))   # leave room: older tight_layout ignores suptitle
        else:
            fig.tight_layout()
        p = out_dir / name
        fig.savefig(p, dpi=200, bbox_inches="tight")
        saved.append(p)

    # ---- 1. GD loss vs iteration ----
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.semilogy(np.arange(1, len(gd_info["loss_history"]) + 1), gd_info["loss_history"], lw=1.5)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Training loss J(B,C)  (mixed units: m$^2$ + m$^2$ + rad$^2$)")
    ax.set_title(f"Gradient-descent loss ({gd_info['iterations']} iterations, "
                 f"lr = {gd_info['learning_rate']:.3g})")
    ax.grid(True, which="both", alpha=0.3)
    save(fig, loss_name)

    # ---- 2. measured vs predicted ----
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    for k, ax in enumerate(axes):
        meas, pred = Y[:, k], Y_hat[:, k]
        lo, hi = min(meas.min(), pred.min()), max(meas.max(), pred.max())
        pad = 0.05 * (hi - lo if hi > lo else 1.0)
        ax.scatter(meas, pred, s=28, alpha=0.8, edgecolor="k", linewidth=0.4)
        ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], "r--", lw=1.2, label="y = x")
        ax.set_xlim(lo - pad, hi + pad)
        ax.set_ylim(lo - pad, hi + pad)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel(f"Measured {OUTPUT_NAMES[k]} ({OUTPUT_UNITS[k]})")
        ax.set_ylabel(f"Predicted {OUTPUT_NAMES[k]} ({OUTPUT_UNITS[k]})")
        ax.set_title(f"{OUTPUT_NAMES[k]}:  $R^2$ = {r2[k]:.4f}")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper left")
    save(fig, meas_pred_name)

    # ---- 3. residual vs predicted ----
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for k, ax in enumerate(axes):
        ax.scatter(Y_hat[:, k], resid[:, k], s=28, alpha=0.8, edgecolor="k", linewidth=0.4)
        ax.axhline(0.0, color="r", ls="--", lw=1.2)
        ax.set_xlabel(f"Predicted {OUTPUT_NAMES[k]} ({OUTPUT_UNITS[k]})")
        ax.set_ylabel(f"Residual = measured - predicted ({OUTPUT_UNITS[k]})")
        ax.set_title(f"{OUTPUT_NAMES[k]}: residual std = {metrics['resid_std'][k]:.4f} {OUTPUT_UNITS[k]}")
        ax.grid(True, alpha=0.3)
    save(fig, resid_name)

    # ---- 4. command vs response (diagnostic only) ----
    # Each panel: the response against its "own" command channel. The other command
    # components are generally nonzero, so scatter about the line is EXPECTED.
    # Dashed line = model slice: B[k,k]*v + C[k] + sum_{j!=k} B[k,j]*mean(v_j).
    v_mean = V.mean(axis=0)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for k, ax in enumerate(axes):
        ax.scatter(V[:, k], Y[:, k], s=28, alpha=0.8, edgecolor="k", linewidth=0.4, label="experiments")
        v_line = np.linspace(V[:, k].min(), V[:, k].max(), 100)
        offset = C[k] + sum(B[k, j] * v_mean[j] for j in range(3) if j != k)
        ax.plot(v_line, B[k, k] * v_line + offset, "r--", lw=1.2,
                label="model slice (other commands at dataset mean)")
        ax.set_xlabel(f"{INPUT_NAMES[k]} ({INPUT_UNITS[k]})")
        ax.set_ylabel(f"{OUTPUT_NAMES[k]} ({OUTPUT_UNITS[k]})")
        ax.set_title(f"{INPUT_NAMES[k]} vs {OUTPUT_NAMES[k]} (diagnostic)")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, loc="best")
    save(fig, cmd_resp_name)

    return saved


# =============================================================================
# 8. PRINTING HELPERS
# =============================================================================
def print_section(title: str) -> None:
    print("\n" + "-" * 60)
    print(title)
    print("-" * 60)


def fmt_matrix(M: np.ndarray, decimals: int = 5) -> str:
    """Readable matrix/vector text."""
    with np.printoptions(precision=decimals, suppress=True, linewidth=120,
                         formatter={"float_kind": lambda x: f"{x: .{decimals}f}"}):
        return np.array2string(np.atleast_1d(M), separator=", ")


def fmt_equation(coeffs: np.ndarray, const: float) -> str:
    """'a vx + b vy + c vyaw + d' with proper +/- signs."""
    text = f"{coeffs[0]:.5f} {INPUT_NAMES[0]}"
    for c, n in zip(coeffs[1:], INPUT_NAMES[1:]):
        text += f" {'+' if c >= 0 else '-'} {abs(c):.5f} {n}"
    text += f" {'+' if const >= 0 else '-'} {abs(const):.5f}"
    return text


def print_metrics_block(title: str, m: dict, loss_label: str = "overall training loss J") -> None:
    print(title)
    for k, name in enumerate(OUTPUT_NAMES):
        print(f"  {name:<13s} MSE = {m['mse'][k]:.6f}  RMSE = {m['rmse'][k]:.6f} {OUTPUT_UNITS[k]}  "
              f"MAE = {m['mae'][k]:.6f} {OUTPUT_UNITS[k]}  R2 = {m['r2'][k]:.4f}")
    print(f"  {loss_label} = (1/N) sum ||residual||^2 = {m['overall_loss']:.6f}")
    print("  residual statistics (measured - predicted):")
    for k, name in enumerate(OUTPUT_NAMES):
        print(f"    {name:<13s} mean = {m['resid_mean'][k]: .6f}  std = {m['resid_std'][k]:.6f}  "
              f"min = {m['resid_min'][k]: .6f}  max = {m['resid_max'][k]: .6f}  ({OUTPUT_UNITS[k]})")


def print_fit_quality(m: dict) -> None:
    """One RMSE / MAE / R2 line per output, as in the final summary."""
    for k, name in enumerate(OUTPUT_NAMES):
        print(f"    {name + ':':<14s} RMSE = {m['rmse'][k]:.6f}, MAE = {m['mae'][k]:.6f}, "
              f"R2 = {m['r2'][k]:.4f}")


def report_dx_outlier_diagnostic(diag: dict, out_dir: Path, threshold: float) -> List[Path]:
    """
    Second half of the diagnostic refit (`diag` from fit_model_excluding_dx_outliers): save the
    retained/excluded tables and the refit's figures, and print the final summary of the refit
    on its own fitting experiments. Returns the saved figure paths.
    """
    n_all, n_out, n_kept = diag["n_original"], diag["n_excluded"], diag["n_remaining"]
    B_f, C_f, B_inv_f = diag["B"], diag["C"], diag["B_inv"]
    m_f = diag["metrics"]

    # ----------------------------------------------------------------- tables
    filtered_path = out_dir / FILTERED_TABLE_NAME
    diag["filtered_table"].to_csv(filtered_path, index=False)
    excluded_path = out_dir / EXCLUDED_DX_NAME
    diag["excluded_table"].to_csv(excluded_path, index=False)
    print(f"\nRetained experiments saved : {filtered_path}")
    print(f"Excluded experiments saved : {excluded_path}")

    # ------------------------------------------------------------------ plots
    saved = make_plots(diag["filtered_table"], diag["V"], diag["Y"], B_f, C_f, m_f, diag["gd_info"], out_dir,
                       file_names=tuple(t.format(n=n_kept) for t in DIAG_PLOT_TEMPLATES),
                       suptitle=f"{n_kept}-sample diagnostic refit "
                                f"({n_out} delta_x over-prediction samples excluded)")
    print(f"\nPlots of the {n_kept}-sample diagnostic refit saved:")
    for p in saved:
        print(f"    {p}")

    # ---------------------------------------------------------- final summary
    print("\n" + "=" * 60)
    print(f"{n_kept}-SAMPLE DIAGNOSTIC REFIT")
    print("DELTA_X OVER-PREDICTION SAMPLES EXCLUDED")
    print("=" * 60)
    print(f"\nOriginal experiments = {n_all}")
    print(f"Excluded experiments = {n_out}")
    print(f"Remaining experiments = {n_kept}")
    print(f"(excluded: predicted_dx - measured_dx > {threshold:.3f} m under the original {n_all}-sample model)")
    print("\nExcluded folders:")
    for name in diag["excluded_folders"]:
        print(f"    {name}")
    print(f"\nB_{n_kept} =\n" + fmt_matrix(B_f))
    print(f"\nC_{n_kept} =\n" + fmt_matrix(C_f))
    print(f"\ndet(B_{n_kept}) = {diag['det']:.6f}")
    print(f"condition_number(B_{n_kept}) = {diag['cond']:.4f}")
    if B_inv_f is not None:
        print(f"\nB_inv_{n_kept} =\n" + fmt_matrix(B_inv_f))
    else:
        print(f"\nB_inv_{n_kept}: not computed ({diag['inv_msg']})")

    print(f"\nFit quality on retained {n_kept} samples:")
    print_fit_quality(m_f)
    print("=" * 60)

    return saved


# =============================================================================
# 9. COMBINED MODEL: RETAINED CURRENT EXPERIMENTS + OLD EXPERIMENTS
# =============================================================================
def fit_combined_model(filtered_table: pd.DataFrame, old_table: pd.DataFrame, current_root: Path) -> dict:
    """
    Third model: fit Delta_p_body = B V + C on `filtered_table` (the retained current experiments,
    from fit_model_excluding_dx_outliers) together with `old_table` (from build_old_dataset), with
    the same fit_gradient_descent, checked against fit_least_squares.

    Both tables are tagged with dataset_source (current_<n> / old_<n>) and experiment_path before
    they are concatenated. The delta_x outlier rule is NOT re-applied to the old experiments.
    The model is evaluated with evaluate_model on the complete combined fitting set only.
    """
    n_current, n_old = len(filtered_table), len(old_table)
    current_source, old_source = f"current_{n_current}", f"old_{n_old}"

    filtered_table = filtered_table.copy()
    filtered_table["experiment_path"] = [str(current_root / name) for name in filtered_table["folder_name"]]
    filtered_table["dataset_source"] = current_source
    old_table = old_table.copy()
    old_table["dataset_source"] = old_source

    combined_table = pd.concat([filtered_table, old_table], ignore_index=True)
    lead = ["dataset_source", "experiment_path"]
    combined_table = combined_table[lead + [c for c in combined_table.columns if c not in lead]]
    n_combined = len(combined_table)

    # Experiments are identified by their (resolved) path, never by folder name alone.
    resolved = combined_table["experiment_path"].map(lambda p: str(Path(p).resolve()))
    if resolved.duplicated().any():
        raise ValueError(f"experiment path(s) included twice: {sorted(set(resolved[resolved.duplicated()]))}")

    print(f"Retained current experiments : {n_current}")
    print(f"Valid old experiments        : {n_old}")
    print(f"Combined experiments         : {n_combined}")
    n_repeated = int(combined_table.duplicated(subset=INPUT_NAMES, keep=False).sum())
    print(f"Experiments sharing their (vx, vy, vyaw) with another one: {n_repeated} "
          "(kept: separate observations)")

    V_c = combined_table[INPUT_NAMES].to_numpy(dtype=float)
    Y_c = combined_table[OUTPUT_NAMES].to_numpy(dtype=float)

    # ------------------------------------------------------- gradient descent
    print_section(f"GRADIENT DESCENT ON THE {n_combined} COMBINED EXPERIMENTS (same fit_gradient_descent)")
    B_c, C_c, gd_info_c = fit_gradient_descent(V_c, Y_c)
    print(f"learning rate (from Hessian)  : {gd_info_c['learning_rate']:.6g}")
    print(f"iterations                    : {gd_info_c['iterations']}")
    print(f"converged (grad-norm < {GD_GRAD_TOL:g}) : {gd_info_c['converged']}")
    print(f"final gradient norm           : {gd_info_c['final_grad_norm']:.3e}")
    if not gd_info_c["converged"]:
        print("[WARN] gradient descent hit the iteration limit before meeting the tolerance")

    # ---------------------------------------------- least-squares sanity check
    B_c_ls, C_c_ls = fit_least_squares(V_c, Y_c)
    max_dB = float(np.max(np.abs(B_c - B_c_ls)))
    max_dC = float(np.max(np.abs(C_c - C_c_ls)))
    print_section(f"LEAST-SQUARES SANITY CHECK, {n_combined} EXPERIMENTS (verification only)")
    print(f"B_{n_combined}_GD =\n" + fmt_matrix(B_c, 6))
    print(f"C_{n_combined}_GD =\n" + fmt_matrix(C_c, 6))
    print(f"B_{n_combined}_LS =\n" + fmt_matrix(B_c_ls, 6))
    print(f"C_{n_combined}_LS =\n" + fmt_matrix(C_c_ls, 6))
    print(f"    max |B_GD - B_LS| = {max_dB:.3e}")
    print(f"    max |C_GD - C_LS| = {max_dC:.3e}")
    if max(max_dB, max_dC) < LS_AGREEMENT_TOL:
        print(f"    gradient descent and least squares agree (both < {LS_AGREEMENT_TOL:g})")
    else:
        print(f"[WARN] gradient descent and least squares differ by more than {LS_AGREEMENT_TOL:g}")

    # ---------------------------------------------------------------- inverse
    det_c, cond_c, B_inv_c, inv_msg_c = compute_inverse(B_c)
    print_section(f"INVERSE OF THE {n_combined}-SAMPLE COMBINED MODEL")
    print(f"det(B_{n_combined})              = {det_c:.6f}")
    print(f"condition_number(B_{n_combined}) = {cond_c:.4f}")
    if B_inv_c is None:
        print(f"[WARN] {inv_msg_c}")
    else:
        print(f"B_inv_{n_combined} =\n" + fmt_matrix(B_inv_c))

    # ------------------------------------------------------------- evaluation
    m_c = evaluate_model(V_c, Y_c, B_c, C_c)
    print_section(f"FIT QUALITY OF THE {n_combined}-SAMPLE COMBINED MODEL ON ITS {n_combined} FITTING EXPERIMENTS")
    print_metrics_block("", m_c)

    return {
        "combined_table": combined_table,
        "n_current": n_current, "n_old": n_old, "n_combined": n_combined,
        "V": V_c, "Y": Y_c,
        "B": B_c, "C": C_c, "gd_info": gd_info_c, "max_dB_ls": max_dB, "max_dC_ls": max_dC,
        "metrics": m_c,
        "det": det_c, "cond": cond_c, "B_inv": B_inv_c, "inv_msg": inv_msg_c,
    }


def report_combined_model(comb: dict, out_dir: Path) -> Tuple[Path, List[Path]]:
    """
    Second half of the combined model (`comb` from fit_combined_model): save the combined training
    table and the combined model's figures, and print the final summary of the model on its own
    fitting experiments. Returns (path of the saved table, saved figure paths).
    """
    n_old, n = comb["n_old"], comb["n_combined"]
    B_c, C_c = comb["B"], comb["C"]

    # ------------------------------------------------------------------ table
    table_path = out_dir / COMBINED_TABLE_TEMPLATE.format(n=n)
    comb["combined_table"].to_csv(table_path, index=False)
    print(f"\nCombined training table saved : {table_path}")

    # ------------------------------------------------------------------ plots
    saved = make_plots(comb["combined_table"], comb["V"], comb["Y"], B_c, C_c, comb["metrics"], comb["gd_info"],
                       out_dir, file_names=tuple(t.format(n=n) for t in COMBINED_PLOT_TEMPLATES),
                       suptitle=f"{n}-sample combined affine model")
    print(f"\nPlots of the {n}-sample combined model saved:")
    for p in saved:
        print(f"    {p}")

    # ---------------------------------------------------------- final summary
    print("\n" + "=" * 60)
    print("COMBINED AFFINE MODEL: CURRENT RETAINED DATA + OLD DATA")
    print("=" * 60)
    print(f"\nCurrent retained samples = {comb['n_current']}")
    print(f"Old valid samples        = {n_old}")
    print(f"Total combined samples   = {n}")
    print(f"\nB_{n} =\n" + fmt_matrix(B_c))
    print(f"\nC_{n} =\n" + fmt_matrix(C_c))
    print(f"\ndet(B_{n}) = {comb['det']:.6f}")
    print(f"condition_number(B_{n}) = {comb['cond']:.4f}")
    if comb["B_inv"] is not None:
        print(f"\nB_inv_{n} =\n" + fmt_matrix(comb["B_inv"]))
    else:
        print(f"\nB_inv_{n}: not computed ({comb['inv_msg']})")
    print(f"\nGD vs least squares: max |dB| = {comb['max_dB_ls']:.3e}, max |dC| = {comb['max_dC_ls']:.3e}")

    print(f"\nFit quality on the {n} combined fitting samples:")
    print_fit_quality(comb["metrics"])
    print("=" * 60)

    return table_path, saved


def run_combined_model_analysis(root: Path, old_root: Path, table: pd.DataFrame, diag: dict) -> None:
    """
    Build the old dataset under `old_root`, fit the combined model on diag["filtered_table"] + the
    old experiments and report it. `table` holds all current experiments; build_old_dataset uses it
    only to recognise an old entry that is really one of them.
    """
    print("\n" + "=" * 60)
    print("COMBINED AFFINE MODEL: RETAINED CURRENT + OLD EXPERIMENTS")
    print("=" * 60)
    if not old_root.is_dir():
        print(f"[WARN] old-data directory does not exist: {old_root} -- combined model skipped")
        return
    if old_root.resolve() == root.resolve():
        print(f"[WARN] old-data directory is the current root {root} -- combined model skipped")
        return

    print(f"Scanning {old_root} recursively for {FOLDER_GLOB} experiments")
    old_table, skipped, duplicates, n_dirs = build_old_dataset(old_root, root, table)

    print_section("OLD DATASET SUMMARY")
    print(f"Fd_* directories found (recursive) : {n_dirs}")
    print(f"Skipped                            : {len(skipped)}")
    for rel, reason in skipped:
        print(f"    - {rel}: {reason}")
    print(f"Same recording already included    : {len(duplicates)}")
    for rel, reason in duplicates:
        print(f"    - {rel}: {reason}")
    print(f"Valid old experiments              : {len(old_table)}")
    if len(old_table) == 0:
        print("[WARN] no valid old experiments -- combined model skipped")
        return
    print("Old experiments used:")
    for path in old_table["experiment_path"]:
        print(f"    {path}")

    print_section("COMBINED DATASET")
    comb = fit_combined_model(diag["filtered_table"], old_table, root)
    report_combined_model(comb, root)


# =============================================================================
# 10. MAIN
# =============================================================================
def main() -> None:
    parser = argparse.ArgumentParser(description="Refit the Go2 affine Move() transition model.")

    ##----- Loading the Folder that has only 65 parameters------------------------------------------######
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT,
                        help=f"directory containing the Fd_* experiment folders (default: {DEFAULT_ROOT})")
    parser.add_argument("--no-show", action="store_true",
                        help="save the figures but do not open plot windows (headless machines)")
    parser.add_argument("--dx-threshold", type=float, default=DX_OVERPREDICTION_THRESHOLD,
                        help="report experiments with predicted - measured delta_x_body above this "
                             f"many metres (default: {DX_OVERPREDICTION_THRESHOLD}; 0 = all over-predictions)")

    ####---- Loading the Folder which has other 20 data samples -------------------------------######
    parser.add_argument("--old-root", type=Path, default=None,
                        help="directory searched recursively for the old Fd_* experiments of the combined "
                             f"model (default: <root>/{OLD_DATA_DIRNAME})")
    args = parser.parse_args()
    root: Path = args.root.expanduser()

    if not root.is_dir():
        raise SystemExit(f"Root directory does not exist: {root}")

    # ------------------------------------------------------------------ data
    print("=" * 60)
    ###---- "Folder_GLOB" is the pattern to search for the experiment folders, e.g., "Fd_*" ----######
    print(f"Scanning {root} for {FOLDER_GLOB} experiments")
    print("=" * 60)
    table, skipped, n_dirs = build_dataset(root)

    print("\n" + "-" * 60)
    print("DATASET SUMMARY")
    print("-" * 60)
    print(f"Fd_* directories found : {n_dirs}")
    print(f"Valid experiments      : {len(table)}")
    print(f"Skipped                : {len(skipped)}")
    for name, reason in skipped:
        print(f"    - {name}: {reason}")

    if len(table) < 4:
        raise SystemExit("Need at least 4 valid experiments to fit 4 parameters per output "
                         "(3 slopes + 1 intercept).")

    table_path = root / TRAINING_TABLE_NAME
    table.to_csv(table_path, index=False)
    print(f"Training table saved   : {table_path}")

    V = table[INPUT_NAMES].to_numpy(dtype=float)        # N x 3
    Y = table[OUTPUT_NAMES].to_numpy(dtype=float)       # N x 3

    # ------------------------------------------------------- gradient descent
    #---- fitting gradient descent on the initial 65 exeriments to get the B and C matrices for the affine model--##
    B_gd, C_gd, gd_info = fit_gradient_descent(V, Y)
    print("\n" + "-" * 60)
    print("GRADIENT DESCENT")
    print("-" * 60)
    print(f"learning rate (from Hessian)  : {gd_info['learning_rate']:.6g}")
    print(f"iterations                    : {gd_info['iterations']}")
    print(f"converged (grad-norm < {GD_GRAD_TOL:g}) : {gd_info['converged']}")
    print(f"final gradient norm           : {gd_info['final_grad_norm']:.3e}")
    if not gd_info["converged"]:
        print("[WARN] gradient descent hit the iteration limit before meeting the tolerance")

    # ---------------------------------------------- least-squares sanity check
    B_ls, C_ls = fit_least_squares(V, Y)
    print("\n" + "-" * 60)
    print("LEAST-SQUARES SANITY CHECK (verification only; GD result is what is used)")
    print("-" * 60)
    print("B_gradient_descent =\n" + fmt_matrix(B_gd, 6))
    print("C_gradient_descent =\n" + fmt_matrix(C_gd, 6))
    print("B_least_squares =\n" + fmt_matrix(B_ls, 6))
    print("C_least_squares =\n" + fmt_matrix(C_ls, 6))
    print("B_gradient_descent - B_least_squares =\n" + fmt_matrix(B_gd - B_ls, 3))
    print(f"    max |dB| = {np.max(np.abs(B_gd - B_ls)):.3e}")
    print("C_gradient_descent - C_least_squares =\n" + fmt_matrix(C_gd - C_ls, 3))
    print(f"    max |dC| = {np.max(np.abs(C_gd - C_ls)):.3e}")

    # ------------------------------------------------------------- evaluation
    m_new = evaluate_model(V, Y, B_gd, C_gd)
    print("\n" + "-" * 60)
    print("FIT QUALITY OF NEW MODEL (gradient-descent B, C) ON THE NEW DATASET")
    print("-" * 60)
    print_metrics_block("", m_new)

    # ---------------------------------------------------------------- inverse
    det_new, cond_new, B_inv_new, inv_msg = compute_inverse(B_gd)
    print("\n" + "-" * 60)
    print("INVERSE MODEL")
    print("-" * 60)
    print(f"det(B)              = {det_new:.6f}")
    print(f"condition_number(B) = {cond_new:.4f}")
    if B_inv_new is None:
        print(f"[WARN] {inv_msg}")
    else:
        print("B_inv =\n" + fmt_matrix(B_inv_new))
        print("Controller form: V = B_inv (Delta_p_body - C)")

    # ------------------------------------------------------------------ plots
    n_all = len(table)
    saved = make_plots(table, V, Y, B_gd, C_gd, m_new, gd_info, root)
    print(f"\nPlots of the {n_all}-sample model saved:")
    for p in saved:
        print(f"    {p}")

    # ---------------------------------------------------------- final summary
    print("\n" + "=" * 60)
    print("NEW GO2 MOVE AFFINE TRANSITION MODEL")
    print("=" * 60)
    print(f"\nNumber of valid experiments = {n_all}")
    print(f"\nB_{n_all} =\n" + fmt_matrix(B_gd))
    print(f"\nC_{n_all} =\n" + fmt_matrix(C_gd))
    print(f"\ndet(B_{n_all}) = {det_new:.6f}")
    print(f"condition_number(B_{n_all}) = {cond_new:.4f}")
    if B_inv_new is not None:
        print(f"\nB_inv_{n_all} =\n" + fmt_matrix(B_inv_new))
    else:
        print(f"\nB_inv_{n_all}: not computed ({inv_msg})")

    print("\nRegression equations:")
    for k, name in enumerate(OUTPUT_NAMES):
        print(f"\n{name} =\n    {fmt_equation(B_gd[k], C_gd[k])}")

    print(f"\nFit quality on all {n_all} samples:")
    for k, name in enumerate(OUTPUT_NAMES):
        print(f"    {name + ':':<14s} RMSE = {m_new['rmse'][k]:.6f}, MAE = {m_new['mae'][k]:.6f}, "
              f"R2 = {m_new['r2'][k]:.4f}")
    print("=" * 60)

    # ----------------------------------------------- delta_x over-predictions
    # Identified exactly once, from the residuals of the model above.
    print("\n" + "-" * 60)
    print("DELTA_X_BODY OVER-PREDICTIONS (predicted - measured > threshold)")
    print("-" * 60)
    dx_outliers = find_delta_x_prediction_errors(table, V, Y, m_new["Y_hat"], m_new["residual"], root,
                                                 threshold=args.dx_threshold)

    # ------------------------ diagnostic refit without the delta_x over-predictions
    # Sensitivity check only: the model above and everything it saved stay as they are.
    print("\n" + "=" * 60)
    print("AFFINE MODEL AFTER DELTA_X OUTLIER EXCLUSION (diagnostic refit)")
    print(f"excluded: predicted_dx - measured_dx > {args.dx_threshold:.3f} m under the "
          f"{len(table)}-sample model")
    print("=" * 60)
    diag = fit_model_excluding_dx_outliers(table, dx_outliers)
    if diag is not None:
        report_dx_outlier_diagnostic(diag, root, args.dx_threshold)

    # ------------------- combined model: retained current experiments + old experiments
    # Additional analysis: the two models above and everything they saved stay as they are.
    if diag is None:
        print("\nCombined model skipped: it is built on the retained experiments of the delta_x refit, "
              "which was not run")
    else:
        old_root = args.old_root.expanduser() if args.old_root is not None else root / OLD_DATA_DIRNAME
        run_combined_model_analysis(root, old_root, table, diag)

    # All figures are already saved. Show them (one call opens all four windows at once).
    if SHOW_PLOTS and not args.no_show:
        plt.show()
    plt.close("all")


if __name__ == "__main__":
    main()
