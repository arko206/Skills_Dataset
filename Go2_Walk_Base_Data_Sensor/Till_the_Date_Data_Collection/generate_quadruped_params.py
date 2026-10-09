#!/usr/bin/env python3
"""
generate_quadruped_params.py
============================

Generates the command-velocity parameter set (V_x, V_y, V_yaw) for the quadruped
kinematic trajectory dataset.

For each of the 13 motion primitives (1, 2, 3, 4a, 4b, 5, ..., 12), exactly
N_SAMPLES = 5 combinations are produced, giving 13 x 5 = 65 rows in total.

Sampling rules
--------------
1. PRIMARY axes (the "driving" velocities, e.g. V_x in [0.15, 2.0]):
       np.linspace(low, high, 5)  -> evenly spaced, endpoints included.
       Guarantees uniform coverage of the validated operating range.

2. NOISE axes (the "zero-equivalent" bands, e.g. V_y in (0, 0.08)):
       uniform random samples in the OPEN interval (low, high).
       These keep every command component non-zero (supervisor requirement)
       while staying below the stationary-detection thresholds
       (0.08 m/s linear, 0.10 rad/s angular).

Outputs
-------
- A pandas DataFrame printed to the console.
- 'quadruped_trajectory_params.csv' written to the current working directory.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd


# =============================================================================
# CONFIGURATION
# =============================================================================

# Number of parameter combinations per motion primitive.
N_SAMPLES = 5

# Fixed seed so the random "noise" values are identical on every run.
# This makes the dataset reproducible and citable (report the seed in your paper).
RANDOM_SEED = 42

# Number of decimal places kept for each command. Commands are rounded because
# the robot's velocity interface does not benefit from float64 precision, and
# rounded values make the filenames/logs readable.
ROUND_DECIMALS = 4

# When a primitive has TWO or THREE primary axes (primitives 5-12), zipping the
# linspace arrays directly pairs low-with-low and high-with-high, e.g.
#     (0.15, 0.15), (0.61, 0.61), ..., (2.0, 2.0)
# which only covers the DIAGONAL of the (V_x, V_yaw) space and makes the two
# axes perfectly correlated in the dataset.
#
# If True, the 2nd/3rd primary axes are randomly permuted (seeded). Each axis
# still contains exactly the same 5 linspace values (rule 1 is preserved), but
# the pairing becomes a Latin-hypercube design: uniform marginal coverage AND
# no artificial correlation between axes.
#
# Set to False to reproduce the plain diagonal pairing.
DECORRELATE_PRIMARY_AXES = True

# Output file name.
OUTPUT_CSV = "quadruped_trajectory_params.csv"

# Prefix used for suggested recording filenames, matching the existing
# convention in the dataset (e.g. "Tf_stationary_0_0_0_0_0_15").
FILENAME_PREFIX = "Tf_stationary"


# =============================================================================
# RANGE DEFINITIONS
# =============================================================================

@dataclass(frozen=True)
class AxisRange:
    """A velocity range for one axis, tagged with how it should be sampled."""
    kind: str      # "primary" -> np.linspace, "noise" -> uniform random
    low: float
    high: float

    def __post_init__(self):
        # Guard against typos such as a reversed range.
        if self.kind not in ("primary", "noise"):
            raise ValueError(f"Unknown axis kind '{self.kind}'")
        if not self.low < self.high:
            raise ValueError(f"Range must satisfy low < high, got ({self.low}, {self.high})")


def P(low, high):
    """Primary (linspace) range, closed interval [low, high]."""
    return AxisRange("primary", low, high)


def N(low, high):
    """Noise (uniform random) range, open interval (low, high)."""
    return AxisRange("noise", low, high)


# ---- Validated operating bounds ---------------------------------------------
# Linear, primary
VX_FWD     = P(0.15, 2.0)      # forward walking               [m/s]
VY_LEFT    = P(0.15, 0.8)       # lateral left                   [m/s]
VY_RIGHT   = P(-0.8, -0.15)     # lateral right                  [m/s]

# Angular, primary (asymmetric bounds reflect the measured directional bias:
# ~60 deg in-place turn reached near +2.0 rad/s but already by about -1.8 rad/s)
VYAW_LEFT  = P(0.15, 2.0)      # counter-clockwise (positive)   [rad/s]
VYAW_RIGHT = P(-1.8, -0.15)    # clockwise (negative)           [rad/s]

# Zero-equivalent noise bands (below the stationary-frame thresholds)
V_LIN_NOISE = N(0.0, 0.08)     # linear noise band              [m/s]
V_ANG_NOISE = N(0.0, 0.1)      # angular noise band             [rad/s]


# ---- Motion primitive table -------------------------------------------------
# Each entry: (primitive_id, descriptive name, V_x range, V_y range, V_yaw range)
MOTION_PRIMITIVES = [
    ("1",  "Forward only",                                 VX_FWD,      V_LIN_NOISE, V_ANG_NOISE),
    ("2",  "Lateral left only",                            V_LIN_NOISE, VY_LEFT,     V_ANG_NOISE),
    ("3",  "Lateral right only",                           V_LIN_NOISE, VY_RIGHT,    V_ANG_NOISE),
    ("4a", "Yaw only (positive/left)",                     V_LIN_NOISE, V_LIN_NOISE, VYAW_LEFT),
    ("4b", "Yaw only (negative/right)",                    V_LIN_NOISE, V_LIN_NOISE, VYAW_RIGHT),
    ("5",  "Forward + yaw (left turn)",                    VX_FWD,      V_LIN_NOISE, VYAW_LEFT),
    ("6",  "Forward + yaw (right turn)",                   VX_FWD,      V_LIN_NOISE, VYAW_RIGHT),
    ("7",  "Lateral left + left turn",                     V_LIN_NOISE, VY_LEFT,     VYAW_LEFT),
    ("8",  "Lateral left + right turn",                    V_LIN_NOISE, VY_LEFT,     VYAW_RIGHT),
    ("9",  "Lateral right + left turn",                    V_LIN_NOISE, VY_RIGHT,    VYAW_LEFT),
    ("10", "Lateral right + right turn",                   V_LIN_NOISE, VY_RIGHT,    VYAW_RIGHT),
    ("11", "Forward-left + left turn",                     VX_FWD,      VY_LEFT,     VYAW_LEFT),
    ("12", "Forward-right + right turn",                   VX_FWD,      VY_RIGHT,    VYAW_RIGHT),
]


# =============================================================================
# SAMPLING FUNCTIONS
# =============================================================================

def sample_primary(axis: AxisRange, n: int, rng: np.random.Generator, permute: bool) -> np.ndarray:
    """
    Rule 1: evenly spaced values across the closed range [low, high].

    If `permute` is True the ORDER of the values is shuffled (the values
    themselves are unchanged), which decorrelates this axis from any other
    primary axis in the same primitive.
    """
    values = np.linspace(axis.low, axis.high, n)
    values = np.round(values, ROUND_DECIMALS)
    if permute:
        values = rng.permutation(values)
    return values


def sample_noise(axis: AxisRange, n: int, rng: np.random.Generator) -> np.ndarray:
    """
    Rule 2: uniform random values strictly inside the open range (low, high).

    np.random's uniform draws from the half-open interval [low, high), so a value
    exactly equal to `low` (here 0.0) is theoretically possible. More practically,
    a tiny draw such as 0.00003 would round to 0.0 at ROUND_DECIMALS = 4, which
    would violate the "all components non-zero" requirement.

    We therefore use rejection sampling: any value that is not strictly inside
    (low, high) AFTER rounding is redrawn.
    """
    values = np.round(rng.uniform(axis.low, axis.high, n), ROUND_DECIMALS)
    bad = (values <= axis.low) | (values >= axis.high)
    while np.any(bad):
        values[bad] = np.round(rng.uniform(axis.low, axis.high, bad.sum()), ROUND_DECIMALS)
        bad = (values <= axis.low) | (values >= axis.high)
    return values


def value_to_tag(v: float) -> str:
    """
    Convert a velocity to the filename token convention already used in the
    dataset: '.' becomes '_', trailing zeros are trimmed, but at least one
    decimal digit is kept.
        0.15 -> '0_15'    2.0 -> '2_0'    -0.5 -> '-0_5'    0.0412 -> '0_0412'
    """
    s = f"{v:.{ROUND_DECIMALS}f}".rstrip("0")
    if s.endswith("."):
        s += "0"
    return s.replace(".", "_")


# =============================================================================
# DATASET GENERATION
# =============================================================================

def generate_dataset(seed: int = RANDOM_SEED) -> pd.DataFrame:
    """Build the full parameter table for all motion primitives."""
    # A dedicated Generator (numpy's recommended API) gives the same uniform
    # distribution as np.random.uniform, but without touching global random state.
    rng = np.random.default_rng(seed)

    rows = []
    for pid, name, vx_rng, vy_rng, vyaw_rng in MOTION_PRIMITIVES:
        sampled = {}
        primary_axes_seen = 0

        # Sample each axis according to its kind.
        for axis_name, axis in (("V_x", vx_rng), ("V_y", vy_rng), ("V_yaw", vyaw_rng)):
            if axis.kind == "primary":
                # The first primary axis stays in ascending order; any further
                # primary axes are permuted (if enabled) to avoid diagonal pairing.
                permute = DECORRELATE_PRIMARY_AXES and primary_axes_seen > 0
                sampled[axis_name] = sample_primary(axis, N_SAMPLES, rng, permute)
                primary_axes_seen += 1
            else:
                sampled[axis_name] = sample_noise(axis, N_SAMPLES, rng)

        # Assemble one row per sample.
        for i in range(N_SAMPLES):
            vx, vy, vyaw = sampled["V_x"][i], sampled["V_y"][i], sampled["V_yaw"][i]
            rows.append({
                "primitive_id":   pid,
                "primitive_name": name,
                "sample_idx":     i + 1,
                "V_x":            vx,
                "V_y":            vy,
                "V_yaw":          vyaw,
                # Sampling method per axis, useful when analysing the dataset later.
                "V_x_kind":       vx_rng.kind,
                "V_y_kind":       vy_rng.kind,
                "V_yaw_kind":     vyaw_rng.kind,
                # Suggested recording filename in the existing naming convention.
                "suggested_filename": f"{FILENAME_PREFIX}_{value_to_tag(vx)}_{value_to_tag(vy)}_{value_to_tag(vyaw)}.csv",
            })

    return pd.DataFrame(rows)


# =============================================================================
# VALIDATION
# =============================================================================

def validate_dataset(df: pd.DataFrame) -> None:
    """
    Sanity checks run before the CSV is written. Any failure raises an
    AssertionError so a bad parameter set can never reach the robot.
    """
    # 1. Correct number of rows: 13 primitives x 5 samples.
    expected_rows = len(MOTION_PRIMITIVES) * N_SAMPLES
    assert len(df) == expected_rows, f"Expected {expected_rows} rows, got {len(df)}"

    # 2. Exactly N_SAMPLES per primitive.
    counts = df.groupby("primitive_id").size()
    assert (counts == N_SAMPLES).all(), f"Uneven sample counts:\n{counts}"

    # 3. Every value inside its declared range.
    #    Primary -> closed [low, high]; noise -> open (low, high).
    ranges = {pid: {"V_x": vx, "V_y": vy, "V_yaw": vyaw}
              for pid, _, vx, vy, vyaw in MOTION_PRIMITIVES}
    tol = 10 ** (-ROUND_DECIMALS)  # allow for rounding at the endpoints
    for _, row in df.iterrows():
        for axis_name, axis in ranges[row["primitive_id"]].items():
            v = row[axis_name]
            if axis.kind == "primary":
                ok = (axis.low - tol) <= v <= (axis.high + tol)
            else:
                ok = axis.low < v < axis.high
            assert ok, (f"Primitive {row['primitive_id']} sample {row['sample_idx']}: "
                        f"{axis_name}={v} outside {axis.kind} range ({axis.low}, {axis.high})")

    # 4. Supervisor requirement: no component is exactly zero.
    assert (df[["V_x", "V_y", "V_yaw"]] != 0).all().all(), "Found a zero velocity component"

    # 5. Primary linspace axes must span the full range (both endpoints present).
    for pid, _, vx, vy, vyaw in MOTION_PRIMITIVES:
        sub = df[df["primitive_id"] == pid]
        for axis_name, axis in (("V_x", vx), ("V_y", vy), ("V_yaw", vyaw)):
            if axis.kind == "primary":
                assert np.isclose(sub[axis_name].min(), axis.low, atol=tol), f"{pid}/{axis_name}: low endpoint missing"
                assert np.isclose(sub[axis_name].max(), axis.high, atol=tol), f"{pid}/{axis_name}: high endpoint missing"

    print(f"[OK] Validation passed: {len(df)} rows, {len(counts)} primitives, "
          f"{N_SAMPLES} samples each, all components non-zero and within bounds.")


# =============================================================================
# MAIN
# =============================================================================

def main():
    df = generate_dataset(RANDOM_SEED)
    validate_dataset(df)

    # Print the core table (wide display so rows do not wrap).
    with pd.option_context("display.max_rows", None,
                           "display.width", 200,
                           "display.float_format", f"{{:.{ROUND_DECIMALS}f}}".format):
        print(df[["primitive_id", "primitive_name", "sample_idx", "V_x", "V_y", "V_yaw"]].to_string(index=False))

    # Export to CSV.
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nSaved {len(df)} parameter combinations to '{OUTPUT_CSV}' "
          f"(seed={RANDOM_SEED}, decorrelated={DECORRELATE_PRIMARY_AXES}).")


if __name__ == "__main__":
    main()
