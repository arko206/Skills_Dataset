#!/usr/bin/env python3
"""
Extract the HandStand -> Stand transition from Unitree Go2 HandStandMove trials
================================================================================

Skill sequence of each HandStandMove trial:

    Stand -> Stand-to-HandStand -> HandStandMove -> [HandStand-to-Stand transition]

This script extracts ONLY the bracketed part:

    START = the LAST Base row with z_floor_m >= 0.50 (last HandStand-height sample)
    END   = the very last Base row of the trial (final four-leg standing pose)

    A brief dip below 0.50 m BEFORE the last HandStand-height sample is not a
    boundary: the rule is strictly "last row >= 0.50 -> final row".

    stage Base = every Base row from START through END, inclusive
    stage leg  = every leg row whose base_common_time_ns exactly equals one of
                 the stage Base common_time_ns values (all frames, all columns)

No stationary_candidate, no other state detection, no interpolation, no
nearest-neighbour matching. Timestamps stay int64 (they are ~1e18 and would
lose precision as floats). A trial that never reaches z_floor_m >= 0.50 is
reported and skipped.

INPUT  (read only):  HandStandMove/Trial_N/Tf_stationary_HandStandMove_Trial_N.csv
                     HandStandMove/Trial_N/Tf_leg_stationary_HandStandMove_Trial_N.csv
OUTPUT (new):        Stage_HS_Stand/Stage_HS_Stand_trial_N/Tf_HS_to_Stand_HandStandMove_Trial_N.csv
                     Stage_HS_Stand/Stage_HS_Stand_trial_N/Tf_leg_HS_to_Stand_HandStandMove_Trial_N.csv
Mapping: Trial_N -> Stage_HS_Stand_trial_N. Rows are filtered; columns and values are unchanged.
(Separate from the Stand -> HandStand dataset in Stage_Stand_HS, which is never touched.)
"""

from pathlib import Path

import numpy as np
import pandas as pd

# =============================================================================
# CONFIGURATION
# =============================================================================
DATASET_ROOT = Path("/home/unitree-arka/Go2_Skill_Base_Data_Sensor/HandStandMove")    # read only
OUTPUT_ROOT = Path("/home/unitree-arka/Go2_Skill_Base_Data_Sensor/Stage_HS_Stand")    # written

TRIAL_NUMBERS = list(range(1, 66))      # Trials 1-65
Z_HANDSTAND_M = 0.50                    # START = last row with z_floor_m >= 0.50

BASE_TIME_COL = "common_time_ns"
LEG_TIME_COL = "base_common_time_ns"
Z_COL = "z_floor_m"


class TrialError(Exception):
    """A trial that cannot be extracted; it is reported and skipped (nothing saved)."""


# =============================================================================
# FILE PATHS
# =============================================================================
def base_input_path(n):
    return DATASET_ROOT / f"Trial_{n}" / f"Tf_stationary_HandStandMove_Trial_{n}.csv"


def leg_input_path(n):
    return DATASET_ROOT / f"Trial_{n}" / f"Tf_leg_stationary_HandStandMove_Trial_{n}.csv"


def stage_dir(n):
    return OUTPUT_ROOT / f"Stage_HS_Stand_trial_{n}"


def base_output_path(n):
    return stage_dir(n) / f"Tf_HS_to_Stand_HandStandMove_Trial_{n}.csv"


def leg_output_path(n):
    return stage_dir(n) / f"Tf_leg_HS_to_Stand_HandStandMove_Trial_{n}.csv"


def is_inside(path, root):
    """True if path is root or below it (works on Python 3.8, which lacks is_relative_to)."""
    try:
        Path(path).resolve().relative_to(Path(root).resolve())
        return True
    except ValueError:
        return False


# =============================================================================
# LOADING
# =============================================================================
def load_trial(n):
    base_path, leg_path = base_input_path(n), leg_input_path(n)
    if not base_path.is_file():
        raise TrialError(f"Base CSV not found: {base_path}")
    if not leg_path.is_file():
        raise TrialError(f"Leg CSV not found: {leg_path}")

    base = pd.read_csv(base_path)
    leg = pd.read_csv(leg_path)

    for col in (BASE_TIME_COL, Z_COL):
        if col not in base.columns:
            raise TrialError(f"column '{col}' missing from {base_path.name}")
    if LEG_TIME_COL not in leg.columns:
        raise TrialError(f"column '{LEG_TIME_COL}' missing from {leg_path.name}")

    # Timestamps must have been read as integers; a float column (e.g. caused by a
    # blank cell) would make exact matching of ~1e18 nanosecond values unreliable.
    if not pd.api.types.is_integer_dtype(base[BASE_TIME_COL]):
        raise TrialError(f"'{BASE_TIME_COL}' is not integer-typed ({base[BASE_TIME_COL].dtype})")
    if not pd.api.types.is_integer_dtype(leg[LEG_TIME_COL]):
        raise TrialError(f"'{LEG_TIME_COL}' is not integer-typed ({leg[LEG_TIME_COL].dtype})")
    if len(base) == 0:
        raise TrialError("Base CSV has no rows")
    return base, leg


# =============================================================================
# HANDSTAND -> STAND EXTRACTION
# =============================================================================
def extract_stage(base, leg):
    """
    Extract the HandStand -> Stand transition.

    START = last chronological Base row with z_floor_m >= 0.50
    END   = final chronological Base row.

    Returns (stage_base, stage_leg, start_index, resorted).
    Base is sorted by common_time_ns only if it is not already in order (stable sort).
    """
    resorted = not base[BASE_TIME_COL].is_monotonic_increasing

    if resorted:
        print(f"  [WARNING] Base rows are not in time order; resorting by {BASE_TIME_COL}")
        base_sorted = base.sort_values(BASE_TIME_COL, kind="stable")
    else:
        base_sorted = base

    z = pd.to_numeric(base_sorted[Z_COL], errors="coerce").to_numpy()
    reached = np.flatnonzero(z >= Z_HANDSTAND_M)          # NaN never counts as reached
    if reached.size == 0:
        raise TrialError(f"never reaches {Z_COL} >= {Z_HANDSTAND_M} "
                         f"(max {np.nanmax(z):.4f} m); no HandStand starting pose, trial skipped")
    start_index = int(reached[-1])                       # LAST HandStand-height sample

    stage_base = base_sorted.iloc[start_index:]          # START through the final row, inclusive

    selected_times = set(stage_base[BASE_TIME_COL].tolist())
    stage_leg = leg[leg[LEG_TIME_COL].isin(selected_times)]

    resorted_leg = not stage_leg[LEG_TIME_COL].is_monotonic_increasing
    if resorted_leg:
        print(f"  [WARNING] leg rows are not in time order; resorting by {LEG_TIME_COL}")
        stage_leg = stage_leg.sort_values(LEG_TIME_COL, kind="stable")   # keeps frame order within a timestamp
    
    return stage_base, stage_leg, start_index, resorted


# =============================================================================
# VALIDATION
# =============================================================================
def validate(base, stage_base, stage_leg, start_index):
    """The six required checks. Raises TrialError on the first failure."""
    # 1. final extracted row is exactly the original final (latest) Base row
    final_original = base.loc[base[BASE_TIME_COL].idxmax()]
    if not stage_base.iloc[-1].equals(final_original):
        raise TrialError("check 1 failed: final extracted row differs from the original final Base row")

    z = pd.to_numeric(stage_base[Z_COL], errors="coerce").to_numpy()
    # 2. first extracted row is at HandStand height
    if not z[0] >= Z_HANDSTAND_M:
        raise TrialError(f"check 2 failed: first extracted z = {z[0]} < {Z_HANDSTAND_M}")
    # 3. no later extracted row is still at HandStand height
    if np.any(z[1:] >= Z_HANDSTAND_M):
        raise TrialError("check 3 failed: a later row is still at HandStand height")

    # 4. every selected leg row belongs to the selected Base timestamp set
    selected_times = set(stage_base[BASE_TIME_COL].tolist())
    if not set(stage_leg[LEG_TIME_COL].tolist()) <= selected_times:
        raise TrialError("check 4 failed: leg rows with timestamps outside the selected Base set")

    print(f"Initial z_floor_m : "
        f"{stage_base[Z_COL].iloc[0]:.6f} m "
        f"(row {start_index})"
    )

    # 5. output Base timestamps strictly increasing
    t_base = stage_base[BASE_TIME_COL].to_numpy()
    if not np.all(np.diff(t_base) > 0):
        raise TrialError("check 5 failed: Base timestamps not strictly increasing (duplicates?)")
    # 6. output leg timestamps non-decreasing
    if not np.all(np.diff(stage_leg[LEG_TIME_COL].to_numpy()) >= 0):
        raise TrialError("check 6 failed: leg base_common_time_ns not non-decreasing")


# =============================================================================
# SAVING
# =============================================================================
def save(n, stage_base, stage_leg):
    base_out, leg_out = base_output_path(n), leg_output_path(n)
    sources = {base_input_path(n).resolve(), leg_input_path(n).resolve()}
    for out in (base_out, leg_out):
        if out.resolve() in sources or is_inside(out, DATASET_ROOT) or not is_inside(out, OUTPUT_ROOT):
            raise TrialError(f"unsafe output path, refusing to write: {out}")
    stage_dir(n).mkdir(parents=True, exist_ok=True)
    stage_base.to_csv(base_out, index=False)     # index=False: original columns only
    stage_leg.to_csv(leg_out, index=False)


# =============================================================================
# PER-TRIAL PROCESSING AND REPORT
# =============================================================================
def process_trial(n):
    print(f"\nTrial {n}  ->  {stage_dir(n).name}")
    base, leg = load_trial(n)
    stage_base, stage_leg, start_index, resorted = extract_stage(base, leg)
    validate(base, stage_base, stage_leg, start_index)

    if resorted:
        print("  [WARNING] Base rows were not in time order; sorted by common_time_ns")
    if len(stage_base) == 1:
        print(f"  [WARNING] the final Base row is still at {Z_COL} >= {Z_HANDSTAND_M}; "
              "stage contains a single row (no final four-leg pose)")

    n_leg_times = stage_leg[LEG_TIME_COL].nunique()
    print(f"  Base input rows            : {len(base)}")
    print(f"  Leg input rows             : {len(leg)}")
    print(f"  Initial Base timestamp     : {int(stage_base[BASE_TIME_COL].iloc[0])}")
    print(f"  Final Base timestamp       : {int(stage_base[BASE_TIME_COL].iloc[-1])}"
          f"  (+{(stage_base[BASE_TIME_COL].iloc[-1] - stage_base[BASE_TIME_COL].iloc[0]) / 1e9:.3f} s)")
    print(f"  Initial z_floor_m          : {stage_base[Z_COL].iloc[0]:.6f} m   (row {start_index})")
    print(f"  Final z_floor_m            : {stage_base[Z_COL].iloc[-1]:.6f} m")
    print(f"  Extracted Base rows        : {len(stage_base)}")
    print(f"  Extracted leg rows         : {len(stage_leg)}")
    print(f"  Base timestamps in leg data: {n_leg_times} of {len(stage_base)}")
    if n_leg_times < len(stage_base):
        print(f"  [WARNING] {len(stage_base) - n_leg_times} extracted Base timestamp(s) have no leg rows")
    print("  Checks 1-6                 : OK")

    save(n, stage_base, stage_leg)
    print(f"  Saved: {base_output_path(n).name}, {leg_output_path(n).name}")
    return len(stage_base), len(stage_leg)


def main():
    if is_inside(OUTPUT_ROOT, DATASET_ROOT) or is_inside(DATASET_ROOT, OUTPUT_ROOT):
        raise SystemExit("[ERROR] OUTPUT_ROOT and DATASET_ROOT must be separate folders")
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    print(f"Input  : {DATASET_ROOT}")
    print(f"Output : {OUTPUT_ROOT}")
    print(f"Stage  : last row with {Z_COL} >= {Z_HANDSTAND_M} -> final Base row (inclusive)")

    ok, failed = [], []
    for n in TRIAL_NUMBERS:
        try:
            ok.append((n, *process_trial(n)))
        except TrialError as exc:
            print(f"  [WARNING] Trial {n} skipped, nothing saved: {exc}")
            failed.append((n, str(exc)))
        except Exception as exc:           # unreadable file etc. -- report and continue
            print(f"  [WARNING] Trial {n} skipped, nothing saved ({type(exc).__name__}): {exc}")
            failed.append((n, f"{type(exc).__name__}: {exc}"))

    print("\n" + "=" * 64)
    print(f"SUMMARY: {len(ok)} of {len(TRIAL_NUMBERS)} trials saved, {len(failed)} skipped")
    print("=" * 64)
    for n, reason in failed:
        print(f"  Trial {n}: {reason}")


if __name__ == "__main__":
    main()
