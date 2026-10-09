#!/usr/bin/env python3
"""
Extract CANDIDATE HandStand-stage intervals from Unitree Go2 HandStandMove trials
==================================================================================

PRELIMINARY RULE (for RViz inspection only -- NOT a formal HandStand definition):

    start row = FIRST base row with z_floor_m >= 0.50
    end row   = LAST  base row with z_floor_m >= 0.50

    filtered base = EVERY base row from start row to end row, inclusive
                    (rows inside the interval that dip below 0.50 m are KEPT,
                     so the result is one continuous trajectory segment)

    filtered leg  = EVERY leg row whose base_common_time_ns is one of the
                    common_time_ns values of the filtered base rows
                    (all frames, all transformation columns)

No other criterion (pitch, roll, velocity, stationary flags, smoothing,
duration) is used. The z_floor_m >= 0.50 rule and all filtering /
synchronization / validation logic are unchanged from the previous version;
only the trial range and the output location changed.

TRIALS
    Trials 1-65 are processed.

INPUT (read only, never modified)
    /home/unitree-arka/Go2_Skill_Base_Data_Sensor/HandStandMove/Trial_N/
        Tf_stationary_HandStandMove_Trial_N.csv
        Tf_leg_stationary_HandStandMove_Trial_N.csv

OUTPUT (separate dataset, a SIBLING of HandStandMove, not inside it)
    /home/unitree-arka/Go2_Skill_Base_Data_Sensor/Stage_HS_and_Move/Stage_N_HS&Move/
        Tf_handstand_stage_HandStandMove_Trial_N.csv
        Tf_leg_handstand_stage_HandStandMove_Trial_N.csv

    Strict one-to-one mapping: Trial_N -> Stage_N_HS&Move. A Stage folder is
    created only when its trial passes every check, so a failed trial never
    leaves an empty or partial Stage folder behind.

    Note: '&' is a special character in the shell. Quote these paths in a
    terminal, e.g.  ls "Stage_HS_and_Move/Stage_12_HS&Move"
"""

from pathlib import Path

import numpy as np
import pandas as pd


# =============================================================================
# CONFIGURATION
# =============================================================================
# INPUT root: original experimental dataset (read only)
DATASET_ROOT = Path("/home/unitree-arka/Go2_Skill_Base_Data_Sensor/HandStandMove")

# OUTPUT root: new filtered dataset, sibling of DATASET_ROOT
OUTPUT_ROOT = Path("/home/unitree-arka/Go2_Skill_Base_Data_Sensor/Stage_HS_and_Move")

TRIAL_NUMBERS = list(range(1, 66))     # Trials 1-65

Z_THRESHOLD_M = 0.50                   # preliminary rule: z_floor_m >= 0.50

BASE_TIME_COL = "common_time_ns"
BASE_Z_COL = "z_floor_m"
LEG_TIME_COL = "base_common_time_ns"
LEG_FRAME_COL = "frame_name"           # used only for reporting, if present

# Base pose columns printed for debugging at the start/end rows (if present)
DEBUG_POSE_COLS = ["x_floor_m", "y_floor_m", "z_floor_m",
                   "roll_floor_rad", "pitch_floor_rad", "yaw_floor_rad"]


def base_input_path(n):
    return DATASET_ROOT / f"Trial_{n}" / f"Tf_stationary_HandStandMove_Trial_{n}.csv"


def leg_input_path(n):
    return DATASET_ROOT / f"Trial_{n}" / f"Tf_leg_stationary_HandStandMove_Trial_{n}.csv"


# --- input paths: always under DATASET_ROOT ---------------------------------
# (base_input_path / leg_input_path above)

# --- output paths: always under OUTPUT_ROOT ----------------------------------
def stage_name(n):
    return f"Stage_{n}_HS&Move"


def stage_output_dir(n):
    """Trial_N -> OUTPUT_ROOT/Stage_N_HS&Move (strict one-to-one mapping)."""
    return OUTPUT_ROOT / stage_name(n)


def base_output_path(n):
    return stage_output_dir(n) / f"Tf_handstand_stage_HandStandMove_Trial_{n}.csv"


def leg_output_path(n):
    return stage_output_dir(n) / f"Tf_leg_handstand_stage_HandStandMove_Trial_{n}.csv"


def is_inside(path, root):
    """True if `path` is `root` or lies anywhere below it (both resolved).
    Written with relative_to() so it also works on Python 3.8, which has no
    Path.is_relative_to()."""
    try:
        Path(path).resolve().relative_to(Path(root).resolve())
        return True
    except ValueError:
        return False


def check_roots_are_separate():
    """
    Run once before any trial: the output dataset must not live inside the
    input dataset (and vice versa), so generated files can never land in a
    HandStandMove/Trial_N folder. Raises SystemExit if misconfigured.
    """
    if is_inside(OUTPUT_ROOT, DATASET_ROOT):
        raise SystemExit(f"[ERROR] OUTPUT_ROOT {OUTPUT_ROOT} is inside DATASET_ROOT {DATASET_ROOT}; "
                         "refusing to run")
    if is_inside(DATASET_ROOT, OUTPUT_ROOT):
        raise SystemExit(f"[ERROR] DATASET_ROOT {DATASET_ROOT} is inside OUTPUT_ROOT {OUTPUT_ROOT}; "
                         "refusing to run")


class TrialError(Exception):
    """Raised when a trial fails a check; the trial is reported and skipped."""


# =============================================================================
# FILE DISCOVERY
# =============================================================================
def load_trial_csvs(n):
    """
    Checks 1-5: both CSVs exist and contain the required columns.

    The timestamp columns must be INTEGER typed. common_time_ns values are
    ~1.7e18, larger than 2^53, so if pandas ever parsed them as float (which
    happens when a column contains a blank/NaN), neighbouring nanosecond
    timestamps would collapse onto the same float and exact matching between
    the base and leg files would be unreliable. That is reported as an error
    instead of matching on imprecise values.
    """
    base_path, leg_path = base_input_path(n), leg_input_path(n)

    if not base_path.is_file():                                     # check 1
        raise TrialError(f"Base CSV not found: {base_path}")
    if not leg_path.is_file():                                      # check 2
        raise TrialError(f"Leg CSV not found: {leg_path}")

    base_df = pd.read_csv(base_path)
    leg_df = pd.read_csv(leg_path)

    for col in (BASE_TIME_COL, BASE_Z_COL):                         # checks 3, 4
        if col not in base_df.columns:
            raise TrialError(f"column '{col}' missing from {base_path.name}")
    if LEG_TIME_COL not in leg_df.columns:                          # check 5
        raise TrialError(f"column '{LEG_TIME_COL}' missing from {leg_path.name}")

    if not pd.api.types.is_integer_dtype(base_df[BASE_TIME_COL]):
        raise TrialError(f"'{BASE_TIME_COL}' is not integer-typed in {base_path.name} "
                         f"(dtype {base_df[BASE_TIME_COL].dtype}); exact timestamp matching is unsafe")
    if not pd.api.types.is_integer_dtype(leg_df[LEG_TIME_COL]):
        raise TrialError(f"'{LEG_TIME_COL}' is not integer-typed in {leg_path.name} "
                         f"(dtype {leg_df[LEG_TIME_COL].dtype}); exact timestamp matching is unsafe")

    return base_df, leg_df


# =============================================================================
# BASE HANDSTAND INTERVAL DETECTION
# =============================================================================
def find_handstand_interval(base_df):
    """
    Sort by common_time_ns (stable; only reorders if the file is not already in
    time order), then find the FIRST and LAST positions with z_floor_m >= 0.50.

    Returns (base_sorted, start_pos, end_pos, was_resorted).
    start_pos / end_pos are POSITIONS in the time-sorted table (0-based), which
    are what .iloc slicing uses. The original CSV row label is still available
    through base_sorted.index[pos] for reporting.
    """
    was_resorted = not base_df[BASE_TIME_COL].is_monotonic_increasing
    base_sorted = base_df.sort_values(BASE_TIME_COL, kind="stable") if was_resorted else base_df

    z = pd.to_numeric(base_sorted[BASE_Z_COL], errors="coerce").to_numpy()
    qualifying = np.flatnonzero(z >= Z_THRESHOLD_M)   # NaN z never qualifies

    if qualifying.size == 0:                                        # check 6
        raise TrialError(f"no base sample with {BASE_Z_COL} >= {Z_THRESHOLD_M} "
                         f"(max z = {np.nanmax(z):.4f} m)")

    start_pos = int(qualifying[0])
    end_pos = int(qualifying[-1])
    if start_pos > end_pos:                                         # check 7
        raise TrialError(f"start index {start_pos} > end index {end_pos}")

    return base_sorted, start_pos, end_pos, was_resorted


# =============================================================================
# BASE FILTERING
# =============================================================================
def filter_base(base_sorted, start_pos, end_pos):
    """
    Keep EVERY row from start_pos to end_pos inclusive -- NOT only the rows
    with z >= 0.50. Columns are untouched.
    """
    filtered_base = base_sorted.iloc[start_pos:end_pos + 1]
    if filtered_base.empty:                                         # check 8
        raise TrialError("filtered base trajectory is empty")
    return filtered_base


# =============================================================================
# LEG SYNCHRONIZATION / FILTERING
# =============================================================================
def filter_leg(leg_df, filtered_base):
    """
    Keep EVERY leg row (all frames, all columns) whose base_common_time_ns
    exactly equals one of the filtered base common_time_ns values.
    Original row order of the leg file is preserved.
    """
    selected_times = set(filtered_base[BASE_TIME_COL].tolist())
    filtered_leg = leg_df[leg_df[LEG_TIME_COL].isin(selected_times)]
    return filtered_leg, selected_times


# =============================================================================
# VALIDATION
# =============================================================================
def validate_sync(filtered_base, filtered_leg, selected_times):
    """
    Check 9:  every selected base timestamp has leg rows.
    Check 10: no leg timestamp outside the selected base timestamps / interval.
    Returns a dict of diagnostics; raises TrialError on failure.
    """
    leg_times = set(filtered_leg[LEG_TIME_COL].tolist())

    missing_in_leg = selected_times - leg_times                     # check 9
    if missing_in_leg:
        example = sorted(missing_in_leg)[:3]
        raise TrialError(f"{len(missing_in_leg)} selected base timestamp(s) have no leg rows "
                         f"(e.g. {example})")

    extra_in_leg = leg_times - selected_times                       # check 10
    t_start = int(filtered_base[BASE_TIME_COL].iloc[0])
    t_end = int(filtered_base[BASE_TIME_COL].iloc[-1])
    out_of_interval = [t for t in leg_times if t < t_start or t > t_end]
    if extra_in_leg or out_of_interval:
        raise TrialError(f"filtered leg file contains {len(extra_in_leg)} timestamp(s) not in the "
                         f"selected base set and {len(out_of_interval)} outside "
                         f"[{t_start}, {t_end}]")

    # Rows per base timestamp (= number of leg frames recorded at each instant)
    rows_per_time = filtered_leg.groupby(LEG_TIME_COL).size()
    return {
        "unique_leg_times": len(leg_times),
        "rows_per_time_values": sorted(rows_per_time.unique().tolist()),
        "rows_per_time_consistent": rows_per_time.nunique() == 1,
        "n_frames": filtered_leg[LEG_FRAME_COL].nunique() if LEG_FRAME_COL in filtered_leg.columns else None,
    }


# =============================================================================
# SAVING
# =============================================================================
def save_outputs(n, filtered_base, filtered_leg):
    """
    Write the filtered copies into OUTPUT_ROOT/Stage_N_HS&Move.

    Safeguards (all checked BEFORE anything is created or written):
      1. no output path may resolve to either source CSV of this trial;
      2. every output path must lie inside OUTPUT_ROOT;
      3. no output path may lie anywhere inside DATASET_ROOT.
    The Stage folder is created only after these checks pass.
    """
    base_out, leg_out = base_output_path(n), leg_output_path(n)
    sources = {base_input_path(n).resolve(), leg_input_path(n).resolve()}
    for out in (base_out, leg_out):
        if out.resolve() in sources:
            raise TrialError(f"output path would overwrite a source CSV: {out}")
        if not is_inside(out, OUTPUT_ROOT):
            raise TrialError(f"output path is not inside OUTPUT_ROOT {OUTPUT_ROOT}: {out}")
        if is_inside(out, DATASET_ROOT):
            raise TrialError(f"output path lies inside the source dataset {DATASET_ROOT}: {out}")

    stage_output_dir(n).mkdir(parents=True, exist_ok=True)

    # index=False: only the original columns are written (no added index column)
    filtered_base.to_csv(base_out, index=False)
    filtered_leg.to_csv(leg_out, index=False)
    return base_out, leg_out


# =============================================================================
# CONSOLE SUMMARY
# =============================================================================
def print_pose(label, row):
    values = ", ".join(f"{c}={row[c]:.4f}" for c in DEBUG_POSE_COLS if c in row.index)
    if values:
        print(f"  {label:<19s}: {values}")


def process_trial(n):
    """Run all steps for one trial. Returns a short status string for the final table."""
    print("\n" + "=" * 64)
    print(f"Trial {n}")
    print("=" * 64)
    print(f"  Input Trial        : Trial_{n}")
    print(f"  Output Stage       : {stage_name(n)}")

    base_df, leg_df = load_trial_csvs(n)
    base_sorted, start_pos, end_pos, was_resorted = find_handstand_interval(base_df)
    filtered_base = filter_base(base_sorted, start_pos, end_pos)
    filtered_leg, selected_times = filter_leg(leg_df, filtered_base)
    sync = validate_sync(filtered_base, filtered_leg, selected_times)

    start_row = base_sorted.iloc[start_pos]
    end_row = base_sorted.iloc[end_pos]
    n_below = int((pd.to_numeric(filtered_base[BASE_Z_COL], errors="coerce") < Z_THRESHOLD_M).sum())
    dup_base_times = int(filtered_base[BASE_TIME_COL].duplicated().sum())

    if was_resorted:
        print("  [WARNING] base CSV was not in time order; rows were sorted by "
              f"{BASE_TIME_COL} before selection (saved output is time-sorted)")
    if dup_base_times:
        print(f"  [WARNING] {dup_base_times} duplicated {BASE_TIME_COL} value(s) inside the selected interval")

    print(f"  Original base rows : {len(base_df)}")
    print(f"  Start index        : {start_pos}  (original CSV row {base_sorted.index[start_pos]})")
    print(f"  Start z            : {start_row[BASE_Z_COL]:.4f} m")
    print(f"  Start timestamp    : {int(start_row[BASE_TIME_COL])}")
    print_pose("Start pose", start_row)
    print(f"  End index          : {end_pos}  (original CSV row {base_sorted.index[end_pos]})")
    print(f"  End z              : {end_row[BASE_Z_COL]:.4f} m")
    print(f"  End timestamp      : {int(end_row[BASE_TIME_COL])}")
    print_pose("End pose", end_row)
    duration_s = (int(end_row[BASE_TIME_COL]) - int(start_row[BASE_TIME_COL])) / 1e9
    print(f"  Interval duration  : {duration_s:.3f} s")
    print(f"  Filtered base rows : {len(filtered_base)}  "
          f"({n_below} of them below {Z_THRESHOLD_M} m, kept intentionally)")
    print(f"  Filtered leg rows  : {len(filtered_leg)}")
    print(f"  Unique leg timestamps : {sync['unique_leg_times']}")
    print(f"  Base timestamps successfully matched with leg timestamps : True")
    print(f"  Synchronization valid : True")
    if sync["rows_per_time_consistent"]:
        frames = f", {sync['n_frames']} distinct frame names" if sync["n_frames"] is not None else ""
        print(f"  Leg rows per base timestamp : {sync['rows_per_time_values'][0]} "
              f"at every timestamp (consistent{frames})")
    else:
        print(f"  [WARNING] leg rows per base timestamp are NOT consistent: "
              f"observed counts {sync['rows_per_time_values']}")

    base_out, leg_out = save_outputs(n, filtered_base, filtered_leg)
    print(f"  Output directory   : {stage_output_dir(n)}")
    print(f"  Saved: {base_out.name}")
    print(f"  Saved: {leg_out.name}")

    status = "OK" if sync["rows_per_time_consistent"] else "OK (inconsistent leg rows/timestamp)"
    return status, len(filtered_base), len(filtered_leg)


def warn_if_stale_outputs(n):
    """
    A trial that fails now writes nothing. If its Stage folder still holds files
    from an earlier successful run, say so (they are NOT deleted automatically).
    """
    stale = [p.name for p in (base_output_path(n), leg_output_path(n)) if p.is_file()]
    if stale:
        print(f"  [WARNING] {stage_name(n)} still contains output(s) from a previous run: {stale} "
              "-- they do not reflect this run")


def main():
    check_roots_are_separate()
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    print(f"Input root   : {DATASET_ROOT}")
    print(f"Output root  : {OUTPUT_ROOT}")
    print(f"Trials       : {TRIAL_NUMBERS[0]}-{TRIAL_NUMBERS[-1]}")
    print(f"Rule         : first..last base row with {BASE_Z_COL} >= {Z_THRESHOLD_M} m "
          "(preliminary candidate interval, inclusive)")

    results = []
    for n in TRIAL_NUMBERS:
        try:
            status, nb, nl = process_trial(n)
            results.append((n, status, nb, nl))
        except TrialError as exc:
            print(f"  [ERROR] Trial {n} skipped, nothing saved: {exc}")
            warn_if_stale_outputs(n)
            results.append((n, f"ERROR: {exc}", None, None))
        except Exception as exc:  # unexpected (e.g. unreadable CSV) -- report and continue
            print(f"  [ERROR] Trial {n} skipped, nothing saved (unexpected {type(exc).__name__}): {exc}")
            warn_if_stale_outputs(n)
            results.append((n, f"ERROR: {type(exc).__name__}: {exc}", None, None))

    print("\n" + "=" * 64)
    print("SUMMARY")
    print("=" * 64)
    print(f"  {'Trial':<10s}{'Stage':<18s}{'base rows':>10s}{'leg rows':>10s}   status")
    for n, status, nb, nl in results:
        nb_s = "-" if nb is None else str(nb)
        nl_s = "-" if nl is None else str(nl)
        print(f"  {'Trial_' + str(n):<10s}{stage_name(n):<18s}{nb_s:>10s}{nl_s:>10s}   {status}")
    n_ok = sum(1 for r in results if r[1].startswith("OK"))
    failed = [r[0] for r in results if not r[1].startswith("OK")]
    print(f"\n  {n_ok}/{len(results)} trials saved, {len(results) - n_ok} skipped with errors")
    if failed:
        print(f"  Failed trials: {failed}")
    print(f"  Output dataset: {OUTPUT_ROOT}")


if __name__ == "__main__":
    main()