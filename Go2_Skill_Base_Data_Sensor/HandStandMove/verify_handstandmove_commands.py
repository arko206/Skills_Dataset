#!/usr/bin/env python3
"""
verify_handstandmove_commands.py
================================

Read-only verification of the Unitree Go2 HandStandMove dataset.

For every Trial_N folder, this script checks that the velocity command
recorded in

    Trial_N/cmd_HandStandMove_Trial_N.csv

matches row N of the generated parameter file (pandas row index N - 1).

Checks performed per trial
--------------------------
- Trial_N folder exists
- cmd_HandStandMove_Trial_N.csv exists and is not empty
- Required columns are present
- trial_number == N in every row
- mode == "HandStandMove" in every row
- Exactly one HANDSTAND_MOVE_START row and one HANDSTAND_MOVE_END row
- vx, vy, vyaw are valid numbers
- START and END rows contain the same velocities
- START and END velocities match the expected parameter row

All comparisons are numerical (numpy.isclose), so 0.1500 and 0.15 are equal.

SAFETY: this script is strictly READ-ONLY. It only opens files for reading.
It never writes, renames, moves, overwrites or deletes anything.
"""

import os
import sys

import numpy as np
import pandas as pd


# =============================================================================
# CONFIGURATION (edit these paths)
# =============================================================================

PARAMETER_CSV = "/home/unitree-arka/Go2_Skill_Base_Data_Sensor/HandStandMove/Experimental_HS&Move/quadruped_HS&Move_params.csv"

DATASET_ROOT = "/home/unitree-arka/Go2_Skill_Base_Data_Sensor/HandStandMove"

# Numerical tolerance for velocity comparisons.
ATOL = 1e-6

# Expected values inside each command CSV.
EXPECTED_MODE = "HandStandMove"
START_EVENT = "HANDSTAND_MOVE_START"
END_EVENT = "HANDSTAND_MOVE_END"

# Column names.
PARAM_VELOCITY_COLUMNS = ["V_x", "V_y", "V_yaw"]          # in the parameter CSV
CMD_VELOCITY_COLUMNS = ["vx", "vy", "vyaw"]               # in each command CSV
CMD_REQUIRED_COLUMNS = ["event", "mode", "trial_number"] + CMD_VELOCITY_COLUMNS

SEPARATOR = "-" * 50
DOUBLE_SEPARATOR = "=" * 50


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def trial_folder(n):
    """Path to the Trial_N folder."""
    return os.path.join(DATASET_ROOT, f"Trial_{n}")


def command_csv_path(n):
    """Path to Trial_N/cmd_HandStandMove_Trial_N.csv."""
    return os.path.join(trial_folder(n), f"cmd_HandStandMove_Trial_{n}.csv")


def to_number(value):
    """
    Convert a value to float. Returns np.nan for empty or non-numeric entries,
    so they can be reported as invalid instead of crashing the script.
    """
    number = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    return float(number) if pd.notna(number) else np.nan


def format_velocity(number, raw_value=None):
    """Print a velocity with four decimal places, or flag it as invalid."""
    if np.isnan(number):
        return f"INVALID ({raw_value!r})"
    return f"{number:.4f}"


def print_velocities(title, velocities, raw_values=None):
    """Print a block of vx / vy / vyaw values."""
    print(title)
    for axis in CMD_VELOCITY_COLUMNS:
        raw = raw_values[axis] if raw_values is not None else None
        print(f"{axis:<4} = {format_velocity(velocities[axis], raw)}")
    print()


def load_parameter_csv(path):
    """
    Load the generated parameter file and convert V_x, V_y, V_yaw to numbers.
    Stops the script if the file or its columns are unusable, because no
    trial can be verified without it.
    """
    if not os.path.isfile(path):
        sys.exit(f"ERROR: parameter CSV not found: {path}")

    params = pd.read_csv(path)
    params.columns = params.columns.str.strip()

    missing_columns = [c for c in PARAM_VELOCITY_COLUMNS if c not in params.columns]
    if missing_columns:
        sys.exit(f"ERROR: parameter CSV is missing columns: {missing_columns}")

    if params.empty:
        sys.exit("ERROR: parameter CSV contains no rows.")

    # Keep the original text for reporting, and a numeric copy for comparison.
    for column in PARAM_VELOCITY_COLUMNS:
        params[column + "_raw"] = params[column]
        params[column] = pd.to_numeric(params[column], errors="coerce")

    return params


# =============================================================================
# PER-TRIAL VERIFICATION
# =============================================================================

def verify_trial(n, param_row):
    """
    Verify one trial.

    Returns one of: "PASS", "FAIL", "MISSING".
    Prints a detailed report for the trial.
    """
    issues = []          # general problems (mode, trial_number, missing rows ...)
    mismatches = []      # velocity axes that do not match the expected row

    # ---- Expected velocities from row N of the parameter CSV -----------------
    expected = {
        cmd: float(param_row[par]) if pd.notna(param_row[par]) else np.nan
        for cmd, par in zip(CMD_VELOCITY_COLUMNS, PARAM_VELOCITY_COLUMNS)
    }
    expected_raw = {
        cmd: param_row[par + "_raw"]
        for cmd, par in zip(CMD_VELOCITY_COLUMNS, PARAM_VELOCITY_COLUMNS)
    }

    # Optional informative label from primitive_id / sample_idx (not used for mapping).
    label = ""
    if "primitive_id" in param_row.index and "sample_idx" in param_row.index:
        label = f"  [primitive {param_row['primitive_id']}, sample {param_row['sample_idx']}]"

    # ---- 1. Folder and file existence -----------------------------------------
    if not os.path.isdir(trial_folder(n)):
        print(SEPARATOR)
        print(f"Trial {n}: MISSING{label}")
        print(SEPARATOR)
        print(f"Folder not found: {trial_folder(n)}\n")
        return "MISSING"

    csv_path = command_csv_path(n)
    if not os.path.isfile(csv_path):
        print(SEPARATOR)
        print(f"Trial {n}: MISSING{label}")
        print(SEPARATOR)
        print(f"Command CSV not found: {csv_path}\n")
        return "MISSING"

    # ---- 2. Read the command CSV (read-only) ----------------------------------
    try:
        # dtype=str keeps every value as text, so non-numeric entries can be
        # detected and reported instead of being silently converted.
        cmd = pd.read_csv(csv_path, dtype=str, skipinitialspace=True)
    except pd.errors.EmptyDataError:
        cmd = None
    except Exception as error:  # e.g. malformed CSV
        issues.append(f"Could not read command CSV: {error}")
        cmd = None

    if cmd is None or cmd.empty:
        if not issues:
            issues.append("Command CSV is empty")
        print(SEPARATOR)
        print(f"Trial {n}: FAIL{label}")
        print(SEPARATOR)
        print_velocities("Expected:", expected, expected_raw)
        print("Issues:")
        for issue in issues:
            print(f"- {issue}")
        print()
        return "FAIL"

    cmd.columns = cmd.columns.str.strip()
    cmd = cmd.apply(lambda col: col.str.strip() if col.dtype == object else col)

    missing_columns = [c for c in CMD_REQUIRED_COLUMNS if c not in cmd.columns]
    if missing_columns:
        print(SEPARATOR)
        print(f"Trial {n}: FAIL{label}")
        print(SEPARATOR)
        print_velocities("Expected:", expected, expected_raw)
        print("Issues:")
        print(f"- Command CSV is missing columns: {missing_columns}\n")
        return "FAIL"

    # ---- 3. Expected parameter row must itself be valid -----------------------
    for axis in CMD_VELOCITY_COLUMNS:
        if np.isnan(expected[axis]):
            issues.append(f"Parameter CSV row {n} has invalid {axis}: {expected_raw[axis]!r}")

    # ---- 4. trial_number == N in every row ------------------------------------
    for i, raw_trial in enumerate(cmd["trial_number"]):
        value = to_number(raw_trial)
        if np.isnan(value) or not np.isclose(value, n, atol=ATOL):
            issues.append(f"Incorrect trial_number in row {i + 1}: {raw_trial!r} (expected {n})")

    # ---- 5. mode == "HandStandMove" in every row ------------------------------
    for i, raw_mode in enumerate(cmd["mode"]):
        if raw_mode != EXPECTED_MODE:
            issues.append(f"Incorrect mode in row {i + 1}: {raw_mode!r} (expected {EXPECTED_MODE!r})")

    # ---- 6. Locate START and END rows -----------------------------------------
    event_rows = {}
    for event in (START_EVENT, END_EVENT):
        matches = cmd[cmd["event"] == event]
        if len(matches) == 0:
            issues.append(f"Missing {event} row")
        else:
            if len(matches) > 1:
                issues.append(f"{len(matches)} {event} rows found (expected 1); using the first")
            event_rows[event] = matches.iloc[0]

    other_events = sorted(set(cmd["event"].dropna()) - {START_EVENT, END_EVENT})
    if other_events:
        # Not a failure on its own, just reported for information.
        print_note = f"Note: additional event rows present: {other_events}"
    else:
        print_note = None

    # ---- 7. Parse recorded velocities -----------------------------------------
    recorded = {}      # event -> {axis: float}
    recorded_raw = {}  # event -> {axis: original text}
    for event, row in event_rows.items():
        recorded_raw[event] = {axis: row[axis] for axis in CMD_VELOCITY_COLUMNS}
        recorded[event] = {axis: to_number(row[axis]) for axis in CMD_VELOCITY_COLUMNS}
        for axis in CMD_VELOCITY_COLUMNS:
            if np.isnan(recorded[event][axis]):
                issues.append(f"Invalid/non-numeric {axis} in {event}: {row[axis]!r}")

    # ---- 8. START and END must contain the same velocities --------------------
    if START_EVENT in recorded and END_EVENT in recorded:
        for axis in CMD_VELOCITY_COLUMNS:
            start_value = recorded[START_EVENT][axis]
            end_value = recorded[END_EVENT][axis]
            if np.isnan(start_value) or np.isnan(end_value):
                continue  # already reported as invalid
            if not np.isclose(start_value, end_value, atol=ATOL):
                issues.append(f"START and END differ in {axis}: "
                              f"{start_value:.4f} vs {end_value:.4f}")

    # ---- 9. Recorded velocities must match the expected parameter row ---------
    for axis in CMD_VELOCITY_COLUMNS:
        if np.isnan(expected[axis]):
            continue  # already reported
        wrong_events = []
        for event in recorded:
            value = recorded[event][axis]
            if np.isnan(value):
                continue  # already reported as invalid
            if not np.isclose(value, expected[axis], atol=ATOL):
                wrong_events.append("START" if event == START_EVENT else "END")
        if wrong_events:
            mismatches.append(f"{axis}  ({', '.join(wrong_events)})")

    # ---- 10. Print the report --------------------------------------------------
    status = "PASS" if not issues and not mismatches else "FAIL"

    print(SEPARATOR)
    print(f"Trial {n}: {status}{label}")
    print(SEPARATOR)
    print_velocities("Expected:", expected, expected_raw)
    for event in (START_EVENT, END_EVENT):
        if event in recorded:
            print_velocities(f"Recorded {event}:", recorded[event], recorded_raw[event])
        else:
            print(f"Recorded {event}:\nNOT FOUND\n")

    if status == "PASS":
        print("Result: MATCH")
    else:
        if mismatches:
            print("Mismatch:")
            for item in mismatches:
                print(item)
            if issues:
                print()
        if issues:
            print("Issues:")
            for issue in issues:
                print(f"- {issue}")
    if print_note:
        print(print_note)
    print()

    return status


# =============================================================================
# MAIN
# =============================================================================

def main():
    params = load_parameter_csv(PARAMETER_CSV)
    expected_trials = len(params)

    if not os.path.isdir(DATASET_ROOT):
        sys.exit(f"ERROR: dataset root not found: {DATASET_ROOT}")

    passed, failed, missing = [], [], []

    # Trial_N  <->  parameter row index N - 1 (global row order).
    for n in range(1, expected_trials + 1):
        status = verify_trial(n, params.iloc[n - 1])
        if status == "PASS":
            passed.append(n)
        elif status == "FAIL":
            failed.append(n)
        else:
            missing.append(n)

    # Informative only: trial folders beyond the parameter file (e.g. Trial_66).
    extra = []
    for name in os.listdir(DATASET_ROOT):
        if name.startswith("Trial_") and name[len("Trial_"):].isdigit():
            number = int(name[len("Trial_"):])
            if number > expected_trials and os.path.isdir(os.path.join(DATASET_ROOT, name)):
                extra.append(number)

    print(DOUBLE_SEPARATOR)
    print("Verification Summary")
    print(DOUBLE_SEPARATOR)
    print(f"Expected trials : {expected_trials}")
    print(f"Checked trials  : {len(passed) + len(failed)}")
    print(f"Passed          : {len(passed)}")
    print(f"Failed          : {len(failed)}")
    print(f"Missing         : {len(missing)}")
    print()
    print("Failed trials:")
    print(failed)
    print()
    print("Missing trials:")
    print(missing)
    if extra:
        print()
        print(f"Note: trial folders with no matching parameter row: {sorted(extra)}")

    # Non-zero exit code if anything is wrong (useful in shell scripts).
    sys.exit(0 if not failed and not missing else 1)


if __name__ == "__main__":
    main()
