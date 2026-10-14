#!/usr/bin/env python3
"""
HandStand+Move command comparison: relative Base trajectories (3-D)
====================================================================

Compares HandStand+Move Base trajectories in four comparison groups of four
trials each. No trial numbers are stored in this script: at run time the
terminal asks for the four Trial numbers of every group, in the order

    Pair 1 first, Pair 1 second, Pair 2 first, Pair 2 second

The two trials of a pair share a colour; the first member of each pair is drawn
solid and the second dashed. The script does not classify the motion of any
trial - the legend shows only the trial number and its commanded V.

Each group gives two 3-D figures, X-Y-Z and X-Y-Yaw. The figures are only
DISPLAYED (plt.show()) for interactive inspection; nothing is saved
automatically. Rotate the 3-D views and save the ones you want from the figure
window.

Every selected experiment is identified by its TRIAL NUMBER N. Two files are
read per trial (leg CSVs are not loaded):

    Base trajectory (filtered; Stage_N corresponds to Trial_N):
        <STAGE_ROOT>/Stage_N_HS&Move/Tf_handstand_stage_HandStandMove_Trial_N.csv
    Commanded V = (Vx, Vy, Vyaw), columns vx, vy, vyaw:
        <COMMAND_ROOT>/Trial_N/cmd_HandStandMove_Trial_N.csv

The commanded V is read ONLY from the command CSV (each command column must hold
exactly one distinct numeric value). It is not hard-coded in this script and is
never inferred from the measured trajectory or its velocity columns. A trial
whose Base CSV or command CSV is missing or invalid is reported and skipped.

NORMALIZATION (relative to each trajectory's own first sample x0, y0, z0, theta0)
--------------------------------------------------------------------------------
    dx_world = x - x0,   dy_world = y - y0
    X_body   =  cos(theta0) * dx_world + sin(theta0) * dy_world
    Y_body   = -sin(theta0) * dx_world + cos(theta0) * dy_world
    Delta_Z  = z - z0
    Delta_th = unwrap(yaw) - unwrap(yaw)[0]

X_body / Y_body are the horizontal displacement in the INITIAL PLANAR body frame
(the same SE(2) convention as the transition models). Delta_Z is the change in
Base height along the floor-frame vertical axis; it is NOT a body-frame Z
coordinate. np.unwrap only removes artificial +/-pi jumps. Every trajectory
therefore starts at (0, 0, 0) in both representations.

Every recorded sample is plotted after only this deterministic normalization:
no smoothing, filtering, interpolation, resampling, fitting, averaging or
outlier removal. Source CSVs are only read.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers the 3D projection)


# =============================================================================
# CONFIGURATION
# =============================================================================
STAGE_ROOT = Path("/home/unitree-arka/Go2_Skill_Base_Data_Sensor/Stage_HS_and_Move")   # filtered Base CSVs
COMMAND_ROOT = Path("/home/unitree-arka/Go2_Skill_Base_Data_Sensor/HandStandMove")     # command CSVs

REQUIRED_COLS = ["common_time_ns", "x_floor_m", "y_floor_m", "z_floor_m", "yaw_floor_rad"]
COMMAND_COLS = ["vx", "vy", "vyaw"]

# Comparison layout. The trial numbers themselves are entered at run time
# (ask_comparison_groups()); none are stored here.
N_GROUPS = 4
TRIALS_PER_GROUP = 4        # entered as Pair 1 first, Pair 1 second, Pair 2 first, Pair 2 second

# The two trials of a pair share a colour (Pair 1 / Pair 2 of each group);
# the position within the pair is shown by line style.
PAIR_COLORS = ["tab:blue", "tab:orange"]
MEMBER_LINESTYLES = ["-", "--"]          # first member of a pair = solid, second = dashed
# Sample markers are dense enough to hide dash gaps, so the member is ALSO encoded in
# the sample markers: filled dots for the first, hollow dots for the second.
MEMBER_MARKER_FILLED = [True, False]
MEMBER_NAMES = ["first (solid)", "second (dashed)"]   # console report only

# X-Y-Z scaling. X and Y always share one equal metric range (undistorted planar
# geometry, identical in the X-Y-Yaw figure). With True, Delta_Z uses that same range
# too (a true metric cube); during HandStand+Move the height change is ~centimetres
# against metres of travel, so it then draws as a flat line. False gives Delta_Z its
# own range so the height variation stays visible.
EQUAL_XYZ_SCALE = False

LINE_WIDTH = 1.8
SAMPLE_MARKER_SIZE = 3.0   # small dots at every recorded sample
ENDPOINT_SIZE = 90
ENDPOINT_EDGE_WIDTH = 1.6
VIEW = {"elev": 25, "azim": -60}   # identical initial camera for all 8 figures

X_LABEL = "X displacement in initial planar body frame (m)"
Y_LABEL = "Y displacement in initial planar body frame (m)"
Z_LABEL_XYZ = "Relative Base height ΔZ in floor frame (m)"
Z_LABEL_YAW = "Relative yaw Δθ (rad)"
CAPTION = ("X, Y: displacement rotated into each trajectory's initial planar body frame;  "
           "ΔZ = z(t) − z(0) along the floor vertical;  "
           "Δθ = unwrap(θ(t)) − unwrap(θ(0)).")


# =============================================================================
# FILE PATHS
# =============================================================================
def stage_dir(n):
    """Folder of the filtered Base trajectory of Trial n (Stage_n <-> Trial_n)."""
    return STAGE_ROOT / f"Stage_{n}_HS&Move"


def base_csv(n):
    return stage_dir(n) / f"Tf_handstand_stage_HandStandMove_Trial_{n}.csv"


def trial_dir(n):
    return COMMAND_ROOT / f"Trial_{n}"


def command_csv(n):
    return trial_dir(n) / f"cmd_HandStandMove_Trial_{n}.csv"


# =============================================================================
# CSV LOADING / VALIDATION
# =============================================================================
def load_trial_base(n):
    """
    Load the Base CSV of Trial n. Returns a time-sorted DataFrame, or None after
    printing a [WARNING] explaining why the trial cannot be used.
    """
    folder, path = stage_dir(n), base_csv(n)
    if not folder.is_dir():
        print(f"[WARNING] Trial {n}: Base folder not found: {folder}")
        return None
    if not path.is_file():
        print(f"[WARNING] Trial {n}: Base CSV not found: {path}")
        return None

    try:
        df = pd.read_csv(path)
    except Exception as exc:
        print(f"[WARNING] Trial {n}: could not read {path.name}: {exc}")
        return None

    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        print(f"[WARNING] Trial {n}: missing column(s) {missing} in {path.name}")
        return None

    df = df[REQUIRED_COLS].copy()
    for c in REQUIRED_COLS[1:]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    n_bad = int(df[REQUIRED_COLS].isna().any(axis=1).sum())
    if n_bad:
        # Not dropped silently: removing rows would change the trajectory.
        print(f"[WARNING] Trial {n}: {n_bad} row(s) with missing/non-numeric values in "
              f"{REQUIRED_COLS}; trial skipped")
        return None
    if len(df) < 2:
        print(f"[WARNING] Trial {n}: only {len(df)} sample(s); need at least 2")
        return None

    if not df["common_time_ns"].is_monotonic_increasing:
        print(f"[WARNING] Trial {n}: rows not in time order; sorted by common_time_ns")
        df = df.sort_values("common_time_ns", kind="stable")

    return df.reset_index(drop=True)


def single_numeric_value(df, col):
    """The one distinct numeric value of a CSV column (blank / NaN rows ignored)."""
    values = pd.to_numeric(df[col], errors="coerce").dropna().unique()
    if len(values) == 0:
        raise ValueError(f"column '{col}' has no numeric value")
    if len(values) > 1:
        raise ValueError(f"ambiguous command: column '{col}' has {len(values)} distinct "
                         f"values {sorted(values.tolist())}")
    return float(values[0])


def read_command(n):
    """
    Commanded V = (vx, vy, vyaw) of Trial n, read from cmd_HandStandMove_Trial_n.csv.
    The file may have several rows; each command column must contain exactly one
    distinct numeric value. If a trial_number column exists it must equal n.
    Raises ValueError with the reason otherwise.
    """
    path = command_csv(n)
    if not path.is_file():
        raise ValueError("command CSV not found")
    try:
        df = pd.read_csv(path)
    except Exception as exc:
        raise ValueError(f"could not read command CSV: {exc}")

    missing = [c for c in COMMAND_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"missing command column(s) {missing}")
    if "trial_number" in df.columns:
        file_trial = single_numeric_value(df, "trial_number")
        if file_trial != n:
            raise ValueError(f"trial_number={file_trial:g} in command CSV does not match Trial {n}")

    return tuple(single_numeric_value(df, c) for c in COMMAND_COLS)


# =============================================================================
# TERMINAL INPUT OF THE TRIAL NUMBERS
# =============================================================================
def parse_trials(text):
    """
    Parse one group's input line (spaces and/or commas as separators) into exactly
    TRIALS_PER_GROUP distinct positive integers. Raises ValueError with the reason.
    """
    tokens = text.replace(",", " ").split()
    if len(tokens) != TRIALS_PER_GROUP:
        raise ValueError(f"expected exactly {TRIALS_PER_GROUP} Trial numbers, got {len(tokens)}")
    trials = []
    for tok in tokens:
        try:
            n = int(tok)
        except ValueError:
            raise ValueError(f"'{tok}' is not an integer")
        if n <= 0:
            raise ValueError(f"'{tok}' is not a positive Trial number")
        trials.append(n)
    if len(set(trials)) != len(trials):
        raise ValueError("the same Trial number was entered more than once")
    return trials


def build_group(index, trials):
    """
    Group from its entered trials: entries 1-2 form Pair 1, entries 3-4 Pair 2.
    member 0 (first of the pair) is drawn solid, member 1 (second) dashed.
    """
    return {
        "name": f"Group {index}",
        "trajectories": [{"trial": n, "pair": k // 2 + 1, "member": k % 2}
                         for k, n in enumerate(trials)],
    }


def ask_comparison_groups():
    """Ask for the Trial numbers of every group in the terminal; repeat until valid."""
    groups = []
    for index in range(1, N_GROUPS + 1):
        print("\n" + "=" * 50)
        print(f"GROUP {index}")
        print("=" * 50)
        print(f"Enter Trial numbers for Group {index}")
        print("in the order (separated by spaces or commas):")
        print("    Pair 1 - trajectory A (solid)")
        print("    Pair 1 - trajectory B (dashed)")
        print("    Pair 2 - trajectory A (solid)")
        print("    Pair 2 - trajectory B (dashed)")
        while True:
            try:
                text = input("\nTrial numbers: ")
            except (EOFError, KeyboardInterrupt):
                raise SystemExit("\n[ERROR] input cancelled; no figures created")
            try:
                trials = parse_trials(text)
                break
            except ValueError as exc:
                print(f"[ERROR] {exc}; please enter them again")
        groups.append(build_group(index, trials))
    return groups


# =============================================================================
# INITIAL-POSE NORMALIZATION (initial planar body frame)
# =============================================================================
def relative_values(df):
    """
    X_body, Y_body : (x, y) - (x0, y0), rotated by -theta0 into the initial planar body frame
    DZ             : z - z0 (floor-frame vertical; not a body-frame coordinate)
    DYaw           : unwrap(yaw) - unwrap(yaw)[0]
    """
    x = df["x_floor_m"].to_numpy(dtype=float)
    y = df["y_floor_m"].to_numpy(dtype=float)
    z = df["z_floor_m"].to_numpy(dtype=float)
    yaw = np.unwrap(df["yaw_floor_rad"].to_numpy(dtype=float))

    dx_world, dy_world = x - x[0], y - y[0]
    c, s = np.cos(yaw[0]), np.sin(yaw[0])
    return {
        "X": c * dx_world + s * dy_world,
        "Y": -s * dx_world + c * dy_world,
        "Z": z - z[0],
        "Yaw": yaw - yaw[0],
    }


def command_label(t):
    """Legend entry: trial number and the commanded V read from its command CSV."""
    vx, vy, vyaw = t["cmd"]
    return f"Trial {t['trial']} | V=({vx:.4f},{vy:+.4f},{vyaw:+.4f})"


def group_trials(group):
    return [t["trial"] for t in group["trajectories"]]


def report_trial(t, df, r):
    """Console report for one selected trajectory."""
    n = t["trial"]
    vx, vy, vyaw = t["cmd"]
    first, last = df.iloc[0], df.iloc[-1]
    print(f"\nTrial {n}   (pair {t['pair']}, {MEMBER_NAMES[t['member']]} member)")
    print(f"  Base CSV          : {base_csv(n)}")
    print(f"  Command CSV       : {command_csv(n)}")
    print(f"  Command           : Vx={vx:.4f}  Vy={vy:+.4f}  Vyaw={vyaw:+.4f}")
    print(f"  Number of samples : {len(df)}")
    print(f"  Duration          : {(last.common_time_ns - first.common_time_ns) / 1e9:.3f} s")
    print(f"  Original initial  : X={first.x_floor_m:.4f} m  Y={first.y_floor_m:.4f} m  "
          f"Z={first.z_floor_m:.4f} m  Yaw={first.yaw_floor_rad:.4f} rad")
    print(f"  Original final    : X={last.x_floor_m:.4f} m  Y={last.y_floor_m:.4f} m  "
          f"Z={last.z_floor_m:.4f} m  Yaw={last.yaw_floor_rad:.4f} rad")
    print(f"  Normalized final  : X_body={r['X'][-1]:+.4f} m  Y_body={r['Y'][-1]:+.4f} m  "
          f"Delta_Z={r['Z'][-1]:+.4f} m  Delta_Yaw={r['Yaw'][-1]:+.4f} rad")
    first_norm = np.array([r["X"][0], r["Y"][0], r["Z"][0], r["Yaw"][0]])
    status = "[OK]" if np.allclose(first_norm, 0.0) else "[WARNING] not at origin"
    print(f"  Normalized first  : X_body={first_norm[0]:+.1e}  Y_body={first_norm[1]:+.1e}  "
          f"Delta_Z={first_norm[2]:+.1e}  Delta_Yaw={first_norm[3]:+.1e}  {status}")


def load_comparison_group(group):
    """
    Load and normalize the selected trials of one group -> list of (trajectory meta, relative values).
    The trajectory meta is the entry built from the terminal input plus "cmd" read from the command CSV.
    """
    print("\n" + "=" * 70)
    print(f"{group['name']}: Trials {', '.join(str(n) for n in group_trials(group))}")
    print("=" * 70)
    loaded = []
    for t in group["trajectories"]:
        n = t["trial"]
        df = load_trial_base(n)
        try:
            cmd = read_command(n)
        except ValueError as exc:
            print(f"[WARNING] Trial {n}: {exc} ({command_csv(n)}); trial skipped")
            cmd = None
        if df is None or cmd is None:
            continue
        t = {**t, "cmd": cmd}
        r = relative_values(df)
        report_trial(t, df, r)
        loaded.append((t, r))
    missing = [n for n in group_trials(group) if n not in {m["trial"] for m, _ in loaded}]
    if missing:
        print(f"\n[WARNING] {group['name']}: trials not plotted: {missing}")
    return loaded


# =============================================================================
# GROUP AXIS LIMITS (shared by all trajectories of one group)
# =============================================================================
def padded_limits(values, margin=0.08):
    """(min, max) of the values with a small margin; guards a zero range."""
    lo, hi = float(np.min(values)), float(np.max(values))
    span = hi - lo if hi > lo else 1e-3
    return lo - margin * span, hi + margin * span


def group_limits(loaded):
    """
    X/Y: one square region (equal metric range on X and Y), used by BOTH figures
         of the group so they can be compared directly.
    Z  : with EQUAL_XYZ_SCALE the same range as X/Y (true metric cube), otherwise
         its own padded range.
    Yaw: its own padded range in radians (never forced to the metric scale).
    """
    X = np.concatenate([r["X"] for _, r in loaded])
    Y = np.concatenate([r["Y"] for _, r in loaded])
    Z = np.concatenate([r["Z"] for _, r in loaded])
    Yaw = np.concatenate([r["Yaw"] for _, r in loaded])

    spans = [np.ptp(X), np.ptp(Y)] + ([np.ptp(Z)] if EQUAL_XYZ_SCALE else [])
    half = max(max(spans) / 2 * 1.1, 1e-3)
    mid = lambda v: (v.min() + v.max()) / 2
    xlim = (mid(X) - half, mid(X) + half)
    ylim = (mid(Y) - half, mid(Y) + half)
    zlim = (mid(Z) - half, mid(Z) + half) if EQUAL_XYZ_SCALE else padded_limits(Z)
    return {"x": xlim, "y": ylim, "z": zlim, "yaw": padded_limits(Yaw)}


# =============================================================================
# SHARED PLOTTING HELPER
# =============================================================================
def set_equal_box(fig, ax):
    """
    Draw the 3D box as a true cube, so equal axis ranges give equal on-screen
    lengths. matplotlib >= 3.3 has set_box_aspect (default box is 4:4:3).
    Older versions (e.g. Ubuntu 20.04's 3.1.2) always use a cube in data space
    but stretch it to fill the axes rectangle, so make that rectangle square.
    """
    if hasattr(ax, "set_box_aspect"):
        ax.set_box_aspect([1, 1, 1])
        return
    fig_w, fig_h = fig.get_size_inches()
    pos = ax.get_position()
    side = min(pos.width * fig_w, pos.height * fig_h)   # inches
    ax.set_position([pos.x0, pos.y0, side / fig_w, side / fig_h])


def plot_comparison(group, loaded, third_key, limits, equal_box, zlabel, suffix):
    """
    One figure, one 3D axes, one INDEPENDENT ax.plot() call per trajectory
    (trajectories are never concatenated). Colour = pair, line style = first / second
    member of the pair. No title is drawn; the group is identified only by the window
    title, which is also the default file name when saving from the figure window.
    """
    window_title = (f"{group['name']} - Trials "
                    f"{'-'.join(str(n) for n in group_trials(group))} - {suffix}")
    fig = plt.figure(num=window_title, figsize=(12, 8), clear=True)
    ax = fig.add_subplot(111, projection="3d")
    legend_handles = []

    for t, r in loaded:
        x, y, w = r["X"], r["Y"], r[third_key]
        color = PAIR_COLORS[(t["pair"] - 1) % len(PAIR_COLORS)]
        ls = MEMBER_LINESTYLES[t["member"]]

        face = color if MEMBER_MARKER_FILLED[t["member"]] else "white"
        ax.plot(x, y, w, color=color, linestyle=ls, linewidth=LINE_WIDTH,
                marker="o", markersize=SAMPLE_MARKER_SIZE, markerfacecolor=face,
                markeredgecolor=color, markeredgewidth=0.8)
        ax.scatter(x[0], y[0], w[0], color=color, marker="o", s=ENDPOINT_SIZE,
                   edgecolor="black", linewidths=ENDPOINT_EDGE_WIDTH, zorder=10)   # start
        ax.scatter(x[-1], y[-1], w[-1], color=color, marker="s", s=ENDPOINT_SIZE,
                   edgecolor="black", linewidths=ENDPOINT_EDGE_WIDTH, zorder=10)  # end
        legend_handles.append(Line2D([0], [0], color=color, linestyle=ls, linewidth=2.2,
                                     marker="o", markersize=5, markerfacecolor=face,
                                     markeredgecolor=color, label=command_label(t)))

    legend_handles += [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="gray",
               markeredgecolor="black", markersize=9, label="Start (all at origin)"),
        Line2D([0], [0], marker="s", color="w", markerfacecolor="gray",
               markeredgecolor="black", markersize=9, label="End"),
    ]

    ax.set_xlim(*limits["x"])
    ax.set_ylim(*limits["y"])
    ax.set_zlim(*(limits["z"] if third_key == "Z" else limits["yaw"]))
    ax.view_init(**VIEW)
    ax.set_xlabel(X_LABEL, labelpad=10)
    ax.set_ylabel(Y_LABEL, labelpad=10)
    ax.set_zlabel(zlabel, labelpad=10)
    ax.legend(handles=legend_handles, loc="upper left", bbox_to_anchor=(1.08, 1.0),
              fontsize=9, title="Command V = (Vx, Vy, Vyaw)", title_fontsize=9)
    fig.text(0.5, 0.01, CAPTION, ha="center", fontsize=8.5, style="italic", color="#444444")
    if equal_box:
        set_equal_box(fig, ax)

    return fig


# =============================================================================
# MAIN EXECUTION
# =============================================================================
def main():
    groups = ask_comparison_groups()

    created = 0
    for group in groups:
        loaded = load_comparison_group(group)
        if not loaded:
            print(f"[WARNING] {group['name']}: no trial could be loaded; both figures skipped")
            continue
        limits = group_limits(loaded)
        print(f"\n  Shared limits: X [{limits['x'][0]:.3f}, {limits['x'][1]:.3f}] m, "
              f"Y [{limits['y'][0]:.3f}, {limits['y'][1]:.3f}] m, "
              f"dZ [{limits['z'][0]:.3f}, {limits['z'][1]:.3f}] m, "
              f"dYaw [{limits['yaw'][0]:.3f}, {limits['yaw'][1]:.3f}] rad")
        plot_comparison(group, loaded, "Z", limits, EQUAL_XYZ_SCALE, Z_LABEL_XYZ, "X-Y-Z")
        plot_comparison(group, loaded, "Yaw", limits, False, Z_LABEL_YAW, "X-Y-Yaw")
        created += 2

    print(f"\n{created} of {2 * len(groups)} figures created")
    plt.show()


if __name__ == "__main__":
    main()
