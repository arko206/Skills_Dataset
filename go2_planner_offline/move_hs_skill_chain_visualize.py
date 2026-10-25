"""Plot the MOVE + HandStand skill-chain plan of Move_HS_skill_chain_planner.

Reads (current directory):
    planner.cfg, query.cfg                map, robot geometry, obstacles, start / goal
    move_hs_rrt_tree.txt                  search tree of the best run
    move_hs_skill_chain_waypoints.txt     x, y, theta per final-path node
    move_hs_skill_chain_steps.csv         mode / incoming skill / command per final-path node
Writes:
    move_hs_skill_chain_plan.png

Every waypoint is drawn with the footprint of its OWN stored mode (from the CSV);
nothing is assumed about which waypoints are STAND or HANDSTAND.
"""
import csv
import os

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.collections import LineCollection
import numpy as np

TREE_FILE = 'move_hs_rrt_tree.txt'
WAYPOINTS_FILE = 'move_hs_skill_chain_waypoints.txt'
STEPS_FILE = 'move_hs_skill_chain_steps.csv'
OUT_PNG = 'move_hs_skill_chain_plan.png'

# HandStand footprint (must match HANDSTAND_LENGTH / HANDSTAND_WIDTH in the planner)
HANDSTAND_L = 0.135
HANDSTAND_W = 0.18

PATH_COLOR = '#1a4fa0'
MODE_STYLE = {
    'STAND':     {'color': '#2a78d6', 'alpha': 0.18, 'label': 'STAND footprint'},
    'HANDSTAND': {'color': '#eb6834', 'alpha': 0.35, 'label': 'HANDSTAND footprint'},
}
SWITCH_STYLE = {
    'STAND_TO_HANDSTAND': {'marker': '^', 'color': '#6250d6', 'short': 'S2H'},
    'HANDSTAND_TO_STAND': {'marker': 'v', 'color': '#1baf7a', 'short': 'H2S'},
}
SHORT_SKILL = {'MOVE': 'MOVE', 'STAND_TO_HANDSTAND': 'S2H',
               'HANDSTAND_MOVE': 'HS_MOVE', 'HANDSTAND_TO_STAND': 'H2S'}


def load_config(filepath):
    config = {}
    if not os.path.exists(filepath):
        return config
    with open(filepath, 'r') as f:
        for line in f:
            idx = line.find('#')
            if idx != -1:
                line = line[:idx]
            line = line.strip()
            if not line:
                continue
            parts = line.split('=')
            if len(parts) == 2:
                try:
                    config[parts[0].strip()] = float(parts[1].strip())
                except ValueError:
                    pass
    return config


def rect_points(x, y, th, length, width):
    """Corners of a length x width rectangle (length along body X) at SE(2) pose (x, y, th)."""
    c, s = np.cos(th), np.sin(th)
    R = np.array(((c, -s), (s, c)))
    pts = np.array([[-length/2, -width/2], [length/2, -width/2],
                    [length/2, width/2], [-length/2, width/2]]).dot(R.T)
    pts[:, 0] += x
    pts[:, 1] += y
    return pts


# Rounded corner polygon for true circle-vs-rectangle inflation
def get_rounded_rect_points(l, w, r, num_pts=10):
    hl, hw = l/2, w/2
    pts = []
    for a in np.linspace(0, np.pi/2, num_pts):             # top right arc
        pts.append([hl + r*np.cos(a), hw + r*np.sin(a)])
    for a in np.linspace(np.pi/2, np.pi, num_pts):         # top left arc
        pts.append([-hl + r*np.cos(a), hw + r*np.sin(a)])
    for a in np.linspace(np.pi, 3*np.pi/2, num_pts):       # bottom left arc
        pts.append([-hl + r*np.cos(a), -hw + r*np.sin(a)])
    for a in np.linspace(3*np.pi/2, 2*np.pi, num_pts):     # bottom right arc
        pts.append([hl + r*np.cos(a), -hw + r*np.sin(a)])
    return np.array(pts)


def draw_obstacles(ax, cfg):
    i = 0
    while f'obs.{i}.x' in cfg:
        x, y, theta = cfg[f'obs.{i}.x'], cfg[f'obs.{i}.y'], cfg[f'obs.{i}.theta']
        L, W = cfg[f'obs.{i}.length'], cfg[f'obs.{i}.width']
        total = cfg.get('safety_margin', 0.05)
        c, s = np.cos(theta), np.sin(theta)
        R = np.array(((c, -s), (s, c)))

        # Plot inflated (true Minkowski sum of rect and circle)
        pts_inf = get_rounded_rect_points(L, W, total).dot(R.T)
        pts_inf[:, 0] += x
        pts_inf[:, 1] += y
        ax.add_patch(patches.Polygon(pts_inf, closed=True, color='red', alpha=0.1,
                                     label='Inflated' if i == 0 else ""))
        # Plot original
        ax.add_patch(patches.Polygon(rect_points(x, y, theta, L, W), closed=True, color='red', alpha=0.5,
                                     label='Obstacle' if i == 0 else ""))
        i += 1


def draw_tree(ax):
    """Tree edges, coloured by the mode of the child node (columns: x y theta parent cost mode skill)."""
    if not os.path.exists(TREE_FILE):
        print(f"No {TREE_FILE}; tree not drawn")
        return
    nodes = []
    with open(TREE_FILE, 'r') as f:
        f.readline()  # Skip Seed line
        for line in f:
            parts = line.split()
            if len(parts) < 4:
                continue
            try:
                nodes.append((float(parts[0]), float(parts[1]), int(parts[3]),
                              parts[5] if len(parts) >= 6 else 'STAND'))
            except ValueError:
                continue
    edges = {'STAND': [], 'HANDSTAND': []}
    for x, y, pidx, mode in nodes:
        if 0 <= pidx < len(nodes):
            px, py = nodes[pidx][0], nodes[pidx][1]
            edges.setdefault(mode, []).append(((px, py), (x, y)))
    tree_colors = {'STAND': '#666666', 'HANDSTAND': '#c8763f'}
    for mode, segs in edges.items():
        if not segs:
            continue
        col = tree_colors.get(mode, '#666666')
        ax.add_collection(LineCollection(segs, colors=col, linewidths=0.5, alpha=0.3, zorder=2,
                                         label=f'RRT tree ({mode})'))
        ends = np.array([seg[1] for seg in segs])
        ax.scatter(ends[:, 0], ends[:, 1], s=1, c=col, alpha=0.3, zorder=2)
    print(f"Plotted {sum(len(v) for v in edges.values())} tree edges "
          f"(STAND {len(edges['STAND'])}, HANDSTAND {len(edges['HANDSTAND'])})")


def load_plan():
    """Waypoints (N x 3) and the typed steps (list of dicts), checked against each other."""
    wps = np.loadtxt(WAYPOINTS_FILE, delimiter=',', ndmin=2)
    with open(STEPS_FILE, 'r', newline='') as f:
        steps = list(csv.DictReader(f))
    if len(steps) != len(wps):
        raise ValueError(f"{STEPS_FILE} has {len(steps)} rows but {WAYPOINTS_FILE} has {len(wps)} waypoints")
    for wp, st in zip(wps, steps):
        if abs(wp[0] - float(st['x'])) > 1e-5 or abs(wp[1] - float(st['y'])) > 1e-5:
            raise ValueError(f"waypoint {st['wp']} differs between {WAYPOINTS_FILE} and {STEPS_FILE}")
    return wps, steps


def skill_sequence(steps):
    """Compressed incoming-skill sequence, e.g. 'MOVE x3 -> S2H -> HS_MOVE x4 -> H2S -> MOVE'."""
    groups = []
    for st in steps[1:]:
        k = st['incoming_skill']
        if groups and groups[-1][0] == k:
            groups[-1][1] += 1
        else:
            groups.append([k, 1])
    return ' -> '.join(SHORT_SKILL.get(k, k) + (f' x{n}' if n > 1 else '') for k, n in groups)


def draw_plan(ax, cfg, wps, steps):
    ax.plot(wps[:, 0], wps[:, 1], '-', color=PATH_COLOR, linewidth=2.5, label='Planned path', zorder=4)

    stand_l, stand_w = cfg.get('robot_length', 0.4), cfg.get('robot_width', 0.35)
    dims = {'STAND': (stand_l, stand_w), 'HANDSTAND': (HANDSTAND_L, HANDSTAND_W)}
    labeled = set()
    for wp, st in zip(wps, steps):
        mode = st['mode']
        RL, RW = dims[mode]
        style = MODE_STYLE[mode]
        ax.add_patch(patches.Polygon(rect_points(wp[0], wp[1], wp[2], RL, RW), closed=True,
                                     facecolor=style['color'], edgecolor=style['color'],
                                     alpha=style['alpha'], linewidth=1.0, zorder=3,
                                     label=style['label'] if mode not in labeled else ""))
        labeled.add(mode)
        ax.plot(wp[0], wp[1], 'k.', markersize=4, zorder=5)
        ax.arrow(wp[0], wp[1], 0.08*np.cos(wp[2]), 0.08*np.sin(wp[2]),
                 head_width=0.04, head_length=0.04, fc='k', ec='k', zorder=6)

    # Mode switches: marker at the pose where the switch is executed (the parent waypoint)
    for i in range(1, len(steps)):
        skill = steps[i]['incoming_skill']
        if skill not in SWITCH_STYLE:
            continue
        style = SWITCH_STYLE[skill]
        x, y = wps[i-1][0], wps[i-1][1]
        ax.plot(x, y, linestyle='none', marker=style['marker'], markersize=12,
                markerfacecolor=style['color'], markeredgecolor='black', zorder=7,
                label=skill if skill not in labeled else "")
        labeled.add(skill)
        ax.annotate(f"{style['short']} (WP{i-1}->{i})", (x, y), xytext=(8, 6), textcoords='offset points',
                    fontsize=8, zorder=8,
                    bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='0.6', alpha=0.85))


def main():
    cfg1 = load_config('planner.cfg')
    cfg2 = load_config('query.cfg')
    cfg = {**cfg1, **cfg2}

    fig, ax = plt.subplots(figsize=(10, 9))
    ax.set_xlim(cfg.get('x_min', -2), cfg.get('x_max', 2))
    ax.set_ylim(cfg.get('y_min', -2), cfg.get('y_max', 3))

    draw_obstacles(ax, cfg)

    goal_x, goal_y = cfg.get('goal_x', 0), cfg.get('goal_y', 0)
    ax.add_patch(patches.Circle((goal_x, goal_y), cfg.get('goal_tol', 0.1), color='green', alpha=0.3,
                                label='Goal region (STAND)'))
    ax.plot(cfg.get('start_x', 0), cfg.get('start_y', 0), 'bo', markersize=8, label='Start (STAND)')

    try:
        draw_tree(ax)
    except Exception as e:
        print("Could not load or plot tree:", e)

    subtitle = 'no plan loaded'
    try:
        wps, steps = load_plan()
        draw_plan(ax, cfg, wps, steps)
        subtitle = skill_sequence(steps) or 'start already satisfies the goal'
        print("Skill sequence:", subtitle)
    except Exception as e:
        print("Could not load or plot plan:", e)

    ax.set_aspect('equal')
    ax.set_xlabel('x [m]')
    ax.set_ylabel('y [m]')
    ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1.0), fontsize=8, framealpha=0.9)
    ax.set_title('SE(2) Skill-Chain RRT Plan (MOVE + HandStand)\n' + subtitle, fontsize=10)
    plt.savefig(OUT_PNG, dpi=150, bbox_inches='tight')
    print(f"Saved {OUT_PNG}")


if __name__ == '__main__':
    main()
