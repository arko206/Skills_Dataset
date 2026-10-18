import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import os
import csv

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

def load_virtual_poses(predicted_file):
    """
    Read the open-loop plan written by go2_navg_exec.cpp (one row per Move() step) and build the
    virtual-pose sequence P0..PN:
        P0     = start_virtual_pose of step 0
        P(i+1) = predicted landing pose after step i
    The values are used exactly as written by GenerateOpenLoopPlan(); no model is evaluated here.
    Returns (rows, virtual_poses as an (N+1) x 3 array); both are empty if nothing can be loaded.
    """
    if not os.path.exists(predicted_file):
        print(f"Could not find {predicted_file}")
        print("Run go2_navg_exec first (it writes this file). No predicted poses will be plotted.")
        return [], np.empty((0, 3))

    try:
        with open(predicted_file, 'r', newline='') as f:
            rows = list(csv.DictReader(f))

        virtual_poses = []
        if rows:
            first = rows[0]
            virtual_poses.append([float(first['start_x']), float(first['start_y']), float(first['start_theta'])])
        for row in rows:
            virtual_poses.append([float(row['pred_x']), float(row['pred_y']), float(row['pred_theta'])])
            row['cmd'] = (float(row['vx']), float(row['vy']), float(row['vyaw']))

        # Each step should start where the previous one was predicted to land.
        for i in range(1, len(rows)):
            start = [float(rows[i]['start_x']), float(rows[i]['start_y']), float(rows[i]['start_theta'])]
            if not np.allclose(start, virtual_poses[i], atol=1e-6):
                print(f"[WARN] step {rows[i]['step']} starts at {start}, not at the predicted landing pose "
                      f"of the previous step {virtual_poses[i]}")
    except (KeyError, ValueError, TypeError) as e:
        print(f"Could not read {predicted_file}: {e!r}")
        return [], np.empty((0, 3))

    if not rows:
        print(f"{predicted_file} contains no Move() steps. No predicted poses will be plotted.")
    return rows, np.array(virtual_poses, dtype=float).reshape(-1, 3)

def print_virtual_waypoints(rows, virtual_poses):
    print("=" * 60)
    print("OPEN-LOOP MODEL-PREDICTED VIRTUAL WAYPOINTS")
    print("=" * 60)
    print(f"Total controller Move() steps : {len(rows)}")
    print(f"Total virtual poses           : {len(virtual_poses)}")
    print()
    header = (f"{'Pose':<6}{'Source':<22}{'x':>10}{'y':>10}{'theta':>10}   "
              f"{'Target':<8}{'Move(vx, vy, vyaw)':<29}Status")
    print(header)
    print("-" * len(header))
    for i, (x, y, theta) in enumerate(virtual_poses):
        if i == 0:
            source, target, move, status = "Initial Virtual Pose", "-", "-", "-"
        else:
            row = rows[i - 1]
            vx, vy, vyaw = row['cmd']
            source = f"Step {row['step']} prediction"
            target = f"WP{row['target_wp']}"
            move = f"({vx:.4f}, {vy:.4f}, {vyaw:.4f})"
            status = row['status']
        print(f"{'P' + str(i):<6}{source:<22}{x:>10.4f}{y:>10.4f}{theta:>10.4f}   "
              f"{target:<8}{move:<29}{status}")
    print("-" * len(header))
    print(f"Loaded {len(rows)} predicted controller steps and {len(virtual_poses)} virtual poses.")

def main():
    cfg1 = load_config('planner.cfg')
    cfg2 = load_config('query.cfg')
    
    cfg = {**cfg1, **cfg2}

    # Model-predicted virtual landing poses from go2_navg_exec.cpp (NOT measured robot poses).
    # Loaded and printed before anything is plotted.
    predicted_file = "open_loop_predicted_poses.csv"
    rows, virtual_poses = load_virtual_poses(predicted_file)
    if rows:
        print_virtual_waypoints(rows, virtual_poses)

    fig, ax = plt.subplots(figsize=(8, 8))
    
    ax.set_xlim(cfg.get('x_min', -2), cfg.get('x_max', 2))
    ax.set_ylim(cfg.get('y_min', -2), cfg.get('y_max', 3))
    
    i = 0
    while f'obs.{i}.x' in cfg:
        x = cfg[f'obs.{i}.x']
        y = cfg[f'obs.{i}.y']
        theta = cfg[f'obs.{i}.theta']
        L = cfg[f'obs.{i}.length']
        W = cfg[f'obs.{i}.width']
        total = cfg.get('safety_margin', 0.05)
        
        # Rounded corner polygon for true circle-vs-rectangle inflation
        def get_rounded_rect_points(l, w, r, num_pts=10):
            hl, hw = l/2, w/2
            pts = []
            # Top right arc
            angles = np.linspace(0, np.pi/2, num_pts)
            for a in angles:
                pts.append([hl + r*np.cos(a), hw + r*np.sin(a)])
            # Top left arc
            angles = np.linspace(np.pi/2, np.pi, num_pts)
            for a in angles:
                pts.append([-hl + r*np.cos(a), hw + r*np.sin(a)])
            # Bottom left arc
            angles = np.linspace(np.pi, 3*np.pi/2, num_pts)
            for a in angles:
                pts.append([-hl + r*np.cos(a), -hw + r*np.sin(a)])
            # Bottom right arc
            angles = np.linspace(3*np.pi/2, 2*np.pi, num_pts)
            for a in angles:
                pts.append([hl + r*np.cos(a), -hw + r*np.sin(a)])
            return np.array(pts)

        c, s = np.cos(theta), np.sin(theta)
        R = np.array(((c, -s), (s, c)))

        # Plot inflated (true Minkowski sum of rect and circle)
        pts_inf = get_rounded_rect_points(L, W, total)
        pts_inf = pts_inf.dot(R.T)
        pts_inf[:, 0] += x
        pts_inf[:, 1] += y
        poly_inf = patches.Polygon(pts_inf, closed=True, color='red', alpha=0.1, label='Inflated' if i==0 else "")
        ax.add_patch(poly_inf)

        # Plot original
        corners = np.array([
            [-L/2, -W/2], [L/2, -W/2], [L/2, W/2], [-L/2, W/2]
        ])
        corners = corners.dot(R.T)
        corners[:, 0] += x
        corners[:, 1] += y
        poly = patches.Polygon(corners, closed=True, color='red', alpha=0.5, label='Obstacle' if i==0 else "")
        ax.add_patch(poly)
        i += 1
        
    start_x, start_y = cfg.get('start_x', 0), cfg.get('start_y', 0)
    start_th = cfg.get('start_theta', 0)
    goal_x, goal_y = cfg.get('goal_x', 0), cfg.get('goal_y', 0)
    
    goal_circle = patches.Circle((goal_x, goal_y), cfg.get('goal_tol', 0.1), color='green', alpha=0.3)
    ax.add_patch(goal_circle)

    ax.plot(start_x, start_y, 'bo', markersize=8, label='Start')

    if len(virtual_poses) > 0:
        # Affine-model predicted open-loop trajectory P0 -> P1 -> ... -> PN (not the RRT path)
        ax.plot(virtual_poses[:, 0], virtual_poses[:, 1], 'b-', linewidth=2.5,
                label='Predicted Virtual Path', zorder=4)

        RL = cfg.get('robot_length', 0.4)
        RW = cfg.get('robot_width', 0.35)

        for i, (x, y, theta) in enumerate(virtual_poses):
            # Plot robot rectangle
            c, s = np.cos(theta), np.sin(theta)
            R_mat = np.array(((c, -s), (s, c)))
            rect_pts = np.array([
                [-RL/2, -RW/2], [RL/2, -RW/2], [RL/2, RW/2], [-RL/2, RW/2]
            ]).dot(R_mat.T)
            rect_pts[:, 0] += x
            rect_pts[:, 1] += y

            # Translucent robot footprint
            robot_poly = patches.Polygon(rect_pts, closed=True, color='blue', alpha=0.15, zorder=3,
                                         label='Predicted Robot Pose' if i == 0 else "")
            ax.add_patch(robot_poly)

            # Pose centre
            ax.plot(x, y, 'k.', markersize=5, zorder=5)

            # Heading arrow at every virtual pose
            dx = 0.08 * np.cos(theta)
            dy = 0.08 * np.sin(theta)
            ax.arrow(x, y, dx, dy, head_width=0.04, head_length=0.04, fc='k', ec='k', zorder=6)

            # P0 = initial virtual pose, P(i+1) = landing predicted after controller step i
            ax.text(x + 0.02, y + 0.02, f"P{i}", fontsize=8, zorder=7)

        print(f"Plotted {len(virtual_poses)} predicted virtual poses")

    ax.set_aspect('equal')
    # ax.legend()
    plt.title('SE(2) Open-Loop Model-Predicted Trajectory')
    plt.savefig('controller_prediction_viz.png', dpi=150, bbox_inches='tight')
    print("Saved controller_prediction_viz.png")

if __name__ == '__main__':
    main()
