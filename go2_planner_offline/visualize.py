import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import os

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

def main():
    cfg1 = load_config('planner.cfg')
    cfg2 = load_config('query.cfg')
    
    cfg = {**cfg1, **cfg2}

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

    try:
        from matplotlib.collections import LineCollection
        tree_edges = []
        nodes = []
        node_orientations = {} # To store orientations for each (x,y) point
        if os.path.exists("rrt_tree.txt"):
            with open("rrt_tree.txt", "r") as f:
                header = f.readline()  # Skip Seed line
                
                # New format: x y theta parent cost
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split()
                    if len(parts) >= 4:
                        try:
                            x = float(parts[0])
                            y = float(parts[1])
                            th = float(parts[2])
                            pidx = int(parts[3])
                            nodes.append({'x': x, 'y': y, 'th': th, 'parent': pidx})
                        except ValueError:
                            # Skip lines that cannot be converted to floats
                            continue
                    else:
                        # Skip lines that don't have enough parts
                        continue
                
                for i, n in enumerate(nodes):
                    if n['parent'] != -1 and n['parent'] < len(nodes):
                        p = nodes[n['parent']]
                        tree_edges.append(((p['x'], p['y']), (n['x'], n['y'])))
                        
                    # Keep orientation info for junctions
                    # Round coordinates to avoid floating point issues for dict keys
                    node_orientations.setdefault((round(n['x'], 3), round(n['y'], 3)), []).append(n['th'])
        
        if tree_edges:
            # Use a slightly darker gray and higher zorder to ensure it's on top of background
            lc = LineCollection(tree_edges, colors='#666666', linewidths=0.5, alpha=0.3, zorder=2)
            ax.add_collection(lc)
            
            # Plot tree nodes as small dots so rotation steps are visible
            nodes = np.array([edge[1] for edge in tree_edges])
            ax.scatter(nodes[:,0], nodes[:,1], s=1, c='#666666', alpha=0.3, zorder=2)
            
            print(f"Successfully plotted {len(tree_edges)} tree edges")
        else:
            print("No valid tree edges found in rrt_tree.txt")
    except Exception as e:
        print("Could not load or plot tree:", e)

    try:
        wps = np.loadtxt('se2_waypoints.txt', delimiter=',')
        if len(wps) > 0:
            if len(wps.shape) == 1:
                wps = wps.reshape(1, -1)
            ax.plot(wps[:, 0], wps[:, 1], 'b-', linewidth=2.5, label='Path', zorder=4)
            
            RL = cfg.get('robot_length', 0.4)
            RW = cfg.get('robot_width', 0.35)
            
            for i, wp in enumerate(wps):
                # Plot robot rectangle
                c, s = np.cos(wp[2]), np.sin(wp[2])
                R_mat = np.array(((c, -s), (s, c)))
                rect_pts = np.array([
                    [-RL/2, -RW/2], [RL/2, -RW/2], [RL/2, RW/2], [-RL/2, RW/2]
                ]).dot(R_mat.T)
                rect_pts[:, 0] += wp[0]
                rect_pts[:, 1] += wp[1]
                
                # Translucent robot footprint
                robot_poly = patches.Polygon(rect_pts, closed=True, color='blue', alpha=0.15, zorder=3, 
                                             label='Robot' if i == 0 else "")
                ax.add_patch(robot_poly)

                # Waypoint dot
                ax.plot(wp[0], wp[1], 'k.', markersize=4, zorder=5)

                # Advanced De-clutter:
                # Only show arrow for the LAST waypoint at this (x, y) location 
                # before moving to a new one, or if it's the start/end.
                is_first = (i == 0)
                is_last = (i == len(wps) - 1)
                
                show_arrow = is_first or is_last
                if not show_arrow:
                    # Check if NEXT waypoint is at a different location (Start of movement)
                    next_wp = wps[i+1]
                    dist_next = np.sqrt((wp[0]-next_wp[0])**2 + (wp[1]-next_wp[1])**2)
                    if dist_next > 0.01:
                        show_arrow = True
                    else:
                        # Check if PREVIOUS waypoint was at a different location (End of movement)
                        prev_wp = wps[i-1]
                        dist_prev = np.sqrt((wp[0]-prev_wp[0])**2 + (wp[1]-prev_wp[1])**2)
                        if dist_prev > 0.01:
                            show_arrow = True

                if show_arrow:
                    dx = 0.08 * np.cos(wp[2])
                    dy = 0.08 * np.sin(wp[2])
                    ax.arrow(wp[0], wp[1], dx, dy, head_width=0.04, head_length=0.04, fc='k', ec='k', zorder=6)
    except Exception as e:
        print("Could not load or plot waypoints:", e)
        
    ax.set_aspect('equal')
    # ax.legend()
    plt.title('SE(2) RRT Planner')
    plt.savefig('plan_viz.png', dpi=150)
    print("Saved plan_viz.png")

if __name__ == '__main__':
    main()