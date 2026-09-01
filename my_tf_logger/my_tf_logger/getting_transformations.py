#!/usr/bin/env python3
import os
import json
from datetime import datetime
from typing import Dict, Any, Tuple

import numpy as np
from scipy.spatial.transform import Rotation as R

import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from rclpy.time import Time
import tf2_ros


def transform_to_matrix(transform_msg) -> np.ndarray:
    """
    geometry_msgs/Transform -> 4x4 homogeneous matrix
    """
    tx = transform_msg.translation.x
    ty = transform_msg.translation.y
    tz = transform_msg.translation.z

    qx = transform_msg.rotation.x
    qy = transform_msg.rotation.y
    qz = transform_msg.rotation.z
    qw = transform_msg.rotation.w

    T = np.eye(4, dtype=float)
    T[:3, :3] = R.from_quat([qx, qy, qz, qw]).as_matrix()
    T[:3, 3] = [tx, ty, tz]
    return T


def matrix_to_pose_dict(T: np.ndarray) -> Dict[str, Any]:
    """
    Convert 4x4 matrix into a readable dict with translation, quaternion, RPY, and raw matrix.
    """
    rot = R.from_matrix(T[:3, :3])
    quat = rot.as_quat()  # x, y, z, w
    rpy = rot.as_euler('xyz', degrees=False)

    return {
        "translation": {
            "x": float(T[0, 3]),
            "y": float(T[1, 3]),
            "z": float(T[2, 3]),
        },
        "quaternion_xyzw": {
            "x": float(quat[0]),
            "y": float(quat[1]),
            "z": float(quat[2]),
            "w": float(quat[3]),
        },
        "rpy_xyz_rad": {
            "roll": float(rpy[0]),
            "pitch": float(rpy[1]),
            "yaw": float(rpy[2]),
        },
        "T": T.tolist(),
    }


def se3_to_se2_floor_xy(T: np.ndarray) -> Dict[str, float]:
    """
    Convert a 4x4 SE(3) transform into an SE(2) pose on the floor XY plane.

    Assumes:
      - floor frame is the world frame
      - planning happens in floor x-y plane
      - heading is yaw about floor z-axis

    Returns:
      {x, y, theta}
    """
    x = float(T[0, 3])
    y = float(T[1, 3])

    # heading from rotation matrix projected onto floor XY plane
    # first column of R = local x-axis expressed in world frame
    theta = float(np.arctan2(T[1, 0], T[0, 0]))

    return {
        "x": x,
        "y": y,
        "theta": theta,
    }


def se3_to_se2_custom_plane(
    T: np.ndarray,
    pos_indices: Tuple[int, int] = (0, 2),
    heading_mode: str = "xz_from_x_axis",
) -> Dict[str, float]:
    """
    Generic helper for converting SE(3) -> SE(2) on a chosen plane.

    Example:
      pos_indices=(0,2) gives position = (x,z)

    heading_mode options:
      - "xz_from_x_axis": theta = atan2(R[2,0], R[0,0])
      - "xy_from_x_axis": theta = atan2(R[1,0], R[0,0])

    Returns:
      {u, v, theta}
    """
    u = float(T[pos_indices[0], 3])
    v = float(T[pos_indices[1], 3])

    if heading_mode == "xz_from_x_axis":
        theta = float(np.arctan2(T[2, 0], T[0, 0]))
    elif heading_mode == "xy_from_x_axis":
        theta = float(np.arctan2(T[1, 0], T[0, 0]))
    else:
        raise ValueError(f"Unsupported heading_mode: {heading_mode}")

    return {
        "u": u,
        "v": v,
        "theta": theta,
    }


def load_saved_matrix(json_path: str, key: str) -> np.ndarray:
    """
    Utility for labmates: load one saved 4x4 matrix from JSON.
    """
    with open(json_path, "r") as f:
        data = json.load(f)

    return np.array(data["transforms"][key]["T"], dtype=float)





import matplotlib.pyplot as plt


from matplotlib.patches import Circle, Polygon


from matplotlib.patches import Circle, Polygon

def plot_floor_se2_entities(output_data: Dict[str, Any], save_fig: bool = True) -> None:

    tf_data = output_data["transforms"]

    poses = {
        "base_link": tf_data["floor_to_base_link"]["se2_floor_xy"],
        "object_2": tf_data["floor_to_object_2"]["se2_floor_xy"],
        "object_3": tf_data["floor_to_object_3"]["se2_floor_xy"],
        "new_obstacle": tf_data["floor_to_new_obstacle"]["se2_floor_xy"]
    }

    fig, ax = plt.subplots(figsize=(8, 8))

    arrow_len = 0.12
    robot_radius = 0.05

    # ---------- helper: rotated rectangle ----------
    def rotated_rectangle(cx, cy, theta, length, width):
        hl = length / 2.0
        hw = width / 2.0

        corners = [
            (+hl, +hw),
            (+hl, -hw),
            (-hl, -hw),
            (-hl, +hw),
        ]

        c = np.cos(theta)
        s = np.sin(theta)

        world = []
        for x, y in corners:
            wx = cx + c * x - s * y
            wy = cy + s * x + c * y
            world.append((wx, wy))

        return world

    # ---------- plot base_link as circle ----------
    base = poses["base_link"]
    bx, by, bth = base["x"], base["y"], base["theta"]

    ax.plot(bx, by, "bo", label="base_link")
    base_circle = Circle((bx, by), robot_radius, fill=False, edgecolor="blue", linewidth=2)
    ax.add_patch(base_circle)

    ax.arrow(
        bx, by,
        arrow_len * np.cos(bth),
        arrow_len * np.sin(bth),
        head_width=0.04,
        head_length=0.05,
        color="blue"
    )

    # ---------- plot object_2 (obstacle block) ----------
    obj2 = poses["object_2"]
    o2x, o2y, o2th = obj2["x"], obj2["y"], obj2["theta"]

    obs_length = 0.25
    obs_width = 0.35

    rect2 = Polygon(
        rotated_rectangle(o2x, o2y, o2th, obs_length, obs_width),
        closed=True,
        fill=False,
        edgecolor="red",
        linewidth=2,
        label="Obstacle (object_2)"
    )
    ax.add_patch(rect2)
    ax.plot(o2x, o2y, "ro")

    # ---------- inflated obstacle ----------
    inflated_rect2 = Polygon(
        rotated_rectangle(
            o2x, o2y, o2th,
            obs_length + 2.0 * robot_radius,
            obs_width + 2.0 * robot_radius
        ),
        closed=True,
        fill=False,
        edgecolor="orange",
        linewidth=2,
        linestyle="--",
        label="Inflated obstacle"
    )
    ax.add_patch(inflated_rect2)



    # ---------- plot second new obstacle (second obstacle block) ----------
    obs_new = poses["new_obstacle"]
    obs_new_x, obs_new_y, obs_new_theta = obs_new["x"], obs_new["y"], obs_new["theta"]

    obs_new_length = 0.57
    obs_new_width = 0.42

    obs_new_rect2 = Polygon(
        rotated_rectangle(obs_new_x, obs_new_y, obs_new_theta, obs_new_length, obs_new_width),
        closed=True,
        fill=False,
        edgecolor="Violet",
        linewidth=2,
        label="New Obstacle"
    )
    ax.add_patch(obs_new_rect2)
    ax.plot(obs_new_x, obs_new_y, "ro")

    # ---------- inflated obstacle ----------
    inflated_rect2_obs = Polygon(
        rotated_rectangle(
            obs_new_x, obs_new_y, obs_new_theta,
            obs_new_length + 2.0 * robot_radius,
            obs_new_width + 2.0 * robot_radius
        ),
        closed=True,
        fill=False,
        edgecolor="orange",
        linewidth=2,
        linestyle="--",
        label="Inflated rectangle obstacle"
    )
    ax.add_patch(inflated_rect2_obs)





    # ---------- plot object_3 (goal block) ----------
    obj3 = poses["object_3"]
    o3x, o3y, o3th = obj3["x"], obj3["y"], obj3["theta"]

    rect3 = Polygon(
        rotated_rectangle(o3x, o3y, o3th, 0.32, 0.54),
        closed=True,
        fill=False,
        edgecolor="green",
        linewidth=2,
        linestyle="--",
        label="Goal (object_3)"
    )
    ax.add_patch(rect3)
    ax.plot(o3x, o3y, "go")

    # ---------- dashed relations ----------
    ax.plot([bx, o2x], [by, o2y], "--", alpha=0.7)
    ax.plot([bx, o3x], [by, o3y], "--", alpha=0.7)
    ax.plot([o2x, o3x], [o2y, o3y], "--", alpha=0.7)

    # ---------- formatting ----------
    ax.set_xlabel("Floor X [m]")
    ax.set_ylabel("Floor Y [m]")
    ax.set_title("SE(2) Planning Visualization (Floor Frame)")
    ax.grid(True)
    ax.axis("equal")
    ax.legend()
    plt.tight_layout()

    if save_fig:
        fig_path = "/home/unitree-arka/Desktop/floor_se2_plot.png"
        plt.savefig(fig_path, dpi=200, bbox_inches="tight")
        print(f"Saved SE(2) plot to: {fig_path}")

    #plt.show()


import sys
class FloorTransformSaver(Node):
    def __init__(self):
        super().__init__("floor_transform_saver")

        self.declare_parameter("floor_frame", "floor")
        self.declare_parameter("base_frame", "base_link_floor")
        self.declare_parameter("object2_frame", "object_2_floor")
        self.declare_parameter("object3_frame", "object_3_floor")
        self.declare_parameter("new_obstacle_frame", "new_obstacle_floor")
        self.declare_parameter(
            "save_path",
            "/home/unitree-arka/Desktop/floor_world_transforms.json"
        )

        self.floor_frame = self.get_parameter("floor_frame").value
        self.base_frame = self.get_parameter("base_frame").value
        self.object2_frame = self.get_parameter("object2_frame").value
        self.object3_frame = self.get_parameter("object3_frame").value
        self.save_path = self.get_parameter("save_path").value
        self.new_obstacle_frame = self.get_parameter("new_obstacle_frame").value

        self.tf_buffer = tf2_ros.Buffer(cache_time=Duration(seconds=10.0))
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.timer = self.create_timer(1.0, self.lookup_and_save_once)
        self.saved_once = False

        self.get_logger().info("FloorTransformSaver started.")
        self.get_logger().info(f"Will save transforms to: {self.save_path}")

    def lookup_matrix(self, parent: str, child: str) -> np.ndarray:
        t = self.tf_buffer.lookup_transform(
            parent,
            child,
            Time(),
            timeout=Duration(seconds=1.0)
        )
        return transform_to_matrix(t.transform)

    def lookup_and_save_once(self):
        if self.saved_once:
            return

        try:
            T_floor_base = self.lookup_matrix(self.floor_frame, self.base_frame)
            T_floor_obj2 = self.lookup_matrix(self.floor_frame, self.object2_frame)
            T_floor_obj3 = self.lookup_matrix(self.floor_frame, self.object3_frame)
            T_floor_new_obs = self.lookup_matrix(self.floor_frame, self.new_obstacle_frame)
        except Exception as e:
            self.get_logger().warn(f"Transform lookup failed: {e}")
            return

        output = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "reference_frame": self.floor_frame,
            "transforms": {
                "floor_to_base_link": {
                    **matrix_to_pose_dict(T_floor_base),
                    "se2_floor_xy": se3_to_se2_floor_xy(T_floor_base),
                    "se2_xz_debug": se3_to_se2_custom_plane(
                        T_floor_base,
                        pos_indices=(0, 2),
                        heading_mode="xz_from_x_axis"
                    ),
                },
                "floor_to_object_2": {
                    **matrix_to_pose_dict(T_floor_obj2),
                    "se2_floor_xy": se3_to_se2_floor_xy(T_floor_obj2),
                    "se2_xz_debug": se3_to_se2_custom_plane(
                        T_floor_obj2,
                        pos_indices=(0, 2),
                        heading_mode="xz_from_x_axis"
                    ),
                },
                "floor_to_object_3": {
                    **matrix_to_pose_dict(T_floor_obj3),
                    "se2_floor_xy": se3_to_se2_floor_xy(T_floor_obj3),
                    "se2_xz_debug": se3_to_se2_custom_plane(
                        T_floor_obj3,
                        pos_indices=(0, 2),
                        heading_mode="xz_from_x_axis"
                    ),
                },
                "floor_to_new_obstacle": {
                    **matrix_to_pose_dict(T_floor_new_obs),
                    "se2_floor_xy": se3_to_se2_floor_xy(T_floor_new_obs),
                    "se2_xz_debug": se3_to_se2_custom_plane(
                        T_floor_new_obs,
                        pos_indices=(0, 2),
                        heading_mode="xz_from_x_axis"
                    ),
                },


            }
        }

        os.makedirs(os.path.dirname(self.save_path), exist_ok=True)
        with open(self.save_path, "w") as f:
            json.dump(output, f, indent=2)

        # plot_floor_se2_entities(output, save_fig=True)
        

        self.get_logger().info("Saved transforms successfully.")
        # self.get_logger().info(
        #     f"floor->base_link SE2: {output['transforms']['floor_to_base_link']['se2_floor_xy']}"
        # )
        # self.get_logger().info(
        #     f"floor->object_2 SE2: {output['transforms']['floor_to_object_2']['se2_floor_xy']}"
        # )
        # self.get_logger().info(
        #     f"floor->object_3 SE2: {output['transforms']['floor_to_object_3']['se2_floor_xy']}"
        # )

        # self.get_logger().info(
        #     f"floor->new_obstacle SE2: {output['transforms']['floor_to_new_obstacle']['se2_floor_xy']}"
        # )



        base_se2 = output["transforms"]["floor_to_base_link"]["se2_floor_xy"]
        obj2_se2 = output["transforms"]["floor_to_object_2"]["se2_floor_xy"]
        obj3_se2 = output["transforms"]["floor_to_object_3"]["se2_floor_xy"]
        new_obs_se2 = output["transforms"]["floor_to_new_obstacle"]["se2_floor_xy"]
        self.get_logger().info(
            f"[base_link]     x={base_se2['x']:.6f}, y={base_se2['y']:.6f}, theta={base_se2['theta']:.6f}"
        )
        self.get_logger().info(
            f"[object_2]      x={obj2_se2['x']:.6f}, y={obj2_se2['y']:.6f}, theta={obj2_se2['theta']:.6f}"
        )
        self.get_logger().info(
            f"[object_3]      x={obj3_se2['x']:.6f}, y={obj3_se2['y']:.6f}, theta={obj3_se2['theta']:.6f}"
        )
        self.get_logger().info(
            f"[new_obstacle]  x={new_obs_se2['x']:.6f}, y={new_obs_se2['y']:.6f}, theta={new_obs_se2['theta']:.6f}"
        )

        # start_x     = 0.4951160401189585
        # start_y     = -0.3499195050119639
        # start_theta = 1.6011516741485452

        # goal_x      =  0.21732685504398908
        # goal_y      = 1.0844167946790706
        # goal_theta  =2.096970272082196

        # obs.0.x      = 0.9306163706562364
        # obs.0.y      = 0.3148186935880498
        # obs.0.theta  = 3.0018571020993536

        # obs.1.x      =-0.3563868039202832
        # obs.1.y      =0.6987057373628113
        # obs.1.theta  = -0.05723210515595908



        outstr = "\n"
        outstr = outstr + "start_x = " + str(base_se2['x']) + "\n"
        outstr = outstr + "start_y = " + str(base_se2['y']) + "\n"
        outstr = outstr + "start_theta = " + str(base_se2['theta']) + "\n"
        outstr = outstr+ "\n"

        outstr = outstr + "goal_x = " + str(obj3_se2['x']) + "\n"
        outstr = outstr + "goal_y = " + str(obj3_se2['y']) + "\n"
        outstr = outstr + "goal_theta = " + str(obj3_se2['theta']) + "\n"
        outstr = outstr+ "\n"

        outstr = outstr + "obs.0.x = " + str(obj2_se2['x']) + "\n"
        outstr = outstr + "obs.0.y = " + str(obj2_se2['y']) + "\n"
        outstr = outstr + "obs.0.theta = " + str(obj2_se2['theta']) + "\n"
        outstr = outstr+ "\n"

        outstr = outstr + "obs.1.x = " + str(new_obs_se2['x']) + "\n"
        outstr = outstr + "obs.1.y = " + str(new_obs_se2['y']) + "\n"
        outstr = outstr + "obs.1.theta = " + str(new_obs_se2['theta']) + "\n"
        outstr = outstr+ "\n"

        docker_path = "/home/unitree-arka/Desktop/"
        filename = docker_path+"query.cfg"
        outfile = open(filename, "w")

        outfile.write(outstr)
        outfile.close()
        print("\n", outstr, "\n")


        self.saved_once = True

        sys.exit()



def main():
    rclpy.init()
    node = FloorTransformSaver()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()



if __name__ == "__main__":
    main()
