#!/usr/bin/env python3
import os
from datetime import datetime
from collections import deque

import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from rclpy.time import Time

import numpy as np
import tf2_ros
from geometry_msgs.msg import TransformStamped, Point
from visualization_msgs.msg import Marker, MarkerArray  

from scipy.spatial.transform import Rotation as R
import json


# ============================================================
# FIXED CALIBRATION MATRIX P*(Rc->T1)
# ============================================================
# P_star_Rc_to_T1 = np.array([
#     [0.998434, -0.051057, -0.022868, 0.002303],
#     [-0.023986, -0.021393, -0.999483, -0.047973],
#     [0.050541, 0.998467, -0.022584, -0.089412],
#     [0.000000, 0.000000, 0.000000, 1.000000]
# ])

# P_star_Rc_to_T1 = np.array([
#     [0.993767, -0.110441, -0.015138, -0.003802],
# [-0.016527, -0.011670 ,-0.999795, -0.048409],
# [0.110242, 0.993814, -0.013423, -0.090582],
# [0.000000, 0.000000 ,0.000000, 1.000000]
# ])

# P_star_Rc_to_T1 = np.array([
#     [0.999212, -0.038571, 0.009362, 0.026219],
#     [0.008859, -0.013171, -0.999874, -0.046079],
#     [0.038689, 0.999169, -0.012819, -0.095978],
#     [0.000000, 0.000000, 0.000000, 1.000000]
# ])

###---- During the Demo --------------------------####
# P_star_Rc_to_T1 = np.array([
#     [0.998833, 0.039814, -0.027331, 0.008661],
#     [-0.025861, -0.036982, -0.998981, -0.049915],
#     [-0.040784, 0.998523, -0.035909, -0.107167],
#     [0.000000, 0.000000, 0.000000, 1.000000]
# ])


###----- After the Demo -----------------------------------####

P_star_Rc_to_T1 = np.array ([
    [0.997492, 0.059323, -0.038611, 0.002513],
[-0.034889, -0.062547, -0.997432, -0.041843],
[-0.061586, 0.996277, -0.060320, -0.112739],
[0.000000, 0.000000, 0.000000, 1.000000]
])






# ============================================================
# HELPERS
# ============================================================
def quat_to_rot(qx, qy, qz, qw):
    """Quaternion (x,y,z,w) -> rotation matrix and RPY."""
    quat_array = np.array([qx, qy, qz, qw], dtype=float)
    rotation = R.from_quat(quat_array)
    rpy = rotation.as_euler('xyz', degrees=False)
    return rotation.as_matrix(), rpy


def translation_error(prev_T, curr_T):
    prev_p = prev_T[:3, 3]
    curr_p = curr_T[:3, 3]
    return np.linalg.norm(curr_p - prev_p)


def rotation_error(prev_T, curr_T):
    prev_R = prev_T[:3, :3]
    curr_R = curr_T[:3, :3]

    R_err = prev_R.T @ curr_R
    trace_val = np.trace(R_err)

    cos_theta = (trace_val - 1.0) / 2.0
    cos_theta = np.clip(cos_theta, -1.0, 1.0)

    return np.arccos(cos_theta)


def rpy_error(prev_rpy, curr_rpy):
    """Absolute component-wise RPY difference."""
    prev_rpy = np.array(prev_rpy, dtype=float)
    curr_rpy = np.array(curr_rpy, dtype=float)
    return np.abs(curr_rpy - prev_rpy)


def euclidean_distance_from_camera(T_matrix):
    p = T_matrix[:3, 3]
    return np.linalg.norm(p)

###--- Reading the bound values from the configuration file path ------######
def load_planner_bounds(cfg_path):
    """
    Read x_min, x_max, y_min, y_max from planner.cfg.
    Lines may contain comments after '#'.
    """
    bounds = {
        "x_min": -0.5,
        "x_max": 2.0,
        "y_min": -0.5,
        "y_max": 2.0,
    }

    cfg_path = os.path.expanduser(cfg_path)

    if not os.path.exists(cfg_path):
        return bounds

    with open(cfg_path, "r") as f:
        for line in f:
            line = line.split("#", 1)[0].strip()

            if not line or "=" not in line:
                continue

            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip()

            if key in bounds:
                try:
                    bounds[key] = float(value)
                except ValueError:
                    pass

    return bounds


# ============================================================
# NODE
# ============================================================
class TFLogger(Node):
    def __init__(self):
        super().__init__('tf_logger')

        # --------------------------------------------------
        # PARAMETERS
        # --------------------------------------------------
        self.declare_parameter('parent_frame', 'camera')
        self.declare_parameter('child_frame', 'object_1')
        self.declare_parameter('intermediate_frame', 'object_2')
        self.declare_parameter('goal_frame', 'object_3')
        ##--- new addition for floor-tag parameters------###
        # Persistent world frame used by RViz
        self.declare_parameter('floor_frame', 'floor')

        # Raw floor-tag frame published by apriltag_ros
        self.declare_parameter('floor_tag_frame', 'floor_tag_raw')

        ###----- end of new additioj-------------------####
        self.declare_parameter('second_obstacle', 'new_obstacle')
        self.declare_parameter('rate_hz', 10.0)
        self.declare_parameter('deque_maxlen', 1000)

        self.declare_parameter(
            'planner_config_file',
            '/home/unitree-arka/go2_planner_offline/planner.cfg'
        )

        self.parent_frame = self.get_parameter('parent_frame').get_parameter_value().string_value
        self.child_frame = self.get_parameter('child_frame').get_parameter_value().string_value
        self.intermediate_frame = self.get_parameter('intermediate_frame').get_parameter_value().string_value
        self.goal_frame = self.get_parameter('goal_frame').get_parameter_value().string_value
        self.floor_frame = self.get_parameter('floor_frame').get_parameter_value().string_value

        ##---- newest addition of parameters ---------#####
        self.floor_tag_frame = (
            self.get_parameter('floor_tag_frame')
            .get_parameter_value()
            .string_value
        )


        if self.floor_frame == self.floor_tag_frame:
            raise RuntimeError(
                "floor_frame and floor_tag_frame must be different; "
                "otherwise a TF cycle will be created."
            )
        ##---------###############
        self.second_obstacle_frame = self.get_parameter('second_obstacle').get_parameter_value().string_value
        self.rate_hz = self.get_parameter('rate_hz').get_parameter_value().double_value
        self.deque_maxlen = self.get_parameter('deque_maxlen').get_parameter_value().integer_value
        # --------------------------------------------------
        self.planner_config_file = (
            self.get_parameter('planner_config_file')
            .get_parameter_value()
            .string_value
        )

        self.floor_bounds = load_planner_bounds(self.planner_config_file)

        self.get_logger().info(
            "Loaded planner bounds from planner.cfg: "
            f"x=[{self.floor_bounds['x_min']:.2f}, {self.floor_bounds['x_max']:.2f}], "
            f"y=[{self.floor_bounds['y_min']:.2f}, {self.floor_bounds['y_max']:.2f}]"
        )



        # TF BUFFER / LISTENER / BROADCASTER
        # --------------------------------------------------
        self.buffer = tf2_ros.Buffer(cache_time=Duration(seconds=10.0))
        self.listener = tf2_ros.TransformListener(self.buffer, self, spin_thread=False)
        self.broadcaster = tf2_ros.TransformBroadcaster(self)

        # --------------------------------------------------
        # RViz CUBE MARKER PUBLISHER
        # --------------------------------------------------
        self.marker_pub = self.create_publisher(
            MarkerArray,
            "/go2_environment_cubes",
            10
        )

        # Object dimensions in metres: (x_length, y_length, z_length)
        self.object2_cube_size = (0.215, 0.96, 0.01) # (0.25, 0.35, 0.48
        self.new_obstacle_cube_size = (0.39, 0.90, 0.01) # (0.57, 0.42, 0.18)
        #self.goal_cube_size = (0.32, 0.54, 0.001) 

        self.goal_circle_radius = 0.215      # 21.5 cm
        self.goal_circle_thickness = 0.01    # 1 cm visual thickness

        # --------------------------------------------------
        # FLOOR LOCK STATE (Transformation from camera to floor)
        # --------------------------------------------------
        self.last_valid_C_to_floor = None
        self.have_floor_lock = False

        # --------------------------------------------------
        # PERSISTENT FLOOR-BASED OBJECT LOCKS
        # --------------------------------------------------
        self.last_valid_floor_to_object2 = None
        self.last_valid_floor_to_object3 = None
        self.last_valid_floor_to_new_obstacle = None

        # Last valid robot pose and floor -> map bridge.
        # These keep the URDF visible when object_1 disappears.
        self.last_valid_floor_to_base = None
        self.last_valid_floor_to_map = None

        # --------------------------------------------------
        # DEQUES
        # --------------------------------------------------
        maxlen = None if self.deque_maxlen <= 0 else self.deque_maxlen

        self.cam_to_obj1_history = deque(maxlen=maxlen)
        self.cam_to_obj2_history = deque(maxlen=maxlen)
        self.cam_to_obj3_history = deque(maxlen=maxlen)
        self.cam_to_base_history = deque(maxlen=maxlen)
        self.cam_to_floor_history = deque(maxlen=maxlen)
        self.cam_to_secondobstacle_history = deque(maxlen=maxlen)

        self.floor_to_baselink_history = deque(maxlen=maxlen)
        self.floor_to_object2_history = deque(maxlen=maxlen)
        self.floor_to_object3_history = deque(maxlen=maxlen)
        self.floor_to_secondobstacle_history = deque(maxlen=maxlen)

        self.relative_distance_history = deque(maxlen=maxlen)

        # --------------------------------------------------
        # TIMER
        # --------------------------------------------------
        period = 1.0 / max(self.rate_hz, 0.1)
        self.timer = self.create_timer(period, self.tick)

        self.get_logger().info(
            f"Logging TF '{self.parent_frame}' -> '{self.child_frame}' at {self.rate_hz:.1f} Hz"
        )
        self.get_logger().info("Using deques for live transform history.")
        self.get_logger().info(
            "Deques initialized for: camera->object_1, camera->object_2, camera->object_3, camera->base_link"
        )
        self.get_logger().info("Floor latch enabled: last valid camera->floor will be reused if floor tag disappears.")

        # self.json_save_counter = 0
        # self.json_save_every_n_ticks = 10

    # =======================================================
    # DEQUE STORAGE
    # =======================================================
    def append_relative_distance_history(self, timestamp_str, p_base, p_obj2, p_obj3):
        dist_base_to_obj2 = float(np.linalg.norm(p_obj2 - p_base))
        dist_base_to_obj3 = float(np.linalg.norm(p_obj3 - p_base))
        dist_obj2_to_obj3 = float(np.linalg.norm(p_obj3 - p_obj2))

        entry = {
            "timestamp": timestamp_str,
            "base_link_to_object_2": dist_base_to_obj2,
            "base_link_to_object_3": dist_base_to_obj3,
            "object_2_to_object_3": dist_obj2_to_obj3,
        }

        self.relative_distance_history.append(entry)

        # self.get_logger().info(
        #     f"[relative_distances] "
        #     f"base->obj2={dist_base_to_obj2:.6f} m | "
        #     f"base->obj3={dist_base_to_obj3:.6f} m | "
        #     f"obj2->obj3={dist_obj2_to_obj3:.6f} m"
        # )

    def append_transform_history(self, history_deque, name, T_matrix, timestamp_str):
        tx, ty, tz = T_matrix[:3, 3]

        quat = R.from_matrix(T_matrix[:3, :3]).as_quat()
        _, rpy = quat_to_rot(quat[0], quat[1], quat[2], quat[3])

        euclidean_distance = euclidean_distance_from_camera(T_matrix)

        if len(history_deque) == 0:
            trans_err = 0.0
            rot_err = 0.0
            rpy_err = np.array([0.0, 0.0, 0.0])
        else:
            prev_entry = history_deque[-1]
            prev_T = prev_entry["T"]
            prev_rpy = prev_entry["rotation_rpy"]

            trans_err = translation_error(prev_T, T_matrix)
            rot_err = rotation_error(prev_T, T_matrix)
            rpy_err = rpy_error(prev_rpy, rpy)

        entry = {
            "timestamp": timestamp_str,
            "name": name,
            "translation": [tx, ty, tz],
            "rotation_rpy": [rpy[0], rpy[1], rpy[2]],
            "euclidean_distance": float(euclidean_distance),
            "T": T_matrix.copy(),
            "translation_error": float(trans_err),
            "rotation_error": float(rot_err),
            "rotation_error_deg": float(np.degrees(rot_err)),
            "rpy_error": [float(rpy_err[0]), float(rpy_err[1]), float(rpy_err[2])],
        }

        history_deque.append(entry)

        # self.get_logger().info(
        #     f"[{name}] "
        #     f"trans=({tx:.6f}, {ty:.6f}, {tz:.6f}) | "
        #     f"dist_from_camera={euclidean_distance:.6f} m | "
        #     f"rpy=({rpy[0]:.6f}, {rpy[1]:.6f}, {rpy[2]:.6f}) | "
        #     f"Δtrans={trans_err:.6f} m | "
        #     f"Δrot={rot_err:.6f} rad ({np.degrees(rot_err):.3f} deg)"
        # )

    def get_deque_data(self, deque_name):
        deques = {
            'camera_to_object1': self.cam_to_obj1_history,
            'camera_to_object2': self.cam_to_obj2_history,
            'camera_to_object3': self.cam_to_obj3_history,
            'camera_to_baselink': self.cam_to_base_history,
            'camera_to_secondobstacle': self.cam_to_secondobstacle_history,
            'relative_distances': self.relative_distance_history,
            'floor_to_baselink': self.floor_to_baselink_history,
            'floor_to_object2': self.floor_to_object2_history,
            'floor_to_object3': self.floor_to_object3_history,
            'floor_to_secondobstacle': self.floor_to_secondobstacle_history
        }

        if deque_name in deques:
            return list(deques[deque_name])
        return None

    def print_deque_summary(self, deque_name):
        data = self.get_deque_data(deque_name)
        if data is None:
            self.get_logger().info(f"Invalid deque name: {deque_name}")
            return

        self.get_logger().info(f"\n========== {deque_name.upper()} DEQUE SUMMARY ==========")
        self.get_logger().info(f"Total entries: {len(data)}")

        if len(data) > 0:
            latest = data[-1]
            self.get_logger().info(f"Latest timestamp: {latest['timestamp']}")

            if deque_name == "relative_distances":
                self.get_logger().info(f"base_link_to_object_2: {latest['base_link_to_object_2']}")
                self.get_logger().info(f"base_link_to_object_3: {latest['base_link_to_object_3']}")
                self.get_logger().info(f"object_2_to_object_3: {latest['object_2_to_object_3']}")
            else:
                self.get_logger().info(f"Translation (x,y,z): {latest['translation']}")
                self.get_logger().info(f"Euclidean distance from camera: {latest['euclidean_distance']}")
                self.get_logger().info(f"Rotation (r,p,y) rad: {latest['rotation_rpy']}")
                self.get_logger().info(f"Translation Error: {latest['translation_error']}")
                self.get_logger().info(f"Rotation Error (rad): {latest['rotation_error']}")
                self.get_logger().info(f"Rotation Error (deg): {latest['rotation_error_deg']}")
                self.get_logger().info(f"RPY Error: {latest['rpy_error']}")

    # =======================================================
    # SAVE JSON
    # =======================================================
    def save_deques_to_json(self):
        save_dir = "/home/unitree-arka/Desktop/Deque_Debug_Demo_Outputs"
        os.makedirs(save_dir, exist_ok=True)

        data_map = {
            "camera_to_object_1.json": list(self.cam_to_obj1_history),
            "camera_to_object_2.json": list(self.cam_to_obj2_history),
            "camera_to_object_3.json": list(self.cam_to_obj3_history),
            "camera_to_base_link.json": list(self.cam_to_base_history),
            "camera_to_floor.json": list(self.cam_to_floor_history),
            "camera_to_secondobstacle.json":list(self.cam_to_secondobstacle_history),
            "floor_to_base_link.json": list(self.floor_to_baselink_history),
            "floor_to_object_2.json": list(self.floor_to_object2_history),
            "floor_to_object_3.json": list(self.floor_to_object3_history),
            "floor_to_secondobstacle.json": list(self.floor_to_secondobstacle_history),
            "relative_distances.json": list(self.relative_distance_history),
        }

        for filename, data in data_map.items():
            serializable_data = []
            for entry in data:
                serializable_entry = dict(entry)
                if "T" in serializable_entry and isinstance(serializable_entry["T"], np.ndarray):
                    serializable_entry["T"] = serializable_entry["T"].tolist()
                serializable_data.append(serializable_entry)

            with open(os.path.join(save_dir, filename), "w") as f:
                json.dump(serializable_data, f, indent=2)



    def transform_to_matrix(self, transform_stamped):
        """Convert a TransformStamped message into a 4x4 matrix."""
        tr = transform_stamped.transform.translation
        q = transform_stamped.transform.rotation

        rotation_matrix, _ = quat_to_rot(
            q.x,
            q.y,
            q.z,
            q.w
        )

        matrix = np.eye(4, dtype=float)
        matrix[:3, :3] = rotation_matrix
        matrix[:3, 3] = np.array([
            tr.x,
            tr.y,
            tr.z
        ])

        return matrix
    
    def get_latched_floor_transform(
        self,
        raw_child_frame,
        P_floor_to_C,
        cache_attribute
    ):
        """
        Obtain floor -> target from the latest camera -> target detection.

        If the raw AprilTag transform is unavailable, reuse the last valid
        floor-based transform stored in cache_attribute.
        """
        try:
            detected_transform = self.buffer.lookup_transform(
                self.parent_frame,
                raw_child_frame,
                Time(),
                timeout=Duration(seconds=0.2)
            )

            P_C_to_target = self.transform_to_matrix(
                detected_transform
            )

            P_floor_to_target = (
                P_floor_to_C @ P_C_to_target
            )

            setattr(
                self,
                cache_attribute,
                P_floor_to_target.copy()
            )

            return P_floor_to_target, True

        except Exception as error:
            cached_transform = getattr(
                self,
                cache_attribute
            )

            if cached_transform is not None:
                return cached_transform.copy(), False

            self.get_logger().warn(
                f"No transform or previous lock exists for "
                f"'{raw_child_frame}': {error}"
            )

            return None, False
        

    def broadcast_locked_robot_pose(self, stamp_msg):
        """
        Publish the last valid robot/world bridge.

        Returns True when a cached robot pose exists.
        """
        if (
            self.last_valid_floor_to_map is None
            or self.last_valid_floor_to_base is None
        ):
            return False

        self.broadcast_transform(
            self.floor_frame,
            "map",
            self.last_valid_floor_to_map,
            stamp_msg
        )

        self.broadcast_transform(
            self.floor_frame,
            "base_link_floor",
            self.last_valid_floor_to_base,
            stamp_msg
        )

        return True
    
    def offset_transform_local(self, T_matrix, dx=0.0, dy=0.0, dz=0.0):
        """
        Apply an offset in the local coordinate frame of T_matrix.

        This is useful because the AprilTag frame is usually at the tag centre,
        while the cube marker should be centred inside the physical object.
        """
        T_offset = np.eye(4, dtype=float)
        T_offset[:3, 3] = np.array([dx, dy, dz], dtype=float)
        return T_matrix @ T_offset


    def matrix_to_marker_pose(self, T_matrix):
        """
        Convert a 4x4 transform matrix into a Marker pose.
        """
        from geometry_msgs.msg import Pose

        pose = Pose()

        pose.position.x = float(T_matrix[0, 3])
        pose.position.y = float(T_matrix[1, 3])
        pose.position.z = float(T_matrix[2, 3])

        quat = R.from_matrix(T_matrix[:3, :3]).as_quat()
        pose.orientation.x = float(quat[0])
        pose.orientation.y = float(quat[1])
        pose.orientation.z = float(quat[2])
        pose.orientation.w = float(quat[3])

        return pose


    def make_cube_marker(
        self,
        marker_id,
        name,
        T_floor_to_cube_center,
        size_xyz,
        color_rgba,
        stamp_msg
    ):
        """
        Create one RViz cube marker in the floor frame.
        """
        marker = Marker()

        marker.header.frame_id = self.floor_frame
        marker.header.stamp = stamp_msg

        marker.ns = "go2_environment_cubes"
        marker.id = marker_id
        marker.type = Marker.CUBE
        marker.action = Marker.ADD

        marker.pose = self.matrix_to_marker_pose(T_floor_to_cube_center)

        marker.scale.x = float(size_xyz[0])
        marker.scale.y = float(size_xyz[1])
        marker.scale.z = float(size_xyz[2])

        marker.color.r = float(color_rgba[0])
        marker.color.g = float(color_rgba[1])
        marker.color.b = float(color_rgba[2])
        marker.color.a = float(color_rgba[3])

        marker.lifetime = Duration(seconds=0.0).to_msg()

        return marker


    #--- Adding Cyclinder Marker-------------------######
    def make_cylinder_marker(
        self,
        marker_id,
        name,
        T_floor_to_circle_center,
        radius,
        thickness,
        color_rgba,
        stamp_msg
    ):
        """
        Create one flat circular goal marker in the floor frame.

        RViz CYLINDER uses:
        scale.x = diameter
        scale.y = diameter
        scale.z = height/thickness
        """
        marker = Marker()

        marker.header.frame_id = self.floor_frame
        marker.header.stamp = stamp_msg

        marker.ns = "go2_environment_cubes"
        marker.id = marker_id
        marker.type = Marker.CYLINDER
        marker.action = Marker.ADD

        marker.pose = self.matrix_to_marker_pose(T_floor_to_circle_center)

        marker.scale.x = float(2.0 * radius)
        marker.scale.y = float(2.0 * radius)
        marker.scale.z = float(thickness)

        marker.color.r = float(color_rgba[0])
        marker.color.g = float(color_rgba[1])
        marker.color.b = float(color_rgba[2])
        marker.color.a = float(color_rgba[3])

        marker.lifetime = Duration(seconds=0.0).to_msg()

        return marker

    def make_axis_arrow_marker(
        self,
        marker_id,
        name,
        start_xyz,
        end_xyz,
        color_rgba,
        stamp_msg
    ):
        """
        Create one RViz arrow marker in the floor frame.

        For Marker.ARROW with two points:
        scale.x = shaft diameter
        scale.y = head diameter
        scale.z = head length
        """
        marker = Marker()

        marker.header.frame_id = self.floor_frame
        marker.header.stamp = stamp_msg

        marker.ns = "go2_floor_axes"
        marker.id = marker_id
        marker.type = Marker.ARROW
        marker.action = Marker.ADD

        marker.pose.orientation.w = 1.0

        start = Point()
        start.x = float(start_xyz[0])
        start.y = float(start_xyz[1])
        start.z = float(start_xyz[2])

        end = Point()
        end.x = float(end_xyz[0])
        end.y = float(end_xyz[1])
        end.z = float(end_xyz[2])

        marker.points.append(start)
        marker.points.append(end)

        marker.scale.x = 0.025   # shaft diameter
        marker.scale.y = 0.080   # head diameter
        marker.scale.z = 0.120   # head length

        marker.color.r = float(color_rgba[0])
        marker.color.g = float(color_rgba[1])
        marker.color.b = float(color_rgba[2])
        marker.color.a = float(color_rgba[3])

        marker.lifetime = Duration(seconds=0.0).to_msg()

        return marker

    def append_floor_axis_markers(self, marker_array, stamp_msg):
        """
        Add X and Y floor-frame axes using planner.cfg bounds.
        """
        x_min = self.floor_bounds["x_min"]
        x_max = self.floor_bounds["x_max"]
        y_min = self.floor_bounds["y_min"]
        y_max = self.floor_bounds["y_max"]

        z_axis = 0.025

        # X-axis: from x_min to x_max at y = 0
        marker_array.markers.append(
            self.make_axis_arrow_marker(
                marker_id=100,
                name="floor_x_axis",
                start_xyz=(x_min, 0.0, z_axis),
                end_xyz=(x_max, 0.0, z_axis),
                color_rgba=(1.0, 0.0, 0.0, 0.80),
                stamp_msg=stamp_msg
            )
        )

        # Y-axis: from y_min to y_max at x = 0
        marker_array.markers.append(
            self.make_axis_arrow_marker(
                marker_id=101,
                name="floor_y_axis",
                start_xyz=(0.0, y_min, z_axis),
                end_xyz=(0.0, y_max, z_axis),
                color_rgba=(0.0, 1.0, 0.0, 0.80),
                stamp_msg=stamp_msg
            )
        )



    def publish_environment_cubes(
        self,
        P_floor_to_obj2,
        P_floor_to_obj3,
        P_floor_to_second_obstacle,
        stamp_msg
    ):
        """
        Publish obstacle and goal cube markers for RViz.

        The TF frames are located at AprilTag centres. Since the obstacle tags
        are placed on top of the physical objects, the cube centre should be
        shifted downward by half of the object height in the tag/object local
        z direction.

        If a cube appears above the tag instead of below it, change the sign
        of the z offset from negative to positive.
        """
        marker_array = MarkerArray()
        self.append_floor_axis_markers(marker_array, stamp_msg)

        # object_2 obstacle: 0.25 x 0.35 x 0.48
        if P_floor_to_obj2 is not None:
            obj2_height = self.object2_cube_size[2]

            P_floor_to_obj2_cube_center = self.offset_transform_local(
                P_floor_to_obj2,
                dz=-0.5 * obj2_height
            )

            marker_array.markers.append(
                self.make_cube_marker(
                    marker_id=0,
                    name="object_2_cube",
                    T_floor_to_cube_center=P_floor_to_obj2_cube_center,
                    size_xyz=self.object2_cube_size,
                    color_rgba=(1.0, 0.0, 0.0, 0.45),
                    stamp_msg=stamp_msg
                )
            )

        # goal tag: 0.32 x 0.54 x 0.001
       # goal: flat circular target region, radius 21.5 cm
        if P_floor_to_obj3 is not None:
            P_floor_to_goal_circle_center = self.offset_transform_local(
                P_floor_to_obj3,
                dz=-0.5 * self.goal_circle_thickness
            )

            marker_array.markers.append(
                self.make_cylinder_marker(
                    marker_id=1,
                    name="goal_circle",
                    T_floor_to_circle_center=P_floor_to_goal_circle_center,
                    radius=self.goal_circle_radius,
                    thickness=self.goal_circle_thickness,
                    color_rgba=(0.0, 1.0, 0.0, 0.45),
                    stamp_msg=stamp_msg
                )
            )

        # new_obstacle: 0.57 x 0.42 x 0.18
        if P_floor_to_second_obstacle is not None:
            new_obs_height = self.new_obstacle_cube_size[2]

            P_floor_to_new_obstacle_cube_center = self.offset_transform_local(
                P_floor_to_second_obstacle,
                dz=-0.5 * new_obs_height
            )

            marker_array.markers.append(
                self.make_cube_marker(
                    marker_id=2,
                    name="new_obstacle_cube",
                    T_floor_to_cube_center=P_floor_to_new_obstacle_cube_center,
                    size_xyz=self.new_obstacle_cube_size,
                    color_rgba=(0.7, 0.0, 1.0, 0.45),
                    stamp_msg=stamp_msg
                )
            )

        self.marker_pub.publish(marker_array)


    # =======================================================
    # BROADCAST HELPER
    # =======================================================
    def broadcast_transform(self, parent, child, T_matrix, stamp_msg):
        msg = TransformStamped()
        msg.header.stamp = stamp_msg
        msg.header.frame_id = parent
        msg.child_frame_id = child
        msg.transform.translation.x = float(T_matrix[0, 3])
        msg.transform.translation.y = float(T_matrix[1, 3])
        msg.transform.translation.z = float(T_matrix[2, 3])

        quat = R.from_matrix(T_matrix[:3, :3]).as_quat()
        msg.transform.rotation.x = float(quat[0])
        msg.transform.rotation.y = float(quat[1])
        msg.transform.rotation.z = float(quat[2])
        msg.transform.rotation.w = float(quat[3])

        self.broadcaster.sendTransform(msg)

    # =======================================================
    # MAIN TICK
    # =======================================================
    def tick(self):

                # --------------------------------------------------
        # CAMERA -> RAW FLOOR TAG
        #
        # The AprilTag node publishes:
        # camera -> floor_tag_raw
        #
        # We invert this measurement and continuously publish:
        # floor -> camera
        #
        # When the floor tag disappears, the final valid
        # measurement remains latched and is reused.
        # --------------------------------------------------
        T_floor = None
        floor_visible_now = False

        try:
            t_floor = self.buffer.lookup_transform(
                self.parent_frame,
                self.floor_tag_frame,
                Time(),
                timeout=Duration(seconds=0.2)
            )

            tr_floor = t_floor.transform.translation
            q_floor = t_floor.transform.rotation

            rot_matrix_floor, _ = quat_to_rot(
                q_floor.x,
                q_floor.y,
                q_floor.z,
                q_floor.w
            )

            T_floor = np.eye(4, dtype=float)
            T_floor[:3, :3] = rot_matrix_floor
            T_floor[:3, 3] = np.array([
                tr_floor.x,
                tr_floor.y,
                tr_floor.z
            ])

            # Save the latest valid camera -> floor-tag transform.
            self.last_valid_C_to_floor = T_floor.copy()
            self.have_floor_lock = True
            floor_visible_now = True

            self.get_logger().info(
                "Raw floor tag visible: updating floor lock."
            )

        except Exception as e:
            if self.have_floor_lock:
                T_floor = self.last_valid_C_to_floor.copy()

                self.get_logger().warn(
                    f"Raw floor tag unavailable: {e}. "
                    f"Using latched camera -> floor transform."
                )
            else:
                self.get_logger().warn(
                    f"Raw floor tag unavailable and no floor lock exists: {e}"
                )
                return

        # Convert camera -> floor-tag into persistent floor -> camera.
        P_floor_to_C = np.linalg.inv(T_floor)

        # Keep floor -> camera alive even when the raw tag disappears.
        floor_stamp_msg = self.get_clock().now().to_msg()

        self.broadcast_transform(
            self.floor_frame,
            self.parent_frame,
            P_floor_to_C,
            floor_stamp_msg
        )

        # --------------------------------------------------
        # INDEPENDENT PERSISTENT WORLD-OBJECT LOCKS
        # --------------------------------------------------
        P_floor_to_obj2, object2_visible = (
            self.get_latched_floor_transform(
                self.intermediate_frame,
                P_floor_to_C,
                "last_valid_floor_to_object2"
            )
        )

        P_floor_to_obj3, object3_visible = (
            self.get_latched_floor_transform(
                self.goal_frame,
                P_floor_to_C,
                "last_valid_floor_to_object3"
            )
        )

        P_floor_to_second_obstacle, second_obstacle_visible = (
            self.get_latched_floor_transform(
                self.second_obstacle_frame,
                P_floor_to_C,
                "last_valid_floor_to_new_obstacle"
            )
        )

        persistent_stamp = self.get_clock().now().to_msg()


        if P_floor_to_obj2 is not None:
            self.broadcast_transform(
                self.floor_frame,
                "object_2_floor",
                P_floor_to_obj2,
                persistent_stamp
            )

        if P_floor_to_obj3 is not None:
            self.broadcast_transform(
                self.floor_frame,
                "object_3_floor",
                P_floor_to_obj3,
                persistent_stamp
            )

        if P_floor_to_second_obstacle is not None:
            self.broadcast_transform(
                self.floor_frame,
                "new_obstacle_floor",
                P_floor_to_second_obstacle,
                persistent_stamp
            )

        # --------------------------------------------------
        # RViz CUBE MARKERS FOR OBSTACLES AND GOAL
        # --------------------------------------------------
        self.publish_environment_cubes(
            P_floor_to_obj2,
            P_floor_to_obj3,
            P_floor_to_second_obstacle,
            persistent_stamp
        )



        # --------------------------------------------------
        # CAMERA -> OBJECT_1
        # --------------------------------------------------
        try:
            t = self.buffer.lookup_transform(
                self.parent_frame,
                self.child_frame,
                Time(),
                timeout=Duration(seconds=0.2)
            )
        except Exception as e:
            self.get_logger().warn(
                f"Robot-mounted object_1 unavailable: {e}. "
                f"Using the last valid robot pose."
            )

            # Static object frames were already processed above.
            # Keep the robot frozen at its last valid floor pose.
            self.broadcast_locked_robot_pose(
                self.get_clock().now().to_msg()
            )

            return

        common_time = Time.from_msg(t.header.stamp)
        stamp_msg = common_time.to_msg()
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        tr = t.transform.translation
        q = t.transform.rotation
        rot_matrix, roll_pitch_yaw_radians = quat_to_rot(q.x, q.y, q.z, q.w)

        T = np.eye(4, dtype=float)
        T[:3, :3] = rot_matrix
        T[:3, 3] = np.array([tr.x, tr.y, tr.z])


        # --------------------------------------------------
        # CAMERA -> TAG-T1
        # --------------------------------------------------
        P_C_to_T1 = T

        # --------------------------------------------------
        # BASE -> HEAD_UPPER
        # --------------------------------------------------
        P_base_to_Head_Upper = np.eye(4)
        rot_matrix_base_to_Head_upper, _ = quat_to_rot(0, 0, 0, 1)
        P_base_to_Head_Upper[:3, :3] = rot_matrix_base_to_Head_upper
        P_base_to_Head_Upper[:3, 3] = np.array([0.285, 0, 0.01])

        # --------------------------------------------------
        # HEAD_UPPER -> Rc
        # --------------------------------------------------
        P_Head_Upper_to_Rc = np.eye(4)
        rot_matrix_Head_Upper_to_Rc, _ = quat_to_rot(-0.5, 0.500002, -0.5, 0.499998)
        P_Head_Upper_to_Rc[:3, :3] = rot_matrix_Head_Upper_to_Rc
        P_Head_Upper_to_Rc[:3, 3] = np.array([0.045, 0, 0.03])

        # --------------------------------------------------
        # BASE -> Rc / BASE -> T1
        # --------------------------------------------------
        P_base_to_Rc = P_base_to_Head_Upper @ P_Head_Upper_to_Rc
        P_base_to_T1 = P_base_to_Head_Upper @ P_Head_Upper_to_Rc @ P_star_Rc_to_T1

        # --------------------------------------------------
        # T1 -> Rc / T1 -> BASE
        # --------------------------------------------------
        P_T1_to_Rc = np.linalg.inv(P_star_Rc_to_T1)
        P_T1_to_base = P_T1_to_Rc @ np.linalg.inv(P_base_to_Rc)

        # --------------------------------------------------
        # CAMERA -> BASE
        # --------------------------------------------------
        P_C_to_base = P_C_to_T1 @ P_T1_to_base

        # --------------------------------------------------
        # FLOOR -> BASE / OBJECTS
        # --------------------------------------------------
         # floor -> camera
        #P_floor_to_C = np.linalg.inv(T_floor)
        P_floor_to_base = P_floor_to_C @ P_C_to_base
        quat_floor_to_base = R.from_matrix(P_floor_to_base[:3, :3]).as_quat()
        _, rpy_floor_to_base = quat_to_rot(
            quat_floor_to_base[0], quat_floor_to_base[1], quat_floor_to_base[2], quat_floor_to_base[3]
        )


    


        #---- computation of transformation from floor to map of the robot urdf----####
                # --------------------------------------------------
        # MAP -> BASE_LINK FROM THE EXISTING URDF TF TREE
        # --------------------------------------------------
        try:
            t_map_to_base = self.buffer.lookup_transform(
                "map",
                "base_link",
                common_time,
                timeout=Duration(seconds=0.2)
            )
        except Exception as e:
            self.get_logger().warn(
                f"TF map -> base_link unavailable: {e}. "
                f"Using the last valid robot bridge."
            )

            self.broadcast_locked_robot_pose(
                self.get_clock().now().to_msg()
            )
            return

        tr_map_base = t_map_to_base.transform.translation
        q_map_base = t_map_to_base.transform.rotation

        R_map_to_base, _ = quat_to_rot(
            q_map_base.x,
            q_map_base.y,
            q_map_base.z,
            q_map_base.w
        )

        P_map_to_base = np.eye(4, dtype=float)
        P_map_to_base[:3, :3] = R_map_to_base
        P_map_to_base[:3, 3] = np.array([
            tr_map_base.x,
            tr_map_base.y,
            tr_map_base.z
        ])

        # floor->base = floor->map @ map->base
        # Therefore:
        # floor->map = floor->base @ inverse(map->base)
        P_floor_to_map = (
            P_floor_to_base @ np.linalg.inv(P_map_to_base)
        )

        # Save the latest valid robot localization result.
        self.last_valid_floor_to_base = (
            P_floor_to_base.copy()
        )

        self.last_valid_floor_to_map = (
            P_floor_to_map.copy()
        )
      
        self.broadcast_locked_robot_pose(
            self.get_clock().now().to_msg()
        )


        # self.append_transform_history(
        #     self.cam_to_base_history,
        #     "camera_to_base_link",
        #     P_C_to_base,
        #     ts
        # )

        

        # self.append_transform_history(
        #     self.floor_to_baselink_history,
        #     "floor_to_baselink",
        #     P_floor_to_base,
        #     ts
        # )


        # self.json_save_counter += 1
        # if self.json_save_counter >= self.json_save_every_n_ticks:
        #     self.save_deques_to_json()
        #     self.json_save_counter = 0


# ============================================================
# MAIN
# ============================================================
def main():
    rclpy.init()
    node = TFLogger()
    try:
        rclpy.spin(node)
    finally:
        #node.save_deques_to_json()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()