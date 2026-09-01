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
from geometry_msgs.msg import TransformStamped

from scipy.spatial.transform import Rotation as R
import json


# ============================================================
# FIXED CALIBRATION MATRIX P*(Rc->T1)
# ============================================================
# P_star_Rc_to_T1 = np.array([
#     [0.999837, -0.002961, -0.017820,  0.009035],
#     [-0.017815, 0.001778, -0.999840, -0.056825],
#     [0.002993,  0.999994,  0.001725, -0.084882],
#     [0.000000,  0.000000,  0.000000,  1.000000]
# ])

P_star_Rc_to_T1 = np.array([
    [0.998434, -0.051057, -0.022868, 0.002303],
    [-0.023986, -0.021393, -0.999483, -0.047973],
    [0.050541, 0.998467, -0.022584, -0.089412],
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

    return np.arccos(cos_theta)  # radians


def rpy_error(prev_rpy, curr_rpy):
    """Absolute component-wise RPY difference."""
    prev_rpy = np.array(prev_rpy, dtype=float)
    curr_rpy = np.array(curr_rpy, dtype=float)
    return np.abs(curr_rpy - prev_rpy)


def euclidean_distance_from_camera(T_matrix):
    p = T_matrix[:3, 3]
    return np.linalg.norm(p)


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
        self.declare_parameter('floor_frame', 'floor')  # new parameter for floor frame
        self.declare_parameter('rate_hz', 10.0)
        self.declare_parameter('deque_maxlen', 1000)


        # Live bridge file for Docker
        self.declare_parameter(
            'bridge_pose_file',
            '/home/arka/unitree_sdk2/live_bridge/Trial_floor_robot_pose_one.txt'
        )


       

        self.parent_frame = self.get_parameter('parent_frame').get_parameter_value().string_value
        self.child_frame = self.get_parameter('child_frame').get_parameter_value().string_value
        self.intermediate_frame = self.get_parameter('intermediate_frame').get_parameter_value().string_value
        self.goal_frame = self.get_parameter('goal_frame').get_parameter_value().string_value

        ##----- adding the floor frame-----------###
        self.floor_frame = self.get_parameter('floor_frame').get_parameter_value().string_value

        self.rate_hz = self.get_parameter('rate_hz').get_parameter_value().double_value
        self.deque_maxlen = self.get_parameter('deque_maxlen').get_parameter_value().integer_value

        self.bridge_pose_file = self.get_parameter('bridge_pose_file').get_parameter_value().string_value

        # --------------------------------------------------
        # TF BUFFER / LISTENER / BROADCASTER
        # --------------------------------------------------
        self.buffer = tf2_ros.Buffer(cache_time=Duration(seconds=10.0))
        self.listener = tf2_ros.TransformListener(self.buffer, self, spin_thread=False)
        self.broadcaster = tf2_ros.TransformBroadcaster(self)

        # --------------------------------------------------
        # DEQUES
        # --------------------------------------------------
        maxlen = None if self.deque_maxlen <= 0 else self.deque_maxlen

        self.cam_to_obj1_history = deque(maxlen=maxlen)
        self.cam_to_obj2_history = deque(maxlen=maxlen)
        self.cam_to_obj3_history = deque(maxlen=maxlen)
        self.cam_to_base_history = deque(maxlen=maxlen)
        ##----- preserving the camera to floor history as well ---------##
        self.cam_to_floor_history = deque(maxlen=maxlen)
        
        ####------ saving floor to base-link, tag-2 and tag-3 transformations------####
        self.floor_to_baselink_history = deque(maxlen=maxlen)
        self.floor_to_object2_history = deque(maxlen=maxlen)
        self.floor_to_object3_history = deque(maxlen=maxlen)

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

        self.json_save_counter = 0 
        self.json_save_every_n_ticks = 10   # save once every 10 ticks = once per second at 10 Hz

    def append_relative_distance_history(
        self,
        timestamp_str,
        p_base,
        p_obj2,
        p_obj3
    ):
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

        self.get_logger().info(
            f"[relative_distances] "
            f"base->obj2={dist_base_to_obj2:.6f} m | "
            f"base->obj3={dist_base_to_obj3:.6f} m | "
            f"obj2->obj3={dist_obj2_to_obj3:.6f} m"
        )



    # =======================================================
    # DEQUE STORAGE
    # =======================================================
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

        self.get_logger().info(
            f"[{name}] "
            f"trans=({tx:.6f}, {ty:.6f}, {tz:.6f}) | "
            f"dist_from_camera={euclidean_distance:.6f} m | "
            f"rpy=({rpy[0]:.6f}, {rpy[1]:.6f}, {rpy[2]:.6f}) | "
            f"Δtrans={trans_err:.6f} m | "
            f"Δrot={rot_err:.6f} rad ({np.degrees(rot_err):.3f} deg)"
        )

    def get_deque_data(self, deque_name):
        deques = {
            'camera_to_object1': self.cam_to_obj1_history,
            'camera_to_object2': self.cam_to_obj2_history,
            'camera_to_object3': self.cam_to_obj3_history,
            'camera_to_baselink': self.cam_to_base_history,
            'relative_distances': self.relative_distance_history,
            'floor_to_baselink':self.floor_to_baselink_history,
            'floor_to_object2':self.floor_to_object2_history,
            'floor_to_object3':self.floor_to_object3_history
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

   ### Saving Deque Files In Json Format ####

    def save_deques_to_json(self):
        save_dir = "/home/arka/Desktop/Deque_Debug_Demo_Outputs"
        os.makedirs(save_dir, exist_ok=True)

        data_map = {
            "camera_to_object_1.json": list(self.cam_to_obj1_history),
            "camera_to_object_2.json": list(self.cam_to_obj2_history),
            "camera_to_object_3.json": list(self.cam_to_obj3_history),
            "camera_to_base_link.json": list(self.cam_to_base_history),
            "camera_to_floor.json": list(self.cam_to_floor_history),
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



    # =======================================================
    # MAIN TICK
    # =======================================================
    def tick(self):
        try:
            t = self.buffer.lookup_transform(
                self.parent_frame,
                self.child_frame,
                Time(),
                timeout=Duration(seconds=0.2)
            )
        except Exception as e:
            self.get_logger().warn(f"TF for object_1 not available yet: {e}")
            return

        common_time = Time.from_msg(t.header.stamp)

        # --------------------------------------------------
        # CAMERA -> OBJECT_1
        # --------------------------------------------------
        tr = t.transform.translation
        q = t.transform.rotation
        rot_matrix, roll_pitch_yaw_radians = quat_to_rot(q.x, q.y, q.z, q.w)

        T = np.eye(4, dtype=float)
        T[:3, :3] = rot_matrix
        T[:3, 3] = np.array([tr.x, tr.y, tr.z])
        r, p, y = roll_pitch_yaw_radians

        # --------------------------------------------------
        # CAMERA -> OBJECT_2
        # --------------------------------------------------
        try:
            t2 = self.buffer.lookup_transform(
                self.parent_frame,
                self.intermediate_frame,
                common_time,
                timeout=Duration(seconds=0.2)
            )
        except Exception as e:
            self.get_logger().warn(f"TF for object_2 not available at common time: {e}")
            return

        tr2 = t2.transform.translation
        q2 = t2.transform.rotation
        rot_matrix2, roll_pitch_yaw_radians2 = quat_to_rot(q2.x, q2.y, q2.z, q2.w)

        T2 = np.eye(4, dtype=float)
        T2[:3, :3] = rot_matrix2
        T2[:3, 3] = np.array([tr2.x, tr2.y, tr2.z])
        r2, p2, y2 = roll_pitch_yaw_radians2

        # --------------------------------------------------
        # CAMERA -> OBJECT_3
        # --------------------------------------------------
        try:
            t3 = self.buffer.lookup_transform(
                self.parent_frame,
                self.goal_frame,
                common_time,
                timeout=Duration(seconds=0.2)
            )
        except Exception as e:
            self.get_logger().warn(f"TF for object_3 not available at common time: {e}")
            return

        tr3 = t3.transform.translation
        q3 = t3.transform.rotation
        rot_matrix3, roll_pitch_yaw_radians3 = quat_to_rot(q3.x, q3.y, q3.z, q3.w)

        T3 = np.eye(4, dtype=float)
        T3[:3, :3] = rot_matrix3
        T3[:3, 3] = np.array([tr3.x, tr3.y, tr3.z])
        r3, p3, y3 = roll_pitch_yaw_radians3


        ##--- CAMERA -> FLOOR (new) ---##
        try:
            t_floor = self.buffer.lookup_transform(
                self.parent_frame,
                self.floor_frame,
                common_time,
                timeout=Duration(seconds=0.2)
            )
        except Exception as e:
            self.get_logger().warn(f"TF for floor not available at common time: {e}")
            return
        tr_floor = t_floor.transform.translation
        q_floor = t_floor.transform.rotation
        rot_matrix_floor, roll_pitch_yaw_radians_floor = quat_to_rot(q_floor.x, q_floor.y, q_floor.z, q_floor.w)

        T_floor = np.eye(4, dtype=float)
        T_floor[:3, :3] = rot_matrix_floor
        T_floor[:3, 3] = np.array([tr_floor.x, tr_floor.y, tr_floor.z])
        r_floor, p_floor, y_floor = roll_pitch_yaw_radians_floor




        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        stamp_msg = common_time.to_msg()

        self.get_logger().info(
            f"Common time = {t.header.stamp.sec}.{t.header.stamp.nanosec:09d}, "
            f"object_2 time = {t2.header.stamp.sec}.{t2.header.stamp.nanosec:09d}, "
            f"object_3 time = {t3.header.stamp.sec}.{t3.header.stamp.nanosec:09d}"
        )

        # --------------------------------------------------
        # CAMERA -> TAG-T1
        # --------------------------------------------------
        P_C_to_T1 = T

        # Tag -> Camera
        P_T1_to_C = np.linalg.inv(P_C_to_T1)
        quat_T1_to_C = R.from_matrix(P_T1_to_C[:3, :3]).as_quat()
        _, rpy_T1_to_C = quat_to_rot(
            quat_T1_to_C[0], quat_T1_to_C[1], quat_T1_to_C[2], quat_T1_to_C[3]
        )

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

        # BASE -> Rc
        P_base_to_Rc = P_base_to_Head_Upper @ P_Head_Upper_to_Rc

        # BASE -> T1
        P_base_to_T1 = P_base_to_Head_Upper @ P_Head_Upper_to_Rc @ P_star_Rc_to_T1

        # T1 -> Rc
       #P_T1_to_Rc = np.linalg.inv(P_base_to_T1) @ P_base_to_Rc
        P_T1_to_Rc = np.linalg.inv(P_star_Rc_to_T1)
        quat_T1_to_Rc = R.from_matrix(P_T1_to_Rc[:3, :3]).as_quat()
        _, rpy_T1_to_Rc = quat_to_rot(
            quat_T1_to_Rc[0], quat_T1_to_Rc[1], quat_T1_to_Rc[2], quat_T1_to_Rc[3]
        )

        # T1 -> BASE
        P_T1_to_base = P_T1_to_Rc @ np.linalg.inv(P_base_to_Rc)
        quat_T1_to_base = R.from_matrix(P_T1_to_base[:3, :3]).as_quat()
        _, rpy_T1_to_base = quat_to_rot(
            quat_T1_to_base[0], quat_T1_to_base[1], quat_T1_to_base[2], quat_T1_to_base[3]
        )

        # --------------------------------------------------
        # CAMERA -> BASE
        # --------------------------------------------------
        P_C_to_base = P_C_to_T1 @ P_T1_to_base
        quat_C_to_base = R.from_matrix(P_C_to_base[:3, :3]).as_quat()
        _, rpy_C_to_base = quat_to_rot(
            quat_C_to_base[0], quat_C_to_base[1], quat_C_to_base[2], quat_C_to_base[3]
        )
        ##------Getting the correct transformations from a fixed floor -----####
        ##--- Floor --> Camera (new)--##
        P_floor_to_C = np.linalg.inv(T_floor)
        quat_floor_to_C = R.from_matrix(P_floor_to_C[:3, :3]).as_quat()
        _, rpy_floor_to_C = quat_to_rot(
            quat_floor_to_C[0], quat_floor_to_C[1], quat_floor_to_C[2], quat_floor_to_C[3]
        )

        ##---Floor-> Base (new)---##
        P_floor_to_base = P_floor_to_C @ P_C_to_base
        quat_floor_to_base = R.from_matrix(P_floor_to_base[:3, :3]).as_quat()
        _, rpy_floor_to_base = quat_to_rot(
            quat_floor_to_base[0], quat_floor_to_base[1], quat_floor_to_base[2], quat_floor_to_base[3]
        )

        ###--- Floor -> Object_2 (new) ---###
        P_floor_to_obj2 = P_floor_to_C @ T2
        quat_floor_to_obj2 = R.from_matrix(P_floor_to_obj2[:3, :3]).as_quat()
        _, rpy_floor_to_obj2 = quat_to_rot(
            quat_floor_to_obj2[0], quat_floor_to_obj2[1], quat_floor_to_obj2[2], quat_floor_to_obj2[3]
        )

        ###--- Floor -> Object_3 (new) ---###
        P_floor_to_obj3 = P_floor_to_C @ T3
        quat_floor_to_obj3 = R.from_matrix(P_floor_to_obj3[:3, :3]).as_quat()
        _, rpy_floor_to_obj3 = quat_to_rot(
            quat_floor_to_obj3[0], quat_floor_to_obj3[1], quat_floor_to_obj3[2], quat_floor_to_obj3[3]
        )







        # --------------------------------------------------
        # BRIDGE FILE FOR DOCKER
        # --------------------------------------------------
        bridge_dir = os.path.dirname(self.bridge_pose_file)
        if bridge_dir:
            os.makedirs(bridge_dir, exist_ok=True)

        x_bridge = P_C_to_base[0, 3]
        z_bridge = P_C_to_base[2, 3]
        theta_bridge = rpy_C_to_base[1]   # kept as before for now

        with open(self.bridge_pose_file, 'w') as f:
            f.write(f"{x_bridge:.6f} {z_bridge:.6f} {theta_bridge:.6f}\n")

        self.get_logger().info(
            f"Bridge pose written: x={x_bridge:.6f}, z={z_bridge:.6f}, theta={theta_bridge:.6f}"
        )

        # --------------------------------------------------
        # CAMERA -> Rc
        # --------------------------------------------------
        P_C_to_Rc = P_C_to_T1 @ P_T1_to_Rc

        # Rc -> obstacle
        P_Rc_to_ObstacleFoam = np.linalg.inv(P_C_to_Rc) @ T2
        # Rc -> goal
        P_Rc_to_GoalFoam = np.linalg.inv(P_C_to_Rc) @ T3

        # --------------------------------------------------
        # BROADCASTS
        # --------------------------------------------------
        
        ###-- floor -> camera (new) --###
        msg_floor_camera = TransformStamped()
        msg_floor_camera.header.stamp = stamp_msg
        msg_floor_camera.header.frame_id = self.floor_frame
        msg_floor_camera.child_frame_id = self.parent_frame
        msg_floor_camera.transform.translation.x = P_floor_to_C[0, 3]
        msg_floor_camera.transform.translation.y = P_floor_to_C[1, 3]
        msg_floor_camera.transform.translation.z = P_floor_to_C[2, 3]
        quat_floor_to_C = R.from_matrix(P_floor_to_C[:3, :3]).as_quat()
        msg_floor_camera.transform.rotation.x = quat_floor_to_C[0]
        msg_floor_camera.transform.rotation.y = quat_floor_to_C[1]
        msg_floor_camera.transform.rotation.z = quat_floor_to_C[2]
        msg_floor_camera.transform.rotation.w = quat_floor_to_C[3]
        self.broadcaster.sendTransform(msg_floor_camera)



        # camera -> base_link
        msg = TransformStamped()
        msg.header.stamp = stamp_msg
        msg.header.frame_id = self.parent_frame
        msg.child_frame_id = "base_link"
        msg.transform.translation.x = P_C_to_base[0, 3]
        msg.transform.translation.y = P_C_to_base[1, 3]
        msg.transform.translation.z = P_C_to_base[2, 3]
        quat_base = R.from_matrix(P_C_to_base[:3, :3]).as_quat()
        msg.transform.rotation.x = quat_base[0]
        msg.transform.rotation.y = quat_base[1]
        msg.transform.rotation.z = quat_base[2]
        msg.transform.rotation.w = quat_base[3]
        self.broadcaster.sendTransform(msg)

        # camera -> object_1
        msg2 = TransformStamped()
        msg2.header.stamp = stamp_msg
        msg2.header.frame_id = self.parent_frame
        msg2.child_frame_id = self.child_frame
        msg2.transform.translation.x = P_C_to_T1[0, 3]
        msg2.transform.translation.y = P_C_to_T1[1, 3]
        msg2.transform.translation.z = P_C_to_T1[2, 3]
        quat_C_to_T1 = R.from_matrix(P_C_to_T1[:3, :3]).as_quat()
        msg2.transform.rotation.x = quat_C_to_T1[0]
        msg2.transform.rotation.y = quat_C_to_T1[1]
        msg2.transform.rotation.z = quat_C_to_T1[2]
        msg2.transform.rotation.w = quat_C_to_T1[3]
        self.broadcaster.sendTransform(msg2)

        # camera -> object_2
        msg3 = TransformStamped()
        msg3.header.stamp = stamp_msg
        msg3.header.frame_id = self.parent_frame
        msg3.child_frame_id = self.intermediate_frame
        msg3.transform.translation.x = T2[0, 3]
        msg3.transform.translation.y = T2[1, 3]
        msg3.transform.translation.z = T2[2, 3]
        quat_C_to_ObstacleFoam = R.from_matrix(T2[:3, :3]).as_quat()
        msg3.transform.rotation.x = quat_C_to_ObstacleFoam[0]
        msg3.transform.rotation.y = quat_C_to_ObstacleFoam[1]
        msg3.transform.rotation.z = quat_C_to_ObstacleFoam[2]
        msg3.transform.rotation.w = quat_C_to_ObstacleFoam[3]
        self.broadcaster.sendTransform(msg3)

        # camera -> object_3
        msg4 = TransformStamped()
        msg4.header.stamp = stamp_msg
        msg4.header.frame_id = self.parent_frame
        msg4.child_frame_id = self.goal_frame
        msg4.transform.translation.x = T3[0, 3]
        msg4.transform.translation.y = T3[1, 3]
        msg4.transform.translation.z = T3[2, 3]
        quat_C_to_GoalFoam = R.from_matrix(T3[:3, :3]).as_quat()
        msg4.transform.rotation.x = quat_C_to_GoalFoam[0]
        msg4.transform.rotation.y = quat_C_to_GoalFoam[1]
        msg4.transform.rotation.z = quat_C_to_GoalFoam[2]
        msg4.transform.rotation.w = quat_C_to_GoalFoam[3]
        self.broadcaster.sendTransform(msg4)

        # --------------------------------------------------
        # APPEND TO DEQUES
        # --------------------------------------------------
        self.append_transform_history(
            self.cam_to_obj1_history,
            "camera_to_object_1",
            T,
            ts
        )

        self.append_transform_history(
            self.cam_to_obj2_history,
            "camera_to_object_2",
            T2,
            ts
        )

        self.append_transform_history(
            self.cam_to_obj3_history,
            "camera_to_object_3",
            T3,
            ts
        )

        self.append_transform_history(
            self.cam_to_base_history,
            "camera_to_base_link",
            P_C_to_base,
            ts
        )

        self.append_transform_history
        (
            self.cam_to_floor_history,
            "camera_to_floor",
            T_floor,
            ts
        )

        self.append_transform_history
        (
            self.floor_to_baselink_history,
            "floor_to_baselink",
            P_floor_to_base,
            ts
        )

        self.append_transform_history
        (
            self.floor_to_object2_history,
            "floor_to_object2",
            P_floor_to_obj2,
            ts
        )

        self.append_transform_history
        (
            self.floor_to_object3_history,
            "floor_to_object3",
            P_floor_to_obj3,
            ts
        )

        p_base = P_C_to_base[:3, 3]
        p_obj2 = T2[:3, 3]
        p_obj3 = T3[:3, 3]

        self.append_relative_distance_history(
            ts,
            p_base,
            p_obj2,
            p_obj3
        )

        self.json_save_counter += 1
        if self.json_save_counter >= self.json_save_every_n_ticks:
            self.save_deques_to_json()
            self.json_save_counter = 0

# ============================================================
# MAIN
# ============================================================
def main():
    rclpy.init()
    node = TFLogger()
    try:
        rclpy.spin(node)
    finally:
        node.save_deques_to_json()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()