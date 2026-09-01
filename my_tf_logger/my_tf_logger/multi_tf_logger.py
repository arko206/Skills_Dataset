#!/usr/bin/env python3
import os
from datetime import datetime

import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from rclpy.time import Time

import numpy as np
import tf2_ros
from scipy.spatial.transform import Rotation as R


def quat_to_rot(qx, qy, qz, qw):
    """Quaternion (x,y,z,w) -> 3x3 rotation matrix using SciPy."""
    quat_array = np.array([qx, qy, qz, qw])
    rotation = R.from_quat(quat_array)   # SciPy expects [x, y, z, w]
    return rotation.as_matrix()


def transform_to_matrix(t):
    """Convert TransformStamped to 4x4 homogeneous matrix."""
    tr = t.transform.translation
    q  = t.transform.rotation
    rot_matrix = quat_to_rot(q.x, q.y, q.z, q.w)
    T = np.eye(4, dtype=float)
    T[:3, :3] = rot_matrix
    T[:3,  3] = np.array([tr.x, tr.y, tr.z])
    return T


class MultiTFLogger(Node):
    """
    Logs multiple transforms and points at a fixed rate into a file, 
    keeping each transform as a full 4x4 matrix block.
    """
    def __init__(self):
        super().__init__('multi_tf_logger')

        self.declare_parameter('rate_hz', 10.0)
        self.declare_parameter('output_file', os.path.expanduser('~/DemoTag_Redo_20th_multi_tf_log.txt'))

        self.rate_hz     = self.get_parameter('rate_hz').get_parameter_value().double_value
        self.output_file = self.get_parameter('output_file').get_parameter_value().string_value

        self.buffer = tf2_ros.Buffer(cache_time=Duration(seconds=10.0))
        self.listener = tf2_ros.TransformListener(self.buffer, self, spin_thread=True)

        period = 1.0 / max(self.rate_hz, 0.1)
        self.timer = self.create_timer(period, self.tick)

        self.get_logger().info(
            f"Logging transforms at {self.rate_hz:.1f} Hz to {self.output_file}"
        )

        # Write header once
        if not os.path.exists(self.output_file):
            with open(self.output_file, 'w') as f:
                f.write("# timestamp_iso\n")
                f.write("# P(C->T1), P(C->T2), P(Rc->T2), P_Rc(x,y,z), P_c(x,y,z)\n\n")

    def lookup_tf_safe(self, parent, child):
        try:
            t = self.buffer.lookup_transform(parent, child, Time())
            return transform_to_matrix(t)
        except Exception as e:
            self.get_logger().warn(f"TF {parent}->{child} not available: {e}")
            return None

    def tick(self):
        parent_c   = "camera"
        parent_rc  = "camera_frame"
        child_t1   = "object_1"
        child_t2   = "object_2"
        child_rc   = "go2_object"
        
        ###-- Step (a) Getting the Transformation from Camera to Tag-T1 --###
        T_c_t1  = self.lookup_tf_safe(parent_c, child_t1)

        ###-- Step (b) Getting the Transformation from Camera to Tag-T2 --###
        T_c_t2  = self.lookup_tf_safe(parent_c, child_t2)

        ###-- Step (c) Getting the Transformation from Robot Camera to Tag-T2 --###
        T_rc_t2 = self.lookup_tf_safe(parent_rc, child_rc)

        if T_c_t1 is None or T_c_t2 is None or T_rc_t2 is None:
            return

       
        ####-- Step (d) Computing the Transformation from Tag-T2 to Camera--###
        T_t2_c = np.linalg.inv(T_c_t2)

        ##-- Step (e) Getting the transformation computed for the webcamera frame in the robot camera frame--##
        T_rc_c = T_rc_t2 @ T_t2_c

        #---Computing the inverse transformation from the camera frame to the robot camera frame--##
        T_c_rc = np.linalg.inv(T_rc_c)


        translation_vector = T_c_rc[:3, 3]
        rotation_matrix = T_c_rc[:3, :3]
        r = R.from_matrix(rotation_matrix)
        quat = r.as_quat()  # [x, y, z, w]

        roll_c_rc, pitch_c_rc, yaw_c_rc = r.as_euler('xyz')

        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")


        ####-------Estinating Tag-T1 in the Rc frame-------------####
        T_rc_t1 = T_rc_t2 @ T_t2_c @ T_c_t1
        T_rc_t1_pred = T_rc_c @ T_c_t1

        roll_rc, pitch_rc, yaw_rc = R.from_matrix(T_rc_t1[:3, :3]).as_euler('xyz')

        # Print to console
        self.get_logger().info(
            f"\n[{ts}] Given these Logged transforms:\n"
            f"P(parent_c->child_t1):\n{T_c_t1}\n\n"
            f"P(parent_c->child_t2):\n{T_c_t2}\n\n"
            f"P(parent_rc->child_rc):\n{T_rc_t2}\n\n"
            f"For C->Rc Translation : {translation_vector}, Rotation (r, p, y): {roll_c_rc, pitch_c_rc, yaw_c_rc}\n"
            f"P(Rc->T1) (predicted):\n{T_rc_t1_pred}\n"
            f"P(Rc->T1) (computed):\n{T_rc_t1}\n"
        )

        # Save to file in readable matrix format
        with open(self.output_file, 'a') as f:
            f.write(f"\n[{ts}]\n")
            f.write("P(C->T1):\n")
            np.savetxt(f, T_c_t1, fmt="%.6f")
            f.write("\nP(C->T2):\n")
            np.savetxt(f, T_c_t2, fmt="%.6f")
            f.write("\nP(Rc->T2):\n")
            np.savetxt(f, T_rc_t2, fmt="%.6f")
            f.write("\nFinally Obtained P(Rc->T1) (computed):\n")
            np.savetxt(f, T_rc_t1, fmt="%.6f")
            f.write("\n" + "="*50 + "\n")

        


def main():
    rclpy.init()
    node = MultiTFLogger()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
    
