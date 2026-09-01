#!/usr/bin/env python3
import os
from datetime import datetime

import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from rclpy.time import Time

import numpy as np
import tf2_ros
from geometry_msgs.msg import TransformStamped

from visualization_msgs.msg import Marker
from std_msgs.msg import ColorRGBA


# Fixed calibration matrix P*(Rc->T1)


# P_star_Rc_to_T1 = np.array([
# [0.997663, 0.068232 ,0.003653, 0.002185],
# [0.004445, -0.011456, -0.999925, -0.053204],
# [-0.068185, 0.997604, -0.011732, -0.093665],
# [0.000000, 0.000000 ,0.000000 ,1.000000]])


# P_star_Rc_to_T1 = np.array([
# [0.997663, 0.068232 ,0.003653, 0.002185],
# [0.004445, -0.011456, -0.999925, -0.053204],
# [-0.068185, 0.997604, -0.011732, -0.093665],
# [0.000000, 0.000000 ,0.000000 ,1.000000]])


P_star_Rc_to_T1= np.array([
    [0.999666, 0.025621, -0.003426, -0.005381],
    [-0.003067, -0.014019, -0.999897, -0.053450],
    [-0.025667, 0.999573, -0.013935, -0.091940],
    [  0.000000, 0.000000 ,0.000000 ,1.000000]])






from scipy.spatial.transform import Rotation as R

def quat_to_rot(qx, qy, qz, qw):
    """Quaternion (x,y,z,w) -> 3x3 rotation matrix using SciPy."""
    quat_array = np.array([qx, qy, qz, qw])
    rotation = R.from_quat(quat_array)   # SciPy expects [x, y, z, w]

    print("The value of rotation is:", rotation)
    print(" The Rotation matrix is:", rotation.as_matrix())


    # Convert to RPY in 'xyz' order (roll, pitch, yaw) in radians
    roll_pitch_yaw_radians = rotation.as_euler('xyz', degrees=False)

    return rotation.as_matrix(), roll_pitch_yaw_radians


###--- To compute the change in quaterion between two consecutive frames ---###
def quat_angle_delta(q1, q2):
    """Angle between two unit quaternions q1, q2 (x,y,z,w). Returns radians."""
    q1 = np.asarray(q1, dtype=float)
    q2 = np.asarray(q2, dtype=float)
    q1 = q1 / np.linalg.norm(q1)
    q2 = q2 / np.linalg.norm(q2)
    dot = float(np.clip(np.abs(np.dot(q1, q2)), 0.0, 1.0))
    return 2.0 * np.arccos(dot)

###-- Function for shutting down the node after the transform stabilizes for certain time duration ---###
def stop_and_exit(self):
    self.get_logger().info("Stopping logger and shutting down...")
    self.timer.cancel()
    self.destroy_node()
    rclpy.shutdown()


class TFLogger(Node):
    """
    Polls tf at a fixed rate and logs the transform parent->child
    as a 4x4 homogeneous matrix to a file. Also prints to console.
    """
    def __init__(self):
        super().__init__('tf_logger')

        # Parameters (you can override with --ros-args -p ...)
        self.declare_parameter('parent_frame', 'camera_frame')
        self.declare_parameter('child_frame', 'object')
        self.declare_parameter('rate_hz', 10.0)  # how often to sample tf
        self.declare_parameter('output_file', os.path.expanduser('~/home/arka/Go2_Walk_Base_Data_Sensor/First_Base_to_cam_movement.txt'))

        ##----- Declaring the parameter for debugging purpose ----##
        self.declare_parameter('object_one_to_camera_frame', os.path.expanduser('~/home/arka/Demo_Movement/Debugging_Storage_Files/First_Tag_to_cam_movement.txt'))
        self.declare_parameter('object_one_to_robot_camera', os.path.expanduser('~/home/arka/Demo_Movement/Debugging_Storage_Files/First_Tag_to_robotcam_movement.txt'))
        self.declare_parameter('object_one_to_robot_base', os.path.expanduser('~/home/arka/Demo_Movement/Debugging_Storage_Files/First_Tag_to_robotbase_movement.txt'))

        self.parent_frame = self.get_parameter('parent_frame').get_parameter_value().string_value
        self.child_frame  = self.get_parameter('child_frame').get_parameter_value().string_value
        self.rate_hz      = self.get_parameter('rate_hz').get_parameter_value().double_value
        self.output_file  = self.get_parameter('output_file').get_parameter_value().string_value

        ###--- For debugging purpose----####
        self.object_one_to_camera_frame = self.get_parameter('object_one_to_camera_frame').get_parameter_value().string_value
        self.object_one_to_robot_camera = self.get_parameter('object_one_to_robot_camera').get_parameter_value().string_value
        self.object_one_to_robot_base = self.get_parameter('object_one_to_robot_base').get_parameter_value().string_value
        

        ###---declaring the parameter for stopping the logging after certain time duration---###
                # --- Auto-stop when transform stabilizes ---
        self.declare_parameter('stop_translation_eps', 0.002)  # meters (e.g., 2mm)
        self.declare_parameter('stop_rotation_eps', 0.01)      # radians (~0.57 deg)
        self.declare_parameter('stop_consecutive', 10)         # stable frames required

        self.stop_translation_eps = self.get_parameter('stop_translation_eps').get_parameter_value().double_value
        self.stop_rotation_eps    = self.get_parameter('stop_rotation_eps').get_parameter_value().double_value
        self.stop_consecutive     = self.get_parameter('stop_consecutive').get_parameter_value().integer_value

        self.prev_P_C_to_base = None
        self.prev_q_C_to_base = None
        self.stable_count = 0


        # ##--Adding the marker publisher --##
        # self.marker_pub = self.create_publisher(Marker, 'camera_to_base_markers', 10)
        # self.marker_id = 0


        # TF buffer/listener
        self.buffer = tf2_ros.Buffer(cache_time=Duration(seconds=10.0))
        self.listener = tf2_ros.TransformListener(self.buffer, self, spin_thread=True)

        # TF broadcaster
        self.broadcaster = tf2_ros.TransformBroadcaster(self)

        # Timer to query at fixed rate
        period = 1.0 / max(self.rate_hz, 0.1)
        self.timer = self.create_timer(period, self.tick)

        self.get_logger().info(
            f"Logging TF '{self.parent_frame}' -> '{self.child_frame}' "
            f"at {self.rate_hz:.1f} Hz to {self.output_file}"
        )

        # Ensure file header exists
        if not os.path.exists(self.output_file):
            with open(self.output_file, 'w') as f:
                f.write("# timestamp_iso  T(4x4 row-major)\n")

    def tick(self):
        # Query the latest available transform
        ###------Loking for the transformation from Camera Frame to Tag-T1 frame ----###
        try:
            t: tf2_ros.TransformStamped = self.buffer.lookup_transform(
                self.parent_frame, self.child_frame, Time()
            )
        except Exception as e:
            self.get_logger().warn(f"TF not available yet: {e}")
            return

        tr = t.transform.translation
        q  = t.transform.rotation

        # 4x4 matrix
        rot_matrix, roll_pitch_yaw_radians = quat_to_rot(q.x, q.y, q.z, q.w)
        T = np.eye(4, dtype=float)
        T[:3, :3] = rot_matrix
        T[:3,  3] = np.array([tr.x, tr.y, tr.z])

        r, p, y = roll_pitch_yaw_radians

        # Timestamp in human-readable local time
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Pretty print to console
        self.get_logger().info(
            f"\n[{ts}] {self.parent_frame} -> {self.child_frame}\n{T}"
        )


        ### Step 1: Computation of Transformation from Webcam to Tag-T1 frame ---- ###
        P_C_to_T1 = T

        ## Step 2: Invert the Transformation to get P(T1->C)
        P_T1_to_C = np.linalg.inv(P_C_to_T1)
        quat_T1_to_C = R.from_matrix(P_T1_to_C[:3, :3]).as_quat()

        rot_matrix_T1_to_C, rpy_T1_to_C = quat_to_rot(quat_T1_to_C[0], quat_T1_to_C[1], quat_T1_to_C[2], quat_T1_to_C[3])

        ### Step 3: Obtaining the Transformation from Base to Head_Upper frame
        P_base_to_Head_Upper = np.eye(4)
        rot_matrix_base_to_Head_upper, rpy_base_to_Head_Upper = quat_to_rot(0, 0, 0, 1)  # rotation
        P_base_to_Head_Upper[:3, :3] = rot_matrix_base_to_Head_upper
        P_base_to_Head_Upper[:3,  3] = np.array([0.285, 0, 0.01]) 


        ###Step 4: Obtaining the Transformation from Head_Upper frame to Rc frame
        P_Head_Upper_to_Rc = np.eye(4)
        rot_matrix_Head_Upper_to_Rc, rpy_Head_Upper_to_Rc = quat_to_rot(-0.5, 0.500002, -0.5, 0.499998) 
        P_Head_Upper_to_Rc[:3, :3] = rot_matrix_Head_Upper_to_Rc
        P_Head_Upper_to_Rc[:3,  3] = np.array([0.045, 0, 0.03])

        #### Step 5: Composing the Transformation from Base to Rc frame
        P_base_to_Rc = P_base_to_Head_Upper @ P_Head_Upper_to_Rc


        ### Step 6: Transformation from Base frame to Tag-T1 frame
        P_base_to_T1 = P_base_to_Head_Upper @ P_Head_Upper_to_Rc @ P_star_Rc_to_T1


         ### Step 8: Computing the Transformation from Tag-T1 to Robot Camera Frame
        P_T1_to_Rc = np.linalg.inv(P_base_to_T1) @ P_base_to_Rc
        quat_T1_to_Rc = R.from_matrix(P_T1_to_Rc[:3, :3]).as_quat()
        rot_matrix_T1_to_Rc, rpy_T1_to_Rc = quat_to_rot(quat_T1_to_Rc[0], quat_T1_to_Rc[1], quat_T1_to_Rc[2], quat_T1_to_Rc[3])

        print(" The transformation matrix from Tag-T1 to Robot Camera frame is:", P_T1_to_Rc)


        ###---Checking_T1 to Rc-----####
        P_check_T1_to_Rc = np.linalg.inv(P_star_Rc_to_T1)
        quat_check_T1_to_Rc = R.from_matrix(P_check_T1_to_Rc[:3, :3]).as_quat()
        rot_matrix_check_T1_to_Rc, rpy_check_T1_to_Rc = quat_to_rot(quat_check_T1_to_Rc[0], quat_check_T1_to_Rc[1], quat_check_T1_to_Rc[2], quat_check_T1_to_Rc[3])

        print(" The checking transformation matrix from Tag-T1 to Robot Camera frame is:", P_check_T1_to_Rc)


        ## Step 7: Computing the Transformation from Tag-T1 to Robot Base Frame
        P_T1_to_base = P_T1_to_Rc @ np.linalg.inv(P_base_to_Rc)
        quat_T1_to_base = R.from_matrix(P_T1_to_base[:3, :3]).as_quat()
        rot_matrix_T1_to_base, rpy_T1_to_base = quat_to_rot(quat_T1_to_base[0], quat_T1_to_base[1], quat_T1_to_base[2], quat_T1_to_base[3])


       

        

        ## Step 9: Computing the Transformation from Camera to Base frame
        P_C_to_base = P_C_to_T1 @ P_T1_to_base
        quat_C_to_base = R.from_matrix(P_C_to_base[:3, :3]).as_quat()
        rot_matrix_C_to_base, rpy_C_to_base = quat_to_rot(quat_C_to_base[0], quat_C_to_base[1], quat_C_to_base[2], quat_C_to_base[3])   

        




        # Broadcast camera->base_link
        msg = TransformStamped()
        msg.header.stamp =  t.header.stamp 
        msg.header.frame_id = self.parent_frame     # "camera"
        msg.child_frame_id = "base_link"

        msg.transform.translation.x = P_C_to_base[0, 3]
        msg.transform.translation.y = P_C_to_base[1, 3]
        msg.transform.translation.z = P_C_to_base[2, 3]

        quat_C_to_base = R.from_matrix(P_C_to_base[:3, :3]).as_quat()
        msg.transform.rotation.x = quat_C_to_base[0]
        msg.transform.rotation.y = quat_C_to_base[1]
        msg.transform.rotation.z = quat_C_to_base[2]
        msg.transform.rotation.w = quat_C_to_base[3]

        self.broadcaster.sendTransform(msg)


       




         # ## Brodcast base_link-> Tag-T1
        msg2 = TransformStamped()
        msg2.header.stamp = t.header.stamp #self.get_clock().now().to_msg()
        msg2.header.frame_id = "base_link"
        msg2.child_frame_id = self.child_frame     # "Tag-T1"

        msg2.transform.translation.x = P_base_to_T1[0, 3]
        msg2.transform.translation.y = P_base_to_T1[1, 3]
        msg2.transform.translation.z = P_base_to_T1[2, 3]
        quat_base_to_T1 = R.from_matrix(P_base_to_T1[:3, :3]).as_quat()
        msg2.transform.rotation.x = quat_base_to_T1[0]
        msg2.transform.rotation.y = quat_base_to_T1[1]
        msg2.transform.rotation.z = quat_base_to_T1[2]
        msg2.transform.rotation.w = quat_base_to_T1[3]

        self.broadcaster.sendTransform(msg2)

        # ---------------- Auto-stop logic ----------------
        t_now = P_C_to_base[:3, 3]
        q_now = quat_C_to_base  # (x,y,z,w)

        if self.prev_P_C_to_base is not None:
            t_prev = self.prev_P_C_to_base[:3, 3]
            dt = float(np.linalg.norm(t_now - t_prev))  # meters
            dtheta = float(quat_angle_delta(self.prev_q_C_to_base, q_now))  # radians

            if (dt < self.stop_translation_eps) and (dtheta < self.stop_rotation_eps):
                self.stable_count += 1
            else:
                self.stable_count = 0

            self.get_logger().info(
                f"Stability check: dt={dt:.6f} m, dtheta={dtheta:.6f} rad, "
                f"stable={self.stable_count}/{self.stop_consecutive}"
            )

            if self.stable_count >= self.stop_consecutive:
                self.get_logger().info(
                    f"Transform stabilized for {self.stop_consecutive} consecutive samples. Stopping node."
                )
                # stop timer then shutdown cleanly
                self.timer.cancel()
                rclpy.shutdown()
                return
            

        self.prev_P_C_to_base = P_C_to_base.copy()
        self.prev_q_C_to_base = q_now.copy()
        # ---------------------------------------------------


         # Append to file in same format
        with open(self.output_file, 'a') as f:
            f.write(f"[{ts}] {self.parent_frame} -> {self.child_frame}\n")
            np.savetxt(f, T, fmt="%.6f")
            f.write("Translation (x,y,z): ist %.6f %.6f %.6f\n" % (tr.x, tr.y, tr.z))
            f.write("Rotation (r,p,y): ist %.6f %.6f %.6f\n" % (r, p, y))
            f.write("Transformation matrix from Camera to Base frame:\n")
            np.savetxt(f, P_C_to_base, fmt="%.6f")
            f.write ("Translation vector from Camera to Base frame: %.6f %.6f %.6f\n" % (P_C_to_base[0,3], P_C_to_base[1,3], P_C_to_base[2,3]))
            f.write ("Rotation (r,p,y) from Camera to Base frame: %.6f %.6f %.6f\n" % (rpy_C_to_base[0], rpy_C_to_base[1], rpy_C_to_base[2]))
            f.write("\n")

        with open(self.object_one_to_camera_frame, 'a') as f:
            f.write("Transformation matrix from Tag-1 to Camera frame:\n")
            np.savetxt(f, P_T1_to_C, fmt="%.6f")
            f.write ("Translation vector from Tag-1 to Camera frame: %.6f %.6f %.6f\n" % (P_T1_to_C[0,3], P_T1_to_C[1,3], P_T1_to_C[2,3]))
            f.write("Rotation (r,p,y) from Tag-1 to Camera frame: %.6f %.6f %.6f\n" % (rpy_T1_to_C[0], rpy_T1_to_C[1], rpy_T1_to_C[2]))
            f.write("\n")

        with open(self.object_one_to_robot_camera, 'a') as f:
            f.write("Transformation matrix from Tag-1 to Robot Camera frame:\n")
            np.savetxt(f, P_T1_to_Rc, fmt="%.6f")
            f.write ("Translation vector from Tag-1 to Robot Camera frame: %.6f %.6f %.6f\n" % (P_T1_to_Rc[0,3], P_T1_to_Rc[1,3], P_T1_to_Rc[2,3]))
            f.write("Rotation (r,p,y) from Tag-1 to Robot Camera frame: %.6f %.6f %.6f\n" % (rpy_T1_to_Rc[0], rpy_T1_to_Rc[1], rpy_T1_to_Rc[2]))
            f.write("\n")

        with open(self.object_one_to_robot_base, 'a') as f:
            f.write("Transformation matrix from Tag-1 to Robot Base frame:\n")
            np.savetxt(f, P_T1_to_base, fmt="%.6f")
            f.write ("Translation vector from Tag-1 to Robot Base frame: %.6f %.6f %.6f\n" % (P_T1_to_base[0,3], P_T1_to_base[1,3], P_T1_to_base[2,3]))
            f.write("Rotation (r,p,y) from Tag-1 to Robot Base frame: %.6f %.6f %.6f\n" % (rpy_T1_to_base[0], rpy_T1_to_base[1], rpy_T1_to_base[2]))
            f.write("\n")







def main():
    rclpy.init()
    node = TFLogger()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
