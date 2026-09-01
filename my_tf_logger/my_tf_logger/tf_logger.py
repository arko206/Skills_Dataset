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


# Fixed calibration matrix P*(Rc->T1)
P_star_Rc_to_T1 = np.array([
   [ 0.999978, -0.003756, -0.005498, 0.000640],
   [-0.005572, -0.019972, -0.999785, -0.051589],
   [ 0.003645,  0.999793, -0.019993, -0.095150],
   [ 0.000000,  0.000000,  0.000000, 1.000000]
])





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

        ### Step 3: Composing the Transformation from Rc to C frame
        P_Rc_to_C = P_star_Rc_to_T1 @ P_T1_to_C


        ### Step 4: Obtaining the Transformation from Base to Head_Upper frame
        P_base_to_Head_Upper = np.eye(4)
        rot_matrix_base_to_Head_upper, rpy_base_to_Head_Upper = quat_to_rot(0, 0, 0, 1)  # rotation
        P_base_to_Head_Upper[:3, :3] = rot_matrix_base_to_Head_upper
        P_base_to_Head_Upper[:3,  3] = np.array([0.285, 0, 0.01]) 


        ###Step 5: Obtaining the Transformation from Head_Upper frame to Rc frame
        P_Head_Upper_to_Rc = np.eye(4)
        rot_matrix_Head_Upper_to_Rc, rpy_Head_Upper_to_Rc = quat_to_rot(-0.5, 0.500002, -0.5, 0.499998) 
        P_Head_Upper_to_Rc[:3, :3] = rot_matrix_Head_Upper_to_Rc
        P_Head_Upper_to_Rc[:3,  3] = np.array([0.045, 0, 0.03])

        #### Step 6: Composing the Transformation from Base to Rc frame
        P_base_to_Rc = P_base_to_Head_Upper @ P_Head_Upper_to_Rc

        ### Step 7: Computing the Transformation from Base to Camera frame
        P_base_to_C = P_base_to_Rc @ P_Rc_to_C
        quat_base_to_C = R.from_matrix(P_base_to_C[:3, :3]).as_quat()

        rot_matrix_base_to_C, rpy_base_to_C = quat_to_rot(quat_base_to_C[0], quat_base_to_C[1], quat_base_to_C[2], quat_base_to_C[3])
        
        ####-- Computing the Transformation from Camera to Base frame --####
        P_C_to_base = np.linalg.inv(P_base_to_C)

        quat_C_to_base = R.from_matrix(P_C_to_base[:3, :3]).as_quat()
        rot_matrix_C_to_base, rpy_C_to_base = quat_to_rot(quat_C_to_base[0], quat_C_to_base[1], quat_C_to_base[2], quat_C_to_base[3])


        ## Step 8: Computing the Transformation from Base to Tag-T1 frame
        #P_base_to_T1 = P_base_to_C @ P_C_to_T1

        #P_base_to_T1_check = P_base_to_Head_Upper @ P_Head_Upper_to_Rc @ P_star_Rc_to_T1

        #new_P_base_to_C = P_base_to_T1 @ P_T1_to_C



        


       












        ### Step 8: For Checking and Validation purpose, Computing the Transformation from Camera frame to Base frame
        P_T1_to_Head_Upper = np.linalg.inv(P_star_Rc_to_T1) @ np.linalg.inv(P_Head_Upper_to_Rc)
        P_C_to_base_check = P_C_to_T1 @ P_T1_to_Head_Upper @ np.linalg.inv(P_base_to_Head_Upper)

        P_base_to_C_check = np.linalg.inv(P_C_to_base_check)

        print("The Transformation matrix from Camera to Base frame is:\n", P_C_to_base_check)
        print("\Estimated Transformation matrix from Camera to Base frame is:\n", P_C_to_base)


        print("The Transformation Matrix from Base to Camera frame is:\n", P_base_to_C_check)
        print("Estimated Transformation Matrix from Base to Camera frame is:\n", P_base_to_C)

        

         ## Step 8: Computing the Transformation from Base to Tag-T1 frame
        P_base_to_T1 = P_base_to_C @ P_C_to_T1

        P_base_to_T1_check = P_base_to_Head_Upper @ P_Head_Upper_to_Rc @ P_star_Rc_to_T1

        # new_P_base_to_C = P_base_to_T1_check @ P_T1_to_C

        # new_P_C_to_base = np.linalg.inv(new_P_base_to_C)



        # ### Step 9: Checking the Transformation from Camera to Tag-T1 frame
        # P_C_to_T1_check = new_P_C_to_base @ P_base_to_T1_check





         # Broadcast camera->base_link
        msg = TransformStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
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

        msg2.transform.translation.x = P_base_to_T1_check[0, 3]
        msg2.transform.translation.y = P_base_to_T1_check[1, 3]
        msg2.transform.translation.z = P_base_to_T1_check[2, 3]
        quat_base_to_T1 = R.from_matrix(P_base_to_T1_check[:3, :3]).as_quat()
        msg2.transform.rotation.x = quat_base_to_T1[0]
        msg2.transform.rotation.y = quat_base_to_T1[1]
        msg2.transform.rotation.z = quat_base_to_T1[2]
        msg2.transform.rotation.w = quat_base_to_T1[3]

        self.broadcaster.sendTransform(msg2)


        # msg3 = TransformStamped()
        # msg3.header.stamp = t.header.stamp #self.get_clock().now().to
        # msg3.header.frame_id = self.parent_frame     # "camera"
        # msg3.child_frame_id = "Tag-T1_check"
        # msg3.transform.translation.x = P_C_to_T1_check[0, 3]
        # msg3.transform.translation.y = P_C_to_T1_check[1, 3]
        # msg3.transform.translation.z = P_C_to_T1_check[2, 3]
        # quat_C_to_T1 = R.from_matrix(P_C_to_T1_check[:3, :3]).as_quat()
        # msg3.transform.rotation.x = quat_C_to_T1[0]
        # msg3.transform.rotation.y = quat_C_to_T1[1]
        # msg3.transform.rotation.z = quat_C_to_T1[2]
        # msg3.transform.rotation.w = quat_C_to_T1[3]


        ## Step 10: Checking the Transformation from Tag-T1 frame to Robot Camera frame
        P_T1_to_Robot_Cam = P_T1_to_C @ np.linalg.inv(P_Rc_to_C)

        quat_T1_to_Robot_Cam = R.from_matrix(P_T1_to_Robot_Cam[:3, :3]).as_quat()

        rot_matrix_T1_to_Robot_Cam, rpy_T1_to_Robot_Cam = quat_to_rot(quat_T1_to_Robot_Cam[0], quat_T1_to_Robot_Cam[1], quat_T1_to_Robot_Cam[2], quat_T1_to_Robot_Cam[3])


        ## sTEP 11: Checking the Transformation from Tag-T1 frame to Robot Base frame
        P_T1_to_Robot_Base = P_T1_to_C @ np.linalg.inv(P_base_to_C)

        quat_T1_to_Robot_Base = R.from_matrix(P_T1_to_Robot_Base[:3, :3]).as_quat()
        rot_matrix_T1_to_Robot_Base, rpy_T1_to_Robot_Base = quat_to_rot(quat_T1_to_Robot_Base[0], quat_T1_to_Robot_Base[1], quat_T1_to_Robot_Base[2], quat_T1_to_Robot_Base[3])







        # Append to file in same format
        with open(self.output_file, 'a') as f:
            f.write(f"[{ts}] {self.parent_frame} -> {self.child_frame}\n")
            np.savetxt(f, T, fmt="%.6f")
            f.write("Translation (x,y,z): ist %.6f %.6f %.6f\n" % (tr.x, tr.y, tr.z))
            f.write("Rotation (r,p,y): ist %.6f %.6f %.6f\n" % (r, p, y))
            f.write("Transformation matrix from Base to Camera frame:\n")
            np.savetxt(f, P_base_to_C, fmt="%.6f")
            f.write ("Translation vector from Camera to Base frame: %.6f %.6f %.6f\n" % (P_C_to_base[0,3], P_C_to_base[1,3], P_C_to_base[2,3]))
            f.write ("Rotation (r,p,y) from Camera to Base frame: %.6f %.6f %.6f\n" % (rpy_C_to_base[0], rpy_C_to_base[1], rpy_C_to_base[2]))
            f.write("\n")

        with open(self.object_one_to_camera_frame, 'a') as f:
            f.write("Transformation matrix from Tag-1 to Camera frame:\n")
            np.savetxt(f, P_C_to_T1, fmt="%.6f")
            f.write ("Translation vector from Tag-1 to Camera frame: %.6f %.6f %.6f\n" % (P_T1_to_C[0,3], P_T1_to_C[1,3], P_T1_to_C[2,3]))
            f.write("Rotation (r,p,y) from Tag-1 to Camera frame: %.6f %.6f %.6f\n" % (rpy_T1_to_C[0], rpy_T1_to_C[1], rpy_T1_to_C[2]))
            f.write("\n")

        with open(self.object_one_to_robot_camera, 'a') as f:
            f.write("Transformation matrix from Tag-1 to Robot Camera frame:\n")
            np.savetxt(f, P_T1_to_Robot_Cam, fmt="%.6f")
            f.write ("Translation vector from Tag-1 to Robot Camera frame: %.6f %.6f %.6f\n" % (P_T1_to_Robot_Cam[0,3], P_T1_to_Robot_Cam[1,3], P_T1_to_Robot_Cam[2,3]))
            f.write("Rotation (r,p,y) from Tag-1 to Robot Camera frame: %.6f %.6f %.6f\n" % (rpy_T1_to_Robot_Cam[0], rpy_T1_to_Robot_Cam[1], rpy_T1_to_Robot_Cam[2]))
            f.write("\n")

        with open(self.object_one_to_robot_base, 'a') as f:
            f.write("Transformation matrix from Tag-1 to Robot Base frame:\n")
            np.savetxt(f, P_T1_to_Robot_Base, fmt="%.6f")
            f.write ("Translation vector from Tag-1 to Robot Base frame: %.6f %.6f %.6f\n" % (P_T1_to_Robot_Base[0,3], P_T1_to_Robot_Base[1,3], P_T1_to_Robot_Base[2,3]))
            f.write("Rotation (r,p,y) from Tag-1 to Robot Base frame: %.6f %.6f %.6f\n" % (rpy_T1_to_Robot_Base[0], rpy_T1_to_Robot_Base[1], rpy_T1_to_Robot_Base[2]))
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
