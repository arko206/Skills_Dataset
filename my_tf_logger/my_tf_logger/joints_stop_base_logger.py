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



P_star_Rc_to_T1 = np.array([
    [0.998833, 0.039814, -0.027331, 0.008661],
    [-0.025861, -0.036982, -0.998981, -0.049915],
    [-0.040784, 0.998523, -0.035909, -0.107167],
    [0.000000, 0.000000, 0.000000, 1.000000]
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


##-- Adding transformation matrix output for debugging purpose ---##
def transform_to_matrix(transform):
    tr = transform.translation
    q = transform.rotation

    rot = R.from_quat([q.x, q.y, q.z, q.w]).as_matrix()

    T = np.eye(4)
    T[:3, :3] = rot
    T[:3, 3] = [tr.x, tr.y, tr.z]

    return T



class TFLogger(Node):
    """
    Polls tf at a fixed rate and logs the transform parent->child
    as a 4x4 homogeneous matrix to a file. Also prints to console.
    """
    def __init__(self):
        super().__init__('tf_logger')

        # Parameters (you can override with --ros-args -p ...)
        self.declare_parameter('parent_frame', 'camera')
        self.declare_parameter('child_frame', 'object_1')
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
        

        # --- Wait-for-motion logic (start recording stop-condition only after motion) ---
        self.declare_parameter('start_motion_translation_eps', 0.01)  # meters (e.g., 1 cm)
        self.declare_parameter('start_motion_rotation_eps', 0.03)     # radians (~1.7 deg)
        self.declare_parameter('start_motion_consecutive', 3)         # moving frames required

        self.start_motion_translation_eps = self.get_parameter(
            'start_motion_translation_eps'
        ).get_parameter_value().double_value

        self.start_motion_rotation_eps = self.get_parameter(
            'start_motion_rotation_eps'
        ).get_parameter_value().double_value

        self.start_motion_consecutive = self.get_parameter(
            'start_motion_consecutive'
        ).get_parameter_value().integer_value

        self.has_moved = False
        self.moving_count = 0



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



    
    ##-- Getting the Transforation Matrix from Parent Frame to Child Frame for debugging purpose ---##
    def get_transform_matrix(self, parent_frame, child_frame):
        try:
            t_all: tf2_ros.TransformStamped  = self.buffer.lookup_transform(parent_frame, child_frame, Time())
            T = transform_to_matrix(t_all.transform)
            return T
        except Exception as e:
            self.get_logger().error(f"Error getting transform {parent_frame} -> {child_frame}: {e}")
            return None
        

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



        ## Step 10: Computing the Transformation from base_link to Front Left Calf Joint of the robot

        ##------ (a)----- Getting the transformation from base to FL hip joint -----##
        P_base_to_FLhip = np.eye(4)
        rot_matrix_base_to_FLhip, rpy_base_to_FLhip = quat_to_rot(0, 0, 0, 1)  # rotation
        P_base_to_FLhip[:3, :3] = rot_matrix_base_to_FLhip
        P_base_to_FLhip[:3,  3] = np.array([0.1934, 0.0465, 0]) 

        ###----- (b) ----- Getting the transformation from FL hip joint to FL thigh joint -----###
        P_FLhip_to_FLthigh = np.eye(4)
        rot_matrix_FLhip_to_FLthigh, rpy_FLhip_to_FLthigh = quat_to_rot(0, -0.00010828, 0, 1)  # rotation
        P_FLhip_to_FLthigh[:3, :3] = rot_matrix_FLhip_to_FLthigh
        P_FLhip_to_FLthigh[:3,  3] = np.array([0, 0.0955, 0])

        ###---- (c) ----- Getting the transformation from FL thigh joint to FL calf joint -----###
        P_FLthigh_to_FLcalf = np.eye(4)
        rot_matrix_FLthigh_to_FLcalf, rpy_FLthigh_to_FLcalf = quat_to_rot(0, -0.77714, 0, 0.62932)  # rotation
        P_FLthigh_to_FLcalf[:3, :3] = rot_matrix_FLthigh_to_FLcalf
        P_FLthigh_to_FLcalf[:3,  3] = np.array([0, 0, -0.213])


        ###---new add--##
        P_FLcalf_to_FLcalflower = np.eye(4)
        rot_matrix_FLcalf_to_FLcalflower, rpy_FLcalf_to_FLcalflower = quat_to_rot(0, 0.0249974, 0, 0.999687)# rotation
        P_FLcalf_to_FLcalflower[:3, :3] = rot_matrix_FLcalf_to_FLcalflower
        P_FLcalf_to_FLcalflower[:3,  3] = np.array([0.02, 0, -0.148])

        P_FLcalflower_to_FLcalflower1 = np.eye(4)
        rot_matrix_FLcalflower_to_FLcalflower1, rpy_FLcalflower_to_FLcalflower1 = quat_to_rot(0, 0.2377, 0, 0.97134)  # rotation
        P_FLcalflower_to_FLcalflower1[:3, :3] = rot_matrix_FLcalflower_to_FLcalflower1
        P_FLcalflower_to_FLcalflower1[:3,  3] = np.array([-0.01, 0, -0.04])

        P_FLcalf_to_FL_foot = np.eye(4)
        rot_matrix_FLcalf_to_FL_foot, rpy_FLcalf_to_FL_foot = quat_to_rot(0, 0, 0, 1)  # rotation
        P_FLcalf_to_FL_foot[:3, :3] = rot_matrix_FLcalf_to_FL_foot
        P_FLcalf_to_FL_foot[:3,  3] = np.array([0, 0, -0.213])



        ##--(d)---- Composing the transformation from base to FL calf joint -----##  
        P_base_to_FLcalf = P_base_to_FLhip @ P_FLhip_to_FLthigh @ P_FLthigh_to_FLcalf
        quat_base_to_FLcalf = R.from_matrix(P_base_to_FLcalf[:3, :3]).as_quat()

        ##--(e)--- Getting the transformation from Camera to FL calf joint for visualization in RViz -----##
        P_C_to_FLcalf = P_C_to_base @ P_base_to_FLcalf
        quat_C_to_FLcalf = R.from_matrix(P_C_to_FLcalf[:3, :3]).as_quat()
        rot_matrix_C_to_FLcalf, rpy_C_to_FLcalf = quat_to_rot(quat_C_to_FLcalf[0], quat_C_to_FLcalf[1], quat_C_to_FLcalf[2], quat_C_to_FLcalf[3])


        ### Step 11: Computing the Transformation from Base link to Front Right Calf Joint of the robot
        
        ##-- (a) --- Getting the transformation from base to FR hip joint -----##
        P_base_to_FRhip = self.get_transform_matrix("base_link", "FR_hip")

        ##-- (b) --- Getting the transformation from FR hip joint to FR thigh joint -----##
        P_FRhip_to_FRthigh = self.get_transform_matrix("FR_hip", "FR_thigh")

        ##-- (c) --- Getting the transformation from FR thigh joint to FR calf joint -----##
        P_FRthigh_to_FRcalf = self.get_transform_matrix("FR_thigh", "FR_calf")


        if (P_base_to_FRhip is None) or (P_FRhip_to_FRthigh is None) or (P_FRthigh_to_FRcalf is None):
            self.get_logger().warn("Missing one of the FR leg transforms, skipping this tick.")
            return

        ##-- (d) --- Composing the transformation from base to FR calf joint -----##        
        P_base_to_FRcalf = P_base_to_FRhip @ P_FRhip_to_FRthigh @ P_FRthigh_to_FRcalf
        quat_base_to_FRcalf = R.from_matrix(P_base_to_FRcalf[:3, :3]).as_quat()

        ##-- (e)--- Getting the transformation from Camera to FR calf joint for visualization in RViz -----##
        P_C_to_FRcalf = P_C_to_base @ P_base_to_FRcalf
        quat_C_to_FRcalf = R.from_matrix(P_C_to_FRcalf[:3, :3]).as_quat()
        rot_matrix_C_to_FRcalf, rpy_C_to_FRcalf = quat_to_rot(quat_C_to_FRcalf[0], quat_C_to_FRcalf[1], quat_C_to_FRcalf[2], quat_C_to_FRcalf[3])



        ### Step 12: Computing the Transformaiom from Base Link to Rear Right Calf Joint of the robot

        ##-- (a) --- Getting the transformation from base to RR hip joint -----##
        P_base_to_RRhip = self.get_transform_matrix("base_link", "RR_hip")
        ##-- (b) --- Getting the transformation from RR hip joint to RR thigh joint -----##
        P_RRhip_to_RRthigh = self.get_transform_matrix("RR_hip", "RR_thigh")
        ##-- (c) --- Getting the transformation from RR thigh joint to RR calf joint -----##
        P_RRthigh_to_RRcalf = self.get_transform_matrix("RR_thigh", "RR_calf")

        if (P_base_to_RRhip is None) or (P_RRhip_to_RRthigh is None) or (P_RRthigh_to_RRcalf is None):
            self.get_logger().warn("Missing one of the RR leg transforms, skipping this tick.")
            return
        ##-- (d) --- Composing the transformation from base to RR calf joint -----##        
        P_base_to_RRcalf = P_base_to_RRhip @ P_RRhip_to_RRthigh @ P_RRthigh_to_RRcalf
        quat_base_to_RRcalf = R.from_matrix(P_base_to_RRcalf[:3, :3]).as_quat() 

        ##-- (e)--- Getting the transformation from Camera to RR calf joint for visualization in RViz -----##
        P_C_to_RRcalf = P_C_to_base @ P_base_to_RRcalf
        quat_C_to_RRcalf = R.from_matrix(P_C_to_RRcalf[:3, :3]).as_quat()
        rot_matrix_C_to_RRcalf, rpy_C_to_RRcalf = quat_to_rot(quat_C_to_RRcalf[0], quat_C_to_RRcalf[1], quat_C_to_RRcalf[2], quat_C_to_RRcalf[3])


        ### Step 13: Computing the Transformaiom from Base Link to Rear Left Calf Joint of the robot   

        ##-- (a) --- Getting the transformation from base to RL hip joint -----##
        P_base_to_RLhip = self.get_transform_matrix("base_link", "RL_hip")
        ##-- (b) --- Getting the transformation from RL hip joint to RL thigh joint -----##
        P_RLhip_to_RLthigh = self.get_transform_matrix("RL_hip", "RL_thigh")
        ##-- (c) --- Getting the transformation from RL thigh joint to RL calf joint -----##
        P_RLthigh_to_RLcalf = self.get_transform_matrix("RL_thigh", "RL_calf")

        if (P_base_to_RLhip is None) or (P_RLhip_to_RLthigh is None) or (P_RLthigh_to_RLcalf is None):
            self.get_logger().warn("Missing one of the RL leg transforms, skipping this tick.")
            return
        ##-- (d) --- Composing the transformation from base to RL calf joint -----##        
        P_base_to_RLcalf = P_base_to_RLhip @ P_RLhip_to_RLthigh @ P_RLthigh_to_RLcalf
        quat_base_to_RLcalf = R.from_matrix(P_base_to_RLcalf[:3, :3]).as_quat() 

        ##-- (e)--- Getting the transformation from Camera to RL calf joint for visualization in RViz -----##
        P_C_to_RLcalf = P_C_to_base @ P_base_to_RLcalf
        quat_C_to_RLcalf = R.from_matrix(P_C_to_RLcalf[:3, :3]).as_quat()
        rot_matrix_C_to_RLcalf, rpy_C_to_RLcalf = quat_to_rot(quat_C_to_RLcalf[0], quat_C_to_RLcalf[1], quat_C_to_RLcalf[2], quat_C_to_RLcalf[3])




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





        # ###-- Broadcast base_link-> FL_hip for visualization in RViz ---###
        msg3 = TransformStamped()
        msg3.header.stamp = t.header.stamp
        msg3.header.frame_id = "base_link"
        msg3.child_frame_id = "Front_Left_Hip"
        msg3.transform.translation.x = P_base_to_FLhip[0, 3]
        msg3.transform.translation.y = P_base_to_FLhip[1, 3]
        msg3.transform.translation.z = P_base_to_FLhip[2, 3]
        quat_base_to_FLhip = R.from_matrix(P_base_to_FLhip[:3, :3]).as_quat()
        msg3.transform.rotation.x = quat_base_to_FLhip[0]
        msg3.transform.rotation.y = quat_base_to_FLhip[1]
        msg3.transform.rotation.z = quat_base_to_FLhip[2]
        msg3.transform.rotation.w = quat_base_to_FLhip[3]
        self.broadcaster.sendTransform(msg3)


        # ##-- Broadacst FL_hip -> FL_thigh for visualization in RViz ---##
        msg4 = TransformStamped()
        msg4.header.stamp = t.header.stamp
        msg4.header.frame_id = "Front_Left_Hip"
        msg4.child_frame_id = "Front_Left_Thigh"
        msg4.transform.translation.x = P_FLhip_to_FLthigh[0, 3]
        msg4.transform.translation.y = P_FLhip_to_FLthigh[1, 3]
        msg4.transform.translation.z = P_FLhip_to_FLthigh[2, 3]
        quat_FLhip_to_FLthigh = R.from_matrix(P_FLhip_to_FLthigh[:3, :3]).as_quat()
        msg4.transform.rotation.x = quat_FLhip_to_FLthigh[0]
        msg4.transform.rotation.y = quat_FLhip_to_FLthigh[1]      
        msg4.transform.rotation.z = quat_FLhip_to_FLthigh[2]
        msg4.transform.rotation.w = quat_FLhip_to_FLthigh[3]
        self.broadcaster.sendTransform(msg4)


        # # Broadcast FL_thigh to FL_calf joint for visualization in RViz ---##
        msg5 = TransformStamped()
        msg5.header.stamp = t.header.stamp
        msg5.header.frame_id = "Front_Left_Thigh"
        msg5.child_frame_id = "Front_Left_Calf"
        msg5.transform.translation.x = P_FLthigh_to_FLcalf[0, 3]
        msg5.transform.translation.y = P_FLthigh_to_FLcalf[1, 3]
        msg5.transform.translation.z = P_FLthigh_to_FLcalf[2, 3]
        quat_FLthigh_to_FLcalf = R.from_matrix(P_FLthigh_to_FLcalf[:3, :3]).as_quat()
        msg5.transform.rotation.x = quat_FLthigh_to_FLcalf[0]
        msg5.transform.rotation.y = quat_FLthigh_to_FLcalf[1]
        msg5.transform.rotation.z = quat_FLthigh_to_FLcalf[2]
        msg5.transform.rotation.w = quat_FLthigh_to_FLcalf[3]
        self.broadcaster.sendTransform(msg5)

        # Broadcast FL_calf-> FL_calflower joint for visualization in RViz ---##
        msg6 = TransformStamped()
        msg6.header.stamp = t.header.stamp
        msg6.header.frame_id = "Front_Left_Calf"
        msg6.child_frame_id = "Front_Left_Calflower"
        msg6.transform.translation.x = P_FLcalf_to_FLcalflower[0, 3]
        msg6.transform.translation.y = P_FLcalf_to_FLcalflower[1, 3]
        msg6.transform.translation.z = P_FLcalf_to_FLcalflower[2, 3]
        quat_FLcalf_to_FLcalflower = R.from_matrix(P_FLcalf_to_FLcalflower[:3, :3]).as_quat()
        msg6.transform.rotation.x = quat_FLcalf_to_FLcalflower[0]
        msg6.transform.rotation.y = quat_FLcalf_to_FLcalflower[1]
        msg6.transform.rotation.z = quat_FLcalf_to_FLcalflower[2]
        msg6.transform.rotation.w = quat_FLcalf_to_FLcalflower[3]
        self.broadcaster.sendTransform(msg6)

         # Broadcast FL_calfLOWER-> FL_calflower1 joint for visualization in RViz ---##
        msg7 = TransformStamped()
        msg7.header.stamp = t.header.stamp
        msg7.header.frame_id = "Front_Left_Calflower"
        msg7.child_frame_id = "Front_Left_Calflower1"
        msg7.transform.translation.x = P_FLcalflower_to_FLcalflower1[0, 3]
        msg7.transform.translation.y = P_FLcalflower_to_FLcalflower1[1, 3]
        msg7.transform.translation.z = P_FLcalflower_to_FLcalflower1[2, 3]
        quat_FLcalflower_to_FLcalflower1 = R.from_matrix(P_FLcalflower_to_FLcalflower1[:3, :3]).as_quat()
        msg7.transform.rotation.x = quat_FLcalflower_to_FLcalflower1[0]
        msg7.transform.rotation.y = quat_FLcalflower_to_FLcalflower1[1]
        msg7.transform.rotation.z = quat_FLcalflower_to_FLcalflower1[2]
        msg7.transform.rotation.w = quat_FLcalflower_to_FLcalflower1[3]
        self.broadcaster.sendTransform(msg7)


         # Broadcast FL_calf-> FL_foot joint for visualization in RViz ---##
        msg8 = TransformStamped()
        msg8.header.stamp = t.header.stamp
        msg8.header.frame_id = "Front_Left_Calf"
        msg8.child_frame_id = "Front_Left_foot"
        msg8.transform.translation.x = P_FLcalf_to_FL_foot[0, 3]
        msg8.transform.translation.y = P_FLcalf_to_FL_foot[1, 3]
        msg8.transform.translation.z = P_FLcalf_to_FL_foot[2, 3]
        quat_FLcalf_to_FL_foot = R.from_matrix(P_FLcalf_to_FL_foot[:3, :3]).as_quat()
        msg8.transform.rotation.x = quat_FLcalf_to_FL_foot[0]
        msg8.transform.rotation.y = quat_FLcalf_to_FL_foot[1]
        msg8.transform.rotation.z = quat_FLcalf_to_FL_foot[2]
        msg8.transform.rotation.w = quat_FLcalf_to_FL_foot[3]
        self.broadcaster.sendTransform(msg8)

        # ---------------- Auto-stop logic ----------------
        # ---------------- Wait-for-motion + Auto-stop logic ----------------
        t_now = P_C_to_base[:3, 3]
        q_now = quat_C_to_base  # (x,y,z,w)

        if self.prev_P_C_to_base is not None:
            t_prev = self.prev_P_C_to_base[:3, 3]
            dt = float(np.linalg.norm(t_now - t_prev))  # meters
            dtheta = float(quat_angle_delta(self.prev_q_C_to_base, q_now))  # radians

            # 1) First: detect motion (only until we confirm motion happened)
            if not self.has_moved:
                if (dt > self.start_motion_translation_eps) or (dtheta > self.start_motion_rotation_eps):
                    self.moving_count += 1
                else:
                    self.moving_count = 0

                self.get_logger().info(
                    f"[WAIT MOTION] dt={dt:.6f} m, dtheta={dtheta:.6f} rad, "
                    f"moving={self.moving_count}/{self.start_motion_consecutive}"
                )

                if self.moving_count >= self.start_motion_consecutive:
                    self.has_moved = True
                    self.stable_count = 0  # reset stability counter after motion starts
                    self.get_logger().info("Motion detected. Stability-based auto-stop is now enabled.")

            # 2) After motion detected: do stability check to stop
            else:
                if (dt < self.stop_translation_eps) and (dtheta < self.stop_rotation_eps):
                    self.stable_count += 1
                else:
                    self.stable_count = 0

                self.get_logger().info(
                    f"[STABILITY] dt={dt:.6f} m, dtheta={dtheta:.6f} rad, "
                    f"stable={self.stable_count}/{self.stop_consecutive}"
                )

                if self.stable_count >= self.stop_consecutive:
                    self.get_logger().info(
                        f"Transform stabilized for {self.stop_consecutive} consecutive samples. Stopping node."
                    )
                    self.timer.cancel()
                    self.destroy_node()
                    rclpy.shutdown()
                    return

        # Update previous
        self.prev_P_C_to_base = P_C_to_base.copy()
        self.prev_q_C_to_base = q_now.copy()
        # -------------------------------------------------------------------
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
            f.write("Transformation matrix from Camera to Front Right Calf Joint:\n")
            np.savetxt(f, P_C_to_FRcalf, fmt="%.6f")
            f.write("Translation vector from Camera to Front Right Calf Joint: %.6f %.6f %.6f\n" % (P_C_to_FRcalf[0,3], P_C_to_FRcalf[1,3], P_C_to_FRcalf[2,3]))
            f.write("Rotation (r,p,y) from Camera to Front Right Calf Joint: %.6f %.6f %.6f\n" % (rpy_C_to_FRcalf[0], rpy_C_to_FRcalf[1], rpy_C_to_FRcalf[2]))
            f.write("\n")
            f.write("Transformation matrix from Camera to Front Left Calf Joint:\n")
            np.savetxt(f, P_C_to_FLcalf, fmt="%.6f")
            f.write("Translation vector from Camera to Front Left Calf Joint: %.6f %.6f %.6f\n" % (P_C_to_FLcalf[0,3], P_C_to_FLcalf[1,3], P_C_to_FLcalf[2,3]))
            f.write("Rotation (r,p,y) from Camera to Front Left Calf Joint: %.6f %.6f %.6f\n" % (rpy_C_to_FLcalf[0], rpy_C_to_FLcalf[1], rpy_C_to_FLcalf[2]))
            f.write("\n")
            f.write("Transformation matrix from Camera to Rear Right Calf Joint:\n")
            np.savetxt(f, P_C_to_RRcalf, fmt="%.6f")
            f.write("Translation vector from Camera to Rear Right Calf Joint: %.6f %.6f %.6f\n" % (P_C_to_RRcalf[0,3], P_C_to_RRcalf[1,3], P_C_to_RRcalf[2,3]))
            f.write("Rotation (r,p,y) from Camera to Rear Right Calf Joint: %.6f %.6f %.6f\n" % (rpy_C_to_RRcalf[0], rpy_C_to_RRcalf[1], rpy_C_to_RRcalf[2]))
            f.write("\n")
            f.write("Transformation matrix from Camera to Rear Left Calf Joint:\n")
            np.savetxt(f, P_C_to_RLcalf, fmt="%.6f")
            f.write("Translation vector from Camera to Rear Left Calf Joint: %.6f %.6f %.6f\n" % (P_C_to_RLcalf[0,3], P_C_to_RLcalf[1,3], P_C_to_RLcalf[2,3]))
            f.write("Rotation (r,p,y) from Camera to Rear Left Calf Joint: %.6f %.6f %.6f\n" % (rpy_C_to_RLcalf[0], rpy_C_to_RLcalf[1], rpy_C_to_RLcalf[2]))
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
