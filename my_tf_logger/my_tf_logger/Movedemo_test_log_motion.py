#!/usr/bin/env python3
"""
AprilTag-based Unitree Go2 floor-relative base-pose recorder.

TF structure used by this node:

    floor
      |-- camera
      |     `-- object_1_raw (published by apriltag_ros)
      |   ---    object_1     (published by this logger)
      |
      `-- map
            `-- base_link           (existing robot/URDF TF tree)

IMPORTANT:
- This node does NOT publish any direct floor -> robot-base transform.
- The robot URDF therefore has only one base_link parent: map.
- The measured floor -> base transform is used internally for:
    1. continuously recording the base pose with respect to floor,
    2. calculating floor -> map.

The transform relationship is:

    floor_T_base = floor_T_map @ map_T_base

Therefore:

    floor_T_map = floor_T_base @ inverse(map_T_base)

No obstacle tags, goal tags, RViz obstacle markers, deques for debugging,
or JSON storage are included.
"""

import csv
import os
import time
from datetime import datetime

import numpy as np
import rclpy
import tf2_ros
from geometry_msgs.msg import TransformStamped
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.time import Time
from scipy.spatial.transform import Rotation as R, Slerp


# ============================================================
# FIXED CALIBRATION MATRIX P*(Rc -> T1)
# ============================================================
# P_star_Rc_to_T1 = np.array(
#     [
#         [0.998833, 0.039814, -0.027331, 0.008661],
#         [-0.025861, -0.036982, -0.998981, -0.049915],
#         [-0.040784, 0.998523, -0.035909, -0.107167],
#         [0.000000, 0.000000, 0.000000, 1.000000],
#     ],
#     dtype=float,
# )


###----- After the Demo -----------------------------------####

P_star_Rc_to_T1 = np.array ([
 [0.997492, 0.059323, -0.038611, 0.002513],
[-0.034889, -0.062547, -0.997432, -0.041843],
[-0.061586, 0.996277, -0.060320, -0.112739],
[0.000000, 0.000000, 0.000000, 1.000000],
],
dtype=float,
)


# ============================================================
# HELPER FUNCTIONS
# ============================================================
def velocity_token(value):
    """
    Convert a velocity to one stable filename token.

    Examples:
        0.15  -> "0.15"
        0.0   -> "0.0"
        -0.1  -> "-0.1"
    """
    text = f"{float(value):.3f}"

    while text.endswith("0"):
        text = text[:-1]

    if text.endswith("."):
        text += "0"

    if text == "-0.0":
        text = "0.0"

    return text


def quat_to_rot(qx, qy, qz, qw):
    """Quaternion (x, y, z, w) -> rotation matrix and XYZ RPY."""
    quat = np.array([qx, qy, qz, qw], dtype=float)
    rotation = R.from_quat(quat)
    rpy = rotation.as_euler("xyz", degrees=False)
    return rotation.as_matrix(), rpy


def transform_stamped_to_matrix(transform_stamped):
    """Convert TransformStamped into a 4x4 homogeneous matrix."""
    translation = transform_stamped.transform.translation
    quaternion = transform_stamped.transform.rotation

    rotation_matrix, _ = quat_to_rot(
        quaternion.x,
        quaternion.y,
        quaternion.z,
        quaternion.w,
    )

    matrix = np.eye(4, dtype=float)
    matrix[:3, :3] = rotation_matrix
    matrix[:3, 3] = np.array(
        [
            translation.x,
            translation.y,
            translation.z,
        ],
        dtype=float,
    )

    return matrix


# ============================================================
# ROS 2 NODE
# ============================================================
class TFLogger(Node):
    MODE_CONFIG = {
        "Move": {
            "start_event": "MOVE_START",
        },
        "FrontJump": {
            "start_event": "JUMP_START",
        },
        "HandStand": {
            "start_event": "HANDSTAND_START",
        },
        "HandStandMove": {
            "start_event": "HANDSTAND_MOVE_START",
        },
    }

    def __init__(self):
        super().__init__("tf_logger")

        # --------------------------------------------------
        # FRAME PARAMETERS
        # --------------------------------------------------
        self.declare_parameter("parent_frame", "camera")
        self.declare_parameter("child_frame", "object_1")
        self.declare_parameter("raw_object_frame","object_1_raw")
        self.declare_parameter("floor_frame", "floor")
        self.declare_parameter("floor_tag_frame", "floor_tag_raw")
        self.declare_parameter("map_frame", "map")
        self.declare_parameter("base_frame", "base_link")

        ##-- Adding extra 10 secs duration with MOVE START so TF logger can record the last few seconds of robot motion 
        # after MOVE_STOP is detected ---##
        self.declare_parameter(
            "recording_duration_sec",
                    10.0,
                )
            
        self.declare_parameter("rate_hz", 30.0)

        # --------------------------------------------------
        # TRIAL VELOCITY AND DATASET PARAMETERS
        # --------------------------------------------------
        # Pass the same values to this logger and the C++ movement
        # executable. They determine the folder and all CSV names.
        self.declare_parameter("vx", 0.15)
        self.declare_parameter("vy", 0.0)
        self.declare_parameter("vyaw", 0.0)

        self.declare_parameter("mode", "Move")

        # --------------------------------------------------
        # STORAGE MODE
        # --------------------------------------------------
        # "dataset":
        #     Fd_{vx}_{vy}_{vyaw}
        #
        # "trial":
        #     Trials_{vx}_{vy}_{vyaw}/
        #         Trial_{trial_number}_{vx}_{vy}_{vyaw}
        #
        self.declare_parameter("storage_mode", "dataset")
        self.declare_parameter("trial_number", -1)

        self.declare_parameter(
            "dataset_root",
            os.path.expanduser(
                "~/Go2_Walk_Base_Data_Sensor"
            ),
        )

        self.declare_parameter(
            "skill_dataset_root",
            os.path.expanduser(
                "~/Go2_Skill_Base_Data_Sensor"
            ),
        )


        self.parent_frame = (
            self.get_parameter("parent_frame")
            .get_parameter_value()
            .string_value
        )
        self.child_frame = (
            self.get_parameter("child_frame")
            .get_parameter_value()
            .string_value
        )

        self.raw_object_frame = (
            self.get_parameter("raw_object_frame")
            .get_parameter_value()
            .string_value
        )

        self.floor_frame = (
            self.get_parameter("floor_frame")
            .get_parameter_value()
            .string_value
        )
        self.floor_tag_frame = (
            self.get_parameter("floor_tag_frame")
            .get_parameter_value()
            .string_value
        )
        self.map_frame = (
            self.get_parameter("map_frame")
            .get_parameter_value()
            .string_value
        )
        self.base_frame = (
            self.get_parameter("base_frame")
            .get_parameter_value()
            .string_value
        )
        self.rate_hz = (
            self.get_parameter("rate_hz")
            .get_parameter_value()
            .double_value
        )


        self.vx = (
            self.get_parameter("vx")
            .get_parameter_value()
            .double_value
        )
        self.vy = (
            self.get_parameter("vy")
            .get_parameter_value()
            .double_value
        )
        self.vyaw = (
            self.get_parameter("vyaw")
            .get_parameter_value()
            .double_value
        )

        self.mode = (
            self.get_parameter("mode")
            .get_parameter_value()
            .string_value
            .strip()
        )








        ##--- Adding the storage mode and trial number parameters to the TF logger node 
        # so that it can create the appropriate directory structure and file names 
        # for storing the recorded data. The storage mode can be 
        # either "dataset" or "trial", and the trial number is used 
        # to differentiate between different trials when using the "trial" storage mode. The dataset root parameter specifies the root directory where the data will be stored.
        self.storage_mode = (
            self.get_parameter("storage_mode")
            .get_parameter_value()
            .string_value
        )

        self.trial_number = (
            self.get_parameter("trial_number")
            .get_parameter_value()
            .integer_value
        )

         ##-- Validating storage mode and trail number parameters to ensure they are set correctly.--###
        # ============================================================
        # MODE / STORAGE VALIDATION
        # ============================================================

        if self.mode not in self.MODE_CONFIG:
            raise RuntimeError(
                "Unsupported mode. "
                "Expected one of: "
                f"{sorted(self.MODE_CONFIG.keys())}. "
                f"Received: {self.mode}"
            )

        if self.storage_mode not in ("dataset", "trial"):
            raise RuntimeError(
                "storage_mode must be either 'dataset' or 'trial'."
            )

        # Move keeps the original dataset/trial behaviour.
        if self.mode == "Move":

            if (
                self.storage_mode == "trial"
                and self.trial_number < 1
            ):
                raise RuntimeError(
                    "trial_number must be >= 1 when "
                    "mode='Move' and storage_mode='trial'."
                )

            if not all(
                np.isfinite(value)
                for value in (
                    self.vx,
                    self.vy,
                    self.vyaw,
                )
            ):
                raise RuntimeError(
                    "vx, vy, and vyaw must all be finite "
                    "when mode='Move'."
                )

        # Skill commands always use numbered trials.
        else:

            if self.storage_mode != "dataset":
                raise RuntimeError(
                    "FrontJump, HandStand, and HandStandMove "
                    "currently use storage_mode='dataset'."
                )

            if self.trial_number < 1:
                raise RuntimeError(
                    "trial_number must be >= 1 for "
                    f"mode='{self.mode}'."
                )

        self.start_event_name = (
            self.MODE_CONFIG[
                self.mode
            ][
                "start_event"
            ]
        )


        self.dataset_root = os.path.expanduser(
            self.get_parameter("dataset_root")
            .get_parameter_value()
            .string_value
        )

        self.skill_dataset_root = os.path.expanduser(
            self.get_parameter("skill_dataset_root")
            .get_parameter_value()
            .string_value
        )

     

        # ============================================================
        # OUTPUT DIRECTORY / FILE SELECTION
        # ============================================================

        if self.mode == "Move":

            # --------------------------------------------------------
            # MOVE
            # --------------------------------------------------------
            vx_token = velocity_token(self.vx)
            vy_token = velocity_token(self.vy)
            vyaw_token = velocity_token(self.vyaw)

            self.velocity_suffix = (
                f"{vx_token}_{vy_token}_{vyaw_token}"
            )

            self.file_suffix = self.velocity_suffix

            if self.storage_mode == "dataset":

                self.trial_directory = os.path.join(
                    self.dataset_root,
                    f"Fd_{self.velocity_suffix}",
                )

            else:

                trials_directory = os.path.join(
                    self.dataset_root,
                    f"Trials_{self.velocity_suffix}",
                )

                self.trial_directory = os.path.join(
                    trials_directory,
                    (
                        f"Trial_{self.trial_number}_"
                        f"{self.velocity_suffix}"
                    ),
                )

            self.command_event_file = os.path.join(
                self.trial_directory,
                f"cmd_w_{self.velocity_suffix}.csv",
            )

        else:

            # --------------------------------------------------------
            # DISCRETE SKILLS
            #
            # FrontJump
            # HandStand
            # HandStandMove
            # --------------------------------------------------------

            self.velocity_suffix = None

            self.file_suffix = (
                f"{self.mode}_Trial_{self.trial_number}"
            )

            self.trial_directory = os.path.join(
                self.skill_dataset_root,
                self.mode,
                f"Trial_{self.trial_number}",
            )

            self.command_event_file = os.path.join(
                self.trial_directory,
                (
                    f"cmd_{self.mode}_"
                    f"Trial_{self.trial_number}.csv"
                ),
            )

        # ============================================================
        # COMMON OUTPUT FILES
        # ============================================================

        self.output_file = os.path.join(
            self.trial_directory,
            f"Tf_{self.file_suffix}.csv",
        )

        self.filtered_output_file = os.path.join(
            self.trial_directory,
            f"Tf_filtered_{self.file_suffix}.csv",
        )

        self.stationary_output_file = os.path.join(
            self.trial_directory,
            f"Tf_stationary_{self.file_suffix}.csv",
        )

        self.leg_raw_output_file = os.path.join(
            self.trial_directory,
            f"Tf_leg_raw_{self.file_suffix}.csv",
        )

        self.leg_stationary_output_file = os.path.join(
            self.trial_directory,
            f"Tf_leg_stationary_{self.file_suffix}.csv",
        )

        

        ###--- Adding extra 10 secs duration with MOVE START so TF logger can record the last few seconds of robot motion-----####
        self.recording_duration_sec = (
            self.get_parameter("recording_duration_sec")
            .get_parameter_value()
            .double_value
        )

        if self.recording_duration_sec <= 0.0:
            raise RuntimeError(
                "recording_duration_sec must be greater than zero."
            )

        self.recording_duration_ns = int(
            self.recording_duration_sec * 1_000_000_000
        )


        if self.floor_frame == self.floor_tag_frame:
            raise RuntimeError(
                "floor_frame and floor_tag_frame must be different."
            )

        if self.floor_frame == self.map_frame:
            raise RuntimeError(
                "floor_frame and map_frame must be different."
            )


        # --------------------------------------------------
        # TF BUFFER / LISTENER / BROADCASTER
        # --------------------------------------------------
        self.buffer = tf2_ros.Buffer(
            cache_time=Duration(seconds=10.0)
        )

        self.listener = tf2_ros.TransformListener(
            self.buffer,
            self,
            spin_thread=False,
        )

        self.broadcaster = tf2_ros.TransformBroadcaster(self)

        # --------------------------------------------------
        # FLOOR LOCK STATE
        # --------------------------------------------------
        self.last_valid_C_to_floor = None
        self.have_floor_lock = False

        # --------------------------------------------------
        # LAST VALID ROBOT LOCALIZATION
        # --------------------------------------------------
        self.last_valid_floor_to_base = None
        self.last_valid_floor_to_map = None

        # --------------------------------------------------
        # LAST VALID ROBOT-TAG MEASUREMENT
        # --------------------------------------------------
        self.last_valid_C_to_T1 = None
        self.have_object_lock = False

        # --------------------------------------------------
        # COMMAND-WINDOW RECORDING STATE
        # --------------------------------------------------
        self.recording_finished = False
        self.samples_written = 0

        # Used to reject a MOVE_START row left behind by an older trial.
        self.logger_start_wall_time_ns = time.time_ns()

        # A new MOVE_START must be detected before the fixed
        # 10-second recording window can begin.
        self.active_command_start_ns = None

        # Calculated as:
        #
        # MOVE_START + recording_duration_sec
        #
        self.recording_end_wall_time_ns = None
        # Prevent repeated cached object_1 transforms from being saved.
        self.last_processed_object_stamp_ns = None

        # Last raw TF timestamp saved for each leg frame.
        # This prevents writing the exact same cached joint TF repeatedly.
        self.last_leg_stamp_ns = {}

       

        # --------------------------------------------------
        # TIMER
        # --------------------------------------------------

        self.initialize_output_file()
        self.initialize_leg_raw_output_file()

        period = 1.0 / max(self.rate_hz, 0.1)
        self.timer = self.create_timer(period, self.tick)

        self.get_logger().info(
            f"Sampling AprilTag localization at {self.rate_hz:.1f} Hz."
        )
        
        self.get_logger().info(
            f"[TRIAL] Directory: {self.trial_directory}"
        )
        self.get_logger().info(
            f"[TRIAL] Command events: {self.command_event_file}"
        )
        self.get_logger().info(
            f"[TRIAL] Raw TF CSV: {self.output_file}"
        )
        self.get_logger().info(
            f"[TRIAL] Filtered TF CSV: {self.filtered_output_file}"
        )
        self.get_logger().info(
            f"[TRIAL] Stationary TF CSV: {self.stationary_output_file}"
        )
        self.get_logger().info(
            "URDF TF path: "
            f"{self.floor_frame} -> {self.map_frame} -> "
            f"{self.base_frame}"
        )
        self.get_logger().info(
            "This node does not publish a second floor-relative base frame."
        )
        self.get_logger().info(
            "Recording every fresh AprilTag pose from logger startup."
        )
        self.get_logger().info(
            "Repeated cached object_1 transforms will not be saved."
        )
        ##-----Newly added to inform the user about the command_event_file 
        # path where the MOVE_START and MOVE_STOP timestamps are being stored ---###
        self.get_logger().info(
            f"[TRIAL] "
            f"mode={self.mode}, "
            f"trial_number={self.trial_number}"
        )

        if self.mode == "Move":
            self.get_logger().info(
                f"[TRIAL] "
                f"vx={self.vx}, "
                f"vy={self.vy}, "
                f"vyaw={self.vyaw}"
            )

        self.get_logger().info(
            "Waiting for a new "
            f"{self.start_event_name} in: "
            f"{self.command_event_file}"
        )

        self.get_logger().info(
            f"After {self.start_event_name}, "
            "recording will stop at "
            f"{self.start_event_name} + "
            f"{self.recording_duration_sec:.3f} seconds."
        )
        ###---- end of adding recorded timestanp-------------###########
        self.get_logger().info(
            "The raw CSV records fresh poses from logger startup. "
            "The filtered CSV will contain the fixed window from "
            "MOVE_START to MOVE_START + recording_duration_sec."
        )

        self.tag_visible = False
        

    # ========================================================
    # TF BROADCASTING
    # ========================================================
    def broadcast_transform(
        self,
        parent,
        child,
        matrix,
        stamp_msg,
    ):
        """Broadcast one 4x4 homogeneous transform."""
        message = TransformStamped()

        message.header.stamp = stamp_msg
        message.header.frame_id = parent
        message.child_frame_id = child

        message.transform.translation.x = float(matrix[0, 3])
        message.transform.translation.y = float(matrix[1, 3])
        message.transform.translation.z = float(matrix[2, 3])

        quaternion = R.from_matrix(
            matrix[:3, :3]
        ).as_quat()

        message.transform.rotation.x = float(quaternion[0])
        message.transform.rotation.y = float(quaternion[1])
        message.transform.rotation.z = float(quaternion[2])
        message.transform.rotation.w = float(quaternion[3])

        self.broadcaster.sendTransform(message)

    
    # ========================================================
    # FLOOR-RELATIVE LEG-FRAME TRANSFORMATIONS
    # ========================================================
    def compute_and_broadcast_floor_leg_transforms(
        self,
        P_floor_to_base,
        common_time,
        stamp_msg,
    ):
        """
        Estimate the floor-relative poses of all leg frames.

        For every leg frame L:

            P_floor_to_L
                = P_floor_to_base @ P_base_to_L

        P_floor_to_base:
            obtained from AprilTag localization.

        P_base_to_L:
            obtained from the existing robot TF tree, which
            contains the URDF geometry and current encoder-driven
            joint configuration.

        Auxiliary floor-relative frames are broadcast only for
        visualization. Existing URDF frame names are NOT reused,
        because that would create multiple TF parents.
        """

        leg_prefixes = (
            "FL",
            "FR",
            "RL",
            "RR",
        )

        leg_frame_suffixes = (
            "hip",
            "thigh",
            "calf",
            "calflower",
            "calflower1",
            "foot",
        )

        floor_leg_transforms = {}

        for leg_prefix in leg_prefixes:

            for frame_suffix in leg_frame_suffixes:

                link_frame = (
                    f"{leg_prefix}_{frame_suffix}"
                )

                # ------------------------------------------
                # BASE_LINK -> CURRENT LEG FRAME
                # ------------------------------------------

                ##--- for saving the transformation data from leg transfromations---##
                ##-- "common_time" is replaced with "Time()" because leg transormations comes from "/jointstate" publisher
                ##-- at different frequency than april tag detection-------------------#####
                try:
                    t_base_to_link = (
                        self.buffer.lookup_transform(
                            self.base_frame,
                            link_frame,
                            common_time,
                            timeout=Duration(
                                seconds=0.0
                            ),
                        )
                    )

                except Exception as error:
                    self.get_logger().warn(
                        f"TF {self.base_frame} -> "
                        f"{link_frame} unavailable "
                        f"at common_time: {error}"
                    )

                    return None

                P_base_to_link = (
                    transform_stamped_to_matrix(
                        t_base_to_link
                    )
                )

                # ------------------------------------------
                # FLOOR -> CURRENT LEG FRAME
                #
                # floor_T_link =
                #     floor_T_base @ base_T_link
                # ------------------------------------------
                P_floor_to_link = (
                    P_floor_to_base
                    @ P_base_to_link
                )

                # Store it so it can later be written
                # into the dataset.
                floor_leg_transforms[
                    link_frame
                ] = P_floor_to_link.copy()

                # ------------------------------------------
                # BROADCAST AUXILIARY FLOOR-RELATIVE FRAME
                #
                # Do NOT use link_frame directly as child.
                # The original link already has a parent
                # inside the URDF TF tree.
                # ------------------------------------------
                floor_visual_frame = (
                    f"floor_{link_frame}"
                )

                self.broadcast_transform(
                    self.floor_frame,
                    floor_visual_frame,
                    P_floor_to_link,
                    stamp_msg,
                )

        return floor_leg_transforms

    # ========================================================
    # RAW LEG-FRAME TRANSFORM RECORDING
    # ========================================================
    def collect_current_leg_transforms(self):
        """
        Record the latest available base_link -> leg-frame
        transformations.

        IMPORTANT:
        ---------
        Unlike the AprilTag/base samples, these transforms are
        NOT forced to use the AprilTag common_time.

        For each leg frame we request:

            lookup_transform(
                base_link,
                leg_frame,
                Time(),
            )

        Time() means:
            give the latest transform currently available in
            the TF buffer.

        Each returned transform is saved together with its own
        ROS timestamp. Later, the raw leg stream will be
        synchronized/interpolated to the timestamps contained
        in Tf_stationary_*.csv.

        Returns:
            Number of new leg-frame rows written during this call.
        """

        leg_prefixes = (
            "FL",
            "FR",
            "RL",
            "RR",
        )

        leg_frame_suffixes = (
            "hip",
            "thigh",
            "calf",
            "calflower",
            "calflower1",
            "foot",
        )

        rows_to_write = []

        # This is only the local time at which this logger
        # collected the TF data. It is NOT the leg sensor time.
        collection_wall_time_ns = time.time_ns()

        for leg_prefix in leg_prefixes:

            for frame_suffix in leg_frame_suffixes:

                link_frame = (
                    f"{leg_prefix}_{frame_suffix}"
                )

                # ------------------------------------------
                # LATEST BASE_LINK -> LEG FRAME
                # ------------------------------------------
                try:
                    t_base_to_link = (
                        self.buffer.lookup_transform(
                            self.base_frame,
                            link_frame,
                            Time(),
                            timeout=Duration(
                                seconds=0.0
                            ),
                        )
                    )

                except Exception as error:
                    self.get_logger().warn(
                        "[LEG RECORDING] "
                        f"TF {self.base_frame} -> "
                        f"{link_frame} unavailable: "
                        f"{error}"
                    )

                    # Very important:
                    # failure of one frame must NOT prevent
                    # attempting the remaining 23 frames.
                    continue

                # ------------------------------------------
                # TRUE TF TIMESTAMP
                # ------------------------------------------
                leg_ros_sec = int(
                    t_base_to_link.header.stamp.sec
                )

                leg_ros_nanosec = int(
                    t_base_to_link.header.stamp.nanosec
                )

                leg_time_ns = (
                    leg_ros_sec * 1_000_000_000
                    + leg_ros_nanosec
                )

                # Reject an invalid zero timestamp.
                if leg_time_ns <= 0:
                    self.get_logger().warn(
                        "[LEG RECORDING] "
                        f"{link_frame} returned an invalid "
                        "zero TF timestamp."
                    )
                    continue

                # ------------------------------------------
                # AVOID DUPLICATE CACHED TF SAMPLES
                # ------------------------------------------
                previous_leg_stamp_ns = (
                    self.last_leg_stamp_ns.get(
                        link_frame
                    )
                )

                if (
                    previous_leg_stamp_ns is not None
                    and leg_time_ns
                    == previous_leg_stamp_ns
                ):
                    # The TF buffer returned exactly the same
                    # cached measurement as last time.
                    continue

                # ------------------------------------------
                # BASE_LINK -> LEG MATRIX
                # ------------------------------------------
                P_base_to_link = (
                    transform_stamped_to_matrix(
                        t_base_to_link
                    )
                )

                base_to_link_flat = (
                    P_base_to_link
                    .reshape(-1)
                    .tolist()
                )

                row = [
                    leg_time_ns,
                    leg_ros_sec,
                    leg_ros_nanosec,
                    collection_wall_time_ns,
                    link_frame,

                    *[
                        f"{value:.9f}"
                        for value
                        in base_to_link_flat
                    ],
                ]

                rows_to_write.append(row)

                # Mark this leg TF timestamp as saved only
                # after the row has been prepared.
                self.last_leg_stamp_ns[
                    link_frame
                ] = leg_time_ns

        # ----------------------------------------------
        # APPEND ALL SUCCESSFUL LEG FRAMES AT ONCE
        # ----------------------------------------------
        if not rows_to_write:
            return 0

        try:
            with open(
                self.leg_raw_output_file,
                "a",
                newline="",
                encoding="utf-8",
            ) as csv_file:

                writer = csv.writer(csv_file)
                writer.writerows(rows_to_write)

        except OSError as error:
            self.get_logger().error(
                "[LEG RECORDING] Failed to append "
                f"raw leg transforms: {error}"
            )

            return 0

        return len(rows_to_write)


    ###--- This function is for when the TF look ups not vaiable, use the last collected transformations ---###
    ##--  So, in the transformation case, "Floor" is the main parent --------------###
    ##-- (1) Published TRansformations for ---######
    ##---  (a) Floor to Camera--####
    ###-- (b)  Floor to Map --###
    ###--- (c)  Camera to Tag on Robot's head---###
    def broadcast_locked_robot_pose(self, stamp_msg):
        """
        Keep the last valid floor-rooted TF structure alive.

        This does not create CSV samples.
        """
        published_anything = False

        if self.last_valid_C_to_floor is not None:
            P_floor_to_C_locked = np.linalg.inv(
                self.last_valid_C_to_floor
            )

            self.broadcast_transform(
                self.floor_frame,
                self.parent_frame,
                P_floor_to_C_locked,
                stamp_msg,
            )

            published_anything = True

        if self.last_valid_floor_to_map is not None:
            self.broadcast_transform(
                self.floor_frame,
                self.map_frame,
                self.last_valid_floor_to_map,
                stamp_msg,
            )

            published_anything = True

        if self.last_valid_C_to_T1 is not None:
            self.broadcast_transform(
                self.parent_frame,
                self.child_frame,
                self.last_valid_C_to_T1,
                stamp_msg,
            )

            published_anything = True

        return published_anything
        
    # ========================================================
    # CSV RECORDING
    # ========================================================
    @staticmethod
    def csv_header():
        """
        CSV layout:

        T00 ... T33
            = floor -> base transformation matrix

        C_T1_T00 ... C_T1_T33
            = camera -> object_1 transformation matrix
        """

        floor_to_base_matrix_columns = [
            f"T{row}{column}"
            for row in range(4)
            for column in range(4)
        ]

        camera_to_tag_matrix_columns = [
            f"C_T1_T{row}{column}"
            for row in range(4)
            for column in range(4)
        ]

        return [
            "timestamp_iso",
            "wall_time_ns",

            # Common ROS timestamp used by both matrices.
            "ros_sec",
            "ros_nanosec",
            "common_time_ns",

            # Floor-relative base pose.
            "x_floor_m",
            "y_floor_m",
            "z_floor_m",
            "roll_floor_rad",
            "pitch_floor_rad",
            "yaw_floor_rad",
            "qx_floor",
            "qy_floor",
            "qz_floor",
            "qw_floor",

            # P_floor_to_base.
            *floor_to_base_matrix_columns,

            # P_C_to_T1.
            *camera_to_tag_matrix_columns,
        ]
    ####---- Contents of Recorded Pose Samples --------------------------######
    def make_pose_sample(
        self,
        P_floor_to_base,
        P_C_to_T1,
        transform_stamp,
    ):
        """
        Create one synchronized sample containing:

            1. floor -> base
            2. camera -> object_1

        Both matrices correspond to transform_stamp.
        """

        rotation = R.from_matrix(
            P_floor_to_base[:3, :3]
        )

        quaternion = rotation.as_quat()

        rpy = rotation.as_euler(
            "xyz",
            degrees=False,
        )

        pose_wall_time_ns = time.time_ns()

        common_time_ns = (
            int(transform_stamp.sec) * 1_000_000_000
            + int(transform_stamp.nanosec)
        )

        return {
            # Local/logger timestamps.
            "timestamp_iso": datetime.now().isoformat(
                timespec="milliseconds"
            ),
            "wall_time_ns": pose_wall_time_ns,

            # Common AprilTag ROS timestamp.
            "ros_sec": int(transform_stamp.sec),
            "ros_nanosec": int(transform_stamp.nanosec),
            "common_time_ns": common_time_ns,

            # Floor-relative base pose.
            "x": float(P_floor_to_base[0, 3]),
            "y": float(P_floor_to_base[1, 3]),
            "z": float(P_floor_to_base[2, 3]),
            "roll": float(rpy[0]),
            "pitch": float(rpy[1]),
            "yaw": float(rpy[2]),
            "qx": float(quaternion[0]),
            "qy": float(quaternion[1]),
            "qz": float(quaternion[2]),
            "qw": float(quaternion[3]),

            # The two synchronized matrices.
            "floor_to_base_matrix": (
                P_floor_to_base.copy()
            ),
            "camera_to_tag_matrix": (
                P_C_to_T1.copy()
            ),
        }

    @staticmethod
    def leg_raw_csv_header():
        """
        CSV layout for raw robot leg-frame transforms.

        Each row stores one:

            base_link -> leg_frame

        transform using the leg TF's own ROS timestamp.

        These raw transforms are NOT yet synchronized to the
        AprilTag/base timestamps. Synchronization will be done
        later using Tf_stationary_*.csv.
        """

        base_to_leg_matrix_columns = [
            f"B_T{row}{column}"
            for row in range(4)
            for column in range(4)
        ]

        return [
            "leg_time_ns",
            "leg_ros_sec",
            "leg_ros_nanosec",
            "collection_wall_time_ns",
            "frame_name",
            *base_to_leg_matrix_columns,
        ]

    def initialize_leg_raw_output_file(self):
        """
        Create a fresh raw leg-transform CSV.

        The file stores base_link -> leg-frame transforms using
        each TF measurement's own ROS timestamp.
        """

        output_directory = os.path.dirname(
            self.leg_raw_output_file
        )

        if output_directory:
            os.makedirs(
                output_directory,
                exist_ok=True,
            )

        with open(
            self.leg_raw_output_file,
            "w",
            newline="",
            encoding="utf-8",
        ) as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(
                self.leg_raw_csv_header()
            )

        # Remove a synchronized leg file from an older run.
        # It must not look current if this new run has not yet
        # produced a synchronized result.
        try:
            if os.path.isfile(
                self.leg_stationary_output_file
            ):
                os.remove(
                    self.leg_stationary_output_file
                )

        except OSError as error:
            self.get_logger().warn(
                "Could not remove previous synchronized "
                f"leg CSV {self.leg_stationary_output_file}: "
                f"{error}"
            )

        self.get_logger().info(
            "[LEG RECORDING] Raw leg CSV initialized: "
            f"{self.leg_raw_output_file}"
        )



    ####-- Initializing the CSV Output File ---------------######
    def initialize_output_file(self):
        """Create a fresh CSV when the TF logger starts."""
        output_directory = os.path.dirname(
            self.output_file
        )

        if output_directory:
            os.makedirs(
                output_directory,
                exist_ok=True,
            )

        with open(
            self.output_file,
            "w",
            newline="",
            encoding="utf-8",
        ) as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(self.csv_header())

        # Remove derived files from an older run of the same velocity
        # combination. This prevents stale results from looking current
        # if the new filtering step fails.
        for stale_file in (
            self.filtered_output_file,
            self.stationary_output_file,
        ):
            try:
                if os.path.isfile(stale_file):
                    os.remove(stale_file)
            except OSError as error:
                self.get_logger().warn(
                    "Could not remove previous derived CSV "
                    f"{stale_file}: {error}"
                )

    def write_pose_sample(self, sample):
        """
        Append one synchronized transform sample to the CSV.
        """

        floor_to_base_flat = (
            sample["floor_to_base_matrix"]
            .reshape(-1)
            .tolist()
        )

        camera_to_tag_flat = (
            sample["camera_to_tag_matrix"]
            .reshape(-1)
            .tolist()
        )

        row = [
            sample["timestamp_iso"],
            sample["wall_time_ns"],
            sample["ros_sec"],
            sample["ros_nanosec"],
            sample["common_time_ns"],

            f'{sample["x"]:.9f}',
            f'{sample["y"]:.9f}',
            f'{sample["z"]:.9f}',
            f'{sample["roll"]:.9f}',
            f'{sample["pitch"]:.9f}',
            f'{sample["yaw"]:.9f}',
            f'{sample["qx"]:.9f}',
            f'{sample["qy"]:.9f}',
            f'{sample["qz"]:.9f}',
            f'{sample["qw"]:.9f}',

            # P_floor_to_base: T00 ... T33.
            *[
                f"{value:.9f}"
                for value in floor_to_base_flat
            ],

            # P_C_to_T1: C_T1_T00 ... C_T1_T33.
            *[
                f"{value:.9f}"
                for value in camera_to_tag_flat
            ],
        ]

        with open(
            self.output_file,
            "a",
            newline="",
            encoding="utf-8",
        ) as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(row)

        self.samples_written += 1

    
    def filter_rows_to_recording_window(self):
        """
        Create a filtered trajectory using:

            start boundary:
                pose nearest to COMMAND_START

            end boundary:
                pose nearest to
                COMMAND_START + recording_duration_sec

        The complete inclusive range between these two poses is saved.
        """
        if self.active_command_start_ns is None:
            self.get_logger().error(
                "Cannot filter trajectory: "
                f"{self.start_event_name} was not detected."
            )
            return False

        if self.recording_end_wall_time_ns is None:
            self.get_logger().error(
                "Cannot filter trajectory: "
                "recording end time was not calculated."
            )
            return False

        if not os.path.isfile(self.output_file):
            self.get_logger().error(
                "Cannot filter trajectory because the raw CSV "
                f"does not exist: {self.output_file}"
            )
            return False

        filtered_directory = os.path.dirname(
            self.filtered_output_file
        )

        if filtered_directory:
            os.makedirs(
                filtered_directory,
                exist_ok=True,
            )

        total_rows = 0
        invalid_rows = 0
        valid_rows = []

        try:
            with open(
                self.output_file,
                "r",
                newline="",
                encoding="utf-8",
            ) as input_csv:
                reader = csv.DictReader(input_csv)

                if reader.fieldnames is None:
                    self.get_logger().error(
                        "Cannot filter trajectory: "
                        "raw CSV has no header."
                    )
                    return False

                if "wall_time_ns" not in reader.fieldnames:
                    self.get_logger().error(
                        "Cannot filter trajectory: "
                        "wall_time_ns column is missing."
                    )
                    return False

                fieldnames = list(reader.fieldnames)

                for row in reader:
                    total_rows += 1

                    try:
                        pose_wall_time_ns = int(
                            row["wall_time_ns"]
                        )

                    except (
                        TypeError,
                        ValueError,
                        KeyError,
                    ):
                        invalid_rows += 1
                        continue

                    valid_rows.append(
                        (
                            pose_wall_time_ns,
                            row,
                        )
                    )

        except (OSError, csv.Error) as error:
            self.get_logger().error(
                "Failed to read raw trajectory CSV: "
                f"{error}"
            )
            return False

        if not valid_rows:
            self.get_logger().error(
                "No valid timestamped pose rows were found."
            )
            return False

        # Make filtering robust to any out-of-order CSV rows.
        valid_rows.sort(
            key=lambda item: item[0]
        )

        # --------------------------------------------------
        # POSE NEAREST TO MOVE_START
        # --------------------------------------------------
        start_index = min(
            range(len(valid_rows)),
            key=lambda index: abs(
                valid_rows[index][0]
                - self.active_command_start_ns
            ),
        )

        # --------------------------------------------------
        # POSE NEAREST TO COMMAND_START + 10 SECONDS
        # --------------------------------------------------
        end_index = min(
            range(len(valid_rows)),
            key=lambda index: abs(
                valid_rows[index][0]
                - self.recording_end_wall_time_ns
            ),
        )

        if start_index > end_index:
            self.get_logger().error(
                "Selected start boundary occurs after "
                "the selected recording-end boundary."
            )
            return False

        # Keep the complete inclusive trajectory.
        selected_rows = valid_rows[
            start_index:end_index + 1
        ]

        if not selected_rows:
            self.get_logger().error(
                "No rows were selected for the recording window."
            )
            return False

        selected_start_ns = selected_rows[0][0]
        selected_end_ns = selected_rows[-1][0]

        start_offset_ms = (
            selected_start_ns
            - self.active_command_start_ns
        ) / 1_000_000.0

        end_offset_ms = (
            selected_end_ns
            - self.recording_end_wall_time_ns
        ) / 1_000_000.0

        selected_duration_seconds = (
            selected_end_ns
            - selected_start_ns
        ) / 1_000_000_000.0

        try:
            with open(
                self.filtered_output_file,
                "w",
                newline="",
                encoding="utf-8",
            ) as output_csv:
                writer = csv.DictWriter(
                    output_csv,
                    fieldnames=fieldnames,
                )

                writer.writeheader()

                for _, row in selected_rows:
                    writer.writerow(row)

        except (OSError, csv.Error) as error:
            self.get_logger().error(
                "Failed to write filtered trajectory CSV: "
                f"{error}"
            )
            return False

        self.get_logger().info(
            "[FILTER] Fixed-duration filtering completed."
        )

        self.get_logger().info(
            "[FILTER] "
            f"{self.start_event_name}={self.active_command_start_ns}, "
            f"TARGET_END={self.recording_end_wall_time_ns}"
        )

        self.get_logger().info(
            "[FILTER] Selected start pose timestamp: "
            f"{selected_start_ns}"
        )

        self.get_logger().info(
            "[FILTER] Selected end pose timestamp: "
            f"{selected_end_ns}"
        )

        self.get_logger().info(
            "[FILTER] Selected start boundary offset: "
            f"{start_offset_ms:.3f} ms"
        )

        self.get_logger().info(
            "[FILTER] Selected end boundary offset: "
            f"{end_offset_ms:.3f} ms"
        )

        self.get_logger().info(
            "[FILTER] Selected trajectory duration: "
            f"{selected_duration_seconds:.9f} s"
        )

        self.get_logger().info(
            "[FILTER] "
            f"raw_rows={total_rows}, "
            f"kept_rows={len(selected_rows)}, "
            f"invalid_rows={invalid_rows}"
        )

        self.get_logger().info(
            "[FILTER] Saved filtered trajectory: "
            f"{self.filtered_output_file}"
        )

        return True

    def filter_rows_to_first_stationary_timestamp(self):
        """
        Read the fixed 10-second filtered trajectory, calculate
        finite-difference velocities in the robot frame, and find the
        first confirmed stationary timestamp.

        Stationary means:

            abs(vx_body) <= 0.08 m/s
            abs(vy_body) <= 0.08 m/s
            abs(vyaw)    <= 0.10 rad/s

        The condition must hold for five consecutive velocity intervals.

        The first timestamp in that five-interval sequence is selected
        as the stationary timestamp.
        """

        linear_threshold_mps = 0.08
        angular_threshold_radps = 0.10
        required_stationary_intervals = 5

        if not os.path.isfile(self.filtered_output_file):
            self.get_logger().error(
                "Cannot detect the stationary timestamp because "
                "the fixed-duration filtered CSV does not exist: "
                f"{self.filtered_output_file}"
            )
            return False

        # Use the velocity-specific stationary filename created
        # during node initialization.
        stationary_output_file = self.stationary_output_file

        rows = []

        # =====================================================
        # 1. READ THE FIXED-DURATION FILTERED CSV
        # =====================================================
        try:
            with open(
                self.filtered_output_file,
                "r",
                newline="",
                encoding="utf-8",
            ) as input_csv:
                reader = csv.DictReader(input_csv)

                if reader.fieldnames is None:
                    self.get_logger().error(
                        "Cannot detect the stationary timestamp: "
                        "filtered CSV has no header."
                    )
                    return False

                required_columns = {
                    "wall_time_ns",
                    "common_time_ns",
                    "x_floor_m",
                    "y_floor_m",
                    "z_floor_m",
                    "roll_floor_rad",
                    "pitch_floor_rad",
                    "yaw_floor_rad",
                }

                missing_columns = (
                    required_columns - set(reader.fieldnames)
                )

                if missing_columns:
                    self.get_logger().error(
                        "Cannot detect the stationary timestamp. "
                        "Missing CSV columns: "
                        f"{sorted(missing_columns)}"
                    )
                    return False

                original_fieldnames = list(reader.fieldnames)
                ##-- "sensor_time_ns" is the same as "common_time_ns" in the CSV, so we can use either one for velocity calculations. Here, we are 
                # using "common_time_ns" as the sensor timestamp.
                for row in reader:
                    try:
                            rows.append(
                            {
                                "csv_row": row,

                                "wall_time_ns": int(
                                    row["wall_time_ns"]
                                ),

                                "sensor_time_ns": int(
                                    row["common_time_ns"]
                                ),

                                "x": float(
                                    row["x_floor_m"]
                                ),

                                "y": float(
                                    row["y_floor_m"]
                                ),

                                "z": float(
                                    row["z_floor_m"]
                                ),

                                "roll": float(
                                    row["roll_floor_rad"]
                                ),

                                "pitch": float(
                                    row["pitch_floor_rad"]
                                ),

                                "yaw": float(
                                    row["yaw_floor_rad"]
                                ),
                            }
                        )

                    except (
                        KeyError,
                        TypeError,
                        ValueError,
                    ):
                        self.get_logger().warn(
                            "Skipping one invalid pose row while "
                            "calculating body-frame velocities."
                        )

        except (OSError, csv.Error) as error:
            self.get_logger().error(
                "Failed to read the fixed-duration filtered CSV: "
                f"{error}"
            )
            return False

        if len(rows) < (
            required_stationary_intervals + 2
        ):
            self.get_logger().error(
                "Not enough valid pose rows to detect motion "
                "followed by five stationary intervals."
            )
            return False

        rows.sort(
            key=lambda item: item["sensor_time_ns"]
        )

        # Add calculated velocity columns to the final CSV.
        velocity_columns = [
            "delta_time_sec",

            # Linear velocity.
            "vx_body_mps",
            "vy_body_mps",
            "vz_floor_mps",

            # Keep the old yaw-rate value for backward
            # compatibility with the Move dataset.
            "vyaw_body_radps",

            # Full 3-D angular velocity used for
            # heterogeneous-skill stationary detection.
            "omega_x_body_radps",
            "omega_y_body_radps",
            "omega_z_body_radps",

            "stationary_candidate",
        ]

        ##-- builds the final csv header for the stationary output file---####
        ##--original_fieldnames are the column names already present in the filtered CSV.
        ##--velocity_columns are new columns added by this function:
        output_fieldnames = (
            original_fieldnames
            + [
                column
                for column in velocity_columns
                if column not in original_fieldnames
            ]
        )

        # The first row has no preceding pose, so it has no
        # finite-difference velocity.
        first_csv_row = rows[0]["csv_row"]

        first_csv_row["delta_time_sec"] = ""
        first_csv_row["vx_body_mps"] = ""
        first_csv_row["vy_body_mps"] = ""
        first_csv_row["vyaw_body_radps"] = ""
        first_csv_row["vz_floor_mps"] = ""
        first_csv_row["omega_x_body_radps"] = ""
        first_csv_row["omega_y_body_radps"] = ""
        first_csv_row["omega_z_body_radps"] = ""
        ##--- Put "0" for additional safeguard so that first pose in filtered csv must not be considered as stationary candidate---##
        first_csv_row["stationary_candidate"] = "0"

        motion_observed = False
        stationary_run_indices = []
        confirmed_stationary_indices = None

        # =====================================================
        # 2. CALCULATE VELOCITIES BETWEEN CONSECUTIVE POSES
        # =====================================================
        for index in range(1, len(rows)):
            previous = rows[index - 1]
            current = rows[index]

            # Use the AprilTag/ROS sensor timestamps for velocity
            # calculation rather than logger-write timing.
            delta_time_sec = (
                current["sensor_time_ns"]
                - previous["sensor_time_ns"]
            ) / 1_000_000_000.0

            current_csv_row = current["csv_row"]

            if delta_time_sec <= 0.0:

                current_csv_row["delta_time_sec"] = ""

                current_csv_row["vx_body_mps"] = ""
                current_csv_row["vy_body_mps"] = ""
                current_csv_row["vz_floor_mps"] = ""

                current_csv_row["vyaw_body_radps"] = ""

                current_csv_row["omega_x_body_radps"] = ""
                current_csv_row["omega_y_body_radps"] = ""
                current_csv_row["omega_z_body_radps"] = ""

                current_csv_row["stationary_candidate"] = ""

                stationary_run_indices = []
                continue

            delta_x_floor = (
                current["x"] - previous["x"]
            )

            delta_y_floor = (
                current["y"] - previous["y"]
            )

            velocity_x_floor = (
                delta_x_floor / delta_time_sec
            )

            velocity_y_floor = (
                delta_y_floor / delta_time_sec
            )

            velocity_z_floor = (
                current["z"] - previous["z"]
            ) / delta_time_sec

            # =================================================
            # FLOOR-FRAME VELOCITY -> ROBOT-FRAME VELOCITY
            # =================================================
            yaw_i = previous["yaw"]

            cosine_yaw = np.cos(yaw_i)
            sine_yaw = np.sin(yaw_i)

            velocity_x_body = (
                cosine_yaw * velocity_x_floor
                + sine_yaw * velocity_y_floor
            )

            velocity_y_body = (
                -sine_yaw * velocity_x_floor
                + cosine_yaw * velocity_y_floor
            )

            previous_rotation = R.from_euler(
                "xyz",
                [
                    previous["roll"],
                    previous["pitch"],
                    previous["yaw"],
                ],
                degrees=False,
            )

            current_rotation = R.from_euler(
                "xyz",
                [
                    current["roll"],
                    current["pitch"],
                    current["yaw"],
                ],
                degrees=False,
            )

            relative_rotation = (
                previous_rotation.inv()
                * current_rotation
            )

            angular_velocity_body = (
                relative_rotation.as_rotvec()
                / delta_time_sec
            )

            omega_x_body = float(
                angular_velocity_body[0]
            )

            omega_y_body = float(
                angular_velocity_body[1]
            )

            omega_z_body = float(
                angular_velocity_body[2]
            )

            # =================================================
            # YAW ANGULAR VELOCITY
            # =================================================
            raw_delta_yaw = (
                current["yaw"] - previous["yaw"]
            )

            # Wrap the difference to [-pi, pi]. This prevents a
            # false angular-velocity jump when yaw crosses +/-pi.
            wrapped_delta_yaw = np.arctan2(
                np.sin(raw_delta_yaw),
                np.cos(raw_delta_yaw),
            )

            velocity_yaw_body = (
                wrapped_delta_yaw / delta_time_sec
            )

            current_csv_row["delta_time_sec"] = (
                f"{delta_time_sec:.9f}"
            )

            current_csv_row["vx_body_mps"] = (
                f"{velocity_x_body:.9f}"
            )

            current_csv_row["vy_body_mps"] = (
                f"{velocity_y_body:.9f}"
            )

            current_csv_row["vyaw_body_radps"] = (
                f"{velocity_yaw_body:.9f}"
            )

            current_csv_row["vz_floor_mps"] = (
                f"{velocity_z_floor:.9f}"
            )

            current_csv_row["omega_x_body_radps"] = (
                f"{omega_x_body:.9f}"
            )

            current_csv_row["omega_y_body_radps"] = (
                f"{omega_y_body:.9f}"
            )

            current_csv_row["omega_z_body_radps"] = (
                f"{omega_z_body:.9f}"
            )


            stationary_candidate = (
                abs(velocity_x_body)
                <= linear_threshold_mps

                and abs(velocity_y_body)
                <= linear_threshold_mps

                and abs(velocity_z_floor)
                <= linear_threshold_mps

                and abs(omega_x_body)
                <= angular_threshold_radps

                and abs(omega_y_body)
                <= angular_threshold_radps

                and abs(omega_z_body)
                <= angular_threshold_radps
            )

            current_csv_row["stationary_candidate"] = (
                            "1" if stationary_candidate else "0"
                        )

            

            # =================================================
            # 3. FIRST WAIT UNTIL MOVEMENT HAS BEEN OBSERVED
            # =================================================
            # The robot may remain stationary briefly between MOVE_START
            # and its physical response. Those initial stationary samples
            # must not be mistaken for the final stop.
            if not motion_observed:
                if not stationary_candidate:
                    motion_observed = True

                    self.get_logger().info(
                        "[STATIONARY FILTER] Robot motion observed at "
                        f"wall_time_ns={current['wall_time_ns']}."
                    )

                continue

            # =================================================
            # 4. REQUIRE FIVE CONSECUTIVE STATIONARY INTERVALS
            # =================================================
            if stationary_candidate:
                stationary_run_indices.append(index)

                if (
                    len(stationary_run_indices)
                    >= required_stationary_intervals
                ):
                    confirmed_stationary_indices = list(
                        stationary_run_indices[
                            -required_stationary_intervals:
                        ]
                    )
                    break

            else:
                # A moving interval breaks the stationary sequence.
                stationary_run_indices = []

        if not motion_observed:
            self.get_logger().error(
                "No non-stationary velocity interval was detected "
                f"after {self.start_event_name}. "
                "The stationary boundary cannot be identified."
            )
            return False

        if confirmed_stationary_indices is None:
            self.get_logger().error(
                "No sequence of "
                f"{required_stationary_intervals} consecutive "
                "stationary velocity intervals was found within "
                "the 10-second recording window."
            )
            return False

        # =====================================================
        # 5. SELECT THE FIRST TIMESTAMP IN THE CONFIRMED RUN
        # =====================================================
        stationary_row_index = (
            confirmed_stationary_indices[0]
        )

        stationary_wall_time_ns = rows[
            stationary_row_index
        ]["wall_time_ns"]

        

        stationary_timestamp_iso = rows[
            stationary_row_index
        ]["csv_row"].get(
            "timestamp_iso",
            "",
        )

        stationary_confirmation_timestamps = [
            rows[index]["wall_time_ns"]
            for index in confirmed_stationary_indices
        ]

        stationary_confirmation_sensor_timestamps = [
                    rows[index]["sensor_time_ns"]
                    for index in confirmed_stationary_indices
                ]

        # Include every row from the beginning of the fixed filtered
        # trajectory through the first confirmed stationary timestamp.
        selected_rows = rows[
            :stationary_row_index + 1
        ]

        # =====================================================
        # 6. WRITE THE STATIONARY-FILTERED CSV
        # =====================================================
        try:
            with open(
                stationary_output_file,
                "w",
                newline="",
                encoding="utf-8",
            ) as output_csv:
                writer = csv.DictWriter(
                    output_csv,
                    fieldnames=output_fieldnames,
                )

                writer.writeheader()

                for item in selected_rows:
                    writer.writerow(
                        item["csv_row"]
                    )

        except (OSError, csv.Error) as error:
            self.get_logger().error(
                "Failed to write the stationary-filtered CSV: "
                f"{error}"
            )
            return False
        ##-- taking the first filtered timestamp as the actual start of the movement duration calculation---##
        selected_csv_start_ns = rows[0]["wall_time_ns"]

        selected_trajectory_duration_sec = (
            stationary_wall_time_ns
            - selected_csv_start_ns
        ) / 1_000_000_000.0

        command_to_stationary_sec = (
            stationary_wall_time_ns
            - self.active_command_start_ns
        ) / 1_000_000_000.0

        ##--- because it subtacts with the between detection timestamps----------------##
        selected_start_sensor_time_ns = (
            rows[0]["sensor_time_ns"]
        )

        stationary_sensor_time_ns = (
            rows[stationary_row_index]["sensor_time_ns"]
        )

        sensor_trajectory_duration_sec = (
            stationary_sensor_time_ns
            - selected_start_sensor_time_ns
        ) / 1_000_000_000.0


        self.get_logger().info(
            f"[STATIONARY FILTER] First confirmed stationary "
            f"timestamp: {stationary_wall_time_ns}"
        )

        if stationary_timestamp_iso:
            self.get_logger().info(
                f"[STATIONARY FILTER] First confirmed stationary "
                f"ISO timestamp: {stationary_timestamp_iso}"
            )

        self.get_logger().info(
            f"[STATIONARY FILTER] Five confirmation timestamps: "
            f"{stationary_confirmation_timestamps}"
        )

        self.get_logger().info(
            f"[STATIONARY FILTER] Selected TF trajectory duration: "
            f"{selected_trajectory_duration_sec:.9f} seconds."
        )

        self.get_logger().info(
            f"[STATIONARY FILTER] "
           f"{self.start_event_name}-to-stationary confirmation: "
            f"{command_to_stationary_sec:.9f} seconds."
        )
        ##---- added----##########
        self.get_logger().info(
            f"[STATIONARY FILTER] AprilTag-timestamp trajectory duration: "
            f"{sensor_trajectory_duration_sec:.9f} seconds."
        )
        self.get_logger().info(
            f"[STATIONARY FILTER] Stationary confirmation sensor timestamps: "
            f"{stationary_confirmation_sensor_timestamps}"
        )

        self.get_logger().info(
            f"[STATIONARY FILTER] Thresholds: "
            f"|vx_body|, |vy_body|, |vz_floor| "
            f"<= {linear_threshold_mps:.3f} m/s; "
            f"|omega_x|, |omega_y|, |omega_z| "
            f"<= {angular_threshold_radps:.3f} rad/s."
        )

        self.get_logger().info(
            f"[STATIONARY FILTER] Saved trajectory from {self.start_event_name} "
            "to the first confirmed stationary timestamp: "
            f"{stationary_output_file}"
        )

        return True

        # ========================================================
    # SYNCHRONIZE RAW LEG POSES TO STATIONARY BASE TIMESTAMPS
    # ========================================================
    def synchronize_leg_poses_to_stationary_base(self):
        """
        Synchronize the raw base_link -> leg-frame TF stream
        to the AprilTag/base timestamps contained in:

            Tf_stationary_*.csv

        For every stationary base timestamp t_base and every
        leg frame:

            1. Find raw leg TF samples at t0 and t1 such that

                   t0 <= t_base <= t1

            2. Interpolate translation linearly.

            3. Interpolate orientation with quaternion SLERP.

            4. Obtain synchronized:

                   P_base_to_link(t_base)

            5. Combine with the recorded AprilTag base pose:

                   P_floor_to_link(t_base)
                       =
                   P_floor_to_base(t_base)
                       @
                   P_base_to_link(t_base)

        The resulting Tf_leg_stationary_*.csv therefore uses
        exactly the same base timestamps as Tf_stationary_*.csv.

        Returns:
            True only if all 24 leg frames were successfully
            synchronized for every stationary base sample.
        """

        # ====================================================
        # MAXIMUM ALLOWED LEG INTERPOLATION GAP
        # ====================================================
        # This is an experimental data-quality threshold.
        #
        # ====================================================
        # LEG INTERPOLATION GAP WARNING THRESHOLD
        # ====================================================
        # This threshold is used only as a data-quality flag.
        #
        # If:
        #
        #     t1 - t0 <= 100 ms
        #
        # the synchronized sample is marked normally.
        #
        # If:
        #
        #     t1 - t0 > 100 ms
        #
        # the pose is STILL interpolated and saved, but
        # large_gap_warning is set to 1 and a warning is
        # reported at the end.
        max_interpolation_gap_ms = 100.0

        max_interpolation_gap_ns = int(
            max_interpolation_gap_ms * 1_000_000
        )

        # ----------------------------------------------------
        # 1. EXPECTED LEG FRAMES
        # ----------------------------------------------------
        leg_prefixes = (
            "FL",
            "FR",
            "RL",
            "RR",
        )

        leg_frame_suffixes = (
            "hip",
            "thigh",
            "calf",
            "calflower",
            "calflower1",
            "foot",
        )

        expected_frames = [
            f"{leg_prefix}_{frame_suffix}"
            for leg_prefix in leg_prefixes
            for frame_suffix in leg_frame_suffixes
        ]

        # ----------------------------------------------------
        # 2. MATRIX COLUMN NAMES
        # ----------------------------------------------------
        base_matrix_columns = [
            f"T{row}{column}"
            for row in range(4)
            for column in range(4)
        ]

        raw_leg_matrix_columns = [
            f"B_T{row}{column}"
            for row in range(4)
            for column in range(4)
        ]

        floor_leg_matrix_columns = [
            f"F_T{row}{column}"
            for row in range(4)
            for column in range(4)
        ]

        # ----------------------------------------------------
        # 3. CHECK INPUT FILES
        # ----------------------------------------------------
        if not os.path.isfile(
            self.stationary_output_file
        ):
            self.get_logger().error(
                "[LEG SYNC] Cannot synchronize leg poses "
                "because stationary base CSV does not exist: "
                f"{self.stationary_output_file}"
            )
            return False

        if not os.path.isfile(
            self.leg_raw_output_file
        ):
            self.get_logger().error(
                "[LEG SYNC] Cannot synchronize leg poses "
                "because raw leg CSV does not exist: "
                f"{self.leg_raw_output_file}"
            )
            return False

        # ====================================================
        # 4. READ STATIONARY BASE TRAJECTORY
        # ====================================================
        base_samples = []

        try:
            with open(
                self.stationary_output_file,
                "r",
                newline="",
                encoding="utf-8",
            ) as base_csv:

                reader = csv.DictReader(base_csv)

                if reader.fieldnames is None:
                    self.get_logger().error(
                        "[LEG SYNC] Stationary base CSV "
                        "has no header."
                    )
                    return False

                required_base_columns = {
                    "common_time_ns",
                    *base_matrix_columns,
                }

                missing_base_columns = (
                    required_base_columns
                    - set(reader.fieldnames)
                )

                if missing_base_columns:
                    self.get_logger().error(
                        "[LEG SYNC] Stationary base CSV "
                        "is missing columns: "
                        f"{sorted(missing_base_columns)}"
                    )
                    return False

                for row in reader:

                    try:
                        base_time_ns = int(
                            row["common_time_ns"]
                        )

                        base_matrix_values = [
                            float(row[column])
                            for column
                            in base_matrix_columns
                        ]

                        P_floor_to_base = (
                            np.array(
                                base_matrix_values,
                                dtype=float,
                            )
                            .reshape(4, 4)
                        )

                    except (
                        KeyError,
                        TypeError,
                        ValueError,
                    ):
                        self.get_logger().warn(
                            "[LEG SYNC] Skipping one "
                            "invalid stationary base row."
                        )
                        continue

                    if not np.all(
                        np.isfinite(P_floor_to_base)
                    ):
                        self.get_logger().warn(
                            "[LEG SYNC] Skipping one "
                            "non-finite base matrix."
                        )
                        continue

                    base_samples.append(
                        (
                            base_time_ns,
                            P_floor_to_base,
                        )
                    )

        except (OSError, csv.Error) as error:
            self.get_logger().error(
                "[LEG SYNC] Failed to read stationary "
                f"base CSV: {error}"
            )
            return False

        if not base_samples:
            self.get_logger().error(
                "[LEG SYNC] No valid stationary "
                "base samples were found."
            )
            return False

        base_samples.sort(
            key=lambda item: item[0]
        )

        # ====================================================
        # 5. READ RAW LEG TRANSFORMS
        # ====================================================
        leg_samples = {
            frame_name: []
            for frame_name in expected_frames
        }

        try:
            with open(
                self.leg_raw_output_file,
                "r",
                newline="",
                encoding="utf-8",
            ) as leg_csv:

                reader = csv.DictReader(leg_csv)

                if reader.fieldnames is None:
                    self.get_logger().error(
                        "[LEG SYNC] Raw leg CSV "
                        "has no header."
                    )
                    return False

                required_leg_columns = {
                    "leg_time_ns",
                    "frame_name",
                    *raw_leg_matrix_columns,
                }

                missing_leg_columns = (
                    required_leg_columns
                    - set(reader.fieldnames)
                )

                if missing_leg_columns:
                    self.get_logger().error(
                        "[LEG SYNC] Raw leg CSV "
                        "is missing columns: "
                        f"{sorted(missing_leg_columns)}"
                    )
                    return False

                for row in reader:

                    frame_name = row.get(
                        "frame_name",
                        "",
                    ).strip()

                    if frame_name not in leg_samples:
                        continue

                    try:
                        leg_time_ns = int(
                            row["leg_time_ns"]
                        )

                        matrix_values = [
                            float(row[column])
                            for column
                            in raw_leg_matrix_columns
                        ]

                        P_base_to_link = (
                            np.array(
                                matrix_values,
                                dtype=float,
                            )
                            .reshape(4, 4)
                        )

                    except (
                        KeyError,
                        TypeError,
                        ValueError,
                    ):
                        self.get_logger().warn(
                            "[LEG SYNC] Skipping one "
                            "invalid raw leg row."
                        )
                        continue

                    if not np.all(
                        np.isfinite(P_base_to_link)
                    ):
                        continue

                    leg_samples[
                        frame_name
                    ].append(
                        (
                            leg_time_ns,
                            P_base_to_link,
                        )
                    )

        except (OSError, csv.Error) as error:
            self.get_logger().error(
                "[LEG SYNC] Failed to read raw "
                f"leg CSV: {error}"
            )
            return False

        # ----------------------------------------------------
        # SORT + REMOVE DUPLICATE TIMESTAMPS
        # ----------------------------------------------------
        for frame_name in expected_frames:

            # Dictionary removes duplicate timestamps.
            # If duplicates somehow exist, the last matrix
            # for that timestamp is retained.
            unique_samples = {}

            for (
                leg_time_ns,
                P_base_to_link,
            ) in leg_samples[frame_name]:

                unique_samples[
                    leg_time_ns
                ] = P_base_to_link

            leg_samples[frame_name] = sorted(
                unique_samples.items(),
                key=lambda item: item[0],
            )

        # ====================================================
        # 6. CREATE SYNCHRONIZED OUTPUT FILE
        # ====================================================
        output_directory = os.path.dirname(
            self.leg_stationary_output_file
        )

        if output_directory:
            os.makedirs(
                output_directory,
                exist_ok=True,
            )

        output_header = [
            "base_sample_index",
            "base_common_time_ns",
            "frame_name",

            # Raw leg timestamps surrounding t_base.
            "source_t0_ns",
            "source_t1_ns",

            # Interpolation information.
            "interpolation_alpha",
            "source_t0_offset_ms",
            "source_t1_offset_ms",

            # Total distance between surrounding samples:
            #
            #     interpolation_gap_ms
            #         = (t1 - t0)
            #
            "interpolation_gap_ms",
            "large_gap_warning",

            # Final synchronized floor -> leg matrix.
            *floor_leg_matrix_columns,
        ]

        written_rows = 0
        missing_rows = 0
        missing_examples = []
        large_gap_rows = 0
        large_gap_examples = []

        try:
            with open(
                self.leg_stationary_output_file,
                "w",
                newline="",
                encoding="utf-8",
            ) as output_csv:

                writer = csv.writer(output_csv)
                writer.writerow(output_header)

                # ============================================
                # 7. SYNCHRONIZE EACH BASE TIMESTAMP
                # ============================================
                for (
                    base_sample_index,
                    base_sample,
                ) in enumerate(base_samples):

                    (
                        base_time_ns,
                        P_floor_to_base,
                    ) = base_sample

                    for frame_name in expected_frames:

                        samples = leg_samples[
                            frame_name
                        ]

                        if not samples:
                            missing_rows += 1

                            if len(
                                missing_examples
                            ) < 10:
                                missing_examples.append(
                                    (
                                        base_time_ns,
                                        frame_name,
                                        "no raw samples",
                                    )
                                )

                            continue

                        sample_times = np.array(
                            [
                                sample[0]
                                for sample in samples
                            ],
                            dtype=np.int64,
                        )

                        # First raw leg timestamp >= base time.
                        right_index = int(
                            np.searchsorted(
                                sample_times,
                                base_time_ns,
                                side="left",
                            )
                        )

                        # ------------------------------------
                        # EXACT TIMESTAMP MATCH
                        # ------------------------------------
                        if (
                            right_index
                            < len(samples)
                            and samples[
                                right_index
                            ][0] == base_time_ns
                        ):
                            t0_ns = base_time_ns
                            t1_ns = base_time_ns

                            P_base_to_link_sync = (
                                samples[
                                    right_index
                                ][1].copy()
                            )

                            alpha = 0.0

                            # Exact timestamp match:
                            # there is no interpolation interval.
                            interpolation_gap_ns = 0
                            interpolation_gap_ms = 0.0

                        # ------------------------------------
                        # NO LEFT OR RIGHT BRACKET
                        # ------------------------------------
                        elif (
                            right_index == 0
                            or right_index
                            >= len(samples)
                        ):
                            missing_rows += 1

                            if len(
                                missing_examples
                            ) < 10:
                                missing_examples.append(
                                    (
                                        base_time_ns,
                                        frame_name,
                                        "no surrounding bracket",
                                    )
                                )

                            continue

                        # ------------------------------------
                        # INTERPOLATE BETWEEN t0 AND t1
                        # ------------------------------------
                        else:
                            (
                                t0_ns,
                                P0,
                            ) = samples[
                                right_index - 1
                            ]

                            (
                                t1_ns,
                                P1,
                            ) = samples[
                                right_index
                            ]

                            if t1_ns <= t0_ns:
                                missing_rows += 1
                                continue

                            # ==================================
                            # INTERPOLATION GAP QUALITY CHECK
                            # ==================================
                            interpolation_gap_ns = (
                                t1_ns - t0_ns
                            )

                            interpolation_gap_ms = (
                                interpolation_gap_ns
                                / 1_000_000.0
                            )

                            if (
                            interpolation_gap_ns
                            > max_interpolation_gap_ns
                            ):
                                # --------------------------------------------------
                                # QUALITY WARNING ONLY
                                #
                                # The synchronized pose is still calculated and
                                # saved, but we record that the interpolation used
                                # a larger-than-recommended temporal bracket.
                                # --------------------------------------------------
                                large_gap_rows += 1

                                if len(
                                    large_gap_examples
                                ) < 10:
                                    large_gap_examples.append(
                                        (
                                            base_time_ns,
                                            frame_name,
                                            t0_ns,
                                            t1_ns,
                                            interpolation_gap_ms,
                                        )
                                    )

                                # IMPORTANT:
                                # Do NOT increment missing_rows.
                                # Do NOT continue.
                                #
                                # Execution deliberately falls through to alpha
                                # calculation, translation interpolation and SLERP.

                            # ==================================
                            # INTERPOLATION FACTOR
                            # ==================================
                            alpha = (
                                base_time_ns
                                - t0_ns
                            ) / (
                                t1_ns
                                - t0_ns
                            )

                            # ------------------------------
                            # Translation interpolation
                            # ------------------------------
                            p0 = P0[:3, 3]
                            p1 = P1[:3, 3]

                            p_sync = (
                                (1.0 - alpha) * p0
                                + alpha * p1
                            )

                            # ------------------------------
                            # Rotation interpolation
                            # using quaternion SLERP
                            # ------------------------------
                            rotations = R.from_matrix(
                                np.stack(
                                    [
                                        P0[:3, :3],
                                        P1[:3, :3],
                                    ],
                                    axis=0,
                                )
                            )

                            slerp = Slerp(
                                [0.0, 1.0],
                                rotations,
                            )

                            R_sync = (
                                slerp(
                                    [alpha]
                                )
                                .as_matrix()[0]
                            )

                            P_base_to_link_sync = (
                                np.eye(
                                    4,
                                    dtype=float,
                                )
                            )

                            P_base_to_link_sync[
                                :3,
                                :3
                            ] = R_sync

                            P_base_to_link_sync[
                                :3,
                                3
                            ] = p_sync

                        # ==================================================
                        # QUALITY FLAG FOR THE FINAL SYNCHRONIZED SAMPLE
                        #
                        # At this point:
                        #   - exact matches have interpolation_gap_ns = 0
                        #   - interpolated samples have interpolation_gap_ns = t1 - t0
                        #   - impossible cases have already used 'continue'
                        # ==================================================
                        large_gap_warning = (
                            1
                            if interpolation_gap_ns
                            > max_interpolation_gap_ns
                            else 0
                        )   

                        # ====================================
                        # 8. FLOOR -> LEG AT BASE TIMESTAMP
                        # ====================================
                        P_floor_to_link = (
                            P_floor_to_base
                            @ P_base_to_link_sync
                        )

                        floor_to_link_flat = (
                            P_floor_to_link
                            .reshape(-1)
                            .tolist()
                        )

                        source_t0_offset_ms = (
                            base_time_ns
                            - t0_ns
                        ) / 1_000_000.0

                        source_t1_offset_ms = (
                            t1_ns
                            - base_time_ns
                        ) / 1_000_000.0

                        writer.writerow(
                            [
                                base_sample_index,
                                base_time_ns,
                                frame_name,
                                t0_ns,
                                t1_ns,
                                f"{alpha:.9f}",

                                (
                                    f"{source_t0_offset_ms:.6f}"
                                ),

                                (
                                    f"{source_t1_offset_ms:.6f}"
                                ),

                                (
                                    f"{interpolation_gap_ms:.6f}"
                                ),

                                # 0 = normal interpolation bracket
                                # 1 = bracket exceeded 100 ms,
                                #     but pose was still interpolated/saved
                                str(large_gap_warning),

                                *[
                                    f"{value:.9f}"
                                    for value
                                    in floor_to_link_flat
                                ],
                            ]
                        )

                        written_rows += 1

        except (OSError, csv.Error) as error:
            self.get_logger().error(
                "[LEG SYNC] Failed to write "
                f"synchronized leg CSV: {error}"
            )
            return False

        # ====================================================
        # 9. VALIDATE COMPLETENESS
        # ====================================================
        expected_rows = (
            len(base_samples)
            * len(expected_frames)
        )

        self.get_logger().info(
            "[LEG SYNC] "
            f"Base samples={len(base_samples)}, "
            f"frames/base={len(expected_frames)}, "
            f"expected_rows={expected_rows}, "
            f"written_rows={written_rows}, "
            f"missing_rows={missing_rows}, "
            f"large_gap_rows={large_gap_rows}, "
            f"warning_threshold={max_interpolation_gap_ms:.1f} ms."
        )

        if large_gap_examples:
            self.get_logger().warn(
                "[LEG SYNC] Large interpolation gaps detected. "
                "These poses were STILL interpolated and saved, "
                "but should be treated as lower-confidence samples. "
                f"Warning threshold: "
                f"{max_interpolation_gap_ms:.1f} ms. "
                "Examples "
                "(base_time_ns, frame_name, "
                "t0_ns, t1_ns, gap_ms): "
                f"{large_gap_examples}"
            )

        if missing_examples:
            self.get_logger().warn(
                "[LEG SYNC] Example missing "
                f"synchronizations: {missing_examples}"
            )

        if written_rows == 0:
            self.get_logger().error(
                "[LEG SYNC] No synchronized "
                "leg poses were generated."
            )
            return False

        if written_rows != expected_rows:
            self.get_logger().error(
                "[LEG SYNC] Synchronized leg file "
                "was created but is incomplete. "
                f"Expected {expected_rows} rows, "
                f"created {written_rows}."
            )
            return False

        self.get_logger().info(
            "[LEG SYNC] Complete synchronized "
            "floor-relative leg trajectory saved: "
            f"{self.leg_stationary_output_file}"
        )

        return True

    
    ###--- Reading the Timestamps of COMMAND_START and COMMAND_STOP from the command_event_file to check if the robot is moving or not ---###
    def read_command_start(self):
        """
        Read the latest start-event timestamp belonging to
        this logger trial.

        Move:
            event = MOVE_START
            validated using vx, vy, vyaw

        FrontJump:
            event = JUMP_START
            validated using mode and trial_number

        HandStand:
            event = HANDSTAND_START

        HandStandMove:
            event = HANDSTAND_MOVE_START

        Returns:
            wall_time_ns of the matching start event,
            or None if none is currently available.
        """

        if not os.path.isfile(
            self.command_event_file
        ):
            return None

        command_start_ns = None

        try:
            with open(
                self.command_event_file,
                "r",
                newline="",
                encoding="utf-8",
            ) as event_file:

                reader = csv.DictReader(
                    event_file
                )

                for row in reader:

                    event_name = row.get(
                        "event",
                        "",
                    ).strip()

                    # ------------------------------------------
                    # START EVENT MUST MATCH CURRENT MODE
                    # ------------------------------------------
                    if (
                        event_name
                        != self.start_event_name
                    ):
                        continue

                    timestamp_text = row.get(
                        "wall_time_ns",
                        "",
                    ).strip()

                    if not timestamp_text:
                        continue

                    # ==========================================
                    # MOVE VALIDATION
                    # ==========================================
                    if self.mode == "Move":

                        try:
                            row_vx = float(
                                row.get("vx", "")
                            )

                            row_vy = float(
                                row.get("vy", "")
                            )

                            row_vyaw = float(
                                row.get("vyaw", "")
                            )

                        except (
                            TypeError,
                            ValueError,
                        ):
                            continue

                        command_matches_trial = (
                            np.isclose(
                                row_vx,
                                self.vx,
                                atol=1e-6,
                            )
                            and np.isclose(
                                row_vy,
                                self.vy,
                                atol=1e-6,
                            )
                            and np.isclose(
                                row_vyaw,
                                self.vyaw,
                                atol=1e-6,
                            )
                        )

                        if not command_matches_trial:

                            self.get_logger().warn(
                                "Ignoring MOVE_START because "
                                "its velocity command does not "
                                "match this logger trial: "
                                f"CSV=({row_vx}, "
                                f"{row_vy}, "
                                f"{row_vyaw}), "
                                f"logger=({self.vx}, "
                                f"{self.vy}, "
                                f"{self.vyaw})"
                            )

                            continue

                    # ==========================================
                    # DISCRETE-SKILL VALIDATION
                    # ==========================================
                    else:

                        row_mode = row.get(
                            "mode",
                            "",
                        ).strip()

                        try:
                            row_trial_number = int(
                                row.get(
                                    "trial_number",
                                    "",
                                )
                            )

                        except (
                            TypeError,
                            ValueError,
                        ):
                            continue

                        command_matches_trial = (
                            row_mode == self.mode
                            and row_trial_number
                            == self.trial_number
                        )

                        if not command_matches_trial:

                            self.get_logger().warn(
                                f"Ignoring "
                                f"{self.start_event_name} "
                                "because its skill/trial "
                                "does not match this logger: "
                                f"CSV=({row_mode}, "
                                f"{row_trial_number}), "
                                f"logger=({self.mode}, "
                                f"{self.trial_number})"
                            )

                            continue

                    # Latest matching row wins.
                    command_start_ns = int(
                        timestamp_text
                    )

        except (
            OSError,
            ValueError,
            csv.Error,
        ) as error:

            self.get_logger().warn(
                "Could not read "
                f"{self.start_event_name} yet: "
                f"{error}"
            )

            return None

        return command_start_ns

    ####-----Function that estimates how 'COMMAND_START' and COMMAND_STOP timestamps 
    # are being stored in the command_event_file and then calculates 
    # the recording_end_wall_time_ns based on the COMMAND_START timestamp and 
    # the recording_duration_sec parameter ---####
    def update_command_start(self):
        """
        Accept one new start event created after this logger
        started and calculate the fixed recording deadline.

        recording_end_wall_time_ns
            =
        command_start
            +
        recording_duration_ns
        """

        # One command/skill execution per logger run.
        if (
            self.active_command_start_ns
            is not None
        ):
            return

        command_start_ns = (
            self.read_command_start()
        )

        if command_start_ns is None:
            return

        # Reject stale command CSV from an older run.
        if (
            command_start_ns
            < self.logger_start_wall_time_ns
        ):
            return

        self.active_command_start_ns = (
            command_start_ns
        )

        self.recording_end_wall_time_ns = (
            self.active_command_start_ns
            + self.recording_duration_ns
        )

        self.get_logger().info(
            "[RECORDING WINDOW] "
            f"New {self.start_event_name} detected: "
            f"{self.active_command_start_ns}"
        )

        self.get_logger().info(
            "[RECORDING WINDOW] "
            "Target recording end: "
            f"{self.recording_end_wall_time_ns}"
        )

        self.get_logger().info(
            "[RECORDING WINDOW] "
            "Recording duration after "
            f"{self.start_event_name}: "
            f"{self.recording_duration_sec:.3f} seconds."
        )

    
    

    def finish_recording(self):
        """
        Stop CSV recording when the fixed duration after the
        current command start event has elapsed.

        The node remains alive so the final floor -> map transform
        continues to be published for RViz.
        """
        if self.recording_finished:
            return

        self.recording_finished = True

        self.get_logger().info(
            f"{self.recording_duration_sec:.3f}-second "
            "recording boundary reached. "
            f"Raw recording completed with "
            f"{self.samples_written} samples."
        )

        self.get_logger().info(
            f"Saved raw trajectory: {self.output_file}"
        )

        filter_success = (
            self.filter_rows_to_recording_window()
        )

        if filter_success:
            self.get_logger().info(
                f"{self.start_event_name}-to-recording-end trajectory "
                "created successfully."
            )

            stationary_filter_success = (
                self.filter_rows_to_first_stationary_timestamp()
            )

            if stationary_filter_success:
                self.get_logger().info(
                    f"{self.start_event_name}-to-stationary trajectory "
                    "created successfully."
                )

                # ------------------------------------------
                # SYNCHRONIZE RAW LEG TF DATA TO THE
                # FINAL STATIONARY BASE TIMESTAMPS
                # ------------------------------------------
                leg_sync_success = (
                    self.synchronize_leg_poses_to_stationary_base()
                )

                if leg_sync_success:
                    self.get_logger().info(
                        "[LEG SYNC] Synchronized "
                        "floor-relative leg trajectory "
                        "created successfully."
                    )

                else:
                    self.get_logger().error(
                        "[LEG SYNC] Leg synchronization "
                        "was incomplete or failed. "
                        "The raw leg CSV has been preserved "
                        "for inspection."
                    )

            else:
                self.get_logger().error(
                    "Stationary-boundary filtering failed. "
                    "The fixed-duration filtered CSV has been preserved."
                )

        else:
            self.get_logger().error(
                "Fixed-duration trajectory filtering failed. "
                "The raw trajectory has been preserved."
            )

        self.get_logger().info(
            "CSV recording has stopped."
        )

        self.get_logger().info(
            "The final floor -> map bridge will continue "
            "to be published for RViz."
        )

        self.get_logger().info(
            "Press Ctrl+C when you are ready to terminate "
            "the TF logger."
        )

    # ========================================================
    # MAIN TICK FUNCTION
    # ========================================================
    def tick(self):
        if self.recording_finished:
            # Recording has ended, but keep the final TF bridge alive.
            self.broadcast_locked_robot_pose(
                ###--- when the logger stops in collecting new transforms, the last transformations will remain active in Rviz---####
                ##-- This happens when the final Filtered CSV File gets created and the program stops recording ----######
                self.get_clock().now().to_msg()
            )
            return

        ##---- calling the function to collect the timestamp before Command Command with non-zero velocity being sent---###
        ###---- and timestamp after Command Command with zero velocity values being sent ---------------#####
        # Read only COMMAND_START from the C++ command-event CSV.
        self.update_command_start()

        if (
            self.recording_end_wall_time_ns is not None
            and time.time_ns()
            >= self.recording_end_wall_time_ns
        ):
            self.finish_recording()
            return

        
        # --------------------------------------------------

        ##-- (a) Camera -> Object-1
        # --------------------------------------------------

        try:
            t_object_1_raw = self.buffer.lookup_transform(
                self.parent_frame,
                self.raw_object_frame,
                Time(),
                timeout=Duration(seconds=0.0),
            )

             ##-- getting the current timestamp ---#####
            raw_object_stamp_ns = (
                int(t_object_1_raw.header.stamp.sec) * 1_000_000_000
                + int(t_object_1_raw.header.stamp.nanosec)
            )

            # A TF lookup may return the last cached transform even after
            # the raw AprilTag has disappeared. Therefore, availability
            # alone does not mean that a fresh detection was received.
            raw_tag_is_fresh = (
                self.last_processed_object_stamp_ns is None
                or raw_object_stamp_ns
                > self.last_processed_object_stamp_ns
            )

            ##-- pick the common time from the published message timestamp for camera to object-1 -----######
            if raw_tag_is_fresh:
                common_time = Time.from_msg(
                    t_object_1_raw.header.stamp
                )

                stamp_msg = t_object_1_raw.header.stamp

                P_C_to_T1 = (
                    transform_stamped_to_matrix(
                        t_object_1_raw
                    )
                )

                self.last_valid_C_to_T1 = (
                    P_C_to_T1.copy()
                )

                self.have_object_lock = True
                ###--- it means that the tag is visisble -----#####
                self.tag_visible = True

            else:
                # The buffer returned the same cached raw AprilTag pose.
                self.tag_visible = False

        except Exception as error:
            self.tag_visible = False

            ## -- if previous Transform lookup do not have any collected transformations from camera to Object-1 -----####
            if not self.have_object_lock:
                self.get_logger().warn(
                    "Object tag unavailable and no "
                    f"object lock exists: {error}"
                )
                return

        ###-- If the camera to tag-1 lookup did not find any transformation values, then there is no fresh timestamp obtained----##
        ##-- so, will take the present time as the commom time ------#####
        if not self.tag_visible:
            common_time = self.get_clock().now()
            stamp_msg = common_time.to_msg()

      
         ###--- Only if you have "raw_tag_is_fresh" sets to True, then publish the camera to Tag-1 transformation--------####
        if self.tag_visible:
            self.broadcast_transform(
                self.parent_frame,
                self.child_frame,
                P_C_to_T1,
                stamp_msg,
            )

        
        ##-- Using the same timestamp when from camera to tag on robot's head getting detetcted
        
        ### -- (b) Camera -> floor transform ------------####
        try:
            t_floor = self.buffer.lookup_transform(
                self.parent_frame,
                self.floor_tag_frame,
                common_time,
                timeout=Duration(seconds=0.0),
            )

            P_C_to_floor = (
                transform_stamped_to_matrix(t_floor)
            )

            self.last_valid_C_to_floor = (
                P_C_to_floor.copy()
            )
            self.have_floor_lock = True

        except Exception as error:
            if not self.have_floor_lock:
                self.get_logger().warn(
                    "Raw floor tag unavailable and no "
                    f"floor lock exists: {error}"
                )
                return

            P_C_to_floor = (
                self.last_valid_C_to_floor.copy()
            )

        # --------------------------------------------------
        # 2. FLOOR -> CAMERA
        # --------------------------------------------------
        P_floor_to_C = np.linalg.inv(
            P_C_to_floor
        )

        self.broadcast_transform(
            self.floor_frame,
            self.parent_frame,
            P_floor_to_C,
            stamp_msg,
        )

        

        # --------------------------------------------------
        # 4. BASE -> HEAD_UPPER
        # --------------------------------------------------
        P_base_to_Head_Upper = np.eye(
            4,
            dtype=float,
        )

        R_base_to_Head_Upper, _ = quat_to_rot(
            0.0,
            0.0,
            0.0,
            1.0,
        )

        P_base_to_Head_Upper[:3, :3] = (
            R_base_to_Head_Upper
        )

        P_base_to_Head_Upper[:3, 3] = np.array(
            [0.285, 0.0, 0.01],
            dtype=float,
        )

        # --------------------------------------------------
        # 5. HEAD_UPPER -> ROBOT CAMERA Rc
        # --------------------------------------------------
        P_Head_Upper_to_Rc = np.eye(
            4,
            dtype=float,
        )

        R_Head_Upper_to_Rc, _ = quat_to_rot(
            -0.5,
            0.500002,
            -0.5,
            0.499998,
        )

        P_Head_Upper_to_Rc[:3, :3] = (
            R_Head_Upper_to_Rc
        )

        P_Head_Upper_to_Rc[:3, 3] = np.array(
            [0.045, 0.0, 0.03],
            dtype=float,
        )

        # --------------------------------------------------
        # 6. BASE -> Rc AND BASE -> T1
        # --------------------------------------------------
        P_base_to_Rc = (
            P_base_to_Head_Upper
            @ P_Head_Upper_to_Rc
        )

        P_base_to_T1 = (
            P_base_to_Head_Upper
            @ P_Head_Upper_to_Rc
            @ P_star_Rc_to_T1
        )



        # --------------------------------------------------
        # RAW OBJECT TAG UNAVAILABLE:
        # PREDICT CAMERA -> OBJECT_1 FROM ROBOT TF
        # --------------------------------------------------
        ##-- here the camera to tag-1 transformation gets printed from already collected transfornation values --##
        ##-- the frame involved in computation of "P_C_to_T1" are the fixed frames -----#####################
        ###---The tag invisible cases arise from two scenarios:-
        ##--- (a) robot stop moves but the tag remains invisible (because out of field of view of the camera)--------####
        ##--- (b) robot continues to move, but the tag remains invisisble , but recording does not stop------########
        if not self.tag_visible:
            if (
                self.last_valid_C_to_floor is None
                or self.last_valid_floor_to_map is None
            ):
                self.get_logger().warn(
                    "Cannot predict object_1 because no complete "
                    "localization lock exists yet."
                )
                return

            current_time = self.get_clock().now()
            stamp_msg = current_time.to_msg()

            try:
                t_map_to_base = self.buffer.lookup_transform(
                    self.map_frame,
                    self.base_frame,
                    Time(),
                    timeout=Duration(seconds=0.0),
                )

            except Exception as error:
                self.get_logger().warn(
                    f"Cannot predict object_1: {error}"
                )

                self.broadcast_locked_robot_pose(
                    stamp_msg
                )
                return

            P_map_to_base = (
                transform_stamped_to_matrix(
                    t_map_to_base
                )
            )
            ##--- estimating with the help of fixed transforms ----#######
            P_C_to_T1_predicted = (
                self.last_valid_C_to_floor
                @ self.last_valid_floor_to_map
                @ P_map_to_base
                @ P_base_to_T1
            )

            self.last_valid_C_to_T1 = (
                P_C_to_T1_predicted.copy()
            )

            self.broadcast_transform(
                self.floor_frame,
                self.map_frame,
                self.last_valid_floor_to_map,
                stamp_msg,
            )

            self.broadcast_transform(
                self.parent_frame,
                self.child_frame,
                P_C_to_T1_predicted,
                stamp_msg,
            )

            return


        # --------------------------------------------------
        # 7. T1 -> Rc AND T1 -> BASE
        # --------------------------------------------------
        P_T1_to_Rc = np.linalg.inv(
            P_star_Rc_to_T1
        )

        P_T1_to_base = (
            P_T1_to_Rc
            @ np.linalg.inv(P_base_to_Rc)
        )

        # --------------------------------------------------
        # 8. CAMERA -> BASE
        # --------------------------------------------------
        P_C_to_base = (
            P_C_to_T1
            @ P_T1_to_base
        )

        # --------------------------------------------------
        # 9. FLOOR -> BASE
        #
        # This is the AprilTag-measured robot base pose used
        # for recording and movement detection.
        # --------------------------------------------------
        P_floor_to_base = (
            P_floor_to_C
            @ P_C_to_base
        )

       

       

        # --------------------------------------------------
        # 10. MAP -> BASE_LINK FROM EXISTING ROBOT TF TREE
        # --------------------------------------------------
        try:
            t_map_to_base = self.buffer.lookup_transform(
                self.map_frame,
                self.base_frame,
                common_time,
                timeout=Duration(seconds=0.0),
            )

        except Exception as error:
            self.get_logger().warn(
                f"TF {self.map_frame} -> "
                f"{self.base_frame} unavailable: {error}. "
                "Reusing the last valid floor -> map bridge."
            )

            self.broadcast_locked_robot_pose(
                stamp_msg
            )
            return

        P_map_to_base = (
            transform_stamped_to_matrix(
                t_map_to_base
            )
        )

        # --------------------------------------------------
        # 11. FLOOR -> MAP
        #
        # floor_T_base = floor_T_map @ map_T_base
        #
        # floor_T_map =
        #     floor_T_base @ inverse(map_T_base)
        # --------------------------------------------------
        P_floor_to_map = (
            P_floor_to_base
            @ np.linalg.inv(P_map_to_base)
        )

        self.last_valid_floor_to_base = (
            P_floor_to_base.copy()
        )

        self.last_valid_floor_to_map = (
            P_floor_to_map.copy()
        )

        # Publish only the bridge that attaches the existing
        # map -> base_link URDF tree to the floor frame.
        self.broadcast_transform(
            self.floor_frame,
            self.map_frame,
            P_floor_to_map,
            stamp_msg,
        )

       # --------------------------------------------------
        # 13. MARK THIS FRESH APRILTAG POSE AS PROCESSED
        # --------------------------------------------------
        self.last_processed_object_stamp_ns = raw_object_stamp_ns

        sample = self.make_pose_sample(
            P_floor_to_base,
            P_C_to_T1,
            t_object_1_raw.header.stamp,
        )

        self.write_pose_sample(sample)

        self.get_logger().info(
            "[RECORDING] "
            f"object_stamp_ns={raw_object_stamp_ns}, "
            f"wall_time_ns={sample['wall_time_ns']}, "
            f"samples={self.samples_written}"
        )

         # --------------------------------------------------
        # 14. RECORD LATEST ROBOT LEG TRANSFORMS
        #
        # IMPORTANT:
        # The AprilTag/base pose has ALREADY been saved above.
        #
        # Leg transforms therefore cannot prevent a valid
        # base sample from being recorded.
        #
        # Leg TFs use their own actual ROS timestamps and
        # will be synchronized later to Tf_stationary.
        # --------------------------------------------------
        saved_leg_rows = (
            self.collect_current_leg_transforms()
        )

        if saved_leg_rows > 0:
            self.get_logger().info(
                "[LEG RECORDING] "
                f"Saved {saved_leg_rows} new "
                "base_link -> leg-frame transforms."
            )
        




# ============================================================
# MAIN
# ============================================================
def main():
    rclpy.init()
    node = TFLogger()

    try:
        rclpy.spin(node)

    except KeyboardInterrupt:
        node.get_logger().info(
            "Interrupted by user."
        )

    finally:
        node.destroy_node()

        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()



