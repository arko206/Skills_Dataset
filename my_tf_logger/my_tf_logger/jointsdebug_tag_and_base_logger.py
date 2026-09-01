#!/usr/bin/env python3
import numpy as np
import rclpy
from rclpy.node import Node
import tf2_ros
from geometry_msgs.msg import TransformStamped, Point
from visualization_msgs.msg import Marker
from std_msgs.msg import ColorRGBA
from scipy.spatial.transform import Rotation as R


class TFReplay(Node):
    """
    Replays Camera→Base and Camera→Tag-T1 transformations from a saved log file
    and visualizes both trajectories in RViz.
    """
    def __init__(self):
        super().__init__('tf_replay_camera_base_tag')

        # Change this to your log file path
        self.filepath = '/home/arka/Desktop/Go2_movement_collection/Walk/Left_Walk/Left_Vx_Vyaw/left_x_yaw_70.txt'

        # Publishers
        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)
        self.marker_pub = self.create_publisher(Marker, 'camera_replay_markers', 10)

        # Load both transforms from log
        (self.tag_transforms,
            self.base_transforms,
            self.fl_transforms,
            self.fr_transforms,
            self.rl_transforms,
            self.rr_transforms) = self.load_transforms(self.filepath)
        
        self.idx = 0

        # Timer for playback
        self.timer = self.create_timer(0.1, self.tick)

        self.get_logger().info(
        f"Loaded: tag={len(self.tag_transforms)}, base={len(self.base_transforms)}, "
        f"FL={len(self.fl_transforms)}, FR={len(self.fr_transforms)}, "
        f"RL={len(self.rl_transforms)}, RR={len(self.rr_transforms)}"
    )


    def load_transforms(self, filepath):
        tag_Ts, base_Ts = [], []
        fl_Ts, fr_Ts, rl_Ts, rr_Ts = [], [], [], []

        with open(filepath, 'r') as f:
            lines = f.readlines()

        i = 0
        while i < len(lines):
            line = lines[i]

            # --- Camera → Tag ---
            if 'camera -> object_1' in line:
                tag_mat = np.array([[float(x) for x in lines[i + j + 1].split()] for j in range(4)])
                tag_Ts.append(tag_mat)
                i += 5

            # --- Camera → Base ---
            elif 'Transformation matrix from Camera to Base frame:' in line:
                base_mat = np.array([[float(x) for x in lines[i + j + 1].split()] for j in range(4)])
                base_Ts.append(base_mat)
                i += 5

            # --- Camera → FR calf ---
            elif 'Transformation matrix from Camera to Front Right Calf Joint:' in line:
                fr_mat = np.array([[float(x) for x in lines[i + j + 1].split()] for j in range(4)])
                fr_Ts.append(fr_mat)
                i += 5

            # --- Camera → FL calf ---
            elif 'Transformation matrix from Camera to Front Left Calf Joint:' in line:
                fl_mat = np.array([[float(x) for x in lines[i + j + 1].split()] for j in range(4)])
                fl_Ts.append(fl_mat)
                i += 5

            # --- Camera → RR calf ---
            elif 'Transformation matrix from Camera to Rear Right Calf Joint:' in line:
                rr_mat = np.array([[float(x) for x in lines[i + j + 1].split()] for j in range(4)])
                rr_Ts.append(rr_mat)
                i += 5

            # --- Camera → RL calf ---
            elif 'Transformation matrix from Camera to Rear Left Calf Joint:' in line:
                rl_mat = np.array([[float(x) for x in lines[i + j + 1].split()] for j in range(4)])
                rl_Ts.append(rl_mat)
                i += 5

            else:
                i += 1

        return tag_Ts, base_Ts, fl_Ts, fr_Ts, rl_Ts, rr_Ts

    def tick(self):
        if (not self.base_transforms or not self.tag_transforms or
            not self.fl_transforms or not self.fr_transforms or
            not self.rl_transforms or not self.rr_transforms):
            self.get_logger().warn("Missing one or more transform streams in file. Check parsing / file content.")
            return

        # Loop through synchronized transforms
        min_len = min(
            len(self.tag_transforms),
            len(self.base_transforms),
            len(self.fl_transforms),
            len(self.fr_transforms),
            len(self.rl_transforms),
            len(self.rr_transforms)
        )

        idx = self.idx % min_len
        self.idx += 1

        T_tag = self.tag_transforms[idx]
        T_base = self.base_transforms[idx]
        T_fl = self.fl_transforms[idx]
        T_fr = self.fr_transforms[idx]
        T_rl = self.rl_transforms[idx]
        T_rr = self.rr_transforms[idx]

        # Publish both transforms as TF
        self.publish_tf(T_tag, "camera", "Tag_T1")
        self.publish_tf(T_base, "camera", "base_link")
        self.publish_tf(T_fl, "camera", "Front_Left_Calf")
        self.publish_tf(T_fr, "camera", "Front_Right_Calf")
        self.publish_tf(T_rl, "camera", "Rear_Left_Calf")
        self.publish_tf(T_rr, "camera", "Rear_Right_Calf")

        # Publish markers in RViz
        self.publish_markers(T_tag[:3, 3], T_base[:3, 3])

        self.publish_path_marker(T_fl[:3,3], "fl_path", 10, 0.9, 0.2, 0.2)
        self.publish_path_marker(T_fr[:3,3], "fr_path", 11, 0.2, 0.9, 0.2)
        self.publish_path_marker(T_rl[:3,3], "rl_path", 12, 0.2, 0.2, 0.9)
        self.publish_path_marker(T_rr[:3,3], "rr_path", 13, 0.9, 0.9, 0.2)

    def publish_tf(self, T, parent, child):
        """Broadcast a TF frame from parent→child."""
        msg = TransformStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = parent
        msg.child_frame_id = child
        msg.transform.translation.x = T[0, 3]
        msg.transform.translation.y = T[1, 3]
        msg.transform.translation.z = T[2, 3]

        
        quat = R.from_matrix(T[:3, :3]).as_quat()
        msg.transform.rotation.x = quat[0]
        msg.transform.rotation.y = quat[1]
        msg.transform.rotation.z = quat[2]
        msg.transform.rotation.w = quat[3]
        self.tf_broadcaster.sendTransform(msg)


    def publish_path_marker(self, position, namespace, marker_id, r, g, b):
        if not hasattr(self, namespace):
            marker = Marker()
            marker.header.frame_id = 'camera'
            marker.ns = namespace
            marker.id = marker_id
            marker.type = Marker.LINE_STRIP
            marker.scale.x = 0.01
            marker.color = ColorRGBA(r=r, g=g, b=b, a=0.8)
            marker.points = []
            setattr(self, namespace, marker)

        marker = getattr(self, namespace)
        marker.points.append(Point(x=position[0], y=position[1], z=position[2]))

        if len(marker.points) > 500:
            marker.points.pop(0)

        marker.header.stamp = self.get_clock().now().to_msg()
        self.marker_pub.publish(marker)

    def publish_markers(self, t_tag, t_base):
        """Publishes markers for Camera→Base and Camera→Tag-T1 trajectories."""
        # === Tag marker (Blue) ===
        tag_marker = Marker()
        tag_marker.header.frame_id = 'camera'
        tag_marker.header.stamp = self.get_clock().now().to_msg()
        tag_marker.ns = 'camera_to_tag_path'
        tag_marker.id = 1
        tag_marker.type = Marker.SPHERE
        tag_marker.scale.x = tag_marker.scale.y = tag_marker.scale.z = 0.03
        tag_marker.color = ColorRGBA(r=0.1, g=0.3, b=1.0, a=1.0)
        tag_marker.pose.position.x, tag_marker.pose.position.y, tag_marker.pose.position.z = t_tag
        self.marker_pub.publish(tag_marker)

        if not hasattr(self, 'tag_path'):
            self.tag_path = Marker()
            self.tag_path.header.frame_id = 'camera'
            self.tag_path.ns = 'camera_to_tag_path_line'
            self.tag_path.id = 2
            self.tag_path.type = Marker.LINE_STRIP
            self.tag_path.scale.x = 0.01
            self.tag_path.color = ColorRGBA(r=0.2, g=0.9, b=1.0, a=0.8)
            self.tag_path.points = []

        self.tag_path.points.append(Point(x=t_tag[0], y=t_tag[1], z=t_tag[2]))
        if len(self.tag_path.points) > 500:
            self.tag_path.points.pop(0)
        self.tag_path.header.stamp = self.get_clock().now().to_msg()
        self.marker_pub.publish(self.tag_path)

        # === Base marker (Red) ===
        base_marker = Marker()
        base_marker.header.frame_id = 'camera'
        base_marker.header.stamp = self.get_clock().now().to_msg()
        base_marker.ns = 'camera_to_base_path'
        base_marker.id = 3
        base_marker.type = Marker.SPHERE
        base_marker.scale.x = base_marker.scale.y = base_marker.scale.z = 0.03
        base_marker.color = ColorRGBA(r=1.0, g=0.3, b=0.1, a=1.0)
        base_marker.pose.position.x, base_marker.pose.position.y, base_marker.pose.position.z = t_base
        self.marker_pub.publish(base_marker)

        # print("Positions of the base marker with respect to camera are: ", base_marker.pose.position.x, base_marker.pose.position.y, base_marker.pose.position.z)

        if not hasattr(self, 'base_path'):
            self.base_path = Marker()
            self.base_path.header.frame_id = 'camera'
            self.base_path.ns = 'camera_to_base_path_line'
            self.base_path.id = 4
            self.base_path.type = Marker.LINE_STRIP
            self.base_path.scale.x = 0.01
            self.base_path.color = ColorRGBA(r=0.1, g=0.8, b=0.1, a=0.8)
            self.base_path.points = []

        self.base_path.points.append(Point(x=t_base[0], y=t_base[1], z=t_base[2]))
        if len(self.base_path.points) > 500:
            self.base_path.points.pop(0)
        self.base_path.header.stamp = self.get_clock().now().to_msg()
        self.marker_pub.publish(self.base_path)


def main():
    rclpy.init()
    node = TFReplay()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
