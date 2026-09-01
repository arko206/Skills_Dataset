#!/usr/bin/env python3
import time
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
    Replays Camera→Base transformations from a saved log file and visualizes them in RViz.
    """
    def __init__(self):
        super().__init__('tf_replay_camera_to_base')

        # Change this path to your recorded log
        self.filepath = '/home/arka/Desktop/Go2_movement_collection/Front_Jump/Straight_Front_Jump/Front_Jump_10.txt'

        # Publishers
        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)
        self.marker_pub = self.create_publisher(Marker, 'camera_to_base_replay_markers', 10)

        # Load all transformation matrices from the file
        self.transforms = self.load_transforms(self.filepath)
        self.idx = 0

        # Timer for playback (10 Hz)
        self.timer = self.create_timer(0.1, self.tick)

        self.get_logger().info(f"Loaded {len(self.transforms)} transforms from {self.filepath}")

    def load_transforms(self, filepath):
        """
        Extracts all 4x4 transformation matrices from lines after
        'Transformation matrix from Camera to Base frame:'.
        """
        transforms = []
        with open(filepath, 'r') as f:
            lines = f.readlines()

        for i, line in enumerate(lines):
            if "Transformation matrix from Camera to Base frame:" in line:
                try:
                    mat_lines = lines[i + 1:i + 5]
                    mat = np.array([[float(x) for x in l.split()] for l in mat_lines])
                    if mat.shape == (4, 4):
                        transforms.append(mat)
                except Exception as e:
                    self.get_logger().warn(f"Error parsing matrix at line {i}: {e}")
        return transforms

    def tick(self):
        if not self.transforms:
            self.get_logger().warn("No transforms loaded. Nothing to replay.")
            return

        # Loop through the loaded transforms
        T = self.transforms[self.idx % len(self.transforms)]
        self.idx += 1

        # Extract translation and rotation
        translation = T[:3, 3]
        quat = R.from_matrix(T[:3, :3]).as_quat()  # x, y, z, w

        # === Broadcast TF (camera -> base_link) ===
        msg = TransformStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'camera'
        msg.child_frame_id = 'base_link'
        msg.transform.translation.x = translation[0]
        msg.transform.translation.y = translation[1]
        msg.transform.translation.z = translation[2]
        msg.transform.rotation.x = quat[0]
        msg.transform.rotation.y = quat[1]
        msg.transform.rotation.z = quat[2]
        msg.transform.rotation.w = quat[3]
        self.tf_broadcaster.sendTransform(msg)

        # === Visualization Marker ===
        self.publish_marker(translation, quat)

    def publish_marker(self, t, q):
        """Publishes both a sphere (current position) and a trajectory line."""
        # --- Sphere marker (current base position) ---
        marker = Marker()
        marker.header.frame_id = 'camera'
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "camera_to_base_replay"
        marker.id = self.idx
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        marker.pose.position.x = t[0]
        marker.pose.position.y = t[1]
        marker.pose.position.z = t[2]
        marker.pose.orientation.x = q[0]
        marker.pose.orientation.y = q[1]
        marker.pose.orientation.z = q[2]
        marker.pose.orientation.w = q[3]
        marker.scale.x = 0.03
        marker.scale.y = 0.03
        marker.scale.z = 0.03
        marker.color = ColorRGBA(r=1.0, g=0.3, b=0.1, a=1.0)
        self.marker_pub.publish(marker)

        # --- Line marker (trajectory path) ---
        if not hasattr(self, 'path_marker'):
            self.path_marker = Marker()
            self.path_marker.header.frame_id = 'camera'
            self.path_marker.ns = "camera_to_base_path"
            self.path_marker.id = 999
            self.path_marker.type = Marker.LINE_STRIP
            self.path_marker.scale.x = 0.01
            self.path_marker.color = ColorRGBA(r=0.1, g=0.8, b=0.1, a=0.8)
            self.path_marker.points = []

        # Add current point
        pt = Point(x=t[0], y=t[1], z=t[2])
        self.path_marker.points.append(pt)

        # Keep the last 500 points
        if len(self.path_marker.points) > 500:
            self.path_marker.points.pop(0)

        self.path_marker.header.stamp = self.get_clock().now().to_msg()
        self.marker_pub.publish(self.path_marker)


def main():
    rclpy.init()
    node = TFReplay()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
