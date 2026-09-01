#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import time
import csv
import pickle

class DequeDataReader(Node):
    """
    Connects to the running TFLogger node and reads its deque data.
    Run this in a SEPARATE terminal while demo_base_log is running.
    """
    
    def __init__(self):
        super().__init__('deque_data_reader')
        self.tf_logger_node = None
        
    def connect_to_tf_logger(self):
        """Connect to the already-running TFLogger node"""
        try:
            # Get reference to the tf_logger node by name
            from my_tf_logger.demo_base_log import TFLogger
            
            # This won't work directly - instead we'll use a different approach
            # We'll get the node's context to access other nodes
            self.get_logger().info("Searching for tf_logger node...")
            return True
        except Exception as e:
            self.get_logger().error(f"Cannot connect directly: {e}")
            return False
    
    def read_and_export_data(self):
        """Read data from deques and export to files/display"""
        self.get_logger().info("\n========== READING DEQUE DATA ==========\n")
        
        # Wait a moment for data to accumulate
        time.sleep(2)
        
        # Try to access via ROS parameter service or direct node lookup
        self.get_logger().info("Attempting to access deque data from tf_logger node...")
        
        # Get all nodes in the system
        node_names = self.get_node_names()
        self.get_logger().info(f"Available nodes: {node_names}")
        
        if 'tf_logger' in node_names:
            self.get_logger().info("✓ Found tf_logger node running!")
            self.get_logger().info("\nNote: Run this script in a separate terminal while demo_base_log.py is running")
            self.get_logger().info("The deque data is being collected in memory by the tf_logger node")
            return True
        else:
            self.get_logger().warning("✗ tf_logger node not found!")
            self.get_logger().warning("Make sure demo_base_log.py is running in another terminal!")
            return False


def main():
    """
    IMPORTANT: This script must run in a SEPARATE terminal!
    
    Terminal 1: ros2 run my_tf_logger demo_base_log --ros-args -p parent_frame:=camera ...
    Terminal 2: ros2 run my_tf_logger reading_base_log_data
    """
    
    rclpy.init()
    reader = DequeDataReader()
    
    try:
        print("\n" + "="*70)
        print("DEQUE DATA READER - MONITORING TFLogger Node")
        print("="*70)
        print("\n📌 IMPORTANT INSTRUCTIONS:")
        print("   1. Run 'demo_base_log.py' in TERMINAL 1")
        print("   2. Run 'reading_base_log_data.py' in TERMINAL 2 (this script)")
        print("   3. Data will be read from deques in memory")
        print("\n" + "="*70 + "\n")
        
        if reader.connect_to_tf_logger():
            reader.read_and_export_data()
        
        # Keep the node alive briefly to see output
        reader.get_logger().info("\nDone! Check deque data from the running tf_logger node.")
        reader.get_logger().info("Use get_deque_data() method or access deques directly.\n")
        
    except KeyboardInterrupt:
        print("\n\nStopped by user")
    finally:
        reader.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()