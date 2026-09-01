#!/usr/bin/env python3
"""
Utility script to interact with TFLogger deques.
Can be used to access and export transformation data from the running TFLogger node.
"""

import pickle
import json
import csv
import numpy as np
import rclpy
from rclpy.node import Node
from my_tf_logger.demo_base_log import TFLogger


class DequeExporter:
    """Utility class to export deque data to various formats."""
    
    @staticmethod
    def export_to_pickle(deque_data, filename):
        """Export deque data to pickle file."""
        with open(filename, 'wb') as f:
            pickle.dump(deque_data, f)
        print(f"Exported to {filename}")
    
    @staticmethod
    def export_to_csv(deque_data, filename):
        """Export deque data to CSV file."""
        if not deque_data:
            print("No data to export")
            return
        
        # Prepare CSV headers
        headers = ['timestamp', 'trans_x', 'trans_y', 'trans_z', 
                   'rot_r', 'rot_p', 'rot_y',
                   'trans_error', 'rot_error', 'rpy_error']
        
        with open(filename, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writeheader()
            
            for entry in deque_data:
                row = {
                    'timestamp': entry['timestamp'],
                    'trans_x': entry['translation'][0],
                    'trans_y': entry['translation'][1],
                    'trans_z': entry['translation'][2],
                    'rot_r': entry['rotation_rpy'][0],
                    'rot_p': entry['rotation_rpy'][1],
                    'rot_y': entry['rotation_rpy'][2],
                    'trans_error': entry['translation_error'],
                    'rot_error': entry['rotation_error'],
                    'rpy_error': entry['rpy_error']
                }
                writer.writerow(row)
        
        print(f"Exported to {filename}")
    
    @staticmethod
    def export_to_json(deque_data, filename):
        """Export deque data to JSON file (matrices as lists)."""
        if not deque_data:
            print("No data to export")
            return
        
        data_for_json = []
        for entry in deque_data:
            json_entry = {
                'timestamp': entry['timestamp'],
                'transform_matrix': entry['transform_matrix'].tolist(),
                'translation': entry['translation'],
                'rotation_rpy': [float(x) for x in entry['rotation_rpy']],
                'translation_error': float(entry['translation_error']) if entry['translation_error'] is not None else None,
                'rotation_error': float(entry['rotation_error']) if entry['rotation_error'] is not None else None,
                'rpy_error': float(entry['rpy_error']) if entry['rpy_error'] is not None else None
            }
            data_for_json.append(json_entry)
        
        with open(filename, 'w') as f:
            json.dump(data_for_json, f, indent=2)
        
        print(f"Exported to {filename}")
    
    @staticmethod
    def print_statistics(deque_data, deque_name):
        """Print statistical summary of deque data."""
        if not deque_data:
            print(f"No data in {deque_name}")
            return
        
        print(f"\n========== STATISTICS FOR {deque_name} ==========")
        print(f"Total entries: {len(deque_data)}")
        
        # Extract all errors (excluding None values)
        trans_errors = [e['translation_error'] for e in deque_data if e['translation_error'] is not None]
        rot_errors = [e['rotation_error'] for e in deque_data if e['rotation_error'] is not None]
        rpy_errors = [e['rpy_error'] for e in deque_data if e['rpy_error'] is not None]
        
        if trans_errors:
            print(f"\nTranslation Error Statistics:")
            print(f"  Mean: {np.mean(trans_errors):.6f}")
            print(f"  Std:  {np.std(trans_errors):.6f}")
            print(f"  Min:  {np.min(trans_errors):.6f}")
            print(f"  Max:  {np.max(trans_errors):.6f}")
        
        if rot_errors:
            print(f"\nRotation Error Statistics:")
            print(f"  Mean: {np.mean(rot_errors):.6f}")
            print(f"  Std:  {np.std(rot_errors):.6f}")
            print(f"  Min:  {np.min(rot_errors):.6f}")
            print(f"  Max:  {np.max(rot_errors):.6f}")
        
        if rpy_errors:
            print(f"\nRPY Error Statistics:")
            print(f"  Mean: {np.mean(rpy_errors):.6f}")
            print(f"  Std:  {np.std(rpy_errors):.6f}")
            print(f"  Min:  {np.min(rpy_errors):.6f}")
            print(f"  Max:  {np.max(rpy_errors):.6f}")


def main():
    """Example usage of DequeExporter."""
    # Initialize ROS2
    rclpy.init()
    node = TFLogger()
    
    try:
        # Run for a short time to collect data
        print("Running TFLogger for 10 seconds to collect data...")
        # In a real scenario, you would run node.spin() in a separate thread
        # or use rclpy.spin_once() in a loop
        
        # For now, just show how to use the exporter once the node is running
        # This would be called from a separate script or after sufficient runtime
        
        # Example: Export all deques when ready
        for deque_name in ['camera_to_object1', 'camera_to_object2', 'camera_to_object3', 'camera_to_baselink']:
            data = node.get_deque_data(deque_name)
            if data:
                # Export to multiple formats
                exporter = DequeExporter()
                exporter.print_statistics(data, deque_name)
                # exporter.export_to_csv(data, f"{deque_name}_data.csv")
                # exporter.export_to_json(data, f"{deque_name}_data.json")
                # exporter.export_to_pickle(data, f"{deque_name}_data.pkl")
        
        node.print_deque_summary('camera_to_object1')
        node.print_deque_summary('camera_to_object2')
        node.print_deque_summary('camera_to_object3')
        node.print_deque_summary('camera_to_baselink')
        
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
