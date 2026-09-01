#!/usr/bin/env python3
"""
CORRECT WAY TO ACCESS DEQUE DATA
================================

Run this in a SEPARATE terminal while demo_base_log.py is running.
This script will access the deque data from the running TFLogger node.
"""

import argparse
import os
import sys

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
import time
import csv
import pickle
import numpy as np

# Ensure this script imports the local package, not an installed version.
# When running as a script (python3 access_deque_data.py), Python adds the
# script's directory to sys.path, which can prevent importing the actual
# package root (src/my_tf_logger).
_this_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(_this_dir)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from my_tf_logger.demo_base_log import TFLogger


class DequeDataAccess:
    """Access deque data by running the TFLogger node in this process"""
    
    def __init__(self):
        self.node = None
        self.data_collected = False
    
    def start_collection(self, duration_seconds=10, stream_object1=False, watch_n=0):
        """Start collecting data.

        Args:
            duration_seconds: How long to collect data (seconds). If <= 0, runs until
                the user stops it with Ctrl+C.
            stream_object1: If True, prints each new camera->object_1 frame with distance.
            watch_n: If >0, prints the latest `watch_n` camera->object_1 frames once at the end.
        """
        rclpy.init()
        self.node = TFLogger()

        print("\n" + "="*70)
        print("🎯 DEQUE DATA COLLECTION - STARTING")
        print("="*70)

        if duration_seconds and duration_seconds > 0:
            print(f"\n⏱️  Collecting data for {duration_seconds} seconds...")
        else:
            print("\n⏱️  Collecting data until stopped (Ctrl+C)...")
        print("Make sure your TF frames are being published!\n")

        start_time = time.time()
        prev_len = 0

        try:
            while True:
                rclpy.spin_once(self.node, timeout_sec=0.1)
                elapsed = time.time() - start_time
                print(f"\r⏳ Elapsed: {elapsed:.1f}s", end="", flush=True)

                if stream_object1:
                    data = self.node.get_deque_data('camera_to_object1')
                    if data is not None and len(data) > prev_len:
                        for entry in data[prev_len:]:
                            dist = np.linalg.norm(entry['translation'])
                            print(
                                f"\n[{entry['timestamp']}] dist={dist:.6f} m | "
                                f"pos={entry['translation']} | rot={entry['rotation_rpy']}"
                            )
                        prev_len = len(data)

                if duration_seconds and duration_seconds > 0 and elapsed >= duration_seconds:
                    break
        except KeyboardInterrupt:
            print("\n\n⏹️  Stopped by user")

        print("\n\n✅ Data collection complete!\n")
        self.data_collected = True

        if watch_n and watch_n > 0:
            self.watch_object1(watch_n)
    
    def print_summaries(self):
        """Print summaries of all deques"""
        if not self.node:
            print("❌ No data available. Run start_collection() first.")
            return
        
        print("="*70)
        print("📊 DEQUE SUMMARIES")
        print("="*70)
        
        for deque_name in ['camera_to_object1', 'camera_to_object2', 
                          'camera_to_object3', 'camera_to_baselink']:
            self.node.print_deque_summary(deque_name)
            print()
    
    def get_latest_readings(self):
        """Get the latest readings from all deques."""
        if not self.node:
            print("❌ No data available. Run start_collection() first.")
            return

        print("\n" + "="*70)
        print("📍 LATEST READINGS")
        print("="*70 + "\n")

        deques = {
            'camera_to_object1': 'Object-1 (Tag-T1)',
            'camera_to_object2': 'Object-2 (Obstacle Foam)',
            'camera_to_object3': 'Object-3 (Goal Foam)',
            'camera_to_baselink': 'Base-Link'
        }

        for deque_name, label in deques.items():
            data = self.node.get_deque_data(deque_name)
            if data and len(data) > 0:
                latest = data[-1]
                print(f"🎯 {label}")
                print(f"   Timestamp: {latest['timestamp']}")
                print(f"   Position [x,y,z]: {latest['translation']}")
                print(f"   Distance from camera: {np.linalg.norm(latest['translation']):.6f} m")
                print(f"   Rotation [r,p,y]: {latest['rotation_rpy']}")
                print(f"   Translation Error: {latest['translation_error']:.6f}")
                print(f"   Rotation Error: {latest['rotation_error']:.6f}")
                print(f"   RPY Error: {latest['rpy_error']:.6f}")
                print()
            else:
                print(f"⚠️  {label}: No data collected")
                print()

    def watch_object1(self, n=100):
        """Print the last N camera->object_1 entries (distance + timestamp)."""
        if not self.node:
            print("❌ No data available. Run start_collection() first.")
            return

        data = self.node.get_deque_data('camera_to_object1')
        if not data:
            print("⚠️  No camera->object_1 data collected yet.")
            return

        print("\n" + "="*70)
        print(f"👁️  WATCH last {min(n, len(data))} camera->object_1 frames")
        print("="*70 + "\n")

        for entry in data[-n:]:
            dist = np.linalg.norm(entry['translation'])
            print(f"{entry['timestamp']} | dist={dist:.6f} m | pos={entry['translation']} | rot={entry['rotation_rpy']}")
    
    def export_to_csv(self, output_dir="."):
        """Export deque data to CSV files"""
        if not self.node:
            print("❌ No data available. Run start_collection() first.")
            return
        
        print("\n" + "="*70)
        print("💾 EXPORTING TO CSV FILES")
        print("="*70 + "\n")
        
        deques = ['camera_to_object1', 'camera_to_object2', 
                 'camera_to_object3', 'camera_to_baselink']
        
        for deque_name in deques:
            data = self.node.get_deque_data(deque_name)
            if not data:
                print(f"⚠️  {deque_name}: No data to export")
                continue
            
            filename = f"{output_dir}/{deque_name}_data.csv"
            
            try:
                with open(filename, 'w', newline='') as csvfile:
                    fieldnames = ['timestamp', 'x', 'y', 'z', 'roll', 'pitch', 'yaw',
                                 'trans_error', 'rot_error', 'rpy_error']
                    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                    
                    writer.writeheader()
                    for entry in data:
                        writer.writerow({
                            'timestamp': entry['timestamp'],
                            'x': entry['translation'][0],
                            'y': entry['translation'][1],
                            'z': entry['translation'][2],
                            'roll': entry['rotation_rpy'][0],
                            'pitch': entry['rotation_rpy'][1],
                            'yaw': entry['rotation_rpy'][2],
                            'trans_error': entry['translation_error'],
                            'rot_error': entry['rotation_error'],
                            'rpy_error': entry['rpy_error']
                        })
                
                print(f"✅ Exported {len(data)} entries to: {filename}")
            
            except Exception as e:
                print(f"❌ Error exporting {deque_name}: {e}")
        
        print()
    
    def export_to_pickle(self, output_dir="."):
        """Export deque data to Pickle files"""
        if not self.node:
            print("❌ No data available. Run start_collection() first.")
            return
        
        print("="*70)
        print("💾 EXPORTING TO PICKLE FILES")
        print("="*70 + "\n")
        
        deques = ['camera_to_object1', 'camera_to_object2', 
                 'camera_to_object3', 'camera_to_baselink']
        
        for deque_name in deques:
            data = self.node.get_deque_data(deque_name)
            if not data:
                print(f"⚠️  {deque_name}: No data to export")
                continue
            
            filename = f"{output_dir}/{deque_name}_data.pkl"
            
            try:
                with open(filename, 'wb') as f:
                    pickle.dump(data, f)
                
                print(f"✅ Exported {len(data)} entries to: {filename}")
            
            except Exception as e:
                print(f"❌ Error exporting {deque_name}: {e}")
        
        print()
    
    def analyze_errors(self):
        """Print error statistics"""
        if not self.node:
            print("❌ No data available. Run start_collection() first.")
            return
        
        print("\n" + "="*70)
        print("📈 ERROR STATISTICS")
        print("="*70 + "\n")
        
        deques = ['camera_to_object1', 'camera_to_object2', 
                 'camera_to_object3', 'camera_to_baselink']
        
        for deque_name in deques:
            data = self.node.get_deque_data(deque_name)
            if not data or len(data) < 2:
                print(f"⚠️  {deque_name}: Not enough data for analysis")
                continue
            
            # Extract errors (skip None values)
            trans_errors = [e['translation_error'] for e in data 
                           if e['translation_error'] is not None]
            rot_errors = [e['rotation_error'] for e in data 
                         if e['rotation_error'] is not None]
            rpy_errors = [e['rpy_error'] for e in data 
                         if e['rpy_error'] is not None]
            
            print(f"📍 {deque_name}")
            
            if trans_errors:
                print(f"   Translation Error:")
                print(f"      Mean: {np.mean(trans_errors):.6f} m")
                print(f"      Std:  {np.std(trans_errors):.6f} m")
                print(f"      Min:  {np.min(trans_errors):.6f} m")
                print(f"      Max:  {np.max(trans_errors):.6f} m")
            
            if rot_errors:
                print(f"   Rotation Error:")
                print(f"      Mean: {np.mean(rot_errors):.6f}")
                print(f"      Std:  {np.std(rot_errors):.6f}")
                print(f"      Min:  {np.min(rot_errors):.6f}")
                print(f"      Max:  {np.max(rot_errors):.6f}")
            
            if rpy_errors:
                print(f"   RPY Error:")
                print(f"      Mean: {np.mean(rpy_errors):.6f} rad")
                print(f"      Std:  {np.std(rpy_errors):.6f} rad")
                print(f"      Min:  {np.min(rpy_errors):.6f} rad")
                print(f"      Max:  {np.max(rpy_errors):.6f} rad")
            
            print()
    
    def cleanup(self):
        """Clean up resources"""
        if self.node:
            self.node.destroy_node()
        rclpy.shutdown()


def main():
    """Run the data accessor.

    USAGE EXAMPLE:
        Terminal 1: ros2 run my_tf_logger demo_base_log --ros-args -p parent_frame:=camera ...
        Terminal 2: python3 access_deque_data.py --duration 0

    When --duration is 0 or negative, this script will run until interrupted (Ctrl+C).
    """

    parser = argparse.ArgumentParser(description="Access deque data collected by TFLogger")
    parser.add_argument(
        "--duration",
        type=float,
        default=10.0,
        help="How many seconds to collect data (<=0 means run until Ctrl+C).",
    )
    parser.add_argument(
        "--stream",
        action="store_true",
        help="Continuously print camera->object_1 frames with distance (live stream).",
    )
    parser.add_argument(
        "--watch",
        type=int,
        default=0,
        help="After collection, print the last N camera->object_1 frames (distance + pose).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=".",
        help="Directory to write CSV/Pickle output files into.",
    )
    args = parser.parse_args()

    print("\n" + "="*70)
    print("🚀 DEQUE DATA ACCESS SCRIPT")
    print("="*70)
    print("\nThis script collects and accesses deque data from TFLogger")

    accessor = DequeDataAccess()

    try:
        accessor.start_collection(
            duration_seconds=args.duration,
            stream_object1=args.stream,
            watch_n=args.watch,
        )

        # Print summaries
        accessor.print_summaries()

        # Get latest readings (includes camera->object_1 distance)
        accessor.get_latest_readings()

        # Analyze errors
        accessor.analyze_errors()

        # Export data
        accessor.export_to_csv(output_dir=args.output_dir)
        accessor.export_to_pickle(output_dir=args.output_dir)

        print("\n" + "="*70)
        print("✅ COMPLETE!")
        print("="*70 + "\n")

    except KeyboardInterrupt:
        print("\n\n⏹️  Stopped by user")

    finally:
        accessor.cleanup()


if __name__ == '__main__':
    main()
