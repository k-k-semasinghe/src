#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist
import numpy as np
import time


class EnvironmentDiagnostic(Node):
    """Diagnostic tool to check if environment is learnable"""
    
    def __init__(self):
        super().__init__('env_diagnostic')
        
        self.odom_sub = self.create_subscription(Odometry, '/odom', self.odom_callback, 10)
        self.scan_sub = self.create_subscription(LaserScan, '/scan', self.scan_callback, 10)
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        
        self.current_odom = None
        self.current_scan = None
        self.goal_position = np.array([-0.36, -0.70])
        
        self.get_logger().info('🔍 Environment Diagnostic Tool Started')
        self.get_logger().info('Waiting for sensor data...')
        
    def odom_callback(self, msg):
        self.current_odom = msg
        
    def scan_callback(self, msg):
        self.current_scan = msg
        
    def get_laser_info(self):
        """Analyze laser scan data"""
        if self.current_scan is None:
            return None
            
        ranges = np.array(self.current_scan.ranges)
        valid_ranges = ranges[~np.isnan(ranges) & ~np.isinf(ranges)]
        
        if len(valid_ranges) == 0:
            return None
        
        # Divide into sectors
        n = len(ranges)
        front = ranges[max(0, n-15):] if n > 15 else ranges[:15]
        front = front[~np.isnan(front) & ~np.isinf(front)]
        
        left = ranges[n//4:n//4+30] if n > 60 else ranges[:30]
        left = left[~np.isnan(left) & ~np.isinf(left)]
        
        right = ranges[3*n//4:3*n//4+30] if n > 60 else ranges[-30:]
        right = right[~np.isnan(right) & ~np.isinf(right)]
        
        back = ranges[n//2-15:n//2+15] if n > 30 else ranges[n//3:2*n//3]
        back = back[~np.isnan(back) & ~np.isinf(back)]
        
        return {
            'min': np.min(valid_ranges),
            'max': np.max(valid_ranges),
            'mean': np.mean(valid_ranges),
            'front_min': np.min(front) if len(front) > 0 else float('inf'),
            'left_min': np.min(left) if len(left) > 0 else float('inf'),
            'right_min': np.min(right) if len(right) > 0 else float('inf'),
            'back_min': np.min(back) if len(back) > 0 else float('inf'),
            'num_readings': len(valid_ranges)
        }
        
    def run_diagnostic(self):
        """Run comprehensive diagnostic"""
        # Wait for data
        while self.current_odom is None or self.current_scan is None:
            rclpy.spin_once(self, timeout_sec=0.1)
            time.sleep(0.1)
        
        self.get_logger().info('\n' + '='*60)
        self.get_logger().info('📍 INITIAL STATE DIAGNOSTIC')
        self.get_logger().info('='*60)
        
        # Robot position
        robot_pos = np.array([
            self.current_odom.pose.pose.position.x,
            self.current_odom.pose.pose.position.y
        ])
        
        # Robot orientation
        orientation = self.current_odom.pose.pose.orientation
        robot_yaw = np.arctan2(
            2.0 * (orientation.w * orientation.z + orientation.x * orientation.y),
            1.0 - 2.0 * (orientation.y**2 + orientation.z**2)
        )
        
        self.get_logger().info(f'\n🤖 ROBOT STATE:')
        self.get_logger().info(f'  Position: ({robot_pos[0]:.3f}, {robot_pos[1]:.3f})')
        self.get_logger().info(f'  Orientation: {np.degrees(robot_yaw):.1f}°')
        
        # Goal info
        goal_vec = self.goal_position - robot_pos
        goal_dist = np.linalg.norm(goal_vec)
        goal_angle = np.arctan2(goal_vec[1], goal_vec[0])
        relative_angle = goal_angle - robot_yaw
        relative_angle = np.arctan2(np.sin(relative_angle), np.cos(relative_angle))
        
        self.get_logger().info(f'\n🎯 GOAL:')
        self.get_logger().info(f'  Position: ({self.goal_position[0]:.3f}, {self.goal_position[1]:.3f})')
        self.get_logger().info(f'  Distance: {goal_dist:.3f}m')
        self.get_logger().info(f'  Relative angle: {np.degrees(relative_angle):.1f}°')
        
        # Laser scan info
        laser_info = self.get_laser_info()
        if laser_info:
            self.get_logger().info(f'\n📡 LASER SCAN:')
            self.get_logger().info(f'  Number of readings: {laser_info["num_readings"]}')
            self.get_logger().info(f'  Min distance: {laser_info["min"]:.3f}m')
            self.get_logger().info(f'  Max distance: {laser_info["max"]:.3f}m')
            self.get_logger().info(f'  Mean distance: {laser_info["mean"]:.3f}m')
            self.get_logger().info(f'\n  Sector distances:')
            self.get_logger().info(f'    Front: {laser_info["front_min"]:.3f}m')
            self.get_logger().info(f'    Left:  {laser_info["left_min"]:.3f}m')
            self.get_logger().info(f'    Right: {laser_info["right_min"]:.3f}m')
            self.get_logger().info(f'    Back:  {laser_info["back_min"]:.3f}m')
            
            # Safety assessment
            self.get_logger().info(f'\n⚠️  SAFETY ASSESSMENT:')
            if laser_info["min"] < 0.2:
                self.get_logger().warn(f'  ❌ TOO CLOSE TO OBSTACLE! (min: {laser_info["min"]:.3f}m)')
                self.get_logger().warn(f'  Robot spawns with collision imminent!')
            elif laser_info["min"] < 0.3:
                self.get_logger().warn(f'  ⚠️  Very close to obstacle (min: {laser_info["min"]:.3f}m)')
                self.get_logger().warn(f'  Very little room to maneuver')
            elif laser_info["min"] < 0.5:
                self.get_logger().info(f'  ⚠️  Close to obstacle (min: {laser_info["min"]:.3f}m)')
                self.get_logger().info(f'  Learning will be challenging')
            else:
                self.get_logger().info(f'  ✓ Safe clearance (min: {laser_info["min"]:.3f}m)')
        
        # Path assessment
        self.get_logger().info(f'\n🛣️  PATH ASSESSMENT:')
        if laser_info["front_min"] < 0.3:
            self.get_logger().warn(f'  ❌ Front blocked! Cannot move forward')
        
        if abs(np.degrees(relative_angle)) > 90:
            self.get_logger().info(f'  ⚠️  Goal is behind robot (needs U-turn)')
        elif abs(np.degrees(relative_angle)) > 45:
            self.get_logger().info(f'  ⚠️  Goal requires sharp turn')
        else:
            self.get_logger().info(f'  ✓ Goal roughly ahead')
        
        # Test movements
        self.get_logger().info(f'\n🧪 TESTING BASIC MOVEMENTS...')
        self.test_movements()
        
        self.get_logger().info('\n' + '='*60)
        self.get_logger().info('✓ DIAGNOSTIC COMPLETE')
        self.get_logger().info('='*60 + '\n')
        
    def test_movements(self):
        """Test if robot can move"""
        self.get_logger().info('  Testing forward movement...')
        
        # Store initial position
        initial_pos = np.array([
            self.current_odom.pose.pose.position.x,
            self.current_odom.pose.pose.position.y
        ])
        
        # Try to move forward
        cmd = Twist()
        cmd.linear.x = 0.1
        cmd.angular.z = 0.0
        
        for _ in range(5):
            self.cmd_vel_pub.publish(cmd)
            time.sleep(0.2)
            rclpy.spin_once(self, timeout_sec=0.05)
        
        # Stop
        stop_cmd = Twist()
        self.cmd_vel_pub.publish(stop_cmd)
        time.sleep(0.2)
        rclpy.spin_once(self, timeout_sec=0.05)
        
        # Check movement
        final_pos = np.array([
            self.current_odom.pose.pose.position.x,
            self.current_odom.pose.pose.position.y
        ])
        
        distance_moved = np.linalg.norm(final_pos - initial_pos)
        
        if distance_moved < 0.05:
            self.get_logger().warn(f'  ❌ Robot barely moved! ({distance_moved:.3f}m)')
            self.get_logger().warn(f'  Possible issues: stuck, simulation paused, or immediate collision')
        elif distance_moved < 0.15:
            self.get_logger().warn(f'  ⚠️  Limited movement ({distance_moved:.3f}m)')
        else:
            self.get_logger().info(f'  ✓ Robot can move ({distance_moved:.3f}m)')
        
        # Check for collision
        laser_info = self.get_laser_info()
        if laser_info and laser_info["min"] < 0.18:
            self.get_logger().warn(f'  ❌ Collision after movement! (min: {laser_info["min"]:.3f}m)')


def main(args=None):
    rclpy.init(args=args)
    diagnostic = EnvironmentDiagnostic()
    
    try:
        diagnostic.run_diagnostic()
    except KeyboardInterrupt:
        pass
    finally:
        # Stop robot
        stop_cmd = Twist()
        diagnostic.cmd_vel_pub.publish(stop_cmd)
        diagnostic.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()