#!/usr/bin/env python3
"""
Spawn a dynamic obstacle in the Gazebo world to test path replanning.
This script spawns a box obstacle that the robot's sensors will detect.
"""

import sys
import rclpy
from rclpy.node import Node
from gazebo_msgs.srv import SpawnEntity
from geometry_msgs.msg import Pose

class ObstacleSpawner(Node):
    def __init__(self):
        super().__init__('obstacle_spawner')
        self.client = self.create_client(SpawnEntity, '/spawn_entity')
        
        while not self.client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Waiting for /spawn_entity service...')

    def spawn_box(self, name, x, y, z=0.25, size_x=0.3, size_y=0.3, size_z=0.5):
        """
        Spawn a box obstacle at the specified position.
        
        Args:
            name: Unique name for the obstacle
            x, y, z: Position in the world
            size_x, size_y, size_z: Dimensions of the box
        """
        # SDF model string for a simple box
        sdf = f"""
        <?xml version='1.0' ?>
        <sdf version='1.6'>
          <model name='{name}'>
            <static>false</static>
            <link name='link'>
              <collision name='collision'>
                <geometry>
                  <box>
                    <size>{size_x} {size_y} {size_z}</size>
                  </box>
                </geometry>
              </collision>
              <visual name='visual'>
                <geometry>
                  <box>
                    <size>{size_x} {size_y} {size_z}</size>
                  </box>
                </geometry>
                <material>
                  <ambient>1 0 0 1</ambient>
                  <diffuse>1 0 0 1</diffuse>
                </material>
              </visual>
            </link>
          </model>
        </sdf>
        """
        
        # Create the spawn request
        req = SpawnEntity.Request()
        req.name = name
        req.xml = sdf
        req.robot_namespace = ''
        req.initial_pose = Pose()
        req.initial_pose.position.x = x
        req.initial_pose.position.y = y
        req.initial_pose.position.z = z
        req.reference_frame = 'world'
        
        # Call the service
        self.get_logger().info(f'Spawning obstacle "{name}" at ({x}, {y}, {z})')
        future = self.client.call_async(req)
        rclpy.spin_until_future_complete(self, future)
        
        if future.result() is not None:
            self.get_logger().info(f'Successfully spawned obstacle: {future.result().status_message}')
            return True
        else:
            self.get_logger().error(f'Failed to spawn obstacle: {future.exception()}')
            return False


def main(args=None):
    rclpy.init(args=args)
    spawner = ObstacleSpawner()
    
    # Default obstacle position (you can change these)
    # This will spawn an obstacle in the middle of the maze
    obstacle_x = 0.5
    obstacle_y = 0.0
    
    # Check if position provided as command line arguments
    if len(sys.argv) >= 3:
        try:
            obstacle_x = float(sys.argv[1])
            obstacle_y = float(sys.argv[2])
        except ValueError:
            print("Invalid coordinates. Using defaults.")
    
    # Spawn the obstacle
    success = spawner.spawn_box('dynamic_obstacle', obstacle_x, obstacle_y)
    
    if success:
        print(f"\n✓ Obstacle spawned at ({obstacle_x}, {obstacle_y})")
        print("The robot should now detect this obstacle and replan its path!")
        print("\nTo remove this obstacle, run:")
        print(f"  ros2 service call /delete_entity gazebo_msgs/srv/DeleteEntity \"{{name: 'dynamic_obstacle'}}\"")
    
    spawner.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()