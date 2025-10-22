
#!/usr/bin/env python3
"""
Remove a spawned obstacle from the Gazebo world.
"""

import sys
import rclpy
from rclpy.node import Node
from gazebo_msgs.srv import DeleteEntity

class ObstacleRemover(Node):
    def __init__(self):
        super().__init__('obstacle_remover')
        self.client = self.create_client(DeleteEntity, '/delete_entity')
        
        while not self.client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Waiting for /delete_entity service...')

    def remove_entity(self, name):
        """Remove an entity from Gazebo by name."""
        req = DeleteEntity.Request()
        req.name = name
        
        self.get_logger().info(f'Removing entity: {name}')
        future = self.client.call_async(req)
        rclpy.spin_until_future_complete(self, future)
        
        if future.result() is not None:
            if future.result().success:
                self.get_logger().info(f'Successfully removed: {name}')
                return True
            else:
                self.get_logger().error(f'Failed to remove: {future.result().status_message}')
                return False
        else:
            self.get_logger().error(f'Service call failed: {future.exception()}')
            return False


def main(args=None):
    rclpy.init(args=args)
    remover = ObstacleRemover()
    
    # Default obstacle name
    obstacle_name = 'dynamic_obstacle'
    
    # Check if name provided as command line argument
    if len(sys.argv) >= 2:
        obstacle_name = sys.argv[1]
    
    # Remove the obstacle
    success = remover.remove_entity(obstacle_name)
    
    if success:
        print(f"\n✓ Obstacle '{obstacle_name}' removed successfully!")
    else:
        print(f"\n✗ Failed to remove obstacle '{obstacle_name}'")
    
    remover.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()