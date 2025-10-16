from launch import LaunchDescription
from launch.actions import SetEnvironmentVariable
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
import os

def generate_launch_description():
    use_sim_time = True
    
    pkg_slam = FindPackageShare('slam_toolbox').find('slam_toolbox')
    slam_config = os.path.join(pkg_slam, 'config', 'mapper_params_online_async.yaml')
    
    return LaunchDescription([
        SetEnvironmentVariable(name='TURTLEBOT3_MODEL', value='burger'),
        
        Node(
            package='slam_toolbox',
            executable='async_slam_toolbox_node',
            name='slam_toolbox',
            output='screen',
            parameters=[
                {'use_sim_time': use_sim_time},
                {'slam_mode': 'mapping'},
                {'map_update_interval': 5.0},
                {'resolution': 0.05},
                {'max_laser_range': 3.5},
                {'minimum_travel_distance': 0.5},
                {'minimum_travel_heading': 0.5},
            ]
        ),
    ])