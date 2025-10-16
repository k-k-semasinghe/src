# from launch import LaunchDescription
# from launch.actions import IncludeLaunchDescription, SetEnvironmentVariable, TimerAction
# from launch.launch_description_sources import PythonLaunchDescriptionSource
# from launch_ros.substitutions import FindPackageShare
# import os

# def generate_launch_description():
#     # Set TurtleBot3 model
#     set_tb3_model = SetEnvironmentVariable(
#         name='TURTLEBOT3_MODEL', 
#         value='burger'
#     )
    
#     # Get package paths
#     pkg_gazebo_ros = FindPackageShare('gazebo_ros').find('gazebo_ros')
#     pkg_maze_world = FindPackageShare('maze_world').find('maze_world')
#     pkg_turtlebot3_gazebo = FindPackageShare('turtlebot3_gazebo').find('turtlebot3_gazebo')
    
#     # Path to your custom world
#     world_path = os.path.join(pkg_maze_world, 'worlds', 'maze.world')
    
#     # Launch Gazebo with your custom world
#     start_gazebo = IncludeLaunchDescription(
#         PythonLaunchDescriptionSource(
#             os.path.join(pkg_gazebo_ros, 'launch', 'gazebo.launch.py')
#         ),
#         launch_arguments=[
#             ('world', world_path),
#             ('verbose', 'true'),
#             ('pause', 'false'),
#         ]
#     )
    
#     # Spawn TurtleBot3 using the turtlebot3_gazebo spawn script
#     spawn_tb3 = TimerAction(
#         period=3.0,
#         actions=[
#             IncludeLaunchDescription(
#                 PythonLaunchDescriptionSource(
#                     os.path.join(pkg_turtlebot3_gazebo, 'launch', 'spawn_turtlebot3.launch.py')
#                 ),
#                 launch_arguments=[
#                     ('x_pose', '0.0'),
#                     ('y_pose', '0.0'),
#                     ('z_pose', '0.0'),
#                 ]
#             )
#         ]
#     )
    
#     return LaunchDescription([
#         set_tb3_model,
#         start_gazebo,
#         spawn_tb3,
#     ])


import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

def generate_launch_description():
    # Package directories
    pkg_gazebo_ros = get_package_share_directory('gazebo_ros')
    pkg_turtlebot3_gazebo = get_package_share_directory('turtlebot3_gazebo')
    pkg_maze_world = get_package_share_directory('maze_world')
    
    # Paths
    world_file = os.path.join(pkg_maze_world, 'worlds', 'maze.world')
    urdf_file = os.path.join(pkg_turtlebot3_gazebo, 'models', 'turtlebot3_burger', 'model.sdf')
    
    # Launch configuration variables
    use_sim_time = LaunchConfiguration('use_sim_time', default='true')
    x_pose = LaunchConfiguration('x_pose', default='0.0')
    y_pose = LaunchConfiguration('y_pose', default='0.0')
    
    # Gazebo server
    gzserver_cmd = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_gazebo_ros, 'launch', 'gzserver.launch.py')
        ),
        launch_arguments={'world': world_file}.items()
    )
    
    # Gazebo client
    gzclient_cmd = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_gazebo_ros, 'launch', 'gzclient.launch.py')
        )
    )
    
    # Robot State Publisher
    robot_state_publisher_cmd = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_turtlebot3_gazebo, 'launch', 'robot_state_publisher.launch.py')
        ),
        launch_arguments={'use_sim_time': use_sim_time}.items()
    )
    
    # Spawn TurtleBot3
    spawn_turtlebot_cmd = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_turtlebot3_gazebo, 'launch', 'spawn_turtlebot3.launch.py')
        ),
        launch_arguments={
            'x_pose': x_pose,
            'y_pose': y_pose
        }.items()
    )
    
    ld = LaunchDescription()
    
    # Add the commands to the launch description
    ld.add_action(gzserver_cmd)
    ld.add_action(gzclient_cmd)
    ld.add_action(robot_state_publisher_cmd)
    ld.add_action(spawn_turtlebot_cmd)
    
    return ld