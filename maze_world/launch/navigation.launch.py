import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

def generate_launch_description():
    # Get package directories
    pkg_maze_world = get_package_share_directory('maze_world')
    pkg_nav2_bringup = get_package_share_directory('nav2_bringup')
    
    # Paths
    map_file = os.path.join(pkg_maze_world, 'maps', 'maze_map.yaml')
    nav2_params = os.path.join(pkg_maze_world, 'config', 'nav2_params.yaml')
    
    # Launch configuration
    use_sim_time = LaunchConfiguration('use_sim_time', default='true')
    autostart = LaunchConfiguration('autostart', default='true')
    
    # Map server
    map_server_cmd = Node(
        package='nav2_map_server',
        executable='map_server',
        name='map_server',
        output='screen',
        parameters=[{
            'use_sim_time': use_sim_time,
            'yaml_filename': map_file
        }]
    )
    
    # AMCL (localization)
    amcl_cmd = Node(
        package='nav2_amcl',
        executable='amcl',
        name='amcl',
        output='screen',
        parameters=[{
            'use_sim_time': use_sim_time,
            
            # INCREASED particle counts for better tracking
            'max_particles': 5000,  # From 2000
            'min_particles': 1000,  # From 500
            
            # Motion model - be more conservative about motion noise
            'alpha1': 0.05,  # REDUCED from 0.2 - less noise on rotation from translation
            'alpha2': 0.05,  # REDUCED from 0.2 - less noise on rotation from rotation
            'alpha3': 0.05,  # REDUCED from 0.2 - less noise on translation from translation
            'alpha4': 0.05,  # REDUCED from 0.2 - less noise on translation from rotation
            'alpha5': 0.05,  # REDUCED from 0.2
            
            'base_frame_id': 'base_footprint',
            'beam_skip_distance': 0.5,
            'beam_skip_error_threshold': 0.9,
            'beam_skip_threshold': 0.3,
            'do_beamskip': False,
            'global_frame_id': 'map',
            'lambda_short': 0.1,
            'laser_likelihood_max_dist': 2.0,
            'laser_max_range': 3.5,
            'laser_min_range': 0.12,
            'laser_model_type': 'likelihood_field',
            'max_beams': 60,
            'odom_frame_id': 'odom',
            
            # Particle filter parameters
            'pf_err': 0.05,
            'pf_z': 0.99,
            
            # ENABLE recovery from bad localization
            'recovery_alpha_fast': 0.1,  # CHANGED from 0.0
            'recovery_alpha_slow': 0.001,  # CHANGED from 0.0
            
            'resample_interval': 2,  # INCREASED from 1
            'robot_model_type': 'nav2_amcl::DifferentialMotionModel',
            'save_pose_rate': 0.5,
            'sigma_hit': 0.2,
            'tf_broadcast': True,
            
            # INCREASED transform tolerance - critical!
            'transform_tolerance': 2.0,  # INCREASED from 1.0
            
            # Update more frequently
            'update_min_a': 0.1,  # REDUCED from 0.2
            'update_min_d': 0.1,  # REDUCED from 0.25
            
            'z_hit': 0.5,
            'z_max': 0.05,
            'z_rand': 0.5,
            'z_short': 0.05,
            'scan_topic': 'scan',
            'set_initial_pose': True,
            'initial_pose': {
                'x': 0.3,
                'y': 0.0,
                'z': 0.0,
                'yaw': 0.0
            }
        }]
    )
    
    # Lifecycle manager for map_server and amcl
    lifecycle_manager_localization_cmd = Node(
        package='nav2_lifecycle_manager',
        executable='lifecycle_manager',
        name='lifecycle_manager_localization',
        output='screen',
        parameters=[{
            'use_sim_time': use_sim_time,
            'autostart': autostart,
            'node_names': ['map_server', 'amcl']
        }]
    )
    
    # Nav2 bringup (planner, controller, behaviors)
    nav2_bringup_cmd = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_nav2_bringup, 'launch', 'navigation_launch.py')
        ),
        launch_arguments={
            'use_sim_time': use_sim_time,
            'autostart': autostart,
            'params_file': nav2_params
        }.items()
    )
    
    ld = LaunchDescription()
    
    # Add declarations
    ld.add_action(DeclareLaunchArgument('use_sim_time', default_value='true'))
    ld.add_action(DeclareLaunchArgument('autostart', default_value='true'))
    
    # Add nodes
    ld.add_action(map_server_cmd)
    ld.add_action(amcl_cmd)
    ld.add_action(lifecycle_manager_localization_cmd)
    ld.add_action(nav2_bringup_cmd)
    
    return ld