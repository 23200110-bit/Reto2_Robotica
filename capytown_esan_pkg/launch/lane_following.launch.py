import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    config_dir = os.path.join(
        get_package_share_directory('capytown_esan_pkg'),
        'config'
    )
    
    detector_node = Node(
        package='capytown_esan_pkg',
        executable='lane_detector',
        name='lane_detector',
        parameters=[{
            'hsv_params_path': os.path.join(config_dir, 'hsv_params.yaml'),
            'ipm_params_path': os.path.join(config_dir, 'ipm_config.json'),
        }],
        output='screen'
    )
    
    controller_node = Node(
        package='capytown_esan_pkg',
        executable='lane_controller',
        name='lane_controller',
        output='screen'
    )
    
    return LaunchDescription([
        detector_node,
        controller_node
    ])
