import os
from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    pkg_dir = get_package_share_directory('risk_assessment')
    config_file = os.path.join(pkg_dir, 'config', 'risk_params.yaml')

    return LaunchDescription([
        Node(
            package='risk_assessment',
            executable='risk_node',
            name='risk_node',
            output='screen',
            parameters=[config_file],
        ),
    ])
