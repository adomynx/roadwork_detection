import os
from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    pkg_dir = get_package_share_directory('roadwork_detection')

    config_file = os.path.join(pkg_dir, 'config', 'detection_params.yaml')
    model_path = os.path.join(pkg_dir, 'models', 'yolov8x_roadwork.engine')

    venv_path = os.path.expanduser('~/roadwork_project/venv/lib/python3.10/site-packages')
    python_env = {'PYTHONPATH': venv_path + ':' + os.environ.get('PYTHONPATH', '')}

    return LaunchDescription([
        Node(
            package='roadwork_detection',
            executable='detector_node',
            name='roadwork_detector_node',
            output='screen',
            parameters=[
                config_file,
                {'model_path': model_path}
            ],
            additional_env=python_env
        ),
        Node(
            package='roadwork_detection',
            executable='lidar_fusion_node',
            name='lidar_fusion_node',
            output='screen',
            parameters=[config_file],
            additional_env=python_env
        ),
        Node(
            package='roadwork_detection',
            executable='confidence_node',
            name='confidence_node',
            output='screen',
            parameters=[config_file],
            additional_env=python_env
        ),
        Node(
            package='roadwork_detection',
            executable='risk_node',
            name='risk_node',
            output='screen',
            parameters=[config_file],
            additional_env=python_env
        ),
    ])
