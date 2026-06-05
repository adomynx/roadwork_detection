import os
from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    pkg_dir = get_package_share_directory('road_condition')
    model_path = os.path.join(pkg_dir, 'models', 'EfficientNet_training_N.keras')

    venv_path = os.path.expanduser('~/roadwork_project/venv/lib/python3.10/site-packages')
    python_env = {'PYTHONPATH': venv_path + ':' + os.environ.get('PYTHONPATH', '')}

    return LaunchDescription([
        Node(
            package='road_condition',
            executable='road_patch_node',
            name='road_patch_node',
            output='screen',
            parameters=[{
                'patch_output_size': 224,
                'crop_size': 80,
                'camera_fx': 266.9,
                'camera_fy': 266.9,
                'camera_cx': 314.0,
                'camera_cy': 182.8,
                'tire_offset_x': 0.775,
                'camera_height': 1.92,
                'look_ahead_distance': 4.0,
            }],
            additional_env=python_env
        ),
        Node(
            package='road_condition',
            executable='road_condition_node',
            name='road_condition_node',
            output='screen',
            parameters=[{
                'model_path': model_path,
                'confidence_threshold': 0.5,
            }],
            additional_env=python_env
        ),
    ])
