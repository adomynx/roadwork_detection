import os
from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    pkg_dir = get_package_share_directory('weather_detection')
    model_path = os.path.join(pkg_dir, 'models', 'weather_efficientnet.keras')

    venv_path = os.path.expanduser('~/roadwork_project/venv/lib/python3.10/site-packages')
    python_env = {'PYTHONPATH': venv_path + ':' + os.environ.get('PYTHONPATH', '')}

    return LaunchDescription([
        Node(
            package='weather_detection',
            executable='weather_node',
            name='weather_node',
            output='screen',
            parameters=[{
                'model_path': model_path,
                'inference_interval': 5,
            }],
            additional_env=python_env
        ),
    ])
