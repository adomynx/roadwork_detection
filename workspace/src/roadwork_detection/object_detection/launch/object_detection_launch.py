import os

from launch import LaunchDescription
from launch.substitutions import PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    pkg_share = FindPackageShare('object_detection')
    config_file = PathJoinSubstitution(
        [pkg_share, 'config', 'object_detection_params.yaml'])

    # Prepend the project venv site-packages so ultralytics / cv2 / torch
    # resolve from the venv rather than system python — same approach the
    # other packages' launch files use. Adjust python3.10 if your venv
    # uses a different interpreter version.
    venv_site = os.path.expanduser(
        '~/roadwork_project/venv/lib/python3.10/site-packages')
    existing = os.environ.get('PYTHONPATH', '')
    pythonpath = venv_site + ((':' + existing) if existing else '')

    return LaunchDescription([
        Node(
            package='object_detection',
            executable='object_detector_node',
            name='object_detector_node',
            output='screen',
            parameters=[config_file],
            additional_env={'PYTHONPATH': pythonpath},
        ),
    ])
