import os
from glob import glob
from setuptools import setup

package_name = 'roadwork_detection'

setup(
    name=package_name,
    version='0.0.1',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
        (os.path.join('share', package_name, 'models'), glob('models/*')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='agbueker',
    maintainer_email='agbueker@todo.todo',
    description='ROS2 Roadwork Detection using YOLOv8 TensorRT',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'detector_node = roadwork_detection.detector_node:main',
            'distance_node = roadwork_detection.distance_node:main',
            'confidence_node = roadwork_detection.confidence_node:main',
            'risk_node = roadwork_detection.risk_node:main',
            'video_publisher_node = roadwork_detection.video_publisher_node:main',
        ],
    },
)
