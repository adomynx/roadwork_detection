import os
from glob import glob
from setuptools import setup

package_name = 'road_condition'

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
    description='ROS2 Road Condition Estimation using EfficientNet-B0',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'road_patch_node = road_condition.road_patch_node:main',
            'road_condition_node = road_condition.road_condition_node:main',
        ],
    },
)
