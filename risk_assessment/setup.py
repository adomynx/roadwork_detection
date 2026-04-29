import os
from glob import glob
from setuptools import setup

package_name = 'risk_assessment'

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
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='agbueker',
    maintainer_email='agbueker@todo.todo',
    description='ROS2 Risk Assessment Node for Autonomous Vehicles',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'risk_node = risk_assessment.risk_node:main',
        ],
    },
)
