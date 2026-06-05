import os
from glob import glob

from setuptools import setup

package_name = 'object_detection'

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'),
            glob('launch/*.py')),
        (os.path.join('share', package_name, 'config'),
            glob('config/*.yaml')),
        (os.path.join('share', package_name, 'models'),
            glob('models/*.engine')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Kartik Wakekar',
    maintainer_email='kartiktoons@gmail.com',
    description=('General-purpose object detection (YOLOv8x COCO) '
                 'for the perception pipeline.'),
    license='MIT',
    entry_points={
        'console_scripts': [
            'object_detector_node = '
            'object_detection.object_detector_node:main',
        ],
    },
)
