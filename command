
source ~/roadwork_project/venv/bin/activate
source /opt/ros/humble/setup.bash
cd ~/roadwork_project/workspace
rm -rf build install log
colcon build
source install/setup.bash
ros2 launch roadwork_detection detection_launch.py





source /opt/ros/humble/setup.bash
cd ~/roadwork_project/rosbag2_2026_03_17-16_50_01
ros2 bag play . --rate 0.5




source /opt/ros/humble/setup.bash
ros2 run rqt_image_view rqt_image_view

