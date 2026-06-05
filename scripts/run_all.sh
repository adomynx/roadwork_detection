#!/bin/bash
SETUP="source ~/roadwork_project/venv/bin/activate && source /opt/ros/humble/setup.bash && source ~/roadwork_project/workspace/install/setup.bash"

# Kill existing session
tmux kill-session -t pipeline 2>/dev/null

# Create new tmux session
tmux new-session -d -s pipeline -n detection

# T1: Roadwork detection
tmux send-keys -t pipeline:detection "$SETUP && ros2 launch roadwork_detection detection_launch.py" C-m

# T2: Road condition
tmux new-window -t pipeline -n road
tmux send-keys -t pipeline:road "$SETUP && ros2 launch road_condition road_condition_launch.py" C-m

# T3: Risk assessment
tmux new-window -t pipeline -n risk
tmux send-keys -t pipeline:risk "$SETUP && ros2 launch risk_assessment risk_launch.py" C-m

# T4: Weather detection
tmux new-window -t pipeline -n weather
tmux send-keys -t pipeline:weather "$SETUP && ros2 launch weather_detection weather_launch.py" C-m

# T5: Object detection (general YOLOv8x COCO)
tmux new-window -t pipeline -n objects
tmux send-keys -t pipeline:objects "$SETUP && ros2 launch object_detection object_detection_launch.py" C-m

# T6: Rosbag (wait 12s for nodes to start — two YOLO engines now load/warm up)
tmux new-window -t pipeline -n rosbag
tmux send-keys -t pipeline:rosbag "sleep 12 && source /opt/ros/humble/setup.bash && cd ~/roadwork_project/rosbag2_2026_03_17-16_50_01 && ros2 bag play . --rate 0.5" C-m

# T7: Visualize
tmux new-window -t pipeline -n viz
tmux send-keys -t pipeline:viz "sleep 14 && source /opt/ros/humble/setup.bash && ros2 run rqt_image_view rqt_image_view" C-m

echo "Pipeline started in tmux session 'pipeline'"
echo "Attach: tmux attach -t pipeline"
echo "Switch windows: Ctrl+b then number (0-6)"
echo "Kill: tmux kill-session -t pipeline"
