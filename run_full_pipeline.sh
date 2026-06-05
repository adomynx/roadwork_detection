#!/bin/bash

# ==============================
# ROS2 Full Pipeline Runner
# Automatically saves logs per run
# ==============================

set -e

# Create timestamped log folder
RUN_ID=$(date +%Y%m%d_%H%M%S)
RUN_LOG_DIR="$HOME/roadwork_project/run_logs/full_pipeline_$RUN_ID"
mkdir -p "$RUN_LOG_DIR"

echo "===================================="
echo "Starting full ROS2 perception pipeline"
echo "Logs will be saved in:"
echo "$RUN_LOG_DIR"
echo "===================================="

# Source environment
source "$HOME/roadwork_project/venv/bin/activate"
source /opt/ros/humble/setup.bash
cd "$HOME/roadwork_project/workspace"
source install/setup.bash

# Save run info
{
    echo "Run ID: $RUN_ID"
    echo "Run time: $(date)"
    echo "Workspace: $HOME/roadwork_project/workspace"
    echo "ROS distro: $ROS_DISTRO"
    echo "Python: $(which python)"
} > "$RUN_LOG_DIR/run_info.txt"

# Copy weather training log if it exists
if [ -f "$HOME/roadwork_project/weather_model/training_log.json" ]; then
    cp "$HOME/roadwork_project/weather_model/training_log.json" "$RUN_LOG_DIR/weather_training_log.json"
fi

# Save weather model file info if it exists
if [ -f "$HOME/roadwork_project/weather_model/weather_efficientnet.keras" ]; then
    ls -lh "$HOME/roadwork_project/weather_model/weather_efficientnet.keras" > "$RUN_LOG_DIR/weather_model_file_info.txt"
fi

# Start nodes in background and save logs automatically
ros2 launch roadwork_detection detection_launch.py \
    > "$RUN_LOG_DIR/detection_launch.log" 2>&1 &
PID_DETECTION=$!

sleep 3

ros2 launch road_condition road_condition_launch.py \
    > "$RUN_LOG_DIR/road_condition_launch.log" 2>&1 &
PID_ROAD=$!

sleep 3

ros2 launch risk_assessment risk_launch.py \
    > "$RUN_LOG_DIR/risk_launch.log" 2>&1 &
PID_RISK=$!

sleep 3

ros2 launch weather_detection weather_launch.py \
    > "$RUN_LOG_DIR/weather_launch.log" 2>&1 &
PID_WEATHER=$!

sleep 5

# Save active topic list before rosbag starts
ros2 topic list > "$RUN_LOG_DIR/topic_list_before_rosbag.txt" 2>&1 || true

echo "===================================="
echo "All nodes started."
echo "Now playing rosbag..."
echo "Press Ctrl+C to stop everything."
echo "===================================="

# Play rosbag and save log
ros2 bag play "$HOME/roadwork_project/rosbag2_2026_03_17-16_50_01" --rate 0.5 \
    > "$RUN_LOG_DIR/rosbag_play.log" 2>&1 &
PID_ROSBAG=$!

sleep 10

# Save topic list after rosbag starts
ros2 topic list > "$RUN_LOG_DIR/topic_list_after_rosbag.txt" 2>&1 || true

# Save some short topic samples automatically
timeout 20 ros2 topic echo /weather/condition \
    > "$RUN_LOG_DIR/weather_condition_topic.log" 2>&1 || true &

timeout 20 ros2 topic echo /road/condition \
    > "$RUN_LOG_DIR/road_condition_topic.log" 2>&1 || true &

timeout 20 ros2 topic echo /fusion/results \
    > "$RUN_LOG_DIR/fusion_results_topic.log" 2>&1 || true &

timeout 20 ros2 topic echo /metrics/risk \
    > "$RUN_LOG_DIR/risk_topic.log" 2>&1 || true &

# Keep script alive until user stops
trap "echo 'Stopping pipeline...'; kill $PID_DETECTION $PID_ROAD $PID_RISK $PID_WEATHER $PID_ROSBAG 2>/dev/null; echo 'Logs saved in: $RUN_LOG_DIR'; exit 0" INT

wait $PID_ROSBAG

echo "Rosbag finished."
echo "Stopping nodes..."
kill $PID_DETECTION $PID_ROAD $PID_RISK $PID_WEATHER 2>/dev/null || true

echo "===================================="
echo "Pipeline finished."
echo "Logs saved in:"
echo "$RUN_LOG_DIR"
echo "===================================="
