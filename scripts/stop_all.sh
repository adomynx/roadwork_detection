#!/bin/bash

echo "Stopping the detection pipeline..."

# 1. Kill the main tmux session housing all the nodes
tmux kill-session -t pipeline 2>/dev/null

# 2. Safety net: forcefully terminate any lingering ROS 2 launch/run processes
pkill -f "ros2 launch" 2>/dev/null
pkill -f "ros2 run" 2>/dev/null

# 3. Stop the ROS 2 daemon to clear memory completely
ros2 daemon stop 2>/dev/null

echo "=== Pipeline stopped completely ==="
