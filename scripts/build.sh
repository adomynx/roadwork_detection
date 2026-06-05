#!/bin/bash

echo "Setting up environment..."
source ~/roadwork_project/venv/bin/activate
source /opt/ros/humble/setup.bash

# Navigate to workspace
cd ~/roadwork_project/workspace

# Clean previous builds to prevent caching issues
echo "Cleaning old build files..."
rm -rf build/ install/ log/

# Build all packages
echo "Building ROS 2 workspace..."
colcon build --symlink-install

# Source the newly built workspace
source install/setup.bash

echo ""
echo "=== Build complete ==="
