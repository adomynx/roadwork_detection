#!/bin/bash
# Runs object_detection node + rosbag, captures output to a timestamped .md log.

set -o pipefail

LOG_DIR="$HOME/roadwork_project/logs"
mkdir -p "$LOG_DIR"
TS=$(date +%Y%m%d_%H%M%S)
LOG_FILE="$LOG_DIR/object_detection_test_${TS}.md"
NODE_OUT="$LOG_DIR/.node_${TS}.txt"
HZ_OUT="$LOG_DIR/.hz_${TS}.txt"
ECHO_OUT="$LOG_DIR/.echo_${TS}.txt"

source ~/roadwork_project/venv/bin/activate
source /opt/ros/humble/setup.bash
source ~/roadwork_project/workspace/install/setup.bash

cleanup() {
  [[ -n "${BAG_PID:-}" ]] && kill "$BAG_PID" 2>/dev/null
  [[ -n "${NODE_PID:-}" ]] && kill "$NODE_PID" 2>/dev/null
  [[ -n "${HZ_PID:-}" ]] && kill "$HZ_PID" 2>/dev/null
  sleep 1
  pkill -f object_detection_launch 2>/dev/null
  pkill -f object_detector_node 2>/dev/null
  pkill -f "ros2 bag play" 2>/dev/null
  rm -f "$NODE_OUT" "$HZ_OUT" "$ECHO_OUT"
}
trap cleanup EXIT INT TERM

echo "===> Starting object_detector_node (warmup ~10s)..."
ros2 launch object_detection object_detection_launch.py > "$NODE_OUT" 2>&1 &
NODE_PID=$!
sleep 10

echo "===> Playing rosbag at 0.5x..."
ros2 bag play ~/roadwork_project/rosbag2_2026_03_17-16_50_01 --rate 0.5 > /dev/null 2>&1 &
BAG_PID=$!
sleep 5

echo "===> Sampling /objects/results for 15s..."
timeout 12s ros2 topic echo /objects/results --once > "$ECHO_OUT" 2>&1 &
ros2 topic hz /objects/results > "$HZ_OUT" 2>&1 &
HZ_PID=$!
sleep 15

echo "===> Stopping..."
kill "$HZ_PID" "$BAG_PID" "$NODE_PID" 2>/dev/null
sleep 2

# ---- Assemble markdown ----
{
  echo "# Object Detection Standalone Test"
  echo ""
  echo "- **Date:** $(date)"
  echo "- **Host:** $(hostname)"
  echo "- **Rosbag:** \`~/roadwork_project/rosbag2_2026_03_17-16_50_01\` @ 0.5x"
  echo ""
  echo "## /objects/results — topic hz"
  echo '```'
  tail -n 20 "$HZ_OUT" 2>/dev/null || echo "(no hz output)"
  echo '```'
  echo ""
  echo "## /objects/results — sample message"
  echo '```json'
  cat "$ECHO_OUT" 2>/dev/null || echo "(no echo output)"
  echo '```'
  echo ""
  echo "## object_detector_node — filtered stdout"
  echo '```'
  grep -E "Loading TRT|warmed up|OBJ FPS|ready|ERROR|Failed|Traceback" "$NODE_OUT" 2>/dev/null \
    || tail -n 40 "$NODE_OUT"
  echo '```'
  echo ""
  echo "## Summary"
  HZ_RATE=$(grep "average rate" "$HZ_OUT" 2>/dev/null | tail -n 1 | awk '{print $3}')
  FPS_LINE=$(grep "OBJ FPS" "$NODE_OUT" 2>/dev/null | tail -n 1 | sed 's/.*]: //')
  DET_COUNT=$(grep -oP '"count":\s*\K[0-9]+' "$ECHO_OUT" 2>/dev/null | head -n 1)
  CLASSES=$(grep -oP '"class_name":\s*"\K[^"]+' "$ECHO_OUT" 2>/dev/null | sort -u | paste -sd, -)
  echo "- **Publish rate:** ${HZ_RATE:-N/A} Hz"
  echo "- **Last FPS log:** ${FPS_LINE:-N/A}"
  echo "- **Detections in sample frame:** ${DET_COUNT:-N/A}"
  echo "- **Classes seen in sample:** ${CLASSES:-N/A}"
} > "$LOG_FILE"

echo ""
echo "===> Done. Log: $LOG_FILE"
