# ROS2 Roadwork Detection

ROS2 package for real-time object detection using YOLOv8 with TensorRT/ONNX on rosbag camera data.

## Requirements
- Ubuntu 22.04
- ROS2 Humble
- Python 3.10
- Ultralytics (YOLOv8)
- NVIDIA GPU + TensorRT (recommended) or CPU with ONNX Runtime

## Setup
```bash
# Create virtual environment
python3 -m venv venv --system-site-packages
source venv/bin/activate

# Install dependencies
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip3 install ultralytics tensorrt "numpy<2"

# Export YOLOv8 model (TensorRT for GPU)
python3 -c "from ultralytics import YOLO; model = YOLO('yolov8x.pt'); model.export(format='engine', half=True)"

# Or ONNX for CPU
python3 -c "from ultralytics import YOLO; model = YOLO('yolov8s.pt'); model.export(format='onnx')"

# Copy model to package
cp yolov8x.engine src/roadwork_detection/models/
```

## Build
```bash
source /opt/ros/humble/setup.bash
colcon build
source install/setup.bash
```

## Run
```bash
# Terminal 1 - Launch detector
ros2 launch roadwork_detection detection_launch.py

# Terminal 2 - Play rosbag
ros2 bag play <your_rosbag_folder>

# Terminal 3 - Visualize
ros2 run rqt_image_view rqt_image_view
```
Select `/detection/annotated_image` from the dropdown.

## Output Topics
| Topic | Type | Description |
|-------|------|-------------|
| /detection/annotated_image | sensor_msgs/Image | Frames with bounding boxes |
| /detection/results | std_msgs/String | JSON detection data |

## Configuration
Edit `config/detection_params.yaml` to change confidence threshold, resize dimensions, and target classes.
