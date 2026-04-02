# ROS2 Roadwork Detection Pipeline

Real-time roadwork detection and ODD (Operational Design Domain) exit risk assessment for autonomous vehicles using camera-LiDAR fusion. Built for ROS2 Humble.

Implements the metrics from: *"Operational Design Domain Exit Detection and Risk Assessment for Roadworks Scenarios in Autonomous Vehicles"* (ICCAR 2026)

## Pipeline Architecture

```
MCAP Rosbag
├── ZED Camera → detector_node (YOLOv8x TensorRT, 17 classes, ~10ms)
│                     ↓ 2D detections (class + bbox)
├── LiDAR      → lidar_detection_node (ground removal + clustering)
│                     ↓ 3D obstacles (position + size)
│                fusion_node (TF transform, match 3D ↔ 2D)
│                     ↓ classified objects with real 3D distance
│              ┌──────┼──────┐
│              ↓      ↓      ↓
│          distance confidence risk
│           node     node    node
│              ↓      ↓      ↓
└── Live annotated feed with ODD exit metrics overlay
```

## Nodes

| Node | Description |
|------|-------------|
| detector_node | YOLOv8x TensorRT inference on ZED compressed images, metrics overlay |
| lidar_detection_node | Ground removal (RANSAC) + obstacle clustering from point cloud |
| fusion_node | Projects LiDAR 3D obstacles onto camera image, matches with 2D detections |
| confidence_node | Per-frame roadwork existence probability (C_frame) with EMA smoothing |
| risk_node | Combined risk assessment R_frame = max(R_d, R_c) with EMA smoothing |
| video_publisher_node | Reads mp4 files and publishes as ROS2 camera topic (for dataset testing) |

## Topics

| Topic | Type | Description |
|-------|------|-------------|
| /detection/annotated_image | sensor_msgs/Image | Frames with bounding boxes + metrics overlay |
| /detection/results | std_msgs/String | JSON detection data (class, bbox, confidence) |
| /lidar/obstacles | std_msgs/String | JSON 3D obstacle data (centroid, size, distance) |
| /fusion/results | std_msgs/String | JSON fused results (class + real 3D distance) |
| /metrics/distance | std_msgs/Float64 | Nearest object distance (EMA smoothed) |
| /metrics/confidence | std_msgs/Float64 | Roadwork existence probability (EMA smoothed) |
| /metrics/risk | std_msgs/Float64 | ODD exit risk score (EMA smoothed) |

## ROADWork Dataset Classes (17)

Police Officer, Police Vehicle, Cone, Fence, Drum, Barricade, Barrier, Work Vehicle, Vertical Panel, Tubular Marker, Arrow Board, Bike Lane, Work Equipment, Worker, Other Roadwork Objects, TTC Message Board, TTC Sign

## Requirements

- Ubuntu 22.04 + ROS2 Humble
- Python 3.10
- NVIDIA GPU + TensorRT (RTX A5000 recommended)
- Ultralytics YOLOv8
- PyTorch with CUDA

## Setup

```bash
# Create virtual environment
python3 -m venv ~/roadwork_project/venv --system-site-packages
source ~/roadwork_project/venv/bin/activate

# Install dependencies
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip3 install ultralytics tensorrt "numpy<2"

# Place model in models folder
cp yolov8x_roadwork.engine src/roadwork_detection/models/
```

## Build

```bash
source ~/roadwork_project/venv/bin/activate
source /opt/ros/humble/setup.bash
cd ~/roadwork_project/workspace
colcon build
source install/setup.bash
```

## Run with Rosbag

```bash
# Terminal 1 - Launch all nodes
ros2 launch roadwork_detection detection_launch.py

# Terminal 2 - Play rosbag
ros2 bag play <rosbag_folder> --rate 0.5

# Terminal 3 - Visualize
ros2 run rqt_image_view rqt_image_view
# Select /detection/annotated_image
```

## Run with Video Files

```bash
# Update launch file to include video_publisher_node
# Set video_dir parameter to your mp4 folder
ros2 launch roadwork_detection detection_launch.py
```

## Configuration

Edit `config/detection_params.yaml` to tune parameters.

## Model Weights

Model weights are swappable via config. Place any trained .engine file in the models folder and update the launch file:

```python
model_path = os.path.join(pkg_dir, 'models', 'your_model.engine')
```

## Training

The YOLOv8x model was fine-tuned on the ROADWork dataset (Carnegie Mellon KiltHub): 5,318 images, 17 classes, mAP50 = 48.1%. Training took 94 epochs (~4.2 hours on RTX A5000).

## Paper Parameters (Table II)

| Parameter | Value |
|-----------|-------|
| EMA α_d (distance) | 0.15 |
| EMA α_c (confidence) | 0.15 |
| EMA α_r (risk) | 0.20 |
| λ (confidence weighting) | 0.50 |
| N_max (max detection count) | 12 |
| d_near | 5.0m |
| d_far | 30.0m |
| w_c (confidence weight) | 0.50 |
| w_d (dynamic object weight) | 0.50 |
