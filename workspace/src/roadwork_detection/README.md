# ROS2 Autonomous Vehicle Perception Pipeline

Real-time roadwork detection, road condition estimation, and ODD (Operational Design Domain) exit risk assessment for autonomous vehicles. Built for ROS2 Humble.

## Repository Structure

```
├── roadwork_detection/     # Roadwork object detection + LiDAR fusion + metrics
├── road_condition/         # Road surface condition estimation (dry/wet/snow)
└── risk_assessment/        # ODD exit risk assessment
```

## 3 Packages

### 1. roadwork_detection
Camera + LiDAR fusion pipeline for roadwork object detection.

**Nodes:**
- `detector_node` — YOLOv8x TensorRT inference on ZED compressed images, metrics overlay
- `lidar_detection_node` — Ground removal (RANSAC) + obstacle clustering from point cloud
- `fusion_node` — Projects LiDAR 3D obstacles onto camera, matches with 2D detections
- `confidence_node` — Per-frame roadwork existence probability with EMA smoothing
- `video_publisher_node` — Reads mp4 files as ROS2 camera topic (for dataset testing)

### 2. road_condition
Road surface condition estimation using EfficientNet-B0.

**Nodes:**
- `road_patch_node` — Extracts road surface patch using camera projection (tire position)
- `road_condition_node` — Classifies patch as dry/wet/snow using EfficientNet-B0

### 3. risk_assessment
ODD exit risk assessment combining distance + confidence + dynamic objects.

**Nodes:**
- `risk_node` — R_frame = max(R_d, R_c) with EMA smoothing

## Pipeline Architecture

```
MCAP Rosbag
├── ZED Camera ──→ detector_node (YOLOv8x TensorRT, 17 classes, ~9ms)
│                       ↓ 2D detections
├── LiDAR ──→ lidar_detection_node (RANSAC + clustering)
│                       ↓ 3D obstacles
│                  fusion_node (TF transform, match 3D ↔ 2D)
│                       ↓ classified objects + real 3D distance
│              ┌────────┼────────┐
│              ↓        ↓        ↓
│          confidence  distance  risk_node
│            node      (fusion)  (separate pkg)
│              ↓        ↓        ↓
├── ZED Camera ──→ road_patch_node ──→ road_condition_node
│                  (crop road surface)   (EfficientNet-B0)
│                                            ↓
└── All metrics overlaid on annotated image (rqt_image_view)
    Distance | Confidence | Risk | Road: dry/wet/snow
```

## Topics

| Topic | Type | Description |
|-------|------|-------------|
| /detection/annotated_image | Image | Frames with bboxes + metrics overlay |
| /detection/results | String | JSON detection data |
| /lidar/obstacles | String | JSON 3D obstacle data |
| /fusion/results | String | JSON fused results |
| /metrics/distance | Float64 | Nearest object distance (EMA) |
| /metrics/confidence | Float64 | Roadwork existence probability (EMA) |
| /metrics/risk | Float64 | ODD exit risk score (EMA) |
| /road/patch | Image | Cropped road surface patch |
| /road/condition | String | Road condition: dry/wet/snow |

## ROADWork Dataset Classes (17)

Police Officer, Police Vehicle, Cone, Fence, Drum, Barricade, Barrier, Work Vehicle, Vertical Panel, Tubular Marker, Arrow Board, Bike Lane, Work Equipment, Worker, Other Roadwork Objects, TTC Message Board, TTC Sign

## Requirements

- Ubuntu 22.04 + ROS2 Humble
- Python 3.10
- NVIDIA GPU + TensorRT (RTX A5000 recommended)
- Ultralytics YOLOv8, PyTorch with CUDA
- TensorFlow (for road condition)

## Setup

```bash
python3 -m venv ~/roadwork_project/venv --system-site-packages
source ~/roadwork_project/venv/bin/activate
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip3 install ultralytics tensorrt tensorflow "numpy<2"
```

## Build

```bash
source ~/roadwork_project/venv/bin/activate
source /opt/ros/humble/setup.bash
cd ~/roadwork_project/workspace
colcon build
source install/setup.bash
```

## Run

```bash
# Terminal 1 - Roadwork detection (detector + LiDAR + fusion + confidence)
ros2 launch roadwork_detection detection_launch.py

# Terminal 2 - Road condition (patch + EfficientNet)
ros2 launch road_condition road_condition_launch.py

# Terminal 3 - Risk assessment
ros2 launch risk_assessment risk_launch.py

# Terminal 4 - Play rosbag
ros2 bag play <rosbag_folder> --rate 0.5

# Terminal 5 - Visualize
ros2 run rqt_image_view rqt_image_view
# Select /detection/annotated_image
```

## Model Weights

Swappable via config — place any trained model in the models folder:
- `roadwork_detection/models/yolov8x_roadwork.engine` — YOLOv8x TensorRT
- `road_condition/models/EfficientNet_training_N.keras` — EfficientNet-B0

## Training

- YOLOv8x fine-tuned on ROADWork dataset (Carnegie Mellon KiltHub): 5,318 images, 17 classes, mAP50 = 48.1%
- EfficientNet-B0 trained on Boreas road patches: 70,322 images, 3 classes (dry/snow/wet)

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

## References

- ICCAR 2026: "Operational Design Domain Exit Detection and Risk Assessment for Roadworks Scenarios in Autonomous Vehicles"
- ROADWork Dataset: Carnegie Mellon KiltHub
- Boreas Dataset: University of Toronto
