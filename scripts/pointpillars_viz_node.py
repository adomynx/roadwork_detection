#!/usr/bin/env python3
"""
Phase A visualization: run KITTI-pretrained PointPillars on live
/rslidar_points and publish 3D detections as visualization markers.

Standalone ROS2 node — no painting, no /objects/results yet.
Use to confirm boxes land on the right LiDAR returns in RViz.

Publishes:
  /pointpillars/markers   (MarkerArray, lifetime ~0.2s)
  /pointpillars/info      (String, JSON summary per scan)
"""

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import String, ColorRGBA
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point

# ---- OpenPCDet -------------------------------------------------------------
sys.path.insert(0, str(Path.home() / 'roadwork_project/OpenPCDet'))
from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.models import build_network, load_data_to_gpu
from pcdet.utils import common_utils
from pcdet.datasets.processor.data_processor import DataProcessor
from pcdet.datasets.processor.point_feature_encoder import PointFeatureEncoder

OPCD_ROOT = Path.home() / 'roadwork_project/OpenPCDet'
CFG = OPCD_ROOT / 'tools/cfgs/kitti_models/pointpillar.yaml'
CKPT = OPCD_ROOT / 'checkpoints/pointpillar_7728.pth'
CLASSES = ['Car', 'Pedestrian', 'Cyclist']
DEFAULT_SCORE_THRESHOLD = 0.3
DEFAULT_MAX_DISTANCE = 60.0  # meters; KITTI range extends to ~70m, reliability drops past 50m

# Per-class display color (RGBA, 0..1)
CLASS_COLORS = {
    'Car':        (0.0, 1.0, 0.2, 0.9),   # green
    'Pedestrian': (1.0, 0.4, 0.0, 0.9),   # orange
    'Cyclist':    (0.2, 0.6, 1.0, 0.9),   # blue
}


def parse_pointcloud2(msg):
    """PointCloud2 -> Nx4 float32 (x, y, z, intensity normalized to [0,1])."""
    dt = np.dtype({
        'names': [f.name for f in msg.fields],
        'formats': [{1: 'i1', 2: 'u1', 3: 'i2', 4: 'u2',
                     5: 'i4', 6: 'u4', 7: 'f4', 8: 'f8'}[f.datatype]
                    for f in msg.fields],
        'offsets': [f.offset for f in msg.fields],
        'itemsize': msg.point_step,
    })
    arr = np.frombuffer(msg.data, dtype=dt)
    mask = np.isfinite(arr['x']) & np.isfinite(arr['y']) & np.isfinite(arr['z'])
    arr = arr[mask]
    intensity = (arr['intensity'].astype(np.float32) / 255.0
                 if 'intensity' in arr.dtype.names
                 else np.zeros(len(arr), dtype=np.float32))
    return np.stack([arr['x'].astype(np.float32),
                     arr['y'].astype(np.float32),
                     arr['z'].astype(np.float32),
                     intensity], axis=1)


def box_to_corners(x, y, z, dx, dy, dz, yaw):
    """7-DoF box -> 8 corner points in world frame."""
    cx, sx = np.cos(yaw), np.sin(yaw)
    # local-frame corners (centered at origin)
    hx, hy, hz = dx / 2, dy / 2, dz / 2
    local = np.array([
        [+hx, +hy, -hz], [+hx, -hy, -hz], [-hx, -hy, -hz], [-hx, +hy, -hz],
        [+hx, +hy, +hz], [+hx, -hy, +hz], [-hx, -hy, +hz], [-hx, +hy, +hz],
    ])
    R = np.array([[cx, -sx, 0], [sx, cx, 0], [0, 0, 1]])
    return (local @ R.T) + np.array([x, y, z])


# 12 line segments connecting the 8 corners
_EDGES = [(0, 1), (1, 2), (2, 3), (3, 0),
          (4, 5), (5, 6), (6, 7), (7, 4),
          (0, 4), (1, 5), (2, 6), (3, 7)]


class PointPillarsVizNode(Node):
    def __init__(self):
        super().__init__('pointpillars_viz_node')

        # ---- Tunable parameters (live-adjustable via `ros2 param set`) ----
        self.declare_parameter('score_threshold', DEFAULT_SCORE_THRESHOLD)
        self.declare_parameter('max_distance', DEFAULT_MAX_DISTANCE)

        # ---- Model ----
        self.get_logger().info('Loading PointPillars (KITTI pretrained)...')
        _prev = os.getcwd()
        os.chdir(OPCD_ROOT / 'tools')
        try:
            cfg_from_yaml_file(str(CFG), cfg)
        finally:
            os.chdir(_prev)
        logger = common_utils.create_logger()

        class DummyDataset:
            class_names = CLASSES
            point_feature_encoder = PointFeatureEncoder(
                cfg.DATA_CONFIG.POINT_FEATURE_ENCODING,
                point_cloud_range=np.array(cfg.DATA_CONFIG.POINT_CLOUD_RANGE),
            )
            point_cloud_range = np.array(cfg.DATA_CONFIG.POINT_CLOUD_RANGE)
            voxel_size = cfg.DATA_CONFIG.DATA_PROCESSOR[-1].VOXEL_SIZE
            grid_size = np.round(
                (point_cloud_range[3:6] - point_cloud_range[0:3]) /
                np.array(voxel_size)).astype(np.int64)
            depth_downsample_factor = None

        self.dataset = DummyDataset()
        self.pcr = self.dataset.point_cloud_range
        self.dp = DataProcessor(
            cfg.DATA_CONFIG.DATA_PROCESSOR,
            point_cloud_range=self.pcr,
            training=False,
            num_point_features=4,
        )
        self.model = build_network(
            model_cfg=cfg.MODEL, num_class=len(CLASSES), dataset=self.dataset)
        self.model.load_params_from_file(
            filename=str(CKPT), logger=logger, to_cpu=False)
        self.model.cuda().eval()
        self.get_logger().info('Model ready')

        # ---- ROS plumbing ----
        self.sub = self.create_subscription(
            PointCloud2, '/rslidar_points',
            self.pc_callback, qos_profile_sensor_data)
        self.marker_pub = self.create_publisher(
            MarkerArray, '/pointpillars/markers', 10)
        self.info_pub = self.create_publisher(
            String, '/pointpillars/info', 10)
        self.detections_pub = self.create_publisher(
            String, '/pointpillars/detections', 10)

        self.frame_count = 0
        self.last_log_time = time.time()
        self.get_logger().info('pointpillars_viz_node ready')

    # ----------------------------------------------------------------------
    def pc_callback(self, msg):
        t0 = time.time()
        points = parse_pointcloud2(msg)
        m = ((points[:, 0] >= self.pcr[0]) & (points[:, 0] <= self.pcr[3]) &
             (points[:, 1] >= self.pcr[1]) & (points[:, 1] <= self.pcr[4]) &
             (points[:, 2] >= self.pcr[2]) & (points[:, 2] <= self.pcr[5]))
        points = points[m]
        if len(points) < 100:
            return

        try:
            data_dict = self.dp.forward({'points': points, 'use_lead_xyz': True})
            data_dict['batch_size'] = 1
            data_dict['voxel_coords'] = np.pad(
                data_dict['voxel_coords'], ((0, 0), (1, 0)), constant_values=0)
            load_data_to_gpu(data_dict)
            with torch.no_grad():
                pred_dicts, _ = self.model.forward(data_dict)
        except Exception as e:
            self.get_logger().error(f'Inference failed: {e}')
            return

        boxes = pred_dicts[0]['pred_boxes'].cpu().numpy()
        scores = pred_dicts[0]['pred_scores'].cpu().numpy()
        labels = pred_dicts[0]['pred_labels'].cpu().numpy()

        # Read live parameter values each frame so tuning takes effect immediately
        score_threshold = self.get_parameter('score_threshold').value
        max_distance = self.get_parameter('max_distance').value

        distances = np.hypot(boxes[:, 0], boxes[:, 1]) if len(boxes) else np.array([])
        keep = (scores >= score_threshold) & (distances <= max_distance)
        boxes, scores, labels = boxes[keep], scores[keep], labels[keep]

        inference_ms = (time.time() - t0) * 1000.0

        # ---- Build markers ----
        ma = MarkerArray()

        # First: a DELETEALL to clear previous frame's boxes
        clr = Marker()
        clr.header = msg.header
        clr.action = Marker.DELETEALL
        ma.markers.append(clr)

        for i, (b, s, l) in enumerate(zip(boxes, scores, labels)):
            x, y, z, dx, dy, dz, yaw = b
            cname = CLASSES[l - 1]
            color = CLASS_COLORS[cname]

            # Box edges as a LINE_LIST
            line = Marker()
            line.header = msg.header
            line.ns = 'boxes'
            line.id = i
            line.type = Marker.LINE_LIST
            line.action = Marker.ADD
            line.scale.x = 0.08  # line width in meters
            line.color = ColorRGBA(r=color[0], g=color[1], b=color[2], a=color[3])
            line.lifetime.sec = 0
            line.lifetime.nanosec = 200_000_000  # 0.2s
            corners = box_to_corners(x, y, z, dx, dy, dz, yaw)
            for a, b_ in _EDGES:
                p1 = Point(x=float(corners[a, 0]),
                           y=float(corners[a, 1]),
                           z=float(corners[a, 2]))
                p2 = Point(x=float(corners[b_, 0]),
                           y=float(corners[b_, 1]),
                           z=float(corners[b_, 2]))
                line.points.append(p1)
                line.points.append(p2)
            ma.markers.append(line)

            # Floating text label above each box
            txt = Marker()
            txt.header = msg.header
            txt.ns = 'labels'
            txt.id = i
            txt.type = Marker.TEXT_VIEW_FACING
            txt.action = Marker.ADD
            txt.pose.position.x = float(x)
            txt.pose.position.y = float(y)
            txt.pose.position.z = float(z + dz / 2 + 0.5)
            txt.scale.z = 0.8  # text height
            txt.color = ColorRGBA(r=1.0, g=1.0, b=1.0, a=1.0)
            txt.text = f'{cname} {s:.2f}'
            txt.lifetime.sec = 0
            txt.lifetime.nanosec = 200_000_000
            ma.markers.append(txt)

        self.marker_pub.publish(ma)

        # ---- Structured detections (for fusion_node) ----
        det_msg = String()
        detections = []
        for b, s, l in zip(boxes, scores, labels):
            x, y, z, dx, dy, dz, yaw = b
            detections.append({
                'class_id': int(l),
                'class_name': CLASSES[l - 1],
                'score': float(s),
                'pos': [float(x), float(y), float(z)],
                'dims': [float(dx), float(dy), float(dz)],
                'yaw': float(yaw),
                'distance_m': float(np.hypot(x, y)),
            })
        det_msg.data = json.dumps({
            'stamp_sec': msg.header.stamp.sec,
            'stamp_nanosec': msg.header.stamp.nanosec,
            'frame_id': msg.header.frame_id,
            'inference_ms': round(inference_ms, 2),
            'count': len(detections),
            'detections': detections,
        })
        self.detections_pub.publish(det_msg)

        # ---- Info ----
        info = String()
        info.data = json.dumps({
            'stamp_sec': msg.header.stamp.sec,
            'frame_id': msg.header.frame_id,
            'inference_ms': round(inference_ms, 2),
            'count': len(boxes),
            'classes': [CLASSES[l - 1] for l in labels],
        })
        self.info_pub.publish(info)

        # ---- Periodic FPS log ----
        self.frame_count += 1
        now = time.time()
        if now - self.last_log_time >= 5.0:
            fps = self.frame_count / (now - self.last_log_time)
            # Per-class breakdown of current frame
            from collections import Counter
            class_counts = Counter(CLASSES[l - 1] for l in labels)
            breakdown = ' '.join(f'{c}:{class_counts.get(c, 0)}' for c in CLASSES)
            self.get_logger().info(
                f'PP FPS: {fps:.1f} | inference: {inference_ms:.1f}ms | '
                f'det: {len(boxes)} ({breakdown}) | '
                f'thresh: {score_threshold:.2f} | max_dist: {max_distance:.0f}m')
            self.frame_count = 0
            self.last_log_time = now


def main(args=None):
    rclpy.init(args=args)
    node = PointPillarsVizNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
