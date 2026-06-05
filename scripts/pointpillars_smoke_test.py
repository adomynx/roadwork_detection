#!/usr/bin/env python3
"""
Phase A smoke test: run KITTI-pretrained PointPillars on one LiDAR scan
from the rosbag. Standalone — no ROS node, no painting.

Outputs: console list of detected 3D boxes (class, position, dims, score).
"""

import sys
from pathlib import Path
import numpy as np
import torch

# OpenPCDet
sys.path.insert(0, str(Path.home() / 'roadwork_project/OpenPCDet'))
from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.models import build_network, load_data_to_gpu
from pcdet.utils import common_utils
from pcdet.datasets.processor.data_processor import DataProcessor
from pcdet.datasets.processor.point_feature_encoder import PointFeatureEncoder

# rosbag reading
from rosbags.highlevel import AnyReader
from rosbags.typesys import Stores, get_typestore

# ---- Config ----------------------------------------------------------------
OPCD_ROOT = Path.home() / 'roadwork_project/OpenPCDet'
CFG = OPCD_ROOT / 'tools/cfgs/kitti_models/pointpillar.yaml'
CKPT = OPCD_ROOT / 'checkpoints/pointpillar_7728.pth'
BAG = Path.home() / 'roadwork_project/rosbag2_2026_03_17-16_50_01'
PC_TOPIC = '/rslidar_points'
CLASSES = ['Car', 'Pedestrian', 'Cyclist']
SCORE_THRESHOLD = 0.3

# ---- Grab one PointCloud2 from the bag -------------------------------------
print('Reading one PointCloud2 from rosbag...')
typestore = get_typestore(Stores.ROS2_HUMBLE)
with AnyReader([BAG], default_typestore=typestore) as reader:
    conn = next(c for c in reader.connections if c.topic == PC_TOPIC)
    _, _, raw = next(reader.messages(connections=[conn]))
    msg = reader.deserialize(raw, conn.msgtype)

# PointCloud2 -> Nx4 (x, y, z, intensity)
dt = np.dtype({
    'names': [f.name for f in msg.fields],
    'formats': [{1: 'i1', 2: 'u1', 3: 'i2', 4: 'u2',
                 5: 'i4', 6: 'u4', 7: 'f4', 8: 'f8'}[f.datatype]
                for f in msg.fields],
    'offsets': [f.offset for f in msg.fields],
    'itemsize': msg.point_step,
})
pts_all = np.frombuffer(msg.data, dtype=dt)
mask = np.isfinite(pts_all['x']) & np.isfinite(pts_all['y']) & np.isfinite(pts_all['z'])
pts_all = pts_all[mask]
intensity = (pts_all['intensity'].astype(np.float32) / 255.0
             if 'intensity' in pts_all.dtype.names
             else np.zeros(len(pts_all), dtype=np.float32))
points = np.stack([
    pts_all['x'].astype(np.float32),
    pts_all['y'].astype(np.float32),
    pts_all['z'].astype(np.float32),
    intensity,
], axis=1)
print(f'  loaded {len(points):,} points '
      f'(x:{points[:,0].min():.1f}..{points[:,0].max():.1f}, '
      f'z:{points[:,2].min():.1f}..{points[:,2].max():.1f})')

# ---- Build PointPillars + load checkpoint ----------------------------------
print('Loading PointPillars (KITTI pretrained)...')
import os
_prev_cwd = os.getcwd()
os.chdir(OPCD_ROOT / 'tools')  # configs use _BASE_CONFIG_ paths relative to tools/
try:
    cfg_from_yaml_file(str(CFG), cfg)
finally:
    os.chdir(_prev_cwd)
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
        (point_cloud_range[3:6] - point_cloud_range[0:3]) / np.array(voxel_size)
    ).astype(np.int64)
    depth_downsample_factor = None


dataset = DummyDataset()
model = build_network(model_cfg=cfg.MODEL, num_class=len(CLASSES), dataset=dataset)
model.load_params_from_file(filename=str(CKPT), logger=logger, to_cpu=False)
model.cuda().eval()

# ---- Crop to PointPillars' KITTI range, preprocess --------------------------
pcr = dataset.point_cloud_range
m = (
    (points[:, 0] >= pcr[0]) & (points[:, 0] <= pcr[3]) &
    (points[:, 1] >= pcr[1]) & (points[:, 1] <= pcr[4]) &
    (points[:, 2] >= pcr[2]) & (points[:, 2] <= pcr[5])
)
points = points[m]
print(f'  {len(points):,} points after KITTI range crop')

dp = DataProcessor(
    cfg.DATA_CONFIG.DATA_PROCESSOR,
    point_cloud_range=pcr,
    training=False,
    num_point_features=points.shape[1],
)
data_dict = dp.forward({'points': points, 'use_lead_xyz': True})
data_dict['batch_size'] = 1
data_dict['voxel_coords'] = np.pad(
    data_dict['voxel_coords'], ((0, 0), (1, 0)), constant_values=0
)
load_data_to_gpu(data_dict)

# ---- Inference --------------------------------------------------------------
print('Running PointPillars inference...')
with torch.no_grad():
    pred_dicts, _ = model.forward(data_dict)

boxes = pred_dicts[0]['pred_boxes'].cpu().numpy()
scores = pred_dicts[0]['pred_scores'].cpu().numpy()
labels = pred_dicts[0]['pred_labels'].cpu().numpy()
keep = scores >= SCORE_THRESHOLD
boxes, scores, labels = boxes[keep], scores[keep], labels[keep]

print(f'\n=== {len(boxes)} detections (score >= {SCORE_THRESHOLD}) ===')
for b, s, l in zip(boxes, scores, labels):
    x, y, z, dx, dy, dz, yaw = b
    dist = float(np.hypot(x, y))
    print(f'  [{CLASSES[l - 1]:10s}] score={s:.2f} '
          f'pos=({x:+6.1f},{y:+6.1f},{z:+5.1f}) '
          f'size=({dx:.1f}x{dy:.1f}x{dz:.1f}) '
          f'yaw={np.degrees(yaw):+5.0f}deg dist={dist:.1f}m')
