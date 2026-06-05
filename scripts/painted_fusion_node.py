#!/usr/bin/env python3
"""
painted_fusion_node.py — Phase B sensor fusion (PointPainting-style re-rank).

Subscribes:
  /objects/results          (std_msgs/String, JSON)  — YOLO 2D detections
  /pointpillars/detections  (std_msgs/String, JSON)  — PointPillars 3D detections

Publishes:
  /objects/fused_3d           (std_msgs/String, JSON)        — rich fused output
  /objects/fused_3d_markers   (visualization_msgs/MarkerArray) — color by match

For each 3D box, projects its center to the camera image and matches to a 2D
box. Score is boosted on class agreement, suppressed when no 2D detection
covers the projected location (only if projection lands inside the image).
"""

import json
import time
from collections import Counter

import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import String, ColorRGBA
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point


# ============================================================================
# Calibration — same TF as existing fusion_node.py (rslidar -> zed_left_camera_optical_frame)
# ============================================================================
T_LIDAR_TO_CAM = np.array([
    [0.0, -1.0,  0.0, -0.060],
    [0.0,  0.0, -1.0, -1.011],
    [1.0,  0.0,  0.0, -0.010],
    [0.0,  0.0,  0.0,  1.0],
])

# ZED 2i left camera intrinsics
FX, FY = 266.9, 266.9
CX, CY = 314.0, 182.8
IMG_W, IMG_H = 640, 360

# COCO class names (from YOLO) -> KITTI class names (from PointPillars)
COCO_TO_KITTI = {
    'car': 'Car', 'truck': 'Car', 'bus': 'Car',
    'person': 'Pedestrian',
    'bicycle': 'Cyclist', 'motorcycle': 'Cyclist',
}

# Marker colors by match status
STATUS_COLORS = {
    'agree':     (0.0, 1.0, 0.2, 0.95),   # bright green — high confidence
    'disagree':  (1.0, 0.9, 0.0, 0.85),   # yellow — semantic conflict
    'no_2d':     (1.0, 0.2, 0.2, 0.75),   # red — likely false positive
    'off_image': (0.6, 0.6, 0.6, 0.6),    # grey — can't verify
    'no_yolo':   (0.4, 0.4, 0.8, 0.7),    # blue-grey — YOLO not yet seen
}


def project_to_image(pos_lidar):
    """3D point in LiDAR frame -> (u, v, Zc); Zc<=0 or off-image means not visible."""
    p = np.array([pos_lidar[0], pos_lidar[1], pos_lidar[2], 1.0])
    pc = T_LIDAR_TO_CAM @ p
    Xc, Yc, Zc = pc[0], pc[1], pc[2]
    if Zc <= 0.1:
        return None, None, Zc
    u = FX * Xc / Zc + CX
    v = FY * Yc / Zc + CY
    return u, v, Zc


def box_corners_world(x, y, z, dx, dy, dz, yaw):
    """7-DoF box -> 8 corner points (in LiDAR/world frame)."""
    hx, hy, hz = dx / 2, dy / 2, dz / 2
    local = np.array([
        [+hx, +hy, -hz], [+hx, -hy, -hz], [-hx, -hy, -hz], [-hx, +hy, -hz],
        [+hx, +hy, +hz], [+hx, -hy, +hz], [-hx, -hy, +hz], [-hx, +hy, +hz],
    ])
    cy_, sy_ = np.cos(yaw), np.sin(yaw)
    R = np.array([[cy_, -sy_, 0], [sy_, cy_, 0], [0, 0, 1]])
    return (local @ R.T) + np.array([x, y, z])


_EDGES = [(0, 1), (1, 2), (2, 3), (3, 0),
          (4, 5), (5, 6), (6, 7), (7, 4),
          (0, 4), (1, 5), (2, 6), (3, 7)]


# ============================================================================
class PaintedFusionNode(Node):
    def __init__(self):
        super().__init__('painted_fusion_node')

        # ---- Tunable parameters (live via `ros2 param set`) ----
        self.declare_parameter('score_boost', 1.30)      # multiplied on class agreement
        self.declare_parameter('score_suppress', 0.50)   # multiplied on no_2d
        self.declare_parameter('publish_markers', True)

        # ---- State ----
        self.latest_yolo = None            # most recent YOLO JSON dict
        self.latest_yolo_time = 0.0        # wall time we got it
        self.frame_count = 0
        self.last_log_time = time.time()

        # ---- ROS plumbing ----
        self.yolo_sub = self.create_subscription(
            String, '/objects/results', self.yolo_cb, 10)
        self.pp_sub = self.create_subscription(
            String, '/pointpillars/detections', self.pp_cb, 10)

        self.fused_pub = self.create_publisher(
            String, '/objects/fused_3d', 10)
        self.marker_pub = self.create_publisher(
            MarkerArray, '/objects/fused_3d_markers', 10)

        self.get_logger().info('painted_fusion_node ready')
        self.get_logger().info(
            '  subs: /objects/results + /pointpillars/detections')
        self.get_logger().info(
            '  pubs: /objects/fused_3d + /objects/fused_3d_markers')

    # ------------------------------------------------------------------
    def yolo_cb(self, msg: String):
        try:
            self.latest_yolo = json.loads(msg.data)
            self.latest_yolo_time = time.time()
        except json.JSONDecodeError as e:
            self.get_logger().warn(f'Bad YOLO JSON: {e}')

    # ------------------------------------------------------------------
    def pp_cb(self, msg: String):
        try:
            pp = json.loads(msg.data)
        except json.JSONDecodeError as e:
            self.get_logger().warn(f'Bad PP JSON: {e}')
            return

        boost = self.get_parameter('score_boost').value
        suppress = self.get_parameter('score_suppress').value
        publish_markers = self.get_parameter('publish_markers').value

        yolo_dets = (self.latest_yolo.get('detections', [])
                     if self.latest_yolo else None)
        yolo_age_ms = ((time.time() - self.latest_yolo_time) * 1000.0
                       if self.latest_yolo else -1.0)

        fused = []
        status_counter = Counter()

        for det in pp.get('detections', []):
            x, y, z = det['pos']
            score_orig = det['score']
            kitti_class = det['class_name']

            # Project 3D center to image
            u, v, Zc = project_to_image((x, y, z))
            in_image = (u is not None and
                        0 <= u < IMG_W and 0 <= v < IMG_H)

            matched_2d = None
            match_status = None

            if yolo_dets is None:
                match_status = 'no_yolo'
                score_fused = score_orig
            elif not in_image:
                match_status = 'off_image'
                score_fused = score_orig
            else:
                # Find smallest 2D box that contains (u, v) — area-min tie-break
                best = None
                for yd in yolo_dets:
                    x1, y1, x2, y2 = yd['bbox']
                    if x1 <= u <= x2 and y1 <= v <= y2:
                        area = (x2 - x1) * (y2 - y1)
                        if best is None or area < best[1]:
                            best = (yd, area)

                if best is None:
                    match_status = 'no_2d'
                    score_fused = score_orig * suppress
                else:
                    matched_2d = best[0]
                    expected = COCO_TO_KITTI.get(matched_2d['class_name'])
                    if expected == kitti_class:
                        match_status = 'agree'
                        score_fused = min(1.0, score_orig * boost)
                    else:
                        match_status = 'disagree'
                        score_fused = score_orig

            status_counter[match_status] += 1

            fused.append({
                'class': kitti_class,
                'score_original': float(score_orig),
                'score_fused': float(score_fused),
                'match_status': match_status,
                'pos': det['pos'],
                'dims': det['dims'],
                'yaw': det['yaw'],
                'distance_m': det['distance_m'],
                'matched_2d_class': (matched_2d['class_name']
                                     if matched_2d else None),
                'matched_2d_bbox': (matched_2d['bbox']
                                    if matched_2d else None),
                'matched_2d_confidence': (matched_2d['confidence']
                                          if matched_2d else None),
                'image_projection': ([float(u), float(v)]
                                     if u is not None else None),
            })

        # ---- Publish JSON ----
        out = String()
        out.data = json.dumps({
            'stamp_sec': pp.get('stamp_sec', 0),
            'stamp_nanosec': pp.get('stamp_nanosec', 0),
            'frame_id': pp.get('frame_id', 'rslidar'),
            'count': len(fused),
            'yolo_age_ms': round(yolo_age_ms, 1),
            'status_counts': dict(status_counter),
            'detections': fused,
        })
        self.fused_pub.publish(out)

        # ---- Publish markers ----
        if publish_markers:
            self.publish_markers(fused, pp.get('frame_id', 'rslidar'))

        # ---- Periodic log ----
        self.frame_count += 1
        now = time.time()
        if now - self.last_log_time >= 5.0:
            fps = self.frame_count / (now - self.last_log_time)
            breakdown = ' '.join(f'{k}:{v}' for k, v in status_counter.items())
            self.get_logger().info(
                f'FUSE FPS: {fps:.1f} | det: {len(fused)} | '
                f'{breakdown} | yolo_age: {yolo_age_ms:.0f}ms')
            self.frame_count = 0
            self.last_log_time = now

    # ------------------------------------------------------------------
    def publish_markers(self, fused, frame_id):
        ma = MarkerArray()
        # Clear previous frame
        clr = Marker()
        clr.header.frame_id = frame_id
        clr.action = Marker.DELETEALL
        ma.markers.append(clr)

        for i, d in enumerate(fused):
            x, y, z = d['pos']
            dx, dy, dz = d['dims']
            yaw = d['yaw']
            color = STATUS_COLORS.get(d['match_status'],
                                      (1.0, 1.0, 1.0, 0.8))

            # Box edges
            line = Marker()
            line.header.frame_id = frame_id
            line.ns = 'fused_boxes'
            line.id = i
            line.type = Marker.LINE_LIST
            line.action = Marker.ADD
            line.scale.x = 0.10  # thicker than PP markers, so fused overlay is visible
            line.color = ColorRGBA(r=color[0], g=color[1],
                                   b=color[2], a=color[3])
            line.lifetime.sec = 0
            line.lifetime.nanosec = 200_000_000
            corners = box_corners_world(x, y, z, dx, dy, dz, yaw)
            for a, b in _EDGES:
                line.points.append(Point(x=float(corners[a, 0]),
                                         y=float(corners[a, 1]),
                                         z=float(corners[a, 2])))
                line.points.append(Point(x=float(corners[b, 0]),
                                         y=float(corners[b, 1]),
                                         z=float(corners[b, 2])))
            ma.markers.append(line)

            # Label: class score_fused (status) [matched_class]
            label = f"{d['class']} {d['score_fused']:.2f} ({d['match_status']})"
            if d['matched_2d_class']:
                label += f" [{d['matched_2d_class']}]"
            txt = Marker()
            txt.header.frame_id = frame_id
            txt.ns = 'fused_labels'
            txt.id = i
            txt.type = Marker.TEXT_VIEW_FACING
            txt.action = Marker.ADD
            txt.pose.position.x = float(x)
            txt.pose.position.y = float(y)
            txt.pose.position.z = float(z + dz / 2 + 0.8)
            txt.scale.z = 0.9
            txt.color = ColorRGBA(r=1.0, g=1.0, b=1.0, a=1.0)
            txt.text = label
            txt.lifetime.sec = 0
            txt.lifetime.nanosec = 200_000_000
            ma.markers.append(txt)

        self.marker_pub.publish(ma)


def main(args=None):
    rclpy.init(args=args)
    node = PaintedFusionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
