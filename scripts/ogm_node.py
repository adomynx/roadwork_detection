#!/usr/bin/env python3
"""
ogm_node.py — Single-scan Occupancy Grid Mapping from LiDAR.

Subscribes:
  /rslidar_points   (sensor_msgs/PointCloud2)

Publishes:
  /ogm/grid         (nav_msgs/OccupancyGrid)

Algorithm — per scan, no accumulation across frames:
  1. Filter points by Z to drop ground returns and high overhangs.
  2. Convert to polar (range, bearing) in the LiDAR's XY plane.
  3. Discretize bearings into n_angle_bins (default 720 = 0.5°).
  4. For each bin, take the minimum range = closest obstacle.
  5. Ray-cast from origin to that endpoint: cells along the ray
     receive a free-space log-odds increment, the endpoint cell
     receives an occupied log-odds increment.
  6. Convert log-odds to probability and emit as OccupancyGrid:
     -1 = unknown, 0..100 = probability of occupancy (in percent).

Parameters (live, via `ros2 param set`):
  resolution     cell size in meters       (default 0.2)
  grid_size_m    side length in meters     (default 60.0)
  z_min, z_max   Z filter in LiDAR frame   (default -1.0, 2.0)
  n_angle_bins   angular discretization    (default 720)
  l_free, l_occ  log-odds increments       (default -0.4, +0.85)
"""

import time
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import PointCloud2
from nav_msgs.msg import OccupancyGrid
from geometry_msgs.msg import Pose, Point, Quaternion


def parse_pointcloud2(msg):
    """PointCloud2 -> Nx3 numpy (x, y, z) in LiDAR frame."""
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
    return np.stack([arr['x'].astype(np.float32),
                     arr['y'].astype(np.float32),
                     arr['z'].astype(np.float32)], axis=1)


class OGMNode(Node):
    def __init__(self):
        super().__init__('ogm_node')

        # ---- Tunable parameters (live via `ros2 param set`) ----
        self.declare_parameter('resolution', 0.2)
        self.declare_parameter('grid_size_m', 60.0)
        self.declare_parameter('z_min', -1.0)
        self.declare_parameter('z_max', 2.0)
        self.declare_parameter('n_angle_bins', 720)
        self.declare_parameter('l_free', -0.4)
        self.declare_parameter('l_occ', 0.85)
        self.declare_parameter('input_topic', '/rslidar_points')
        self.declare_parameter('output_topic', '/ogm/grid')

        input_topic = self.get_parameter('input_topic').value
        output_topic = self.get_parameter('output_topic').value

        # ---- ROS plumbing ----
        self.sub = self.create_subscription(
            PointCloud2, input_topic,
            self.pc_callback, qos_profile_sensor_data)
        self.grid_pub = self.create_publisher(
            OccupancyGrid, output_topic, 10)

        self.frame_count = 0
        self.last_log_time = time.time()

        self.get_logger().info(
            f'ogm_node ready — single-scan OGM | '
            f'subscribing {input_topic}, publishing {output_topic}')

    # ----------------------------------------------------------------------
    def pc_callback(self, msg):
        t0 = time.time()

        # Read parameters each frame for live tuning
        res = float(self.get_parameter('resolution').value)
        size_m = float(self.get_parameter('grid_size_m').value)
        z_min = float(self.get_parameter('z_min').value)
        z_max = float(self.get_parameter('z_max').value)
        n_bins = int(self.get_parameter('n_angle_bins').value)
        L_FREE = float(self.get_parameter('l_free').value)
        L_OCC = float(self.get_parameter('l_occ').value)

        n_cells = int(round(size_m / res))
        half = (n_cells * res) / 2.0
        origin_idx = n_cells // 2

        # ---- Parse points ----
        try:
            points = parse_pointcloud2(msg)
        except Exception as e:
            self.get_logger().warn(f'PointCloud parse failed: {e}')
            return
        if len(points) == 0:
            return

        # ---- Filter Z (drop ground returns and high overhangs) ----
        z = points[:, 2]
        m_z = (z > z_min) & (z < z_max)
        pts = points[m_z]
        if len(pts) == 0:
            self._publish_empty(msg, n_cells, res, half)
            return

        # ---- Polar coordinates ----
        x, y = pts[:, 0], pts[:, 1]
        r = np.hypot(x, y)
        theta = np.arctan2(y, x)

        # ---- Filter range (drop self-returns and out-of-grid hits) ----
        m_r = (r > 0.5) & (r < (half - res))
        r, theta = r[m_r], theta[m_r]
        if len(r) == 0:
            self._publish_empty(msg, n_cells, res, half)
            return

        # ---- Angular binning + scatter-min for closest obstacle per bin ----
        bin_idx = ((theta + np.pi) / (2 * np.pi) * n_bins).astype(np.int32) % n_bins
        min_r = np.full(n_bins, np.inf, dtype=np.float32)
        np.minimum.at(min_r, bin_idx, r)

        # ---- Vectorized ray-casting across all bins ----
        valid_bins = np.isfinite(min_r)
        n_valid = int(valid_bins.sum())
        if n_valid == 0:
            self._publish_empty(msg, n_cells, res, half)
            return

        min_r_v = min_r[valid_bins]
        bin_centers = -np.pi + (np.arange(n_bins) + 0.5) * (2 * np.pi / n_bins)
        cos_v = np.cos(bin_centers[valid_bins])
        sin_v = np.sin(bin_centers[valid_bins])

        # Step at half-cell resolution along the longest ray; shorter rays
        # share the same parametric grid but cover less distance.
        max_r = float(min_r_v.max())
        n_steps = max(2, int(np.ceil(max_r * 2.0 / res)))
        ts = np.linspace(0.0, 1.0, n_steps + 1, dtype=np.float32)  # (S,)

        # Parametric distance for each (bin, step) pair: (n_valid, S)
        t_dist = ts[None, :] * min_r_v[:, None]
        cell_x = t_dist * cos_v[:, None]
        cell_y = t_dist * sin_v[:, None]
        cell_i = np.floor((cell_x + half) / res).astype(np.int32)
        cell_j = np.floor((cell_y + half) / res).astype(np.int32)

        # In-bounds mask
        in_bounds = ((cell_i >= 0) & (cell_i < n_cells) &
                     (cell_j >= 0) & (cell_j < n_cells))

        # ---- Build log-odds grid ----
        grid_log = np.zeros((n_cells, n_cells), dtype=np.float32)

        # Free cells: every (bin, step) except the endpoint (last step)
        free_mask = in_bounds.copy()
        free_mask[:, -1] = False
        if free_mask.any():
            # Flatten then use bincount on linear index for fast scatter
            lin_idx = cell_j[free_mask] * n_cells + cell_i[free_mask]
            counts = np.bincount(lin_idx, minlength=n_cells * n_cells)
            grid_log += counts.reshape((n_cells, n_cells)).astype(np.float32) * L_FREE

        # Occupied cells: endpoint of each ray
        end_valid = in_bounds[:, -1]
        if end_valid.any():
            end_i = cell_i[end_valid, -1]
            end_j = cell_j[end_valid, -1]
            lin_idx = end_j * n_cells + end_i
            counts = np.bincount(lin_idx, minlength=n_cells * n_cells)
            grid_log += counts.reshape((n_cells, n_cells)).astype(np.float32) * L_OCC

        # ---- Log-odds -> probability -> OccupancyGrid format ----
        occ = np.full((n_cells, n_cells), -1, dtype=np.int8)
        visited = grid_log != 0
        if visited.any():
            # Clip log-odds to a safe range before sigmoid to avoid overflow
            # in exp() for cells hit by many free or many occupied observations.
            # The clip range maps to probabilities (~0.001, ~0.999) which is
            # plenty for an int8 OccupancyGrid.
            clipped = np.clip(grid_log[visited], -7.0, 7.0)
            prob = 1.0 / (1.0 + np.exp(-clipped))
            occ[visited] = np.clip(prob * 100.0, 0, 100).astype(np.int8)

        compute_ms = (time.time() - t0) * 1000.0

        # ---- Publish ----
        self._publish_grid(msg, occ, res, half)

        # ---- Periodic log ----
        self.frame_count += 1
        now = time.time()
        if now - self.last_log_time >= 5.0:
            fps = self.frame_count / (now - self.last_log_time)
            n_occ = int((occ >= 65).sum())
            n_free = int(((occ >= 0) & (occ < 35)).sum())
            n_unknown = int((occ < 0).sum())
            self.get_logger().info(
                f'OGM FPS: {fps:.1f} | compute: {compute_ms:.1f}ms | '
                f'cells occ:{n_occ} free:{n_free} unknown:{n_unknown}')
            self.frame_count = 0
            self.last_log_time = now

    # ----------------------------------------------------------------------
    def _publish_grid(self, header_src_msg, occ, res, half):
        msg = OccupancyGrid()
        # Keep the LiDAR frame_id for spatial alignment, but zero the stamp.
        # RViz's tf message filter on OccupancyGrid otherwise tries to align
        # the message timestamp against tf, and with rosbag playback there is
        # no /clock publishing rosbag-time tf, so every message gets dropped
        # with "queue is full". Zeroing the stamp tells RViz to use it as a
        # static map.
        msg.header.frame_id = header_src_msg.header.frame_id
        msg.header.stamp.sec = 0
        msg.header.stamp.nanosec = 0
        msg.info.resolution = float(res)
        msg.info.width = int(occ.shape[1])
        msg.info.height = int(occ.shape[0])
        # Origin = bottom-left corner of the grid in the LiDAR frame.
        msg.info.origin = Pose(
            position=Point(x=-float(half), y=-float(half), z=0.0),
            orientation=Quaternion(x=0.0, y=0.0, z=0.0, w=1.0))
        # nav_msgs/OccupancyGrid expects row-major flat int8.
        msg.data = occ.flatten().tolist()
        self.grid_pub.publish(msg)

    def _publish_empty(self, header_src_msg, n_cells, res, half):
        occ = np.full((n_cells, n_cells), -1, dtype=np.int8)
        self._publish_grid(header_src_msg, occ, res, half)


def main(args=None):
    rclpy.init(args=args)
    node = OGMNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
