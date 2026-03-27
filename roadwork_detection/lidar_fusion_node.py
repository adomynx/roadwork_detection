import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Float64
from sensor_msgs.msg import PointCloud2, CameraInfo
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
import numpy as np
import json
import struct


class LidarFusionNode(Node):
    def __init__(self):
        super().__init__('lidar_fusion_node')

        # Declare parameters
        self.declare_parameter('lidar_to_cam_z_offset', 1.011)
        self.declare_parameter('alpha_d', 0.15)
        self.declare_parameter('default_fx', 1000.0)
        self.declare_parameter('default_fy', 1000.0)
        self.declare_parameter('default_cx', 640.0)
        self.declare_parameter('default_cy', 540.0)
        self.declare_parameter('min_distance', 0.5)
        self.declare_parameter('max_distance', 80.0)
        self.declare_parameter('bbox_expand_px', 10)

        # Get parameters
        self.z_offset = self.get_parameter('lidar_to_cam_z_offset').value
        self.alpha_d = self.get_parameter('alpha_d').value
        self.min_dist = self.get_parameter('min_distance').value
        self.max_dist = self.get_parameter('max_distance').value
        self.bbox_expand = self.get_parameter('bbox_expand_px').value

        # Camera intrinsics (updated from camera_info)
        self.f_x = self.get_parameter('default_fx').value
        self.f_y = self.get_parameter('default_fy').value
        self.c_x = self.get_parameter('default_cx').value
        self.c_y = self.get_parameter('default_cy').value
        self.camera_info_received = False

        # Full transform: rslidar → zed_left_camera_optical_frame
        # Chain from TF static:
        #   rslidar → nemo → zed_camera_link → zed_camera_center
        #   → zed_left_camera_frame → zed_left_camera_optical_frame
        #
        # Translation (all identity rotations until optical):
        #   t = (-0.01, 0.06, 1.011)  [x=baseline, y=height offset, z=fwd offset]
        #
        # Optical rotation (quat x=0.5, y=-0.5, z=0.5, w=-0.5) → R^T:
        #   [[ 0, -1,  0],
        #    [ 0,  0, -1],
        #    [ 1,  0,  0]]
        #
        # T_full = [R^T | R^T * t]
        self.T = np.array([
            [ 0.0, -1.0,  0.0, -0.060 ],
            [ 0.0,  0.0, -1.0, -1.011 ],
            [ 1.0,  0.0,  0.0, -0.010 ],
            [ 0.0,  0.0,  0.0,  1.0   ]
        ])

        # State
        self.latest_detections = []
        self.latest_lidar_points = None
        self.d_min_smoothed = None
        self.frame_count = 0

        # QoS
        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=5
        )

        # Subscribers
        self.lidar_sub = self.create_subscription(
            PointCloud2,
            '/rslidar_points',
            self.lidar_callback,
            sensor_qos
        )
        self.results_sub = self.create_subscription(
            String,
            '/detection/results',
            self.results_callback,
            10
        )
        self.camera_info_sub = self.create_subscription(
            CameraInfo,
            '/zed/zed_node/left_raw/camera_info',
            self.camera_info_callback,
            sensor_qos
        )

        # Publishers
        self.distance_pub = self.create_publisher(Float64, '/metrics/distance', 10)
        self.distance_detail_pub = self.create_publisher(String, '/metrics/distance_detail', 10)

        self.get_logger().info('LiDAR Fusion Node started!')
        self.get_logger().info(f'Z offset: {self.z_offset}m | alpha_d: {self.alpha_d}')
        self.get_logger().info(f'Distance range: {self.min_dist}m - {self.max_dist}m')

    def camera_info_callback(self, msg):
        if not self.camera_info_received:
            # K = [fx, 0, cx, 0, fy, cy, 0, 0, 1]
            if len(msg.k) == 9 and msg.k[0] > 0:
                self.f_x = msg.k[0]
                self.f_y = msg.k[4]
                self.c_x = msg.k[2]
                self.c_y = msg.k[5]
                self.camera_info_received = True
                self.get_logger().info(
                    f'Camera intrinsics received: '
                    f'fx={self.f_x:.1f} fy={self.f_y:.1f} '
                    f'cx={self.c_x:.1f} cy={self.c_y:.1f}'
                )

    def lidar_callback(self, msg):
        """Parse PointCloud2 and store XYZ points."""
        try:
            points = self._parse_pointcloud2(msg)
            self.latest_lidar_points = points
        except Exception as e:
            self.get_logger().error(f'LiDAR parse error: {str(e)}')

    def results_callback(self, msg):
        """On new detections, fuse with latest LiDAR scan."""
        self.frame_count += 1
        try:
            data = json.loads(msg.data)
            detections = data.get('detections', [])

            if len(detections) == 0:
                # Decay smoothed distance toward zero
                if self.d_min_smoothed is not None:
                    self.d_min_smoothed = (1 - self.alpha_d) * self.d_min_smoothed
                    if self.d_min_smoothed < 0.1:
                        self.d_min_smoothed = None
                    else:
                        dist_msg = Float64()
                        dist_msg.data = self.d_min_smoothed
                        self.distance_pub.publish(dist_msg)
                return

            if self.latest_lidar_points is None or len(self.latest_lidar_points) == 0:
                self.get_logger().info(
                    f'Frame {self.frame_count}: No LiDAR points yet, waiting...'
                )
                return

            # Step 1: Transform LiDAR points → camera frame
            pts_cam = self._transform_points(self.latest_lidar_points)

            # Step 2: Filter points in front of camera (Z > 0)
            front_mask = pts_cam[:, 2] > self.min_dist
            pts_cam = pts_cam[front_mask]

            if len(pts_cam) == 0:
                return

            # Step 3: Project to image pixels
            # u = fx * X/Z + cx,  v = fy * Y/Z + cy
            Z = pts_cam[:, 2]
            u = (self.f_x * pts_cam[:, 0] / Z + self.c_x).astype(int)
            v = (self.f_y * pts_cam[:, 1] / Z + self.c_y).astype(int)

            # Step 4: For each detection bbox, find min Z of points inside
            distances = []
            for det in detections:
                bbox = det['bbox']  # [x1, y1, x2, y2]
                x1 = bbox[0] - self.bbox_expand
                y1 = bbox[1] - self.bbox_expand
                x2 = bbox[2] + self.bbox_expand
                y2 = bbox[3] + self.bbox_expand

                # Points inside this bbox
                inside_mask = (u >= x1) & (u <= x2) & (v >= y1) & (v <= y2)
                pts_inside = Z[inside_mask]

                # Filter by max distance
                pts_inside = pts_inside[pts_inside <= self.max_dist]

                if len(pts_inside) == 0:
                    continue

                d_j = float(np.min(pts_inside))
                distances.append({
                    'class': det['class'],
                    'confidence': det['confidence'],
                    'distance': round(d_j, 2),
                    'num_points': int(np.sum(inside_mask))
                })

            if len(distances) == 0:
                self.get_logger().info(
                    f'Frame {self.frame_count}: {len(detections)} detections, '
                    f'no LiDAR points inside bboxes'
                )
                return

            # Step 5: Nearest distance
            d_min = min(d['distance'] for d in distances)

            # Step 6: EMA smoothing
            if self.d_min_smoothed is None:
                self.d_min_smoothed = d_min
            else:
                self.d_min_smoothed = (
                    self.alpha_d * d_min +
                    (1 - self.alpha_d) * self.d_min_smoothed
                )

            # Publish
            dist_msg = Float64()
            dist_msg.data = self.d_min_smoothed
            self.distance_pub.publish(dist_msg)

            # Publish detail
            detail = {
                'frame': self.frame_count,
                'source': 'lidar',
                'd_min_raw': round(d_min, 2),
                'd_min_smoothed': round(self.d_min_smoothed, 2),
                'num_valid_distances': len(distances),
                'all_distances': distances,
                'total_lidar_points': len(self.latest_lidar_points),
            }
            detail_msg = String()
            detail_msg.data = json.dumps(detail)
            self.distance_detail_pub.publish(detail_msg)

            self.get_logger().info(
                f'Frame {self.frame_count}: '
                f'd_min = {d_min:.1f}m | '
                f'd_smoothed = {self.d_min_smoothed:.1f}m | '
                f'objects: {len(distances)}'
            )

        except Exception as e:
            self.get_logger().error(f'Fusion error: {str(e)}')

    def _transform_points(self, points_xyz):
        """Apply T matrix: rslidar -> zed_camera_center."""
        # points_xyz: Nx3 numpy array
        N = len(points_xyz)
        # Homogeneous: Nx4
        ones = np.ones((N, 1))
        pts_h = np.hstack([points_xyz, ones])  # Nx4
        # Apply transform: (4x4) @ (4xN) -> (4xN) -> Nx4
        pts_cam = (self.T @ pts_h.T).T  # Nx4
        return pts_cam[:, :3]  # Nx3

    def _parse_pointcloud2(self, msg):
        """Parse ROS2 PointCloud2 message to Nx3 numpy array."""
        # Find field offsets
        fields = {f.name: f for f in msg.fields}
        if 'x' not in fields or 'y' not in fields or 'z' not in fields:
            return np.array([])

        x_off = fields['x'].offset
        y_off = fields['y'].offset
        z_off = fields['z'].offset
        point_step = msg.point_step
        data = msg.data

        n_points = msg.width * msg.height
        points = np.zeros((n_points, 3), dtype=np.float32)

        for i in range(n_points):
            base = i * point_step
            try:
                x = struct.unpack_from('f', data, base + x_off)[0]
                y = struct.unpack_from('f', data, base + y_off)[0]
                z = struct.unpack_from('f', data, base + z_off)[0]
                points[i] = [x, y, z]
            except Exception:
                continue

        # Remove NaN/Inf
        valid = np.isfinite(points).all(axis=1)
        return points[valid]


def main(args=None):
    rclpy.init(args=args)
    node = LidarFusionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
