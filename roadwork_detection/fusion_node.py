import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo
from std_msgs.msg import String, Float64
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
import numpy as np
import json


class FusionNode(Node):
    def __init__(self):
        super().__init__('fusion_node')

        # Parameters
        self.declare_parameter('alpha_d', 0.15)
        self.declare_parameter('projection_tolerance', 50)

        self.alpha_d = self.get_parameter('alpha_d').value
        self.proj_tol = self.get_parameter('projection_tolerance').value

        # Camera intrinsics
        self.f_x = 266.9
        self.f_y = 266.9
        self.c_x = 314.0
        self.c_y = 182.8
        self.camera_info_received = False

        # TF transform: rslidar -> zed_left_camera_optical_frame
        self.T = np.array([
            [ 0.0, -1.0,  0.0, -0.060],
            [ 0.0,  0.0, -1.0, -1.011],
            [ 1.0,  0.0,  0.0, -0.010],
            [ 0.0,  0.0,  0.0,  1.0  ]
        ])

        # State
        self.latest_obstacles = None
        self.latest_detections = None
        self.d_min_smoothed = None
        self.frame_count = 0

        # QoS
        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=5
        )

        # Subscribers
        self.obstacles_sub = self.create_subscription(
            String, '/lidar/obstacles', self.obstacles_callback, 10)
        self.detections_sub = self.create_subscription(
            String, '/detection/results', self.detections_callback, 10)
        self.camera_info_sub = self.create_subscription(
            CameraInfo, '/zed/zed_node/left_raw/camera_info',
            self.camera_info_callback, sensor_qos)

        # Publishers
        self.fused_pub = self.create_publisher(String, '/fusion/results', 10)
        self.distance_pub = self.create_publisher(Float64, '/metrics/distance', 10)
        self.distance_detail_pub = self.create_publisher(String, '/metrics/distance_detail', 10)

        self.get_logger().info('Fusion Node started!')
        self.get_logger().info(f'alpha_d: {self.alpha_d} | projection_tolerance: {self.proj_tol}px')

    def camera_info_callback(self, msg):
        if not self.camera_info_received:
            if len(msg.k) == 9 and msg.k[0] > 0:
                self.f_x = msg.k[0]
                self.f_y = msg.k[4]
                self.c_x = msg.k[2]
                self.c_y = msg.k[5]
                self.camera_info_received = True
                self.get_logger().info(
                    f'Camera intrinsics: fx={self.f_x:.1f} fy={self.f_y:.1f} '
                    f'cx={self.c_x:.1f} cy={self.c_y:.1f}'
                )

    def obstacles_callback(self, msg):
        try:
            self.latest_obstacles = json.loads(msg.data)
        except Exception as e:
            self.get_logger().error(f'Error parsing obstacles: {str(e)}')

    def detections_callback(self, msg):
        self.frame_count += 1
        try:
            data = json.loads(msg.data)
            detections = data.get('detections', [])

            if len(detections) == 0:
                # Decay
                if self.d_min_smoothed is not None:
                    self.d_min_smoothed = (1 - self.alpha_d) * self.d_min_smoothed
                    if self.d_min_smoothed < 0.1:
                        self.d_min_smoothed = None
                    else:
                        dist_msg = Float64()
                        dist_msg.data = self.d_min_smoothed
                        self.distance_pub.publish(dist_msg)
                return

            if self.latest_obstacles is None:
                self.get_logger().info(
                    f'Frame {self.frame_count}: Waiting for LiDAR obstacles...'
                )
                return

            obstacles = self.latest_obstacles.get('obstacles', [])
            if len(obstacles) == 0:
                return

            # Step 1: Project each obstacle centroid to camera image
            fused_results = []
            for det in detections:
                bbox = det['bbox']  # [x1, y1, x2, y2]
                bbox_cx = (bbox[0] + bbox[2]) / 2
                bbox_cy = (bbox[1] + bbox[3]) / 2

                best_match = None
                best_dist_px = float('inf')

                for obs in obstacles:
                    centroid = np.array(obs['centroid'])

                    # Transform centroid to camera frame
                    pt_h = np.array([centroid[0], centroid[1], centroid[2], 1.0])
                    pt_cam = self.T @ pt_h

                    # Skip if behind camera
                    if pt_cam[2] <= 0.5:
                        continue

                    # Project to image
                    u = self.f_x * pt_cam[0] / pt_cam[2] + self.c_x
                    v = self.f_y * pt_cam[1] / pt_cam[2] + self.c_y

                    # Check if projection falls inside or near the detection bbox
                    inside_x = bbox[0] - self.proj_tol <= u <= bbox[2] + self.proj_tol
                    inside_y = bbox[1] - self.proj_tol <= v <= bbox[3] + self.proj_tol

                    if inside_x and inside_y:
                        # Distance from bbox center to projection
                        dist_px = np.sqrt((u - bbox_cx)**2 + (v - bbox_cy)**2)
                        if dist_px < best_dist_px:
                            best_dist_px = dist_px
                            best_match = obs

                if best_match is not None:
                    fused_results.append({
                        'class': det['class'],
                        'class_id': det['class_id'],
                        'confidence': det['confidence'],
                        'bbox': det['bbox'],
                        'distance_3d': best_match['distance'],
                        'centroid_3d': best_match['centroid'],
                        'size_3d': best_match['size'],
                        'num_lidar_points': best_match['num_points']
                    })

            # Publish fused results
            fused_msg = String()
            fused_msg.data = json.dumps({
                'frame': self.frame_count,
                'num_fused': len(fused_results),
                'num_detections': len(detections),
                'num_obstacles': len(obstacles),
                'fused': fused_results
            })
            self.fused_pub.publish(fused_msg)

            # Publish distance from fused results
            if len(fused_results) > 0:
                d_min = min(f['distance_3d'] for f in fused_results)

                if self.d_min_smoothed is None:
                    self.d_min_smoothed = d_min
                else:
                    self.d_min_smoothed = (
                        self.alpha_d * d_min +
                        (1 - self.alpha_d) * self.d_min_smoothed
                    )

                dist_msg = Float64()
                dist_msg.data = self.d_min_smoothed
                self.distance_pub.publish(dist_msg)

                # Detail
                detail = {
                    'frame': self.frame_count,
                    'source': 'lidar_fusion',
                    'd_min_raw': round(d_min, 2),
                    'd_min_smoothed': round(self.d_min_smoothed, 2),
                    'num_fused': len(fused_results),
                    'all_distances': [
                        {'class': f['class'], 'distance': f['distance_3d']}
                        for f in fused_results
                    ]
                }
                detail_msg = String()
                detail_msg.data = json.dumps(detail)
                self.distance_detail_pub.publish(detail_msg)

                self.get_logger().info(
                    f'Frame {self.frame_count}: '
                    f'{len(fused_results)}/{len(detections)} fused | '
                    f'd_min = {d_min:.1f}m | '
                    f'd_smoothed = {self.d_min_smoothed:.1f}m'
                )
            else:
                self.get_logger().info(
                    f'Frame {self.frame_count}: '
                    f'{len(detections)} detections, '
                    f'{len(obstacles)} obstacles, no matches'
                )

        except Exception as e:
            self.get_logger().error(f'Fusion error: {str(e)}')


def main(args=None):
    rclpy.init(args=args)
    node = FusionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
