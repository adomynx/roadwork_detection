import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Float64
from sensor_msgs.msg import CameraInfo
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
import json


class DistanceNode(Node):
    def __init__(self):
        super().__init__('distance_node')

        # Declare parameters
        self.declare_parameter('camera_height', 1.5)
        self.declare_parameter('alpha_d', 0.15)
        self.declare_parameter('default_fy', 1000.0)
        self.declare_parameter('default_cy', 930.0)

        # Get parameters
        self.h_c = self.get_parameter('camera_height').value
        self.alpha_d = self.get_parameter('alpha_d').value
        self.default_fy = self.get_parameter('default_fy').value
        self.default_cy = self.get_parameter('default_cy').value

        # Camera intrinsics (updated from camera_info if available)
        self.f_y = self.default_fy
        self.c_y = self.default_cy

        # EMA state
        self.d_min_smoothed = None

        # Frame counter
        self.frame_count = 0

        # QoS for camera_info
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )

        # Subscribers
        self.results_sub = self.create_subscription(
            String,
            '/detection/results',
            self.results_callback,
            10
        )
        self.info_sub = self.create_subscription(
            CameraInfo,
            '/camera/camera_info',
            self.camera_info_callback,
            qos_profile
        )

        # Publishers
        self.distance_pub = self.create_publisher(Float64, '/metrics/distance', 10)
        self.distance_detail_pub = self.create_publisher(String, '/metrics/distance_detail', 10)

        self.get_logger().info('Distance Node started!')
        self.get_logger().info(f'Camera height: {self.h_c}m | alpha_d: {self.alpha_d}')

    def camera_info_callback(self, msg):
        # Extract focal length (f_y) and principal point (c_y) from camera_info
        # K matrix: [fx, 0, cx, 0, fy, cy, 0, 0, 1]
        if len(msg.k) >= 6:
            self.f_y = msg.k[4]  # fy
            self.c_y = msg.k[5]  # cy

    def results_callback(self, msg):
        self.frame_count += 1

        try:
            data = json.loads(msg.data)
            detections = data.get('detections', [])

            if len(detections) == 0:
                # Actively decay smoothed value toward zero
                if self.d_min_smoothed is not None:
                    self.d_min_smoothed = (1 - self.alpha_d) * self.d_min_smoothed
                    if self.d_min_smoothed < 0.1:
                        self.d_min_smoothed = None

                    # Publish decaying value
                    if self.d_min_smoothed is not None:
                        dist_msg = Float64()
                        dist_msg.data = self.d_min_smoothed
                        self.distance_pub.publish(dist_msg)

                return

            # Calculate distance for each detection
            distances = []
            for det in detections:
                bbox = det['bbox']  # [x1, y1, x2, y2]
                x_tl, y_tl, x_br, y_br = bbox

                # Bottom edge midpoint
                v_j = y_br

                # Reject invalid detections (at or above horizon)
                if v_j <= self.c_y:
                    continue

                # Distance formula from paper: d_j = (h_c * f_y) / (v_j - c_y)
                d_j = (self.h_c * self.f_y) / (v_j - self.c_y)
                distances.append({
                    'class': det['class'],
                    'confidence': det['confidence'],
                    'distance': round(d_j, 2)
                })

            if len(distances) == 0:
                return

            # Nearest distance: d_min = min(d_j)
            d_min = min(d['distance'] for d in distances)

            # EMA smoothing: d_hat(t) = alpha_d * d_min(t) + (1 - alpha_d) * d_hat(t-1)
            if self.d_min_smoothed is None:
                self.d_min_smoothed = d_min
            else:
                self.d_min_smoothed = (
                    self.alpha_d * d_min +
                    (1 - self.alpha_d) * self.d_min_smoothed
                )

            # Publish smoothed distance
            dist_msg = Float64()
            dist_msg.data = self.d_min_smoothed
            self.distance_pub.publish(dist_msg)

            # Publish detailed info
            detail = {
                'frame': self.frame_count,
                'd_min_raw': round(d_min, 2),
                'd_min_smoothed': round(self.d_min_smoothed, 2),
                'num_valid_distances': len(distances),
                'all_distances': distances
            }
            detail_msg = String()
            detail_msg.data = json.dumps(detail)
            self.distance_detail_pub.publish(detail_msg)

            # Log
            self.get_logger().info(
                f'Frame {self.frame_count}: d_min = {d_min:.1f}m | '
                f'd_smoothed = {self.d_min_smoothed:.1f}m | '
                f'objects: {len(distances)}'
            )

        except Exception as e:
            self.get_logger().error(f'Error in distance calculation: {str(e)}')


def main(args=None):
    rclpy.init(args=args)
    node = DistanceNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
