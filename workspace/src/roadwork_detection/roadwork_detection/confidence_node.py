import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Float64
import json


class ConfidenceNode(Node):
    def __init__(self):
        super().__init__('confidence_node')

        # Declare parameters
        self.declare_parameter('lambda_factor', 0.5)
        self.declare_parameter('n_max', 12)
        self.declare_parameter('alpha_c', 0.15)

        # Get parameters
        self.lambda_factor = self.get_parameter('lambda_factor').value
        self.n_max = self.get_parameter('n_max').value
        self.alpha_c = self.get_parameter('alpha_c').value

        # EMA state
        self.c_frame_smoothed = None

        # Frame counter
        self.frame_count = 0

        # Subscriber
        self.results_sub = self.create_subscription(
            String,
            '/detection/results',
            self.results_callback,
            10
        )

        # Publishers
        self.confidence_pub = self.create_publisher(Float64, '/metrics/confidence', 10)
        self.confidence_detail_pub = self.create_publisher(String, '/metrics/confidence_detail', 10)

        self.get_logger().info('Confidence Node started!')
        self.get_logger().info(
            f'lambda: {self.lambda_factor} | N_max: {self.n_max} | alpha_c: {self.alpha_c}'
        )

    def results_callback(self, msg):
        self.frame_count += 1

        try:
            data = json.loads(msg.data)
            detections = data.get('detections', [])
            N = len(detections)

            if N == 0:
                if self.c_frame_smoothed is not None:
                    self.c_frame_smoothed = (1 - self.alpha_c) * self.c_frame_smoothed
                    if self.c_frame_smoothed < 0.03:
                        self.c_frame_smoothed = None

                    if self.c_frame_smoothed is not None:
                        conf_msg = Float64()
                        conf_msg.data = self.c_frame_smoothed
                        self.confidence_pub.publish(conf_msg)

                return

            # Calculate C_pres: area-weighted average of confidences
            # C_pres = sum(s_j * A_j) / sum(A_j)
            total_weighted_conf = 0.0
            total_area = 0.0

            for det in detections:
                bbox = det['bbox']  # [x1, y1, x2, y2]
                x_tl, y_tl, x_br, y_br = bbox
                s_j = det['confidence']

                # Bounding box area: A_j = (x_br - x_tl)(y_br - y_tl)
                A_j = (x_br - x_tl) * (y_br - y_tl)

                if A_j <= 0:
                    continue

                total_weighted_conf += s_j * A_j
                total_area += A_j

            if total_area == 0:
                return

            C_pres = total_weighted_conf / total_area

            # Detection density factor: C_count = min(1, N / N_max)
            C_count = min(1.0, N / self.n_max)

            # Combined: C_frame = lambda * C_pres + (1 - lambda) * C_count
            C_frame = self.lambda_factor * C_pres + (1 - self.lambda_factor) * C_count

            # EMA smoothing: C_hat(t) = alpha_c * C_frame(t) + (1 - alpha_c) * C_hat(t-1)
            if self.c_frame_smoothed is None:
                self.c_frame_smoothed = C_frame
            else:
                self.c_frame_smoothed = (
                    self.alpha_c * C_frame +
                    (1 - self.alpha_c) * self.c_frame_smoothed
                )

            # Publish smoothed confidence
            conf_msg = Float64()
            conf_msg.data = self.c_frame_smoothed
            self.confidence_pub.publish(conf_msg)

            # Publish detailed info
            detail = {
                'frame': self.frame_count,
                'num_detections': N,
                'C_pres': round(C_pres, 4),
                'C_count': round(C_count, 4),
                'C_frame_raw': round(C_frame, 4),
                'C_frame_smoothed': round(self.c_frame_smoothed, 4)
            }
            detail_msg = String()
            detail_msg.data = json.dumps(detail)
            self.confidence_detail_pub.publish(detail_msg)

            # Log
            self.get_logger().info(
                f'Frame {self.frame_count}: C_pres = {C_pres*100:.1f}% | '
                f'C_count = {C_count*100:.1f}% | '
                f'C_frame = {C_frame*100:.1f}% | '
                f'C_smoothed = {self.c_frame_smoothed*100:.1f}%'
            )

        except Exception as e:
            self.get_logger().error(f'Error in confidence calculation: {str(e)}')


def main(args=None):
    rclpy.init(args=args)
    node = ConfidenceNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
