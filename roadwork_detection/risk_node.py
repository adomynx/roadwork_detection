import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Float64
import json


DYNAMIC_CLASSES = {
    0: 'Police Officer',
    1: 'Police Vehicle',
    7: 'Work Vehicle',
    13: 'Worker'
}


class RiskNode(Node):
    def __init__(self):
        super().__init__('risk_node')

        # Declare parameters
        self.declare_parameter('d_near', 5.0)
        self.declare_parameter('d_far', 30.0)
        self.declare_parameter('w_c', 0.5)
        self.declare_parameter('w_d', 0.5)
        self.declare_parameter('alpha_r', 0.20)

        # Get parameters
        self.d_near = self.get_parameter('d_near').value
        self.d_far = self.get_parameter('d_far').value
        self.w_c = self.get_parameter('w_c').value
        self.w_d = self.get_parameter('w_d').value
        self.alpha_r = self.get_parameter('alpha_r').value

        # State from other nodes
        self.latest_distance = None
        self.latest_confidence = None
        self.latest_detections = None

        # EMA state
        self.r_frame_smoothed = None

        # Frame counter
        self.frame_count = 0

        # Subscribers
        self.distance_sub = self.create_subscription(
            Float64,
            '/metrics/distance',
            self.distance_callback,
            10
        )
        self.confidence_sub = self.create_subscription(
            Float64,
            '/metrics/confidence',
            self.confidence_callback,
            10
        )
        self.results_sub = self.create_subscription(
            String,
            '/detection/results',
            self.results_callback,
            10
        )

        # Publishers
        self.risk_pub = self.create_publisher(Float64, '/metrics/risk', 10)
        self.risk_detail_pub = self.create_publisher(String, '/metrics/risk_detail', 10)

        self.get_logger().info('Risk Node started!')
        self.get_logger().info(
            f'd_near: {self.d_near}m | d_far: {self.d_far}m | '
            f'w_c: {self.w_c} | w_d: {self.w_d} | alpha_r: {self.alpha_r}'
        )

    def distance_callback(self, msg):
        self.latest_distance = msg.data

    def confidence_callback(self, msg):
        self.latest_confidence = msg.data

    def results_callback(self, msg):
        self.frame_count += 1

        try:
            data = json.loads(msg.data)
            detections = data.get('detections', [])
            N_tot = len(detections)

            if N_tot == 0:
                if self.r_frame_smoothed is not None:
                    self.r_frame_smoothed = (1 - self.alpha_r) * self.r_frame_smoothed
                    if self.r_frame_smoothed < 0.01:
                        self.r_frame_smoothed = None

                    if self.r_frame_smoothed is not None:
                        risk_msg = Float64()
                        risk_msg.data = self.r_frame_smoothed
                        self.risk_pub.publish(risk_msg)

                return

            # Wait until we have data from both other nodes
            if self.latest_distance is None or self.latest_confidence is None:
                self.get_logger().info(
                    f'Frame {self.frame_count}: Waiting for distance and confidence data...'
                )
                return

            d_min = self.latest_distance
            C_frame = self.latest_confidence

            # ---- Distance Risk (R_d) ----
            # R_d = 1 if d_min <= d_near
            # R_d = 0 if d_min >= d_far
            # R_d = (d_far - d_min) / (d_far - d_near) otherwise
            if d_min <= self.d_near:
                R_d = 1.0
            elif d_min >= self.d_far:
                R_d = 0.0
            else:
                R_d = (self.d_far - d_min) / (self.d_far - self.d_near)

            # ---- Dynamic Object Ratio (R_dyn) ----
            # R_dyn = N_dyn / N_tot
            N_dyn = 0
            for det in detections:
                cls_id = det.get('class_id', -1)
                if cls_id in DYNAMIC_CLASSES:
                    N_dyn += 1

            R_dyn = N_dyn / N_tot if N_tot > 0 else 0.0

            # ---- Contextual Risk (R_c) ----
            # R_c = (w_c * C_frame + w_d * R_dyn) / (w_c + w_d)
            R_c = (self.w_c * C_frame + self.w_d * R_dyn) / (self.w_c + self.w_d)

            # ---- Final Risk (R_frame) ----
            # R_frame = max(R_d, R_c)
            R_frame = max(R_d, R_c)

            # EMA smoothing: R_hat(t) = alpha_r * R_frame(t) + (1 - alpha_r) * R_hat(t-1)
            if self.r_frame_smoothed is None:
                self.r_frame_smoothed = R_frame
            else:
                self.r_frame_smoothed = (
                    self.alpha_r * R_frame +
                    (1 - self.alpha_r) * self.r_frame_smoothed
                )

            # Publish smoothed risk
            risk_msg = Float64()
            risk_msg.data = self.r_frame_smoothed
            self.risk_pub.publish(risk_msg)

            # Publish detailed info
            detail = {
                'frame': self.frame_count,
                'd_min': round(d_min, 2),
                'C_frame': round(C_frame, 4),
                'N_tot': N_tot,
                'N_dyn': N_dyn,
                'R_d': round(R_d, 4),
                'R_dyn': round(R_dyn, 4),
                'R_c': round(R_c, 4),
                'R_frame_raw': round(R_frame, 4),
                'R_frame_smoothed': round(self.r_frame_smoothed, 4)
            }
            detail_msg = String()
            detail_msg.data = json.dumps(detail)
            self.risk_detail_pub.publish(detail_msg)

            # Log
            self.get_logger().info(
                f'Frame {self.frame_count}: R_d = {R_d*100:.1f}% | '
                f'R_dyn = {R_dyn*100:.1f}% | R_c = {R_c*100:.1f}% | '
                f'R_frame = {R_frame*100:.1f}% | '
                f'R_smoothed = {self.r_frame_smoothed*100:.1f}%'
            )

        except Exception as e:
            self.get_logger().error(f'Error in risk calculation: {str(e)}')


def main(args=None):
    rclpy.init(args=args)
    node = RiskNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
