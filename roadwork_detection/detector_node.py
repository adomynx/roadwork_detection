import time
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from std_msgs.msg import String, Float64
from cv_bridge import CvBridge
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
import cv2
import json
import os
import random
import numpy as np
from datetime import datetime
from ultralytics import YOLO


class RoadworkDetectorNode(Node):
    def __init__(self):
        super().__init__('roadwork_detector_node')

        # Output folder for saving detections
        self.output_dir = os.path.expanduser('~/roadwork_project/detection_results')
        os.makedirs(self.output_dir, exist_ok=True)
        self.get_logger().info(f'Saving detections to: {self.output_dir}')

        # Declare parameters
        self.declare_parameter('model_path', '')
        self.declare_parameter('confidence_threshold', 0.5)
        self.declare_parameter('resize_width', 640)
        self.declare_parameter('resize_height', 480)
        self.declare_parameter('target_classes', [0, 1, 2, 3, 5, 7, 9, 11])

        # Get parameters
        model_path = self.get_parameter('model_path').value
        self.conf_threshold = self.get_parameter('confidence_threshold').value
        self.resize_w = self.get_parameter('resize_width').value
        self.resize_h = self.get_parameter('resize_height').value
        self.target_classes = self.get_parameter('target_classes').value

        # Load YOLOv8 TensorRT model
        if not model_path or not os.path.exists(model_path):
            self.get_logger().error(f'Model not found at: {model_path}')
            raise FileNotFoundError(f'Model not found at: {model_path}')

        self.get_logger().info(f'Loading YOLO model from: {model_path}')
        self.model = YOLO(model_path, task='detect')
        self.get_logger().info('Model loaded successfully!')

        # Warm up the model
        dummy = np.zeros((self.resize_h, self.resize_w, 3), dtype=np.uint8)
        self.model.predict(source=dummy, conf=0.5, verbose=False)
        self.get_logger().info('Model warm-up complete!')

        # CV Bridge
        self.bridge = CvBridge()

        # Store camera info
        self.camera_info = None

        # QoS profile to match rosbag
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )

        # Subscribers
        self.image_sub = self.create_subscription(
            Image,
            '/camera/image_raw',
            self.image_callback,
            10
        )
        self.info_sub = self.create_subscription(
            CameraInfo,
            '/camera/camera_info',
            self.camera_info_callback,
            qos_profile
        )

        # Publishers
        self.annotated_pub = self.create_publisher(Image, '/detection/annotated_image', 10)
        self.results_pub = self.create_publisher(String, '/detection/results', 10)

        # Subscribe to metrics for overlay
        self.latest_distance = None
        self.latest_confidence = None
        self.latest_risk = None

        self.metric_dist_sub = self.create_subscription(
            Float64, '/metrics/distance', self.dist_metric_callback, 10)
        self.metric_conf_sub = self.create_subscription(
            Float64, '/metrics/confidence', self.conf_metric_callback, 10)
        self.metric_risk_sub = self.create_subscription(
            Float64, '/metrics/risk', self.risk_metric_callback, 10)

        # Frame counter
        self.frame_count = 0

        # ROADWork dataset classes (17 classes)
        self.class_names = {
            0: 'Police Officer', 1: 'Police Vehicle', 2: 'Cone',
            3: 'Fence', 4: 'Drum', 5: 'Barricade', 6: 'Barrier',
            7: 'Work Vehicle', 8: 'Vertical Panel', 9: 'Tubular Marker',
            10: 'Arrow Board', 11: 'Bike Lane', 12: 'Work Equipment',
            13: 'Worker', 14: 'Other Roadwork Objects',
            15: 'TTC Message Board', 16: 'TTC Sign'
        }

        # Colors for each class (BGR)
        random.seed(42)
        self.class_colors = {}
        for i in range(17):
            self.class_colors[i] = (
                random.randint(50, 255),
                random.randint(50, 255),
                random.randint(50, 255)
            )

        self.get_logger().info('Roadwork Detector Node started!')
        self.get_logger().info(f'Confidence threshold: {self.conf_threshold}')
        self.get_logger().info(f'Resize: {self.resize_w}x{self.resize_h}')

    def camera_info_callback(self, msg):
        self.camera_info = msg

    def dist_metric_callback(self, msg):
        self.latest_distance = msg.data

    def conf_metric_callback(self, msg):
        self.latest_confidence = msg.data

    def risk_metric_callback(self, msg):
        self.latest_risk = msg.data

    def image_callback(self, msg):
        self.frame_count += 1
        start_time = time.time()

        try:
            # Step 1: Convert ROS Image to OpenCV (BGR)
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            original_h, original_w = cv_image.shape[:2]

            # Step 2: Resize for inference
            resized = cv2.resize(cv_image, (self.resize_w, self.resize_h),
                                 interpolation=cv2.INTER_LINEAR)

            # Step 3: Run YOLOv8 inference (GPU accelerated)
            results = self.model.predict(
                source=resized,
                conf=self.conf_threshold,
                verbose=False
            )

            # Step 4: Process detections
            detections = []
            result = results[0]

            for box in result.boxes:
                cls_id = int(box.cls[0])
                confidence = float(box.conf[0])

                if cls_id not in self.target_classes:
                    continue

                x1, y1, x2, y2 = box.xyxy[0].tolist()

                # Step 5: Scale back to original resolution
                scale_x = original_w / self.resize_w
                scale_y = original_h / self.resize_h
                x1_orig = int(x1 * scale_x)
                y1_orig = int(y1 * scale_y)
                x2_orig = int(x2 * scale_x)
                y2_orig = int(y2 * scale_y)

                class_name = self.class_names.get(cls_id, f'class_{cls_id}')

                detections.append({
                    'class': class_name,
                    'class_id': cls_id,
                    'confidence': round(confidence, 2),
                    'bbox': [x1_orig, y1_orig, x2_orig, y2_orig]
                })

                # Step 6: Draw bounding box on original image
                color = self.class_colors.get(cls_id, (255, 255, 255))
                cv2.rectangle(cv_image, (x1_orig, y1_orig), (x2_orig, y2_orig), color, 3)
                label = f'{class_name} {confidence:.2f}'
                label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
                cv2.rectangle(cv_image,
                              (x1_orig, y1_orig - label_size[1] - 10),
                              (x1_orig + label_size[0], y1_orig),
                              color, -1)
                cv2.putText(cv_image, label, (x1_orig, y1_orig - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

            # Calculate processing time
            processing_time = time.time() - start_time

            # Step 6.5: Draw metrics overlay
            overlay = cv_image.copy()
            cv2.rectangle(overlay, (10, 10), (420, 200), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.7, cv_image, 0.3, 0, cv_image)

            cv2.putText(cv_image, "ODD Exit Metrics", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

            d_val = self.latest_distance
            c_val = self.latest_confidence
            r_val = self.latest_risk

            if d_val is not None:
                d_color = (0, 0, 255) if d_val < 5 else (0, 255, 0) if d_val > 30 else (0, 165, 255)
                cv2.putText(cv_image, f"Distance: {d_val:.1f}m", (20, 80),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, d_color, 2)
            else:
                cv2.putText(cv_image, "Distance: N/A", (20, 80),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)

            if c_val is not None:
                c_color = (0, 0, 255) if c_val > 0.7 else (0, 255, 0) if c_val < 0.3 else (0, 165, 255)
                cv2.putText(cv_image, f"Confidence: {c_val*100:.1f}%", (20, 120),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, c_color, 2)
            else:
                cv2.putText(cv_image, "Confidence: N/A", (20, 120),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)

            if r_val is not None:
                r_color = (0, 0, 255) if r_val > 0.7 else (0, 255, 0) if r_val < 0.3 else (0, 165, 255)
                cv2.putText(cv_image, f"Risk: {r_val*100:.1f}%", (20, 160),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, r_color, 2)
            else:
                cv2.putText(cv_image, "Risk: N/A", (20, 160),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)

            cv2.putText(cv_image, f"Detections: {len(detections)} | {processing_time*1000:.0f}ms",
                        (20, 190), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

            # Save frame if detections found
            if len(detections) > 0:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                img_filename = f'frame_{self.frame_count}_{timestamp}.jpg'
                json_filename = f'frame_{self.frame_count}_{timestamp}.json'

                cv2.imwrite(os.path.join(self.output_dir, img_filename), cv_image)

                with open(os.path.join(self.output_dir, json_filename), 'w') as f:
                    json.dump({
                        'frame': self.frame_count,
                        'processing_time_ms': round(processing_time * 1000, 1),
                        'detections': detections
                    }, f, indent=2)

                self.get_logger().info(f'Saved: {img_filename}')

            # Step 7: Publish annotated image
            annotated_msg = self.bridge.cv2_to_imgmsg(cv_image, encoding='bgr8')
            annotated_msg.header = msg.header
            self.annotated_pub.publish(annotated_msg)

            # Step 8: Publish detection results as JSON
            results_data = {
                'frame': self.frame_count,
                'timestamp': msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9,
                'processing_time_ms': round(processing_time * 1000, 1),
                'num_detections': len(detections),
                'detections': detections
            }
            results_msg = String()
            results_msg.data = json.dumps(results_data)
            self.results_pub.publish(results_msg)

            # Log
            self.get_logger().info(
                f'Frame {self.frame_count}: {len(detections)} detections | '
                f'Processing: {processing_time*1000:.0f}ms | '
                f'D: {d_val:.1f}m | C: {c_val*100:.1f}% | R: {r_val*100:.1f}%'
                if d_val and c_val and r_val else
                f'Frame {self.frame_count}: {len(detections)} detections | '
                f'Processing: {processing_time*1000:.0f}ms | Metrics: waiting...'
            )

        except Exception as e:
            self.get_logger().error(f'Error processing frame: {str(e)}')


def main(args=None):
    rclpy.init(args=args)
    node = RoadworkDetectorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
