import time
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from std_msgs.msg import String
from cv_bridge import CvBridge
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
import cv2
import json
import os
import random
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

        # Warm up the model (first inference is slow)
        import numpy as np
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
            '/arena_camera_node/images',
            self.image_callback,
            qos_profile
        )
        self.info_sub = self.create_subscription(
            CameraInfo,
            '/arena_camera_node/camera_info',
            self.camera_info_callback,
            qos_profile
        )

        # Publishers
        self.annotated_pub = self.create_publisher(Image, '/detection/annotated_image', 10)
        self.results_pub = self.create_publisher(String, '/detection/results', 10)

        # Frame counter
        self.frame_count = 0

        # COCO class names
        self.class_names = {
            0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane',
            5: 'bus', 6: 'train', 7: 'truck', 8: 'boat', 9: 'traffic_light',
            10: 'fire_hydrant', 11: 'stop_sign', 12: 'parking_meter', 13: 'bench',
            14: 'bird', 15: 'cat', 16: 'dog', 17: 'horse', 18: 'sheep', 19: 'cow',
            20: 'elephant', 21: 'bear', 22: 'zebra', 23: 'giraffe', 24: 'backpack',
            25: 'umbrella', 26: 'handbag', 27: 'tie', 28: 'suitcase', 29: 'frisbee',
            30: 'skis', 31: 'snowboard', 32: 'sports_ball', 33: 'kite',
            34: 'baseball_bat', 35: 'baseball_glove', 36: 'skateboard',
            37: 'surfboard', 38: 'tennis_racket', 39: 'bottle', 40: 'wine_glass',
            41: 'cup', 42: 'fork', 43: 'knife', 44: 'spoon', 45: 'bowl',
            46: 'banana', 47: 'apple', 48: 'sandwich', 49: 'orange', 50: 'broccoli',
            51: 'carrot', 52: 'hot_dog', 53: 'pizza', 54: 'donut', 55: 'cake',
            56: 'chair', 57: 'couch', 58: 'potted_plant', 59: 'bed',
            60: 'dining_table', 61: 'toilet', 62: 'tv', 63: 'laptop', 64: 'mouse',
            65: 'remote', 66: 'keyboard', 67: 'cell_phone', 68: 'microwave',
            69: 'oven', 70: 'toaster', 71: 'sink', 72: 'refrigerator', 73: 'book',
            74: 'clock', 75: 'vase', 76: 'scissors', 77: 'teddy_bear',
            78: 'hair_drier', 79: 'toothbrush'
        }

        # Colors for each class (BGR) - auto-generated
        random.seed(42)
        self.class_colors = {}
        for i in range(80):
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

                # Filter for target classes only
                if cls_id not in self.target_classes:
                    continue

                # Get bounding box (on resized image)
                x1, y1, x2, y2 = box.xyxy[0].tolist()

                # Step 5: Scale back to original resolution
                scale_x = original_w / self.resize_w
                scale_y = original_h / self.resize_h
                x1_orig = int(x1 * scale_x)
                y1_orig = int(y1 * scale_y)
                x2_orig = int(x2 * scale_x)
                y2_orig = int(y2 * scale_y)

                # Get class name
                class_name = self.class_names.get(cls_id, f'class_{cls_id}')

                # Store detection
                detections.append({
                    'class': class_name,
                    'class_id': cls_id,
                    'confidence': round(confidence, 2),
                    'bbox': [x1_orig, y1_orig, x2_orig, y2_orig]
                })

                # Step 6: Draw on original image
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
                f'Image stamp: {msg.header.stamp.sec}.{msg.header.stamp.nanosec}'
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
