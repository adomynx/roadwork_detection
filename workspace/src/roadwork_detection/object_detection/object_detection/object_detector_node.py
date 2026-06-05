#!/usr/bin/env python3
"""
object_detector_node.py

General-purpose object detection node for the perception pipeline.
Runs YOLOv8x pretrained on COCO (80 classes) via TensorRT FP16.
Sibling to detector_node — same model architecture, different weights,
different output namespace. Does NOT render the ODD-metric HUD overlay.

Subscribes : /zed/zed_node/left_raw/image_raw_color/compressed
Publishes  : /objects/results            (std_msgs/String, JSON payload)
             /objects/annotated_image    (sensor_msgs/Image, optional)
"""

import json
import os
import time

import cv2
import numpy as np
import rclpy
from ament_index_python.packages import get_package_share_directory
from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage, Image
from std_msgs.msg import String
from ultralytics import YOLO


class ObjectDetectorNode(Node):
    def __init__(self):
        super().__init__('object_detector_node')

        # ---- Parameters ---------------------------------------------------
        self.declare_parameter('model_path', '')
        self.declare_parameter('confidence_threshold', 0.35)
        self.declare_parameter('iou_threshold', 0.45)
        self.declare_parameter('publish_annotated', True)
        self.declare_parameter('input_topic',
                               '/zed/zed_node/left_raw/image_raw_color/compressed')
        self.declare_parameter('results_topic', '/objects/results')
        self.declare_parameter('annotated_topic', '/objects/annotated_image')

        gp = lambda k: self.get_parameter(k).get_parameter_value()
        model_path = gp('model_path').string_value
        self.conf_thresh = gp('confidence_threshold').double_value
        self.iou_thresh = gp('iou_threshold').double_value
        self.publish_annotated = gp('publish_annotated').bool_value
        input_topic = gp('input_topic').string_value
        results_topic = gp('results_topic').string_value
        annotated_topic = gp('annotated_topic').string_value

        # Default model path resolves to the engine bundled in the package share
        if not model_path:
            pkg_share = get_package_share_directory('object_detection')
            model_path = os.path.join(pkg_share, 'models', 'yolov8x_coco.engine')

        if not os.path.exists(model_path):
            self.get_logger().error(f'Model file not found: {model_path}')
            raise FileNotFoundError(model_path)

        # ---- Model --------------------------------------------------------
        self.get_logger().info(f'Loading TRT engine: {model_path}')
        self.model = YOLO(model_path, task='detect')

        # Warmup (avoids first-frame latency spike)
        dummy = np.zeros((640, 640, 3), dtype=np.uint8)
        self.model.predict(dummy, verbose=False,
                           conf=self.conf_thresh, iou=self.iou_thresh)
        self.get_logger().info('Model warmed up')

        # ---- ROS plumbing -------------------------------------------------
        self.bridge = CvBridge()
        self.frame_count = 0
        self.last_log_time = time.time()

        self.image_sub = self.create_subscription(
            CompressedImage, input_topic, self.compressed_image_callback, 10)

        self.results_pub = self.create_publisher(String, results_topic, 10)
        if self.publish_annotated:
            self.annotated_pub = self.create_publisher(
                Image, annotated_topic, 10)
        else:
            self.annotated_pub = None

        self.get_logger().info(
            f'object_detector_node ready — subscribing to {input_topic}, '
            f'publishing {results_topic}'
            + (f' and {annotated_topic}' if self.publish_annotated else ''))

    # ----------------------------------------------------------------------
    def compressed_image_callback(self, msg: CompressedImage):
        try:
            np_arr = np.frombuffer(msg.data, np.uint8)
            frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            if frame is None:
                self.get_logger().warn('Failed to decode compressed image')
                return
        except Exception as e:
            self.get_logger().error(f'Image decode error: {e}')
            return
        self._process_frame(frame, msg.header)

    # ----------------------------------------------------------------------
    def _process_frame(self, frame, header):
        t0 = time.time()
        results = self.model.predict(
            frame, verbose=False,
            conf=self.conf_thresh, iou=self.iou_thresh)
        inference_ms = (time.time() - t0) * 1000.0

        # ---- Extract detections ------------------------------------------
        detections = []
        if results:
            r = results[0]
            if r.boxes is not None and len(r.boxes) > 0:
                names = r.names
                boxes_xyxy = r.boxes.xyxy.cpu().numpy()
                confs = r.boxes.conf.cpu().numpy()
                cls_ids = r.boxes.cls.cpu().numpy().astype(int)
                for box, conf, cls_id in zip(boxes_xyxy, confs, cls_ids):
                    x1, y1, x2, y2 = box.tolist()
                    detections.append({
                        'class_id': int(cls_id),
                        'class_name': names[cls_id],
                        'confidence': float(conf),
                        'bbox': [float(x1), float(y1),
                                 float(x2), float(y2)],
                    })

        # ---- Publish results ---------------------------------------------
        out = String()
        out.data = json.dumps({
            'stamp_sec': header.stamp.sec,
            'stamp_nanosec': header.stamp.nanosec,
            'frame_id': header.frame_id,
            'inference_ms': round(inference_ms, 2),
            'count': len(detections),
            'detections': detections,
        })
        self.results_pub.publish(out)

        # ---- Publish annotated frame -------------------------------------
        if self.annotated_pub is not None:
            annotated = results[0].plot() if results else frame
            img_msg = self.bridge.cv2_to_imgmsg(annotated, encoding='bgr8')
            img_msg.header = header
            self.annotated_pub.publish(img_msg)

        # ---- Periodic FPS log --------------------------------------------
        self.frame_count += 1
        now = time.time()
        if now - self.last_log_time >= 5.0:
            fps = self.frame_count / (now - self.last_log_time)
            self.get_logger().info(
                f'OBJ FPS: {fps:.1f} | inference: {inference_ms:.1f}ms '
                f'| det: {len(detections)}')
            self.frame_count = 0
            self.last_log_time = now


def main(args=None):
    rclpy.init(args=args)
    node = ObjectDetectorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
