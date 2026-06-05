import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import String
import cv2
import numpy as np
import json


class WeatherNode(Node):
    def __init__(self):
        super().__init__('weather_node')

        self.declare_parameter('model_path', '')
        self.declare_parameter('inference_interval', 5)

        model_path = self.get_parameter('model_path').value
        self.inference_interval = self.get_parameter('inference_interval').value
        self.class_names = ['clear', 'fog', 'rain', 'snow']

        if not model_path or not os.path.exists(model_path):
            self.get_logger().error(f'Model not found: {model_path}')
            raise FileNotFoundError(f'Model not found: {model_path}')

        self.get_logger().info(f'Loading weather model from: {model_path}')
        import tensorflow as tf
        tf.config.set_visible_devices([], 'GPU')
        self.model = tf.keras.models.load_model(model_path, compile=False)
        self.get_logger().info('Model loaded!')

        # Warm up
        dummy = np.zeros((1, 224, 224, 3), dtype=np.float32)
        self.model.predict(dummy, verbose=0)
        self.get_logger().info('Warm-up complete!')

        self.frame_count = 0

        self.image_sub = self.create_subscription(
            CompressedImage,
            '/zed/zed_node/left_raw/image_raw_color/compressed',
            self.image_callback, 10)

        self.weather_pub = self.create_publisher(String, '/weather/condition', 10)

        self.get_logger().info(f'Weather Node started! Classes: {self.class_names}')

    def image_callback(self, msg):
        self.frame_count += 1
        if self.frame_count % self.inference_interval != 0:
            return

        try:
            np_arr = np.frombuffer(msg.data, np.uint8)
            cv_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            if cv_image is None:
                return

            img = cv2.resize(cv_image, (224, 224))
            pred = self.model.predict(
                np.expand_dims(img, 0).astype(np.float32), verbose=0)[0]

            class_idx = int(np.argmax(pred))
            condition = self.class_names[class_idx]
            confidence = float(pred[class_idx])

            result = {
                'frame': self.frame_count,
                'condition': condition,
                'confidence': round(confidence, 4),
                'probabilities': {
                    self.class_names[i]: round(float(pred[i]), 4)
                    for i in range(len(self.class_names))
                }
            }

            msg_out = String()
            msg_out.data = json.dumps(result)
            self.weather_pub.publish(msg_out)

            if self.frame_count % 50 == 0:
                self.get_logger().info(
                    f'Frame {self.frame_count}: {condition} ({confidence*100:.0f}%)')

        except Exception as e:
            self.get_logger().error(f'Error: {str(e)}')


def main(args=None):
    rclpy.init(args=args)
    node = WeatherNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
