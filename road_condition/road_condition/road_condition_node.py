import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge
import cv2
import numpy as np
import json
import os


class RoadConditionNode(Node):
    def __init__(self):
        super().__init__('road_condition_node')

        # Parameters
        self.declare_parameter('model_path', '')
        self.declare_parameter('confidence_threshold', 0.5)

        model_path = self.get_parameter('model_path').value
        self.conf_threshold = self.get_parameter('confidence_threshold').value

        # Class labels
        self.class_names = ['dry', 'snow', 'wet']

        # Load Keras model
        if not model_path or not os.path.exists(model_path):
            self.get_logger().error(f'Model not found: {model_path}')
            raise FileNotFoundError(f'Model not found: {model_path}')

        self.get_logger().info(f'Loading EfficientNet model from: {model_path}')
        import tensorflow as tf
        # Force CPU to avoid CuDNN version mismatch
        tf.config.set_visible_devices([], 'GPU')
        self.model = tf.keras.models.load_model(model_path)

        # Warm up
        dummy = np.zeros((1, 224, 224, 3), dtype=np.float32)
        self.model.predict(dummy, verbose=0)
        self.get_logger().info('Model warm-up complete!')

        self.bridge = CvBridge()
        self.frame_count = 0

        # Subscribe to road patch
        self.patch_sub = self.create_subscription(
            Image,
            '/road/patch',
            self.patch_callback,
            10
        )

        # Publish condition result
        self.condition_pub = self.create_publisher(String, '/road/condition', 10)

        self.get_logger().info('Road Condition Node started!')
        self.get_logger().info(f'Classes: {self.class_names}')

    def patch_callback(self, msg):
        self.frame_count += 1
        try:
            # Convert to OpenCV
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

            # Convert BGR to RGB
            rgb_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)

            # Resize to model input size (224x224) if needed
            h, w = rgb_image.shape[:2]
            if h != 224 or w != 224:
                rgb_image = cv2.resize(rgb_image, (224, 224), interpolation=cv2.INTER_LINEAR)

            # Prepare for model: add batch dimension
            input_tensor = np.expand_dims(rgb_image, axis=0).astype(np.float32)

            # Predict
            predictions = self.model.predict(input_tensor, verbose=0)
            probs = predictions[0]

            class_idx = int(np.argmax(probs))
            confidence = float(probs[class_idx])
            condition = self.class_names[class_idx]

            # Publish result
            result = {
                'frame': self.frame_count,
                'condition': condition,
                'confidence': round(confidence, 4),
                'probabilities': {
                    self.class_names[i]: round(float(probs[i]), 4)
                    for i in range(len(self.class_names))
                }
            }
            result_msg = String()
            result_msg.data = json.dumps(result)
            self.condition_pub.publish(result_msg)

            if self.frame_count % 30 == 0:
                self.get_logger().info(
                    f'Frame {self.frame_count}: {condition} '
                    f'({confidence*100:.1f}%) | '
                    f'dry={probs[0]*100:.1f}% '
                    f'snow={probs[1]*100:.1f}% '
                    f'wet={probs[2]*100:.1f}%'
                )

        except Exception as e:
            self.get_logger().error(f'Error in road condition: {str(e)}')


def main(args=None):
    rclpy.init(args=args)
    node = RoadConditionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
