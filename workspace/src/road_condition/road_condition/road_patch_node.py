import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CompressedImage
from cv_bridge import CvBridge
import cv2
import numpy as np


class RoadPatchNode(Node):
    def __init__(self):
        super().__init__('road_patch_node')

        # Parameters
        self.declare_parameter('patch_output_size', 224)
        self.declare_parameter('camera_fx', 266.9)
        self.declare_parameter('camera_fy', 266.9)
        self.declare_parameter('camera_cx', 314.0)
        self.declare_parameter('camera_cy', 182.8)
        self.declare_parameter('tire_offset_x', 0.775)
        self.declare_parameter('camera_height', 1.92)
        self.declare_parameter('look_ahead_distance', 6.0)

        self.patch_size = self.get_parameter('patch_output_size').value
        self.fx = self.get_parameter('camera_fx').value
        self.fy = self.get_parameter('camera_fy').value
        self.cx = self.get_parameter('camera_cx').value
        self.cy = self.get_parameter('camera_cy').value
        self.tire_x = self.get_parameter('tire_offset_x').value
        self.cam_h = self.get_parameter('camera_height').value
        self.look_dist = self.get_parameter('look_ahead_distance').value

        # Calculate crop centers using camera projection--- 
        # Left tire: [-tire_x, cam_h, look_dist]
        # Right tire: [+tire_x, cam_h, look_dist]
        self.left_u = int(self.fx * (-self.tire_x) / self.look_dist + self.cx)
        self.left_v = int(self.fy * self.cam_h / self.look_dist + self.cy)
        self.right_u = int(self.fx * self.tire_x / self.look_dist + self.cx)
        self.right_v = self.left_v  # same vertical position

        self.bridge = CvBridge()
        self.frame_count = 0

        # Subscribe to ZED compressed camera
        self.image_sub = self.create_subscription(
            CompressedImage,
            '/zed/zed_node/left_raw/image_raw_color/compressed',
            self.image_callback,
            10
        )

        # Publish cropped road patches (left and right)
        self.patch_left_pub = self.create_publisher(Image, '/road/patch', 10)
        self.patch_right_pub = self.create_publisher(Image, '/road/patch_right', 10)

        self.get_logger().info('Road Patch Node started!')
        self.get_logger().info(
            f'Left crop center: ({self.left_u}, {self.left_v}) | '
            f'Right crop center: ({self.right_u}, {self.right_v})'
        )

    def _extract_patch(self, image, center_u, center_v):
        h, w = image.shape[:2]
        half = 112  # 224/2

        # Calculate crop box
        x1 = center_u - half
        y1 = center_v - half
        x2 = center_u + half
        y2 = center_v + half

        # If crop goes out of bounds, adjust and use what we can
        x1_safe = max(0, x1)
        y1_safe = max(0, y1)
        x2_safe = min(w, x2)
        y2_safe = min(h, y2)

        patch = image[y1_safe:y2_safe, x1_safe:x2_safe]

        # Resize to 224x224 if needed
        if patch.shape[0] != 224 or patch.shape[1] != 224:
            patch = cv2.resize(patch, (224, 224), interpolation=cv2.INTER_LINEAR)

        return patch

    def image_callback(self, msg):
        self.frame_count += 1
        try:
            np_arr = np.frombuffer(msg.data, np.uint8)
            cv_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            if cv_image is None:
                return

            # Extract left tire patch
            left_patch = self._extract_patch(cv_image, self.left_u, self.left_v)
            left_msg = self.bridge.cv2_to_imgmsg(left_patch, encoding='bgr8')
            left_msg.header = msg.header
            self.patch_left_pub.publish(left_msg)

            # Extract right tire patch
            right_patch = self._extract_patch(cv_image, self.right_u, self.right_v)
            right_msg = self.bridge.cv2_to_imgmsg(right_patch, encoding='bgr8')
            right_msg.header = msg.header
            self.patch_right_pub.publish(right_msg)

            if self.frame_count % 100 == 0:
                self.get_logger().info(
                    f'Frame {self.frame_count}: Published patches | '
                    f'Left: ({self.left_u}, {self.left_v}) | '
                    f'Right: ({self.right_u}, {self.right_v})'
                )

        except Exception as e:
            self.get_logger().error(f'Error in road patch: {str(e)}')


def main(args=None):
    rclpy.init(args=args)
    node = RoadPatchNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
