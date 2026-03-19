import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import os
import glob
import sys


class VideoPublisherNode(Node):
    def __init__(self):
        super().__init__('video_publisher_node')

        self.declare_parameter('video_path', '')
        self.declare_parameter('video_dir', '')
        self.declare_parameter('fps', 0.0)
        self.declare_parameter('loop', False)

        video_path = self.get_parameter('video_path').value
        video_dir = self.get_parameter('video_dir').value
        self.loop = self.get_parameter('loop').value
        self.target_fps = self.get_parameter('fps').value

        # Build video list
        self.video_list = []
        if video_dir and os.path.isdir(video_dir):
            self.video_list = sorted(glob.glob(os.path.join(video_dir, '*.mp4')))
        elif video_path and os.path.isfile(video_path):
            self.video_list = [video_path]

        if not self.video_list:
            self.get_logger().error('No videos found! Set video_path or video_dir parameter.')
            return

        self.get_logger().info(f'Found {len(self.video_list)} videos')

        self.bridge = CvBridge()
        self.image_pub = self.create_publisher(Image, '/camera/image_raw', 10)

        self.current_video_idx = 0
        self.cap = None
        self.frame_count = 0
        self.total_frames_all = 0

        self.open_next_video()

        if self.cap and self.cap.isOpened():
            period = 1.0 / self.actual_fps
            self.timer = self.create_timer(period, self.timer_callback)
            self.get_logger().info(f'Publishing at {self.actual_fps:.1f} FPS')

    def open_next_video(self):
        if self.cap:
            self.cap.release()

        if self.current_video_idx >= len(self.video_list):
            self.get_logger().info('All videos processed!')
            self.cap = None
            return False

        video_path = self.video_list[self.current_video_idx]
        self.cap = cv2.VideoCapture(video_path)

        if not self.cap.isOpened():
            self.get_logger().error(f'Cannot open: {video_path}')
            self.current_video_idx += 1
            return self.open_next_video()

        video_fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.actual_fps = self.target_fps if self.target_fps > 0 else video_fps
        total = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

        self.get_logger().info(
            f'[{self.current_video_idx + 1}/{len(self.video_list)}] '
            f'{os.path.basename(video_path)} | '
            f'Frames: {total} | FPS: {self.actual_fps:.1f}'
        )
        return True

    def timer_callback(self):
        if self.cap is None or not self.cap.isOpened():
            if not self.open_next_video():
                self.get_logger().info(
                    f'Finished all videos! Total frames published: {self.total_frames_all}'
                )
                self.timer.cancel()
                return

        ret, frame = self.cap.read()
        if not ret:
            self.current_video_idx += 1
            if not self.open_next_video():
                self.get_logger().info(
                    f'Finished all videos! Total frames published: {self.total_frames_all}'
                )
                self.timer.cancel()
            return

        self.frame_count += 1
        self.total_frames_all += 1

        msg = self.bridge.cv2_to_imgmsg(frame, encoding='bgr8')
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'camera'
        self.image_pub.publish(msg)

        if self.frame_count % 100 == 0:
            self.get_logger().info(
                f'Published frame {self.frame_count} | Total: {self.total_frames_all}'
            )


def main(args=None):
    rclpy.init(args=args)
    node = VideoPublisherNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node.cap:
            node.cap.release()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

