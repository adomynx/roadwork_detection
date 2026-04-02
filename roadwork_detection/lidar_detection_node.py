import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import String
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
import numpy as np
import json


class LidarDetectionNode(Node):
    def __init__(self):
        super().__init__('lidar_detection_node')

        # Parameters
        self.declare_parameter('ground_threshold', 0.3)
        self.declare_parameter('min_height', -1.5)
        self.declare_parameter('max_height', 3.0)
        self.declare_parameter('min_distance', 1.0)
        self.declare_parameter('max_distance', 50.0)
        self.declare_parameter('cluster_tolerance', 0.8)
        self.declare_parameter('min_cluster_size', 5)
        self.declare_parameter('max_cluster_size', 5000)
        self.declare_parameter('voxel_size', 0.15)

        self.ground_thresh = self.get_parameter('ground_threshold').value
        self.min_height = self.get_parameter('min_height').value
        self.max_height = self.get_parameter('max_height').value
        self.min_dist = self.get_parameter('min_distance').value
        self.max_dist = self.get_parameter('max_distance').value
        self.cluster_tol = self.get_parameter('cluster_tolerance').value
        self.min_cluster = self.get_parameter('min_cluster_size').value
        self.max_cluster = self.get_parameter('max_cluster_size').value
        self.voxel_size = self.get_parameter('voxel_size').value

        # State
        self.frame_count = 0

        # QoS
        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=5
        )

        # Subscriber
        self.lidar_sub = self.create_subscription(
            PointCloud2,
            '/rslidar_points',
            self.lidar_callback,
            sensor_qos
        )

        # Publisher
        self.obstacles_pub = self.create_publisher(String, '/lidar/obstacles', 10)

        self.get_logger().info('LiDAR Detection Node started!')
        self.get_logger().info(
            f'Ground thresh: {self.ground_thresh}m | '
            f'Cluster tol: {self.cluster_tol}m | '
            f'Range: {self.min_dist}-{self.max_dist}m'
        )

    def lidar_callback(self, msg):
        self.frame_count += 1
        try:
            # Step 1: Parse point cloud
            points = self._parse_pointcloud2(msg)
            if len(points) == 0:
                return

            # Step 2: Distance filter (remove too close and too far)
            dist_xy = np.sqrt(points[:, 0]**2 + points[:, 1]**2)
            range_mask = (dist_xy > self.min_dist) & (dist_xy < self.max_dist)
            points = points[range_mask]

            if len(points) == 0:
                return

            # Step 3: Height filter
            height_mask = (points[:, 2] > self.min_height) & (points[:, 2] < self.max_height)
            points = points[height_mask]

            if len(points) == 0:
                return

            # Step 4: Voxel downsampling for speed
            points = self._voxel_downsample(points, self.voxel_size)

            # Step 5: Ground removal using RANSAC plane fitting
            non_ground = self._remove_ground_ransac(points)

            if len(non_ground) < self.min_cluster:
                return

            # Step 6: Euclidean clustering
            clusters = self._euclidean_cluster(non_ground)

            if len(clusters) == 0:
                return

            # Step 7: Build obstacle list
            obstacles = []
            for cluster_points in clusters:
                centroid = np.mean(cluster_points, axis=0)
                mins = np.min(cluster_points, axis=0)
                maxs = np.max(cluster_points, axis=0)
                size = maxs - mins
                distance = float(np.sqrt(centroid[0]**2 + centroid[1]**2 + centroid[2]**2))

                obstacles.append({
                    'centroid': [round(float(centroid[0]), 2),
                                 round(float(centroid[1]), 2),
                                 round(float(centroid[2]), 2)],
                    'bbox_min': [round(float(mins[0]), 2),
                                 round(float(mins[1]), 2),
                                 round(float(mins[2]), 2)],
                    'bbox_max': [round(float(maxs[0]), 2),
                                 round(float(maxs[1]), 2),
                                 round(float(maxs[2]), 2)],
                    'size': [round(float(size[0]), 2),
                             round(float(size[1]), 2),
                             round(float(size[2]), 2)],
                    'distance': round(distance, 2),
                    'num_points': len(cluster_points)
                })

            # Publish
            obs_msg = String()
            obs_msg.data = json.dumps({
                'frame': self.frame_count,
                'num_obstacles': len(obstacles),
                'total_points': len(non_ground),
                'obstacles': obstacles
            })
            self.obstacles_pub.publish(obs_msg)

            if self.frame_count % 10 == 0:
                self.get_logger().info(
                    f'Frame {self.frame_count}: {len(obstacles)} obstacles | '
                    f'{len(non_ground)} non-ground points'
                )

        except Exception as e:
            self.get_logger().error(f'LiDAR detection error: {str(e)}')

    def _remove_ground_ransac(self, points, max_iterations=100, threshold=None):
        if threshold is None:
            threshold = self.ground_thresh

        best_inliers = None
        best_count = 0
        n_points = len(points)

        for _ in range(max_iterations):
            # Random 3 points
            idx = np.random.choice(n_points, 3, replace=False)
            p1, p2, p3 = points[idx]

            # Plane normal
            v1 = p2 - p1
            v2 = p3 - p1
            normal = np.cross(v1, v2)
            norm_len = np.linalg.norm(normal)
            if norm_len < 1e-8:
                continue
            normal = normal / norm_len

            # Check if plane is roughly horizontal (normal mostly vertical)
            if abs(normal[2]) < 0.7:
                continue

            # Distance from all points to plane
            d = np.abs(np.dot(points - p1, normal))
            inliers = d < threshold
            count = np.sum(inliers)

            if count > best_count:
                best_count = count
                best_inliers = inliers

        if best_inliers is None:
            return points

        # Return non-ground points
        return points[~best_inliers]

    def _euclidean_cluster(self, points):
        if len(points) == 0:
            return []

        # Simple grid-based clustering for speed
        # Assign each point to a grid cell
        grid_size = self.cluster_tol
        grid_keys = np.floor(points[:, :2] / grid_size).astype(int)

        # Group points by grid cell
        cell_map = {}
        for i in range(len(points)):
            key = (grid_keys[i, 0], grid_keys[i, 1])
            if key not in cell_map:
                cell_map[key] = []
            cell_map[key].append(i)

        # BFS to merge adjacent cells into clusters
        visited = set()
        clusters = []

        for key in cell_map:
            if key in visited:
                continue

            cluster_indices = []
            queue = [key]
            visited.add(key)

            while queue:
                current = queue.pop(0)
                if current in cell_map:
                    cluster_indices.extend(cell_map[current])

                # Check 8 neighbors
                for dx in [-1, 0, 1]:
                    for dy in [-1, 0, 1]:
                        if dx == 0 and dy == 0:
                            continue
                        neighbor = (current[0] + dx, current[1] + dy)
                        if neighbor not in visited and neighbor in cell_map:
                            visited.add(neighbor)
                            queue.append(neighbor)

            if self.min_cluster <= len(cluster_indices) <= self.max_cluster:
                clusters.append(points[cluster_indices])

        return clusters

    def _voxel_downsample(self, points, voxel_size):
        keys = np.floor(points / voxel_size).astype(int)
        _, unique_idx = np.unique(keys, axis=0, return_index=True)
        return points[unique_idx]

    def _parse_pointcloud2(self, msg):
        fields = {f.name: f for f in msg.fields}
        if 'x' not in fields or 'y' not in fields or 'z' not in fields:
            return np.array([])

        x_off = fields['x'].offset
        y_off = fields['y'].offset
        z_off = fields['z'].offset
        point_step = msg.point_step
        n_points = msg.width * msg.height

        data = np.frombuffer(msg.data, dtype=np.uint8)

        x_starts = np.arange(n_points) * point_step + x_off
        y_starts = np.arange(n_points) * point_step + y_off
        z_starts = np.arange(n_points) * point_step + z_off

        max_idx = len(data) - 4
        valid = (x_starts <= max_idx) & (y_starts <= max_idx) & (z_starts <= max_idx)
        x_starts = x_starts[valid]
        y_starts = y_starts[valid]
        z_starts = z_starts[valid]

        x_bytes = np.array([data[s:s+4] for s in x_starts], dtype=np.uint8)
        y_bytes = np.array([data[s:s+4] for s in y_starts], dtype=np.uint8)
        z_bytes = np.array([data[s:s+4] for s in z_starts], dtype=np.uint8)

        x = x_bytes.view(np.float32).reshape(-1)
        y = y_bytes.view(np.float32).reshape(-1)
        z = z_bytes.view(np.float32).reshape(-1)

        points = np.column_stack([x, y, z])
        mask = np.isfinite(points).all(axis=1) & (np.abs(points).sum(axis=1) > 0.01)
        return points[mask]


def main(args=None):
    rclpy.init(args=args)
    node = LidarDetectionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
