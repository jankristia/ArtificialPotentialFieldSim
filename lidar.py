import numpy as np
from scipy.spatial import KDTree
from scipy.spatial import ConvexHull

class LidarSimulator:
    def __init__(self, max_range=20, num_rays=128):
        self.max_range = max_range
        self.num_rays = num_rays
        self.angles = np.linspace(-1/2*np.pi, 1/2*np.pi, num_rays)
        self.obstacles = []

    def ray_circle_intersection(self, ray_origin, ray_dir, circle_center, circle_radius):
        """ Computes the intersection of a ray with a circular obstacle """
        oc = ray_origin - circle_center
        a = np.dot(ray_dir, ray_dir)
        b = 2.0 * np.dot(oc, ray_dir)
        c = np.dot(oc, oc) - circle_radius**2
        
        discriminant = b**2 - 4*a*c
        if discriminant < 0:
            return None
        
        t1 = (-b - np.sqrt(discriminant)) / (2*a)
        t2 = (-b + np.sqrt(discriminant)) / (2*a)
        
        if t1 > 0 and t1 < self.max_range:
            return t1
        elif t2 > 0 and t2 < self.max_range:
            return t2
        return None
    
    def sense_obstacles(self, boat_x, boat_y, boat_psi, circular_obstacles=[]):
        """ Simulates LiDAR scan by checking exact intersection points with obstacles """
        distances = np.full(self.num_rays, float(self.max_range))  # Default: max range
        ray_origin = np.array([boat_x, boat_y])


        self.obstacles = [(obs.x, obs.y, obs.radius) for obs in circular_obstacles]
        
        for i, angle in enumerate(self.angles):
            ray_angle = angle + boat_psi
            ray_dir = np.array([np.cos(ray_angle), np.sin(ray_angle)])
            
            for obs_x, obs_y, obs_r in self.obstacles:
                circle_center = np.array([obs_x, obs_y])
                intersection = self.ray_circle_intersection(ray_origin, ray_dir, circle_center, obs_r)
                
                if intersection is not None:
                    distances[i] = min(distances[i], intersection)
                    
        return distances
    


    def remove_noise_knn(self, distances, angles, k=5, std_threshold=1, mean_threshold=2):
        """ Removes noisy measurements using k-nearest neighbors and standard deviation. """
        # Convert polar coordinates to Cartesian coordinates
        x_coords = np.array([dist * np.cos(angle) for dist, angle in zip(distances, angles)])
        y_coords = np.array([dist * np.sin(angle) for dist, angle in zip(distances, angles)])
        points = np.column_stack((x_coords, y_coords))
        
        # Build a KD-tree for efficient neighbor searching
        tree = KDTree(points)
        
        to_remove = []
        for i, point in enumerate(points):
            distances_to_neighbors, indices = tree.query(point, k=k+1)  # k+1 because the first is itself
            std_dev = np.std(distances_to_neighbors[1:])  # Exclude the first, which is itself
            
            mean_dev = np.mean(distances_to_neighbors[1:])

            if std_dev > std_threshold or mean_dev > mean_threshold:
                to_remove.append(i)
            
        
        filtered_distances = np.delete(distances, to_remove).tolist()
        filtered_angles = np.delete(angles, to_remove).tolist()
            
        return filtered_distances, filtered_angles
    
    def cluster_lidar_points(self, distances, angles, dist_max=20, threshold=1):
        """Clusters LiDAR points and stores all measurements within each cluster."""
        clusters = []
        current_cluster = []

        for i in range(len(angles)):
            distance = distances[i]
            angle = angles[i]  

            if distance >= dist_max:
                if current_cluster:
                    clusters.append(current_cluster)
                    current_cluster = []
                continue

            if current_cluster and abs(distance - current_cluster[-1][0]) > threshold:
                clusters.append(current_cluster)
                current_cluster = []

            current_cluster.append((distance, angle))

        if current_cluster:
            clusters.append(current_cluster)

        return clusters
        
    def merge_clusters(self, clusters, merge_threshold=3):
        """Merges clusters into objects if the Euclidean distance between them is below a threshold."""
        merged_clusters = [clusters[0]]

        for i in range(1, len(clusters)):
            last_point = merged_clusters[-1][-1]
            first_point = clusters[i][0]

            euclidean_distance = np.sqrt((last_point[0] * np.cos(last_point[1]) - first_point[0] * np.cos(first_point[1])) ** 2 +
                                        (last_point[0] * np.sin(last_point[1]) - first_point[0] * np.sin(first_point[1])) ** 2)

            if euclidean_distance <= merge_threshold:
                merged_clusters[-1].extend(clusters[i])
            else:
                merged_clusters.append(clusters[i])

        return merged_clusters
    
    
    def get_oriented_bounding_box(self, boat_state, cluster):
        """ Computes the minimum-area bounding rectangle for a given cluster. """
        if len(cluster) < 3:
            return None  # Not enough points for a valid bounding box

        x_body = np.array([point[0] * np.cos(point[1]) for point in cluster])
        y_body = np.array([point[0] * np.sin(point[1]) for point in cluster])
        
        # Extract boat state information
        x_boat, y_boat, theta = boat_state[:3]  # Boat position and heading

        # Rotation matrix for body to ENU transformation
        R = np.array([
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta),  np.cos(theta)]
        ])

        # Apply transformation
        points_body = np.column_stack((x_body, y_body))  # Shape: (N,2)
        points = (R @ points_body.T).T + np.array([x_boat, y_boat]) 

        # Check if all points are nearly collinear (low variance in one dimension)
        if np.ptp(points[:, 0]) < 1e-6 or np.ptp(points[:, 1]) < 1e-6:
            # If points are nearly a line, return an axis-aligned bounding box (AABB)
            min_x, max_x = np.min(points[:, 0]), np.max(points[:, 0])
            min_y, max_y = np.min(points[:, 1]), np.max(points[:, 1])
            return np.array([
                [min_x, min_y],
                [max_x, min_y],
                [max_x, max_y],
                [min_x, max_y]
            ])

        try:
            # Compute the convex hull of the points
            hull = ConvexHull(points, qhull_options="QJ")  # QJ joggles points to avoid numerical issues
            hull_points = points[hull.vertices]

            # Initialize variables for finding the minimum bounding box
            min_area = float("inf")
            best_rect = None

            for i in range(len(hull_points)):
                # Compute the edge vector
                edge = hull_points[(i + 1) % len(hull_points)] - hull_points[i]
                edge_angle = np.arctan2(edge[1], edge[0])  # Angle of the edge

                # Rotation matrix for aligning edge with x-axis
                cos_theta, sin_theta = np.cos(-edge_angle), np.sin(-edge_angle)
                rotation_matrix = np.array([[cos_theta, -sin_theta], [sin_theta, cos_theta]])

                # Rotate all points
                rotated_points = np.dot(hull_points, rotation_matrix.T)  # Ensure correct transformation

                # Compute bounding box in rotated space
                min_x, max_x = np.min(rotated_points[:, 0]), np.max(rotated_points[:, 0])
                min_y, max_y = np.min(rotated_points[:, 1]), np.max(rotated_points[:, 1])
                area = (max_x - min_x) * (max_y - min_y)

                if area < min_area:
                    min_area = area
                    best_rect = (min_x, min_y, max_x, max_y, edge_angle)

            # Convert best rectangle back to original space
            min_x, min_y, max_x, max_y, angle = best_rect
            cos_theta, sin_theta = np.cos(angle), np.sin(angle)
            rotation_matrix = np.array([[cos_theta, -sin_theta], [sin_theta, cos_theta]])

            # Define the four rectangle corners in the rotated space
            rect_corners = np.array([
                [min_x, min_y],
                [max_x, min_y],
                [max_x, max_y],
                [min_x, max_y]
            ])

            # Transform back to original space
            rect_corners = np.dot(rect_corners, rotation_matrix.T)

            return rect_corners

        except Exception as e:
            print(f"Warning: ConvexHull failed due to numerical issues ({e}). Returning AABB.")
            # If ConvexHull fails, return a simple axis-aligned bounding box (AABB)
            min_x, max_x = np.min(points[:, 0]), np.max(points[:, 0])
            min_y, max_y = np.min(points[:, 1]), np.max(points[:, 1])
            return np.array([
                [min_x, min_y],
                [max_x, min_y],
                [max_x, max_y],
                [min_x, max_y]
            ])

        
    def clusters_to_oriented_rectangles(self, boat_state, clusters):
        """ Converts all clusters into oriented bounding rectangles. """
        rectangles = []

        for cluster in clusters:
            obb = self.get_oriented_bounding_box(boat_state, cluster)  # Call function on one cluster at a time
            if obb is not None:
                rectangles.append(obb)

        return rectangles
        
    
    
    # ------------------ OLD CLUSTERING --------------------------- #
    
    
    
    
    def cluster_lidar_data(self, boat_state, circular_obstacles):
        """Clusters LiDAR data into detected obstacles, adding a safety margin to each obstacle."""

        lidar_readings = self.sense_obstacles(boat_state[0], boat_state[1], boat_state[2], circular_obstacles)
        obstacle_clusters = []
        prev_dist = None
        dist_diff_threshold = 1.0
        cluster = []

        for dist, angle in zip(lidar_readings, self.angles):
            adjusted_angle = angle + boat_state[2]

            if dist >= self.max_range:
                # End the current cluster if there is one
                if cluster:
                    
                    avg_dist = np.mean([point[0] for point in cluster])
                    
                    start_angle = cluster[0][1]
                    end_angle = cluster[-1][1]
                    obstacle_clusters.append((start_angle, end_angle, avg_dist))
                    cluster = []  # Reset cluster

                prev_dist = None  # Reset for new cluster
                continue  # Skip max range points
            
            elif prev_dist is not None and np.abs(dist - prev_dist) <= dist_diff_threshold and dist < self.max_range:
                cluster.append((dist, adjusted_angle))
            else:
                # Finish the previous cluster and start a new one
                if cluster:
                    avg_dist = np.mean([point[0] for point in cluster])
                    start_angle = cluster[0][1]
                    end_angle = cluster[-1][1]

                    obstacle_clusters.append((start_angle, end_angle, avg_dist))  #  can use avg_dist - self.safety_distance
                    cluster = []  # Reset cluster
                

            prev_dist = dist  # Update for next iteration

        # Final cluster processing
        if cluster:
            avg_dist = np.mean([point[0] for point in cluster])
            start_angle = cluster[0][1]
            end_angle = cluster[-1][1]
            obstacle_clusters.append((start_angle, end_angle, avg_dist))  #  can use avg_dist - self.safety_distance

        return obstacle_clusters
    def cluster_objects(self, obstacle_clusters, radius):
        """Merges obstacle clusters if the Euclidean distance between them is below a threshold."""
        
        merged_clusters = []
        merge_distance_threshold = 4*radius
        boat_width = 2*radius
        
        if not obstacle_clusters:
            return  []# No clusters detected
        
        # Sort clusters by starting angle
        obstacle_clusters.sort(key=lambda x: x[0])
        
        current_cluster = list(obstacle_clusters[0])  # Convert tuple to list for modification

        for next_cluster in obstacle_clusters[1:]:
            start_angle_current, end_angle_current, dist_current = current_cluster
            start_angle_next, end_angle_next, dist_next = next_cluster

            # Convert angles to Cartesian coordinates
            x1 = dist_current * np.cos(end_angle_current)
            y1 = dist_current * np.sin(end_angle_current)
            x2 = dist_next * np.cos(start_angle_next)
            y2 = dist_next * np.sin(start_angle_next)

            # Compute Euclidean distance between the two clusters
            gap_distance = np.hypot(x2 - x1, y2 - y1)

            # Check if clusters are close enough to merge
            if gap_distance < merge_distance_threshold:
                # Merge: expand the current cluster to include the next one
                current_cluster[1] = end_angle_next  # Extend end angle
                current_cluster[2] = min(dist_current, dist_next)  # Use the nearest distance
            else:
                # Add dynamic safety margin before saving the cluster
                avg_dist = current_cluster[2]
                margin = np.arctan(boat_width / avg_dist)  # Dynamic angular margin

                # Apply safety margin
                current_cluster[0] -= margin  # Expand start angle
                current_cluster[1] += margin  # Expand end angle
                merged_clusters.append(tuple(current_cluster))
                current_cluster = list(next_cluster)
        
        # Apply safety margin to the last cluster and append
        avg_dist = current_cluster[2]
        margin = np.arctan(boat_width / avg_dist)
        current_cluster[0] -= margin
        current_cluster[1] += margin
        merged_clusters.append(tuple(current_cluster))
        
        return merged_clusters

    def expand_oriented_rectangle(self, rect_corners, expansion_size=1.0):
        """
        Expands an oriented bounding rectangle by the given expansion size (1 meter default).
        
        Parameters:
            rect_corners (numpy array): The 4x2 array representing the rectangle corners.
            expansion_size (float): The amount to expand on each side.
        
        Returns:
            numpy array: The new expanded rectangle corners (4x2 array).
        """
        # Compute rectangle center
        center = np.mean(rect_corners, axis=0)

        # Compute width and height before expansion
        width = np.linalg.norm(rect_corners[1] - rect_corners[0])  # Distance between first two points
        height = np.linalg.norm(rect_corners[2] - rect_corners[1])  # Distance between second and third points

        # New width and height after expansion
        new_width = width + 2 * expansion_size
        new_height = height + 2 * expansion_size

        # Compute edge vectors
        edge1 = (rect_corners[1] - rect_corners[0]) / max(width, 10**(-6))
        edge2 = (rect_corners[2] - rect_corners[1]) / max(height, 10**(-6))

        # Compute new expanded corners
        new_corners = np.array([
            center - (new_width / 2) * edge1 - (new_height / 2) * edge2,  # Bottom-left
            center + (new_width / 2) * edge1 - (new_height / 2) * edge2,  # Bottom-right
            center + (new_width / 2) * edge1 + (new_height / 2) * edge2,  # Top-right
            center - (new_width / 2) * edge1 + (new_height / 2) * edge2   # Top-left
        ])

        return new_corners

    def expand_oriented_rectangles(self, rectangles, expansion_size=1.0):
        expanded_rectangles = [self.expand_oriented_rectangle(rect, expansion_size) for rect in rectangles]
        return expanded_rectangles


