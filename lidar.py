import numpy as np

class LidarSimulator:
    def __init__(self, static_obstacles, max_range=20, num_rays=128):
        self.max_range = max_range  # Max Lidar range (meters)
        self.num_rays = num_rays  # Number of rays (resolution)
        self.angles = np.linspace(-1/2*np.pi, 1/2*np.pi, num_rays)  # 180° scan in ENU, 0° is East
        # self.angles = np.linspace(np.pi, 0, num_rays)  # 180° scan in NED, 0° is North
        self.static_obstacles = static_obstacles
        self.obstacles = []

    def ray_circle_intersection(self, ray_origin, ray_dir, circle_center, circle_radius):
        """ Computes the intersection of a ray with a circular obstacle """
        oc = ray_origin - circle_center
        a = np.dot(ray_dir, ray_dir)
        b = 2.0 * np.dot(oc, ray_dir)
        c = np.dot(oc, oc) - circle_radius**2
        
        discriminant = b**2 - 4*a*c
        if discriminant < 0:
            return None  # No intersection
        
        t1 = (-b - np.sqrt(discriminant)) / (2*a)
        t2 = (-b + np.sqrt(discriminant)) / (2*a)
        
        if t1 > 0 and t1 < self.max_range:
            return t1  # Entry point
        elif t2 > 0 and t2 < self.max_range:
            return t2  # Exit point
        return None
    
    def sense_obstacles(self, boat_x, boat_y, boat_psi, moving_obstacles=[]):
        """ Simulates LiDAR scan by checking exact intersection points with obstacles """
        distances = np.full(self.num_rays, float(self.max_range))  # Default: max range
        ray_origin = np.array([boat_x, boat_y])


        self.obstacles = self.static_obstacles + [(obs.x, obs.y, obs.radius) for obs in moving_obstacles]
        
        for i, angle in enumerate(self.angles):
            ray_angle = angle + boat_psi
            ray_dir = np.array([np.cos(ray_angle), np.sin(ray_angle)])
            
            for obs_x, obs_y, obs_r in self.obstacles:
                circle_center = np.array([obs_x, obs_y])
                intersection = self.ray_circle_intersection(ray_origin, ray_dir, circle_center, obs_r)
                
                if intersection is not None:
                    distances[i] = min(distances[i], intersection)
                    
        return distances
    
    def cluster_lidar_data(self, boat_state, moving_obstacles):
        """Clusters LiDAR data into detected obstacles, adding a safety margin to each obstacle."""

        lidar_readings = self.sense_obstacles(boat_state[0], boat_state[1], boat_state[2], moving_obstacles)
        obstacle_clusters = []
        prev_dist = None
        dist_diff_threshold = 1.0
        cluster = []

        for dist, angle in zip(lidar_readings, self.angles):
            adjusted_angle = angle + boat_state[2]  # Convert to World frame

            if dist >= self.max_range:
                # End the current cluster if there is one
                if cluster:
                    
                    avg_dist = np.mean([point[0] for point in cluster])
                    
                    start_angle = cluster[0][1]
                    end_angle = cluster[-1][1]
                    obstacle_clusters.append((start_angle, end_angle, avg_dist))  #  can use avg_dist - self.safety_distance
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
        merge_distance_threshold = 4*radius  # Maximum allowed gap (meters) between clusters to merge
        boat_width = 2*radius  # Boat width for dynamic safety margin
        
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
        
        return merged_clusters  # Update with refined obstacles



