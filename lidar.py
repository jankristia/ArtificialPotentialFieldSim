import numpy as np

class LidarSimulator:
    def __init__(self, obstacles, max_range=20, num_rays=64):
        self.max_range = max_range  # Max Lidar range (meters)
        self.num_rays = num_rays  # Number of rays (resolution)
        self.angles = np.linspace(-1/2*np.pi, 1/2*np.pi, num_rays)  # 180° scan
        self.obstacles = obstacles

    def sense_obstacles(self, boat_x, boat_y, boat_psi):
        """Simulates Lidar scan by checking distance to obstacles"""
        distances = np.full(self.num_rays, self.max_range)  # Default: max range

        for i, angle in enumerate(self.angles):
            ray_angle = boat_psi + angle  # Global ray direction

            for obs_x, obs_y, obs_r in self.obstacles:
                dx, dy = obs_x - boat_x, obs_y - boat_y
                distance_to_center = np.hypot(dx, dy)
                angle_to_center = np.arctan2(dy, dx)

                # Compute the signed angular difference
                angle_diff = np.arctan2(np.sin(angle_to_center - ray_angle), np.cos(angle_to_center - ray_angle))

                # Ensure the obstacle is in front of the Lidar ray (within ±90 degrees)
                if abs(angle_diff) < np.pi / 2:
                    # Compute shortest distance along the ray
                    perpendicular_distance = abs(distance_to_center * np.sin(angle_diff))

                    if perpendicular_distance < obs_r:
                        # Compute distance to the edge of the obstacle along the ray
                        distance_along_ray = np.sqrt(distance_to_center**2 - perpendicular_distance**2) - obs_r

                        # Store the closest valid obstacle
                        if 0 < distance_along_ray < distances[i]:
                            distances[i] = max(distance_along_ray, 0.1)  # Ensure non-negative distances
                            
        return distances
