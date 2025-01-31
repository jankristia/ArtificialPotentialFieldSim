import numpy as np


# Lidar Simulator
class LidarSimulator:
    def __init__(self, obstacles, max_range=10, num_rays=72):
        self.max_range = max_range  # Max Lidar range (meters)
        self.num_rays = num_rays  # Number of rays (resolution)
        self.angles = np.linspace(-np.pi, np.pi, num_rays)  # 360° scan
        self.obstacles = obstacles

    def sense_obstacles(self, boat_x, boat_y, boat_psi):
        """Simulates Lidar scan by checking distance to obstacles"""
        distances = np.full(self.num_rays, self.max_range)  # Default: max range

        for i, angle in enumerate(self.angles):
            ray_angle = boat_psi + angle  # Global ray direction
            for obs_x, obs_y, obs_r in self.obstacles:
                # Calculate distance to the obstacle
                dx, dy = obs_x - boat_x, obs_y - boat_y
                distance_to_center = np.hypot(dx, dy)
                angle_to_center = np.arctan2(dy, dx)

                # Check if the ray intersects the obstacle
                if abs(np.sin(angle_to_center - ray_angle)) < (obs_r / distance_to_center):
                    distance = distance_to_center - obs_r
                    if distance < distances[i]:
                        distances[i] = max(distance, 0.1)  # Ensure no negative distances
        return distances
