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
