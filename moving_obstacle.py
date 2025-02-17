import numpy as np

class MovingObstacle:
    def __init__(self, x, y, radius, vx, vy):
        """Initialize a moving obstacle with constant velocity."""
        self.x = x  # NED frame position (meters)
        self.y = y
        self.radius = radius  # Obstacle radius
        self.vx = vx  # Velocity in North (m/s)
        self.vy = vy  # Velocity in East (m/s)

    def update_position(self, dt):
        """Updates obstacle position using constant velocity in NED frame."""
        self.x += self.vx * dt
        self.y += self.vy * dt

    def get_position(self):
        return self.x, self.y, self.radius
