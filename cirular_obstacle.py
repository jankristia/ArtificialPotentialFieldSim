import numpy as np

class CircularObstacle:
    def __init__(self, x, y, radius, vx, vy):
        """Initialize a moving obstacle with constant velocity."""
        self.x = x
        self.y = y
        self.radius = radius
        self.vx = vx
        self.vy = vy

    def update_position(self, dt):
        """Updates obstacle position using constant velocity in NED frame."""
        self.x += self.vx * dt
        self.y += self.vy * dt

    def get_position(self):
        return self.x, self.y, self.radius
