import numpy as np
from cirular_obstacle import CircularObstacle

class ScenarioGenerator:
    def __init__(self, scenario_name="default"):
        self.scenario_name = scenario_name
        self.waypoints = []
        self.circular_obstacles = []
        self.setup_scenario()
    
    def setup_scenario(self):
        if self.scenario_name == "moving_obstacle_head_on":
            self.waypoints = np.array([[10, 10], [50, 50]])
            self.circular_obstacles = [CircularObstacle(30, 30, 1, -0.5, -0.5)]
        elif self.scenario_name == "moving_obstacle_crossing_right":
            self.waypoints = np.array([[10, 10], [50, 50]])
            self.circular_obstacles = [CircularObstacle(25, 5, 1, -0.5, 0.5)]
        elif self.scenario_name == "moving_obstacle_crossing_left":
            self.waypoints = np.array([[10, 10], [50, 50]])
            self.circular_obstacles = [CircularObstacle(5, 25, 1, 0.5, -0.5)]
        elif self.scenario_name == "moving_obstacle_crossing_left_right_and_front":
            self.waypoints = np.array([[10, 10], [50, 50]])
            self.circular_obstacles = [CircularObstacle(5, 25, 1, 0.5, -0.5), CircularObstacle(40, 5, 1, -0.5, 0.5), CircularObstacle(30, 30, 1, -0.5, -0.5), CircularObstacle(10, 15, 2, 0, 0), CircularObstacle(16, 24, 2, 0, 0)] 
        elif self.scenario_name == "moving_obstacle_overtaking":
            self.waypoints = np.array([[10, 10], [50, 50]])
            self.circular_obstacles = [CircularObstacle(10, 10, 1, 0.5, 0.5)]
        elif self.scenario_name == "moving_obstacle_overtaking_head_on_crossing":
            self.waypoints = np.array([[10, 10], [50, 50]])
            self.circular_obstacles = [CircularObstacle(10, 10, 1, 0.15, 0.15), CircularObstacle(45, 45, 1, -0.5, -0.5), CircularObstacle(40, 10, 1, -0.25, 0.65)]
        elif self.scenario_name == "two_moving_obstacles":
            self.waypoints = np.array([[10, 10], [50, 50]])
            self.circular_obstacles = [CircularObstacle(5, 25, 1, 0.5, -0.5), CircularObstacle(45, 45, 1, -0.5, -0.5), CircularObstacle(45, 10, 1, -0.5, 0.5)]
        elif self.scenario_name == "one_small_obstacle":
            self.waypoints = np.array([[10, 10], [50, 50]])
            self.circular_obstacles = [CircularObstacle(30, 30, 2, 0, 0)]
        elif self.scenario_name == "complex_obstacles":
            self.waypoints = np.array([[5, 5], [30, 15], [40, 40], [10,50]])
            self.circular_obstacles = [CircularObstacle(20, 20, 3, 0, 0), CircularObstacle(35, 30, 4, 0, 0), CircularObstacle(45, 25, 2, 0, 0)]
        elif self.scenario_name == "one_large_obstacle":
            self.waypoints = np.array([[10, 10], [50, 50]])
            self.circular_obstacles = [CircularObstacle(34, 34, 5, 0, 0)]
        else:
            # Default scenario
            self.waypoints = np.array([[10, 10], [50, 50]])
            self.circular_obstacles = [ CircularObstacle(15, 40, 2, 0.5, -0.5), CircularObstacle(30, 30, 2, 0, 0)]
    
    def get_scenario(self):
        return self.waypoints, self.circular_obstacles
