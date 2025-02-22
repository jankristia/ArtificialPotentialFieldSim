import numpy as np

class ScenarioGenerator:
    def __init__(self, scenario_name="default"):
        self.scenario_name = scenario_name
        self.waypoints = []
        self.static_obstacles = []
        self.moving_obstacles = []
        self.setup_scenario()
    
    def setup_scenario(self):
        if self.scenario_name == "moving_obstacle_head_on":
            self.waypoints = np.array([[10, 10], [35, 35]])
            self.static_obstacles = []
            self.moving_obstacles = [(30, 30, 1, -0.5, -0.5)]
        elif self.scenario_name == "moving_obstacle_crossing_right":
            self.waypoints = np.array([[10, 10], [50, 50]])
            self.static_obstacles = []
            self.moving_obstacles = [(25, 5, 2, -0.5, 0.5)]
        elif self.scenario_name == "moving_obstacle_crossing_left":
            self.waypoints = np.array([[10, 10], [50, 50]])
            self.static_obstacles = []
            self.moving_obstacles = [(5, 25, 2, 0.5, -0.5)]
        elif self.scenario_name == "moving_obstacle_overtaking":
            self.waypoints = np.array([[10, 10], [50, 50]])
            self.static_obstacles = []
            self.moving_obstacles = [(10, 10, 2, 0.15, 0.15)]
        elif self.scenario_name == "one_small_obstacle":
            self.waypoints = np.array([[10, 10], [50, 50]])
            self.static_obstacles = [(30, 30, 2)]
            self.moving_obstacles = []
        elif self.scenario_name == "complex_obstacles":
            self.waypoints = np.array([[5, 5], [30, 15], [40, 40], [10,50]])
            self.static_obstacles = [(20, 20, 3), (35, 30, 4), (45, 25, 2)]
            self.moving_obstacles = [] # (15, 40, 2, 0.5, 0.5)
        elif self.scenario_name == "two_obstacles":
            self.waypoints = np.array([[10, 10], [50, 50]])
            self.static_obstacles = [(34, 30, 2), (30, 34, 2)]
            self.moving_obstacles = []
        elif self.scenario_name == "one_large_obstacle":
            self.waypoints = np.array([[10, 10], [50, 50]])
            self.static_obstacles = [(34, 34, 5)]
            self.moving_obstacles = []
        else:
            # Default scenario
            self.waypoints = np.array([[10, 10], [50, 50]])
            self.static_obstacles = [(30, 30, 2)]
            self.moving_obstacles = [ (15, 40, 2, 0.5, -0.5)]
    
    def get_scenario(self):
        return self.waypoints, self.static_obstacles, self.moving_obstacles
