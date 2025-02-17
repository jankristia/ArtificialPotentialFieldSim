import numpy as np

class ScenarioGenerator:
    def __init__(self, scenario_name="default"):
        self.scenario_name = scenario_name
        self.waypoints = []
        self.obstacles = []
        self.setup_scenario()
    
    def setup_scenario(self):
        if self.scenario_name == "one_small_obstacle":
            self.waypoints = np.array([[10, 10], [50, 50]])
            self.obstacles = [(30, 30, 2)]  # Circular obstacles
        elif self.scenario_name == "complex_obstacles":
            self.waypoints = np.array([[5, 5], [30, 15], [40, 40], [10,50]])
            self.obstacles = [(20, 20, 3), (35, 30, 4), (45, 25, 2)]
        elif self.scenario_name == "two_obstacles":
            self.waypoints = np.array([[10, 10], [50, 50]])
            self.obstacles = [(34, 30, 2), (30, 34, 2)]  # Circular obstacles
        elif self.scenario_name == "one_large_obstacle":
            self.waypoints = np.array([[10, 10], [50, 50]])
            self.obstacles = [(34, 34, 5)]  # Circular obstacles
        else:
            # Default scenario
            self.waypoints = np.array([[10, 10], [50, 50]])
            self.obstacles = [(30, 30, 2)]
    
    def get_scenario(self):
        return self.waypoints, self.obstacles
