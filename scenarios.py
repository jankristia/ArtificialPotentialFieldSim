import numpy as np

class ScenarioGenerator:
    def __init__(self, scenario_name="default"):
        self.scenario_name = scenario_name
        self.waypoints = []
        self.static_obstacles = []
        self.moving_obstacles = []
        self.setup_scenario()
    
    def setup_scenario(self):
        if self.scenario_name == "moving_obstacle":
            self.waypoints = np.array([[10, 10], [50, 50]])
            self.static_obstacles = [(20, 20, 2)]
            self.moving_obstacles = [(15, 10, 2, -0.5, 0.5), (15, 40, 2, 0.5, -0.5)]
        else:
            # Default scenario
            self.waypoints = np.array([[10, 10], [50, 50]])
            self.static_obstacles = [(30, 30, 2)]
    
    def get_scenario(self):
        return self.waypoints, self.static_obstacles, self.moving_obstacles
