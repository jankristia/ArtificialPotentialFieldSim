import matplotlib.pyplot as plt
import numpy as np

class Render:
    def __init__(self, waypoints, obstacles):
        """Initialize the rendering environment."""
        self.fig, self.ax = plt.subplots(figsize=(10, 10))
        self.ax.set_xlim(0, 60)
        self.ax.set_ylim(0, 60)
        self.ax.set_title("Boat with Obstacles")
        self.ax.set_xlabel("X Position")
        self.ax.set_ylabel("Y Position")
        
        # Draw waypoints
        self.ax.plot(waypoints[:, 0], waypoints[:, 1], 'ro--', label="Waypoints")
        
        # Draw obstacles
        for obs_x, obs_y, obs_r in obstacles:
            circle = plt.Circle((obs_x, obs_y), obs_r, color='r', alpha=0.5)
            self.ax.add_patch(circle)
        
        self.boat_marker = plt.Circle((0, 0), 1, color='b', alpha=0.5)
        self.ax.add_patch(self.boat_marker)
        self.trail, = self.ax.plot([], [], 'b-', linewidth=1)
        self.lidar_lines = [self.ax.plot([], [], 'g-', alpha=0.3)[0] for _ in range(128)]    # Must be the same as num_rays in lidar.py
        self.heading_arrow = [self.ax.arrow(0, 0, 0, 0, head_width=1, color='b')]
        self.trajectory_x = []
        self.trajectory_y = []
        self.obstacle_patches = []
        self.moving_obstacle_patches = []

        self.legend_handles = [
            plt.Line2D([0], [0], color='r', marker='o', markersize=8, lw=2, label="Waypoints"),
            plt.Line2D([0], [0], color='b', marker='o', markersize=8, lw=0, label="Boat"),
            plt.Line2D([0], [0], color='b', lw=2, label="Boat Heading"),
            plt.Line2D([0], [0], color='purple', lw=2, label="Detected Obstacles")
        ]

        self.ax.legend(handles=self.legend_handles, loc='upper right')

    def update_plot(self, boat):
        """Update the plot with new boat position, lidar data, and forces."""
        self.boat_marker.set_center((boat.state[0], boat.state[1]))
        self.trajectory_x.append(boat.state[0])
        self.trajectory_y.append(boat.state[1])
        self.trail.set_data(self.trajectory_x, self.trajectory_y)
        
        # Update lidar visualization
        distances = boat.lidar.sense_obstacles(boat.state[0], boat.state[1], boat.state[2], boat.moving_obstacles)
        for j, (dist, line) in enumerate(zip(distances, boat.lidar.angles)):
            angle = boat.state[2] + line
            end_x = boat.state[0] + dist * np.cos(angle)
            end_y = boat.state[1] + dist * np.sin(angle)
            self.lidar_lines[j].set_data([boat.state[0], end_x], [boat.state[1], end_y])

            if 15 < dist <= 20:
                self.lidar_lines[j].set_color('g')
            elif 10 < dist <= 15:
                self.lidar_lines[j].set_color('y')
            elif 5 < dist <= 10:
                self.lidar_lines[j].set_color('orange')
            elif 0 <= dist <= 5:
                self.lidar_lines[j].set_color('r')
            
        # Update heading arrow
        self.heading_arrow[0].remove()
        self.heading_arrow[0] = self.ax.arrow(boat.state[0], boat.state[1], 2 * np.cos(boat.state[2]), 2 * np.sin(boat.state[2]))

        # Update detected obstacle clusters
        for patch in self.obstacle_patches:
            patch.remove()
        self.obstacle_patches.clear()

        for start_angle, end_angle, avg_dist in boat.obstacle_clusters:
            arc_points = np.linspace(start_angle, end_angle, 10)
            arc_x = boat.state[0] + avg_dist * np.cos(arc_points)
            arc_y = boat.state[1] + avg_dist * np.sin(arc_points)
            patch, = self.ax.plot(arc_x, arc_y, 'purple', linewidth=2)
            self.obstacle_patches.append(patch)
        
        # Update moving obstacles
        for patch in self.moving_obstacle_patches:
            patch.remove()

        self.moving_obstacle_patches.clear()

        for obs in boat.moving_obstacles:
            circle = plt.Circle((obs.x, obs.y), obs.radius, color='orange', alpha=0.5)
            self.ax.add_patch(circle)
            self.moving_obstacle_patches.append(circle)
        
        return self.boat_marker, self.trail, self.lidar_lines + self.heading_arrow
    
    def prepare_plot_for_saving(self):
        """Prepare the plot for saving by removing force arrows and legend."""
        for line in self.lidar_lines:
            line.set_data([], [])

        self.legend_handles = [
            plt.Line2D([0], [0], color='r', marker='o', markersize=8, lw=2, label="Waypoints"),
            plt.Line2D([0], [0], color='b', marker='o', markersize=8, lw=0, label="Boat"),
        ]

        self.ax.legend(handles=self.legend_handles, loc='upper right')