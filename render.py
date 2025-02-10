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
        self.lidar_lines = [self.ax.plot([], [], 'g-', alpha=0.3)[0] for _ in range(72)]
        self.heading_arrow = [self.ax.arrow(0, 0, 0, 0, head_width=1, color='b')]
        self.force_arrows = []
        self.trajectory_x = []
        self.trajectory_y = []

        self.legend_handles = [
            plt.Line2D([0], [0], color='r', marker='o', markersize=8, lw=2, label="Waypoints"),
            plt.Line2D([0], [0], color='b', marker='o', markersize=8, lw=0, label="Boat"),
            plt.Line2D([0], [0], color='g', lw=2, label="Repulsive Force"),
            plt.Line2D([0], [0], color='b', lw=2, label="Attractive Force"),
            plt.Line2D([0], [0], color='m', lw=2, label="Total Force"),
            plt.Line2D([0], [0], color='b', lw=2, label="Boat Heading"),
        ]

        self.ax.legend(handles=self.legend_handles, loc='upper right')

    
    def update_plot(self, boat):
        """Update the plot with new boat position, lidar data, and forces."""
        self.boat_marker.set_center((boat.state[0], boat.state[1]))
        self.trajectory_x.append(boat.state[0])
        self.trajectory_y.append(boat.state[1])
        self.trail.set_data(self.trajectory_x, self.trajectory_y)
        
        # Update lidar visualization
        distances = boat.lidar.sense_obstacles(boat.state[0], boat.state[1], boat.state[2])
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
        
        # Delete previous force arrows
        for arrow in self.force_arrows:
            arrow.remove()
        self.force_arrows.clear()

        # APF force visualization
        scale = 1  # Scaling factor for visualization
        rep_arrow = self.ax.arrow(boat.state[0], boat.state[1], scale * boat.repulsive_force[0], scale * boat.repulsive_force[1],
                                 head_width=0.5, color='g')
        att_arrow = self.ax.arrow(boat.state[0], boat.state[1], scale * boat.attractive_force[0], scale * boat.attractive_force[1],
                                 head_width=0.5, color='b')
        tot_arrow = self.ax.arrow(boat.state[0], boat.state[1], scale * boat.total_force[0], scale * boat.total_force[1],
                                 head_width=0.5, color='m')

        self.force_arrows.extend([rep_arrow, att_arrow, tot_arrow])

        return self.boat_marker, self.trail, self.lidar_lines + self.heading_arrow + self.force_arrows
    
    def prepare_plot_for_saving(self):
        """Prepare the plot for saving by removing force arrows and legend."""
        for arrow in self.force_arrows:
            arrow.remove()
        self.force_arrows.clear()

        for line in self.lidar_lines:
            line.set_data([], [])

        self.legend_handles = [
            plt.Line2D([0], [0], color='r', marker='o', markersize=8, lw=2, label="Waypoints"),
            plt.Line2D([0], [0], color='b', marker='o', markersize=8, lw=0, label="Boat"),
        ]

        self.ax.legend(handles=self.legend_handles, loc='upper right')