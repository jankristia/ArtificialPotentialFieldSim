import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from boat import BoatSimulator
import os
import csv

# Ensure 'log' directory exists
log_dir = "log"
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

# Open CSV file for writing
csv_path = os.path.join(log_dir, "log.csv")
csv_file = open(csv_path, mode='w', newline='')  # Open file in write mode
csv_writer = csv.writer(csv_file)

# Write CSV header
csv_writer.writerow(["Time", "Surge", "Sway", "YawRate", "LeftThrsut", "RightThrust", "DiffThrust", "CrossTrackError", "HeadingError", "Heading", "DesiredHeading"])
simulation_time = [0]  # Time tracker


# Environment setup with obstacles and waypoints
obstacles = [(30, 30, 2), (33, 30, 2),(30, 33, 2)]  # Circular obstacles in the environment
waypoints = np.array([[10, 10], [50,50]])  # Waypoints to follow

# Visualization
fig, ax = plt.subplots(figsize=(10, 10))
ax.set_xlim(0, 60)
ax.set_ylim(0, 60)
ax.set_title("Boat with Lidar and Obstacles")
ax.set_xlabel("X Position")
ax.set_ylabel("Y Position")

# Draw waypoints
ax.plot(waypoints[:, 0], waypoints[:, 1], 'ro--')

# Draw obstacles
for obs_x, obs_y, obs_r in obstacles:
    circle = plt.Circle((obs_x, obs_y), obs_r, color='r', alpha=0.5)
    ax.add_patch(circle)

boat_marker, = ax.plot([], [], 'bo', markersize=8)
boat_marker = plt.Circle((0, 0), 1, color='b', alpha=0.5)
ax.add_patch(boat_marker)
trail, = ax.plot([], [], 'b-', linewidth=1)
lidar_lines = [ax.plot([], [], 'g-', alpha=0.3)[0] for _ in range(72)]
heading_arrow = [ax.arrow(0, 0, 0, 0, head_width=1, color='b')]
force_arrows = []  # Stores force vectors

trajectory_x = []
trajectory_y = []

# Proxy handles for legend
legend_handles = [
    plt.Line2D([0], [0], color='r', marker='o', markersize=8, lw=2, label="Waypoints"),
    plt.Line2D([0], [0], color='b', marker='o', markersize=8, lw=0, label="Boat"),
    plt.Line2D([0], [0], color='g', lw=2, label="Repulsive Force"),
    plt.Line2D([0], [0], color='b', lw=2, label="Attractive Force"),
    plt.Line2D([0], [0], color='m', lw=2, label="Total Force"),
    plt.Line2D([0], [0], color='b', lw=2, label="Boat Heading"),
]

# Animation function
def animate(i):
    if boat.reached_goal or boat.collided:  # Stop if all waypoints are reached
        if boat.collided:
            print("Simulation ended: Boat collided with an obstacle.")
        elif boat.reached_goal:
            print("Simulation ended: Boat reached final waypoint.")
        else:
            print("Simulation ended by weird cause...")

        ani.event_source.stop()  # Stop the animation

        # Erase lidar and force lines for saving plot
        for line in lidar_lines:
            line.set_data([], [])

        for arrow in force_arrows:
            arrow.remove()
        force_arrows.clear()

        # Proxy handles for legend
        legend_handles = [
            plt.Line2D([0], [0], color='r', marker='o', markersize=8, lw=2, label="Waypoints"),
            plt.Line2D([0], [0], color='b', marker='o', markersize=8, lw=0, label="Boat"),
        ]


        # Add legend manually using proxy handles
        ax.legend(handles=legend_handles, loc="upper right")

        # Save the trajectory plot
        plot_path = os.path.join(log_dir, "trajectory_plot.png")
        plt.savefig(plot_path, dpi=300)
        print(f"Trajectory plot saved to {plot_path}")

        # Close the CSV file
        csv_file.close()
        print(f"Boat data saved to {csv_path}")


        return
     
    boat.update()
    boat_marker.set_center((boat.state[0], boat.state[1]))

    # Write data to CSV file
    csv_writer.writerow([simulation_time[0], boat.state[3], boat.state[4], boat.state[5], boat.thrust_left, boat.thrust_right, boat.thrust_diff, boat.cross_track_error, boat.prev_heading_error, boat.state[2], boat.desired_heading])
    simulation_time[0] += 1

    # Store the position in the trajectory list
    trajectory_x.append(boat.state[0])
    trajectory_y.append(boat.state[1])

    trail.set_data(trajectory_x, trajectory_y)

    # Lidar visualization
    distances = boat.lidar.sense_obstacles(boat.state[0], boat.state[1], boat.state[2])
    for j, (dist, line) in enumerate(zip(distances, boat.lidar.angles)):
        angle = boat.state[2] + line
        end_x = boat.state[0] + dist * np.cos(angle)
        end_y = boat.state[1] + dist * np.sin(angle)
        lidar_lines[j].set_data([boat.state[0], end_x], [boat.state[1], end_y])

        if dist > 12:
            lidar_lines[j].set_color('g')
        elif 8 < dist <= 12:
            lidar_lines[j].set_color('y')
        elif 4 < dist <= 8:
            lidar_lines[j].set_color('orange')
        else:
            lidar_lines[j].set_color('r')

    # Boat heading visualization
    heading_arrow[0].remove()
    heading_arrow[0] = ax.arrow(boat.state[0], boat.state[1], 2 * np.cos(boat.state[2]), 2 * np.sin(boat.state[2]))

    # Delete previous force arrows
    for arrow in force_arrows:
        arrow.remove()
    force_arrows.clear()

    # APF force visualization
    scale = 1  # Scaling factor for visualization
    rep_arrow = ax.arrow(boat.state[0], boat.state[1], scale * boat.repulsive_force[0], scale * boat.repulsive_force[1],
                         head_width=0.5, color='g')
    att_arrow = ax.arrow(boat.state[0], boat.state[1], scale * boat.attractive_force[0], scale * boat.attractive_force[1],
                         head_width=0.5, color='b')
    tot_arrow = ax.arrow(boat.state[0], boat.state[1], scale * boat.total_force[0], scale * boat.total_force[1],
                         head_width=0.5, color='m')

    force_arrows.extend([rep_arrow, att_arrow, tot_arrow])

    return [boat_marker, trail] + lidar_lines + force_arrows + heading_arrow

# Add legend manually using proxy handles
ax.legend(handles=legend_handles, loc="upper right")

# Simulation setup
boat = BoatSimulator(waypoints, obstacles)
ani = animation.FuncAnimation(fig, animate, frames=300, interval=50, blit=False)  # Set blit=False to prevent legend removal
plt.show()
