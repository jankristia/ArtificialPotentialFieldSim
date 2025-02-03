import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from boat import BoatSimulator

# Environment setup with obstacles and waypoints
obstacles = [(35, 30, 2)]  # Circular obstacles in the environment
waypoints = np.array([[10, 10], [50, 50]])  # Waypoints to follow

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
trail, = ax.plot([], [], 'b-', linewidth=1)
lidar_lines = [ax.plot([], [], 'g-', alpha=0.3)[0] for _ in range(72)]
force_arrows = []  # Stores force vectors

# Proxy handles for legend
legend_handles = [
    plt.Line2D([0], [0], color='r', marker='o', markersize=8, lw=2, label="Waypoints"),
    plt.Line2D([0], [0], color='b', marker='o', markersize=8, lw=0, label="Boat"),
    plt.Line2D([0], [0], color='g', lw=2, label="Repulsive Force"),
    plt.Line2D([0], [0], color='b', lw=2, label="Attractive Force"),
    plt.Line2D([0], [0], color='m', lw=2, label="Total Force")
]

# Animation function
def animate(i):
    boat.update()
    boat_marker.set_data(boat.state[0], boat.state[1])
    trail.set_data(trail.get_xdata() + [boat.state[0]], trail.get_ydata() + [boat.state[1]])

    # Lidar visualization
    distances = boat.lidar.sense_obstacles(boat.state[0], boat.state[1], boat.state[2])
    for j, (dist, line) in enumerate(zip(distances, boat.lidar.angles)):
        angle = boat.state[2] + line
        end_x = boat.state[0] + dist * np.cos(angle)
        end_y = boat.state[1] + dist * np.sin(angle)
        lidar_lines[j].set_data([boat.state[0], end_x], [boat.state[1], end_y])

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

    return [boat_marker, trail] + lidar_lines + force_arrows

# Add legend manually using proxy handles
ax.legend(handles=legend_handles, loc="upper right")

# Simulation setup
boat = BoatSimulator(waypoints, obstacles)
ani = animation.FuncAnimation(fig, animate, frames=300, interval=50, blit=False)  # Set blit=False to prevent legend removal
plt.show()
