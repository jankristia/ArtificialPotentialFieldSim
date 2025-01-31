import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from boat import BoatSimulator

# Environment setup with obstacles and waypoints
obstacles = [
    (15, 7, 2),   # (x, y, radius)
    (15, 14, 2),
]  # Circular obstacles in the environment

waypoints = np.array([[10, 5], [44, 12]])


# Visualization
fig, ax = plt.subplots(figsize=(10, 6))
ax.set_xlim(0, 45)
ax.set_ylim(0, 15)
ax.set_title("Boat with Lidar and Obstacles")
ax.set_xlabel("X Position")
ax.set_ylabel("Y Position")

# Draw waypoints
ax.plot(waypoints[:, 0], waypoints[:, 1], 'ro--', label="Waypoints")

# Draw obstacles
for obs_x, obs_y, obs_r in obstacles:
    circle = plt.Circle((obs_x, obs_y), obs_r, color='r', alpha=0.5)
    ax.add_patch(circle)

boat_marker, = ax.plot([], [], 'bo', markersize=8, label="Boat")
trail, = ax.plot([], [], 'b-', linewidth=1)
lidar_lines = [ax.plot([], [], 'g-', alpha=0.3)[0] for _ in range(72)]

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

    return [boat_marker, trail] + lidar_lines

# Simulation setup
boat = BoatSimulator(waypoints, obstacles)
ani = animation.FuncAnimation(fig, animate, frames=300, interval=50, blit=True)
plt.legend()
plt.show()
