import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from boat import BoatSimulator
from scenarios import ScenarioGenerator
from csv_logging import open_csv_file, close_and_save_csv_file
from render import Render
from moving_obstacle import MovingObstacle

log_dir, csv_file, csv_writer, simulation_time = open_csv_file()

scenario = ScenarioGenerator("moving_obstacle_head_on")        # moving_obstacle_crossing, moving_obstacle_head_on, moving_obstacle_overtaking, one_small_obstacle, complex_obstacles, two_obstacles, one_large_obstacle
waypoints, static_obstacles, moving_obstacles_data = scenario.get_scenario()

moving_obstacles = [MovingObstacle(*data) for data in moving_obstacles_data]


render = Render(waypoints, static_obstacles)

# Animation function
def animate(i):
    if boat.reached_goal or boat.collided: # End simulation
        if boat.collided:
            print("Simulation ended: Boat collided with an obstacle.")
        elif boat.reached_goal:
            print("Simulation ended: Boat reached final waypoint.")

        ani.event_source.stop()
        
        close_and_save_csv_file(csv_file, log_dir)

        return render.prepare_plot_for_saving()

    boat.update()

    csv_writer.writerow([simulation_time[0], boat.state[3], boat.state[4], boat.state[5], boat.thrust_left, boat.thrust_right, boat.thrust_diff, boat.cross_track_error, boat.prev_heading_error, boat.state[2], boat.desired_heading])
    simulation_time[0] += 1

    return render.update_plot(boat)

# Simulation setup
boat = BoatSimulator(waypoints, static_obstacles, moving_obstacles)
ani = animation.FuncAnimation(render.fig, animate, frames=300, interval=100, blit=False)

# # Save the animation as a video,
# video_path = "log/simulation_video.mp4"  # Change to .gif for GIF output
# Writer = animation.FFMpegWriter(fps=20, metadata={'title': 'USV Simulation'}, bitrate=1800)
# ani.save(video_path, writer=Writer)


plt.show()

