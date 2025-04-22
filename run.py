import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from boat import BoatSimulator
from scenarios import ScenarioGenerator
from csv_logging import open_csv_file, close_and_save_csv_file
from render import Render

log_dir, csv_file, csv_writer, simulation_time = open_csv_file()

scenario = ScenarioGenerator("straight_line")        # two_moving_obstacles, , moving_obstacle_crossing_left_right_and_front, moving_obstacle_crossing_left, moving_obstacle_crossing_right, moving_obstacle_head_on, moving_obstacle_overtaking, one_small_obstacle, complex_obstacles, two_obstacles, one_large_obstacle
waypoints, circular_obstacles, isNoise = scenario.get_scenario()

render = Render(waypoints, circular_obstacles)

# Animation function
def animate(i):
    if boat.reached_goal or boat.collided: # End simulation
        if boat.collided:
            print("Simulation ended: Boat collided with an obstacle.")
        elif boat.reached_goal:
            print("Simulation ended: Boat reached final waypoint.")

        ani.event_source.stop()
        render.prepare_plot_for_saving()
        close_and_save_csv_file(csv_file, log_dir)
        return render.prepare_plot_for_saving()
    boat.update()

    # if simulation_time[0] % 30 == 0:
    #     output_folder = "log/frames"
    #     output_filename = "frame_" + str(simulation_time[0]) + ".pdf"
    #     render.save_plot_in_video(output_folder, output_filename, simulation_time[0])

    csv_writer.writerow([simulation_time[0], boat.state[3], boat.state[4], boat.state[5], boat.pwm_left, boat.pwm_right, boat.pwm_diff, boat.cross_track_error, boat.cross_track_error_no_noise, boat.prev_heading_error, boat.state[2], boat.ColAv_desired_heading, boat.LOS_desired_heading, boat.shortest_object_dist])
    simulation_time[0] += 1

    return render.update_plot(boat)

# Simulation setup
boat = BoatSimulator(waypoints, circular_obstacles, isNoise)
ani = animation.FuncAnimation(render.fig, animate, frames=600, interval=100, blit=False)

# # Save the animation as a video,
# video_path = "log/simulation_video.mp4"  # Change to .gif for GIF output
# Writer = animation.FFMpegWriter(fps=20, metadata={'title': 'USV Simulation'}, bitrate=1800)
# ani.save(video_path, writer=Writer)


plt.show()

