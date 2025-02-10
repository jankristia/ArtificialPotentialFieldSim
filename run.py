import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from boat import BoatSimulator
from scenarios import ScenarioGenerator
from csv_logging import open_csv_file, close_and_save_csv_file
from render import Render

log_dir, csv_file, csv_writer, simulation_time = open_csv_file()

scenario = ScenarioGenerator("simple_path")            # ("two_obstacles")
waypoints, obstacles = scenario.get_scenario()

render = Render(waypoints, obstacles)

# Animation function
def animate(i):
    if boat.reached_goal or boat.collided: # End simulation
        if boat.collided:
            print("Simulation ended: Boat collided with an obstacle.")
        elif boat.reached_goal:
            print("Simulation ended: Boat reached final waypoint.")
        else:
            print("Simulation ended by weird cause...")

        ani.event_source.stop()
        
        close_and_save_csv_file(csv_file, log_dir)

        return render.prepare_plot_for_saving()

    boat.update()

    csv_writer.writerow([simulation_time[0], boat.state[3], boat.state[4], boat.state[5], boat.thrust_left, boat.thrust_right, boat.thrust_diff, boat.cross_track_error, boat.prev_heading_error, boat.state[2], boat.desired_heading])
    simulation_time[0] += 1

    return render.update_plot(boat)

# Simulation setup
boat = BoatSimulator(waypoints, obstacles)
ani = animation.FuncAnimation(render.fig, animate, frames=300, interval=50, blit=False)
plt.show()
