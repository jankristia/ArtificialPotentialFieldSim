import numpy as np
from helpers import calculate_relative_pos_velocity



def calculate_future_absolute_positions(obs, time_horizon=2):
    """ Calculate future absolute positions of the moving obstacles """
    future_x = obs.x + obs.vx * time_horizon
    future_y = obs.y + obs.vy * time_horizon
    return np.array([future_x, future_y])


def compute_velocity_obstacles(vessel_state, moving_obstacles, radius, max_time_horizon=10, time_step=2):
    """ 
    Compute Linear Velocity Obstacles (LVO) considering multiple future time horizons.
    """

    forbidden_headings = []
    future_obstacle_positions = []

    for obs in moving_obstacles:
        # Get initial relative position and velocity
        relative_position, relative_velocity = calculate_relative_pos_velocity(vessel_state, obs)

        # Store all forbidden heading angles across multiple future timesteps
        time_horizon_forbidden_headings = []
        future_positions = []

        # Iterate through multiple future timesteps
        for t in range(1, max_time_horizon + 1, time_step):
            # Predict future absolute position of the obstacle
            future_absolute_position = calculate_future_absolute_positions(obs, t)

            # Compute future relative position (own ship remains fixed)
            future_relative_position = future_absolute_position - vessel_state[:2]

            # Compute distance to the future obstacle position
            distance = np.linalg.norm(future_relative_position)
            combined_radius = radius + obs.radius
            expanded_radius = 1.5*combined_radius  # Can be increased for safety margin

            # if distance < combined_radius:
            #     print(f"PANIC! Collision Imminent at t={t}!")
            #     return [(-np.pi, np.pi)], []  # Full heading restriction and no future positions

            cone_width = np.arcsin(min(1, expanded_radius / (distance + 10**-6)))  # Avoid math domain error
            
            # Compute heading based relative position
            base_heading = np.arctan2(future_relative_position[1], future_relative_position[0])

            # Compute left and right forbidden headings
            left_forbidden_heading = base_heading - cone_width
            right_forbidden_heading = base_heading + cone_width

            # Normalize angles to **ensure correct quadrant**
            left_forbidden_heading = (left_forbidden_heading + np.pi) % (2 * np.pi) - np.pi
            right_forbidden_heading = (right_forbidden_heading + np.pi) % (2 * np.pi) - np.pi

            time_horizon_forbidden_headings.append((right_forbidden_heading, left_forbidden_heading))

            # Store absolute position of the obstacle for rendering
            future_positions.append((future_absolute_position[0], future_absolute_position[1], expanded_radius))

        # Merge heading intervals from different time horizons
        forbidden_headings.extend(time_horizon_forbidden_headings)
        # print(f"Forbidden Headings: {forbidden_headings}")
        future_obstacle_positions.extend(future_positions)

    return forbidden_headings, future_obstacle_positions  # Return both hea