import numpy as np
from helpers import calculate_relative_pos_velocity, ssa
    

def tcpa_dcpa_vo_check(vessel_state, moving_objects, velocity_magnitude, lidar_angles, safety_radius, sensor_range):
    """Check if the current heading is dangerous using TCPA, DCPA, and VO."""
    forbidden_headings = []

    for obs in moving_objects:
        relative_position, relative_velocity = calculate_relative_pos_velocity(vessel_state, obs)
        distance = np.linalg.norm(relative_position)
        relative_heading = np.arctan2(relative_position[1], relative_position[0])
        heading_diff = ssa(relative_heading - vessel_state[2])

        # Only detect objects within sensor range and in front of the vessel
        if distance < sensor_range and np.abs(heading_diff) < np.pi/2:
            if distance < 1e-6:
                continue
                
            tcpa = np.dot(relative_position, relative_velocity) / (np.linalg.norm(relative_velocity)**2 + 1e-6)
            dcpa = np.linalg.norm((relative_position[0]*relative_velocity[1] - relative_position[1]*relative_velocity[0]) / 
                                (np.linalg.norm(relative_velocity) + 1e-6))
            if 0 < tcpa < 10 and 0 < dcpa <10:

                vo_left, vo_right = compute_velocity_obstacle(vessel_state, obs, velocity_magnitude, safety_radius)


                # Check each possible heading for safety
                for candidate_angle in lidar_angles:
                    candidate_heading = ssa(vessel_state[2] + candidate_angle)
                    candidate_velocity = np.array([velocity_magnitude * np.cos(candidate_heading), 
                                                velocity_magnitude * np.sin(candidate_heading)])
                    
                    candidate_relative_velocity = candidate_velocity - np.array([obs.vx, obs.vy])

                    candidate_relative_velocity_heading = np.arctan2(candidate_relative_velocity[1], candidate_relative_velocity[0])

                    # Check if candidate heading lies outside the VO cone
                    if vo_right <= candidate_relative_velocity_heading <= vo_left:
                        forbidden_headings.append(candidate_heading)

    forbidden_intervals = merge_intervals(forbidden_headings)

    return forbidden_intervals


def compute_velocity_obstacle(vessel_state, obstacle, velocity_magnitude, vessel_radius):
    """Compute the Velocity Obstacle (VO) for the given obstacle."""
    relative_position, relative_velocity = calculate_relative_pos_velocity(vessel_state, obstacle)
    distance = np.linalg.norm(relative_position)
    
    vo_radius = obstacle.radius + vessel_radius
    vo_half_angle = np.arcsin(vo_radius / max(distance, vo_radius + 1e-6))
    
    relative_heading = np.arctan2(relative_position[1], relative_position[0])
    
    vo_left = relative_heading + vo_half_angle
    vo_right = relative_heading - vo_half_angle

    return vo_left, vo_right

def merge_intervals(angles):
    """Merge consecutive forbidden angles into angle intervals."""
    if not angles:
        return []
    
    angles = sorted(angles)
    intervals = []
    
    start = angles[0]
    prev = start

    for angle in angles[1:]:
        if abs(angle - prev) > np.radians(2):
            intervals.append((start, prev))
            start = angle
        prev = angle

    intervals.append((start, prev))
    return intervals
