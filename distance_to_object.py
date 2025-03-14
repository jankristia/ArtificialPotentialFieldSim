from shapely.geometry import LineString, Polygon, Point
import numpy as np

def distance_to_nearest_object(rectangles, angle, vessel_state, max_distance=20.0):
    """
    Calculates the minimum distance a ray can travel in a given angle before hitting an object,
    considering the vessel's position and heading.

    Parameters:
        rectangles (list of numpy arrays): List of 4x2 arrays representing the corners of the detected rectangles.
        angle (float): The angle in radians representing the travel direction relative to the vessel's heading.
        vessel_state (tuple): The vessel's (x, y, heading) in ENU frame.
        max_distance (float): The maximum range if no obstacle is hit.

    Returns:
        float: The distance to the closest object in the given direction.
    """
    # Extract vessel state
    vessel_x, vessel_y, vessel_heading = vessel_state[:3]
    
    # Adjust angle for vessel heading
    global_angle = angle + vessel_heading
    
    # Compute ray direction in ENU frame
    ray_dir = np.array([np.cos(global_angle), np.sin(global_angle)])
    
    # Define the ray as a line from vessel's position extending in the given direction
    ray_end = vessel_x + ray_dir[0] * max_distance, vessel_y + ray_dir[1] * max_distance
    ray = LineString([(vessel_x, vessel_y), ray_end])
    
    min_distance = max_distance  # Initialize with max distance
    
    for rect in rectangles:
        polygon = Polygon(rect)  # Create a polygon from rectangle corners
        intersection = ray.intersection(polygon.boundary)

        if not intersection.is_empty:
            if isinstance(intersection, Point):  # Single intersection
                dist = np.linalg.norm(np.array(intersection.coords[0]) - np.array([vessel_x, vessel_y]))
                min_distance = min(min_distance, dist)
            else:  # Multiple intersection points (LineString or MultiPoint)
                for point in intersection.geoms:
                    dist = np.linalg.norm(np.array(point.coords[0]) - np.array([vessel_x, vessel_y]))
                    min_distance = min(min_distance, dist)

    return min_distance

def get_distance_profile(rectangles, angles, vessel_state, max_distance=20.0):
    """
    Computes the safe travel distances for every whole degree from 0 to pi (180°).
    
    Parameters:
        rectangles (list of numpy arrays): List of 4x2 arrays representing detected rectangular objects.
        angles (list of float): List of angles in radians to check for obstacles.
        vessel_state (tuple): The vessel's (x, y, heading) in ENU frame.
        max_distance (float): The max distance to return if no object is hit.
    
    Returns:
        list: Distances corresponding to each angle.
    """
    distances = []
    for angle in angles:
        dist = distance_to_nearest_object(rectangles, angle, vessel_state, max_distance)
        distances.append(dist)
    
    return distances
