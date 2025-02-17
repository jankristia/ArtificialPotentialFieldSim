import numpy as np
from lidar import LidarSimulator
from moving_obstacle import MovingObstacle
# from scipy.integrate import solve_ivp

# Boat Simulator
# # Boat parameters (based on Fossen model)
m = 20.0      # Mass of the boat (kg)
Iz = 8.5     # Moment of inertia (kg.m^2)
X_u_dot = -30  # Added mass in surge
Y_v_dot = -25  # Added mass in sway
N_r_dot = -5  # Added moment of inertia in yaw
Xu = -40     # Linear damping in surge
Yv = -65     # Linear damping in sway
Nr = -30     # Linear damping in yaw
Y_r = -0.15
N_v = -0.12

# Model matrices
M = np.array([
    [m - X_u_dot, 0, 0],
    [0, m - Y_v_dot, 0],
    [0, 0, Iz - N_r_dot]
])

def N(nu):
    u, v, r = nu
    return np.array([
        [-Xu, -m*r, Y_v_dot*v],
        [m*r, -Yv, -X_u_dot*u],
        [-Y_v_dot*v, X_u_dot*u, -Nr]
    ])

def Rzyx(phi, theta, psi):
    cphi = np.cos(phi)
    sphi = np.sin(phi)
    cth = np.cos(theta)
    sth = np.sin(theta)
    cpsi = np.cos(psi)
    spsi = np.sin(psi)

    return np.vstack([
        np.hstack([cpsi*cth, -spsi*cphi+cpsi*sth*sphi, spsi*sphi+cpsi*cphi*sth]),
        np.hstack([spsi*cth, cpsi*cphi+sphi*sth*spsi, -cpsi*sphi+sth*spsi*cphi]),
        np.hstack([-sth, cth*sphi, cth*cphi])
    ])

class BoatSimulator:
    def __init__(self, waypoints, obstacles, moving_obstacles):
        # State: [x, y, psi, u, v, r] (Position & velocity)
        self.state = np.array([0.0, 0.0, 0, 0.0, 0.0, 0.0])  # [x, y, heading, surge vel, sway vel, yaw rate]
        self.dt = 0.1  # Time step
        self.kp = 35  # PD yaw control proportional gain
        self.kd = 3  # PD yaw control derivative gain
        self.prev_heading_error = 0.0  # Previous heading error for PD control
        self.waypoints = waypoints
        self.current_wp_index = 0  # Start with the first waypoint
        self.base_thrust = 20  # Base thrust applied to both thrusters
        self.max_thrust = 100
        self.min_thrust = -60
        self.radius = 1.0  # Radius of the boat (m)
        self.safety_distance = 1.0  # Safety distance from obstacles (m)
        self.thruster_arm = 0.3       # Distance from centerline to thruster (m)
        self.lidar = LidarSimulator(static_obstacles=obstacles)
        self.thresh_next_wp = 10.0  # Threshold to switch waypoints
        self.los_lookahead = 25  # Lookahead distance for LOS guidance
        self.collided = False
        self.reached_goal = False
        self.obstacle_clusters = []
        self.static_obstacles = obstacles
        self.moving_obstacles = moving_obstacles

        # Variables for plotting
        self.thrust_diff = 0.0
        self.thrust_left = 0.0
        self.thrust_right = 0.0
        self.cross_track_error = 0.0
        self.desired_heading = 0.0



    def los_guidance(self):
        """Compute desired heading using Line of Sight (LOS)"""
        if self.current_wp_index >= len(self.waypoints):
            return self.state[2]  # Keep last heading if done
        
        x, y = self.state[0], self.state[1]

        wp_curr = self.waypoints[self.current_wp_index]
        wp_next = self.waypoints[min(self.current_wp_index + 1, len(self.waypoints) - 1)]
        
        dx = wp_next[0] - wp_curr[0]
        dy = wp_next[1] - wp_curr[1]

        pi_p = np.arctan2(dy, dx)  # Path angle

        cross_track_error = (y - wp_curr[1]) * np.cos(pi_p) - (x - wp_curr[0]) * np.sin(pi_p)
        self.cross_track_error = cross_track_error

        psi_d = pi_p - np.arctan(cross_track_error / self.los_lookahead)

        if np.hypot(x-wp_next[0], y-wp_next[1]) < self.thresh_next_wp:
            self.current_wp_index += 1

        return psi_d

    def pd_controller(self, psi_d):
        """PD Controller for yaw control (differential thrust)"""
        psi = self.state[2]
        error = np.arctan2(np.sin(psi_d - psi), np.cos(psi_d - psi))  # Angle wrap
        d_error = (error - self.prev_heading_error) / self.dt
        thrust_diff = self.kp * error + self.kd * d_error  # PD control output
        self.prev_heading_error = error
        return thrust_diff

    def forces(self, thrust_diff):
        """Compute forces and moments from two thrusters"""
        T_left = self.base_thrust + thrust_diff  # Left thruster
        T_right = self.base_thrust - thrust_diff  # Right thruster

        T_left = np.clip(T_left, self.min_thrust, self.max_thrust)
        T_right = np.clip(T_right, self.min_thrust, self.max_thrust)

        # print(f"Thrust left: {thrust_diff} Thrust right: {T_right}")
        self.thrust_diff = thrust_diff
        self.thrust_left = T_left
        self.thrust_right = T_right
        # Compute forces and moment
        surge_force = T_left + T_right  # Total forward force
        yaw_moment = - self.thruster_arm * (T_right - T_left)  # Moment due to thrust difference

        tau = np.array([surge_force, 0, yaw_moment])  # [Fx, Fy, Mz]
        return tau
 
    def cluster_lidar_data(self):
        """Clusters LiDAR data into detected obstacles, adding a safety margin to each obstacle."""
    
        lidar_readings = self.lidar.sense_obstacles(self.state[0], self.state[1], self.state[2], self.moving_obstacles)
        self.obstacle_clusters = []
        prev_dist = None
        dist_diff_threshold = 1.0
        cluster = []
        boat_width = 2*self.radius

        for dist, angle in zip(lidar_readings, self.lidar.angles):
            adjusted_angle = angle + self.state[2]  # Convert to World frame

            if dist >= self.lidar.max_range:
                # End the current cluster if there is one
                if cluster:
                    
                    avg_dist = np.mean([point[0] for point in cluster])
                    
                    start_angle = cluster[0][1]
                    end_angle = cluster[-1][1]
                    self.obstacle_clusters.append((start_angle, end_angle, avg_dist))  #  can use avg_dist - self.safety_distance
                    cluster = []  # Reset cluster

                prev_dist = None  # Reset for new cluster
                continue  # Skip max range points
            
            elif prev_dist is not None and np.abs(dist - prev_dist) <= dist_diff_threshold and dist < self.lidar.max_range:
                cluster.append((dist, adjusted_angle))
            else:
                # Finish the previous cluster and start a new one
                if cluster:
                    avg_dist = np.mean([point[0] for point in cluster])
                    start_angle = cluster[0][1]
                    end_angle = cluster[-1][1]

                    self.obstacle_clusters.append((start_angle, end_angle, avg_dist))  #  can use avg_dist - self.safety_distance
                    cluster = []  # Reset cluster
                
                # Start new cluster
                # cluster.append((dist, adjusted_angle))

            prev_dist = dist  # Update for next iteration

        # Final cluster processing
        if cluster:
            avg_dist = np.mean([point[0] for point in cluster])
            start_angle = cluster[0][1]
            end_angle = cluster[-1][1]
            self.obstacle_clusters.append((start_angle, end_angle, avg_dist))  #  can use avg_dist - self.safety_distance


    def cluster_objects(self):
        """Merges obstacle clusters if the Euclidean distance between them is below a threshold."""
        
        merged_clusters = []
        merge_distance_threshold = 4*self.radius  # Maximum allowed gap (meters) between clusters to merge
        
        if not self.obstacle_clusters:
            return  # No clusters detected
        
        # Sort clusters by starting angle
        self.obstacle_clusters.sort(key=lambda x: x[0])
        
        current_cluster = list(self.obstacle_clusters[0])  # Convert tuple to list for modification

        for next_cluster in self.obstacle_clusters[1:]:
            start_angle_current, end_angle_current, dist_current = current_cluster
            start_angle_next, end_angle_next, dist_next = next_cluster

            # Convert angles to Cartesian coordinates
            x1 = dist_current * np.cos(end_angle_current)
            y1 = dist_current * np.sin(end_angle_current)
            x2 = dist_next * np.cos(start_angle_next)
            y2 = dist_next * np.sin(start_angle_next)

            # Compute Euclidean distance between the two clusters
            gap_distance = np.hypot(x2 - x1, y2 - y1)

            # Check if clusters are close enough to merge
            if gap_distance < merge_distance_threshold:
                # Merge: expand the current cluster to include the next one
                current_cluster[1] = end_angle_next  # Extend end angle
                current_cluster[2] = min(dist_current, dist_next)  # Use the nearest distance
            else:
                # Compute dynamic safety margin before saving the cluster
                avg_dist = current_cluster[2]
                boat_width = 2 * self.radius  # Boat width (2 * radius)
                margin = np.arctan(boat_width / avg_dist)  # Dynamic angular margin

                # Apply safety margin
                current_cluster[0] -= margin  # Expand start angle
                current_cluster[1] += margin  # Expand end angle
                merged_clusters.append(tuple(current_cluster))
                current_cluster = list(next_cluster)
        
        # Apply safety margin to the last cluster and append
        avg_dist = current_cluster[2]
        boat_width = 2 * self.radius
        margin = np.arctan(boat_width / avg_dist)
        current_cluster[0] -= margin
        current_cluster[1] += margin
        merged_clusters.append(tuple(current_cluster))
        
        self.obstacle_clusters = merged_clusters  # Update with refined obstacles


    def regulate_base_thrust(self):
        """Regulate the base thrust based on nearby obstacles"""
        if not self.obstacle_clusters:
            self.base_thrust = 20  # Default value
            return
        else:
            for cluster in self.obstacle_clusters:
                start_angle, end_angle, avg_dist = cluster
                if avg_dist <= 10:
                    self.base_thrust = 10

    def cri_obstacle_avoidance(self, psi_d):
        """Avoid obstacles using Collision Risk Index (CRI) method.
        Evaluates risk at all LiDAR angles based on:
        - Deviation from the desired heading (angle risk)
        - Proximity to obstacles (distance risk)
        """

        self.cluster_lidar_data()  # Clusters LiDAR data into obstacles
        self.cluster_objects()  # Merge clusters if the gap is too small
        self.regulate_base_thrust() # Go slower when close to obstacles

        risk_list = []  # Store tuples of (risk, angle)
        current_angle = self.state[2]

        for dist, angle in zip(self.lidar.sense_obstacles(self.state[0], self.state[1], self.state[2]), self.lidar.angles):
            angle = angle + current_angle
            angle_diff = np.abs(np.arctan2(np.sin(psi_d - angle), np.cos(psi_d - angle)))
            if angle_diff < np.pi/6:  # Reduce threshold for more responsive avoidance
                Ra = 0
            else:
                Ra = np.abs(angle_diff - np.pi/6) * 0.04 * 180 / np.pi  # Adjust weight dynamically

            # Add a distance risk based on if it is in an obstacle cluster
            Rd = 0
            for start_angle, end_angle, avg_dist in self.obstacle_clusters:
                if start_angle <= angle <= end_angle:
                    Rd = max(0, 20 - avg_dist - self.radius)*3  # Prevent negative risk values
                    break
            
            Re = 0 #(np.abs(np.arctan2(np.sin(current_angle - angle), np.cos(current_angle - angle)))) * 0.2


            # print(f"Ra: {Ra} Rd: {Rd} Re: {Re}")
            Rt = Rd + Ra + Re
            risk_list.append((Rt, angle))

        min_risk = min(risk_list, key=lambda x: x[0])[0]
        best_angles = [angle for risk, angle in risk_list if risk == min_risk]

        # If multiple angles have the same minimum risk, choose the one closest to psi_d
        best_angle = min(best_angles, key=lambda a: np.abs(np.arctan2(np.sin(psi_d - a), np.cos(psi_d - a))))

        return best_angle

    

    def state_dot(self, tau):
        """Compute the derivative of the state vector"""
        nu = self.state[3:]  # Velocity state [u, v, r]
        psi = self.state[2]

        eta_dot = Rzyx(0, 0, psi) @ nu  # Transform velocities to body frame
        nu_dot = np.linalg.inv(M) @ (tau - N(nu) @ nu)  # Acceleration in body frame
        state_dot = np.concatenate([eta_dot, nu_dot])  # Combine position and velocity
        # print(f"State dot: {state_dot}")
        return state_dot
    
    def check_collision(self):
        x, y = self.state[:2]
        obstacles = self.lidar.obstacles
        for obs_x, obs_y, obs_r in obstacles:
            if np.hypot(x - obs_x, y - obs_y) < obs_r + self.radius:
                return True
        return False

    def update(self):
        """Simulate boat movement using Fossen's 3-DOF model"""

        if self.current_wp_index == len(self.waypoints) - 1:
            self.reached_goal = True
        if self.check_collision():
            self.collided = True
        
        psi_d = self.los_guidance()
        # print(f"LOS desired: {psi_d}")
        psi_d = self.cri_obstacle_avoidance(psi_d)
        thrust_diff = self.pd_controller(psi_d)  # Compute differential thrust

        tau = self.forces(thrust_diff)  # Compute input forces and moments
        state_dot = self.state_dot(tau)
        
        # self.state[:3] += state_dot[:3] * self.dt  # Update position and heading
        # self.state[3:] += state_dot[3:] * self.dt  # Update velocity state
        self.state[3:] += state_dot[3:] * self.dt  # Update velocity state first
        self.state[:3] += Rzyx(0, 0, self.state[2]) @ self.state[3:] * self.dt  # Update position using new velocity

        # Update moving obstacles
        for obs in self.moving_obstacles:
            obs.update_position(self.dt)

