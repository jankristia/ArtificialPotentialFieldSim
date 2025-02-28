import numpy as np
from lidar import LidarSimulator
from cirular_obstacle import CircularObstacle
from helpers import M, N, Rzyx
from velocity_obstacles import tcpa_dcpa_vo_check


class BoatSimulator:
    def __init__(self, waypoints, obstacles, moving_obstacles):
        # State: [x, y, psi, u, v, r] (Position & velocity)
        self.state = np.array([0.0, 0.0, 1/2*np.pi, 0.0, 0.0, 0.0])  # [x, y, heading, surge vel, sway vel, yaw rate]
        
        # Boat parameters
        self.max_thrust = 100
        self.min_thrust = -60
        self.thruster_arm = 0.3
        self.radius = 1.0  # Radius of the boat (m)
        
        # Control parameters
        self.dt = 0.1
        self.kp_heading = 35 
        self.kd_heading = 3
        self.prev_heading_error = 0.0
        self.kp_velocity = 150
        self.kd_velocity = 10
        self.ki_velocity = 5
        self.prev_velocity_error = 0.0
        self.velocity_integral_error = 0.0
        self.integral_windup_limit = 50
        self.base_surge_velocity = 2.0
        
        # Waypoints and navigation
        self.waypoints = waypoints
        self.current_wp_index = 0
        self.thresh_next_wp = 10.0
        self.los_lookahead = 25

        # LiDAR and obstacles
        self.safety_distance = 1.0
        self.lidar = LidarSimulator(static_obstacles=obstacles)
        self.collided = False
        self.reached_goal = False
        self.obstacle_clusters = []
        self.static_obstacles = obstacles
        self.moving_obstacles = moving_obstacles

        # Velocity Obstacles
        self.forbidden_headings = []

        # Variables for plotting
        self.thrust_diff = 0.0
        self.thrust_left = 0.0
        self.thrust_right = 0.0
        self.cross_track_error = 0.0
        self.desired_heading = 0.0



    def los_guidance(self):
        """Compute desired heading using Line of Sight (LOS)"""
        if self.current_wp_index >= len(self.waypoints):
            return self.state[2]
        
        x, y = self.state[0], self.state[1]

        wp_curr = self.waypoints[self.current_wp_index]
        wp_next = self.waypoints[min(self.current_wp_index + 1, len(self.waypoints) - 1)]
        
        dx = wp_next[0] - wp_curr[0]
        dy = wp_next[1] - wp_curr[1]

        pi_p = np.arctan2(dy, dx)

        cross_track_error = (y - wp_curr[1]) * np.cos(pi_p) - (x - wp_curr[0]) * np.sin(pi_p)
        self.cross_track_error = cross_track_error

        psi_d = pi_p - np.arctan(cross_track_error / self.los_lookahead)

        if np.hypot(x-wp_next[0], y-wp_next[1]) < self.thresh_next_wp:
            self.current_wp_index += 1

        return psi_d

    def pd_heading_controller(self, psi_d):
        """PD Controller for yaw control (differential thrust)"""
        psi = self.state[2]
        error = np.arctan2(np.sin(psi_d - psi), np.cos(psi_d - psi))
        d_error = (error - self.prev_heading_error) / self.dt
        thrust_diff = self.kp_heading * error + self.kd_heading * d_error  # PD control output
        self.prev_heading_error
        return thrust_diff
    
    def pd_velocity_controller(self, surge_velocity_d):
        """PD Controller for surge velocity control"""
        surge_velocity = self.state[3]
        error = surge_velocity_d - surge_velocity
        d_error = (error - self.prev_velocity_error) / self.dt
        self.prev_velocity_error = error

        self.velocity_integral_error += error * self.dt
        self.velocity_integral_error = np.clip(self.velocity_integral_error, -self.integral_windup_limit, self.integral_windup_limit)

        control_output = self.kp_velocity * error + self.kd_velocity * d_error + self.ki_velocity * self.velocity_integral_error
        return np.clip(control_output, self.min_thrust, self.max_thrust)

    def combined_controller(self, psi_d, surge_velocity_d):
        """Combined PD controller for heading and surge velocity"""
        thrust_diff = self.pd_heading_controller(psi_d)
        surge_force = self.pd_velocity_controller(surge_velocity_d)
    
        T_left = surge_force + thrust_diff/2
        T_right = surge_force - thrust_diff/2
        
        T_left = np.clip(T_left, self.min_thrust, self.max_thrust)
        T_right = np.clip(T_right, self.min_thrust, self.max_thrust)

        self.thrust_diff = thrust_diff
        self.thrust_left = T_left
        self.thrust_right = T_right

        # surge_force = (T_left + T_right)
        yaw_moment = - self.thruster_arm * (T_right - T_left)  # Moment due to thrust difference
        tau = np.array([surge_force, 0, yaw_moment])  # [Fx, Fy, Mz]
        return tau    

    def cri_obstacle_avoidance(self, psi_d):
        """Avoid obstacles using Collision Risk Index (CRI) and Velocity Obstacle (VO) method.
        Evaluates risk at all LiDAR angles based on:
        - Deviation from the desired heading (angle risk)
        - Proximity to obstacles (distance risk)
        - If heading is inside Velocity Obstacle
        """

        self.obstacle_clusters = self.lidar.cluster_lidar_data(self.state, self.moving_obstacles)
        self.obstacle_clusters = self.lidar.cluster_objects(self.obstacle_clusters, self.radius)
        risk_list = []
        current_angle = self.state[2]

        for dist, angle in zip(self.lidar.sense_obstacles(self.state[0], self.state[1], self.state[2]), self.lidar.angles):
            angle = angle + current_angle
            angle_diff = np.abs(np.arctan2(np.sin(psi_d - angle), np.cos(psi_d - angle)))
            if angle_diff < np.pi/6:  # Reduce threshold for more responsive avoidance
                Ra = 0
            else:
                Ra = np.abs(angle_diff - np.pi/6) * 0.04 * 180 / np.pi  # Adjust weight dynamically

            # Add distance risk
            Rd = 0
            for start_angle, end_angle, avg_dist in self.obstacle_clusters:
                if start_angle <= angle <= end_angle:
                    Rd = max(0, 20 - avg_dist - self.radius)*3  # Prevent negative risk values
                    break

            # Velocity obstacle risk
            Rvo = 0
            for right_forbidden_heading, left_forbidden_heading in self.forbidden_headings:
                if right_forbidden_heading <= angle <= left_forbidden_heading:
                    Rvo = 40
                    break


            Rt =  Rvo + Ra + Rd # + Rb  
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

        eta_dot = Rzyx(0, 0, psi) @ nu
        nu_dot = np.linalg.inv(M) @ (tau - N(nu) @ nu)
        state_dot = np.concatenate([eta_dot, nu_dot])
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

        # Find magnitude of vessels velocity
        absolute_velocity = np.sqrt(self.state[3]**2 + self.state[4]**2)

        psi_d = self.los_guidance()
        self.forbidden_headings = tcpa_dcpa_vo_check(self.state, self.moving_obstacles, absolute_velocity, self.lidar.angles, 3*self.radius, self.lidar.max_range)

        psi_d = self.cri_obstacle_avoidance(psi_d)

        tau = self.combined_controller(psi_d, self.base_surge_velocity)  # Compute input forces and moments
        
        state_dot = self.state_dot(tau)
        
        self.state[3:] += state_dot[3:] * self.dt  # Update velocity state first
        self.state[:3] += Rzyx(0, 0, self.state[2]) @ self.state[3:] * self.dt  # Update position using new velocity

        # Update moving obstacles
        for obs in self.moving_obstacles:
            obs.update_position(self.dt)
