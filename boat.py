import numpy as np
from lidar import LidarSimulator
from moving_obstacle import MovingObstacle
from model_constants import M, N, Rzyx


class BoatSimulator:
    def __init__(self, waypoints, obstacles, moving_obstacles):
        # State: [x, y, psi, u, v, r] (Position & velocity)
        self.state = np.array([0.0, 0.0, 1/2*np.pi, 0.0, 0.0, 0.0])  # [x, y, heading, surge vel, sway vel, yaw rate]
        
        # Boat parameters
        self.base_thrust = 20
        self.max_thrust = 100
        self.min_thrust = -60
        self.thruster_arm = 0.3
        self.radius = 1.0  # Radius of the boat (m)
        
        # Control parameters
        self.dt = 0.1
        self.kp = 35 
        self.kd = 3
        self.prev_heading_error = 0.0 
        
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

        # Moving obstacle states
        self.collision_risk = False
        self.enter_collision_risk_counter = 0
        self.exit_collision_risk_counter = 0
        self.collision_scenario = None

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
        
        dx = wp_next[0] - wp_curr[0] # North
        dy = wp_next[1] - wp_curr[1] # East

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
 
    def regulate_base_thrust(self):
        """Regulate the base thrust based on nearby obstacles"""
        is_nearby_obstacle = False
        if self.obstacle_clusters == None:
            return
        for cluster in self.obstacle_clusters:
            start_angle, end_angle, avg_dist = cluster
            if avg_dist <= 10:
                is_nearby_obstacle = True
                break
        if is_nearby_obstacle and self.collision_scenario != "overtaking":
            self.base_thrust = 10
        else:
            self.base_thrust = 20

    def calculate_relative_pos_velocity(self, obs):
        """Calculate relative position and velocity of the obstacle"""
        vessel_pos = np.array([self.state[0], self.state[1]])
        vessel_vel = np.array([self.state[3], self.state[4]])
        obs_pos = np.array([obs.x, obs.y])
        obs_vel = np.array([obs.vx, obs.vy])

        R_full = Rzyx(0, 0, self.state[2])
        R_2d = R_full[:2, :2]
        vessel_vel = R_2d @ vessel_vel

        relative_position = obs_pos - vessel_pos
        relative_velocity = obs_vel - vessel_vel

        return relative_position, relative_velocity
    
    def calculate_tcpa_dcpa(self, relative_position, relative_velocity):
        """Calculate Time to Closest Point of Approach (TCPA) and Distance at Closest Point of Approach (DCPA)"""
        distance = abs(np.linalg.norm(relative_position))
        # To only detect obstacles when they are close
        if distance < 20:
            tcpa = -np.dot(relative_position, relative_velocity) / (np.linalg.norm(relative_velocity)**2 + 1e-6)
            dcpa = np.linalg.norm((relative_position[0]*relative_velocity[1] - relative_position[1]*relative_velocity[0]) / (np.linalg.norm(relative_velocity) + 1e-6))
        else:
            tcpa = 100
            dcpa = 100
        return tcpa, dcpa
    
    def determine_collision_risk(self, tcpa, dcpa):
        """Determine collision risk based on TCPA and DCPA"""
        if 0 < tcpa  < 20 and 0 < dcpa < 15:
            return True
        return False
    
    def swich_collision_state(self):
        """Switches if the vessel should be in a collision state based
          on TCPA and DCPA to dynamic obstacles"""
        # NB: Now it assumes to always know the state of the dynamic obstacles, even outside the vision range

        collision_risk_this_step = False
        if len(self.moving_obstacles) > 0:
            obs = self.moving_obstacles[0]
            relative_position, relative_velocity = self.calculate_relative_pos_velocity(obs)
            tcpa, dcpa = self.calculate_tcpa_dcpa(relative_position, relative_velocity)
            if self.determine_collision_risk(tcpa, dcpa):
                collision_risk_this_step = True

        # Update collision state counters
        if collision_risk_this_step:
            self.enter_collision_risk_counter += 1
            self.exit_collision_risk_counter = 0
        else:
            self.exit_collision_risk_counter += 1
            self.enter_collision_risk_counter = 0

        # Change collision state only after 3 consecutive steps
        if self.enter_collision_risk_counter >= 3:
            if not self.collision_risk:
                print("Entered collision state!")
                self.collision_scenario = self.classify_collision_scenario()
                print(f"Collision scenario: {self.collision_scenario}")
            self.collision_risk = True
        elif self.exit_collision_risk_counter >= 3:
            if self.collision_risk:
                print("Exited collision state!")
                self.collision_scenario = None
            self.collision_risk = False

    def classify_collision_scenario(self):
        """Classify the collision scenario with dynamic obstacles as head-on, crossing, or overtaking."""

        def normalize_angle(angle):
            # Correctly normalize to [-pi, pi]
            return (angle + np.pi) % (2 * np.pi) - np.pi

        scenarios = []
        obs = self.moving_obstacles[0]
        relative_position, _ = self.calculate_relative_pos_velocity(obs)
        bearing_to_obs = np.arctan2(relative_position[1], relative_position[0])
        rel_bearing = normalize_angle(bearing_to_obs - self.state[2])
        rel_bearing_deg = np.degrees(rel_bearing)

        if abs(rel_bearing_deg) < 20:
            own_velocity = np.array([self.state[3], self.state[4]])
            obs_velocity = np.array([obs.vx, obs.vy])

            R_full = Rzyx(0, 0, self.state[2])
            R_2d = R_full[:2, :2]
            own_velocity = R_2d @ own_velocity
            if np.dot(own_velocity, obs_velocity) > 0:
                scenarios.append("overtaking")
            else:
                scenarios.append("head-on")
        elif rel_bearing_deg < -20:
            scenarios.append("crossing-right")
        elif rel_bearing_deg > 20:
            scenarios.append("crossing-left")

        if len(scenarios) == 1:
            return scenarios[0]
        else:
            return scenarios

    def cri_obstacle_avoidance(self, psi_d):
        """Avoid obstacles using Collision Risk Index (CRI) method.
        Evaluates risk at all LiDAR angles based on:
        - Deviation from the desired heading (angle risk)
        - Proximity to obstacles (distance risk)
        """

        self.obstacle_clusters = self.lidar.cluster_lidar_data(self.state, self.moving_obstacles)  # Clusters LiDAR data into obstacles
        self.obstacle_clusters = self.lidar.cluster_objects(self.obstacle_clusters, self.radius)  # Merge clusters if the gap is too small
        self.regulate_base_thrust() # Go slower when close to obstacles

        risk_list = []  # Store tuples of (risk, angle)
        current_angle = self.state[2]

        if len(self.moving_obstacles) > 0:
            obs = self.moving_obstacles[0]
            relative_position, relative_velocity = self.calculate_relative_pos_velocity(obs)
            obs_bearing = np.arctan2(relative_position[1], relative_position[0])

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

            
            Rb = 0
            if self.collision_scenario == "head-on":
                diff_bearing = np.arctan2(np.sin(angle - obs_bearing), np.cos(angle - obs_bearing))
                if diff_bearing > 0:
                    Rb = 20
                elif diff_bearing > -np.pi/6:
                    Rb = 10
            elif self.collision_scenario == "crossing-left":
                diff_bearing = np.arctan2(np.sin(angle - obs_bearing), np.cos(angle - obs_bearing))
                if diff_bearing < 0:
                    Rb = 20
                elif diff_bearing > np.pi/6:
                    Rb = 10
            elif self.collision_scenario == "crossing-right":
                diff_bearing = np.arctan2(np.sin(angle - obs_bearing), np.cos(angle - obs_bearing))
                if diff_bearing > 0:
                    Rb = 20
                elif diff_bearing > -np.pi/6:
                    Rb = 10
            elif self.collision_scenario == "overtaking":
                diff_bearing = np.arctan2(np.sin(angle - obs_bearing), np.cos(angle - obs_bearing))
                if diff_bearing > 0:
                    Rb = 20
                elif diff_bearing > -np.pi/6:
                    Rb = 10
            Rt = Rd + Ra + Rb
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

        self.swich_collision_state()

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

