import numpy as np
from lidar import LidarSimulator
from helpers import M, N, Rzyx
from velocity_obstacles import tcpa_dcpa_vo_check
from distance_to_object import get_distance_profile


class BoatSimulator:
    def __init__(self, waypoints, circular_obstacles, isNoise=False):
        # State: [x, y, psi, u, v, r] (Position & velocity)
        self.state = np.array([0.0, 0.0, 2/6*np.pi, 0.0, 0.0, 0.0])  # [x, y, heading, surge vel, sway vel, yaw rate]

        # Boat parameters
        self.max_thrust = 80
        self.min_thrust = -50
        self.thruster_arm = 0.3
        self.radius = 1.0  # Radius of the boat (m)
        self.neutral_pwm = 1500
        self.min_pwm = 1100
        self.max_pwm = 1900

        # Control parameters
        self.dt = 0.1       # Time step (s)
        self.T = 0.3        # Time constant for refrence heading model
        self.kp_heading = 200
        self.kd_heading = 20
        self.prev_heading_error = 0.0
        self.kp_velocity = 150
        self.kd_velocity = 10
        self.ki_velocity = 5
        self.prev_velocity_error = 0.0
        self.velocity_integral_error = 0.0
        self.integral_windup_limit = 100
        self.base_surge_velocity = 1.0
        self.base_pwm = 0
        self.prev_desired_heading = self.state[2]

        # Waypoints and navigation
        self.waypoints = waypoints
        self.current_wp_index = 0
        self.thresh_next_wp = 10.0
        self.los_lookahead = 15

        # LiDAR and obstacles
        self.safety_distance = 1.0
        self.lidar = LidarSimulator()
        self.collided = False
        self.reached_goal = False
        self.circular_obstacles = circular_obstacles
        self.rectangle_obstacles = []
        self.expanded_retangles = []
        self.distance_profile = []
        self.candidate_headings = []


        # Velocity Obstacles
        self.forbidden_headings = []

        # Variables for plotting
        self.pwm_diff = 0
        self.pwm_right = 0.0
        self.pwm_right = 0.0
        self.cross_track_error = 0.0
        self.ColAv_desired_heading = 0.0
        self.LOS_desired_heading = 0.0
        self.shortest_object_dist = 0.0

        # Noise and disturbances
        if isNoise:
            # Velocity of water current
            self.vcx = -0.1
            self.vcy = 0.1
            # GPS noise
            self.gps_noise_std_dev = 0.3
            self.gps_noise = np.random.normal(0, self.gps_noise_std_dev, size=2)
            # Object tracking noise
            self.object_velocity_noise_std_dev = 0.3
            self.object_velocity_noise = np.random.normal(0, self.object_velocity_noise_std_dev, size=2)
            # Heading noise
            self.heading_noise_std_dev = np.deg2rad(0.5)
            self.heading_noise = np.random.normal(0, self.heading_noise_std_dev)
        else:
            # Velocity of water current
            self.vcx = 0
            self.vcy = 0
            # GPS noise
            self.gps_noise_std_dev = 0
            self.gps_noise = np.random.normal(0, self.gps_noise_std_dev, size=2)
            # Object tracking noise
            self.object_velocity_noise_std_dev = 0
            self.object_velocity_noise = np.random.normal(0, self.object_velocity_noise_std_dev, size=2)



    def los_guidance(self):
        """Compute desired heading using Line of Sight (LOS)"""
        if self.current_wp_index >= len(self.waypoints):
            return self.state[2]

        x, y = self.state[0] + self.gps_noise[0], self.state[1] + self.gps_noise[1]

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

    def wp_guidance(self):
        """Compute desired heading using waypoint guidance"""
        if self.current_wp_index >= len(self.waypoints):
            return self.state[2]

        wp_next = self.waypoints[min(self.current_wp_index + 1, len(self.waypoints) - 1)]

        dx = wp_next[0] - (self.state[0] + self.gps_noise[0])
        dy = wp_next[1] - (self.state[1] + self.gps_noise[1])
        psi_d = np.arctan2(dy, dx)

        if np.hypot((self.state[0] + self.gps_noise[0])-wp_next[0], (self.state[1] + self.gps_noise[1])-wp_next[1]) < self.thresh_next_wp:
            self.current_wp_index += 1
        return psi_d

    def pd_heading_controller(self, psi_d):
        """PD Controller for yaw control (differential thrust)"""
        psi = self.state[2] + self.heading_noise
        error = psi_d - psi
        d_error = (error - self.prev_heading_error) / self.dt
        thrust_diff = self.kp_heading * error + self.kd_heading * d_error
        self.prev_heading_error
        return thrust_diff

    def pid_velocity_controller(self, surge_velocity_d):
        """PD Controller for surge velocity control"""
        surge_velocity = self.state[3]
        error = surge_velocity_d - surge_velocity
        d_error = (error - self.prev_velocity_error) / self.dt
        self.prev_velocity_error = error

        self.velocity_integral_error += error * self.dt
        self.velocity_integral_error = np.clip(self.velocity_integral_error, -self.integral_windup_limit, self.integral_windup_limit)

        base_pwm = 1500 + 400/3*surge_velocity_d

        control_output = self.kp_velocity * error + self.kd_velocity * d_error + self.ki_velocity * self.velocity_integral_error + base_pwm
        return np.clip(control_output, self.min_pwm, self.max_pwm)

    def combined_controller(self, psi_d, surge_velocity_d):
        """Combined PD controller for heading and surge velocity"""
        pwm_diff_desired = self.pd_heading_controller(psi_d)
        base_pwm_desired = self.pid_velocity_controller(surge_velocity_d)


        # Apply low-pass filter to thrust pwm
        self.pwm_diff = self.pwm_diff + self.dt / self.T * (pwm_diff_desired - self.pwm_diff)
        self.base_pwm = self.base_pwm + self.dt / self.T * (base_pwm_desired - self.base_pwm)

        # self.pwm_diff = pwm_diff_desired
        # self.base_pwm = base_pwm_desired

        pwm_left = self.base_pwm + self.pwm_diff / 2
        pwm_right = self.base_pwm - self.pwm_diff / 2

        pwm_left = np.clip(pwm_left, self.min_pwm, self.max_pwm)
        pwm_right = np.clip(pwm_right, self.min_pwm, self.max_pwm)

        self.pwm_left = pwm_left
        self.pwm_right = pwm_right

        pwm_out = np.array([pwm_left, pwm_right])
        return pwm_out

    def thrust_model(self, pwm_out):
        """Convert PWM to thrust using a simple linear model"""
        pwm_left, pwm_right = pwm_out

        def pwm_to_thrust(pwm):
            if pwm >= self.neutral_pwm:
                return (pwm - self.neutral_pwm) / (self.max_pwm - self.neutral_pwm) * 100  # scale to [0, 100]
            else:
                return (pwm - self.neutral_pwm) / (self.min_pwm - self.neutral_pwm) * (-60)   # scale to [0, -60]

        # Convert to thrust forces
        thrust_left = pwm_to_thrust(pwm_left)
        thrust_right = pwm_to_thrust(pwm_right)

        # Surge force = sum of the two thrusters
        surge_force = thrust_left + thrust_right

        # Yaw moment = differential thrust * arm length (assume symmetric layout)
        yaw_moment = (thrust_left - thrust_right) * self.thruster_arm

        # No sway force in underactuated system
        tau = np.array([surge_force, 0, yaw_moment])  # [Fx, Fy, Mz]
        return tau


    def cri_obstacle_avoidance(self, psi_d):
        """Avoid obstacles using Collision Risk Index (CRI) and Velocity Obstacle (VO) method.
        Evaluates risk at all LiDAR angles based on:
        - Deviation from the desired heading (angle risk)
        - Proximity to obstacles (distance risk)
        - If heading is inside Velocity Obstacle
        """

        distances = self.lidar.sense_obstacles((self.state[0] + self.gps_noise[0]), (self.state[1] + self.gps_noise[1]), self.state[2] + self.heading_noise, self.circular_obstacles)
        # filtered_distances, filtered_angles = self.lidar.remove_noise_knn(distances, self.lidar.angles)
        clusters_ = self.lidar.cluster_lidar_points(distances, self.lidar.angles)

        if clusters_:
            merged_clusters = self.lidar.merge_clusters(clusters_)
            self.rectangle_obstacles = self.lidar.clusters_to_oriented_rectangles(self.state, merged_clusters)
            self.expanded_retangles = self.lidar.expand_oriented_rectangles(self.rectangle_obstacles, self.radius)
        else:
            self.rectangle_obstacles = []
            self.expanded_retangles = []
        self.candidate_headings =  np.linspace(-1/2*np.pi, 1/2*np.pi, 180)  # self.lidar.angles
        self.distance_profile = get_distance_profile(self.expanded_retangles, self.candidate_headings, self.state, max_distance=20.0)

        risk_list = []
        current_angle = self.state[2] + self.heading_noise
        self.shortest_object_dist = np.min(distances)

        for dist, angle in zip(self.distance_profile, self.candidate_headings):
            angle = angle + current_angle
            angle_diff = np.abs(np.arctan2(np.sin(psi_d - angle), np.cos(psi_d - angle)))
            # if angle_diff < np.pi/6:  # Reduce threshold for more responsive avoidance
            #     Ra = 0
            # else:
            #     Ra = np.abs(angle_diff - np.pi/6) * 0.04 * 180 / np.pi  # Adjust weight dynamically
            Ra =  0.04 * 180 / np.pi * (angle_diff) **2  # Adjust weight dynamically


            # Add distance risk
            Rd = max(0, 20 - dist)*6  # Prevent negative risk values

            # Prevent large changes in psi_d
            heading_change = angle - self.prev_desired_heading
            heading_change = np.arctan2(np.sin(heading_change), np.cos(heading_change))  # Wrap to [-pi, pi]

            # Penalty grows if we try to reverse heading by more than 20 degrees
            if np.abs(heading_change) > np.deg2rad(10):
                R_delta_psi = 180 / np.pi * 0.1 * (np.abs(heading_change)) ** 2  # stronger but only when needed
            else:
                R_delta_psi = 0

            # Velocity obstacle risk
            Rvo = 0
            for right_forbidden_heading, left_forbidden_heading in self.forbidden_headings:
                if right_forbidden_heading <= angle <= left_forbidden_heading:
                    Rvo = 80
                    break

            Rt =  Ra + Rd + R_delta_psi + Rvo
            risk_list.append((Rt, angle))

        min_risk = min(risk_list, key=lambda x: x[0])[0]
        best_angles = [angle for risk, angle in risk_list if risk == min_risk]

        # If multiple angles have the same minimum risk, choose the one closest to psi_d
        best_angle = min(best_angles, key=lambda a: np.abs(np.arctan2(np.sin(psi_d - a), np.cos(psi_d - a))))

        return best_angle

    def state_dot(self, tau):
        nu = self.state[3:]  # [u, v, r]
        psi = self.state[2]  # heading

        # Current velocity in NED (defined in world)
        vc_ned = np.array([self.vcx, self.vcy, 0])

        # Rotate current into body frame
        R = Rzyx(0, 0, psi)
        vc_body = R.T @ vc_ned  # inverse rotation

        # Compute RELATIVE velocity (used for hydrodynamics)
        nu_r = nu - vc_body  # relative to water!

        # η̇ = transformation from body to inertial, using absolute body velocity
        eta_dot = R @ nu

        # ν̇ using relative velocity
        nu_dot = np.linalg.inv(M) @ (tau - N(nu_r) @ nu_r)

        return np.concatenate([eta_dot, nu_dot])


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
        # psi_d = self.wp_guidance()
        self.LOS_desired_heading = psi_d

        self.forbidden_headings = tcpa_dcpa_vo_check(self.state, self.circular_obstacles, absolute_velocity, self.lidar.angles, 3*self.radius, self.lidar.max_range, self.object_velocity_noise)

        psi_d = self.cri_obstacle_avoidance(psi_d)
        psi_d = 0.3 * psi_d + 0.7 * self.prev_desired_heading
        self.prev_desired_heading = psi_d

        self.ColAv_desired_heading = psi_d

        pwm_out = self.combined_controller(psi_d, self.base_surge_velocity)  # Compute input forces and moments

        tau = self.thrust_model(pwm_out)


        state_dot = self.state_dot(tau)

        self.state[3:] += state_dot[3:] * self.dt  # Update velocity state first
        self.state[:3] += Rzyx(0, 0, self.state[2]) @ self.state[3:] * self.dt  # Update position using new velocity

        # Update moving obstacles
        for obs in self.circular_obstacles:
            obs.update_position(self.dt)

        # Update sensor noise
        self.gps_noise = np.random.normal(0, self.gps_noise_std_dev, size=2)
        self.object_velocity_noise = np.random.normal(0, self.object_velocity_noise_std_dev, size=2)
        self.heading_noise = np.random.normal(0, self.heading_noise_std_dev)