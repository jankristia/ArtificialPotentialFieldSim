import numpy as np
from lidar import LidarSimulator
# from scipy.integrate import solve_ivp

# Boat Simulator
# # Boat parameters (based on Fossen model)
m = 20.0      # Mass of the boat (kg)
Iz = 8.5     # Moment of inertia (kg.m^2)
X_u_dot = -30  # Added mass in surge
Y_v_dot = -25  # Added mass in sway
N_r_dot = -6  # Added moment of inertia in yaw
Xu = -40     # Linear damping in surge
Yv = -65     # Linear damping in sway
Nr = -50     # Linear damping in yaw
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
    def __init__(self, waypoints, obstacles):
        # State: [x, y, psi, u, v, r] (Position & velocity)
        self.state = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])  # [x, y, heading, surge vel, sway vel, yaw rate]
        self.dt = 0.1  # Time step
        self.kp = 25  # PD yaw control proportional gain
        self.kd = 2  # PD yaw control derivative gain
        self.prev_heading_error = 0.0  # Previous heading error for PD control
        self.waypoints = waypoints
        self.current_wp_index = 0  # Start with the first waypoint
        self.base_thrust = 30  # Base thrust applied to both thrusters
        self.max_thrust = 100
        self.min_thrust = -60
        self.radius = 1.0  # Radius of the boat (m)
        self.thruster_arm = 0.3       # Distance from centerline to thruster (m)
        self.lidar = LidarSimulator(obstacles=obstacles)  # Lidar sensor
        self.thresh_next_wp = 10.0  # Threshold to switch waypoints
        self.los_lookahead = 10  # Lookahead distance for LOS guidance
        self.collided = False
        self.reached_goal = False

        # Variables for plotting
        self.repulsive_force = np.array([0.0, 0.0])
        self.attractive_force = np.array([0.0, 0.0])
        self.total_force = np.array([0.0, 0.0])
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
    
    def apf_obstacle_avoidance(self, psi_d):
        lidar_readings = self.lidar.sense_obstacles(self.state[0], self.state[1], self.state[2])
        self.repulsive_force = np.array([0.0, 0.0])

        for dist, angle in zip(lidar_readings, self.lidar.angles):
            if dist < self.lidar.max_range:
                if 0 < dist:
                    obstacle_angle = self.state[2] + angle  # Convert to global frame
                    rep_angle = obstacle_angle + np.pi  # Directly opposite
                    self.repulsive_force += np.array([
                        np.cos(rep_angle) / (dist**2) + np.cos(rep_angle) * 0.5,
                        np.sin(rep_angle) / (dist**2) + np.sin(rep_angle) * 0.5
                    ])
        self.attractive_force = 6*np.array([np.cos(psi_d), np.sin(psi_d)])
        self.total_force = self.attractive_force + self.repulsive_force
        psi_d_new = np.arctan2(self.total_force[1], self.total_force[0])
        self.desired_heading = psi_d_new
        return psi_d_new
    
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

        psi_d = self.los_guidance()  # Compute desired heading
        psi_d = self.apf_obstacle_avoidance(psi_d)
        thrust_diff = self.pd_controller(psi_d)  # Compute differential thrust

        tau = self.forces(thrust_diff)  # Compute input forces and moments
        state_dot = self.state_dot(tau)
        
        # self.state[:3] += state_dot[:3] * self.dt  # Update position and heading
        # self.state[3:] += state_dot[3:] * self.dt  # Update velocity state
        self.state[3:] += state_dot[3:] * self.dt  # Update velocity state first
        self.state[:3] += Rzyx(0, 0, self.state[2]) @ self.state[3:] * self.dt  # Update position using new velocity


