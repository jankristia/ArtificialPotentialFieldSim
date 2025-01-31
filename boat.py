import numpy as np
from lidar import LidarSimulator

# Boat Simulator
# # Boat parameters (based on Fossen model)
m = 50.0      # Mass of the boat (kg)
Iz = 10.0     # Moment of inertia (kg.m^2)
X_u_dot = -5  # Added mass in surge
Y_v_dot = -10  # Added mass in sway
N_r_dot = -1  # Added moment of inertia in yaw
Xu = -2.0     # Linear damping in surge
Yv = -5.0     # Linear damping in sway
Nr = -3.0     # Linear damping in yaw
dt = 0.1      # Time step
d = 0.5       # Distance from centerline to thruster (m)
max_thrust = 10.0  # Max thrust per thruster (N)

# Inertia matrix (Including added mass)
M = np.array([
    [m - X_u_dot, 0, 0],
    [0, m - Y_v_dot, 0],
    [0, 0, Iz - N_r_dot]
])

# Damping matrix
D = np.array([
    [-Xu, 0, 0],
    [0, -Yv, 0],
    [0, 0, -Nr]
])

class BoatSimulator:
    def __init__(self, waypoints, obstacles):
        # State: [x, y, psi, u, v, r] (Position & velocity)
        self.state = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0])  # [x, y, heading, surge vel, sway vel, yaw rate]
        self.dt = dt  # Time step
        self.kp = 2.0  # PD yaw control proportional gain
        self.kd = 0.5  # PD yaw control derivative gain
        self.prev_error = 0.0  # Previous heading error for PD control
        self.waypoints = waypoints
        self.current_wp_index = 0  # Start with the first waypoint
        self.base_thrust = 2.5  # Base thrust applied to both thrusters
        self.lidar = LidarSimulator(obstacles=obstacles)  # Lidar sensor

    def los_guidance(self):
        """Compute desired heading using Line of Sight (LOS)"""
        if self.current_wp_index >= len(self.waypoints):
            return self.state[2]  # Keep last heading if done

        wp = self.waypoints[self.current_wp_index]
        dx = wp[0] - self.state[0]
        dy = wp[1] - self.state[1]
        psi_d = np.arctan2(dy, dx)  # Desired heading

        # Switch waypoint if reached
        if np.hypot(dx, dy) < 2.5:  # Threshold to switch waypoints
            self.current_wp_index += 1
            if self.current_wp_index < len(self.waypoints):
                return self.los_guidance()

        return psi_d

    def pd_controller(self, psi_d):
        """PD Controller for yaw control (differential thrust)"""
        psi = self.state[2]
        error = np.arctan2(np.sin(psi_d - psi), np.cos(psi_d - psi))  # Angle wrap
        d_error = (error - self.prev_error) / self.dt
        thrust_diff = self.kp * error + self.kd * d_error  # PD control output
        self.prev_error = error
        return np.clip(thrust_diff, -max_thrust, max_thrust)  # Limit thrust difference

    def forces(self, thrust_diff):
        """Compute forces and moments from two thrusters"""
        T_left = self.base_thrust + thrust_diff  # Left thruster
        T_right = self.base_thrust - thrust_diff  # Right thruster

        # Clip thrust to prevent negative values
        T_left = np.clip(T_left, 0, max_thrust)
        T_right = np.clip(T_right, 0, max_thrust)

        # Compute forces and moment
        surge_force = T_left + T_right  # Total forward force
        yaw_moment = - d * (T_right - T_left)  # Moment due to thrust difference

        tau = np.array([surge_force, 0, yaw_moment])  # [Fx, Fy, Mz]
        return tau

    def update(self):
        """Simulate boat movement using Fossen's 3-DOF model"""
        if self.current_wp_index >= len(self.waypoints):
            return  # Stop if all waypoints are reached

        psi_d = self.los_guidance()  # Compute desired heading
        thrust_diff = self.pd_controller(psi_d)  # Compute differential thrust

        tau = self.forces(thrust_diff)  # Compute input forces and moments
        nu = self.state[3:]  # Velocity state [u, v, r]

        # Solve the equation: M * (d_nu/dt) + D * nu = tau
        d_nu_dt = np.linalg.inv(M) @ (tau - D @ nu)  # Acceleration in body frame
        nu = nu + d_nu_dt * self.dt  # Integrate to update velocity

        # Update position and heading using kinematic equations
        psi = self.state[2]
        J = np.array([
            [np.cos(psi), -np.sin(psi), 0],
            [np.sin(psi), np.cos(psi), 0],
            [0, 0, 1]
        ])
        d_eta_dt = J @ nu  # Transform velocities to world frame
        self.state[:3] += d_eta_dt * self.dt  # Update position and heading
        self.state[3:] = nu  # Update velocity state

