import numpy as np

m = 20.0
Iz = 8.5
X_u_dot = -30
Y_v_dot = -25
N_r_dot = -5
Xu = -40
Yv = -65
Nr = -30
Y_r = -0.15
N_v = -0.12

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

def ssa(angle):
    """Normalize an angle to the range [-π, π]"""
    return (angle + np.pi) % (2 * np.pi) - np.pi

def calculate_relative_pos_velocity(vessel_state, obs):
    """Calculate relative position and velocity of the obstacle"""
    vessel_pos = np.array([vessel_state[0], vessel_state[1]])
    vessel_vel = np.array([vessel_state[3], vessel_state[4]])
    obs_pos = np.array([obs.x, obs.y])
    obs_vel = np.array([obs.vx, obs.vy])

    R_full = Rzyx(0, 0, vessel_state[2])
    R_2d = R_full[:2, :2]
    vessel_vel = R_2d @ vessel_vel

    relative_position =  obs_pos - vessel_pos
    relative_velocity =  vessel_vel - obs_vel

    return relative_position, relative_velocity