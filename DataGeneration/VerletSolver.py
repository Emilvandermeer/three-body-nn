# Three-body problem: velocity Verlet simulation
#
# State vector layout (12 numbers), used everywhere in the project:
#   [x1, y1, x2, y2, x3, y3, vx1, vy1, vx2, vy2, vx3, vy3]

import numpy as np
import matplotlib.pyplot as plt


# Compute acceleration
def compute_acceleration(x, y, M, ax, ay, G):
    for j in range(3):
        # Distance between body j and all bodies (including itself)
        dx = x[j] - x
        dy = y[j] - y

        # Acceleration according to Newton's law
        r3 = np.sqrt(dx**2 + dy**2) ** 3
        r3[j] = 1.0  # avoid dividing by zero for the self-interaction (dx = dy = 0 there)
        ax[:, j] = (-dx * M * G) / r3
        ay[:, j] = (-dy * M * G) / r3

    ax_tot = np.sum(ax, axis=0)
    ay_tot = np.sum(ay, axis=0)
    return ax_tot, ay_tot


def unpack_state(state):
    """Split a 12-number state into x, y, vx, vy (each of length 3)."""
    x = state[0:6:2]
    y = state[1:6:2]
    vx = state[6:12:2]
    vy = state[7:12:2]
    return x, y, vx, vy


def total_energy(x, y, vx, vy, M, G):
    """Total energy (kinetic + potential) at one moment in time."""
    # Kinetic energy: 1/2 * sum of m * v^2
    kinetic = 0.5 * np.sum(M * (vx**2 + vy**2))

    # Potential energy: -G * m_i * m_j / r_ij for each of the 3 pairs
    potential = 0.0
    for i, j in [(0, 1), (0, 2), (1, 2)]:
        r = np.sqrt((x[i] - x[j]) ** 2 + (y[i] - y[j]) ** 2)
        potential -= G * M[i] * M[j] / r

    return kinetic + potential


def simulate(initial_state, M, G, dt, n_steps, save_every=1):
    """Run the velocity Verlet method.

    Returns an array of shape (n_steps // save_every + 1, 12),
    one row per saved moment, in the state layout above.
    """
    # Prepare the arrays to store the solution (3 columns: one per body)
    x = np.zeros((n_steps + 1, 3))
    y = np.zeros((n_steps + 1, 3))
    vx = np.zeros((n_steps + 1, 3))
    vy = np.zeros((n_steps + 1, 3))

    ax = np.zeros((3, 3))
    ay = np.zeros((3, 3))

    # Assign the initial condition to the first row
    x[0, :], y[0, :], vx[0, :], vy[0, :] = unpack_state(initial_state)

    for i in range(1, n_steps + 1):
        # Compute the acceleration at the current time
        ax_tot, ay_tot = compute_acceleration(x[i - 1, :], y[i - 1, :], M, ax, ay, G)

        # Compute the next position given the current acceleration
        x[i, :] = x[i - 1, :] + dt * vx[i - 1, :] + 1 / 2 * (ax_tot * dt**2)
        y[i, :] = y[i - 1, :] + dt * vy[i - 1, :] + 1 / 2 * (ay_tot * dt**2)

        # Compute the acceleration at the next time
        ax_tot_next, ay_tot_next = compute_acceleration(x[i, :], y[i, :], M, ax, ay, G)

        # Compute the next velocity according to the Verlet method
        vx[i, :] = vx[i - 1, :] + 1 / 2 * (ax_tot + ax_tot_next) * dt
        vy[i, :] = vy[i - 1, :] + 1 / 2 * (ay_tot + ay_tot_next) * dt

    # Keep only every save_every-th row
    x, y = x[::save_every], y[::save_every]
    vx, vy = vx[::save_every], vy[::save_every]

    # Pack into one (n_saved, 12) array in the fixed state layout
    states = np.zeros((len(x), 12))
    states[:, 0:6:2] = x
    states[:, 1:6:2] = y
    states[:, 6:12:2] = vx
    states[:, 7:12:2] = vy
    return states


def plot_trajectory(states, filename=None):
    """Plot the paths of the three bodies from a (n_saved, 12) state array."""
    plt.figure()
    plt.grid(True)
    plt.title("Verlet Method")
    plt.xlabel("x")
    plt.ylabel("y")

    colors = ["r", "b", "g"]
    for body in range(3):
        x = states[:, 2 * body]
        y = states[:, 2 * body + 1]
        plt.plot(x, y, colors[body], label=f"Body {body + 1}")
        plt.plot(x[0], y[0], colors[body] + "o")  # starting position

    plt.legend()
    if filename:
        plt.savefig(filename, dpi=150)
    else:
        plt.show()


if __name__ == "__main__":
    # Everything below only runs when you run this file directly,
    # not when another script does `from simulate import simulate`.

    # Physical settings
    M = np.array([1.0, 1.0, 1.0])  # body masses
    G = 1.0  # Newton constant

    # Time settings
    dt = 0.001  # solver time step
    T = 50  # total simulated time 
    n_steps = int(round(T / dt))
    save_every = 10  # save every 10th step -> one row per 0.01 time units

    # Initial conditions
    vx1, vy1 = 0.50, 0.5327
    initial_state = np.array(
        [
            -1.0, # x1
            0.0,  # y1
            1.0,  # x2
            0.0,  # y2
            0.0,  # x3
            0.0,  # y3
            vx1,  # vx1
            vy1,  # vy1
            vx1,  # vx2
            vy1,  # vy2
            -2 * vx1,  # vx3
            -2 * vy1,  # vy3
        ]
    )

    states = simulate(initial_state, M, G, dt, n_steps, save_every)
    print("states shape:", states.shape)

    # Energy check: should stay (nearly) the same along the whole trajectory
    energies = [total_energy(*unpack_state(row), M, G) for row in states]
    print(f"energy at start: {energies[0]:.6f}")
    print(f"energy at end:   {energies[-1]:.6f}")
    print(f"max drift:       {max(abs(e - energies[0]) for e in energies):.2e}")

    plot_trajectory(states)
