import numpy as np
import DataGeneration.VerletSolver as verlet

def random_initial_conditions(rng, M, G, E):
    """Generate random initial conditions for the three-body problem."""
    # Random positions in a unit square
    K_target = 0

    while True:
        pos = rng.uniform(-1, 1, size=(3, 2))

        com = np.sum(M[:, np.newaxis] * pos, axis=0) / np.sum(M)
        pos = pos - com 

        U = verlet.total_energy(pos[:, 0], pos[:, 1], np.zeros(3), np.zeros(3), M, G)
        K_target = E - U
        if K_target > 0:
            break

    vel = rng.normal(-1, 1, size=(3, 2))
    vel = vel - np.sum(M[:, None] * vel, axis=0) / np.sum(M)

    K = 0.5 * np.sum(M[:, None] * vel**2)
    vel = vel * np.sqrt(K_target/ K)

    return np.concatenate([pos.ravel(), vel.ravel()])

if __name__ == "__main__":
    M = np.array([1.0, 1.0, 1.0])
    G = 1.0
    E = -0.8987

    rng = np.random.default_rng(0)
    state = random_initial_conditions(rng, M, G, E)
    # Test 1: does every random start have the right energy?
    for i in range(3):
        state = random_initial_conditions(rng, M, G, E)
        energy = verlet.total_energy(*verlet.unpack_state(state), M, G)
        print(f"start {i}: energy = {energy:.6f}")

    # Test 2: simulate the last one and look at it
    states = verlet.simulate(state, M, G, dt=0.001, n_steps=20000, save_every=10)
    verlet.plot_trajectory(states)