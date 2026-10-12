import os
import time

import numpy as np

import DataGeneration.Arc_Generater as arc_gen
import DataGeneration.VerletSolver as verlet


def is_good_trajectory(states, M, G, E, max_dist = 5, energy_tol = 0.001) -> bool:
    """Check whether a simulated trajectory is usable as training data.

    Rejects the trajectory if
    - any body gets further than max_dist from the origin (escape), or
    - the total energy drifts from E by more than energy_tol (relative),
      which indicates the solver became inaccurate.
    """
    if np.max(np.abs(states[:, :6])) > max_dist:
        return False

    energies = np.array([verlet.total_energy(*verlet.unpack_state(state), M, G) for state in states])
    for i in range(len(energies)):
        if abs(energies[i] - E) / abs(E) > energy_tol:
            return False
    return True

def generate_chunk(rng, chunksize, M, G, E, dt, n_steps, save_every):
    """Generate a chunk of trajectories."""
    trajectories = []
    n_rejected = 0
    while len(trajectories) < chunksize:
        random_state = arc_gen.random_initial_conditions(rng, M, G, E)
        states = verlet.simulate(random_state, M, G, dt, n_steps, save_every)
        if is_good_trajectory(states, M, G, E):
            trajectories.append(states)
        else:
            n_rejected += 1
    return trajectories, n_rejected


if __name__ == "__main__":
    #======= Data Generation Settings =======
    n_chunks = 40
    chunksize = 50
    t_max = 20.0
    output_dir = "data/traj_chunks"
    seed = 0
    #========================================
    M = np.array([1.0, 1.0, 1.0])
    G = 1.0
    E = -0.8987
    dt = 0.001
    n_steps = int(t_max / dt)
    save_every = 10
    
    os.makedirs(output_dir, exist_ok=True)

    start = time.time()
    for c in range(n_chunks):
        filename = f"{output_dir}/chunk_c{c}.npz"

        if os.path.exists(filename):
            print(f"Skipping {filename} (already exists)")
            continue
        rng = np.random.default_rng([seed, c])
        data, n_rejected = generate_chunk(rng, chunksize, M, G, E, dt, n_steps, save_every)
        np.savez(filename, trajectories=data, M = M, G = G, E = E, dt=dt, save_every=save_every, seed=seed, chunk=c)

        print(f"Saved {filename} ({n_rejected} rejected) at {time.time() - start:.1f} seconds")
    end = time.time()
    print(f"Total time: {end - start:.1f} seconds")
