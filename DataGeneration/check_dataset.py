import glob
import numpy as np
import matplotlib.pyplot as plt
import VerletSolver as verlet

out_dir = "data/traj_chunks"  # same folder as in generate_dataset.py

# 1. Are all the files there?
files = sorted(glob.glob(f"{out_dir}/chunk_*.npz"))
print("number of chunk files:", len(files))

# 2. Load everything into one array
data = np.concatenate([np.load(f)["trajectories"] for f in files])
d0 = np.load(files[0])
M, G, E = d0["M"], d0["G"], float(d0["E"])
print("data shape:", data.shape)  # (n_chunks*chunk_size, n_saved, 12)

# 3. No broken numbers
print("any NaN or inf:", not np.all(np.isfinite(data)))

# 4. Every trajectory starts at the right energy and stays there
start_E = [verlet.total_energy(*verlet.unpack_state(traj[0]), M, G) for traj in data]
end_E = [verlet.total_energy(*verlet.unpack_state(traj[-1]), M, G) for traj in data]
print("start energies: min", min(start_E), "max", max(start_E))
print("end energies:   min", min(end_E), "max", max(end_E))

# 5. No body flew away
print("largest position value:", np.max(np.abs(data[:, :, :6])))

# 6. All trajectories are different (no repeated random starts)
starts = data[:, 0, :]
print("unique starts:", len(np.unique(starts, axis=0)), "of", len(starts))

# 7. Look at a few
for i in range(3):
    print(f"number of trajectories: {len(data)}")
    print(f"number of steps in trajectory {i}:", len(data[i]))
    verlet.plot_trajectory(data[i])
