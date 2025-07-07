from __future__ import annotations

import numpy as np
from matplotlib import pyplot as plt
from phonopy import Phonopy
from phonopy.structure.atoms import PhonopyAtoms

# Step 1: Define the unit cell
a = 1.0  # Lattice constant
cell = PhonopyAtoms(
    symbols=["Si"], cell=[[a, 0, 0], [0, 1, 0], [0, 0, 1]], scaled_positions=[[0, 0, 0]]
)

# Step 2: Set up supercell and primitive cell
supercell_matrix = [[5, 0, 0], [0, 1, 0], [0, 0, 1]]  # 5 atoms in x-direction
phonon = Phonopy(unitcell=cell, supercell_matrix=supercell_matrix)

# Step 3: Define force constants manually (nearest-neighbour harmonic model)
num_atoms = len(phonon.supercell)
fc = np.zeros((num_atoms, num_atoms, 3, 3))

k = 10.0  # force constant

# Nearest neighbour harmonic interaction along x
for i in range(num_atoms):
    if i > 0:
        fc[i, i - 1, 0, 0] = -k
    if i < num_atoms - 1:
        fc[i, i + 1, 0, 0] = -k
    fc[i, i, 0, 0] = 2 * k  # diagonal term

phonon.force_constants = fc

# Step 4: Run mesh calculation
mesh = [101, 1, 1]  # Fine mesh for smooth dispersion
phonon.run_mesh(mesh, with_eigenvectors=True, is_mesh_symmetry=False)
mesh_dict = phonon.get_mesh_dict()

qpoints = mesh_dict["qpoints"]
frequencies = mesh_dict["frequencies"]
eigenvectors = mesh_dict["eigenvectors"]

# Reflect q-values in the first Brillouin zone
bz_boundary = 0.5  # For reduced units (unit cell reciprocal = 1/a)
q_vals = qpoints[:, 0]  # Reflect negative q-values
# Fold into BZ

# Sort q-values and corresponding frequencies and eigenvectors for plotting
sorted_indices = np.argsort(q_vals)
q_vals = q_vals[sorted_indices]
frequencies = frequencies[sorted_indices]
eigenvectors = eigenvectors[sorted_indices]

# Plot
fig, ax = plt.subplots(figsize=(8, 6))
for band in range(frequencies.shape[1]):
    ax.plot(q_vals, frequencies[:, band], label=f"Mode {band + 1}")

ax.axvline(bz_boundary, color="r", linestyle="--", label="BZ boundary")
ax.axvline(0, color="r", linestyle="--")
ax.set_xlabel("q (reduced units)")
ax.set_ylabel("Frequency (THz)")
ax.set_title("Phonon Dispersion in 1D Chain")
ax.legend()
ax.grid(visible=True)
plt.tight_layout()
plt.show()
