from __future__ import annotations

import numpy as np
from matplotlib import pyplot as plt
from phonopy import Phonopy
from phonopy.structure.atoms import PhonopyAtoms

# I somehow get negative frequencies near Gamma, which is not expected.
# The form is similar to reported dispersion curves
a_cc = 1.42
a = a_cc * np.sqrt(3)
cell = PhonopyAtoms(symbols=["C", "C"], cell=[[a, 0, 0], [a / 2, a * np.sqrt(3) / 2, 0], [0, 0, 1],]
    , scaled_positions=[[0, 0, 0], [1 / 3, 1 / 3, 0]])
#  n x m supercell
n = 15
m = 15

supercell_matrix = [[n, 0, 0], [0, m, 0], [0, 0, 1]]
phonon = Phonopy(unitcell=cell, supercell_matrix=supercell_matrix)
num_atoms = len(phonon.supercell)
fc = np.zeros((num_atoms, num_atoms, 3, 3), dtype=float)


k = 5.0  # Arbitrary force constant
mass = cell.masses[0]

# Find nearest neighbors for each atom in the supercell
positions = phonon.supercell.get_positions()

for i in range(num_atoms):
    for j in range(num_atoms):
        if i == j:
            continue
        vec = np.array(positions[j]) - np.array(positions[i])
        vec -= np.round(vec / np.array([a * n, a * m, 1])) * np.array([a * n, a * m, 1])
        dist = np.linalg.norm(vec[:2])
        if np.isclose(dist, a_cc, atol=0.05):
            for d in [0, 1]:
                fc[i, j, d, d] = -k
            fc[i, i, 0, 0] += k
            fc[i, i, 1, 1] += k
            # Add a small z spring for stability. This introduces two more bands an am not sure if it is required here
            fc[i, j, 2, 2] = -0.05 * k
            fc[i, i, 2, 2] += 0.05 * k

phonon.force_constants = fc
mesh = (25, 25, 1)
phonon.run_mesh(mesh, with_eigenvectors=True)
mesh_dict: dict[str, np.ndarray] = phonon.get_mesh_dict()
qpoints = mesh_dict["qpoints"]
frequencies = mesh_dict["frequencies"]

path = np.array([  # Plot along path: Gamma to K to M to Gamma. K and M may be the wrong way round somehow due to comparisons with dispersion cuves in papers
    [0.0, 0.0, 0.0],  # Gamma
    [0.5, 0.0, 0.0],  # M
    [1 / 3, 1 / 3, 0.0],  # K
    [0.0, 0.0, 0.0],  # Gamma
])
labels = [r"$\Gamma$", "M", "K", r"$\Gamma$"]


def interpolate_path(path: np.ndarray, n_points: int) -> np.ndarray:
    """
    Interpolates a path in reciprocal space with a specified number of points between each pair of points.

    Parameters
    ----------
    path : np.ndarray
        Array of shape (N, 3) representing the path in reciprocal space.
    n_points : int
        Number of points to interpolate between each pair of points.

    Returns
    -------
    np.ndarray
        Interpolated path as an array of shape (N * n_points, 3).
    """
    points: list[np.ndarray] = []
    for i in range(len(path) - 1):
        seg: np.ndarray = np.linspace(path[i], path[i + 1], n_points, endpoint=False)
        points.append(seg)
    points.append(path[-1][None, :])
    return np.vstack(points)


q_path = interpolate_path(path, n_points=100)
phonon.run_band_structure([q_path], with_eigenvectors=False)
bands: dict[str, np.ndarray] = phonon.get_band_structure_dict()
distances = bands["distances"][0]
band_frequencies = bands["frequencies"][0]

fig, ax = plt.subplots(figsize=(8, 6))
for i in range(band_frequencies.shape[1]):
    ax.plot(distances, band_frequencies[:, i], color="b")
ax.set_xticks([distances[0], distances[100], distances[200], distances[-1]])
ax.set_xticklabels(labels)
ax.set_xlim(distances[0], distances[-1])
ax.set_ylabel("Frequency (THz)")
ax.set_title("Phonon Dispersion of Graphene (toy model)")
ax.grid(visible=True)
plt.tight_layout()
plt.show()
