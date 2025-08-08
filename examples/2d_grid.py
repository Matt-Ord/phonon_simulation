from __future__ import annotations

import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from phonon_simulation.calculating import (
    DispersionPath,
    Lattice2DSystem,
    calculate_2d_modes,
)
from phonon_simulation.plotting import (
    interpolate_path,
    plot_2d_dispersion_band,
    plot_2d_dispersion_band_and_mesh,
    plot_2d_dispersion_mesh,
    plot_2d_lattice,
    plot_2d_mesh_3d_scatter,
    plot_2d_mesh_3d_surface,
    plot_dispersion_path,
    plot_mesh_frequency_difference,
)

start_time = time.time()

path = DispersionPath(
    points=np.array(
        [
            [0.0, 0.0, 0.0],
            [0.5, 0.0, 0.0],
            [0.5, 0.5, 0.0],
            [0.0, 0.0, 0.0],
        ]
    ),
    labels=[r"$\Gamma$", "X", "M", r"$\Gamma$"],
)

system = Lattice2DSystem(
    element="Si",
    lattice_vector_a=(1, 0, 0.0),
    lattice_vector_b=(0, 1, 0.0),
    n_repeatsa=20,
    n_repeatsb=20,  # need to use factors of 2 its seems so Monkhorst Pack grid picks up points along diagonal
    k_nn=12.5,
    k_nnn=2.50,
)


mesh_dict, result, normal_modes_2d = calculate_2d_modes(system, vacancy=False)
folder = Path("./examples")
modes_output = folder / "2d_grid.normal_modes.txt"
modes_output.write_text(normal_modes_2d.to_human_readable(), encoding="utf-8")


fig, ax = plot_dispersion_path(path, system)
fig.show()
plot_2d_lattice(result, system, vacancy=False)

# Calculate band structure (smooth curve)
q_path_band = interpolate_path(path.points, n_points=100)
result.get_phonon().run_band_structure([q_path_band], with_eigenvectors=True)
bands = result.get_phonon().get_band_structure_dict()


plot_2d_dispersion_band_and_mesh(
    bands, mesh_dict, path.points, path.labels, system
)  # Plots both band and mesh in one figure
plot_2d_dispersion_band(
    bands, path.points, path.labels, system
)  # Plots only the band structure
plot_2d_dispersion_mesh(
    bands, mesh_dict, path.points, path.labels, system
)  # Plots only the mesh structure

# Both these two plots use copilot generated code at the moment just to check the mesh produces the expected results
plot_2d_mesh_3d_scatter(mesh_dict, system)  # Plots 3D scatter plot of mesh points
plot_2d_mesh_3d_surface(mesh_dict, system)  # Plots 3D surface plot of mesh points

"""
# --- Print force constants for bonds connected to the central atom ---

positions = result.get_positions()
a_vec = np.array(system.lattice_vector_a)
b_vec = np.array(system.lattice_vector_b)

central_atom_index = find_central_atom_index(system, positions, a_vec, b_vec)

# Get all bonds (do not exclude the central atom here)
nn_pairs, nnn_pairs = find_lattice_bond_pairs(
    positions, system, vacancy=False, central_atom_index=central_atom_index
)

fc = result.get_phonon().force_constants

print(
    f"\nForce constants for bonds connected to the central atom (index {central_atom_index}):"
)
for i, j in nn_pairs + nnn_pairs:
    if central_atom_index in {i, j}:
        print(f"Bond ({i}, {j}):")
        print(fc[i, j])
"""

mesh_dict_no_vac, result_no_vac, _ = calculate_2d_modes(system, vacancy=False)
mesh_dict_vac, result_vac, _ = calculate_2d_modes(system, vacancy=True)

# Plot the absolute difference
plot_mesh_frequency_difference(
    mesh_dict_no_vac, mesh_dict_vac, system, diff_type="absolute"
)  # Currently gives zero difference which is a problem
print("Max abs frequency (no vac):", np.max(np.abs(mesh_dict_no_vac["frequencies"])))
print("Max abs frequency (vac):", np.max(np.abs(mesh_dict_vac["frequencies"])))
print(
    "Max abs difference:",
    np.max(np.abs(mesh_dict_no_vac["frequencies"] - mesh_dict_vac["frequencies"])),
)
print(
    "Any difference?:",
    np.any(
        np.abs(mesh_dict_no_vac["frequencies"] - mesh_dict_vac["frequencies"]) > 1e-8
    ),
)  # Currently is False, which is a problem as there should be some difference


print(
    "Process finished ---%s seconds ---" % (time.time() - start_time)
)  # just to check, has  no real use

plt.show()
