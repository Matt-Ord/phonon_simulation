from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from phonon_simulation.normal_modes import (
    SquareLattice2DSystem,
    calculate_2d_square_modes,
    plot_2d_square_bz_path,
    plot_2d_square_dispersion,
    plot_2d_square_lattice,
)

path = np.array(
    [
        [0.0, 0.0, 0.0],
        [0.5, 0.0, 0.0],
        [0.5, 0.5, 0.0],
        [0.0, 0.0, 0.0],
    ]
)
labels = [r"$\Gamma$", "X", "M", r"$\Gamma$"]

system = SquareLattice2DSystem(
    element="C",
    lattice_constantx=1.0,  # At the moment the code can't handle different lattice constants in x and y but I will be able to add this later
    lattice_constanty=1.0,
    n_repeatsx=6,
    n_repeatsy=5,
    k_nn=12.5,  # These values are used for comparison with a known plot of a 2D square lattice
    k_nnn=6.25,
)

mesh_dict, phonon = calculate_2d_square_modes(system)
plot_2d_square_bz_path(path, labels)
plot_2d_square_lattice(phonon, system)
plot_2d_square_dispersion(mesh_dict, phonon, system, path, labels)
plt.show()
