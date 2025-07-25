from __future__ import annotations

import time

import matplotlib.pyplot as plt
import numpy as np

from phonon_simulation.calculating import (
    DispersionPath,
    Lattice2DSystem,
    calculate_2d__lattice_bonds,
    calculate_2d__modes,
)
from phonon_simulation.plotting import (
    plot_2d__dispersion,
    plot_2d__lattice,
    plot_dispersion_path,
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
    element="C",
    lattice_vector_a=(1, -1, 0.0),
    lattice_vector_b=(1, 1, 0.0),
    n_repeatsa=6,
    n_repeatsb=6,
    k_nn=12.5,
    k_nnn=6.25,
)

mesh_dict, result = calculate_2d__modes(system)
fig, ax = plot_dispersion_path(path)
fig.show()

bonds = calculate_2d__lattice_bonds(result, system)
plot_2d__lattice(result, system, bonds)

plot_2d__dispersion(mesh_dict, result.get_phonon(), system, path.points, path.labels)


print(
    "Process finished ---%s seconds ---" % (time.time() - start_time)
)  # just to check, has  no real use
plt.show()
