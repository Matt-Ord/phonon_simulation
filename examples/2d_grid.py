from __future__ import annotations

import time

import matplotlib.pyplot as plt
import numpy as np

from phonon_simulation.calculating import (
    DispersionPath,
    Lattice2DSystem,
    calculate_2d_modes,
)
from phonon_simulation.plotting import (
    plot_2d_dispersion,
    plot_2d_lattice,
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
    element="Si",
    lattice_vector_a=(1, 0, 0.0),
    lattice_vector_b=(0, 1, 0.0),
    n_repeatsa=6,
    n_repeatsb=6,
    k_nn=12.5,
    k_nnn=2.50,
)

mesh_dict, result = calculate_2d_modes(system)
fig, ax = plot_dispersion_path(path, system)
fig.show()

plot_2d_lattice(result, system)

plot_2d_dispersion(result.get_phonon(), system, path.points, path.labels)

print(
    "Process finished ---%s seconds ---" % (time.time() - start_time)
)  # just to check, has  no real use
plt.show()
