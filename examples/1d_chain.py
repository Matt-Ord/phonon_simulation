from __future__ import annotations

from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt

from phonon_simulation.normal_modes import (
    calculate_normal_modes,
    plot_dispersion,
)
from phonon_simulation.system import NearestNeighbourSystem1D

if __name__ == "__main__":
    chain = NearestNeighbourSystem1D(
        element="Au",
        cell=np.diag([1.0, 1.0, 1.0]),
        n_repeats=21,
        spring_constant=1.0,
    )

    modes = calculate_normal_modes(chain)

    # Save results and plot to a folder
    folder = Path("./examples")
    modes_output = folder / "1d_chain.normal_modes.txt"
    modes_output.write_text(modes.to_human_readable(), encoding="utf-8")

    plot_output = folder / "1d_chain.dispersion_plot.png"
    fig, _ = plot_dispersion(modes)
    fig.savefig(plot_output)
    # View and plot the supercell structure
    print("Supercell structure:")
    print(chain)

    # Optionally, plot the atomic positions in the supercell (1D chain)

    n_atoms = chain.n_repeats[0]
    a = chain.cell[0, 0]
    positions = np.array([[i * a, 0, 0] for i in range(n_atoms)])
    fig_supercell, ax = plt.subplots()
    ax.scatter(positions[:, 0], positions[:, 1], s=50, c="blue")
    ax.set_xlabel("x (Å)")
    ax.set_yticks([])
    ax.set_title("1D Chain Supercell Structure")
    ax.grid(visible=True)
    plt.show()
    input()
