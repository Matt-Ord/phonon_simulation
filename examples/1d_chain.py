from __future__ import annotations

from pathlib import Path

from matplotlib import pyplot as plt
from phonopy.physical_units import get_physical_units

from phonon_simulation.normal_modes import (
    System,
    calculate_normal_modes,
    plot_dispersion,
)

AMU = get_physical_units().AMU
if __name__ == "__main__":
    # An example of simulating a simple 1D chain of atoms
    # To simulate a 1D system we set the lattice constant along the y/z directions to zero
    chain = System(
        lattice_constant=(20.0e-10, 0.0, 0.0),
        n_repeats=(100, 1, 1),
        spring_constant=(16, 0.0, 0.0),
        mass=28.0855 * AMU,  # Silicon mass in kg
    )

    modes = calculate_normal_modes(chain)

    # Save results and plot to a folder
    folder = Path("./examples")
    modes_output = folder / "1d_chain.normal_modes.txt"
    modes_output.write_text(modes.to_human_readable(), encoding="utf-8")

    plot_output = folder / "1d_chain.dispersion_plot.png"
    fig, _ = plot_dispersion(modes)
    fig.savefig(plot_output)
    plt.show()
