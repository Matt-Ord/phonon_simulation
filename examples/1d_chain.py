from __future__ import annotations

from pathlib import Path

from matplotlib import pyplot as plt

from phonon_simulation.normal_modes import (
    System,
    calculate_normal_modes,
    plot_dispersion,
)

if __name__ == "__main__":
    chain = System(
        element="Si",
        lattice_constant=(1, 1, 1),
        n_repeats=(100, 1, 1),
        spring_constant=(1, 0.0, 0.0),
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
