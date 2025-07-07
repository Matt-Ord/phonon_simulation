from __future__ import annotations

from pathlib import Path

from matplotlib import pyplot as plt

from phonon_simulation.normal_modes import (
    System,
    calculate_normal_modes,
    plot_dispersion,
)

one_d_chain = System(
    lattice_constant=(1.0, 0.0, 0.0),
    number_of_repeats=(38, 1, 1),
    spring_constant=(1.0, 0.0, 0.0),
    mass=1.0,
)

# Calculate normal modes and get results dictionary
results = calculate_normal_modes(one_d_chain)


# Save results and plot to a folder
folder = Path(r"./examples")
modes_output = folder / "normal_modes.txt"
modes_output.write_text(results.to_human_readable(), encoding="utf-8")

plot_output = folder / "dispersion_plot.png"
fig, _ = plot_dispersion(results)
fig.savefig(plot_output)
plt.show()
