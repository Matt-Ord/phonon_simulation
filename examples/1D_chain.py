from __future__ import annotations

from phonon_simulation.normal_modes import (
    System,
    calculate_normal_modes,
    plot_dispersion,
    save_results,
)

one_d_chain = System(
    lattice_constant=(1.0, 0.0, 0.0),
    number_of_repeats=(38, 1, 1),
    spring_constant=(1.0, 0.0, 0.0),
    mass=1.0,
)

# Calculate normal modes and get results dictionary
results = calculate_normal_modes(one_d_chain)

# Plot the dispersion relation
plot_dispersion(results["q_vals"], results["dispersion"], one_d_chain)

# Save results and plot to a folder
folder = r"C:\Users\jm\Documents\Physics\Phonon_Project\OutputFolder1"
save_results(results, folder)
