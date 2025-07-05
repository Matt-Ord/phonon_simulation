from __future__ import annotations

from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np


@dataclass
class System:
    """
    Represents a 1D, 2D, or 3D lattice system for phonon calculations.

    Attributes
    ----------
    The physical system containing lattice constant, number of repeats, spring constants, and mass of particles.
    """

    lattice_constant: tuple[float, float, float]
    number_of_repeats: tuple[int, int, int]
    spring_constant: tuple[float, float, float]
    mass: float


def calculate_normal_modes(
    system: System, output_file: str = "normal_modes_output.txt"
) -> None:
    """
    Calculate and plot the normal modes and phonon dispersion relation for a simple 1D chain system.

    Parameters
    ----------
    system : System
        The physical system containing lattice constant, number of repeats, spring constants, and mass of particles.

    This function prints the normal mode frequencies, eigenvectors, wave vectors, and dispersion relation,
    and displays a plot dispersion relation.
    """
    n = system.number_of_repeats[0]
    system.lattice_constant[0]
    k = system.spring_constant[0]
    m = system.mass
    d = np.zeros((n, n))

    for i in range(n):
        d[i, i] = 2 * k / m
        d[i, (i - 1) % n] = -k / m
        d[i, (i + 1) % n] = -k / m

    omega2, modes = np.linalg.eigh(d)
    omega = np.sqrt(np.abs(omega2))
    q_vals = 2 * np.pi * np.arange(n) / (n * system.lattice_constant[0])
    dispersion = np.sqrt(
        (2 * k / m) * (1 - np.cos(q_vals * system.lattice_constant[0]))
    )
    # Prepare output string
    # Generate output file name from input parameters if not provided

    output = []
    output.extend(
        [
            f"Calculating normal modes for system: {system}\n",
            "Normal mode frequencies (omega):\n",
            np.array2string(omega, precision=6, separator=", ") + "\n",
            "Normal modes (eigenvectors):\n",
            np.array2string(modes, precision=6, separator=", ") + "\n",
            "Wave vectors (q):\n",
            np.array2string(q_vals, precision=6, separator=", ") + "\n",
            "Dispersion relation omega(q):\n",
            np.array2string(dispersion, precision=6, separator=", ") + "\n",
        ]
    )

    # Save to file
    with open(output_file, "w", encoding="utf-8") as f:
        f.writelines(output)

    print(f"Output saved to '{output_file}'.")
    # I had to write type: ignore to avoid type errors in the following lines from plt.something. Not sure why but it works with or without the type: ignore comments.
    # Plotting the dispersion relation
    plt.figure(figsize=(6, 4))
    plt.plot(q_vals, dispersion, "o-", label="Dispersion relation")
    plt.axvline(
        np.pi / system.lattice_constant[0],
        color="r",
        linestyle="--",
        label="BZ boundary",
    )
    plt.axvline(-np.pi / system.lattice_constant[0], color="r", linestyle="--")
    plt.xlabel("Wave vector q")
    plt.ylabel("Frequency ω(q)")
    plt.title("Phonon Dispersion Relation for 1D Chain")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plot_file = output_file.rsplit(".", 1)[0] + "_plot.png"
    plt.savefig(plot_file)
    print(f"Plot saved to '{plot_file}'.")
    plt.show()
