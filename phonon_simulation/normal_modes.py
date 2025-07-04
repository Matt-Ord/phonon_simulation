from __future__ import annotations

from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np


@dataclass
class System:
    lattice_constant: tuple[float, float, float]
    number_of_repeats: tuple[int, int, int]
    spring_constant: tuple[float, float, float]
    mass: float


def calculate_normal_modes(system: System) -> None:
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
    print(f"Calculating normal modes for system: {system}")
    print("Normal mode frequencies (omega):", omega)
    print("Normal modes (eigenvectors):\n", modes)
    q_vals = 2 * np.pi * np.arange(n) / (n * system.lattice_constant[0])
    dispersion = np.sqrt(
        (2 * k / m) * (1 - np.cos(q_vals * system.lattice_constant[0]))
    )
    print("Wave vectors (q):", q_vals)
    print("Dispersion relation omega(q):", dispersion)
    plt.figure(figsize=(6, 4))  # type: ignore
    plt.plot(q_vals, dispersion, "o-", label="Dispersion relation")
    plt.axvline(
        np.pi / system.lattice_constant[0],
        color="r",
        linestyle="--",
        label="BZ boundary",
    )
    plt.xlabel("Wave vector q")
    plt.ylabel("Frequency ω(q)")
    plt.title("Phonon Dispersion Relation for 1D Chain")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
