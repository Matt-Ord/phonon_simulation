from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from typing_extensions import TypedDict


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


class NormalModeResults(TypedDict):
    """
    Results of normal mode calculations for a phonon system.

    Attributes
    ----------
    system : System
        The physical system outputs from normal mode calculations.
    """

    system: System
    omega: np.ndarray
    modes: np.ndarray
    q_vals: np.ndarray
    dispersion: np.ndarray


def calculate_normal_modes(system: System) -> NormalModeResults:
    """
    Calculate and plot the normal modes and phonon dispersion relation for a simple 1D chain system.

    Parameters
    ----------
    system : System
        The physical system containing lattice constant, number of repeats, spring constants, and mass of particles.

    This function returns a dictionary containing the normal mode frequencies, eigenvectors
    (modes), wave vectors, and dispersion relation.
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
    return {
        "system": system,
        "omega": omega,
        "modes": modes,
        "q_vals": q_vals,
        "dispersion": dispersion,
    }


def plot_dispersion(q_vals: np.ndarray, dispersion: np.ndarray, system: System) -> None:
    """Plot the phonon dispersion relation for a 1D chain on a graph."""
    system.lattice_constant[0]
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
    plt.grid(visible=True)
    plt.legend()
    plt.tight_layout()


def save_results(results: NormalModeResults, folder: str) -> None:
    """Save the results of normal mode calculations and the plot to a specified folder."""
    system = results["system"]
    file_name = (
        f"1D_N{system.number_of_repeats[0]}"
        f"_a{system.lattice_constant[0]}"
        f"_k{system.spring_constant[0]}"
        f"_m{system.mass}"
    )
    output_file = Path(folder) / f"{file_name}.txt"
    plot_file = Path(folder) / f"{file_name}_plot.png"

    output = [
        f"Calculating normal modes for system: {system}\n",
        "Normal mode frequencies (omega):\n",
        np.array2string(results["omega"], precision=6, separator=", ") + "\n",
        "Wave vectors (q):\n",
        np.array2string(results["q_vals"], precision=6, separator=", ") + "\n",
        "Dispersion relation omega(q):\n",
        np.array2string(results["dispersion"], precision=6, separator=", ") + "\n",
        "Normal modes (eigenvectors):\n",
        np.array2string(results["modes"], precision=6, separator=", ") + "\n",
    ]

    with output_file.open("w", encoding="utf-8") as f:
        f.writelines(output)

    plt.savefig(plot_file)
    plt.show()
