from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import matplotlib.pyplot as plt
import numpy as np

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure


@dataclass(kw_only=True, frozen=True)
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


@dataclass(kw_only=True, frozen=True)
class NormalModeResult:
    """Result of a normal mode calculation for a phonon system."""

    system: System
    omega: np.ndarray[Any, np.dtype[np.floating]]
    modes: np.ndarray[Any, np.dtype[np.floating]]
    q_vals: np.ndarray[Any, np.dtype[np.floating]]
    dispersion: np.ndarray[Any, np.dtype[np.floating]]

    def to_human_readable(self) -> str:
        """Convert the result to a text representation."""
        return (
            f"Calculating normal modes for system: {self.system}\n"
            "Normal mode frequencies (omega):\n"
            f"{np.array2string(self.omega, precision=6, separator=', ')}\n"
            "Wave vectors (q):\n"
            f"{np.array2string(self.q_vals, precision=6, separator=', ')}\n"
            "Dispersion relation omega(q):\n"
            f"{np.array2string(self.dispersion, precision=6, separator=', ')}\n"
            "Normal modes (eigenvectors):\n"
            f"{np.array2string(self.modes, precision=6, separator=', ')}\n"
        )


def calculate_normal_modes(system: System) -> NormalModeResult:
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
    q_vals = np.fft.fftfreq(n, d=system.lattice_constant[0] / (2 * np.pi))
    dispersion = np.sqrt(
        (2 * k / m) * (1 - np.cos(q_vals * system.lattice_constant[0]))
    )
    return NormalModeResult(
        system=system,
        omega=omega,
        modes=modes,
        q_vals=q_vals,
        dispersion=dispersion,
    )


def plot_dispersion(modes: NormalModeResult) -> tuple[Figure, Axes]:
    """Plot the phonon dispersion relation for a 1D chain on a graph."""
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(
        np.fft.ifftshift(modes.q_vals),
        np.fft.ifftshift(modes.dispersion),
        "o-",
        label="Dispersion relation",
    )
    ax.axvline(
        np.pi / modes.system.lattice_constant[0],
        color="r",
        linestyle="--",
        label="BZ boundary",
    )
    ax.axvline(-np.pi / modes.system.lattice_constant[0], color="r", linestyle="--")
    ax.set_xlabel("Wave vector q")
    ax.set_ylabel("Frequency ω(q)")
    ax.set_title("Phonon Dispersion Relation for 1D Chain")
    ax.grid(visible=True)
    ax.legend()
    fig.tight_layout()
    return fig, ax
