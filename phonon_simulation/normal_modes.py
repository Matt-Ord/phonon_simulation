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
    """Represents a lattice system used for phonon calculations."""

    lattice_constant: tuple[float, float, float]
    n_repeats: tuple[int, int, int]
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

    This function returns a `NormalModeResult` containing the normal mode frequencies, eigenvectors
    (modes), wave vectors, and dispersion.

    Parameters
    ----------
    system : System
        The physical system containing lattice constant, number of repeats, spring constants, and mass of particles.

    """
    n = system.n_repeats[0]
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
    ax.set_xlim(
        -np.pi / modes.system.lattice_constant[0],
        np.pi / modes.system.lattice_constant[0],
    )
    ax.set_xlabel("Wave vector $q$")
    ax.set_ylabel("Frequency $\\omega(q)$")
    ax.set_title("Phonon Dispersion Relation")
    ax.grid(visible=True)
    ax.legend()
    fig.tight_layout()
    return fig, ax
