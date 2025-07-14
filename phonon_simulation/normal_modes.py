from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import matplotlib.pyplot as plt
import numpy as np
from phonopy.api_phonopy import Phonopy
from phonopy.physical_units import get_physical_units
from phonopy.structure.atoms import PhonopyAtoms

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

EV = get_physical_units().EV
Angstrom = get_physical_units().Angstrom
AMU = get_physical_units().AMU
VaspToOmega = np.sqrt(EV / AMU) / Angstrom


@dataclass(kw_only=True, frozen=True)
class System:
    """Represents a lattice system used for phonon calculations."""

    element: str
    lattice_constant: tuple[float, float, float]
    n_repeats: tuple[int, int, int]
    spring_constant: tuple[float, float, float]


@dataclass(kw_only=True, frozen=True)
class NormalModeResult:
    """Result of a normal mode calculation for a phonon system."""

    system: System
    omega: np.ndarray[Any, np.dtype[np.floating]]
    modes: np.ndarray[Any, np.dtype[np.floating]]
    q_vals: np.ndarray[Any, np.dtype[np.floating]]
    mass: float

    def to_human_readable(self) -> str:
        """Convert the result to a text representation."""
        return (
            f"Calculating normal modes for system: {self.system}\n"
            "Normal mode frequencies (omega):\n"
            f"{np.array2string(self.omega, precision=6, separator=', ')}\n"
            "Wave vectors (q):\n"
            f"{np.array2string(self.q_vals, precision=6, separator=', ')}\n"
            "Normal modes (eigenvectors):\n"
            f"{np.array2string(self.modes, precision=6, separator=', ')}\n"
        )


def calculate_normal_modes(system: System) -> NormalModeResult:
    """
    Calculate and plot the normal modes and phonon dispersion relation for a simple 1D chain system.

    Returns a NormalModeResult containing frequencies, eigenvectors, and reduced wave vectors.
    """
    cell: PhonopyAtoms = PhonopyAtoms(
        symbols=[system.element],
        cell=[
            [system.lattice_constant[0], 0, 0],
            [0, system.lattice_constant[1], 0],
            [0, 0, system.lattice_constant[2]],
        ],
        scaled_positions=[[0, 0, 0]],
    )
    supercell_matrix: list[list[int]] = [
        [system.n_repeats[0], 0, 0],
        [0, 1, 0],
        [0, 0, 1],
    ]
    phonon: Phonopy = Phonopy(unitcell=cell, supercell_matrix=supercell_matrix)
    num_atoms: int = len(phonon.supercell)
    fc: np.ndarray = np.zeros((num_atoms, num_atoms, 3, 3), dtype=float)
    mass = cell.masses[0]
    # Set up force constants for 1D monoatomic chain
    for i in range(num_atoms):
        fc[i, i, 0, 0] = 2 * system.spring_constant[0]
        fc[i, (i - 1) % num_atoms, 0, 0] = -system.spring_constant[0]
        fc[i, (i + 1) % num_atoms, 0, 0] = -system.spring_constant[0]

    phonon.force_constants = fc
    mesh: tuple[int, int, int] = (101, 1, 1)
    phonon.run_mesh(mesh, with_eigenvectors=True, is_mesh_symmetry=False)
    mesh_dict: dict[str, np.ndarray] = phonon.get_mesh_dict()
    q_vals: np.ndarray = mesh_dict["qpoints"][:, 0]
    frequencies: np.ndarray = mesh_dict["frequencies"][:, 2]
    omega: np.ndarray = mesh_dict["frequencies"] * 1e12 * 2 * np.pi
    sorted_indices = np.argsort(q_vals)
    q_vals = q_vals[sorted_indices]
    frequencies = frequencies[sorted_indices]
    eigenvectors: np.ndarray = mesh_dict["eigenvectors"][sorted_indices]
    omega = omega[sorted_indices]
    modes = eigenvectors[..., 0]
    return NormalModeResult(
        system=system,
        omega=omega,
        modes=modes,
        q_vals=q_vals,
        mass=mass,
    )


def plot_dispersion(modes: NormalModeResult) -> tuple[Figure, Axes]:
    """Plot the phonon dispersion relation for a 1D chain on a graph, including analytical curve."""
    fig, ax = plt.subplots(figsize=(8, 6))
    # Numerical result from phonon simulation
    ax.plot(
        modes.q_vals,
        modes.omega,
        "o",
        label="Numerical",
    )
    ax.set_xlim(-0.6, 0.6)
    ax.axvline(0.5, color="gray", linestyle="--", label="First BZ boundary")
    ax.axvline(-0.5, color="gray", linestyle="--")
    ax.axhline(0, color="k", linestyle="-")
    ax.axvline(0, color="k", linestyle="-")
    ax.set_xlabel("Wave vector $q$ (Reduced units)")
    ax.set_ylabel("Frequency $\\omega(q)$")
    ax.set_title("Phonon Dispersion Relation")
    ax.grid(visible=True)
    ax.legend()
    fig.tight_layout()
    return fig, ax
