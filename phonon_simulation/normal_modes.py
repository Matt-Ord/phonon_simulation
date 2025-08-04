from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
from phonopy.api_phonopy import Phonopy
from phonopy.structure.atoms import PhonopyAtoms

if TYPE_CHECKING:
    from phonon_simulation.system import System


@dataclass(kw_only=True, frozen=True)
class NormalModeGrid:
    """Result of a normal mode calculation for a phonon system."""

    system: System

    omega: np.ndarray[Any, np.dtype[np.floating]]
    """The normal mode frequencies in angular frequency units."""
    modes: np.ndarray[Any, np.dtype[np.floating]]
    """The eigenvectors (normal modes) of the system."""
    q_vals: np.ndarray[Any, np.dtype[np.floating]]
    """The reduced wave vectors for the normal modes."""


def _build_phonopy_system(system: System) -> Phonopy:
    cell = PhonopyAtoms(
        symbols=[system.element],
        cell=system.cell,
        scaled_positions=[[0, 0, 0]],
    )
    supercell_matrix = np.diag(system.n_repeats)
    phonon = Phonopy(unitcell=cell, supercell_matrix=supercell_matrix)

    phonon.force_constants = system.force_constants

    return phonon


def calculate_normal_modes(system: System) -> NormalModeGrid:
    """
    Calculate and plot the normal modes and phonon dispersion relation for a simple 1D chain system.

    Returns a NormalModeResult containing frequencies, eigenvectors, and reduced wave vectors.
    """
    phonon = _build_phonopy_system(system)
    phonon.run_mesh(system.n_repeats, with_eigenvectors=True, is_mesh_symmetry=False)  # type: ignore[arg-type]
    mesh_dict: dict[str, np.ndarray] = phonon.get_mesh_dict()  # type: ignore[return-value]

    sorted_indices = np.argsort(mesh_dict["qpoints"][:, 0])  # cspell: disable-line
    return NormalModeResult(
        system=system,
        omega=mesh_dict["frequencies"][sorted_indices] * 1e12 * 2 * np.pi,
        modes=mesh_dict["eigenvectors"][sorted_indices][..., 0],
        q_vals=mesh_dict["qpoints"][sorted_indices, 0],
    )


@dataclass(frozen=True, kw_only=True)
class DispersionPath:
    """
    Represents a path in reciprocal space for 2D phonon calculations.

    Attributes
    ----------
    path_points : np.ndarray
        Array of k-point coordinates along the path (shape: (N, 3)).
    labels : list[str]
        List of labels for each k-point in the path.

    Methods
    -------
    __post_init__() -> None
        Validates that the number of labels matches the number of path points.
    """

    points: np.ndarray[tuple[int, int], np.dtype[np.floating]]
    labels: list[str]
    n_points: int = 100
    """Number of points to interpolate along the path."""

    def __post_init__(self) -> None:
        assert self.points.shape == (len(self.points), 3), (
            f"Path points must be a 2D array with shape (N, 3), got {self.points.shape}"
        )


@dataclass(kw_only=True, frozen=True)
class NormalModeDispersionResult:
    """Result of a normal mode calculation for a phonon system."""

    system: System
    path: DispersionPath

    distances: np.ndarray[Any, np.dtype[np.floating]]
    """The distances along the dispersion path."""
    band_frequencies: np.ndarray[Any, np.dtype[np.floating]]
    """The eigenvectors (normal modes) of the system."""


def calculate_normal_mode_dispersion(
    system: System, path: DispersionPath
) -> NormalModeDispersionResult:
    """
    Calculate and plot the normal modes and phonon dispersion relation for a simple 1D chain system.

    Returns a NormalModeResult containing frequencies, eigenvectors, and reduced wave vectors.
    """
    phonon = _build_phonopy_system(system)

    def interpolate_path(
        path: np.ndarray, n_points: int = 100
    ) -> np.ndarray[Any, np.dtype[np.floating]]:
        points = []
        for i in range(len(path) - 1):
            seg = np.linspace(path[i], path[i + 1], n_points, endpoint=False)
            points.append(seg)
        points.append(path[-1][None, :])
        return np.vstack(points)

    q_path = interpolate_path(path, n_points=100)
    phonon.run_band_structure([q_path], with_eigenvectors=True)
    bands: dict[str, np.ndarray] = phonon.get_band_structure_dict()
    bands["distances"][0]
    bands["frequencies"][0]

    return NormalModeDispersionResult(
        system=system,
        path=path,
        distances=bands["distances"][0],
        band_frequencies=bands["frequencies"][0],
    )
