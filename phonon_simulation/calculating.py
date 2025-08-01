from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from phonopy.api_phonopy import Phonopy
from phonopy.structure.atoms import PhonopyAtoms


@dataclass(kw_only=True, frozen=True)
class Lattice2DSystem:
    """
    Represents a 2D  lattice system for phonon calculations.

    Attributes
    ----------
    element : str
        The chemical symbol of the element (e.g., "C" for carbon).
    lattice_vector_a : tuple[float, float, float]
        Lattice vector along the a direction, in Angstroms, (1 Å = 1e-10 m).
    lattice_vector_b : tuple[float, float, float]
        Lattice vector along the b direction, in Angstroms, (1 Å = 1e-10 m).
    n_repeatsa : int
        Number of unit cells repeated along the x direction.
    n_repeatsb : int
        Number of unit cells repeated along the y direction.
    k_nn : float
        Nearest neighbor spring constant, in electronvolts per Angstrom d (1 eV/Å² = 16.02 N/m).
    k_nnn : float
        Next nearest neighbor spring constant, in electronvolts per Angstrom d (1 eV/Å² = 16.02 N/m).
    """

    element: str
    lattice_vector_a: tuple[float, float, float]
    lattice_vector_b: tuple[float, float, float]
    n_repeatsa: int
    n_repeatsb: int
    k_nn: float
    k_nnn: float

    @property
    def mass(self) -> float:
        """Return the mass of the element in atomic mass units(1 AMU = 1.6605402e-27 kg)."""
        cell = PhonopyAtoms(
            symbols=[self.element],
            cell=[
                list(self.lattice_vector_a),
                list(self.lattice_vector_b),
                [0, 0, 1],  # For 2D, z is just a dummy direction
            ],
            scaled_positions=[[0, 0, 0]],
        )
        return cell.masses[0]


@dataclass(frozen=True)
class PhononSystem2DResult:
    """
    Stores the results of a 2D phonon system calculation.

    Attributes
    ----------
    cell : PhonopyAtoms
        The atomic structure of the system.
    phonon : Phonopy
        The Phonopy object containing phonon calculation results.
    positions : np.ndarray
        The atomic positions in the supercell.
    """

    cell: PhonopyAtoms
    phonon: Phonopy
    positions: np.ndarray

    def get_cell(self) -> PhonopyAtoms:
        """
        Return the atomic structure (PhonopyAtoms) of the system.

        Returns
        -------
        PhonopyAtoms
            The atomic structure of the system.
        """
        return self.cell

    def get_phonon(self) -> Phonopy:
        """
        Return the Phonopy object containing phonon calculation results.

        Returns
        -------
        Phonopy
            The Phonopy object containing phonon calculation results.
        """
        return self.phonon

    def get_positions(self) -> np.ndarray:
        """
        Return the atomic positions in the supercell.

        Returns
        -------
        np.ndarray
            The atomic positions in the supercell.
        """
        return self.positions


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

    def __post_init__(self) -> None:
        assert self.points.shape == (len(self.points), 3), (
            f"Path points must be a 2D array with shape (N, 3), got {self.points.shape}"
        )


def build_force_constants_2d(
    system: Lattice2DSystem, result: PhononSystem2DResult
) -> np.ndarray:
    """
    Build the force constant matrix for a 2D lattice system including nearest and next nearest neighbor interactions using the atomic positions from a PhononSystem2DResult object and bond pairs found in find_lattice_bond_pairs.

    Parameters
    ----------
    system : Lattice2DSystem
        The 2D lattice system for which to build the force constant matrix.
    result : PhononSystem2DResult
        The result object containing cell, phonon, and positions.

    Returns
    -------
    np.ndarray
        The force constant matrix of shape (num_atoms, num_atoms, 3, 3).
    """
    num_atoms = system.n_repeatsa * system.n_repeatsb
    positions = result.get_positions()
    fc = np.zeros((num_atoms, num_atoms, 3, 3), dtype=float)
    a_vec = np.array(system.lattice_vector_a)
    b_vec = np.array(system.lattice_vector_b)

    nn_pairs, nnn_pairs = find_lattice_bond_pairs(positions, a_vec, b_vec)

    for i, j in nn_pairs:  # Nearest neighbor bonds
        displacement_vector = positions[j] - positions[i]
        direction = displacement_vector / np.linalg.norm(displacement_vector)
        for d1 in range(3):
            for d2 in range(3):
                fc[i, j, d1, d2] += -system.k_nn * direction[d1] * direction[d2]
                fc[i, i, d1, d2] += system.k_nn * direction[d1] * direction[d2]

    for i, j in nnn_pairs:  # Next-nearest neighbor bonds
        displacement_vector = positions[j] - positions[i]
        direction = displacement_vector / np.linalg.norm(displacement_vector)
        for d1 in range(3):
            for d2 in range(3):
                fc[i, j, d1, d2] += -system.k_nnn * direction[d1] * direction[d2]
                fc[i, i, d1, d2] += system.k_nnn * direction[d1] * direction[d2]
    return fc


def calculate_2d_modes(
    system: Lattice2DSystem,
) -> tuple[dict[str, np.ndarray], PhononSystem2DResult]:
    """
    Calculate the phonon modes for a 2D  lattice system.

    Parameters
    ----------
    system : Lattice2DSystem
        The 2D  lattice system for which to calculate the phonon modes.

    Returns
    -------
    tuple[dict[str, np.ndarray], PhononSystem2DResult]
        A tuple containing the mesh dictionary with phonon properties and the PhononSystem2DResult object.
    """
    cell = PhonopyAtoms(
        symbols=[system.element],
        cell=[
            list(system.lattice_vector_a),
            list(system.lattice_vector_b),
            [0, 0, 1],
        ],
        scaled_positions=[[0, 0, 0]],
    )
    supercell_matrix = [[system.n_repeatsa, 0, 0], [0, system.n_repeatsb, 0], [0, 0, 1]]
    phonon = Phonopy(unitcell=cell, supercell_matrix=supercell_matrix)
    positions = phonon.supercell.get_positions()

    temp_result = PhononSystem2DResult(
        cell=cell, phonon=phonon, positions=positions
    )  # Create a temporary result object to pass positions
    fc = build_force_constants_2d(system, temp_result)
    phonon.force_constants = fc
    mesh = (system.n_repeatsa, system.n_repeatsb, 1)
    phonon.run_mesh(
        mesh, with_eigenvectors=True, is_mesh_symmetry=False, is_gamma_center=True
    )
    mesh_dict: dict[str, np.ndarray] = phonon.get_mesh_dict()
    positions = phonon.supercell.get_positions()
    result = PhononSystem2DResult(cell=cell, phonon=phonon, positions=positions)
    return mesh_dict, result


def find_lattice_bond_pairs(
    positions: np.ndarray,
    a_vec: np.ndarray,
    b_vec: np.ndarray,
    rtol: float = 1e-3,
) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    """
    Find nearest neighbor (nn) and next-nearest neighbor (nnn) atom pairs in a 2D lattice.

    Parameters
    ----------
    positions : np.ndarray
        Array of atomic positions in the supercell.
    a_vec : np.ndarray
        Lattice vector along the a direction.
    b_vec : np.ndarray
        Lattice vector along the b direction.
    rtol : float
        Relative tolerance for comparing vectors.

    Returns
    -------
    tuple[list[tuple[int, int]], list[tuple[int, int]]]
        A tuple containing two lists: nn_pairs and nnn_pairs, each as lists of (i, j) index pairs.
    """
    nn_pairs: list[tuple[int, int]] = []
    nnn_pairs: list[tuple[int, int]] = []

    nnn_diag1 = a_vec + b_vec
    nnn_diag2 = a_vec - b_vec
    for i in range(positions.shape[0]):
        for j in range(positions.shape[0]):
            disp = positions[j] - positions[i]

            if np.allclose(disp, a_vec, rtol=rtol) or np.allclose(
                disp, b_vec, rtol=rtol
            ):
                nn_pairs.append((i, j))

            elif np.allclose(disp, nnn_diag1, rtol=rtol) or np.allclose(
                disp, nnn_diag2, rtol=rtol
            ):
                nnn_pairs.append((i, j))
    return nn_pairs, nnn_pairs
