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


def build_force_constants_2d(
    system: Lattice2DSystem, result: PhononSystem2DResult
) -> np.ndarray:
    """
    Build the force constant matrix for a 2D lattice system including nearest and next nearest neighbor interactions.

    using the atomic positions from a PhononSystem2DResult object.

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
    nn_bond_a = np.linalg.norm(a_vec)
    nn_bond_b = np.linalg.norm(b_vec)
    nnn_bond_diag1 = np.linalg.norm(a_vec + b_vec)
    nnn_bond_diag2 = np.linalg.norm(a_vec - b_vec)
    tol = 0.005

    for i in range(num_atoms):
        for j in range(num_atoms):
            if i == j:
                continue
            vec = positions[j] - positions[i]
            dist = np.linalg.norm(vec)
            # Nearest-neighbor: along a_vec or b_vec
            if np.isclose(dist, nn_bond_a, atol=tol) or np.isclose(
                dist, nn_bond_b, atol=tol
            ):
                direction = vec / np.linalg.norm(vec)
                for d1 in range(3):
                    for d2 in range(3):
                        fc[i, j, d1, d2] += -system.k_nn * direction[d1] * direction[d2]
                        fc[i, i, d1, d2] += system.k_nn * direction[d1] * direction[d2]
            # Next-nearest-neighbor: along both diagonals
            elif np.isclose(dist, nnn_bond_diag1, atol=tol) or np.isclose(
                dist, nnn_bond_diag2, atol=tol
            ):
                direction = vec / np.linalg.norm(vec)
                for d1 in range(3):
                    for d2 in range(3):
                        fc[i, j, d1, d2] += (
                            -system.k_nnn * direction[d1] * direction[d2]
                        )
                        fc[i, i, d1, d2] += system.k_nnn * direction[d1] * direction[d2]
    return fc


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


def calculate_2d__modes(
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
    # Create a temporary result object to pass positions
    temp_result = PhononSystem2DResult(cell=cell, phonon=phonon, positions=positions)
    fc = build_force_constants_2d(system, temp_result)
    phonon.force_constants = fc
    mesh = (201, 201, 1)
    phonon.run_mesh(
        mesh, with_eigenvectors=True, is_mesh_symmetry=False, is_gamma_center=True
    )
    mesh_dict: dict[str, np.ndarray] = phonon.get_mesh_dict()
    positions = phonon.supercell.get_positions()
    result = PhononSystem2DResult(cell=cell, phonon=phonon, positions=positions)
    return mesh_dict, result


def calculate_2d__lattice_bonds(
    result: PhononSystem2DResult, system: Lattice2DSystem
) -> list[tuple[int, int, str]]:
    """
    Calculate the bonds (pairs of atom indices and bond type) for a 2D  lattice supercell only for plotting, not used for force calculations.

    Returns a list of tuples: (i, j, bond_type), where bond_type is 'nn' or 'nnn'.
    """
    positions = result.get_positions()
    a_vec = np.array(system.lattice_vector_a)
    b_vec = np.array(system.lattice_vector_b)
    nn_bond_a = np.linalg.norm(a_vec)
    nn_bond_b = np.linalg.norm(b_vec)
    nnn_bond_diag1 = np.linalg.norm(a_vec + b_vec)
    nnn_bond_diag2 = np.linalg.norm(a_vec - b_vec)
    tol = 0.05
    bonds: list[tuple[int, int, str]] = []
    for i in range(positions.shape[0]):
        for j in range(i + 1, positions.shape[0]):
            disp = positions[j] - positions[i]
            dist = np.linalg.norm(disp)
            # Nearest neighbour: along a_vec or b_vec
            if np.isclose(dist, nn_bond_a, atol=tol) or np.isclose(
                dist, nn_bond_b, atol=tol
            ):
                bonds.append((i, j, "nn"))
            # Next nearest neighbour: along diagonals
            elif np.isclose(dist, nnn_bond_diag1, atol=tol) or np.isclose(
                dist, nnn_bond_diag2, atol=tol
            ):
                bonds.append((i, j, "nnn"))
    return bonds
