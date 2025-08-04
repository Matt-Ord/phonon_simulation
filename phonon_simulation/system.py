from __future__ import annotations

from typing import Any

import numpy as np
from phonopy.structure.atoms import PhonopyAtoms  # type: ignore[import]


class System:
    """Represents a lattice system used for phonon calculations."""

    _element: str
    _cell: np.ndarray[tuple[int, int], np.dtype[np.floating]]
    _n_repeats: tuple[int, int, int]
    _force_constants: np.ndarray[Any, np.dtype[np.floating]]

    def __init__(
        self,
        element: str,
        cell: np.ndarray[tuple[int, int], np.dtype[np.floating]],
        n_repeats: tuple[int, int, int],
        force_constants: np.ndarray[Any, np.dtype[np.floating]] | None = None,
    ) -> None:
        """Initialize the System with element, cell, repeats and spring constant."""
        self._element = element
        self._cell = np.array(cell, dtype=np.float64)
        self._n_repeats = n_repeats
        self._force_constants = (
            np.zeros(
                (np.prod(n_repeats), 3, 3),
                dtype=np.float64,
            )
            if force_constants is None
            else force_constants
        )
        assert self._force_constants.shape == (np.prod(n_repeats), 3, 3)

    @property
    def element(self) -> str:
        """Chemical symbol of the element."""
        return self._element

    @property
    def cell(self) -> np.ndarray[tuple[int, int], np.dtype[np.floating]]:
        """Lattice cell of the system."""
        return self._cell

    @property
    def n_repeats(self) -> tuple[int, int, int]:
        """Number of repeats in each direction."""
        return self._n_repeats

    @property
    def force_constants(self) -> np.ndarray[Any, np.dtype[np.floating]]:
        """Force constants of the system."""
        return self._force_constants

    @property
    def mass(self) -> float:
        """Mass of the element in atomic mass units."""
        cell = PhonopyAtoms(
            symbols=[self.element],
            cell=self.cell,
            scaled_positions=[[0, 0, 0]],
        )
        return cell.masses[0]


class NearestNeighbourSystem1D(System):
    """Represents a 1D lattice system used for phonon calculations."""

    def __init__(
        self,
        element: str,
        cell: np.ndarray[tuple[int, int], np.dtype[np.floating]],
        n_repeats: int,
        spring_constant: float = 1.0,
    ) -> None:
        """Initialize the 1D System with element, cell, repeats and spring constant."""
        # TODO: build the force constant matrix
        super().__init__(element, cell, n_repeats=(n_repeats, 1, 1), force_constants=fc)


class NearestNeighbourSystem2D(System):
    """Represents a 2D lattice system used for phonon calculations."""

    def __init__(
        self,
        element: str,
        cell: np.ndarray[tuple[int, int], np.dtype[np.floating]],
        n_repeats: tuple[int, int],
        spring_constant: float = 1.0,
        next_neighbour_spring_constant: float = 0.0,
    ) -> None:
        """Initialize the 2D System with element, cell, repeats and spring constant."""
        # TODO: build the force constant matrix
        super().__init__(element, cell, n_repeats=(*n_repeats, 1), force_constants=fc)

    # def with_nearest_neighbor_forces(
    #     self,
    #     forces: NeighbourhoodForces,
    # ) -> System:
    #     """Return a new System with nearest neighbor spring constant."""
    #     fc = self._fc.copy().reshape(*self.n_repeats, 3, 3)
    #     forces = _neighbourhood_forces_as_full(forces)
    #     mg = np.meshgrid([0, 1, -1], [0, 1, -1], [0, 1, -1], indexing="ij")

    #     print(np.ndindex(3, 3, 3))
    #     print(mg[0][0, 0, 0], mg[0][1, 0, 0], mg[0][2, 0, 0])
    #     print(mg[1][0, 0, 0], mg[1][0, 1, 0], mg[1][0, 2, 0])
    #     print(mg[2][0, 0, 0], mg[2][1, 0, 0], mg[2][2, 0, 0])

    #     print(mg[0][0, 0, 0], mg[0][1, 0, 0], mg[0][2, 0, 0])
    #     print(mg[1][0, 0, 0], mg[1][1, 0, 0], mg[1][2, 0, 0])
    #     print(mg[2][0, 0, 0], mg[2][1, 0, 0], mg[2][2, 0, 0])
    #     fc[*mg] = forces
    #     return System(
    #         element=self.element,
    #         cell=self.cell,
    #         n_repeats=self.n_repeats,
    #         fc=fc.reshape(*self._fc.shape),
    #     )

    # def with_next_nearest_neighbor_forces(
    #     self,
    #     forces: NeighbourhoodForces,
    # ) -> System:
    #     """Return a new System with next nearest neighbor spring constant."""
    #     fc = self._fc.copy().reshape(*self.n_repeats, 3, 3)
    #     forces = _neighbourhood_forces_as_full(forces)
    #     # TODO: get nnn points...
    #     fc[[0, 2, -2], [0, 2, -2], [0, 2, -2], :, :] = forces
    #     return System(
    #         element=self.element,
    #         cell=self.cell,
    #         n_repeats=self.n_repeats,
    #         fc=fc.reshape(*self._fc.shape),
    #     )


# type NeighbourhoodForces = (
#     float | tuple[float, float, float] | np.ndarray[Any, np.dtype[np.floating]]
# )


# def _neighbourhood_forces_as_full(
#     spring_constant: NeighbourhoodForces,
# ) -> np.ndarray[Any, np.dtype[np.floating]]:
#     """Convert a single spring constant to a full tuple."""
#     if isinstance(spring_constant, (float, int)):
#         return _neighbourhood_forces_as_full((spring_constant,) * 3)  # type: ignore[return-value]
#     if isinstance(spring_constant, tuple):
#         kx, ky, kz = spring_constant
#         out = np.zeros((3, 3, 3, 3, 3), dtype=float)

#         out[0, 0, 0, 0, 0] += 2 * kx
#         out[1, 0, 0, 0, 0] += -kx
#         out[-1, 0, 0, 0, 0] += -kx

#         out[0, 0, 0, 1, 1] += 2 * ky
#         out[0, 1, 0, 1, 1] += -ky
#         out[0, -1, 0, 1, 1] += -ky
#         # Z direction neighbors
#         out[0, 0, 0, 2, 2] += 2 * kz
#         out[0, 0, 1, 2, 2] += -kz
#         out[0, 0, -1, 2, 2] += -kz
#         return out
#     return spring_constant

# def _build_force_constant_matrix(
#     system: System,
# ) -> np.ndarray[Any, np.dtype[np.floating]]:
#     return np.tile(system.fc, np.prod(system.n_repeats).item()).reshape(
#         (np.prod(system.n_repeats), *system.fc.shape)
#     )
#     # assert system.n_repeats[1:] == (1, 1), "Only 1D chains are supported."  # noqa: ERA001
#     n_x, n_y, n_z = system.n_repeats
#     kx, ky, kz = system.spring_constant

#     fc = np.zeros((n_x, n_y, n_x, n_y, 3, 3), dtype=float)

#     for ix in range(n_x):
#         for iy in range(n_y):
#             for iz in range(n_z):
#                 # X direction neighbors
#                 fc[ix, iy, iz, ix, iy, 0, 0] += 2 * kx
#                 fc[ix, iy, iz, (ix + 1) % n_x, iy, 0, 0] += -kx
#                 fc[ix, iy, iz, (ix - 1) % n_x, iy, 0, 0] += -kx
#                 # Y direction neighbors
#                 fc[ix, iy, iz, ix, iy, 1, 1] += 2 * ky
#                 fc[ix, iy, iz, ix, (iy + 1) % n_y, 1, 1] += -ky
#                 fc[ix, iy, iz, ix, (iy - 1) % n_y, 1, 1] += -ky
#                 # Z direction neighbors
#                 fc[ix, iy, iz, ix, iy, 2, 2] += 2 * kz
#                 fc[ix, iy, iz, ix, (iy + 1) % n_y, 2, 2] += -kz
#                 fc[ix, iy, iz, ix, (iy - 1) % n_y, 2, 2] += -kz
#     return fc.reshape((n_x * n_y, n_x * n_y, 3, 3))
# def build_force_constants_2d(system: System) -> np.ndarray:
#     """
#     Build the force constant matrix for a 2D lattice system including nearest and next nearest neighbor interactions using the atomic positions from a PhononSystem2DResult object and bond pairs found in find_lattice_bond_pairs.

#     Parameters
#     ----------
#     system : Lattice2DSystem
#         The 2D lattice system for which to build the force constant matrix.
#     result : PhononSystem2DResult
#         The result object containing cell, phonon, and positions.

#     Returns
#     -------
#     np.ndarray
#         The force constant matrix of shape (num_atoms, num_atoms, 3, 3).
#     """
#     num_atoms = system.n_repeatsa * system.n_repeatsb
#     positions = result.get_positions()
#     fc = np.zeros((num_atoms, num_atoms, 3, 3), dtype=float)
#     a_vec = np.array(system.lattice_vector_a)
#     b_vec = np.array(system.lattice_vector_b)

#     nn_pairs, nnn_pairs = find_lattice_bond_pairs(positions, a_vec, b_vec)

#     for i, j in nn_pairs:  # Nearest neighbor bonds
#         displacement_vector = positions[j] - positions[i]
#         direction = displacement_vector / np.linalg.norm(displacement_vector)
#         for d1 in range(3):
#             for d2 in range(3):
#                 fc[i, j, d1, d2] += -system.k_nn * direction[d1] * direction[d2]
#                 fc[i, i, d1, d2] += system.k_nn * direction[d1] * direction[d2]

#     for i, j in nnn_pairs:  # Next-nearest neighbor bonds
#         displacement_vector = positions[j] - positions[i]
#         direction = displacement_vector / np.linalg.norm(displacement_vector)
#         for d1 in range(3):
#             for d2 in range(3):
#                 fc[i, j, d1, d2] += -system.k_nnn * direction[d1] * direction[d2]
#                 fc[i, i, d1, d2] += system.k_nnn * direction[d1] * direction[d2]
#     return fc

# def build_force_constants_2d(system: SquareLattice2DSystem) -> np.ndarray:
#     """
#     Build the force constant matrix for a 2D square lattice system including nearest and next nearest neighbor interactions.

#     Parameters
#     ----------
#     system : SquareLattice2DSystem
#         The 2D square lattice system for which to build the force constant matrix.

#     Returns
#     -------
#     np.ndarray
#         The force constant matrix of shape (num_atoms, num_atoms, 3, 3).
#     """
#     nx = system.n_repeatsx
#     ny = system.n_repeatsy
#     num_atoms = nx * ny
#     fc = np.zeros((num_atoms, num_atoms, 3, 3), dtype=float)
#     cell = PhonopyAtoms(
#         symbols=[system.element],
#         cell=[
#             [system.lattice_constantx, 0, 0],
#             [0, system.lattice_constanty, 0],
#             [0, 0, 1],
#         ],
#         scaled_positions=[[0, 0, 0]],
#     )
#     supercell_matrix = [[nx, 0, 0], [0, ny, 0], [0, 0, 1]]
#     phonon = Phonopy(unitcell=cell, supercell_matrix=supercell_matrix)
#     positions = phonon.supercell.get_positions()

#     for i in range(num_atoms):
#         for j in range(num_atoms):
#             if i == j:
#                 continue
#             vec: np.ndarray = positions[j] - positions[i]
#             vec -= np.round(
#                 vec
#                 / np.array(
#                     [system.lattice_constantx * nx, system.lattice_constanty * ny, 1]
#                 )  # Periodic boundary conditions
#             ) * np.array(
#                 [system.lattice_constantx * nx, system.lattice_constanty * ny, 1]
#             )
#             dist = np.linalg.norm(vec[:2])
#             a = max(system.lattice_constantx, system.lattice_constanty)
#             if np.isclose(dist, a, atol=0.05):  # Neartest-neighbor
#                 direction = vec / np.linalg.norm(vec)
#                 for d1 in range(3):
#                     for d2 in range(3):
#                         fc[i, j, d1, d2] += -system.k_nn * direction[d1] * direction[d2]
#                         fc[i, i, d1, d2] += system.k_nn * direction[d1] * direction[d2]
#             elif np.isclose(dist, np.sqrt(2) * a, atol=0.05):  # Next-nearest-neighbor
#                 direction = vec / np.linalg.norm(vec)
#                 for d1 in range(3):
#                     for d2 in range(3):
#                         fc[i, j, d1, d2] += (
#                             -system.k_nnn * direction[d1] * direction[d2]
#                         )
#                         fc[i, i, d1, d2] += system.k_nnn * direction[d1] * direction[d2]
#     return fc
