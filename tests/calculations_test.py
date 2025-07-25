from __future__ import annotations

import numpy as np

from phonon_simulation.calculating import (
    SquareLattice2DSystem,
    build_force_constants_2d,
    calculate_2d_square_lattice_bonds,
    calculate_2d_square_modes,
)


def test_force_constant_matrix_shape() -> (
    None
):  # tests that the force constant matrix has the correct shape
    system = SquareLattice2DSystem(
        element="C",
        lattice_constantx=1.0,
        lattice_constanty=1.0,
        n_repeatsx=7,
        n_repeatsy=5,
        k_nn=1.0,
        k_nnn=0.5,
    )
    fc = build_force_constants_2d(system)
    assert fc.shape == (35, 35, 3, 3), "Shape should be (num_atoms, num_atoms, 3, 3)"


def test_zero_at_gamma() -> None:
    system = SquareLattice2DSystem(
        element="C",
        lattice_constantx=1.0,
        lattice_constanty=1.0,
        n_repeatsx=4,
        n_repeatsy=4,
        k_nn=1.0,
        k_nnn=0.0,
    )
    mesh_dict, _ = calculate_2d_square_modes(system)
    gamma_idx = np.where(np.all(np.isclose(mesh_dict["qpoints"], [0, 0, 0]), axis=1))[0]
    assert np.allclose(mesh_dict["frequencies"][gamma_idx][:, :2], 0, atol=1e-6)


def test_bonds_types() -> None:
    system = SquareLattice2DSystem(
        element="C",
        lattice_constantx=1.0,
        lattice_constanty=1.0,
        n_repeatsx=2,
        n_repeatsy=2,
        k_nn=1.0,
        k_nnn=0.5,
    )
    _, phonon = calculate_2d_square_modes(system)
    bonds = calculate_2d_square_lattice_bonds(phonon, system)
    bond_types = {b[2] for b in bonds}
    assert "nn" in bond_types
    assert "nnn" in bond_types


def test_lattice_size() -> None:
    system = SquareLattice2DSystem(
        element="C",
        lattice_constantx=1.0,
        lattice_constanty=1.0,
        n_repeatsx=2,
        n_repeatsy=2,
        k_nn=1.0,
        k_nnn=0.5,
    )
    _, phonon = calculate_2d_square_modes(system)
    positions = phonon.supercell.positions
    num_atoms = positions.shape[0]
    expected_atoms = system.n_repeatsx * system.n_repeatsy
    assert num_atoms == expected_atoms, (
        f"Expected there to be {expected_atoms} atoms, but got {num_atoms} atoms."
    )
