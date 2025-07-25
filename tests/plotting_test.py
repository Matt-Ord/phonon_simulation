from __future__ import annotations

import numpy as np

from phonon_simulation.calculating import (
    SquareLattice2DSystem,
    calculate_2d_square_lattice_bonds,
    calculate_2d_square_modes,
)
from phonon_simulation.plotting import (
    plot_2d_square_bz_path,
    plot_2d_square_dispersion,
    plot_2d_square_lattice,
)


def test_plot_2d_square_dispersion_runs() -> None:
    system = SquareLattice2DSystem(
        element="C",
        lattice_constantx=1.0,
        lattice_constanty=1.0,
        n_repeatsx=4,
        n_repeatsy=4,
        k_nn=1.0,
        k_nnn=0.5,
    )
    mesh_dict, phonon = calculate_2d_square_modes(system)
    path = np.array([[0, 0, 0], [0.5, 0, 0], [0.5, 0.5, 0], [0, 0, 0]])
    labels = [r"$\Gamma$", "X", "M", r"$\Gamma$"]
    fig, ax = plot_2d_square_dispersion(mesh_dict, phonon, system, path, labels)
    assert fig is not None
    assert ax is not None


def test_plot_2d_square_bz_path_runs() -> None:
    path = np.array([[0, 0, 0], [0.5, 0, 0], [0.5, 0.5, 0], [0, 0, 0]])
    labels = [r"$\Gamma$", "X", "M", r"$\Gamma$"]
    fig, ax = plot_2d_square_bz_path(path, labels)
    assert fig is not None
    assert ax is not None


def test_plot_2d_square_lattice_runs() -> None:
    system = SquareLattice2DSystem(
        element="C",
        lattice_constantx=1.0,
        lattice_constanty=1.0,
        n_repeatsx=3,
        n_repeatsy=3,
        k_nn=1.0,
        k_nnn=0.5,
    )
    _, phonon = calculate_2d_square_modes(system)
    bonds = calculate_2d_square_lattice_bonds(phonon, system)
    fig, ax = plot_2d_square_lattice(phonon, system, bonds)
    assert fig is not None
    assert ax is not None
