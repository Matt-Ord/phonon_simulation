from __future__ import annotations

import numpy as np
from phonopy.physical_units import get_physical_units

from phonon_simulation import normal_modes

EV = get_physical_units().EV
Angstrom = get_physical_units().Angstrom
AMU = get_physical_units().AMU
VaspToOmega = np.sqrt(EV / AMU) / Angstrom
mesh_size = 101


def test_import() -> None:
    try:
        import phonon_simulation  # noqa: PLC0415
    except ImportError:
        phonon_simulation = None

    assert phonon_simulation is not None, "phonon_simulation module should not be None"


def test_import_normal_modes() -> None:
    """Test that the normal_modes module can be imported and contains expected attributes."""
    assert hasattr(normal_modes, "calculate_normal_modes")
    assert hasattr(normal_modes, "System")


def test_modes_shape() -> None:
    """Test that the shape of the returned modes matches the expected dimensions."""
    system = normal_modes.System(
        element="Si",
        lattice_constant=(5.43, 1.0, 1.0),
        n_repeats=(10, 1, 1),
        spring_constant=(1.0, 0.0, 0.0),
    )
    result = normal_modes.calculate_normal_modes(system)
    assert result.modes.shape[0] == result.q_vals.shape[0], (
        "Modes should match number of q-points"
    )


def test_dispersion_matches_analytical() -> None:
    """Test that the numerical dispersion matches the analytical result."""
    # The value of n_atoms, k and a can be changed to test different scenarios
    n_atoms = 51
    k = 3.0
    a = 7.0
    system = normal_modes.System(
        element="Si",
        lattice_constant=(a, 1.0, 1.0),
        n_repeats=(n_atoms, 1, 1),
        spring_constant=(k, 0.0, 0.0),
    )
    result = normal_modes.calculate_normal_modes(system)
    q_vals = result.q_vals
    acoustic_index = 2
    omega_dim_mode = 2
    omega_numerical = (
        result.omega[:, acoustic_index]
        if result.omega.ndim == omega_dim_mode
        else result.omega
    )
    omega_analytical = (
        2 * np.sqrt(k / result.mass) * np.abs(np.sin(np.pi * q_vals)) * VaspToOmega
    )
    max_fractional_deviation = 1e-8
    non_zero = 1e-8  # Largest value above zero that we consider non-zero
    mask = (
        np.abs(omega_analytical) > non_zero
    )  # Only divide where analytical is nonzero
    deviation = np.zeros_like(omega_analytical)
    deviation[mask] = np.abs(omega_numerical[mask] - omega_analytical[mask]) / np.abs(
        omega_analytical[mask]
    )
    deviation[~mask] = np.abs(omega_numerical[~mask] - omega_analytical[~mask])
    assert np.all(deviation < max_fractional_deviation), (
        f"Numerical dispersion deviates from analytical by more than {max_fractional_deviation}"
    )


def test_phonopy_supercell_atoms() -> None:
    """Test that Phonopy builds the correct number of atoms in the supercell."""
    system = normal_modes.System(
        element="Si",
        lattice_constant=(1, 1.0, 1.0),
        n_repeats=(8, 1, 1),
        spring_constant=(1.0, 0.0, 0.0),
    )
    result = normal_modes.calculate_normal_modes(system)
    testn = 8  # This is an example number that can be changed
    assert result.system.n_repeats[0] == testn, (
        f"Supercell should have {testn} atoms for n_repeats=({testn},1,1) but has {result.system.n_repeats[0]}"  # The number of atoms in the supercell should match n_repeats[0]
    )


def test_qpoints_shape() -> None:
    """Test that the 1D chain example produces the correct number of q-points."""
    system = normal_modes.System(
        element="Si",
        lattice_constant=(1.0, 1.0, 1.0),
        n_repeats=(100, 1, 1),
        spring_constant=(1.0, 0.0, 0.0),
    )
    result = normal_modes.calculate_normal_modes(system)
    assert result.q_vals.shape[0] == mesh_size, (
        "Should have 101 q-points for mesh (101, 1, 1)"
    )


def test_omega_shape() -> None:
    """Test that the omega output matches the number of q-points."""
    system = normal_modes.System(
        element="Si",
        lattice_constant=(1.0, 1.0, 1.0),
        n_repeats=(100, 1, 1),
        spring_constant=(1.0, 0.0, 0.0),
    )
    result = normal_modes.calculate_normal_modes(system)
    assert result.omega.shape[0] == mesh_size, "Omega should match number of q-points"


def test_omega_nonnegative() -> None:
    """Test that all frequencies are non-negative."""
    system = normal_modes.System(
        element="Si",
        lattice_constant=(1.0, 1.0, 1.0),
        n_repeats=(100, 1, 1),
        spring_constant=(1.0, 0.0, 0.0),
    )
    result = normal_modes.calculate_normal_modes(system)
    assert np.all(result.omega >= 0), "All frequencies should be non-negative"


def test_acoustic_zero_at_q0() -> None:
    """Test that the acoustic branch is zero at q=0."""
    system = normal_modes.System(
        element="Si",
        lattice_constant=(1.0, 1.0, 1.0),
        n_repeats=(100, 1, 1),
        spring_constant=(1.0, 0.0, 0.0),
    )
    result = normal_modes.calculate_normal_modes(system)
    acoustic_index = 2
    omega_dim = 2
    omega_numerical = (
        result.omega[:, acoustic_index]
        if result.omega.ndim == omega_dim
        else result.omega
    )
    zero_q_index = np.argmin(np.abs(result.q_vals))
    assert np.isclose(omega_numerical[zero_q_index], 0, atol=1e-8), (
        "Acoustic mode should be zero at q=0"
    )
