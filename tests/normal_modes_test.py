from __future__ import annotations

import numpy as np
from phonopy.physical_units import get_physical_units

from phonon_simulation import normal_modes

EV = get_physical_units().EV
Angstrom = get_physical_units().Angstrom
AMU = get_physical_units().AMU
VASP_TO_OMEGA = np.sqrt(EV / AMU) / Angstrom


def test_dispersion_matches_analytical() -> None:
    k = 3.0
    system = normal_modes.System(
        element="Si",
        cell=np.diag([7.0, 1.0, 1.0]),
        n_repeats=(51, 1, 1),
        spring_constant=(k, 0.0, 0.0),
    )
    result = normal_modes.calculate_normal_modes(system)

    # Get the numerical omega values for the second branch (acoustic)
    omega_numerical = result.omega[:, 2]
    omega_analytical = (
        2
        * np.sqrt(k / result.system.mass)
        * np.abs(np.sin(np.pi * result.q_vals))
        * VASP_TO_OMEGA
    )

    np.testing.assert_allclose(
        omega_numerical,
        omega_analytical,
        rtol=1e-8,
        atol=1e-16,
        err_msg="Numerical dispersion does not match analytical result within tolerance",
    )


def test_normal_modes_result_shape() -> None:
    system = normal_modes.System(
        element="Si",
        cell=np.diag([1, 1.0, 1.0]),
        n_repeats=(8, 1, 1),
        spring_constant=(1.0, 0.0, 0.0),
    )
    result = normal_modes.calculate_normal_modes(system)

    assert result.system == system
    assert result.omega.shape[0] == result.system.n_repeats[0], (
        "Omega should match number of q-points"
    )
    assert result.q_vals.shape[0] == result.system.n_repeats[0], (
        "Q-vals should match number of q-points"
    )
    assert result.modes.shape[0] == result.system.n_repeats[0], (
        "Modes should match number of q-points"
    )


def test_omega_nonnegative() -> None:
    system = normal_modes.System(
        element="Si",
        cell=np.diag([1.0, 1.0, 1.0]),
        n_repeats=(100, 1, 1),
        spring_constant=(1.0, 0.0, 0.0),
    )
    result = normal_modes.calculate_normal_modes(system)
    np.testing.assert_array_compare(np.greater_equal, result.omega, 0)


def test_acoustic_zero_at_q0() -> None:
    system = normal_modes.System(
        element="Si",
        cell=np.diag([1.0, 1.0, 1.0]),
        n_repeats=(101, 1, 1),
        spring_constant=(1.0, 0.0, 0.0),
    )
    result = normal_modes.calculate_normal_modes(system)

    zero_q_index = np.argmin(np.abs(result.q_vals))
    omega_acoustic = result.omega[zero_q_index, 2]
    np.testing.assert_allclose(omega_acoustic, 0, atol=1e-8)
