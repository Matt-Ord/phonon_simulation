from __future__ import annotations

import numpy as np


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
