from __future__ import annotations

from typing import TYPE_CHECKING, Any

import matplotlib.pyplot as plt
import numpy as np

from phonon_simulation.calculating import find_lattice_bond_pairs

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

    from phonon_simulation.normal_modes import (
        DispersionPath,
        NormalModeDispersionResult,
        NormalModeGrid,
    )
    from phonon_simulation.system import System


def plot_dispersion_1d(modes: NormalModeGrid) -> tuple[Figure, Axes]:
    """Plot the phonon dispersion relation for a 1D chain on a graph, including analytical curve."""
    fig, ax = plt.subplots(figsize=(8, 6))

    ax.plot(modes.q_vals, modes.omega, "o", label="Numerical")

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


def plot_2d_square_lattice(
    phonon, system: SquareLattice2DSystem
) -> tuple[Figure, Axes]:
    """
    Plot the atomic positions and bonds of a 2D square lattice supercell, highlighting nearest and next nearest neighbor interactions.

    Parameters
    ----------
    phonon : Phonopy
        The Phonopy object containing the supercell information.
    system : SquareLattice2DSystem
        The 2D square lattice system being visualized.

    Returns
    -------
    tuple[Figure, Axes]
        The matplotlib Figure and Axes objects for the plot.
    """
    supercell: PhonopyAtoms = phonon.supercell
    supercell_positions: np.ndarray[Any, np.dtype[np.floating]] = supercell.positions
    fig_supercell, ax_supercell = plt.subplots()
    ax_supercell.scatter(
        supercell_positions[:, 0], supercell_positions[:, 1], s=50, c="black"
    )
    ax_supercell.set_xlabel("x (Å)")
    ax_supercell.set_ylabel("y (Å)")
    ax_supercell.set_title(
        f"{system.n_repeatsx}x{system.n_repeatsy} supercell of {system.element} atoms with nearest \n and next nearest neighbour interactions"
    )
    ax_supercell.set_aspect("equal")

    a_x: float = system.lattice_constantx
    a_y: float = system.lattice_constanty
    a = {"a_x": a_x, "a_y": a_y}
    nn_bond = min(a_x, a_y)
    nnn_bond = np.sqrt(a_x**2 + a_y**2)
    a.get(max(a))
    nn_label_added = False
    nnn_label_added = False

    for i in range(supercell_positions.shape[0]):
        for j in range(i + 1, supercell_positions.shape[0]):
            dist = np.linalg.norm(supercell_positions[i] - supercell_positions[j])
            if 0.9 * nn_bond < dist < 1.1 * nn_bond:
                ax_supercell.plot(
                    [supercell_positions[i, 0], supercell_positions[j, 0]],
                    [supercell_positions[i, 1], supercell_positions[j, 1]],
                    color="orange",
                    linewidth=1.5,
                    alpha=0.7,
                    label="Nearest neighbour" if not nn_label_added else None,
                )
                nn_label_added = True
            elif 0.9 * nnn_bond < dist < 1.1 * nnn_bond:
                ax_supercell.plot(
                    [supercell_positions[i, 0], supercell_positions[j, 0]],
                    [supercell_positions[i, 1], supercell_positions[j, 1]],
                    color="blue",
                    linestyle="--",
                    linewidth=1.5,
                    alpha=0.7,
                    label="Next nearest neighbour" if not nnn_label_added else None,
                )
                nnn_label_added = True

    _handles, labels = ax_supercell.get_legend_handles_labels()
    if any(labels):
        ax_supercell.legend(loc="lower center", bbox_to_anchor=(0.5, -0.3), ncol=2)
        plt.subplots_adjust(bottom=0.3)  # Make space for legend

    return fig_supercell, ax_supercell


def plot_2d_square_bz_path(path: np.ndarray, labels: list[str]) -> tuple[Figure, Axes]:
    """
    Plot the first Brillouin zone (BZ) of a 2D square lattice and the high-symmetry path used for phonon dispersion calculations.

    Parameters
    ----------
    system : SquareLattice2DSystem
        The 2D square lattice system for which the BZ is plotted.
    path : np.ndarray
        Array of q-points defining the path in the Brillouin zone.
    labels : list of str
        List of labels for the high-symmetry points along the path.

    Returns
    -------
    tuple[Figure, Axes]
        The matplotlib Figure and Axes objects for the plot.
    """
    fig, ax = plt.subplots(figsize=(4, 4))
    # Square BZ from -0.5 to 0.5 in reduced coordinates
    ax.plot([-0.5, 0.5, 0.5, -0.5, -0.5], [-0.5, -0.5, 0.5, 0.5, -0.5], "k-", lw=1)
    ax.plot(path[:, 0], path[:, 1], "k-", lw=2)
    arrowprops = {
        "arrowstyle": "->",
        "color": "k",
        "lw": 1.5,
        "shrinkA": 0,
        "shrinkB": 0,
    }
    for i in range(len(path) - 1):
        mid = (path[i] + path[i + 1]) / 2
        ax.annotate("", xy=mid[:2], xytext=path[i][:2], arrowprops=arrowprops)
    for i, pt in enumerate(path):
        x, y = pt[0], pt[1]
        ax.plot(x, y, "ro")
        if labels[i] == r"$\Gamma$" and x == 0 and y == 0:
            ax.text(x - 0.05, y, labels[i], fontsize=10, va="center", ha="right")
        else:
            ax.text(x + 0.05, y, labels[i], fontsize=10, va="center", ha="center")
    ax.set_xticks([-0.5, 0, 0.5])
    ax.set_xticklabels([r"$-\pi/a$", "0", r"$\pi/a$"])
    ax.set_yticks([-0.5, 0, 0.5])
    ax.set_yticklabels([r"$-\pi/a$", "0", r"$\pi/a$"])
    ax.set_xlim(-0.6, 0.6)
    ax.set_ylim(-0.6, 0.6)
    ax.set_title("FBZ & Path", fontsize=12)
    ax.set_aspect("equal")
    fig.tight_layout()
    return fig, ax


def plot_dispersion_result(
    result: NormalModeDispersionResult,
) -> tuple[Figure, Axes]:
    """
    Plot the phonon dispersion relation for a 2D  lattice system along a specified path in the First Brillouin zone.

    Parameters
    ----------
    phonon : Phonopy
        The Phonopy object used for the calculation.
    system : Lattice2DSystem
        The 2D  lattice system for which the dispersion is plotted.
    path : np.ndarray
        Array of q-points defining the path in the Brillouin zone.
    labels : list of str
        List of labels for the high-symmetry points along the path.

    Returns
    -------
    tuple[Figure, Axes]
        The matplotlib Figure and Axes objects for the plot.
    """
    distances = result.distances
    band_frequencies = result.band_frequencies

    fig, ax = plt.subplots(figsize=(8, 6))
    num_bands = band_frequencies.shape[1]
    cmap = plt.get_cmap("tab10")
    for i in range(num_bands):
        color = cmap(i % cmap.N)
        ax.plot(distances, band_frequencies[:, i], color=color, linewidth=3)

    tick_locations = [distances[0]]
    segment_length = len(distances) // (len(result.path.labels) - 1)
    tick_locations.extend(
        [distances[i * segment_length] for i in range(1, len(result.path.labels) - 1)]
    )
    tick_locations.append(distances[-1])
    ax.set_xticks(tick_locations)
    ax.set_xticklabels(result.path.labels)
    ax.set_xlim(distances[0], distances[-1])
    ax.set_ylabel("Frequency (THz)")
    ax.set_title(
        f"Phonon Dispersion of Lattice using up to next nearest neighbours: \n"
        f"{result.system.n_repeats[0]}x{result.system.n_repeats[1]} supercell made of {result.system.element} atoms"
    )
    ax.grid(visible=True)
    fig.tight_layout()
    return fig, ax


def _plot_dispersion_path_lines(
    path: DispersionPath,
    system: Lattice2DSystem,
    axis: tuple[int, int] = (0, 1),
    *,
    ax: Axes,
) -> None:
    cart_points = reduced_to_cartesian(path.points, system)
    (line,) = ax.plot(
        cart_points[:, axis[0]],
        cart_points[:, axis[1]],
        "k-",
        lw=2,
    )
    line.set_marker("o")
    line.set_markerfacecolor("r")
    line.set_markeredgecolor("r")
    """
    arrowprops = {
        "arrowstyle": "->",
        "color": "k",
        "lw": 1.5,
        "shrinkA": 0,
        "shrinkB": 0,
    }
    # Add arrows between points
    midpoints = (path.points[:-1] + path.points[1:]) / 2
    startpoints = (path.points[:-1] + midpoints) / 2
    for start, end in zip(startpoints, midpoints, strict=False):
        ax.annotate(
            "",
            xy=end[list(axis)],
            xytext=start[list(axis)],
            arrowprops=arrowprops,
        )"""


def _plot_dispersion_path_labels(
    path: DispersionPath,
    system: Lattice2DSystem,
    axis: tuple[int, int] = (0, 1),
    *,
    ax: Axes,
) -> None:
    cart_points = reduced_to_cartesian(path.points, system)
    x_center = 0.0
    a_vec = np.array(system.lattice_vector_a[:2])
    b_vec = np.array(system.lattice_vector_b[:2])
    area = a_vec[0] * b_vec[1] - a_vec[1] * b_vec[0]
    b1 = 2 * np.pi * np.array([b_vec[1], -b_vec[0]]) / area
    qx_lim = 0.5 * np.linalg.norm(b1)
    offset = 0.05 * qx_lim

    for label, pt in zip(path.labels, cart_points, strict=True):
        x, y = pt[list(axis)]
        if x > x_center:
            ax.text(x + offset, y, label, fontsize=10, va="center", ha="left")
        elif x < x_center:
            ax.text(x - offset, y, label, fontsize=10, va="center", ha="right")
        elif x == x_center:
            if y > 0:
                ax.text(x, y + offset, label, fontsize=10, va="bottom", ha="center")
            else:
                ax.text(x, y - offset, label, fontsize=10, va="top", ha="center")
        else:
            ax.text(x, y, label, fontsize=10, va="center", ha="center")


def plot_dispersion_path(
    path: DispersionPath, system: System, axis: tuple[int, int] = (0, 1)
) -> tuple[Figure, Axes]:
    """Plot the first Brillouin zone (BZ) of a 2D lattice and the path used for phonon dispersion calculations. The plot size is set to 2pi/a by 2pi/b where a and b are the magnitudes of the lattice vectors."""
    a_vec = np.array(system.lattice_vector_a[:2])
    b_vec = np.array(system.lattice_vector_b[:2])
    area = a_vec[0] * b_vec[1] - a_vec[1] * b_vec[0]
    b1 = 2 * np.pi * np.array([b_vec[1], -b_vec[0]]) / area
    b2 = 2 * np.pi * np.array([-a_vec[1], a_vec[0]]) / area
    qx_lim = 0.5 * np.linalg.norm(b1)
    qy_lim = 0.5 * np.linalg.norm(b2)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(
        [-qx_lim, qx_lim, qx_lim, -qx_lim, -qx_lim],
        [-qy_lim, -qy_lim, qy_lim, qy_lim, -qy_lim],
        "k-",
        lw=1,
    )
    _plot_dispersion_path_lines(path, system, axis=axis, ax=ax)
    _plot_dispersion_path_labels(path, system, axis=axis, ax=ax)
    ax.set_xticks(np.array([-qx_lim, 0, qx_lim]))
    ax.set_xticklabels([r"$-\pi/a$", "0", r"$\pi/a$"])
    ax.set_yticks(np.array([-qy_lim, 0, qy_lim]))
    ax.set_yticklabels([r"$-\pi/b$", "0", r"$\pi/b$"])
    ax.set_xlim(float(-1.1 * qx_lim), float(1.1 * qx_lim))
    ax.set_ylim(float(-1.1 * qy_lim), float(1.1 * qy_lim))
    ax.set_title("FBZ & Path", fontsize=12)
    ax.set_aspect("equal")
    fig.tight_layout()
    return fig, ax


def plot_2d_lattice(
    result: PhononSystem2DResult,
    system: Lattice2DSystem,
) -> tuple[Figure, Axes]:
    """
    Plot the 2D lattice structure inputted with nearest and next nearest neighbour bonds shown.

    Parameters
    ----------
    result : PhononSystem2DResult
        The result object containing atomic positions.
    system : Lattice2DSystem
        The 2D lattice system being plotted.

    Returns
    -------
    tuple[Figure, Axes]
        The matplotlib Figure and Axes objects for the plot.
    """
    positions = result.get_positions()
    fig, ax = plt.subplots()
    ax.scatter(positions[:, 0], positions[:, 1], s=50, c="black")
    ax.set_xlabel("x (Å)")
    ax.set_ylabel("y (Å)")
    ax.set_title(
        f"{system.n_repeatsa}x{system.n_repeatsb} supercell of {system.element} atoms with nearest \n and next nearest neighbour interactions"
    )
    ax.set_aspect("equal")
    a_vec = np.array(system.lattice_vector_a)
    b_vec = np.array(system.lattice_vector_b)
    nn_pairs, nnn_pairs = find_lattice_bond_pairs(positions, a_vec, b_vec)
    nn_label_added = False  # To make sure each label is only plotted once in the legend
    nnn_label_added = False

    for i, j in nn_pairs:
        ax.plot(
            [positions[i, 0], positions[j, 0]],
            [positions[i, 1], positions[j, 1]],
            color="orange",
            linewidth=1.5,
            alpha=0.7,
            label="Nearest neighbour" if not nn_label_added else None,
        )
        nn_label_added = True
    for i, j in nnn_pairs:
        ax.plot(
            [positions[i, 0], positions[j, 0]],
            [positions[i, 1], positions[j, 1]],
            color="blue",
            linestyle="--",
            linewidth=1.5,
            alpha=0.7,
            label="Next nearest neighbour" if not nnn_label_added else None,
        )
        nnn_label_added = True
    margin = 0.2 * max(np.ptp(positions[:, 0]), np.ptp(positions[:, 1]))
    x_min, x_max = positions[:, 0].min() - margin, positions[:, 0].max() + margin
    y_min, y_max = positions[:, 1].min() - margin, positions[:, 1].max() + margin
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    labels = ax.get_legend_handles_labels()
    if any(labels):
        ax.legend(loc="lower center", bbox_to_anchor=(0.5, -1), ncol=2)
        plt.subplots_adjust(bottom=0.3)
    return fig, ax


def reduced_to_cartesian(path_points, system: Lattice2DSystem) -> np.ndarray:
    """Convert reduced coordinates to cartesian reciprocal coordinates for plotting."""  # This function may not be needed but works for now as the plot of the path is low priority to have working shape and labels for non square or rectangular lattices.
    a_vec = np.array(system.lattice_vector_a[:2])
    b_vec = np.array(system.lattice_vector_b[:2])
    area = a_vec[0] * b_vec[1] - a_vec[1] * b_vec[0]
    b1 = 2 * np.pi * np.array([b_vec[1], -b_vec[0]]) / area
    b2 = 2 * np.pi * np.array([-a_vec[1], a_vec[0]]) / area
    return np.array([p[0] * b1 + p[1] * b2 for p in path_points])
