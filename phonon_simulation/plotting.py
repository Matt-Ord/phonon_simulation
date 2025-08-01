from __future__ import annotations

from typing import TYPE_CHECKING, Any

import matplotlib.pyplot as plt
import numpy as np

from phonon_simulation.calculating import find_lattice_bond_pairs

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure
    from phonopy.api_phonopy import Phonopy

    from phonon_simulation.calculating import (
        DispersionPath,
        Lattice2DSystem,
        PhononSystem2DResult,
    )


def plot_2d_dispersion(
    phonon: Phonopy,
    system: Lattice2DSystem,
    path: np.ndarray,
    labels: list[str],
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

    def interpolate_path(
        path: np.ndarray, n_points: int = 100
    ) -> np.ndarray[Any, np.dtype[np.floating]]:
        points = []
        for i in range(len(path) - 1):
            seg: np.ndarray = np.linspace(
                path[i], path[i + 1], n_points, endpoint=False
            )
            points.append(seg)
        points.append(path[-1][None, :])
        return np.vstack(points)

    q_path = interpolate_path(path, n_points=100)
    phonon.run_band_structure([q_path], with_eigenvectors=True)
    bands: dict[str, np.ndarray] = phonon.get_band_structure_dict()
    distances = bands["distances"][0]
    band_frequencies = bands["frequencies"][0]

    fig, ax = plt.subplots(figsize=(8, 6))
    num_bands = band_frequencies.shape[1]
    cmap = plt.get_cmap("tab10")
    for i in range(num_bands):
        color = cmap(i % cmap.N)
        ax.plot(distances, band_frequencies[:, i], color=color, linewidth=3)

    tick_locations = [distances[0]]
    segment_length = len(distances) // (len(path) - 1)
    tick_locations.extend(
        [distances[i * segment_length] for i in range(1, len(path) - 1)]
    )
    tick_locations.append(distances[-1])
    ax.set_xticks(tick_locations)
    ax.set_xticklabels(labels)
    ax.set_xlim(distances[0], distances[-1])
    ax.set_ylabel("Frequency (THz)")
    ax.set_title(
        f"Phonon Dispersion of  Lattice using up to next nearest neighbours: \n {system.n_repeatsa}x{system.n_repeatsb} supercell made of {system.element} atoms"
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
    path: DispersionPath, system: Lattice2DSystem, axis: tuple[int, int] = (0, 1)
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
