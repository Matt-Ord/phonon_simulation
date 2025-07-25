from __future__ import annotations

from typing import TYPE_CHECKING, Any

import matplotlib.pyplot as plt
import numpy as np

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure
    from phonopy.api_phonopy import Phonopy

    from phonon_simulation.calculating import Lattice2DSystem, PhononSystem2DResult


def plot_2d__dispersion(
    mesh_dict: dict[str, np.ndarray],
    phonon: Phonopy,
    system: Lattice2DSystem,
    path: np.ndarray,
    labels: list[str],
) -> tuple[Figure, Axes]:
    """
    Plot the phonon dispersion relation for a 2D  lattice system along a specified path in the Brillouin zone.

    Parameters
    ----------
    mesh_dict : dict
        Dictionary containing mesh information with phonon properties.
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
    for i in range(band_frequencies.shape[1]):
        ax.plot(distances, band_frequencies[:, i], color="b", linewidth=3)

    qpoints = mesh_dict["qpoints"]
    frequencies = mesh_dict["frequencies"]
    tolerance = 5e-3
    for idx, q in enumerate(q_path):
        matches = np.where(np.all(np.abs(qpoints - q) < tolerance, axis=1))[0]
        if matches.size > 0:
            mesh_freqs = frequencies[matches[0]]
            ax.scatter(
                [distances[idx]] * mesh_freqs.shape[0],
                mesh_freqs,
                color="r",
                s=5,
                zorder=3,
                label="Mesh modes" if idx == 0 else None,
            )

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
    _handles, labels_ = ax.get_legend_handles_labels()
    if "Mesh modes" in labels_ or "All mesh modes" in labels_:
        ax.legend(loc="lower right", bbox_to_anchor=(0.5, -0.25), ncol=2)
    fig.tight_layout()
    return fig, ax


def plot_2d__bz_path(path: np.ndarray, labels: list[str]) -> tuple[Figure, Axes]:
    """
    Plot the first Brillouin zone (BZ) of a 2D  lattice and the high-symmetry path used for phonon dispersion calculations.

    Parameters
    ----------
    system : Lattice2DSystem
        The 2D  lattice system for which the BZ is plotted.
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
    ax.plot(
        [-0.5, 0.5, 0.5, -0.5, -0.5], [-0.5, -0.5, 0.5, 0.5, -0.5], "k-", lw=1
    )  # BZ from -0.5 to 0.5 in reduced coordinates
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


def plot_2d__lattice(
    result: PhononSystem2DResult,
    system: Lattice2DSystem,
    bonds: list[tuple[int, int, str]] | None = None,
) -> tuple[Figure, Axes]:
    """
    Plot the atomic positions and bonds of a 2D  lattice supercell.

    If bonds is None, will not plot bonds.
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

    # Plot bonds if provided
    nn_label_added = False
    nnn_label_added = False
    if bonds is not None:
        for i, j, bond_type in bonds:
            if bond_type == "nn":
                ax.plot(
                    [positions[i, 0], positions[j, 0]],
                    [positions[i, 1], positions[j, 1]],
                    color="orange",
                    linewidth=1.5,
                    alpha=0.7,
                    label="Nearest neighbour" if not nn_label_added else None,
                )
                nn_label_added = True
            elif bond_type == "nnn":
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

    _handles, labels = ax.get_legend_handles_labels()
    if any(labels):
        ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.3), ncol=2)
        plt.subplots_adjust(bottom=0.3)
    return fig, ax
