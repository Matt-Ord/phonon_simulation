from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import matplotlib.pyplot as plt
import numpy as np
from phonopy.api_phonopy import Phonopy
from phonopy.physical_units import get_physical_units
from phonopy.structure.atoms import PhonopyAtoms  # type: ignore[import]

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

EV = get_physical_units().EV
Angstrom = get_physical_units().Angstrom
AMU = get_physical_units().AMU
VaspToOmega = np.sqrt(EV / AMU) / Angstrom


@dataclass(kw_only=True, frozen=True)
class System:
    """Represents a lattice system used for phonon calculations."""

    element: str
    lattice_constant: tuple[float, float, float]
    n_repeats: tuple[int, int, int]
    spring_constant: tuple[float, float, float]

    @property
    def mass(self) -> float:
        """Mass of the element in atomic mass units."""
        cell = PhonopyAtoms(
            symbols=[self.element],
            cell=np.diag(self.lattice_constant),
            scaled_positions=[[0, 0, 0]],
        )
        return cell.masses[0]


@dataclass(kw_only=True, frozen=True)
class NormalModeResult:
    """Result of a normal mode calculation for a phonon system."""

    system: System
    omega: np.ndarray[Any, np.dtype[np.floating]]
    """The normal mode frequencies in angular frequency units."""
    modes: np.ndarray[Any, np.dtype[np.floating]]
    """The eigenvectors (normal modes) of the system."""
    q_vals: np.ndarray[Any, np.dtype[np.floating]]
    """The reduced wave vectors for the normal modes."""

    def to_human_readable(self) -> str:
        """Convert the result to a text representation."""
        return (
            f"Calculating normal modes for system: {self.system}\n"
            "Normal mode frequencies (omega):\n"
            f"{np.array2string(self.omega, precision=6, separator=', ')}\n"
            "Wave vectors (q):\n"
            f"{np.array2string(self.q_vals, precision=6, separator=', ')}\n"
            "Normal modes (eigenvectors):\n"
            f"{np.array2string(self.modes, precision=6, separator=', ')}\n"
        )


def _build_force_constant_matrix(
    system: System,
) -> np.ndarray[Any, np.dtype[np.floating]]:
    # assert system.n_repeats[1:] == (1, 1), "Only 1D chains are supported."  # noqa: ERA001
    n_x, n_y = system.n_repeats[0], system.n_repeats[1]
    kx, ky = system.spring_constant[0], system.spring_constant[1]
    n = n_x * n_y
    fc = np.zeros((n, n, 3, 3), dtype=float)

    def idx(ix: int, iy: int) -> int:
        return ix * n_y + iy

    for ix in range(n_x):
        for iy in range(n_y):
            i = idx(ix, iy)
            # X direction neighbors
            jx_p = idx((ix + 1) % n_x, iy)
            jx_m = idx((ix - 1) % n_x, iy)
            fc[i, i, 0, 0] += 2 * kx
            fc[i, jx_p, 0, 0] += -kx
            fc[i, jx_m, 0, 0] += -kx
            # Y direction neighbors
            jy_p = idx(ix, (iy + 1) % n_y)
            jy_m = idx(ix, (iy - 1) % n_y)
            fc[i, i, 1, 1] += 2 * ky
            fc[i, jy_p, 1, 1] += -ky
            fc[i, jy_m, 1, 1] += -ky
    return fc


def calculate_normal_modes(system: System) -> NormalModeResult:
    """
    Calculate and plot the normal modes and phonon dispersion relation for a simple 1D chain system.

    Returns a NormalModeResult containing frequencies, eigenvectors, and reduced wave vectors.
    """
    cell = PhonopyAtoms(
        symbols=[system.element],
        cell=np.diag(system.lattice_constant),
        scaled_positions=[[0, 0, 0]],
    )
    supercell_matrix = np.diag(system.n_repeats)
    phonon = Phonopy(unitcell=cell, supercell_matrix=supercell_matrix)

    phonon.force_constants = _build_force_constant_matrix(system)
    phonon.run_mesh(system.n_repeats, with_eigenvectors=True, is_mesh_symmetry=False)  # type: ignore[arg-type]
    mesh_dict: dict[str, np.ndarray] = phonon.get_mesh_dict()  # type: ignore[return-value]

    sorted_indices = np.argsort(mesh_dict["qpoints"][:, 0])  # cspell: disable-line
    return NormalModeResult(
        system=system,
        omega=mesh_dict["frequencies"][sorted_indices] * 1e12 * 2 * np.pi,
        modes=mesh_dict["eigenvectors"][sorted_indices][..., 0],
        q_vals=mesh_dict["qpoints"][sorted_indices, 0],  # cspell: disable-line
    )


def plot_dispersion(modes: NormalModeResult) -> tuple[Figure, Axes]:
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


@dataclass(kw_only=True, frozen=True)
class SquareLattice2DSystem:
    """Represents a 2D square lattice system for phonon calculations.

    Attributes
    ----------
    element : str
        The chemical symbol of the element.
    lattice_constantx : float
        Lattice constant along the x direction.
    lattice_constanty : float
        Lattice constant along the y direction.
    n_repeatsx : int
        Number of repeats along the x direction.
    n_repeatsy : int
        Number of repeats along the y direction.
    k_nn : float
        Nearest neighbor spring constant.
    k_nnn : float
        Next nearest neighbor spring constant.
    """

    element: str
    lattice_constantx: float
    lattice_constanty: float
    n_repeatsx: int
    n_repeatsy: int
    k_nn: float
    k_nnn: float

    @property
    def mass(self) -> float:
        """Return the mass of the element in atomic mass units."""
        cell = PhonopyAtoms(
            symbols=[self.element],
            cell=[
                [self.lattice_constantx, 0, 0],
                [0, self.lattice_constanty, 0],
                [0, 0, 1],
            ],
            scaled_positions=[[0, 0, 0]],
        )
        return cell.masses[0]


@dataclass(kw_only=True, frozen=True)
class System2D:
    """Represents a 2D square lattice system for phonon calculations.

    Attributes
    ----------
    element : str
        The chemical symbol of the element.
    lattice_constantx : float
        Lattice constant along the x direction.
    lattice_constanty : float
        Lattice constant along the y direction.
    n_repeatsx : int
        Number of repeats along the x direction.
    n_repeatsy : int
        Number of repeats along the y direction.
    k_nn : float
        Nearest neighbor spring constant.
    k_nnn : float
        Next nearest neighbor spring constant.
    """

    element: str
    lattice_constants: np.ndarray[Any, np.dtype[np.floating]]
    n_repeatsx: int
    n_repeatsy: int
    k_nn: float
    k_nnn: float

    @property
    def mass(self) -> float:
        """Return the mass of the element in atomic mass units."""
        cell = PhonopyAtoms(
            symbols=[self.element],
            cell=[
                [self.lattice_constantx, 0, 0],
                [0, self.lattice_constanty, 0],
                [0, 0, 1],
            ],
            scaled_positions=[[0, 0, 0]],
        )
        return cell.masses[0]


class squaresystem2d(System2D):
    """A class to represent a 2D square lattice system for phonon calculations."""

    def __init__(
        self,
        element: str,
        lattice_constants: tuple[float, float, float],
        n_repeatsx: int,
        n_repeatsy: int,
        k_nn: float,
        k_nnn: float,
    ) -> None:
        super().__init__(
            element=element,
            lattice_constants=np.diag(lattice_constants),
            n_repeatsx=n_repeatsx,
            n_repeatsy=n_repeatsy,
            k_nn=k_nn,
            k_nnn=k_nnn,
        )


def build_force_constants_2d(system: SquareLattice2DSystem) -> np.ndarray:
    """
    Build the force constant matrix for a 2D square lattice system including nearest and next nearest neighbor interactions.

    Parameters
    ----------
    system : SquareLattice2DSystem
        The 2D square lattice system for which to build the force constant matrix.

    Returns
    -------
    np.ndarray
        The force constant matrix of shape (num_atoms, num_atoms, 3, 3).
    """
    nx = system.n_repeatsx
    ny = system.n_repeatsy
    num_atoms = nx * ny
    fc = np.zeros((num_atoms, num_atoms, 3, 3), dtype=float)
    cell = PhonopyAtoms(
        symbols=[system.element],
        cell=[
            [system.lattice_constantx, 0, 0],
            [0, system.lattice_constanty, 0],
            [0, 0, 1],
        ],
        scaled_positions=[[0, 0, 0]],
    )
    supercell_matrix = [[nx, 0, 0], [0, ny, 0], [0, 0, 1]]
    phonon = Phonopy(unitcell=cell, supercell_matrix=supercell_matrix)
    positions = phonon.supercell.get_positions()

    for i in range(num_atoms):
        for j in range(num_atoms):
            if i == j:
                continue
            vec: np.ndarray = positions[j] - positions[i]
            vec -= np.round(
                vec
                / np.array(
                    [system.lattice_constantx * nx, system.lattice_constanty * ny, 1]
                )  # Periodic boundary conditions
            ) * np.array(
                [system.lattice_constantx * nx, system.lattice_constanty * ny, 1]
            )
            dist = np.linalg.norm(vec[:2])
            a = max(system.lattice_constantx, system.lattice_constanty)
            if np.isclose(dist, a, atol=0.05):  # Neartest-neighbor
                direction = vec / np.linalg.norm(vec)
                for d1 in range(3):
                    for d2 in range(3):
                        fc[i, j, d1, d2] += -system.k_nn * direction[d1] * direction[d2]
                        fc[i, i, d1, d2] += system.k_nn * direction[d1] * direction[d2]
            elif np.isclose(dist, np.sqrt(2) * a, atol=0.05):  # Next-nearest-neighbor
                direction = vec / np.linalg.norm(vec)
                for d1 in range(3):
                    for d2 in range(3):
                        fc[i, j, d1, d2] += (
                            -system.k_nnn * direction[d1] * direction[d2]
                        )
                        fc[i, i, d1, d2] += system.k_nnn * direction[d1] * direction[d2]
    return fc


def calculate_2d_square_modes(
    system: SquareLattice2DSystem,
) -> tuple[dict[str, np.ndarray], Phonopy]:
    """
    Calculate the phonon modes for a 2D square lattice system.

    Parameters
    ----------
    system : SquareLattice2DSystem
        The 2D square lattice system for which to calculate the phonon modes.

    Returns
    -------
    tuple[dict[str, np.ndarray], Phonopy]
        A tuple containing the mesh dictionary with phonon properties and the Phonopy object.
    """
    cell = PhonopyAtoms(
        symbols=[system.element],
        cell=[
            [system.lattice_constantx, 0, 0],
            [0, system.lattice_constanty, 0],
            [0, 0, 1],
        ],
        scaled_positions=[[0, 0, 0]],
    )
    supercell_matrix = [[system.n_repeatsx, 0, 0], [0, system.n_repeatsy, 0], [0, 0, 1]]
    phonon = Phonopy(unitcell=cell, supercell_matrix=supercell_matrix)
    fc = build_force_constants_2d(system)
    phonon.force_constants = fc
    mesh = (201, 201, 1)
    phonon.run_mesh(mesh, with_eigenvectors=True, is_mesh_symmetry=False)
    mesh_dict: dict[str, np.ndarray] = phonon.get_mesh_dict()
    return mesh_dict, phonon


def plot_2d_square_dispersion(
    mesh_dict,
    phonon,
    system: SquareLattice2DSystem,
    path: np.ndarray,
    labels: list[str],
) -> tuple[Figure, Axes]:
    """
    Plot the phonon dispersion relation for a 2D square lattice system along a specified path in the Brillouin zone.

    Parameters
    ----------
    mesh_dict : dict
        Dictionary containing mesh information with phonon properties.
    phonon : Phonopy
        The Phonopy object used for the calculation.
    system : SquareLattice2DSystem
        The 2D square lattice system for which the dispersion is plotted.
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
            seg = np.linspace(path[i], path[i + 1], n_points, endpoint=False)
            points.append(seg)
        points.append(path[-1][None, :])
        return np.vstack(points)

    q_path = interpolate_path(path, n_points=100)
    phonon.run_band_structure([q_path], with_eigenvectors=True)
    bands = phonon.get_band_structure_dict()
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

    ax.set_xticks([distances[0], distances[100], distances[200], distances[-1]])
    ax.set_xticklabels(labels)
    ax.set_xlim(distances[0], distances[-1])
    ax.set_ylabel("Frequency (THz)")
    ax.set_title(
        f"Phonon Dispersion of Square Lattice using up to next nearest neighbours: \n {system.n_repeatsx}x{system.n_repeatsy} supercell made of {system.element} atoms"
    )
    ax.grid(visible=True)
    _handles, labels_ = ax.get_legend_handles_labels()
    if "Mesh modes" in labels_ or "All mesh modes" in labels_:
        ax.legend(loc="lower right", bbox_to_anchor=(0.5, -0.25), ncol=2)
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
