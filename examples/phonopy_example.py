from __future__ import annotations

import numpy as np
from matplotlib import pyplot as plt
from phonopy.api_phonopy import Phonopy
from phonopy.physical_units import get_physical_units
from phonopy.structure.atoms import PhonopyAtoms

a: float = 1.0  # Change to desired lattice spacing. Units in Angstroms
symbol: str = "Si"  # Change to desired element
k: float = 1.0  # Change to desired force constant value. # Units in sqrt(eV/Å²)


EV = get_physical_units().EV
Angstrom = get_physical_units().Angstrom
AMU = get_physical_units().AMU
VaspToOmega = np.sqrt(EV / AMU) / Angstrom

cell: PhonopyAtoms = PhonopyAtoms(
    symbols=[symbol],
    cell=[[a, 0, 0], [0, 1, 0], [0, 0, 1]],
    scaled_positions=[[0, 0, 0]],
)
supercell_matrix: list[list[int]] = [[50, 0, 0], [0, 1, 0], [0, 0, 1]]
phonon: Phonopy = Phonopy(unitcell=cell, supercell_matrix=supercell_matrix)
num_atoms: int = len(phonon.supercell)
fc: np.ndarray = np.zeros((num_atoms, num_atoms, 3, 3), dtype=float)
mass: float = cell.masses[0]

for i in range(num_atoms):
    fc[i, i, 0, 0] = 2 * k
    fc[i, (i - 1) % num_atoms, 0, 0] = -k
    fc[i, (i + 1) % num_atoms, 0, 0] = -k

phonon.force_constants = fc
mesh: tuple[float, float, float] = (101.0, 1.0, 1.0)
phonon.run_mesh(mesh, with_eigenvectors=True, is_mesh_symmetry=False)
mesh_dict: dict[str, np.ndarray] = phonon.get_mesh_dict()
qpoints: np.ndarray = mesh_dict["qpoints"]
frequencies: np.ndarray = mesh_dict["frequencies"]
omega: np.ndarray = (
    mesh_dict["frequencies"] * 1e12 * 2 * np.pi
)  # Converts to angular frequency as phonopy frequency calculated is calcluated in THz
eigenvectors: np.ndarray = mesh_dict["eigenvectors"]
q_vals: np.ndarray = qpoints[:, 0]
sorted_indices: np.ndarray = np.argsort(q_vals)

q_vals = q_vals[sorted_indices]
frequencies = frequencies[sorted_indices]
omega = omega[sorted_indices]
eigenvectors = eigenvectors[sorted_indices]

q_analytical: np.ndarray = np.linspace(-0.5, 0.5, 10000)
omega_analytical: np.ndarray = (
    2 * np.sqrt(k / mass) * VaspToOmega * np.abs(np.sin(np.pi * q_analytical))
)

fig, ax = plt.subplots(figsize=(8, 6))
for band in range(omega.shape[1]):
    ax.plot(q_vals, omega[:, band], label=f"Numerical Mode {band + 1}")
ax.plot(
    q_analytical, omega_analytical, "r:", linewidth=4, label="Analytical Dispersion"
)
ax.axvline(0.5, color="gray", linestyle="--", label="BZ boundary")
ax.axvline(-0.5, color="gray", linestyle="--")
ax.set_xlabel("q (reduced units)")
ax.set_ylabel("Frequency (THz)")
ax.set_title(
    f"Phonon Dispersion: Numerical vs Analytical for a 1D Monoatomic Chain of {cell.symbols[0]} atoms"
)
ax.legend()
ax.grid(visible=True)
plt.tight_layout()
plt.show()
