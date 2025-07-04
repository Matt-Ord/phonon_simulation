from __future__ import annotations

# Import System from the appropriate module
from phonon_simulation.normal_modes import System, calculate_normal_modes

one_d_chain: System = System(
    lattice_constant=(1.0, 0.0, 0.0),  # 1D: only x-direction matters
    number_of_repeats=(50, 1, 1),  # 10 atoms in x, 1 in y and z
    spring_constant=(1.0, 0.0, 0.0),  # spring constant in x-direction
    mass=1.0,  # mass of each atom
)

calculate_normal_modes(one_d_chain)
