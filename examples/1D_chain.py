from __future__ import annotations

# Import System from the appropriate module
from phonon_simulation.normal_modes import System, calculate_normal_modes

one_d_chain: System = System(
    lattice_constant=(1.0, 0.0, 0.0),  # 1D: only x-direction matters
    number_of_repeats=(100, 1, 1),  # 10 atoms in x, 1 in y and z
    spring_constant=(1.0, 0.0, 0.0),  # spring constant in x-direction
    mass=1.0,  # mass of each atom
)
# file_name creates a name for files based on input parameters
file_name = (
    f"1D_N{one_d_chain.number_of_repeats[0]}"
    f"_a{one_d_chain.lattice_constant[0]}"
    f"_k{one_d_chain.spring_constant[0]}"
    f"_m{one_d_chain.mass}"
)
folder = r"C:\Users\jm\Documents\Physics\Phonon_Project\OutputFolder1"  # Adjust the output folder path as needed for your device
output_file = rf"{folder}\{file_name}.txt"  # Output file name based on input parameters

calculate_normal_modes(one_d_chain, output_file)
