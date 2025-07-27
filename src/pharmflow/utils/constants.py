# Copyright 2025 PharmFlow Development Team
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
PharmFlow Real Constants and Physical Parameters
"""

import numpy as np
from typing import Dict, List, Any, Tuple
from dataclasses import dataclass
import math

# =============================================================================
# FUNDAMENTAL PHYSICAL CONSTANTS
# =============================================================================

# Universal constants (CODATA 2018 values)
PLANCK_CONSTANT = 6.62607015e-34  # J⋅s
REDUCED_PLANCK_CONSTANT = PLANCK_CONSTANT / (2 * np.pi)  # ℏ in J⋅s
SPEED_OF_LIGHT = 299792458  # m/s (exact)
ELEMENTARY_CHARGE = 1.602176634e-19  # C (exact)
ELECTRON_MASS = 9.1093837015e-31  # kg
PROTON_MASS = 1.67262192369e-27  # kg
NEUTRON_MASS = 1.67492749804e-27  # kg
ATOMIC_MASS_UNIT = 1.66053906660e-27  # kg
AVOGADRO_NUMBER = 6.02214076e23  # mol⁻¹ (exact)
BOLTZMANN_CONSTANT = 1.380649e-23  # J/K (exact)
GAS_CONSTANT = 8.314462618  # J/(mol⋅K) (exact)
FINE_STRUCTURE_CONSTANT = 7.2973525693e-3  # dimensionless
VACUUM_PERMITTIVITY = 8.8541878128e-12  # F/m
VACUUM_PERMEABILITY = 1.25663706212e-6  # H/m
GRAVITATIONAL_CONSTANT = 6.67430e-11  # m³/(kg⋅s²)

# =============================================================================
# UNIT CONVERSIONS
# =============================================================================

class UnitConversions:
    """Real unit conversion constants"""
    
    # Energy conversions (to Joules)
    EV_TO_JOULE = ELEMENTARY_CHARGE  # 1 eV = 1.602176634e-19 J
    CAL_TO_JOULE = 4.184  # 1 cal = 4.184 J (exact)
    KCAL_TO_JOULE = 4184.0  # 1 kcal = 4184 J
    HARTREE_TO_JOULE = 4.3597447222071e-18  # 1 Hartree
    RYDBERG_TO_JOULE = HARTREE_TO_JOULE / 2  # 1 Ry = 0.5 Hartree
    
    # Energy conversions (common in molecular physics)
    HARTREE_TO_EV = HARTREE_TO_JOULE / EV_TO_JOULE  # ≈ 27.211 eV
    HARTREE_TO_KCAL_MOL = HARTREE_TO_JOULE * AVOGADRO_NUMBER / KCAL_TO_JOULE  # ≈ 627.5 kcal/mol
    HARTREE_TO_KJ_MOL = HARTREE_TO_JOULE * AVOGADRO_NUMBER / 1000  # ≈ 2625.5 kJ/mol
    EV_TO_KCAL_MOL = EV_TO_JOULE * AVOGADRO_NUMBER / KCAL_TO_JOULE  # ≈ 23.06 kcal/mol
    
    # Length conversions (to meters)
    ANGSTROM_TO_METER = 1e-10  # 1 Å = 10⁻¹⁰ m
    BOHR_TO_METER = 5.29177210903e-11  # Bohr radius in m
    NANOMETER_TO_METER = 1e-9  # 1 nm = 10⁻⁹ m
    PICOMETER_TO_METER = 1e-12  # 1 pm = 10⁻¹² m
    
    # Mass conversions (to kg)
    AMU_TO_KG = ATOMIC_MASS_UNIT  # 1 u = 1.66053906660e-27 kg
    DALTON_TO_KG = AMU_TO_KG  # 1 Da = 1 u
    
    # Temperature conversions
    @staticmethod
    def celsius_to_kelvin(celsius: float) -> float:
        """Convert Celsius to Kelvin"""
        return celsius + 273.15
    
    @staticmethod
    def fahrenheit_to_kelvin(fahrenheit: float) -> float:
        """Convert Fahrenheit to Kelvin"""
        return (fahrenheit + 459.67) * 5/9
    
    @staticmethod
    def kelvin_to_celsius(kelvin: float) -> float:
        """Convert Kelvin to Celsius"""
        return kelvin - 273.15

# =============================================================================
# MOLECULAR AND ATOMIC CONSTANTS
# =============================================================================

class MolecularConstants:
    """Real molecular and atomic constants"""
    
    # Atomic radii (van der Waals radii in Å)
    VDW_RADII = {
        'H': 1.20, 'He': 1.40,
        'Li': 1.82, 'Be': 1.53, 'B': 1.92, 'C': 1.70, 'N': 1.55, 'O': 1.52, 'F': 1.47, 'Ne': 1.54,
        'Na': 2.27, 'Mg': 1.73, 'Al': 1.84, 'Si': 2.10, 'P': 1.80, 'S': 1.80, 'Cl': 1.75, 'Ar': 1.88,
        'K': 2.75, 'Ca': 2.31, 'Sc': 2.11, 'Ti': 1.87, 'V': 1.79, 'Cr': 1.89, 'Mn': 1.97, 'Fe': 1.94,
        'Co': 1.92, 'Ni': 1.84, 'Cu': 1.32, 'Zn': 1.22, 'Ga': 1.87, 'Ge': 2.11, 'As': 1.85, 'Se': 1.90,
        'Br': 1.85, 'Kr': 2.02, 'Rb': 3.03, 'Sr': 2.49, 'Y': 2.32, 'Zr': 2.23, 'Nb': 2.18, 'Mo': 2.17,
        'Tc': 2.16, 'Ru': 2.13, 'Rh': 2.10, 'Pd': 2.10, 'Ag': 1.72, 'Cd': 1.58, 'In': 1.93, 'Sn': 2.17,
        'Sb': 2.06, 'Te': 2.06, 'I': 1.98, 'Xe': 2.16, 'Cs': 3.43, 'Ba': 2.68, 'La': 2.43, 'Ce': 2.42,
        'Pr': 2.40, 'Nd': 2.39, 'Pm': 2.38, 'Sm': 2.36, 'Eu': 2.35, 'Gd': 2.34, 'Tb': 2.33, 'Dy': 2.31,
        'Ho': 2.30, 'Er': 2.29, 'Tm': 2.27, 'Yb': 2.26, 'Lu': 2.24, 'Hf': 2.23, 'Ta': 2.22, 'W': 2.18,
        'Re': 2.16, 'Os': 2.16, 'Ir': 2.13, 'Pt': 2.13, 'Au': 1.66, 'Hg': 1.55, 'Tl': 1.96, 'Pb': 2.02,
        'Bi': 2.07, 'Po': 1.97, 'At': 2.02, 'Rn': 2.20, 'Fr': 3.48, 'Ra': 2.83, 'Ac': 2.47, 'Th': 2.45,
        'Pa': 2.43, 'U': 2.41, 'Np': 2.39, 'Pu': 2.43, 'Am': 2.44, 'Cm': 2.45, 'Bk': 2.44, 'Cf': 2.45,
        'Es': 2.45, 'Fm': 2.45, 'Md': 2.46, 'No': 2.46, 'Lr': 2.46
    }
    
    # Covalent radii (in Å)
    COVALENT_RADII = {
        'H': 0.31, 'He': 0.28,
        'Li': 1.28, 'Be': 0.96, 'B': 0.84, 'C': 0.73, 'N': 0.71, 'O': 0.66, 'F': 0.57, 'Ne': 0.58,
        'Na': 1.66, 'Mg': 1.41, 'Al': 1.21, 'Si': 1.11, 'P': 1.07, 'S': 1.05, 'Cl': 1.02, 'Ar': 1.06,
        'K': 2.03, 'Ca': 1.76, 'Sc': 1.70, 'Ti': 1.60, 'V': 1.53, 'Cr': 1.39, 'Mn': 1.50, 'Fe': 1.42,
        'Co': 1.38, 'Ni': 1.24, 'Cu': 1.32, 'Zn': 1.22, 'Ga': 1.22, 'Ge': 1.20, 'As': 1.19, 'Se': 1.20,
        'Br': 1.20, 'Kr': 1.16, 'Rb': 2.20, 'Sr': 1.95, 'Y': 1.90, 'Zr': 1.75, 'Nb': 1.64, 'Mo': 1.54,
        'Tc': 1.47, 'Ru': 1.46, 'Rh': 1.42, 'Pd': 1.39, 'Ag': 1.45, 'Cd': 1.44, 'In': 1.42, 'Sn': 1.39,
        'Sb': 1.39, 'Te': 1.38, 'I': 1.39, 'Xe': 1.40, 'Cs': 2.44, 'Ba': 2.15, 'La': 2.07, 'Ce': 2.04,
        'Pr': 2.03, 'Nd': 2.01, 'Pm': 1.99, 'Sm': 1.98, 'Eu': 1.98, 'Gd': 1.96, 'Tb': 1.94, 'Dy': 1.92,
        'Ho': 1.92, 'Er': 1.89, 'Tm': 1.90, 'Yb': 1.87, 'Lu': 1.87, 'Hf': 1.75, 'Ta': 1.70, 'W': 1.62,
        'Re': 1.51, 'Os': 1.44, 'Ir': 1.41, 'Pt': 1.36, 'Au': 1.36, 'Hg': 1.32, 'Tl': 1.45, 'Pb': 1.46,
        'Bi': 1.48, 'Po': 1.40, 'At': 1.50, 'Rn': 1.50, 'Fr': 2.60, 'Ra': 2.21, 'Ac': 2.15, 'Th': 2.06,
        'Pa': 2.00, 'U': 1.96, 'Np': 1.90, 'Pu': 1.87, 'Am': 1.80, 'Cm': 1.69
    }
    
    # Atomic masses (in u, atomic mass units)
    ATOMIC_MASSES = {
        'H': 1.00782503223, 'He': 4.00260325413,
        'Li': 7.0160034366, 'Be': 9.012183065, 'B': 11.00930536, 'C': 12.0000000, 'N': 14.00307400443,
        'O': 15.99491461957, 'F': 18.99840316273, 'Ne': 19.9924401762,
        'Na': 22.9897692820, 'Mg': 23.985041697, 'Al': 26.98153853, 'Si': 27.97692653465,
        'P': 30.97376199842, 'S': 31.9720718004, 'Cl': 34.9688527, 'Ar': 39.9623831237,
        'K': 38.9637064864, 'Ca': 39.962590863, 'Sc': 44.95590828, 'Ti': 47.94794198,
        'V': 50.94395704, 'Cr': 51.94050623, 'Mn': 54.93804391, 'Fe': 55.93493633,
        'Co': 58.93319429, 'Ni': 57.93534241, 'Cu': 62.92959772, 'Zn': 63.92914201,
        'Ga': 68.9255735, 'Ge': 73.921177761, 'As': 74.92159457, 'Se': 79.9165218,
        'Br': 78.9183376, 'Kr': 83.9114977282, 'Rb': 84.9117897379, 'Sr': 87.9056125,
        'Y': 88.9058403, 'Zr': 89.9046977, 'Nb': 92.906373, 'Mo': 97.90540482,
        'Tc': 98.9062508, 'Ru': 101.9043441, 'Rh': 102.905498, 'Pd': 105.9034804,
        'Ag': 106.9050916, 'Cd': 113.90336509, 'In': 114.903878776, 'Sn': 119.90220163,
        'Sb': 120.9038120, 'Te': 129.906222748, 'I': 126.9044719, 'Xe': 131.9041550856,
        'Cs': 132.905451961, 'Ba': 137.90524700, 'La': 138.9063563, 'Ce': 139.9054431,
        'Pr': 140.9076576, 'Nd': 141.9077290, 'Pm': 144.9127559, 'Sm': 151.9197397,
        'Eu': 152.9212380, 'Gd': 157.9241123, 'Tb': 158.9253547, 'Dy': 163.9291819,
        'Ho': 164.9303288, 'Er': 165.9302995, 'Tm': 168.9342179, 'Yb': 173.9388664,
        'Lu': 174.9407752, 'Hf': 179.9465570, 'Ta': 180.9479958, 'W': 183.9509309,
        'Re': 186.9557501, 'Os': 191.9614770, 'Ir': 192.9629216, 'Pt': 194.9647917,
        'Au': 196.9665687, 'Hg': 201.9706434, 'Tl': 204.9744278, 'Pb': 207.9766525,
        'Bi': 208.9803991, 'Po': 208.9824308, 'At': 209.9871479, 'Rn': 222.0175782,
        'Fr': 223.0197360, 'Ra': 226.0254103, 'Ac': 227.0277523, 'Th': 232.0380558,
        'Pa': 231.0358842, 'U': 238.0507884, 'Np': 237.0481736, 'Pu': 244.0642053,
        'Am': 243.0613813, 'Cm': 247.0703541, 'Bk': 247.0703073, 'Cf': 251.0795886,
        'Es': 252.082980, 'Fm': 257.0951061, 'Md': 258.0984315, 'No': 259.10103,
        'Lr': 262.10961, 'Rf': 267.12179, 'Db': 268.12567, 'Sg': 271.13393,
        'Bh': 272.13826, 'Hs': 270.13429, 'Mt': 276.15159, 'Ds': 281.16451,
        'Rg': 280.16514, 'Cn': 285.17712, 'Nh': 284.17873, 'Fl': 289.19042,
        'Mc': 288.19274, 'Lv': 293.20449, 'Ts': 292.20746, 'Og': 294.21392
    }
    
    # Electronegativity values (Pauling scale)
    ELECTRONEGATIVITY = {
        'H': 2.20, 'He': 0.00,
        'Li': 0.98, 'Be': 1.57, 'B': 2.04, 'C': 2.55, 'N': 3.04, 'O': 3.44, 'F': 3.98, 'Ne': 0.00,
        'Na': 0.93, 'Mg': 1.31, 'Al': 1.61, 'Si': 1.90, 'P': 2.19, 'S': 2.58, 'Cl': 3.16, 'Ar': 0.00,
        'K': 0.82, 'Ca': 1.00, 'Sc': 1.36, 'Ti': 1.54, 'V': 1.63, 'Cr': 1.66, 'Mn': 1.55, 'Fe': 1.83,
        'Co': 1.88, 'Ni': 1.91, 'Cu': 1.90, 'Zn': 1.65, 'Ga': 1.81, 'Ge': 2.01, 'As': 2.18, 'Se': 2.55,
        'Br': 2.96, 'Kr': 3.00, 'Rb': 0.82, 'Sr': 0.95, 'Y': 1.22, 'Zr': 1.33, 'Nb': 1.6, 'Mo': 2.16,
        'Tc': 1.9, 'Ru': 2.2, 'Rh': 2.28, 'Pd': 2.20, 'Ag': 1.93, 'Cd': 1.69, 'In': 1.78, 'Sn': 1.96,
        'Sb': 2.05, 'Te': 2.1, 'I': 2.66, 'Xe': 2.60, 'Cs': 0.79, 'Ba': 0.89, 'La': 1.10, 'Ce': 1.12,
        'Pr': 1.13, 'Nd': 1.14, 'Pm': 1.13, 'Sm': 1.17, 'Eu': 1.2, 'Gd': 1.20, 'Tb': 1.1, 'Dy': 1.22,
        'Ho': 1.23, 'Er': 1.24, 'Tm': 1.25, 'Yb': 1.1, 'Lu': 1.27, 'Hf': 1.3, 'Ta': 1.5, 'W': 2.36,
        'Re': 1.9, 'Os': 2.2, 'Ir': 2.20, 'Pt': 2.28, 'Au': 2.54, 'Hg': 2.00, 'Tl': 1.62, 'Pb': 2.33,
        'Bi': 2.02, 'Po': 2.0, 'At': 2.2, 'Rn': 0.0, 'Fr': 0.7, 'Ra': 0.9, 'Ac': 1.1, 'Th': 1.3,
        'Pa': 1.5, 'U': 1.38, 'Np': 1.36, 'Pu': 1.28, 'Am': 1.3, 'Cm': 1.3, 'Bk': 1.3, 'Cf': 1.3,
        'Es': 1.3, 'Fm': 1.3, 'Md': 1.3, 'No': 1.3, 'Lr': 1.3
    }

# =============================================================================
# QUANTUM MECHANICAL CONSTANTS
# =============================================================================

class QuantumConstants:
    """Real quantum mechanical constants"""
    
    # Fundamental quantum constants
    PLANCK_LENGTH = np.sqrt(REDUCED_PLANCK_CONSTANT * GRAVITATIONAL_CONSTANT / SPEED_OF_LIGHT**3)  # ≈ 1.616e-35 m
    PLANCK_TIME = PLANCK_LENGTH / SPEED_OF_LIGHT  # ≈ 5.391e-44 s
    PLANCK_MASS = np.sqrt(REDUCED_PLANCK_CONSTANT * SPEED_OF_LIGHT / GRAVITATIONAL_CONSTANT)  # ≈ 2.176e-8 kg
    PLANCK_ENERGY = PLANCK_MASS * SPEED_OF_LIGHT**2  # ≈ 1.956e9 J
    PLANCK_TEMPERATURE = PLANCK_ENERGY / BOLTZMANN_CONSTANT  # ≈ 1.417e32 K
    
    # Atomic units (Hartree atomic units)
    BOHR_RADIUS = 4 * np.pi * VACUUM_PERMITTIVITY * REDUCED_PLANCK_CONSTANT**2 / (ELECTRON_MASS * ELEMENTARY_CHARGE**2)  # ≈ 5.292e-11 m
    HARTREE_ENERGY = ELECTRON_MASS * ELEMENTARY_CHARGE**4 / (16 * np.pi**2 * VACUUM_PERMITTIVITY**2 * REDUCED_PLANCK_CONSTANT**2)  # ≈ 4.360e-18 J
    ATOMIC_UNIT_TIME = REDUCED_PLANCK_CONSTANT / HARTREE_ENERGY  # ≈ 2.419e-17 s
    ATOMIC_UNIT_VELOCITY = BOHR_RADIUS / ATOMIC_UNIT_TIME  # ≈ 2.187e6 m/s
    
    # Quantum field theory constants
    COMPTON_WAVELENGTH_ELECTRON = PLANCK_CONSTANT / (ELECTRON_MASS * SPEED_OF_LIGHT)  # ≈ 2.426e-12 m
    CLASSICAL_ELECTRON_RADIUS = ELEMENTARY_CHARGE**2 / (4 * np.pi * VACUUM_PERMITTIVITY * ELECTRON_MASS * SPEED_OF_LIGHT**2)  # ≈ 2.818e-15 m
    
    # Magnetic constants
    BOHR_MAGNETON = ELEMENTARY_CHARGE * REDUCED_PLANCK_CONSTANT / (2 * ELECTRON_MASS)  # ≈ 9.274e-24 J/T
    NUCLEAR_MAGNETON = ELEMENTARY_CHARGE * REDUCED_PLANCK_CONSTANT / (2 * PROTON_MASS)  # ≈ 5.051e-27 J/T
    
    # Spectroscopic constants
    RYDBERG_CONSTANT = ELECTRON_MASS * ELEMENTARY_CHARGE**4 / (8 * VACUUM_PERMITTIVITY**2 * PLANCK_CONSTANT**3 * SPEED_OF_LIGHT)  # ≈ 1.097e7 m⁻¹
    RYDBERG_ENERGY = HARTREE_ENERGY / 2  # ≈ 2.180e-18 J

# =============================================================================
# MOLECULAR DYNAMICS CONSTANTS
# =============================================================================

class MDConstants:
    """Real molecular dynamics simulation constants"""
    
    # Force field parameters (typical values)
    LENNARD_JONES_SIGMA_CARBON = 3.4  # Å (for sp3 carbon)
    LENNARD_JONES_EPSILON_CARBON = 0.066  # kcal/mol
    
    LENNARD_JONES_SIGMA_NITROGEN = 3.25  # Å
    LENNARD_JONES_EPSILON_NITROGEN = 0.17  # kcal/mol
    
    LENNARD_JONES_SIGMA_OXYGEN = 3.12  # Å
    LENNARD_JONES_EPSILON_OXYGEN = 0.21  # kcal/mol
    
    LENNARD_JONES_SIGMA_HYDROGEN = 2.5  # Å
    LENNARD_JONES_EPSILON_HYDROGEN = 0.015  # kcal/mol
    
    # Bond parameters (typical values)
    BOND_LENGTH_CC = 1.54  # Å (single bond)
    BOND_LENGTH_CN = 1.47  # Å
    BOND_LENGTH_CO = 1.43  # Å
    BOND_LENGTH_CH = 1.09  # Å
    BOND_LENGTH_NH = 1.01  # Å
    BOND_LENGTH_OH = 0.96  # Å
    
    # Bond force constants (kcal/mol/Å²)
    BOND_FORCE_CC = 310.0
    BOND_FORCE_CN = 340.0
    BOND_FORCE_CO = 350.0
    BOND_FORCE_CH = 340.0
    BOND_FORCE_NH = 434.0
    BOND_FORCE_OH = 553.0
    
    # Angle parameters
    ANGLE_CCC = 112.7  # degrees
    ANGLE_CCN = 109.5  # degrees
    ANGLE_CCO = 109.5  # degrees
    ANGLE_HCH = 107.8  # degrees
    
    # Angle force constants (kcal/mol/rad²)
    ANGLE_FORCE_CCC = 40.0
    ANGLE_FORCE_CCN = 50.0
    ANGLE_FORCE_CCO = 50.0
    ANGLE_FORCE_HCH = 35.0
    
    # Dihedral parameters (kcal/mol)
    DIHEDRAL_BARRIER_CCCC = 1.4  # rotation barrier
    DIHEDRAL_BARRIER_HCCH = 0.16
    
    # Electrostatic parameters
    COULOMB_CONSTANT = 1 / (4 * np.pi * VACUUM_PERMITTIVITY)  # in SI units
    COULOMB_CONSTANT_KCAL = 332.064  # kcal⋅Å/(mol⋅e²) for CHARMM units

# =============================================================================
# CHEMICAL PROPERTY CONSTANTS
# =============================================================================

class ChemicalConstants:
    """Real chemical property constants and thresholds"""
    
    # Drug-likeness rules (Lipinski Rule of Five)
    LIPINSKI_MW_MAX = 500.0  # Da
    LIPINSKI_LOGP_MAX = 5.0
    LIPINSKI_HBD_MAX = 5  # Hydrogen bond donors
    LIPINSKI_HBA_MAX = 10  # Hydrogen bond acceptors
    
    # Veber rules
    VEBER_ROTATABLE_BONDS_MAX = 10
    VEBER_TPSA_MAX = 140.0  # Ų (topological polar surface area)
    
    # Lead-likeness rules
    LEAD_MW_MIN = 200.0  # Da
    LEAD_MW_MAX = 350.0  # Da
    LEAD_LOGP_MIN = 1.0
    LEAD_LOGP_MAX = 3.0
    
    # ADMET thresholds
    BLOOD_BRAIN_BARRIER_LOGPS = -1.0  # LogPS threshold for BBB penetration
    CACO2_PERMEABILITY_THRESHOLD = -5.15  # log(cm/s)
    HUMAN_ORAL_BIOAVAILABILITY_THRESHOLD = 20.0  # %
    HERG_IC50_THRESHOLD = 10.0  # μM (cardiotoxicity)
    
    # Solubility constants
    AQUEOUS_SOLUBILITY_POOR = -4.0  # log(mol/L)
    AQUEOUS_SOLUBILITY_GOOD = -2.0  # log(mol/L)
    
    # Binding affinity ranges
    BINDING_AFFINITY_WEAK = -5.0  # kcal/mol
    BINDING_AFFINITY_MODERATE = -7.0  # kcal/mol
    BINDING_AFFINITY_STRONG = -9.0  # kcal/mol
    BINDING_AFFINITY_VERY_STRONG = -12.0  # kcal/mol

# =============================================================================
# QUANTUM ALGORITHM CONSTANTS
# =============================================================================

class QAOAConstants:
    """Real QAOA and quantum algorithm constants"""
    
    # Circuit depth recommendations
    MIN_QAOA_LAYERS = 1
    MAX_QAOA_LAYERS = 10
    RECOMMENDED_QAOA_LAYERS = 3
    
    # Optimizer parameters
    SPSA_A = 0.628
    SPSA_C = 0.1
    SPSA_ALPHA = 0.602
    SPSA_GAMMA = 0.101
    
    # Convergence criteria
    QAOA_CONVERGENCE_TOLERANCE = 1e-6
    MAX_QAOA_ITERATIONS = 500
    
    # Quantum circuit parameters
    DEFAULT_SHOTS = 1024
    HIGH_PRECISION_SHOTS = 8192
    STATISTICAL_SHOTS = 10000
    
    # Qubit connectivity constraints
    MAX_QUBIT_CONNECTIVITY = 20  # For near-term quantum devices
    TYPICAL_GATE_FIDELITY = 0.999
    TYPICAL_READOUT_FIDELITY = 0.95
    
    # Quantum error mitigation
    ERROR_MITIGATION_OVERHEAD = 3.0  # Multiplicative factor for shots
    ZERO_NOISE_EXTRAPOLATION_POINTS = 3

# =============================================================================
# THERMODYNAMIC CONSTANTS
# =============================================================================

class ThermodynamicConstants:
    """Real thermodynamic constants and standard conditions"""
    
    # Standard conditions
    STANDARD_TEMPERATURE = 298.15  # K (25°C)
    STANDARD_PRESSURE = 101325.0  # Pa (1 atm)
    STANDARD_CONCENTRATION = 1.0  # M
    
    # Phase transition constants
    WATER_FREEZING_POINT = 273.15  # K
    WATER_BOILING_POINT = 373.15  # K
    WATER_CRITICAL_TEMPERATURE = 647.1  # K
    WATER_CRITICAL_PRESSURE = 22064000.0  # Pa
    
    # Entropy constants
    WATER_ENTROPY_LIQUID = 69.9  # J/(mol⋅K) at 298.15 K
    WATER_ENTROPY_GAS = 188.8  # J/(mol⋅K) at 298.15 K
    
    # Heat capacity constants (J/(mol⋅K))
    HEAT_CAPACITY_WATER_LIQUID = 75.3
    HEAT_CAPACITY_WATER_GAS = 33.6
    
    # Solvation free energies (kcal/mol in water)
    SOLVATION_ENERGY_METHANE = 1.99
    SOLVATION_ENERGY_ETHANE = 1.83
    SOLVATION_ENERGY_BENZENE = -0.87
    SOLVATION_ENERGY_PHENOL = -6.62

# =============================================================================
# SPECTROSCOPIC CONSTANTS
# =============================================================================

class SpectroscopyConstants:
    """Real spectroscopic constants and transition energies"""
    
    # IR vibrational frequencies (cm⁻¹)
    IR_FREQ_OH_STRETCH = 3300  # O-H stretch
    IR_FREQ_NH_STRETCH = 3350  # N-H stretch
    IR_FREQ_CH_STRETCH = 2950  # C-H stretch
    IR_FREQ_CO_STRETCH = 1700  # C=O stretch
    IR_FREQ_CC_STRETCH = 1600  # C=C stretch
    IR_FREQ_CN_STRETCH = 1200  # C-N stretch
    
    # UV-Vis absorption wavelengths (nm)
    UV_BENZENE_ABSORPTION = 254  # benzene π→π* transition
    UV_TRYPTOPHAN_ABSORPTION = 280  # tryptophan absorption maximum
    UV_TYROSINE_ABSORPTION = 274  # tyrosine absorption maximum
    UV_PHENYLALANINE_ABSORPTION = 257  # phenylalanine absorption maximum
    
    # NMR chemical shifts (ppm)
    NMR_H_AROMATIC = 7.5  # Aromatic protons
    NMR_H_ALKYL = 2.0  # Alkyl protons
    NMR_H_ALCOHOL = 3.5  # Alcohol protons
    NMR_C_AROMATIC = 130.0  # Aromatic carbons
    NMR_C_ALKYL = 30.0  # Alkyl carbons
    NMR_C_CARBONYL = 200.0  # Carbonyl carbons

# =============================================================================
# COMPUTATIONAL CONSTANTS
# =============================================================================

class ComputationalConstants:
    """Real computational constants for algorithms"""
    
    # Numerical precision
    FLOATING_POINT_TOLERANCE = 1e-15
    OPTIMIZATION_TOLERANCE = 1e-6
    INTEGRATION_TOLERANCE = 1e-8
    
    # Convergence criteria
    SCF_CONVERGENCE_ENERGY = 1e-8  # Hartree
    SCF_CONVERGENCE_DENSITY = 1e-6
    GEOMETRY_CONVERGENCE_ENERGY = 1e-6  # Hartree
    GEOMETRY_CONVERGENCE_GRADIENT = 1e-4  # Hartree/Bohr
    
    # Grid parameters
    DFT_GRID_FINE = 75  # (75,302) Lebedev grid
    DFT_GRID_ULTRAFINE = 99  # (99,590) Lebedev grid
    INTEGRATION_GRID_RADIAL = 100
    INTEGRATION_GRID_ANGULAR = 302
    
    # Basis set parameters
    MIN_EXPONENT = 1e-8  # Minimum Gaussian exponent
    MAX_EXPONENT = 1e8   # Maximum Gaussian exponent
    
    # Monte Carlo parameters
    MC_EQUILIBRATION_STEPS = 1000
    MC_PRODUCTION_STEPS = 10000
    MC_ACCEPTANCE_RATIO_TARGET = 0.5

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

class ConstantUtils:
    """Utility functions for constant calculations"""
    
    @staticmethod
    def wavenumber_to_energy(wavenumber_cm: float) -> float:
        """Convert wavenumber (cm⁻¹) to energy (J)"""
        return wavenumber_cm * 100 * PLANCK_CONSTANT * SPEED_OF_LIGHT
    
    @staticmethod
    def wavelength_to_energy(wavelength_nm: float) -> float:
        """Convert wavelength (nm) to energy (J)"""
        return PLANCK_CONSTANT * SPEED_OF_LIGHT / (wavelength_nm * 1e-9)
    
    @staticmethod
    def energy_to_temperature(energy_j: float) -> float:
        """Convert energy (J) to equivalent temperature (K)"""
        return energy_j / BOLTZMANN_CONSTANT
    
    @staticmethod
    def temperature_to_energy(temperature_k: float) -> float:
        """Convert temperature (K) to thermal energy (J)"""
        return BOLTZMANN_CONSTANT * temperature_k
    
    @staticmethod
    def concentration_to_number_density(concentration_mol_l: float) -> float:
        """Convert molar concentration to number density (m⁻³)"""
        return concentration_mol_l * AVOGADRO_NUMBER * 1000
    
    @staticmethod
    def binding_constant_to_free_energy(ka_m: float, temperature_k: float = STANDARD_TEMPERATURE) -> float:
        """Convert binding constant (M⁻¹) to free energy (J/mol)"""
        return -GAS_CONSTANT * temperature_k * np.log(ka_m)
    
    @staticmethod
    def free_energy_to_binding_constant(delta_g_j_mol: float, temperature_k: float = STANDARD_TEMPERATURE) -> float:
        """Convert free energy (J/mol) to binding constant (M⁻¹)"""
        return np.exp(-delta_g_j_mol / (GAS_CONSTANT * temperature_k))
    
    @staticmethod
    def ic50_to_ki(ic50_m: float, substrate_km_m: float, substrate_conc_m: float) -> float:
        """Convert IC50 to Ki using Cheng-Prusoff equation"""
        return ic50_m / (1 + substrate_conc_m / substrate_km_m)
    
    @staticmethod
    def calculate_molecular_weight(formula: Dict[str, int]) -> float:
        """Calculate molecular weight from atomic composition"""
        mw = 0.0
        for element, count in formula.items():
            if element in MolecularConstants.ATOMIC_MASSES:
                mw += MolecularConstants.ATOMIC_MASSES[element] * count
        return mw
    
    @staticmethod
    def estimate_diffusion_coefficient(molecular_weight_da: float, temperature_k: float = STANDARD_TEMPERATURE) -> float:
        """Estimate diffusion coefficient in water using Stokes-Einstein equation"""
        # Assume spherical molecule with density ~1 g/cm³
        radius_m = ((3 * molecular_weight_da * UnitConversions.AMU_TO_KG) / (4 * np.pi * 1000))**(1/3)
        viscosity_water = 8.9e-4  # Pa⋅s at 298 K
        return BOLTZMANN_CONSTANT * temperature_k / (6 * np.pi * viscosity_water * radius_m)

# =============================================================================
# VALIDATION AND TESTING
# =============================================================================

def validate_constants():
    """Validate that all constants are physically reasonable"""
    
    print("Validating PharmFlow constants...")
    
    # Test fundamental constants
    assert abs(SPEED_OF_LIGHT - 299792458) < 1, "Speed of light incorrect"
    assert abs(PLANCK_CONSTANT - 6.62607015e-34) < 1e-40, "Planck constant incorrect"
    assert abs(ELEMENTARY_CHARGE - 1.602176634e-19) < 1e-25, "Elementary charge incorrect"
    
    # Test unit conversions
    assert abs(UnitConversions.HARTREE_TO_EV - 27.211) < 0.001, "Hartree to eV conversion incorrect"
    assert abs(UnitConversions.HARTREE_TO_KCAL_MOL - 627.5) < 0.1, "Hartree to kcal/mol conversion incorrect"
    
    # Test molecular constants
    assert 1.0 < MolecularConstants.VDW_RADII['C'] < 2.0, "Carbon VdW radius unreasonable"
    assert 10 < MolecularConstants.ATOMIC_MASSES['C'] < 15, "Carbon atomic mass unreasonable"
    assert 2.0 < MolecularConstants.ELECTRONEGATIVITY['C'] < 3.0, "Carbon electronegativity unreasonable"
    
    # Test chemical constants
    assert ChemicalConstants.LIPINSKI_MW_MAX == 500.0, "Lipinski MW threshold incorrect"
    assert ChemicalConstants.LIPINSKI_LOGP_MAX == 5.0, "Lipinski LogP threshold incorrect"
    
    # Test utility functions
    co2_mw = ConstantUtils.calculate_molecular_weight({'C': 1, 'O': 2})
    assert abs(co2_mw - 44.01) < 0.1, "CO2 molecular weight calculation incorrect"
    
    # Test energy conversions
    room_temp_energy = ConstantUtils.temperature_to_energy(298.15)
    assert 3e-21 < room_temp_energy < 5e-21, "Room temperature energy conversion incorrect"
    
    print("✅ All constants validated successfully!")
    
    # Print some key values for verification
    print(f"\nKey Physical Constants:")
    print(f"  Speed of light: {SPEED_OF_LIGHT:e} m/s")
    print(f"  Planck constant: {PLANCK_CONSTANT:e} J⋅s")
    print(f"  Boltzmann constant: {BOLTZMANN_CONSTANT:e} J/K")
    print(f"  Avogadro number: {AVOGADRO_NUMBER:e} mol⁻¹")
    
    print(f"\nKey Unit Conversions:")
    print(f"  1 Hartree = {UnitConversions.HARTREE_TO_EV:.3f} eV")
    print(f"  1 Hartree = {UnitConversions.HARTREE_TO_KCAL_MOL:.1f} kcal/mol")
    print(f"  1 Bohr = {UnitConversions.BOHR_TO_METER:e} m")
    print(f"  1 Å = {UnitConversions.ANGSTROM_TO_METER:e} m")
    
    print(f"\nKey Molecular Properties:")
    print(f"  Carbon VdW radius: {MolecularConstants.VDW_RADII['C']:.2f} Å")
    print(f"  Carbon atomic mass: {MolecularConstants.ATOMIC_MASSES['C']:.6f} u")
    print(f"  Carbon electronegativity: {MolecularConstants.ELECTRONEGATIVITY['C']:.2f}")
    
    print(f"\nKey Chemical Thresholds:")
    print(f"  Lipinski MW max: {ChemicalConstants.LIPINSKI_MW_MAX} Da")
    print(f"  Strong binding affinity: < {ChemicalConstants.BINDING_AFFINITY_STRONG} kcal/mol")
    print(f"  BBB LogPS threshold: {ChemicalConstants.BLOOD_BRAIN_BARRIER_LOGPS}")

if __name__ == "__main__":
    validate_constants()
