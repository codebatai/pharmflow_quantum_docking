"""
Physical and computational constants for PharmFlow Quantum Molecular Docking
"""

import numpy as np
from typing import Dict, List

# ============================================================================
# PHYSICAL CONSTANTS
# ============================================================================

# Fundamental constants
AVOGADRO_NUMBER = 6.02214076e23  # mol^-1
BOLTZMANN_CONSTANT = 1.380649e-23  # J/K
GAS_CONSTANT = 8.314462618  # J/(mol·K)
PLANCK_CONSTANT = 6.62607015e-34  # J·s

# Unit conversion factors
KCAL_TO_JOULE = 4184.0  # J/kcal
JOULE_TO_KCAL = 1.0 / KCAL_TO_JOULE
ANGSTROM_TO_METER = 1e-10  # m/Å
BOHR_TO_ANGSTROM = 0.529177210903  # Å/bohr

# Electrostatic constants
COULOMB_CONSTANT = 332.0637  # kcal·Å/(mol·e^2)
ELEMENTARY_CHARGE = 1.602176634e-19  # C
VACUUM_PERMITTIVITY = 8.8541878128e-12  # F/m

# Thermal energy at room temperature (298.15 K)
RT_KCAL_MOL = 0.592  # kcal/mol
KT_JOULE = BOLTZMANN_CONSTANT * 298.15  # J

# ============================================================================
# MOLECULAR PARAMETERS
# ============================================================================

# Atomic radii (van der Waals radii in Angstroms)
VDW_RADII = {
    'H': 1.20,   'He': 1.40,  'Li': 1.82,  'Be': 1.53,  'B': 1.92,
    'C': 1.70,   'N': 1.55,   'O': 1.52,   'F': 1.47,   'Ne': 1.54,
    'Na': 2.27,  'Mg': 1.73,  'Al': 1.84,  'Si': 2.10,  'P': 1.80,
    'S': 1.80,   'Cl': 1.75,  'Ar': 1.88,  'K': 2.75,   'Ca': 2.31,
    'Sc': 2.11,  'Ti': 1.87,  'V': 1.79,   'Cr': 1.89,  'Mn': 1.97,
    'Fe': 1.94,  'Co': 1.92,  'Ni': 1.84,  'Cu': 1.32,  'Zn': 1.22,
    'Ga': 1.87,  'Ge': 2.11,  'As': 1.85,  'Se': 1.90,  'Br': 1.85,
    'Kr': 2.02,  'Rb': 3.03,  'Sr': 2.49,  'I': 1.98,   'Xe': 2.16
}

# Atomic masses (in atomic mass units)
ATOMIC_MASSES = {
    'H': 1.008,   'He': 4.003,  'Li': 6.941,  'Be': 9.012,  'B': 10.811,
    'C': 12.011,  'N': 14.007,  'O': 15.999,  'F': 18.998,  'Ne': 20.180,
    'Na': 22.990, 'Mg': 24.305, 'Al': 26.982, 'Si': 28.086, 'P': 30.974,
    'S': 32.065,  'Cl': 35.453, 'Ar': 39.948, 'K': 39.098,  'Ca': 40.078,
    'Sc': 44.956, 'Ti': 47.867, 'V': 50.942,  'Cr': 51.996, 'Mn': 54.938,
    'Fe': 55.845, 'Co': 58.933, 'Ni': 58.693, 'Cu': 63.546, 'Zn': 65.409,
    'Ga': 69.723, 'Ge': 72.640, 'As': 74.922, 'Se': 78.960, 'Br': 79.904,
    'Kr': 83.798, 'Rb': 85.468, 'Sr': 87.620, 'I': 126.905, 'Xe': 131.293
}

# Covalent radii (in Angstroms)
COVALENT_RADII = {
    'H': 0.31,   'He': 0.28,  'Li': 1.28,  'Be': 0.96,  'B': 0.84,
    'C': 0.76,   'N': 0.71,   'O': 0.66,   'F': 0.57,   'Ne': 0.58,
    'Na': 1.66,  'Mg': 1.41,  'Al': 1.21,  'Si': 1.11,  'P': 1.07,
    'S': 1.05,   'Cl': 1.02,  'Ar': 1.06,  'K': 2.03,   'Ca': 1.76
}

# Electronegativity values (Pauling scale)
ELECTRONEGATIVITY = {
    'H': 2.20,   'He': 0.00,  'Li': 0.98,  'Be': 1.57,  'B': 2.04,
    'C': 2.55,   'N': 3.04,   'O': 3.44,   'F': 3.98,   'Ne': 0.00,
    'Na': 0.93,  'Mg': 1.31,  'Al': 1.61,  'Si': 1.90,  'P': 2.19,
    'S': 2.58,   'Cl': 3.16,  'Ar': 0.00,  'K': 0.82,   'Ca': 1.00,
    'Br': 2.96,  'I': 2.66
}

# ============================================================================
# FORCE FIELD PARAMETERS
# ============================================================================

# MMFF94 van der Waals parameters (epsilon in kcal/mol, sigma in Angstroms)
MMFF94_VDW_PARAMS = {
    'C': {'epsilon': 0.070, 'sigma': 3.50},  # sp3 carbon
    'Car': {'epsilon': 0.070, 'sigma': 3.55},  # aromatic carbon
    'N': {'epsilon': 0.170, 'sigma': 3.25},  # sp3 nitrogen
    'Nar': {'epsilon': 0.170, 'sigma': 3.25},  # aromatic nitrogen
    'O': {'epsilon': 0.170, 'sigma': 3.07},  # oxygen
    'S': {'epsilon': 0.200, 'sigma': 3.60},  # sulfur
    'H': {'epsilon': 0.030, 'sigma': 2.50},  # hydrogen
    'F': {'epsilon': 0.061, 'sigma': 2.94},  # fluorine
    'Cl': {'epsilon': 0.227, 'sigma': 3.52},  # chlorine
    'Br': {'epsilon': 0.389, 'sigma': 3.73},  # bromine
    'I': {'epsilon': 0.550, 'sigma': 4.00},  # iodine
    'P': {'epsilon': 0.200, 'sigma': 3.74}   # phosphorus
}

# Hydrogen bond parameters
HBOND_ENERGY_RANGE = (-5.0, -0.5)  # kcal/mol
HBOND_DISTANCE_RANGE = (2.5, 3.5)  # Angstroms
HBOND_ANGLE_CUTOFF = 120.0  # degrees

# Hydrophobic interaction parameters
HYDROPHOBIC_ENERGY = -0.5  # kcal/mol per contact
HYDROPHOBIC_DISTANCE_CUTOFF = 4.5  # Angstroms

# ============================================================================
# QUANTUM COMPUTING PARAMETERS
# ============================================================================

# QAOA algorithm parameters
DEFAULT_QAOA_LAYERS = 3
MAX_QAOA_LAYERS = 10
DEFAULT_QAOA_ITERATIONS = 200
MAX_QAOA_ITERATIONS = 1000

# Quantum circuit parameters
MAX_QUBITS = 100
MEASUREMENT_SHOTS = 8192
QUANTUM_NOISE_THRESHOLD = 0.01

# Optimization parameters
CONVERGENCE_TOLERANCE = 1e-6
MAX_OPTIMIZATION_ITERATIONS = 500
GRADIENT_TOLERANCE = 1e-4
PARAMETER_BOUNDS = (-2*np.pi, 2*np.pi)

# Smoothing filter parameters
DEFAULT_SMOOTHING_FACTOR = 0.1
MIN_SMOOTHING_FACTOR = 0.01
MAX_SMOOTHING_FACTOR = 0.5
ROUGHNESS_THRESHOLD = 2.0

# ============================================================================
# MOLECULAR ENCODING PARAMETERS
# ============================================================================

# Binary encoding bit allocations
POSITION_ENCODING_BITS = 18  # 6 bits per coordinate (x, y, z)
ROTATION_ENCODING_BITS = 12  # 4 bits per Euler angle (α, β, γ)
BOND_ENCODING_BITS = 3       # 3 bits per rotatable bond

# Coordinate ranges for encoding
POSITION_RANGE = (-10.0, 10.0)  # Angstroms
ROTATION_RANGE = (0.0, 2*np.pi)  # radians
BOND_ANGLE_RANGE = (0.0, 2*np.pi)  # radians

# Grid resolutions
POSITION_RESOLUTION = 64  # 2^6 positions per axis
ROTATION_RESOLUTION = 16  # 2^4 angles per rotation
BOND_RESOLUTION = 8       # 2^3 angles per bond

# ============================================================================
# PHARMACOPHORE PARAMETERS
# ============================================================================

# Pharmacophore types and properties
PHARMACOPHORE_TYPES = {
    'hydrophobic': {'color': 'yellow', 'radius': 1.5, 'tolerance': 2.0},
    'hydrogen_bond_donor': {'color': 'cyan', 'radius': 1.0, 'tolerance': 1.5},
    'hydrogen_bond_acceptor': {'color': 'red', 'radius': 1.0, 'tolerance': 1.5},
    'aromatic': {'color': 'orange', 'radius': 2.0, 'tolerance': 1.0},
    'positive_ionizable': {'color': 'blue', 'radius': 1.5, 'tolerance': 2.0},
    'negative_ionizable': {'color': 'magenta', 'radius': 1.5, 'tolerance': 2.0}
}

# Pharmacophore interaction strengths (kcal/mol)
PHARMACOPHORE_INTERACTIONS = {
    ('hydrogen_bond_donor', 'hydrogen_bond_acceptor'): -3.0,
    ('hydrogen_bond_acceptor', 'hydrogen_bond_donor'): -3.0,
    ('positive_ionizable', 'negative_ionizable'): -5.0,
    ('negative_ionizable', 'positive_ionizable'): -5.0,
    ('hydrophobic', 'hydrophobic'): -1.0,
    ('aromatic', 'aromatic'): -2.0
}

# ============================================================================
# ENERGY EVALUATION PARAMETERS
# ============================================================================

# Standard energy component weights for molecular docking
STANDARD_ENERGY_WEIGHTS = {
    'vdw_energy': 0.35,
    'electrostatic': 0.25,
    'hydrogen_bonds': 0.20,
    'hydrophobic': 0.12,
    'solvation': 0.05,
    'internal_strain': 0.03
}

# Energy cutoffs and thresholds
ENERGY_CUTOFF_DISTANCE = 12.0  # Angstroms
VDW_CUTOFF_DISTANCE = 8.0      # Angstroms
ELECTROSTATIC_CUTOFF_DISTANCE = 12.0  # Angstroms
CLASH_DISTANCE_THRESHOLD = 0.8  # × sum of VDW radii

# Solvation parameters (Generalized Born model)
GB_PROBE_RADIUS = 1.4  # Angstroms (water probe)
GB_SURFACE_TENSION = 0.005  # kcal/(mol·Å²)
GB_DIELECTRIC_CONSTANT = 78.5  # water at 298K

# ============================================================================
# CLASSICAL OPTIMIZATION PARAMETERS
# ============================================================================

# Optimization algorithm settings
CLASSICAL_MAX_ITERATIONS = 100
CLASSICAL_CONVERGENCE_TOLERANCE = 1e-4
CLASSICAL_STEP_SIZE = 0.01
CLASSICAL_LINE_SEARCH_TOLERANCE = 1e-6

# Molecular dynamics parameters
MD_TEMPERATURE = 298.15  # K
MD_PRESSURE = 1.0        # atm
MD_TIME_STEP = 1.0       # fs
MD_EQUILIBRATION_STEPS = 1000
MD_PRODUCTION_STEPS = 5000

# ============================================================================
# DRUG-LIKENESS PARAMETERS
# ============================================================================

# Lipinski's Rule of Five
LIPINSKI_MW_MAX = 500.0      # Da
LIPINSKI_LOGP_MAX = 5.0      # dimensionless
LIPINSKI_HBD_MAX = 5         # count
LIPINSKI_HBA_MAX = 10        # count

# Extended drug-likeness criteria
VEBER_TPSA_MAX = 140.0       # Ų
VEBER_ROTBONDS_MAX = 10      # count
EGAN_LOGP_RANGE = (-2.0, 6.0)
EGAN_TPSA_RANGE = (0.0, 200.0)

# ADMET thresholds
SOLUBILITY_THRESHOLD = -4.0   # log S
PERMEABILITY_THRESHOLD = -5.0 # log Papp
BBB_THRESHOLD = 0.1          # brain/blood ratio
CYP_INHIBITION_THRESHOLD = 10.0  # μM

# ============================================================================
# VALIDATION AND BENCHMARKING PARAMETERS
# ============================================================================

# Accuracy thresholds
RMSD_SUCCESS_THRESHOLD = 2.0  # Angstroms
RMSD_GOOD_THRESHOLD = 1.5     # Angstroms
RMSD_EXCELLENT_THRESHOLD = 1.0  # Angstroms

# Binding affinity prediction accuracy
AFFINITY_CORRELATION_THRESHOLD = 0.7  # Pearson R
AFFINITY_RMSE_THRESHOLD = 1.5         # log units

# Computational performance targets
TARGET_SPEEDUP_FACTOR = 10.0
TARGET_SUCCESS_RATE = 0.85
TARGET_EFFICIENCY = 0.90  # successful poses / total attempts

# ============================================================================
# BENCHMARK DATASETS
# ============================================================================

# Standard benchmarking datasets
BENCHMARK_DATASETS = {
    'PDBbind_core': {
        'size': 285,
        'description': 'High-quality protein-ligand complexes',
        'reference': 'Wang et al., J. Med. Chem. 2004'
    },
    'CASF_2016': {
        'size': 285,
        'description': 'Comparative Assessment of Scoring Functions',
        'reference': 'Su et al., J. Chem. Inf. Model. 2019'
    },
    'DUD_E': {
        'size': 22886,
        'description': 'Directory of Useful Decoys Enhanced',
        'reference': 'Mysinger et al., J. Chem. Inf. Model. 2012'
    },
    'ChEMBL': {
        'size': 1000000,
        'description': 'Large-scale bioactivity database',
        'reference': 'Gaulton et al., Nucleic Acids Res. 2017'
    }
}

# ============================================================================
# ERROR HANDLING AND LOGGING
# ============================================================================

# Error tolerance levels
ERROR_TOLERANCE_STRICT = 1e-8
ERROR_TOLERANCE_NORMAL = 1e-6
ERROR_TOLERANCE_RELAXED = 1e-4

# Logging levels and formats
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
LOG_DATE_FORMAT = '%Y-%m-%d %H:%M:%S'

# Performance monitoring thresholds
MEMORY_WARNING_THRESHOLD = 0.8  # 80% of available memory
CPU_WARNING_THRESHOLD = 0.9     # 90% CPU usage
DISK_WARNING_THRESHOLD = 0.9    # 90% disk usage

# ============================================================================
# SYSTEM CONFIGURATION
# ============================================================================

# Parallel processing parameters
DEFAULT_N_CORES = 4
MAX_N_CORES = 16
THREAD_POOL_SIZE = 8

# Memory management
DEFAULT_MEMORY_LIMIT = '4GB'
MAX_MOLECULE_SIZE = 1000  # atoms
MAX_BATCH_SIZE = 100      # molecules

# File I/O parameters
MAX_FILE_SIZE = 100 * 1024 * 1024  # 100 MB
SUPPORTED_FORMATS = ['pdb', 'sdf', 'mol2', 'xyz', 'smiles']
TEMP_DIR_PREFIX = 'pharmflow_tmp_'

# ============================================================================
# VERSION AND METADATA
# ============================================================================

PHARMFLOW_VERSION = "1.0.0"
QISKIT_MIN_VERSION = "1.0.0"
RDKIT_MIN_VERSION = "2023.3.1"
PYTHON_MIN_VERSION = "3.8"

# Citations and references
CITATIONS = {
    'pharmflow': 'PharmFlow Team, "QAOA and Pharmacophore-Optimized Quantum Molecular Docking", 2025',
    'qiskit': 'Qiskit contributors, "Qiskit: An Open-source Framework for Quantum Computing", 2023',
    'rdkit': 'RDKit Team, "RDKit: Open-source cheminformatics", 2023'
}

# API endpoints and services
API_ENDPOINTS = {
    'protein_db': 'https://www.rcsb.org/pdb/',
    'chembl': 'https://www.ebi.ac.uk/chembl/',
    'pubchem': 'https://pubchem.ncbi.nlm.nih.gov/',
    'uniprot': 'https://www.uniprot.org/'
}
