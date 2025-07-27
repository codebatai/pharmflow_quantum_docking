"""
PharmFlow Constants and Configuration Parameters
Central repository for all system constants, defaults, and configuration values
"""

import numpy as np
from typing import Dict, List, Tuple, Any

# ====================================
# QUANTUM COMPUTING CONSTANTS
# ====================================

# QAOA Configuration
DEFAULT_QAOA_LAYERS = 3
MAX_QAOA_LAYERS = 10
MIN_QAOA_LAYERS = 1

# Quantum Circuit Parameters
MEASUREMENT_SHOTS = 1024
MAX_MEASUREMENT_SHOTS = 8192
MIN_MEASUREMENT_SHOTS = 256

# Quantum Backends
SUPPORTED_BACKENDS = [
    'qasm_simulator',
    'statevector_simulator', 
    'aer_simulator',
    'ibmq_qasm_simulator',
    'ibmq_lima',
    'ibmq_belem',
    'ibmq_quito'
]

# Optimization Parameters
OPTIMIZATION_TOLERANCE = 1e-6
MAX_OPTIMIZATION_ITERATIONS = 500
DEFAULT_OPTIMIZATION_ITERATIONS = 200

# ====================================
# MOLECULAR ENCODING CONSTANTS
# ====================================

# Position Encoding
POSITION_ENCODING_BITS = 18  # 6 bits per coordinate (x, y, z)
COORDINATE_BITS = 6
POSITION_RANGE = (-10.0, 10.0)  # Angstroms

# Rotation Encoding
ROTATION_ENCODING_BITS = 12  # 4 bits per angle (α, β, γ)
ANGLE_BITS = 4
ROTATION_RANGE = (0.0, 2 * np.pi)  # Radians

# Bond Encoding
BOND_ENCODING_BITS = 3  # 3 bits per rotatable bond
MAX_ROTATABLE_BONDS = 20

# ====================================
# PHARMACOPHORE DEFINITIONS
# ====================================

# Pharmacophore Types
PHARMACOPHORE_TYPES = {
    'hydrophobic': {
        'code': 0,
        'color': '#FFD700',
        'radius': 1.5,
        'importance': 0.8
    },
    'hydrogen_bond_donor': {
        'code': 1,
        'color': '#FF4500',
        'radius': 1.2,
        'importance': 1.0
    },
    'hydrogen_bond_acceptor': {
        'code': 2,
        'color': '#0000FF',
        'radius': 1.2,
        'importance': 1.0
    },
    'positive_ionizable': {
        'code': 3,
        'color': '#FF0000',
        'radius': 1.8,
        'importance': 1.2
    },
    'negative_ionizable': {
        'code': 4,
        'color': '#8B0000',
        'radius': 1.8,
        'importance': 1.2
    },
    'aromatic': {
        'code': 5,
        'color': '#800080',
        'radius': 2.0,
        'importance': 0.9
    },
    'halogen_bond': {
        'code': 6,
        'color': '#00FF00',
        'radius': 1.5,
        'importance': 0.7
    }
}

# Pharmacophore Interaction Matrix
PHARMACOPHORE_INTERACTIONS = {
    ('hydrogen_bond_donor', 'hydrogen_bond_acceptor'): -2.5,
    ('hydrogen_bond_acceptor', 'hydrogen_bond_donor'): -2.5,
    ('positive_ionizable', 'negative_ionizable'): -3.0,
    ('negative_ionizable', 'positive_ionizable'): -3.0,
    ('hydrophobic', 'hydrophobic'): -1.5,
    ('aromatic', 'aromatic'): -2.0,
    ('halogen_bond', 'hydrogen_bond_acceptor'): -1.8,
    ('hydrogen_bond_acceptor', 'halogen_bond'): -1.8
}

# Distance Parameters
INTERACTION_DISTANCE_RANGE = (2.0, 6.0)  # Angstroms
OPTIMAL_INTERACTION_DISTANCE = 3.5  # Angstroms
DISTANCE_TOLERANCE = 1.0  # Angstroms

# ====================================
# ENERGY CALCULATION CONSTANTS
# ====================================

# Standard Energy Weights
STANDARD_ENERGY_WEIGHTS = {
    'van_der_waals': 0.25,
    'electrostatic': 0.35,
    'hydrogen_bonding': 0.25,
    'hydrophobic': 0.10,
    'entropy': 0.05
}

# Energy Scaling Factors
VDW_SCALING = 1.0
ELECTROSTATIC_SCALING = 1.2
HYDROGEN_BOND_SCALING = 1.5
HYDROPHOBIC_SCALING = 0.8

# Energy Ranges (kcal/mol)
BINDING_AFFINITY_RANGE = (-15.0, 0.0)
FAVORABLE_BINDING_THRESHOLD = -7.0
STRONG_BINDING_THRESHOLD = -10.0

# ====================================
# MOLECULAR DESCRIPTORS
# ====================================

# Lipinski's Rule of Five
LIPINSKI_LIMITS = {
    'molecular_weight': 500.0,  # Da
    'logp': 5.0,
    'hbd': 5,  # Hydrogen bond donors
    'hba': 10,  # Hydrogen bond acceptors
    'rotatable_bonds': 10
}

# Extended Drug-likeness Rules
VEBER_RULES = {
    'tpsa': 140.0,  # Å²
    'rotatable_bonds': 10
}

GHOSE_FILTER = {
    'molecular_weight': (160.0, 480.0),
    'logp': (-0.4, 5.6),
    'atoms': (20, 70),
    'molar_refractivity': (40.0, 130.0)
}

# ====================================
# ADMET PARAMETERS
# ====================================

# Absorption Parameters
ABSORPTION_THRESHOLDS = {
    'caco2_permeability': -5.15,  # log Papp (cm/s)
    'mdck_permeability': -6.0,
    'pgp_substrate_probability': 0.7,
    'bioavailability_score': 0.55
}

# Distribution Parameters
DISTRIBUTION_THRESHOLDS = {
    'bbb_permeability': -1.0,  # log BB
    'cns_permeability': -2.0,  # log PS
    'vdss': 0.71,  # L/kg
    'protein_binding': 0.9
}

# Metabolism Parameters
METABOLISM_THRESHOLDS = {
    'cyp3a4_substrate': 0.5,
    'cyp2d6_substrate': 0.5,
    'cyp1a2_inhibitor': 0.5,
    'cyp2c9_inhibitor': 0.5,
    'cyp2d6_inhibitor': 0.5,
    'cyp3a4_inhibitor': 0.5
}

# Excretion Parameters
EXCRETION_THRESHOLDS = {
    'clearance_hepatocyte': 0.5,  # ml/min/kg
    'clearance_microsome': 0.5,
    'half_life': 4.0  # hours
}

# Toxicity Parameters
TOXICITY_THRESHOLDS = {
    'acute_toxicity': 3.0,  # -log LD50 (mol/kg)
    'skin_sensitization': 0.5,
    'hepatotoxicity': 0.5,
    'carcinogenicity': 0.5,
    'mutagenicity': 0.5,
    'herg_cardiotoxicity': 0.5
}

# ====================================
# VISUALIZATION CONSTANTS
# ====================================

# Color Schemes
COLOR_PALETTES = {
    'binding_affinity': ['#2E8B57', '#32CD32', '#FFD700', '#FF6347', '#DC143C'],
    'admet_scores': ['#FF0000', '#FF4500', '#FFD700', '#9AFF9A', '#008000'],
    'quantum_classical': ['#4169E1', '#DC143C', '#32CD32', '#FF1493'],
    'pharmacophores': ['#FFD700', '#FF4500', '#0000FF', '#FF0000', '#8B0000', '#800080']
}

# Plot Parameters
FIGURE_SIZE = (12, 8)
DPI = 300
FONT_SIZE = 12
TITLE_FONT_SIZE = 14
AXIS_FONT_SIZE = 10

# 3D Visualization
MOLECULE_SPHERE_SIZE = 0.3
BOND_CYLINDER_RADIUS = 0.1
PHARMACOPHORE_SPHERE_SIZE = 0.5

# ====================================
# FILE FORMAT CONSTANTS
# ====================================

# Supported Molecular File Formats
MOLECULAR_FILE_FORMATS = {
    'protein': ['.pdb', '.pdbx', '.mmcif', '.ent'],
    'ligand': ['.sdf', '.mol', '.mol2', '.smiles', '.pdb'],
    'trajectory': ['.dcd', '.xtc', '.trr', '.nc']
}

# Database Identifiers
DATABASE_PREFIXES = {
    'pdb': 'PDB:',
    'chembl': 'CHEMBL',
    'pubchem': 'CID:',
    'zinc': 'ZINC',
    'drugbank': 'DB'
}

# ====================================
# PERFORMANCE CONSTANTS
# ====================================

# Memory Management
MAX_MEMORY_GB = 16
CHUNK_SIZE = 1000
BATCH_SIZE = 100

# Parallel Processing
DEFAULT_N_JOBS = 4
MAX_N_JOBS = 16

# Caching
CACHE_SIZE_MB = 512
CACHE_EXPIRY_HOURS = 24

# ====================================
# ERROR HANDLING CONSTANTS
# ====================================

# Numerical Tolerances
NUMERICAL_TOLERANCE = 1e-10
CONVERGENCE_TOLERANCE = 1e-6
ZERO_THRESHOLD = 1e-12

# Retry Parameters
MAX_RETRIES = 3
RETRY_DELAY = 1.0  # seconds
BACKOFF_FACTOR = 2.0

# ====================================
# LOGGING CONFIGURATION
# ====================================

# Log Levels
LOG_LEVELS = {
    'DEBUG': 10,
    'INFO': 20,
    'WARNING': 30,
    'ERROR': 40,
    'CRITICAL': 50
}

# Log Format
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
DATE_FORMAT = '%Y-%m-%d %H:%M:%S'

# ====================================
# VALIDATION CONSTANTS
# ====================================

# Molecular Validation
MIN_ATOMS = 5
MAX_ATOMS = 1000
MIN_MOLECULAR_WEIGHT = 100.0  # Da
MAX_MOLECULAR_WEIGHT = 2000.0  # Da

# Protein Validation
MIN_PROTEIN_RESIDUES = 50
MAX_PROTEIN_RESIDUES = 10000
MIN_BINDING_SITE_RESIDUES = 3
MAX_BINDING_SITE_RESIDUES = 50

# ====================================
# UNIT CONVERSIONS
# ====================================

# Energy Units
KCAL_TO_KJ = 4.184
HARTREE_TO_KCAL = 627.509
EV_TO_KCAL = 23.061

# Distance Units
ANGSTROM_TO_NM = 0.1
BOHR_TO_ANGSTROM = 0.529177

# ====================================
# PHYSICAL CONSTANTS
# ====================================

# Fundamental Constants
BOLTZMANN_CONSTANT = 1.380649e-23  # J/K
GAS_CONSTANT = 8.314462618  # J/(mol·K)
AVOGADRO_NUMBER = 6.02214076e23  # mol⁻¹
PLANCK_CONSTANT = 6.62607015e-34  # J·s

# Temperature
ROOM_TEMPERATURE = 298.15  # K
BODY_TEMPERATURE = 310.15  # K

# ====================================
# ALGORITHM PARAMETERS
# ====================================

# Optimization Algorithm Settings
OPTIMIZATION_ALGORITHMS = {
    'COBYLA': {
        'maxiter': 200,
        'tol': 1e-6,
        'rhobeg': 1.0,
        'rhoend': 1e-6
    },
    'SPSA': {
        'maxiter': 200,
        'learning_rate': 0.01,
        'perturbation': 0.1
    },
    'Adam': {
        'maxiter': 200,
        'learning_rate': 0.001,
        'beta1': 0.9,
        'beta2': 0.999
    }
}

# Smoothing Filter Parameters
SMOOTHING_PARAMETERS = {
    'gaussian_sigma': 0.1,
    'kernel_size': 3,
    'filter_strength': 0.5
}

# ====================================
# EXPERIMENTAL CONSTANTS
# ====================================

# Assay Conditions
STANDARD_CONDITIONS = {
    'temperature': 298.15,  # K
    'ph': 7.4,
    'ionic_strength': 0.15,  # M
    'pressure': 101325  # Pa
}

# Measurement Uncertainties
EXPERIMENTAL_UNCERTAINTY = {
    'binding_affinity': 0.5,  # kcal/mol
    'ic50': 0.3,  # log units
    'ki': 0.3,  # log units
    'kd': 0.3  # log units
}

# ====================================
# VERSION INFORMATION
# ====================================

VERSION = "1.0.0"
API_VERSION = "v1"
SCHEMA_VERSION = "1.0"

# ====================================
# CONFIGURATION DEFAULTS
# ====================================

# Default Configuration Dictionary
DEFAULT_CONFIG = {
    'quantum': {
        'backend': 'qasm_simulator',
        'optimizer': 'COBYLA',
        'num_qaoa_layers': DEFAULT_QAOA_LAYERS,
        'shots': MEASUREMENT_SHOTS,
        'noise_mitigation': True
    },
    'classical': {
        'force_field': 'MMFF94',
        'charge_method': 'Gasteiger',
        'conformer_generation': True
    },
    'optimization': {
        'max_iterations': DEFAULT_OPTIMIZATION_ITERATIONS,
        'tolerance': OPTIMIZATION_TOLERANCE,
        'parallel': True,
        'n_jobs': DEFAULT_N_JOBS
    },
    'analysis': {
        'calculate_admet': True,
        'generate_visualizations': True,
        'export_results': True
    }
}
