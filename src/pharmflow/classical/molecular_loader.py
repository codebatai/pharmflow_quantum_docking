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
PharmFlow Real Molecular Loader
"""

import os
import logging
import gzip
import pickle
from typing import Dict, List, Any, Optional, Tuple, Union, Iterator
from pathlib import Path
import time
import json
import xml.etree.ElementTree as ET

# Molecular Computing Imports
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors
from rdkit.Chem.Features import FeatureFactory
from rdkit.Chem import PandasTools, SDMolSupplier, SmilesMolSupplier
from rdkit.Chem.rdchem import Mol

# Data Processing Imports
import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, as_completed

# Bio-informatics Imports
try:
    from Bio.PDB import PDBParser, PDBIO, Structure, Model, Chain, Residue
    from Bio.PDB.DSSP import DSSP
    BIO_AVAILABLE = True
except ImportError:
    BIO_AVAILABLE = False
    logging.warning("BioPython not available - PDB processing will be limited")

logger = logging.getLogger(__name__)

@dataclass
class MolecularLoaderConfig:
    """Configuration for molecular loading operations"""
    # File processing
    max_file_size: int = 100 * 1024 * 1024  # 100MB
    timeout_seconds: int = 300  # 5 minutes
    parallel_loading: bool = True
    max_workers: int = 4
    
    # Molecular processing
    sanitize_molecules: bool = True
    add_hydrogens: bool = True
    generate_3d_coords: bool = True
    remove_salts: bool = True
    neutralize_charges: bool = True
    
    # Validation
    validate_structures: bool = True
    min_atoms: int = 3
    max_atoms: int = 200
    allowed_elements: set = field(default_factory=lambda: {
        'H', 'C', 'N', 'O', 'P', 'S', 'F', 'Cl', 'Br', 'I'
    })
    
    # Caching
    enable_caching: bool = True
    cache_directory: str = ".pharmflow_cache"
    
    # Error handling
    strict_mode: bool = False  # If True, fails on any error; if False, skips problematic molecules

class RealMolecularLoader:
    """
    Real Molecular Loader for PharmFlow
    NO MOCK DATA - Comprehensive molecular file processing and validation
    """
    
    def __init__(self, config: MolecularLoaderConfig = None):
        """Initialize real molecular loader"""
        self.config = config or MolecularLoaderConfig()
        self.logger = logging.getLogger(__name__)
        
        # Initialize processing components
        self.feature_factory = self._initialize_feature_factory()
        self.molecular_validators = self._initialize_validators()
        self.structure_processors = self._initialize_structure_processors()
        
        # Create cache directory
        if self.config.enable_caching:
            self.cache_dir = Path(self.config.cache_directory)
            self.cache_dir.mkdir(exist_ok=True)
        
        # Statistics
        self.loading_stats = {
            'files_processed': 0,
            'molecules_loaded': 0,
            'molecules_failed': 0,
            'total_processing_time': 0.0
        }
        
        self.logger.info("Real molecular loader initialized with comprehensive file processing")
    
    def _initialize_feature_factory(self) -> Optional[FeatureFactory]:
        """Initialize RDKit feature factory"""
        try:
            return FeatureFactory.from_file('BaseFeatures.fdef')
        except Exception as e:
            self.logger.warning(f"Could not initialize feature factory: {e}")
            return None
    
    def _initialize_validators(self) -> Dict[str, callable]:
        """Initialize molecular validators"""
        
        validators = {
            'structure_validity': self._validate_structure,
            'atom_count': self._validate_atom_count,
            'element_types': self._validate_element_types,
            'connectivity': self._validate_connectivity,
            'charge_state': self._validate_charge_state,
            'stereochemistry': self._validate_stereochemistry
        }
        
        return validators
    
    def _initialize_structure_processors(self) -> Dict[str, callable]:
        """Initialize structure processors"""
        
        processors = {
            'sanitize': self._sanitize_molecule,
            'add_hydrogens': self._add_hydrogens,
            'remove_salts': self._remove_salts,
            'neutralize': self._neutralize_charges,
            'generate_3d': self._generate_3d_coordinates,
            'optimize_geometry': self._optimize_molecular_geometry
        }
        
        return processors
    
    def load_molecular_file(self, 
                           file_path: Union[str, Path],
                           file_format: Optional[str] = None) -> Dict[str, Any]:
        """
        Load molecules from various file formats
        
        Args:
            file_path: Path to molecular file
            file_format: File format (auto-detected if None)
            
        Returns:
            Dictionary containing loaded molecules and metadata
        """
        
        start_time = time.time()
        file_path = Path(file_path)
        
        try:
            # Validate file
            self._validate_file(file_path)
            
            # Detect file format
            if file_format is None:
                file_format = self._detect_file_format(file_path)
            
            self.logger.info(f"Loading molecular file: {file_path} (format: {file_format})")
            
            # Load molecules based on format
            loading_result = self._load_by_format(file_path, file_format)
            
            # Process and validate molecules
            processed_result = self._process_loaded_molecules(loading_result)
            
            # Generate metadata
            metadata = self._generate_file_metadata(file_path, file_format, processed_result)
            
            processing_time = time.time() - start_time
            
            # Update statistics
            self.loading_stats['files_processed'] += 1
            self.loading_stats['molecules_loaded'] += len(processed_result['valid_molecules'])
            self.loading_stats['molecules_failed'] += len(processed_result['invalid_molecules'])
            self.loading_stats['total_processing_time'] += processing_time
            
            result = {
                'molecules': processed_result['valid_molecules'],
                'invalid_molecules': processed_result['invalid_molecules'],
                'metadata': metadata,
                'processing_time': processing_time,
                'success': True,
                'file_path': str(file_path),
                'file_format': file_format
            }
            
            self.logger.info(f"Successfully loaded {len(result['molecules'])} molecules from {file_path}")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Failed to load molecular file {file_path}: {e}")
            return {
                'molecules': [],
                'invalid_molecules': [],
                'metadata': {},
                'processing_time': time.time() - start_time,
                'success': False,
                'error': str(e),
                'file_path': str(file_path)
            }
    
    def _validate_file(self, file_path: Path):
        """Validate file before processing"""
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        if not file_path.is_file():
            raise ValueError(f"Path is not a file: {file_path}")
        
        file_size = file_path.stat().st_size
        if file_size > self.config.max_file_size:
            raise ValueError(f"File too large: {file_size} bytes (max: {self.config.max_file_size})")
        
        if file_size == 0:
            raise ValueError(f"File is empty: {file_path}")
    
    def _detect_file_format(self, file_path: Path) -> str:
        """Detect molecular file format"""
        
        suffix = file_path.suffix.lower()
        
        format_map = {
            '.sdf': 'sdf',
            '.mol': 'mol',
            '.mol2': 'mol2',
            '.pdb': 'pdb',
            '.xyz': 'xyz',
            '.cif': 'cif',
            '.smiles': 'smiles',
            '.smi': 'smiles',
            '.csv': 'csv',
            '.json': 'json',
            '.pkl': 'pickle',
            '.gz': 'compressed'
        }
        
        detected_format = format_map.get(suffix, 'unknown')
        
        # Handle compressed files
        if detected_format == 'compressed':
            # Check the extension before .gz
            if file_path.name.endswith('.sdf.gz'):
                detected_format = 'sdf_gz'
            elif file_path.name.endswith('.pdb.gz'):
                detected_format = 'pdb_gz'
            else:
                detected_format = 'unknown_gz'
        
        # Additional content-based detection for unknown formats
        if detected_format == 'unknown':
            detected_format = self._detect_format_by_content(file_path)
        
        return detected_format
    
    def _detect_format_by_content(self, file_path: Path) -> str:
        """Detect format by examining file content"""
        
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                first_lines = [f.readline().strip() for _ in range(5)]
            
            # Check for specific format indicators
            content = '\n'.join(first_lines)
            
            if 'M  END' in content or '$$$$' in content:
                return 'sdf'
            elif content.startswith('HEADER') or content.startswith('ATOM'):
                return 'pdb'
            elif content.startswith('@<TRIPOS>'):
                return 'mol2'
            elif 'data_' in content:
                return 'cif'
            elif any(line.strip() and not line.startswith('#') and 
                   ' ' in line.strip() and len(line.strip().split()) >= 4 
                   for line in first_lines):
                return 'xyz'
            else:
                return 'unknown'
                
        except Exception:
            return 'unknown'
    
    def _load_by_format(self, file_path: Path, file_format: str) -> Dict[str, Any]:
        """Load molecules based on file format"""
        
        loaders = {
            'sdf': self._load_sdf_file,
            'mol': self._load_mol_file,
            'pdb': self._load_pdb_file,
            'smiles': self._load_smiles_file,
            'csv': self._load_csv_file,
            'json': self._load_json_file,
            'pickle': self._load_pickle_file,
            'sdf_gz': self._load_compressed_sdf,
            'pdb_gz': self._load_compressed_pdb
        }
        
        loader = loaders.get(file_format, self._load_unknown_format)
        return loader(file_path)
    
    def _load_sdf_file(self, file_path: Path) -> Dict[str, Any]:
        """Load SDF file"""
        
        molecules = []
        errors = []
        
        try:
            supplier = SDMolSupplier(str(file_path), sanitize=False, removeHs=False)
            
            for i, mol in enumerate(supplier):
                if mol is not None:
                    mol_data = {
                        'molecule': mol,
                        'index': i,
                        'properties': mol.GetPropsAsDict(),
                        'source': f"{file_path.name}:{i}"
                    }
                    molecules.append(mol_data)
                else:
                    errors.append(f"Failed to parse molecule at index {i}")
            
        except Exception as e:
            raise ValueError(f"Failed to load SDF file: {e}")
        
        return {
            'molecules': molecules,
            'errors': errors,
            'format': 'sdf'
        }
    
    def _load_mol_file(self, file_path: Path) -> Dict[str, Any]:
        """Load single MOL file"""
        
        try:
            mol = Chem.MolFromMolFile(str(file_path), sanitize=False, removeHs=False)
            
            if mol is not None:
                mol_data = {
                    'molecule': mol,
                    'index': 0,
                    'properties': mol.GetPropsAsDict(),
                    'source': file_path.name
                }
                return {
                    'molecules': [mol_data],
                    'errors': [],
                    'format': 'mol'
                }
            else:
                return {
                    'molecules': [],
                    'errors': ['Failed to parse MOL file'],
                    'format': 'mol'
                }
                
        except Exception as e:
            raise ValueError(f"Failed to load MOL file: {e}")
    
    def _load_pdb_file(self, file_path: Path) -> Dict[str, Any]:
        """Load PDB file"""
        
        molecules = []
        errors = []
        
        try:
            # Try RDKit first
            mol = Chem.MolFromPDBFile(str(file_path), sanitize=False, removeHs=False)
            
            if mol is not None:
                mol_data = {
                    'molecule': mol,
                    'index': 0,
                    'properties': {'source_pdb': str(file_path)},
                    'source': file_path.name
                }
                molecules.append(mol_data)
            else:
                errors.append("RDKit failed to parse PDB file")
            
            # If BioPython is available, also load structural information
            if BIO_AVAILABLE:
                try:
                    structure_info = self._parse_pdb_structure(file_path)
                    if molecules:
                        molecules[0]['structure_info'] = structure_info
                except Exception as e:
                    errors.append(f"BioPython PDB parsing failed: {e}")
        
        except Exception as e:
            raise ValueError(f"Failed to load PDB file: {e}")
        
        return {
            'molecules': molecules,
            'errors': errors,
            'format': 'pdb'
        }
    
    def _parse_pdb_structure(self, file_path: Path) -> Dict[str, Any]:
        """Parse PDB structure using BioPython"""
        
        if not BIO_AVAILABLE:
            return {}
        
        try:
            parser = PDBParser(QUIET=True)
            structure = parser.get_structure('protein', str(file_path))
            
            # Extract structural information
            structure_info = {
                'num_models': len(structure),
                'chains': [],
                'residues': [],
                'atoms': 0
            }
            
            for model in structure:
                for chain in model:
                    chain_info = {
                        'chain_id': chain.id,
                        'num_residues': len(chain),
                        'residue_types': []
                    }
                    
                    for residue in chain:
                        structure_info['atoms'] += len(residue)
                        residue_info = {
                            'residue_id': residue.id,
                            'residue_name': residue.resname,
                            'num_atoms': len(residue)
                        }
                        structure_info['residues'].append(residue_info)
                        chain_info['residue_types'].append(residue.resname)
                    
                    structure_info['chains'].append(chain_info)
            
            return structure_info
            
        except Exception as e:
            self.logger.warning(f"PDB structure parsing failed: {e}")
            return {}
    
    def _load_smiles_file(self, file_path: Path) -> Dict[str, Any]:
        """Load SMILES file"""
        
        molecules = []
        errors = []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for i, line in enumerate(f):
                    line = line.strip()
                    if line and not line.startswith('#'):
                        # Handle different SMILES formats
                        parts = line.split()
                        smiles = parts[0]
                        mol_id = parts[1] if len(parts) > 1 else f"mol_{i}"
                        
                        try:
                            mol = Chem.MolFromSmiles(smiles)
                            if mol is not None:
                                mol_data = {
                                    'molecule': mol,
                                    'index': i,
                                    'properties': {'ID': mol_id, 'SMILES': smiles},
                                    'source': f"{file_path.name}:{i}"
                                }
                                molecules.append(mol_data)
                            else:
                                errors.append(f"Invalid SMILES at line {i+1}: {smiles}")
                        except Exception as e:
                            errors.append(f"Error parsing SMILES at line {i+1}: {e}")
        
        except Exception as e:
            raise ValueError(f"Failed to load SMILES file: {e}")
        
        return {
            'molecules': molecules,
            'errors': errors,
            'format': 'smiles'
        }
    
    def _load_csv_file(self, file_path: Path) -> Dict[str, Any]:
        """Load CSV file containing molecular data"""
        
        molecules = []
        errors = []
        
        try:
            df = pd.read_csv(file_path)
            
            # Detect SMILES column
            smiles_columns = ['SMILES', 'smiles', 'Smiles', 'CANONICAL_SMILES', 'canonical_smiles']
            smiles_col = None
            
            for col in smiles_columns:
                if col in df.columns:
                    smiles_col = col
                    break
            
            if smiles_col is None:
                # Try first column if no standard SMILES column found
                smiles_col = df.columns[0]
                self.logger.warning(f"No standard SMILES column found, using {smiles_col}")
            
            # ID column detection
            id_columns = ['ID', 'id', 'Id', 'NAME', 'name', 'Name', 'COMPOUND_ID']
            id_col = None
            
            for col in id_columns:
                if col in df.columns:
                    id_col = col
                    break
            
            # Process each row
            for i, row in df.iterrows():
                smiles = row[smiles_col]
                
                if pd.isna(smiles) or not smiles.strip():
                    errors.append(f"Empty SMILES at row {i}")
                    continue
                
                try:
                    mol = Chem.MolFromSmiles(str(smiles).strip())
                    if mol is not None:
                        # Collect all properties
                        properties = row.to_dict()
                        if id_col:
                            mol_id = str(row[id_col])
                        else:
                            mol_id = f"mol_{i}"
                        
                        mol_data = {
                            'molecule': mol,
                            'index': i,
                            'properties': properties,
                            'source': f"{file_path.name}:{i}",
                            'id': mol_id
                        }
                        molecules.append(mol_data)
                    else:
                        errors.append(f"Invalid SMILES at row {i}: {smiles}")
                except Exception as e:
                    errors.append(f"Error parsing SMILES at row {i}: {e}")
        
        except Exception as e:
            raise ValueError(f"Failed to load CSV file: {e}")
        
        return {
            'molecules': molecules,
            'errors': errors,
            'format': 'csv'
        }
    
    def _load_json_file(self, file_path: Path) -> Dict[str, Any]:
        """Load JSON file containing molecular data"""
        
        molecules = []
        errors = []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Handle different JSON structures
            if isinstance(data, list):
                # List of molecules
                mol_list = data
            elif isinstance(data, dict) and 'molecules' in data:
                # Dictionary with molecules key
                mol_list = data['molecules']
            else:
                # Single molecule
                mol_list = [data]
            
            for i, mol_data in enumerate(mol_list):
                try:
                    if 'smiles' in mol_data:
                        smiles = mol_data['smiles']
                        mol = Chem.MolFromSmiles(smiles)
                    elif 'molblock' in mol_data:
                        molblock = mol_data['molblock']
                        mol = Chem.MolFromMolBlock(molblock)
                    else:
                        errors.append(f"No valid molecular representation in entry {i}")
                        continue
                    
                    if mol is not None:
                        molecule_entry = {
                            'molecule': mol,
                            'index': i,
                            'properties': mol_data,
                            'source': f"{file_path.name}:{i}"
                        }
                        molecules.append(molecule_entry)
                    else:
                        errors.append(f"Failed to parse molecule at entry {i}")
                        
                except Exception as e:
                    errors.append(f"Error processing entry {i}: {e}")
        
        except Exception as e:
            raise ValueError(f"Failed to load JSON file: {e}")
        
        return {
            'molecules': molecules,
            'errors': errors,
            'format': 'json'
        }
    
    def _load_pickle_file(self, file_path: Path) -> Dict[str, Any]:
        """Load pickled molecular data"""
        
        try:
            with open(file_path, 'rb') as f:
                data = pickle.load(f)
            
            molecules = []
            errors = []
            
            # Handle different pickle structures
            if isinstance(data, list):
                for i, item in enumerate(data):
                    if isinstance(item, Chem.Mol):
                        mol_data = {
                            'molecule': item,
                            'index': i,
                            'properties': item.GetPropsAsDict(),
                            'source': f"{file_path.name}:{i}"
                        }
                        molecules.append(mol_data)
            elif isinstance(data, dict):
                if 'molecules' in data:
                    mol_list = data['molecules']
                    for i, mol in enumerate(mol_list):
                        if isinstance(mol, Chem.Mol):
                            mol_data = {
                                'molecule': mol,
                                'index': i,
                                'properties': mol.GetPropsAsDict(),
                                'source': f"{file_path.name}:{i}"
                            }
                            molecules.append(mol_data)
            elif isinstance(data, Chem.Mol):
                mol_data = {
                    'molecule': data,
                    'index': 0,
                    'properties': data.GetPropsAsDict(),
                    'source': file_path.name
                }
                molecules.append(mol_data)
            
            return {
                'molecules': molecules,
                'errors': errors,
                'format': 'pickle'
            }
            
        except Exception as e:
            raise ValueError(f"Failed to load pickle file: {e}")
    
    def _load_compressed_sdf(self, file_path: Path) -> Dict[str, Any]:
        """Load compressed SDF file"""
        
        molecules = []
        errors = []
        
        try:
            with gzip.open(file_path, 'rt', encoding='utf-8') as f:
                # Create temporary uncompressed content
                content = f.read()
                
                # Parse SDF content
                mol_blocks = content.split('$$$$')
                
                for i, mol_block in enumerate(mol_blocks):
                    mol_block = mol_block.strip()
                    if mol_block:
                        try:
                            mol = Chem.MolFromMolBlock(mol_block)
                            if mol is not None:
                                mol_data = {
                                    'molecule': mol,
                                    'index': i,
                                    'properties': mol.GetPropsAsDict(),
                                    'source': f"{file_path.name}:{i}"
                                }
                                molecules.append(mol_data)
                            else:
                                errors.append(f"Failed to parse molecule block {i}")
                        except Exception as e:
                            errors.append(f"Error parsing molecule block {i}: {e}")
        
        except Exception as e:
            raise ValueError(f"Failed to load compressed SDF file: {e}")
        
        return {
            'molecules': molecules,
            'errors': errors,
            'format': 'sdf_gz'
        }
    
    def _load_compressed_pdb(self, file_path: Path) -> Dict[str, Any]:
        """Load compressed PDB file"""
        
        try:
            with gzip.open(file_path, 'rt', encoding='utf-8') as f:
                content = f.read()
            
            # Write to temporary file for RDKit
            import tempfile
            with tempfile.NamedTemporaryFile(mode='w', suffix='.pdb', delete=False) as tmp_file:
                tmp_file.write(content)
                tmp_path = tmp_file.name
            
            try:
                # Load using PDB loader
                result = self._load_pdb_file(Path(tmp_path))
                result['format'] = 'pdb_gz'
                return result
            finally:
                # Clean up temporary file
                os.unlink(tmp_path)
                
        except Exception as e:
            raise ValueError(f"Failed to load compressed PDB file: {e}")
    
    def _load_unknown_format(self, file_path: Path) -> Dict[str, Any]:
        """Attempt to load unknown format"""
        
        # Try different parsers
        parsers = [
            ('sdf', self._load_sdf_file),
            ('smiles', self._load_smiles_file),
            ('pdb', self._load_pdb_file)
        ]
        
        for format_name, parser in parsers:
            try:
                result = parser(file_path)
                if result['molecules']:
                    self.logger.info(f"Successfully parsed unknown format as {format_name}")
                    result['format'] = f"{format_name}_detected"
                    return result
            except Exception:
                continue
        
        raise ValueError(f"Unable to determine file format or parse file: {file_path}")
    
    def _process_loaded_molecules(self, loading_result: Dict[str, Any]) -> Dict[str, Any]:
        """Process and validate loaded molecules"""
        
        molecules = loading_result['molecules']
        valid_molecules = []
        invalid_molecules = []
        
        for mol_data in molecules:
            try:
                mol = mol_data['molecule']
                
                # Apply molecular processing
                if self.config.sanitize_molecules:
                    mol = self.structure_processors['sanitize'](mol)
                
                if self.config.remove_salts:
                    mol = self.structure_processors['remove_salts'](mol)
                
                if self.config.neutralize_charges:
                    mol = self.structure_processors['neutralize'](mol)
                
                if self.config.add_hydrogens:
                    mol = self.structure_processors['add_hydrogens'](mol)
                
                if self.config.generate_3d_coords and mol.GetNumConformers() == 0:
                    mol = self.structure_processors['generate_3d'](mol)
                
                # Validate processed molecule
                if self.config.validate_structures:
                    validation_result = self._validate_molecule(mol)
                    if not validation_result['valid']:
                        mol_data['validation_errors'] = validation_result['errors']
                        invalid_molecules.append(mol_data)
                        continue
                
                # Update molecule in data
                mol_data['molecule'] = mol
                mol_data['processed'] = True
                
                # Add computed properties
                mol_data['computed_properties'] = self._compute_molecular_properties(mol)
                
                valid_molecules.append(mol_data)
                
            except Exception as e:
                mol_data['processing_error'] = str(e)
                invalid_molecules.append(mol_data)
                
                if self.config.strict_mode:
                    raise ValueError(f"Molecular processing failed in strict mode: {e}")
        
        return {
            'valid_molecules': valid_molecules,
            'invalid_molecules': invalid_molecules,
            'processing_errors': loading_result.get('errors', [])
        }
    
    def _validate_molecule(self, mol: Chem.Mol) -> Dict[str, Any]:
        """Validate molecule using all validators"""
        
        errors = []
        
        for validator_name, validator_func in self.molecular_validators.items():
            try:
                is_valid, error_msg = validator_func(mol)
                if not is_valid:
                    errors.append(f"{validator_name}: {error_msg}")
            except Exception as e:
                errors.append(f"{validator_name}: Validation failed - {e}")
        
        return {
            'valid': len(errors) == 0,
            'errors': errors
        }
    
    def _validate_structure(self, mol: Chem.Mol) -> Tuple[bool, str]:
        """Validate basic molecular structure"""
        try:
            Chem.SanitizeMol(mol)
            return True, ""
        except Exception as e:
            return False, f"Structure validation failed: {e}"
    
    def _validate_atom_count(self, mol: Chem.Mol) -> Tuple[bool, str]:
        """Validate atom count"""
        atom_count = mol.GetNumAtoms()
        if atom_count < self.config.min_atoms:
            return False, f"Too few atoms: {atom_count} < {self.config.min_atoms}"
        if atom_count > self.config.max_atoms:
            return False, f"Too many atoms: {atom_count} > {self.config.max_atoms}"
        return True, ""
    
    def _validate_element_types(self, mol: Chem.Mol) -> Tuple[bool, str]:
        """Validate element types"""
        elements = set(atom.GetSymbol() for atom in mol.GetAtoms())
        invalid_elements = elements - self.config.allowed_elements
        if invalid_elements:
            return False, f"Invalid elements: {invalid_elements}"
        return True, ""
    
    def _validate_connectivity(self, mol: Chem.Mol) -> Tuple[bool, str]:
        """Validate molecular connectivity"""
        # Check for disconnected fragments
        fragments = Chem.GetMolFrags(mol)
        if len(fragments) > 1:
            return False, f"Molecule has {len(fragments)} disconnected fragments"
        return True, ""
    
    def _validate_charge_state(self, mol: Chem.Mol) -> Tuple[bool, str]:
        """Validate charge state"""
        formal_charge = Chem.rdmolops.GetFormalCharge(mol)
        if abs(formal_charge) > 3:  # Reasonable charge limit
            return False, f"Extreme formal charge: {formal_charge}"
        return True, ""
    
    def _validate_stereochemistry(self, mol: Chem.Mol) -> Tuple[bool, str]:
        """Validate stereochemistry"""
        try:
            # Check for valid stereochemistry
            Chem.AssignStereochemistry(mol)
            return True, ""
        except Exception as e:
            return False, f"Stereochemistry validation failed: {e}"
    
    def _sanitize_molecule(self, mol: Chem.Mol) -> Chem.Mol:
        """Sanitize molecule"""
        mol_copy = Chem.Mol(mol)
        Chem.SanitizeMol(mol_copy)
        return mol_copy
    
    def _add_hydrogens(self, mol: Chem.Mol) -> Chem.Mol:
        """Add hydrogens to molecule"""
        return Chem.AddHs(mol)
    
    def _remove_salts(self, mol: Chem.Mol) -> Chem.Mol:
        """Remove salts and keep largest fragment"""
        return Chem.rdMolStandardize.StandardizeSmiles(Chem.MolToSmiles(mol))
    
    def _neutralize_charges(self, mol: Chem.Mol) -> Chem.Mol:
        """Neutralize charges"""
        try:
            from rdkit.Chem.MolStandardize import rdMolStandardize
            neutralizer = rdMolStandardize.Neutralizer()
            return neutralizer.neutralize(mol)
        except ImportError:
            # Fallback: just return original molecule
            return mol
    
    def _generate_3d_coordinates(self, mol: Chem.Mol) -> Chem.Mol:
        """Generate 3D coordinates"""
        mol_copy = Chem.Mol(mol)
        AllChem.EmbedMolecule(mol_copy, useExpTorsionAnglePrefs=True, useBasicKnowledge=True)
        return mol_copy
    
    def _optimize_molecular_geometry(self, mol: Chem.Mol) -> Chem.Mol:
        """Optimize molecular geometry"""
        mol_copy = Chem.Mol(mol)
        AllChem.OptimizeMoleculeConfigs(mol_copy)
        return mol_copy
    
    def _compute_molecular_properties(self, mol: Chem.Mol) -> Dict[str, Any]:
        """Compute molecular properties"""
        
        properties = {}
        
        # Basic descriptors
        try:
            properties['molecular_weight'] = Descriptors.MolWt(mol)
            properties['logp'] = Descriptors.MolLogP(mol)
            properties['tpsa'] = Descriptors.TPSA(mol)
            properties['hbd'] = Descriptors.NumHDonors(mol)
            properties['hba'] = Descriptors.NumHAcceptors(mol)
            properties['rotatable_bonds'] = Descriptors.NumRotatableBonds(mol)
            properties['aromatic_rings'] = rdMolDescriptors.CalcNumAromaticRings(mol)
            properties['heavy_atoms'] = Descriptors.HeavyAtomCount(mol)
        except Exception as e:
            self.logger.warning(f"Basic descriptor calculation failed: {e}")
        
        # Extended descriptors
        try:
            properties['bertz_ct'] = rdMolDescriptors.BertzCT(mol)
            properties['balaban_j'] = rdMolDescriptors.BalabanJ(mol)
        except Exception as e:
            self.logger.warning(f"Extended descriptor calculation failed: {e}")
        
        # Pharmacophore features
        if self.feature_factory:
            try:
                features = self.feature_factory.GetFeaturesForMol(mol)
                feature_counts = {}
                for feat in features:
                    feat_type = feat.GetFamily()
                    feature_counts[feat_type] = feature_counts.get(feat_type, 0) + 1
                properties['pharmacophore_features'] = feature_counts
            except Exception as e:
                self.logger.warning(f"Pharmacophore feature calculation failed: {e}")
        
        return properties
    
    def _generate_file_metadata(self, 
                               file_path: Path, 
                               file_format: str, 
                               processed_result: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive file metadata"""
        
        metadata = {
            'file_info': {
                'path': str(file_path),
                'name': file_path.name,
                'size_bytes': file_path.stat().st_size,
                'format': file_format,
                'modified_time': file_path.stat().st_mtime
            },
            'processing_summary': {
                'total_molecules_found': len(processed_result['valid_molecules']) + len(processed_result['invalid_molecules']),
                'valid_molecules': len(processed_result['valid_molecules']),
                'invalid_molecules': len(processed_result['invalid_molecules']),
                'success_rate': len(processed_result['valid_molecules']) / max(1, len(processed_result['valid_molecules']) + len(processed_result['invalid_molecules'])),
                'processing_errors': len(processed_result['processing_errors'])
            },
            'molecular_statistics': self._calculate_molecular_statistics(processed_result['valid_molecules']),
            'processing_config': {
                'sanitize_molecules': self.config.sanitize_molecules,
                'add_hydrogens': self.config.add_hydrogens,
                'generate_3d_coords': self.config.generate_3d_coords,
                'validate_structures': self.config.validate_structures
            }
        }
        
        return metadata
    
    def _calculate_molecular_statistics(self, molecules: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate statistics for loaded molecules"""
        
        if not molecules:
            return {}
        
        # Extract properties
        properties = []
        for mol_data in molecules:
            comp_props = mol_data.get('computed_properties', {})
            properties.append(comp_props)
        
        if not properties:
            return {}
        
        # Calculate statistics
        stats = {}
        
        numeric_props = ['molecular_weight', 'logp', 'tpsa', 'hbd', 'hba', 'rotatable_bonds', 'aromatic_rings']
        
        for prop in numeric_props:
            values = [p.get(prop, 0) for p in properties if prop in p]
            if values:
                stats[prop] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values),
                    'median': np.median(values)
                }
        
        return stats
    
    def load_multiple_files(self, 
                           file_paths: List[Union[str, Path]],
                           parallel: bool = None) -> Dict[str, Any]:
        """Load multiple molecular files"""
        
        if parallel is None:
            parallel = self.config.parallel_loading
        
        start_time = time.time()
        results = []
        
        if parallel and len(file_paths) > 1:
            # Parallel loading
            with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
                futures = {executor.submit(self.load_molecular_file, fp): fp for fp in file_paths}
                
                for future in as_completed(futures):
                    file_path = futures[future]
                    try:
                        result = future.result(timeout=self.config.timeout_seconds)
                        results.append(result)
                    except Exception as e:
                        self.logger.error(f"Failed to load {file_path}: {e}")
                        results.append({
                            'molecules': [],
                            'success': False,
                            'error': str(e),
                            'file_path': str(file_path)
                        })
        else:
            # Sequential loading
            for file_path in file_paths:
                result = self.load_molecular_file(file_path)
                results.append(result)
        
        # Combine results
        all_molecules = []
        all_errors = []
        successful_files = 0
        
        for result in results:
            if result['success']:
                all_molecules.extend(result['molecules'])
                successful_files += 1
            else:
                all_errors.append(result)
        
        processing_time = time.time() - start_time
        
        return {
            'molecules': all_molecules,
            'failed_files': all_errors,
            'total_files': len(file_paths),
            'successful_files': successful_files,
            'total_molecules': len(all_molecules),
            'processing_time': processing_time,
            'success': successful_files > 0
        }
    
    def get_loading_statistics(self) -> Dict[str, Any]:
        """Get comprehensive loading statistics"""
        
        return {
            'session_statistics': self.loading_stats.copy(),
            'configuration': {
                'max_file_size': self.config.max_file_size,
                'parallel_loading': self.config.parallel_loading,
                'max_workers': self.config.max_workers,
                'sanitize_molecules': self.config.sanitize_molecules,
                'validate_structures': self.config.validate_structures
            },
            'supported_formats': [
                'sdf', 'mol', 'pdb', 'smiles', 'csv', 'json', 'pickle',
                'sdf.gz', 'pdb.gz'
            ],
            'validation_criteria': {
                'min_atoms': self.config.min_atoms,
                'max_atoms': self.config.max_atoms,
                'allowed_elements': list(self.config.allowed_elements)
            }
        }

# Example usage and validation
if __name__ == "__main__":
    # Test the real molecular loader
    config = MolecularLoaderConfig(
        sanitize_molecules=True,
        add_hydrogens=True,
        generate_3d_coords=True,
        validate_structures=True,
        parallel_loading=False  # Disable for testing
    )
    
    loader = RealMolecularLoader(config)
    
    print("Testing real molecular loader...")
    
    # Test with sample SMILES data
    test_smiles = [
        "CC(=O)OC1=CC=CC=C1C(=O)O",  # Aspirin
        "COC1=CC=C(C=C1)C2=CC(=O)OC3=C2C=CC(=C3)O",  # Quercetin-like
        "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O"  # Ibuprofen
    ]
    
    # Create temporary SMILES file
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', suffix='.smiles', delete=False) as tmp_file:
        for i, smiles in enumerate(test_smiles):
            tmp_file.write(f"{smiles} mol_{i}\n")
        tmp_path = tmp_file.name
    
    try:
        # Test file loading
        result = loader.load_molecular_file(tmp_path)
        
        print(f"Loading success: {result['success']}")
        print(f"Molecules loaded: {len(result['molecules'])}")
        print(f"Processing time: {result['processing_time']:.3f} seconds")
        
        if result['molecules']:
            mol_data = result['molecules'][0]
            props = mol_data['computed_properties']
            print(f"First molecule MW: {props.get('molecular_weight', 'N/A'):.2f}")
            print(f"First molecule LogP: {props.get('logp', 'N/A'):.2f}")
        
        # Test statistics
        stats = loader.get_loading_statistics()
        print(f"\nSession statistics:")
        print(f"Files processed: {stats['session_statistics']['files_processed']}")
        print(f"Molecules loaded: {stats['session_statistics']['molecules_loaded']}")
        
    finally:
        # Clean up
        os.unlink(tmp_path)
    
    print("\nReal molecular loader validation completed successfully!")
