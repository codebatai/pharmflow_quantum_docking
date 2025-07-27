"""
Molecular structure loading and processing utilities
Supports PDB, SDF, MOL2, and SMILES formats with comprehensive validation
"""

import os
import gzip
import logging
from typing import Dict, List, Any, Optional, Union, Tuple
from pathlib import Path
import numpy as np

from rdkit import Chem
from rdkit.Chem import AllChem, rdMolDescriptors, Descriptors
from rdkit.Chem.rdForceFieldHelpers import UFFOptimizeMolecule, MMFFOptimizeMolecule
from Bio.PDB import PDBParser, Structure, Model, Chain, Residue
from Bio.PDB.PDBIO import PDBIO
import warnings

from ..utils.constants import ATOMIC_MASSES, VDW_RADII, SUPPORTED_FORMATS

logger = logging.getLogger(__name__)

class MolecularLoader:
    """
    Comprehensive molecular structure loader supporting multiple formats
    """
    
    def __init__(self, validate_structures: bool = True):
        """
        Initialize molecular loader
        
        Args:
            validate_structures: Whether to validate loaded structures
        """
        self.validate_structures = validate_structures
        self.logger = logging.getLogger(__name__)
        
        # Initialize parsers
        self.pdb_parser = PDBParser(QUIET=True)
        
        # Structure cache
        self._structure_cache = {}
        
        # Supported file formats
        self.supported_formats = SUPPORTED_FORMATS
        
        self.logger.info("Molecular loader initialized")
    
    def load_protein(self, file_path: str, chain_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Load protein structure from PDB file
        
        Args:
            file_path: Path to PDB file
            chain_id: Specific chain to load (None for all chains)
            
        Returns:
            Protein structure dictionary
        """
        try:
            # Check file existence and format
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"PDB file not found: {file_path}")
            
            file_path = str(Path(file_path).resolve())
            
            # Check cache
            cache_key = f"{file_path}_{chain_id}"
            if cache_key in self._structure_cache:
                self.logger.debug(f"Loading protein from cache: {file_path}")
                return self._structure_cache[cache_key]
            
            # Load PDB structure
            structure_id = Path(file_path).stem
            
            # Handle compressed files
            if file_path.endswith('.gz'):
                with gzip.open(file_path, 'rt') as f:
                    pdb_content = f.read()
                # Write temporary uncompressed file
                temp_path = file_path.replace('.gz', '.tmp')
                with open(temp_path, 'w') as f:
                    f.write(pdb_content)
                structure = self.pdb_parser.get_structure(structure_id, temp_path)
                os.remove(temp_path)
            else:
                structure = self.pdb_parser.get_structure(structure_id, file_path)
            
            # Extract protein information
            protein_data = self._extract_protein_data(structure, chain_id)
            protein_data['file_path'] = file_path
            protein_data['structure_id'] = structure_id
            
            # Validate structure if requested
            if self.validate_structures:
                self._validate_protein_structure(protein_data)
            
            # Cache result
            self._structure_cache[cache_key] = protein_data
            
            self.logger.info(f"Successfully loaded protein: {file_path}")
            return protein_data
            
        except Exception as e:
            self.logger.error(f"Failed to load protein {file_path}: {e}")
            raise ValueError(f"Protein loading error: {e}")
    
    def load_ligand(self, file_path: str, mol_id: Optional[str] = None) -> Chem.Mol:
        """
        Load ligand molecule from various formats
        
        Args:
            file_path: Path to ligand file or SMILES string
            mol_id: Molecule identifier
            
        Returns:
            RDKit molecule object
        """
        try:
            # Determine if input is file path or SMILES string
            if os.path.exists(file_path):
                mol = self._load_ligand_from_file(file_path)
            else:
                # Assume it's a SMILES string
                mol = self._load_ligand_from_smiles(file_path)
            
            if mol is None:
                raise ValueError(f"Could not load ligand from: {file_path}")
            
            # Set molecule identifier
            if mol_id:
                mol.SetProp("_Name", mol_id)
            
            # Add hydrogens if not present
            mol = Chem.AddHs(mol)
            
            # Generate 3D coordinates if needed
            if mol.GetNumConformers() == 0:
                self._generate_3d_coordinates(mol)
            
            # Validate structure if requested
            if self.validate_structures:
                self._validate_ligand_structure(mol)
            
            self.logger.info(f"Successfully loaded ligand: {file_path}")
            return mol
            
        except Exception as e:
            self.logger.error(f"Failed to load ligand {file_path}: {e}")
            raise ValueError(f"Ligand loading error: {e}")
    
    def load_multiple_ligands(self, file_paths: List[str]) -> List[Chem.Mol]:
        """
        Load multiple ligands from file list
        
        Args:
            file_paths: List of ligand file paths
            
        Returns:
            List of RDKit molecule objects
        """
        molecules = []
        
        for i, file_path in enumerate(file_paths):
            try:
                mol = self.load_ligand(file_path, mol_id=f"ligand_{i}")
                molecules.append(mol)
            except Exception as e:
                self.logger.warning(f"Failed to load ligand {file_path}: {e}")
        
        self.logger.info(f"Loaded {len(molecules)} out of {len(file_paths)} ligands")
        return molecules
    
    def _extract_protein_data(self, structure: Structure, chain_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Extract comprehensive protein data from Bio.PDB structure
        
        Args:
            structure: Bio.PDB structure object
            chain_id: Specific chain ID to extract
            
        Returns:
            Protein data dictionary
        """
        protein_data = {
            'structure': structure,
            'chains': {},
            'atoms': [],
            'coordinates': [],
            'residues': [],
            'secondary_structure': {},
            'binding_sites': [],
            'metadata': {}
        }
        
        for model in structure:
            for chain in model:
                if chain_id is None or chain.id == chain_id:
                    chain_data = self._extract_chain_data(chain)
                    protein_data['chains'][chain.id] = chain_data
                    
                    # Accumulate atoms and coordinates
                    protein_data['atoms'].extend(chain_data['atoms'])
                    protein_data['coordinates'].extend(chain_data['coordinates'])
                    protein_data['residues'].extend(chain_data['residues'])
        
        # Convert coordinates to numpy array
        if protein_data['coordinates']:
            protein_data['coordinates'] = np.array(protein_data['coordinates'])
        
        # Extract metadata
        protein_data['metadata'] = self._extract_metadata(structure)
        
        return protein_data
    
    def _extract_chain_data(self, chain: Chain) -> Dict[str, Any]:
        """Extract data from a protein chain"""
        chain_data = {
            'chain_id': chain.id,
            'atoms': [],
            'coordinates': [],
            'residues': [],
            'sequence': ''
        }
        
        for residue in chain:
            if residue.id[0] == ' ':  # Standard amino acid residues
                residue_data = self._extract_residue_data(residue)
                chain_data['residues'].append(residue_data)
                chain_data['sequence'] += self._get_residue_single_letter(residue.resname)
                
                # Extract atomic data
                for atom in residue:
                    atom_data = {
                        'name': atom.name,
                        'element': atom.element,
                        'coord': atom.coord,
                        'bfactor': atom.bfactor,
                        'occupancy': atom.occupancy,
                        'residue': residue.resname,
                        'residue_id': residue.id[1]
                    }
                    chain_data['atoms'].append(atom_data)
                    chain_data['coordinates'].append(atom.coord)
        
        return chain_data
    
    def _extract_residue_data(self, residue: Residue) -> Dict[str, Any]:
        """Extract data from a protein residue"""
        return {
            'name': residue.resname,
            'id': residue.id[1],
            'insertion_code': residue.id[2],
            'single_letter': self._get_residue_single_letter(residue.resname),
            'atoms': [atom.name for atom in residue],
            'center': self._calculate_residue_center(residue)
        }
    
    def _calculate_residue_center(self, residue: Residue) -> np.ndarray:
        """Calculate geometric center of residue"""
        coords = [atom.coord for atom in residue]
        return np.mean(coords, axis=0) if coords else np.zeros(3)
    
    def _get_residue_single_letter(self, resname: str) -> str:
        """Convert three-letter residue name to single letter"""
        aa_dict = {
            'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C',
            'GLN': 'Q', 'GLU': 'E', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I',
            'LEU': 'L', 'LYS': 'K', 'MET': 'M', 'PHE': 'F', 'PRO': 'P',
            'SER': 'S', 'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V'
        }
        return aa_dict.get(resname, 'X')
    
    def _extract_metadata(self, structure: Structure) -> Dict[str, Any]:
        """Extract metadata from PDB structure"""
        header = structure.header
        return {
            'resolution': header.get('resolution'),
            'structure_method': header.get('structure_method'),
            'deposition_date': header.get('deposition_date'),
            'release_date': header.get('release_date'),
            'name': header.get('name'),
            'head': header.get('head'),
            'idcode': header.get('idcode')
        }
    
    def _load_ligand_from_file(self, file_path: str) -> Optional[Chem.Mol]:
        """Load ligand from file based on extension"""
        file_ext = Path(file_path).suffix.lower()
        
        if file_ext == '.sdf':
            return self._load_from_sdf(file_path)
        elif file_ext == '.mol':
            return self._load_from_mol(file_path)
        elif file_ext == '.mol2':
            return self._load_from_mol2(file_path)
        elif file_ext == '.pdb':
            return self._load_ligand_from_pdb(file_path)
        elif file_ext in ['.smi', '.smiles']:
            return self._load_from_smiles_file(file_path)
        else:
            raise ValueError(f"Unsupported file format: {file_ext}")
    
    def _load_from_sdf(self, file_path: str) -> Optional[Chem.Mol]:
        """Load molecule from SDF file"""
        try:
            suppl = Chem.SDMolSupplier(file_path)
            mol = next(suppl)
            return mol
        except Exception as e:
            self.logger.warning(f"SDF loading failed: {e}")
            return None
    
    def _load_from_mol(self, file_path: str) -> Optional[Chem.Mol]:
        """Load molecule from MOL file"""
        try:
            mol = Chem.MolFromMolFile(file_path)
            return mol
        except Exception as e:
            self.logger.warning(f"MOL loading failed: {e}")
            return None
    
    def _load_from_mol2(self, file_path: str) -> Optional[Chem.Mol]:
        """Load molecule from MOL2 file"""
        try:
            mol = Chem.MolFromMol2File(file_path)
            return mol
        except Exception as e:
            self.logger.warning(f"MOL2 loading failed: {e}")
            return None
    
    def _load_ligand_from_pdb(self, file_path: str) -> Optional[Chem.Mol]:
        """Load ligand from PDB file"""
        try:
            mol = Chem.MolFromPDBFile(file_path)
            return mol
        except Exception as e:
            self.logger.warning(f"PDB ligand loading failed: {e}")
            return None
    
    def _load_from_smiles_file(self, file_path: str) -> Optional[Chem.Mol]:
        """Load molecule from SMILES file"""
        try:
            with open(file_path, 'r') as f:
                smiles = f.readline().strip()
            return self._load_ligand_from_smiles(smiles)
        except Exception as e:
            self.logger.warning(f"SMILES file loading failed: {e}")
            return None
    
    def _load_ligand_from_smiles(self, smiles: str) -> Optional[Chem.Mol]:
        """Load ligand from SMILES string"""
        try:
            mol = Chem.MolFromSmiles(smiles)
            return mol
        except Exception as e:
            self.logger.warning(f"SMILES parsing failed: {e}")
            return None
    
    def _generate_3d_coordinates(self, mol: Chem.Mol) -> None:
        """Generate 3D coordinates for molecule"""
        try:
            # Embed molecule in 3D
            AllChem.EmbedMolecule(mol, randomSeed=42, useExpTorsionAnglePrefs=True)
            
            # Optimize geometry
            try:
                # Try MMFF94 first
                if MMFFOptimizeMolecule(mol) != 0:
                    # Fall back to UFF if MMFF94 fails
                    UFFOptimizeMolecule(mol)
            except:
                self.logger.warning("Force field optimization failed")
                
        except Exception as e:
            self.logger.warning(f"3D coordinate generation failed: {e}")
    
    def _validate_protein_structure(self, protein_data: Dict[str, Any]) -> None:
        """Validate protein structure quality"""
        if not protein_data['atoms']:
            raise ValueError("Protein structure contains no atoms")
        
        if not protein_data['residues']:
            raise ValueError("Protein structure contains no residues")
        
        # Check for minimum number of atoms
        if len(protein_data['atoms']) < 10:
            self.logger.warning("Protein structure has very few atoms")
        
        # Check coordinate validity
        coords = protein_data['coordinates']
        if len(coords) > 0:
            if np.any(np.isnan(coords)) or np.any(np.isinf(coords)):
                raise ValueError("Protein structure contains invalid coordinates")
    
    def _validate_ligand_structure(self, mol: Chem.Mol) -> None:
        """Validate ligand structure quality"""
        if mol is None:
            raise ValueError("Ligand molecule is None")
        
        if mol.GetNumAtoms() == 0:
            raise ValueError("Ligand molecule contains no atoms")
        
        if mol.GetNumAtoms() > 200:
            self.logger.warning("Ligand molecule is unusually large (>200 atoms)")
        
        # Check for valid valences
        try:
            Chem.SanitizeMol(mol)
        except:
            raise ValueError("Ligand molecule has invalid valences")
        
        # Check molecular properties
        mw = Descriptors.MolWt(mol)
        if mw > 1000:  # Da
            self.logger.warning(f"Ligand molecular weight is high: {mw:.1f} Da")
    
    def get_binding_site_residues(self, 
                                 protein_data: Dict[str, Any], 
                                 ligand_center: np.ndarray,
                                 radius: float = 8.0) -> List[Dict[str, Any]]:
        """
        Find binding site residues within radius of ligand center
        
        Args:
            protein_data: Protein structure data
            ligand_center: Center coordinates of ligand
            radius: Search radius in Angstroms
            
        Returns:
            List of binding site residues
        """
        binding_site_residues = []
        
        for residue in protein_data['residues']:
            residue_center = residue['center']
            distance = np.linalg.norm(residue_center - ligand_center)
            
            if distance <= radius:
                residue_copy = residue.copy()
                residue_copy['distance_to_ligand'] = distance
                binding_site_residues.append(residue_copy)
        
        # Sort by distance
        binding_site_residues.sort(key=lambda x: x['distance_to_ligand'])
        
        return binding_site_residues
    
    def calculate_protein_properties(self, protein_data: Dict[str, Any]) -> Dict[str, float]:
        """
        Calculate basic protein properties
        
        Args:
            protein_data: Protein structure data
            
        Returns:
            Dictionary of protein properties
        """
        properties = {
            'num_atoms': len(protein_data['atoms']),
            'num_residues': len(protein_data['residues']),
            'num_chains': len(protein_data['chains']),
            'molecular_weight': 0.0,
            'center_of_mass': np.zeros(3)
        }
        
        # Calculate molecular weight and center of mass
        total_mass = 0.0
        weighted_coords = np.zeros(3)
        
        for atom in protein_data['atoms']:
            element = atom['element']
            mass = ATOMIC_MASSES.get(element, 12.0)  # Default to carbon mass
            
            total_mass += mass
            weighted_coords += mass * atom['coord']
        
        if total_mass > 0:
            properties['molecular_weight'] = total_mass
            properties['center_of_mass'] = weighted_coords / total_mass
        
        return properties
    
    def save_structure(self, 
                      protein_data: Dict[str, Any], 
                      output_path: str,
                      format: str = 'pdb') -> None:
        """
        Save protein structure to file
        
        Args:
            protein_data: Protein structure data
            output_path: Output file path
            format: Output format ('pdb')
        """
        try:
            if format.lower() == 'pdb':
                structure = protein_data['structure']
                io = PDBIO()
                io.set_structure(structure)
                io.save(output_path)
                self.logger.info(f"Structure saved to {output_path}")
            else:
                raise ValueError(f"Unsupported output format: {format}")
                
        except Exception as e:
            self.logger.error(f"Failed to save structure: {e}")
            raise
    
    def clear_cache(self) -> None:
        """Clear structure cache"""
        self._structure_cache.clear()
        self.logger.info("Structure cache cleared")
    
    def get_cache_info(self) -> Dict[str, int]:
        """Get cache information"""
        return {
            'cached_structures': len(self._structure_cache),
            'cache_size_mb': self._estimate_cache_size()
        }
    
    def _estimate_cache_size(self) -> float:
        """Estimate cache size in MB"""
        # Rough estimation based on number of cached structures
        return len(self._structure_cache) * 5.0  # Assume ~5MB per structure
