"""
Pharmacophore-Based Quantum Encoder
Handles molecular encoding using pharmacophore patterns
"""

import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors
from rdkit.Chem.Pharmacophore import FeatMaps
import logging

logger = logging.getLogger(__name__)

class PharmacophoreEncoder:
    """Pharmacophore-based quantum encoding for molecular docking"""
    
    def __init__(self):
        """Initialize pharmacophore encoder"""
        self.pharmacophore_types = {
            'hydrophobic': 0,
            'hydrogen_bond_donor': 1,
            'hydrogen_bond_acceptor': 2,
            'aromatic': 3,
            'positive_ionizable': 4,
            'negative_ionizable': 5
        }
        
        # Encoding parameters
        self.position_bits = 6  # 6 bits per coordinate (64 positions)
        self.angle_bits = 4     # 4 bits per angle (16 angles)
        self.bond_bits = 3      # 3 bits per rotatable bond
        
        logger.info("Pharmacophore encoder initialized")
    
    def extract_pharmacophores(self, 
                              protein: Any,
                              ligand: Any,
                              binding_site_residues: Optional[List[int]] = None) -> List[Dict]:
        """
        Extract pharmacophore features from protein-ligand system
        
        Args:
            protein: Protein structure object
            ligand: Ligand molecule object
            binding_site_residues: Specific binding site residues
            
        Returns:
            List of pharmacophore feature dictionaries
        """
        pharmacophores = []
        
        # Extract ligand pharmacophores
        ligand_features = self._extract_ligand_pharmacophores(ligand)
        pharmacophores.extend(ligand_features)
        
        # Extract protein binding site pharmacophores
        if binding_site_residues:
            protein_features = self._extract_protein_pharmacophores(
                protein, binding_site_residues
            )
            pharmacophores.extend(protein_features)
        
        logger.info(f"Extracted {len(pharmacophores)} pharmacophore features")
        return pharmacophores
    
    def _extract_ligand_pharmacophores(self, ligand: Chem.Mol) -> List[Dict]:
        """Extract pharmacophore features from ligand"""
        features = []
        
        # Hydrophobic centers
        hydrophobic_atoms = self._find_hydrophobic_atoms(ligand)
        for atom_idx in hydrophobic_atoms:
            features.append({
                'type': 'hydrophobic',
                'atom_idx': atom_idx,
                'position': self._get_atom_position(ligand, atom_idx),
                'source': 'ligand'
            })
        
        # Hydrogen bond donors
        hbd_atoms = self._find_hb_donors(ligand)
        for atom_idx in hbd_atoms:
            features.append({
                'type': 'hydrogen_bond_donor',
                'atom_idx': atom_idx,
                'position': self._get_atom_position(ligand, atom_idx),
                'source': 'ligand'
            })
        
        # Hydrogen bond acceptors
        hba_atoms = self._find_hb_acceptors(ligand)
        for atom_idx in hba_atoms:
            features.append({
                'type': 'hydrogen_bond_acceptor',
                'atom_idx': atom_idx,
                'position': self._get_atom_position(ligand, atom_idx),
                'source': 'ligand'
            })
        
        # Aromatic centers
        aromatic_rings = self._find_aromatic_rings(ligand)
        for ring_center in aromatic_rings:
            features.append({
                'type': 'aromatic',
                'position': ring_center,
                'source': 'ligand'
            })
        
        return features
    
    def _extract_protein_pharmacophores(self, 
                                       protein: Any,
                                       binding_site_residues: List[int]) -> List[Dict]:
        """Extract pharmacophore features from protein binding site"""
        features = []
        
        # Simplified protein pharmacophore extraction
        # In practice, this would analyze protein residues
        
        for residue_id in binding_site_residues:
            # Mock protein pharmacophore extraction
            # Replace with actual protein analysis
            features.append({
                'type': 'hydrogen_bond_acceptor',
                'residue_id': residue_id,
                'position': [0.0, 0.0, 0.0],  # Would be actual residue position
                'source': 'protein'
            })
        
        return features
    
    def encode_docking_problem(self, 
                              protein: Any,
                              ligand: Any,
                              pharmacophores: List[Dict]) -> Tuple[np.ndarray, float]:
        """
        Encode molecular docking problem as QUBO matrix
        
        Args:
            protein: Protein structure
            ligand: Ligand molecule
            pharmacophores: Extracted pharmacophore features
            
        Returns:
            QUBO matrix and offset value
        """
        # Calculate total qubits needed
        ligand_pharmacophores = [p for p in pharmacophores if p['source'] == 'ligand']
        
        # Encoding scheme:
        # - Position encoding: 18 qubits (6 bits × 3 coordinates)
        # - Rotation encoding: 12 qubits (4 bits × 3 angles)
        # - Conformation encoding: 3n qubits (n rotatable bonds)
        # - Pharmacophore selection: m qubits (m pharmacophores)
        
        num_rotatable_bonds = self._count_rotatable_bonds(ligand)
        num_pharmacophores = len(ligand_pharmacophores)
        
        total_qubits = 18 + 12 + (3 * num_rotatable_bonds) + num_pharmacophores
        
        # Initialize QUBO matrix
        qubo_matrix = np.zeros((total_qubits, total_qubits))
        
        # Add energy terms
        self._add_position_energy_terms(qubo_matrix, pharmacophores)
        self._add_rotation_energy_terms(qubo_matrix)
        self._add_conformation_energy_terms(qubo_matrix, num_rotatable_bonds)
        self._add_pharmacophore_interaction_terms(qubo_matrix, pharmacophores)
        
        offset = 0.0  # Constant energy offset
        
        logger.info(f"Encoded docking problem: {total_qubits} qubits")
        return qubo_matrix, offset
    
    def _add_position_energy_terms(self, 
                                  qubo_matrix: np.ndarray,
                                  pharmacophores: List[Dict]):
        """Add position-dependent energy terms to QUBO matrix"""
        
        # Position encoding uses qubits 0-17 (18 qubits total)
        position_start = 0
        
        # Add quadratic penalty for unfavorable positions
        for i in range(18):
            qubo_matrix[position_start + i, position_start + i] = -0.1  # Favor occupied positions
            
            # Add pairwise interactions
            for j in range(i + 1, 18):
                if self._positions_interact(i, j):
                    qubo_matrix[position_start + i, position_start + j] = 0.05
    
    def _add_rotation_energy_terms(self, qubo_matrix: np.ndarray):
        """Add rotation-dependent energy terms"""
        
        # Rotation encoding uses qubits 18-29 (12 qubits total)
        rotation_start = 18
        
        # Add rotational energy penalties
        for i in range(12):
            qubo_matrix[rotation_start + i, rotation_start + i] = -0.05
    
    def _add_conformation_energy_terms(self, 
                                      qubo_matrix: np.ndarray,
                                      num_rotatable_bonds: int):
        """Add conformational energy terms"""
        
        # Conformation encoding starts after rotation qubits
        conformation_start = 30
        conformation_qubits = 3 * num_rotatable_bonds
        
        for i in range(conformation_qubits):
            # Favor extended conformations slightly
            qubo_matrix[conformation_start + i, conformation_start + i] = -0.02
    
    def _add_pharmacophore_interaction_terms(self, 
                                           qubo_matrix: np.ndarray,
                                           pharmacophores: List[Dict]):
        """Add pharmacophore interaction energy terms"""
        
        ligand_pharmacophores = [p for p in pharmacophores if p['source'] == 'ligand']
        protein_pharmacophores = [p for p in pharmacophores if p['source'] == 'protein']
        
        # Pharmacophore selection qubits start after conformation qubits
        pharmacophore_start = 30 + (3 * self._count_rotatable_bonds_from_pharmacophores(pharmacophores))
        
        # Add favorable interactions between complementary pharmacophores
        for i, lig_pharm in enumerate(ligand_pharmacophores):
            for j, prot_pharm in enumerate(protein_pharmacophores):
                if self._pharmacophores_complement(lig_pharm, prot_pharm):
                    # Add negative (favorable) interaction
                    qubit_i = pharmacophore_start + i
                    if qubit_i < qubo_matrix.shape[0]:
                        qubo_matrix[qubit_i, qubit_i] -= 0.5
    
    def decode_quantum_results(self, 
                              bitstrings: List[str],
                              pharmacophores: List[Dict]) -> List[Dict]:
        """
        Decode quantum measurement results to molecular poses
        
        Args:
            bitstrings: Quantum measurement bitstrings
            pharmacophores: Pharmacophore features
            
        Returns:
            List of decoded molecular poses
        """
        poses = []
        
        for bitstring in bitstrings:
            try:
                pose = self._decode_single_bitstring(bitstring, pharmacophores)
                poses.append(pose)
            except Exception as e:
                logger.warning(f"Failed to decode bitstring {bitstring}: {e}")
        
        return poses
    
    def _decode_single_bitstring(self, 
                                bitstring: str,
                                pharmacophores: List[Dict]) -> Dict:
        """Decode a single bitstring to molecular pose"""
        
        bits = [int(b) for b in bitstring[::-1]]  # Reverse for correct order
        
        # Decode position (qubits 0-17)
        position = self._decode_position(bits[0:18])
        
        # Decode rotation (qubits 18-29)
        rotation = self._decode_rotation(bits[18:30])
        
        # Decode conformation (remaining qubits)
        conformation = self._decode_conformation(bits[30:])
        
        return {
            'position': position,
            'rotation': rotation,
            'conformation': conformation,
            'bitstring': bitstring,
            'pharmacophores': pharmacophores
        }
    
    def _decode_position(self, position_bits: List[int]) -> Tuple[float, float, float]:
        """Decode position from binary representation"""
        
        # Each coordinate uses 6 bits
        x_bits = position_bits[0:6]
        y_bits = position_bits[6:12]
        z_bits = position_bits[12:18]
        
        # Convert to decimal and normalize to [-10, 10] range
        x = self._bits_to_float(x_bits, -10.0, 10.0)
        y = self._bits_to_float(y_bits, -10.0, 10.0)
        z = self._bits_to_float(z_bits, -10.0, 10.0)
        
        return (x, y, z)
    
    def _decode_rotation(self, rotation_bits: List[int]) -> Tuple[float, float, float]:
        """Decode rotation angles from binary representation"""
        
        # Each angle uses 4 bits
        alpha_bits = rotation_bits[0:4]
        beta_bits = rotation_bits[4:8]
        gamma_bits = rotation_bits[8:12]
        
        # Convert to angles in [0, 2π] range
        alpha = self._bits_to_float(alpha_bits, 0.0, 2 * np.pi)
        beta = self._bits_to_float(beta_bits, 0.0, 2 * np.pi)
        gamma = self._bits_to_float(gamma_bits, 0.0, 2 * np.pi)
        
        return (alpha, beta, gamma)
    
    def _decode_conformation(self, conformation_bits: List[int]) -> List[float]:
        """Decode conformational angles from binary representation"""
        
        # Each rotatable bond uses 3 bits
        num_bonds = len(conformation_bits) // 3
        angles = []
        
        for i in range(num_bonds):
            bond_bits = conformation_bits[i*3:(i+1)*3]
            angle = self._bits_to_float(bond_bits, 0.0, 2 * np.pi)
            angles.append(angle)
        
        return angles
    
    def _bits_to_float(self, bits: List[int], min_val: float, max_val: float) -> float:
        """Convert binary representation to float in given range"""
        
        if not bits:
            return min_val
        
        # Convert to decimal
        decimal = sum(bit * (2 ** i) for i, bit in enumerate(bits))
        max_decimal = (2 ** len(bits)) - 1
        
        # Normalize to range
        normalized = decimal / max_decimal
        return min_val + normalized * (max_val - min_val)
    
    # Helper methods for pharmacophore analysis
    
    def _find_hydrophobic_atoms(self, mol: Chem.Mol) -> List[int]:
        """Find hydrophobic atoms in molecule"""
        hydrophobic_atoms = []
        
        for atom in mol.GetAtoms():
            if atom.GetSymbol() == 'C' and atom.GetHybridization() in [Chem.HybridizationType.SP3, Chem.HybridizationType.SP2]:
                # Simple hydrophobic definition: carbon atoms
                hydrophobic_atoms.append(atom.GetIdx())
        
        return hydrophobic_atoms
    
    def _find_hb_donors(self, mol: Chem.Mol) -> List[int]:
        """Find hydrogen bond donor atoms"""
        donors = []
        
        for atom in mol.GetAtoms():
            if atom.GetSymbol() in ['N', 'O'] and atom.GetTotalNumHs() > 0:
                donors.append(atom.GetIdx())
        
        return donors
    
    def _find_hb_acceptors(self, mol: Chem.Mol) -> List[int]:
        """Find hydrogen bond acceptor atoms"""
        acceptors = []
        
        for atom in mol.GetAtoms():
            if atom.GetSymbol() in ['N', 'O', 'F'] and len([n for n in atom.GetNeighbors()]) < 4:
                acceptors.append(atom.GetIdx())
        
        return acceptors
    
    def _find_aromatic_rings(self, mol: Chem.Mol) -> List[Tuple[float, float, float]]:
        """Find aromatic ring centers"""
        ring_centers = []
        
        ring_info = mol.GetRingInfo()
        for ring in ring_info.AtomRings():
            if len(ring) in [5, 6]:  # Common aromatic ring sizes
                # Check if aromatic
                aromatic = all(mol.GetAtomWithIdx(i).GetIsAromatic() for i in ring)
                if aromatic:
                    # Calculate ring center
                    center = self._calculate_ring_center(mol, ring)
                    ring_centers.append(center)
        
        return ring_centers
    
    def _calculate_ring_center(self, mol: Chem.Mol, ring: Tuple[int]) -> Tuple[float, float, float]:
        """Calculate the geometric center of a ring"""
        if mol.GetNumConformers() == 0:
            return (0.0, 0.0, 0.0)
        
        conf = mol.GetConformer()
        positions = [conf.GetAtomPosition(i) for i in ring]
        
        center_x = sum(pos.x for pos in positions) / len(positions)
        center_y = sum(pos.y for pos in positions) / len(positions)
        center_z = sum(pos.z for pos in positions) / len(positions)
        
        return (center_x, center_y, center_z)
    
    def _get_atom_position(self, mol: Chem.Mol, atom_idx: int) -> Tuple[float, float, float]:
        """Get atom position coordinates"""
        if mol.GetNumConformers() == 0:
            return (0.0, 0.0, 0.0)
        
        conf = mol.GetConformer()
        pos = conf.GetAtomPosition(atom_idx)
        return (pos.x, pos.y, pos.z)
    
    def _count_rotatable_bonds(self, mol: Chem.Mol) -> int:
        """Count rotatable bonds in molecule"""
        return rdMolDescriptors.CalcNumRotatableBonds(mol)
    
    def _count_rotatable_bonds_from_pharmacophores(self, pharmacophores: List[Dict]) -> int:
        """Estimate rotatable bonds from pharmacophores"""
        # Simplified estimation
        ligand_pharmacophores = [p for p in pharmacophores if p['source'] == 'ligand']
        return max(1, len(ligand_pharmacophores) // 3)
    
    def _positions_interact(self, pos1: int, pos2: int) -> bool:
        """Check if two positions interact"""
        # Simple interaction check based on proximity
        return abs(pos1 - pos2) <= 3
    
    def _pharmacophores_complement(self, pharm1: Dict, pharm2: Dict) -> bool:
        """Check if two pharmacophores are complementary"""
        complementary_pairs = [
            ('hydrogen_bond_donor', 'hydrogen_bond_acceptor'),
            ('hydrogen_bond_acceptor', 'hydrogen_bond_donor'),
            ('hydrophobic', 'hydrophobic'),
            ('aromatic', 'aromatic')
        ]
        
        return (pharm1['type'], pharm2['type']) in complementary_pairs
