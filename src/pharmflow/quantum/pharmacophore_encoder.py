"""
Pharmacophore Encoder for Quantum Molecular Docking
Encodes molecular docking problems as QUBO matrices using pharmacophore features
"""

import numpy as np
from typing import Dict, List, Any, Optional, Tuple
import logging
from rdkit import Chem
from rdkit.Chem import AllChem, rdMolDescriptors, Descriptors
from scipy.spatial.distance import cdist

from ..utils.constants import (
    POSITION_ENCODING_BITS, ROTATION_ENCODING_BITS, BOND_ENCODING_BITS,
    POSITION_RANGE, ROTATION_RANGE, PHARMACOPHORE_TYPES, PHARMACOPHORE_INTERACTIONS
)

logger = logging.getLogger(__name__)

class PharmacophoreEncoder:
    """
    Advanced pharmacophore-based encoder for quantum molecular docking
    Converts molecular docking problems into QUBO formulations
    """
    
    def __init__(self):
        """Initialize pharmacophore encoder"""
        self.logger = logging.getLogger(__name__)
        
        # Encoding bit allocations
        self.position_bits = POSITION_ENCODING_BITS  # 18 bits (6 per coordinate)
        self.rotation_bits = ROTATION_ENCODING_BITS  # 12 bits (4 per angle)
        self.angle_bits = 4  # 4 bits per rotation angle
        self.bond_bits = BOND_ENCODING_BITS  # 3 bits per rotatable bond
        
        # Coordinate encoding ranges
        self.position_range = POSITION_RANGE  # (-10, 10) Angstroms
        self.rotation_range = ROTATION_RANGE  # (0, 2π) radians
        
        # Pharmacophore type mappings
        self.pharmacophore_types = {
            ptype: idx for idx, ptype in enumerate(PHARMACOPHORE_TYPES.keys())
        }
        
        # Interaction strength matrix
        self.interaction_matrix = self._build_interaction_matrix()
        
        self.logger.info("Pharmacophore encoder initialized")
    
    def extract_pharmacophores(self,
                              protein: Dict[str, Any],
                              ligand: Chem.Mol,
                              binding_site_residues: Optional[List[int]] = None) -> List[Dict[str, Any]]:
        """
        Extract pharmacophore features from protein-ligand system
        
        Args:
            protein: Protein structure dictionary
            ligand: RDKit molecule object
            binding_site_residues: List of binding site residue IDs
            
        Returns:
            List of pharmacophore feature dictionaries
        """
        try:
            pharmacophores = []
            
            # Extract ligand pharmacophores
            ligand_features = self._extract_ligand_pharmacophores(ligand)
            pharmacophores.extend(ligand_features)
            
            # Extract protein pharmacophores from binding site
            if binding_site_residues and protein.get('residues'):
                protein_features = self._extract_protein_pharmacophores(
                    protein, binding_site_residues
                )
                pharmacophores.extend(protein_features)
            
            self.logger.info(f"Extracted {len(pharmacophores)} pharmacophore features")
            return pharmacophores
            
        except Exception as e:
            self.logger.error(f"Pharmacophore extraction failed: {e}")
            return []
    
    def encode_docking_problem(self,
                              protein: Dict[str, Any],
                              ligand: Chem.Mol,
                              pharmacophores: List[Dict[str, Any]]) -> Tuple[np.ndarray, float]:
        """
        Encode molecular docking problem as QUBO matrix
        
        Args:
            protein: Protein structure
            ligand: Ligand molecule
            pharmacophores: Extracted pharmacophore features
            
        Returns:
            QUBO matrix and offset constant
        """
        try:
            # Calculate total number of qubits needed
            num_rotatable_bonds = self._count_rotatable_bonds(ligand)
            num_pharmacophore_qubits = len(pharmacophores)
            
            total_qubits = (
                self.position_bits +  # Position encoding (18 bits)
                self.rotation_bits +  # Rotation encoding (12 bits)
                num_rotatable_bonds * self.bond_bits +  # Conformational encoding
                num_pharmacophore_qubits  # Pharmacophore selection
            )
            
            self.logger.info(f"Encoding problem with {total_qubits} qubits")
            
            # Initialize QUBO matrix
            qubo_matrix = np.zeros((total_qubits, total_qubits))
            offset = 0.0
            
            # Encode position constraints
            position_offset = 0
            self._encode_position_constraints(
                qubo_matrix, position_offset, self.position_bits
            )
            
            # Encode rotation constraints
            rotation_offset = self.position_bits
            self._encode_rotation_constraints(
                qubo_matrix, rotation_offset, self.rotation_bits
            )
            
            # Encode conformational constraints
            conformation_offset = self.position_bits + self.rotation_bits
            self._encode_conformational_constraints(
                qubo_matrix, conformation_offset, num_rotatable_bonds
            )
            
            # Encode pharmacophore interactions
            pharmacophore_offset = conformation_offset + num_rotatable_bonds * self.bond_bits
            interaction_energy, interaction_offset = self._encode_pharmacophore_interactions(
                qubo_matrix, pharmacophore_offset, pharmacophores
            )
            
            offset += interaction_offset
            
            # Add energy penalties for invalid configurations
            self._add_constraint_penalties(qubo_matrix, total_qubits)
            
            self.logger.info(f"QUBO encoding completed: {qubo_matrix.shape[0]}x{qubo_matrix.shape[1]} matrix")
            
            return qubo_matrix, offset
            
        except Exception as e:
            self.logger.error(f"QUBO encoding failed: {e}")
            # Return minimal valid QUBO
            return np.array([[-1.0]]), 0.0
    
    def decode_quantum_results(self,
                              bitstrings: List[str],
                              pharmacophores: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Decode quantum measurement results back to molecular poses
        
        Args:
            bitstrings: List of measurement bitstrings
            pharmacophores: Original pharmacophore features
            
        Returns:
            List of decoded molecular poses
        """
        poses = []
        
        for bitstring in bitstrings:
            try:
                pose = self._decode_single_bitstring(bitstring, pharmacophores)
                poses.append(pose)
            except Exception as e:
                self.logger.warning(f"Failed to decode bitstring {bitstring}: {e}")
        
        return poses
    
    def _extract_ligand_pharmacophores(self, ligand: Chem.Mol) -> List[Dict[str, Any]]:
        """Extract pharmacophore features from ligand molecule"""
        features = []
        
        try:
            # Hydrophobic features
            hydrophobic_atoms = self._find_hydrophobic_atoms(ligand)
            for atom_idx in hydrophobic_atoms:
                pos = self._get_atom_position(ligand, atom_idx)
                features.append({
                    'type': 'hydrophobic',
                    'position': pos,
                    'atom_idx': atom_idx,
                    'source': 'ligand'
                })
            
            # Hydrogen bond donors
            hbd_atoms = self._find_hb_donors(ligand)
            for atom_idx in hbd_atoms:
                pos = self._get_atom_position(ligand, atom_idx)
                features.append({
                    'type': 'hydrogen_bond_donor',
                    'position': pos,
                    'atom_idx': atom_idx,
                    'source': 'ligand'
                })
            
            # Hydrogen bond acceptors
            hba_atoms = self._find_hb_acceptors(ligand)
            for atom_idx in hba_atoms:
                pos = self._get_atom_position(ligand, atom_idx)
                features.append({
                    'type': 'hydrogen_bond_acceptor',
                    'position': pos,
                    'atom_idx': atom_idx,
                    'source': 'ligand'
                })
            
            # Aromatic rings
            aromatic_rings = self._find_aromatic_rings(ligand)
            for ring_center in aromatic_rings:
                features.append({
                    'type': 'aromatic',
                    'position': ring_center,
                    'source': 'ligand'
                })
            
            # Ionizable groups
            positive_atoms = self._find_positive_ionizable(ligand)
            for atom_idx in positive_atoms:
                pos = self._get_atom_position(ligand, atom_idx)
                features.append({
                    'type': 'positive_ionizable',
                    'position': pos,
                    'atom_idx': atom_idx,
                    'source': 'ligand'
                })
            
            negative_atoms = self._find_negative_ionizable(ligand)
            for atom_idx in negative_atoms:
                pos = self._get_atom_position(ligand, atom_idx)
                features.append({
                    'type': 'negative_ionizable',
                    'position': pos,
                    'atom_idx': atom_idx,
                    'source': 'ligand'
                })
            
            return features
            
        except Exception as e:
            self.logger.error(f"Ligand pharmacophore extraction failed: {e}")
            return []
    
    def _extract_protein_pharmacophores(self,
                                       protein: Dict[str, Any],
                                       binding_site_residues: List[int]) -> List[Dict[str, Any]]:
        """Extract pharmacophore features from protein binding site"""
        features = []
        
        try:
            # Map residue IDs to residue objects
            residue_dict = {res['id']: res for res in protein.get('residues', [])}
            
            for res_id in binding_site_residues:
                if res_id not in residue_dict:
                    continue
                
                residue = residue_dict[res_id]
                res_name = residue['name']
                res_center = residue['center']
                
                # Classify residue pharmacophore types
                if res_name in ['ARG', 'LYS', 'HIS']:
                    features.append({
                        'type': 'positive_ionizable',
                        'position': tuple(res_center),
                        'residue_id': res_id,
                        'residue_name': res_name,
                        'source': 'protein'
                    })
                    
                elif res_name in ['ASP', 'GLU']:
                    features.append({
                        'type': 'negative_ionizable',
                        'position': tuple(res_center),
                        'residue_id': res_id,
                        'residue_name': res_name,
                        'source': 'protein'
                    })
                    
                elif res_name in ['SER', 'THR', 'TYR']:
                    features.append({
                        'type': 'hydrogen_bond_donor',
                        'position': tuple(res_center),
                        'residue_id': res_id,
                        'residue_name': res_name,
                        'source': 'protein'
                    })
                    
                elif res_name in ['ASN', 'GLN']:
                    features.append({
                        'type': 'hydrogen_bond_acceptor',
                        'position': tuple(res_center),
                        'residue_id': res_id,
                        'residue_name': res_name,
                        'source': 'protein'
                    })
                    
                elif res_name in ['PHE', 'TRP', 'TYR']:
                    features.append({
                        'type': 'aromatic',
                        'position': tuple(res_center),
                        'residue_id': res_id,
                        'residue_name': res_name,
                        'source': 'protein'
                    })
                    
                elif res_name in ['ALA', 'VAL', 'LEU', 'ILE', 'MET', 'PRO']:
                    features.append({
                        'type': 'hydrophobic',
                        'position': tuple(res_center),
                        'residue_id': res_id,
                        'residue_name': res_name,
                        'source': 'protein'
                    })
            
            return features
            
        except Exception as e:
            self.logger.error(f"Protein pharmacophore extraction failed: {e}")
            return []
    
    def _find_hydrophobic_atoms(self, mol: Chem.Mol) -> List[int]:
        """Find hydrophobic atoms in molecule"""
        hydrophobic_atoms = []
        
        for atom in mol.GetAtoms():
            if atom.GetSymbol() == 'C':
                # Aliphatic carbons
                if not atom.GetIsAromatic():
                    # Check if carbon is not adjacent to heteroatoms
                    neighbors = [n.GetSymbol() for n in atom.GetNeighbors()]
                    if all(symbol in ['C', 'H'] for symbol in neighbors):
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
            if atom.GetSymbol() in ['N', 'O', 'F']:
                # Check for lone pairs (simplified)
                formal_charge = atom.GetFormalCharge()
                valence = atom.GetTotalValence()
                
                # Simple heuristic for lone pair presence
                if (atom.GetSymbol() == 'O' and valence <= 2) or \
                   (atom.GetSymbol() == 'N' and valence <= 3) or \
                   (atom.GetSymbol() == 'F' and valence <= 1):
                    acceptors.append(atom.GetIdx())
        
        return acceptors
    
    def _find_aromatic_rings(self, mol: Chem.Mol) -> List[Tuple[float, float, float]]:
        """Find aromatic ring centers"""
        ring_centers = []
        
        # Get ring information
        ring_info = mol.GetRingInfo()
        
        for ring in ring_info.AtomRings():
            # Check if ring is aromatic
            if all(mol.GetAtomWithIdx(idx).GetIsAromatic() for idx in ring):
                center = self._calculate_ring_center(mol, ring)
                ring_centers.append(center)
        
        return ring_centers
    
    def _find_positive_ionizable(self, mol: Chem.Mol) -> List[int]:
        """Find positive ionizable atoms"""
        positive_atoms = []
        
        for atom in mol.GetAtoms():
            # Basic amines, amidines, guanidines
            if atom.GetSymbol() == 'N':
                # Check for positive formal charge or basic nitrogen
                if atom.GetFormalCharge() > 0:
                    positive_atoms.append(atom.GetIdx())
                elif len([n for n in atom.GetNeighbors() if n.GetSymbol() == 'H']) >= 2:
                    # Primary or secondary amine
                    positive_atoms.append(atom.GetIdx())
        
        return positive_atoms
    
    def _find_negative_ionizable(self, mol: Chem.Mol) -> List[int]:
        """Find negative ionizable atoms"""
        negative_atoms = []
        
        for atom in mol.GetAtoms():
            # Carboxylates, phosphates, sulfonates
            if atom.GetSymbol() == 'O':
                # Check for negative formal charge or carboxyl oxygen
                if atom.GetFormalCharge() < 0:
                    negative_atoms.append(atom.GetIdx())
                else:
                    # Check if part of carboxyl group
                    for neighbor in atom.GetNeighbors():
                        if neighbor.GetSymbol() == 'C':
                            # Check for C=O pattern
                            carbon_neighbors = [n.GetSymbol() for n in neighbor.GetNeighbors()]
                            if carbon_neighbors.count('O') >= 2:
                                negative_atoms.append(atom.GetIdx())
                                break
        
        return negative_atoms
    
    def _calculate_ring_center(self, mol: Chem.Mol, ring: Tuple[int, ...]) -> Tuple[float, float, float]:
        """Calculate geometric center of aromatic ring"""
        conf = mol.GetConformer()
        coords = []
        
        for atom_idx in ring:
            pos = conf.GetAtomPosition(atom_idx)
            coords.append([pos.x, pos.y, pos.z])
        
        center = np.mean(coords, axis=0)
        return tuple(center)
    
    def _get_atom_position(self, mol: Chem.Mol, atom_idx: int) -> Tuple[float, float, float]:
        """Get 3D position of atom"""
        conf = mol.GetConformer()
        pos = conf.GetAtomPosition(atom_idx)
        return (pos.x, pos.y, pos.z)
    
    def _count_rotatable_bonds(self, mol: Chem.Mol) -> int:
        """Count rotatable bonds in molecule"""
        return Descriptors.NumRotatableBonds(mol)
    
    def _encode_position_constraints(self, qubo_matrix: np.ndarray, offset: int, num_bits: int):
        """Encode position constraints in QUBO matrix"""
        # Add quadratic penalty for position encoding
        penalty_strength = 1.0
        
        # Group position bits by coordinate (6 bits per coordinate)
        for coord in range(3):
            coord_offset = offset + coord * 6
            
            # Add constraints to favor reasonable position values
            for i in range(6):
                qubo_matrix[coord_offset + i, coord_offset + i] -= penalty_strength
    
    def _encode_rotation_constraints(self, qubo_matrix: np.ndarray, offset: int, num_bits: int):
        """Encode rotation constraints in QUBO matrix"""
        # Add quadratic penalty for rotation encoding
        penalty_strength = 0.8
        
        # Group rotation bits by angle (4 bits per angle)
        for angle in range(3):
            angle_offset = offset + angle * 4
            
            # Add constraints to favor reasonable rotation values
            for i in range(4):
                qubo_matrix[angle_offset + i, angle_offset + i] -= penalty_strength
    
    def _encode_conformational_constraints(self, qubo_matrix: np.ndarray, offset: int, num_bonds: int):
        """Encode conformational constraints"""
        penalty_strength = 0.6
        
        # Add torsional energy penalties
        for bond in range(num_bonds):
            bond_offset = offset + bond * self.bond_bits
            
            for i in range(self.bond_bits):
                qubo_matrix[bond_offset + i, bond_offset + i] -= penalty_strength
                
                # Add coupling between adjacent bonds
                if bond < num_bonds - 1:
                    next_bond_offset = offset + (bond + 1) * self.bond_bits
                    if i < self.bond_bits:
                        qubo_matrix[bond_offset + i, next_bond_offset + i] += 0.2
    
    def _encode_pharmacophore_interactions(self,
                                         qubo_matrix: np.ndarray,
                                         offset: int,
                                         pharmacophores: List[Dict[str, Any]]) -> Tuple[float, float]:
        """Encode pharmacophore interaction energies"""
        total_interaction_energy = 0.0
        interaction_offset = 0.0
        
        num_pharmacophores = len(pharmacophores)
        
        # Encode pairwise pharmacophore interactions
        for i in range(num_pharmacophores):
            for j in range(i + 1, num_pharmacophores):
                pharm_i = pharmacophores[i]
                pharm_j = pharmacophores[j]
                
                # Check if pharmacophores can interact
                if self._pharmacophores_complement(pharm_i, pharm_j):
                    interaction_strength = self._calculate_interaction_strength(pharm_i, pharm_j)
                    
                    # Add interaction term to QUBO
                    qubo_matrix[offset + i, offset + j] += interaction_strength
                    total_interaction_energy += interaction_strength
        
        # Add individual pharmacophore selection penalties
        for i in range(num_pharmacophores):
            selection_penalty = self._calculate_selection_penalty(pharmacophores[i])
            qubo_matrix[offset + i, offset + i] += selection_penalty
        
        return total_interaction_energy, interaction_offset
    
    def _pharmacophores_complement(self, pharm1: Dict[str, Any], pharm2: Dict[str, Any]) -> bool:
        """Check if two pharmacophores can interact favorably"""
        type1 = pharm1['type']
        type2 = pharm2['type']
        
        # Check complementarity rules
        complementary_pairs = [
            ('hydrogen_bond_donor', 'hydrogen_bond_acceptor'),
            ('hydrogen_bond_acceptor', 'hydrogen_bond_donor'),
            ('positive_ionizable', 'negative_ionizable'),
            ('negative_ionizable', 'positive_ionizable'),
            ('hydrophobic', 'hydrophobic'),
            ('aromatic', 'aromatic')
        ]
        
        return (type1, type2) in complementary_pairs
    
    def _calculate_interaction_strength(self, pharm1: Dict[str, Any], pharm2: Dict[str, Any]) -> float:
        """Calculate interaction strength between pharmacophores"""
        type1 = pharm1['type']
        type2 = pharm2['type']
        
        # Get base interaction strength
        base_strength = PHARMACOPHORE_INTERACTIONS.get((type1, type2), 0.0)
        
        # Apply distance-dependent scaling
        if 'position' in pharm1 and 'position' in pharm2:
            pos1 = np.array(pharm1['position'])
            pos2 = np.array(pharm2['position'])
            distance = np.linalg.norm(pos1 - pos2)
            
            # Optimal interaction distance (Angstroms)
            optimal_distance = 3.5
            distance_factor = np.exp(-(distance - optimal_distance)**2 / 2.0)
            
            base_strength *= distance_factor
        
        return base_strength
    
    def _calculate_selection_penalty(self, pharmacophore: Dict[str, Any]) -> float:
        """Calculate selection penalty for pharmacophore"""
        # Different pharmacophore types have different selection preferences
        penalties = {
            'hydrogen_bond_donor': -1.5,
            'hydrogen_bond_acceptor': -1.5,
            'positive_ionizable': -2.0,
            'negative_ionizable': -2.0,
            'aromatic': -1.0,
            'hydrophobic': -0.5
        }
        
        return penalties.get(pharmacophore['type'], 0.0)
    
    def _add_constraint_penalties(self, qubo_matrix: np.ndarray, total_qubits: int):
        """Add constraint penalties to prevent invalid configurations"""
        penalty_strength = 10.0
        
        # Add constraint to prevent all-zero solutions
        for i in range(total_qubits):
            qubo_matrix[i, i] -= penalty_strength * 0.1
    
    def _build_interaction_matrix(self) -> np.ndarray:
        """Build pharmacophore interaction strength matrix"""
        num_types = len(self.pharmacophore_types)
        matrix = np.zeros((num_types, num_types))
        
        for (type1, type2), strength in PHARMACOPHORE_INTERACTIONS.items():
            idx1 = self.pharmacophore_types.get(type1, 0)
            idx2 = self.pharmacophore_types.get(type2, 0)
            matrix[idx1, idx2] = strength
            matrix[idx2, idx1] = strength  # Symmetric
        
        return matrix
    
    def _decode_single_bitstring(self, bitstring: str, pharmacophores: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Decode single bitstring to molecular pose"""
        bits = [int(b) for b in bitstring[::-1]]  # Reverse for correct bit order
        
        # Decode position (first 18 bits)
        position_bits = bits[:18] if len(bits) >= 18 else bits + [0] * (18 - len(bits))
        position = self._decode_position(position_bits)
        
        # Decode rotation (next 12 bits)
        rotation_bits = bits[18:30] if len(bits) >= 30 else [0] * 12
        rotation = self._decode_rotation(rotation_bits)
        
        # Decode conformation (remaining bits for bonds)
        remaining_bits = bits[30:] if len(bits) > 30 else []
        conformation = self._decode_conformation(remaining_bits)
        
        # Decode pharmacophore selection
        num_pharmacophores = len(pharmacophores)
        pharmacophore_offset = 30 + len(remaining_bits) - num_pharmacophores
        pharmacophore_bits = bits[max(0, pharmacophore_offset):max(0, pharmacophore_offset + num_pharmacophores)]
        
        selected_pharmacophores = []
        for i, bit in enumerate(pharmacophore_bits):
            if bit == 1 and i < len(pharmacophores):
                selected_pharmacophores.append(pharmacophores[i])
        
        return {
            'position': position,
            'rotation': rotation,
            'conformation': conformation,
            'bitstring': bitstring,
            'pharmacophores': selected_pharmacophores
        }
    
    def _decode_position(self, bits: List[int]) -> Tuple[float, float, float]:
        """Decode position from binary representation"""
        # Extract coordinates (6 bits each)
        x_bits = bits[:6]
        y_bits = bits[6:12]
        z_bits = bits[12:18]
        
        x = self._bits_to_float(x_bits, *self.position_range)
        y = self._bits_to_float(y_bits, *self.position_range)
        z = self._bits_to_float(z_bits, *self.position_range)
        
        return (x, y, z)
    
    def _decode_rotation(self, bits: List[int]) -> Tuple[float, float, float]:
        """Decode rotation angles from binary representation"""
        # Extract angles (4 bits each)
        alpha_bits = bits[:4]
        beta_bits = bits[4:8]
        gamma_bits = bits[8:12]
        
        alpha = self._bits_to_float(alpha_bits, *self.rotation_range)
        beta = self._bits_to_float(beta_bits, *self.rotation_range)
        gamma = self._bits_to_float(gamma_bits, *self.rotation_range)
        
        return (alpha, beta, gamma)
    
    def _decode_conformation(self, bits: List[int]) -> List[float]:
        """Decode conformational angles from binary representation"""
        angles = []
        
        # Group bits by bond (3 bits per bond)
        for i in range(0, len(bits), self.bond_bits):
            bond_bits = bits[i:i + self.bond_bits]
            if len(bond_bits) == self.bond_bits:
                angle = self._bits_to_float(bond_bits, *self.rotation_range)
                angles.append(angle)
        
        return angles
    
    def _bits_to_float(self, bits: List[int], min_val: float, max_val: float) -> float:
        """Convert bit array to float value in range"""
        if not bits:
            return min_val
        
        # Convert to decimal
        decimal = sum(bit * (2 ** i) for i, bit in enumerate(bits))
        max_decimal = (2 ** len(bits)) - 1
        
        if max_decimal == 0:
            return min_val
        
        # Scale to range
        normalized = decimal / max_decimal
        return min_val + normalized * (max_val - min_val)
    
    def _positions_interact(self, pos1_idx: int, pos2_idx: int) -> bool:
        """Check if two position indices represent interacting regions"""
        # Simplified interaction check based on position encoding
        # In practice, would use actual 3D coordinates
        
        # Map indices to 3D grid positions
        grid_size = 2 ** 6  # 6 bits per coordinate
        
        x1, y1, z1 = pos1_idx % grid_size, (pos1_idx // grid_size) % grid_size, pos1_idx // (grid_size ** 2)
        x2, y2, z2 = pos2_idx % grid_size, (pos2_idx // grid_size) % grid_size, pos2_idx // (grid_size ** 2)
        
        # Calculate Manhattan distance
        distance = abs(x1 - x2) + abs(y1 - y2) + abs(z1 - z2)
        
        # Interaction cutoff
        return distance <= 5
