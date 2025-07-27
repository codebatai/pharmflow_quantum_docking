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
Test suite for PharmFlow Pharmacophore Encoder
Comprehensive testing of pharmacophore-based quantum encoding functionality
"""

import pytest
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
import sys
import os

# Add src to path for testing
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from pharmflow.quantum.pharmacophore_encoder import PharmacophoreEncoder
from pharmflow.utils.constants import PHARMACOPHORE_TYPES

class TestPharmacophoreEncoder:
    """Test cases for PharmFlow Pharmacophore Encoder"""
    
    @pytest.fixture
    def encoder(self):
        """Create pharmacophore encoder for testing"""
        return PharmacophoreEncoder()
    
    @pytest.fixture
    def simple_ligand(self):
        """Create simple ligand molecule for testing"""
        # Ethanol - simple molecule with known pharmacophoric features
        smiles = "CCO"
        mol = Chem.MolFromSmiles(smiles)
        mol = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol, randomSeed=42)
        AllChem.MMFFOptimizeMolecule(mol)
        return mol
    
    @pytest.fixture
    def complex_ligand(self):
        """Create more complex ligand molecule for testing"""
        # Ibuprofen - has multiple pharmacophoric features
        smiles = "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O"
        mol = Chem.MolFromSmiles(smiles)
        mol = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol, randomSeed=42)
        AllChem.MMFFOptimizeMolecule(mol)
        return mol
    
    @pytest.fixture
    def mock_protein(self):
        """Create mock protein structure for testing"""
        return {
            'structure': None,
            'chains': {'A': {'residues': []}},
            'atoms': [],
            'coordinates': np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2]]),
            'residues': [
                {'name': 'ARG', 'id': 1, 'center': np.array([0, 0, 0])},
                {'name': 'ASP', 'id': 2, 'center': np.array([5, 5, 5])}
            ]
        }
    
    def test_encoder_initialization(self, encoder):
        """Test pharmacophore encoder initialization"""
        assert encoder.position_bits == 6
        assert encoder.angle_bits == 4
        assert encoder.bond_bits == 3
        assert len(encoder.pharmacophore_types) == 6
        
        # Check pharmacophore type mappings
        expected_types = [
            'hydrophobic', 'hydrogen_bond_donor', 'hydrogen_bond_acceptor',
            'aromatic', 'positive_ionizable', 'negative_ionizable'
        ]
        for ptype in expected_types:
            assert ptype in encoder.pharmacophore_types
    
    def test_ligand_pharmacophore_extraction(self, encoder, simple_ligand):
        """Test extraction of pharmacophores from ligand"""
        ligand_features = encoder._extract_ligand_pharmacophores(simple_ligand)
        
        # Should find some pharmacophoric features in ethanol
        assert isinstance(ligand_features, list)
        assert len(ligand_features) > 0
        
        # Check feature structure
        for feature in ligand_features:
            assert 'type' in feature
            assert 'position' in feature
            assert 'source' in feature
            assert feature['source'] == 'ligand'
            assert feature['type'] in encoder.pharmacophore_types
    
    def test_complex_ligand_pharmacophores(self, encoder, complex_ligand):
        """Test pharmacophore extraction from complex molecule"""
        ligand_features = encoder._extract_ligand_pharmacophores(complex_ligand)
        
        # Ibuprofen should have multiple pharmacophoric features
        assert len(ligand_features) >= 2
        
        # Should find aromatic and hydrogen bond features
        feature_types = [f['type'] for f in ligand_features]
        assert 'aromatic' in feature_types or 'hydrophobic' in feature_types
    
    def test_hydrophobic_atom_identification(self, encoder, complex_ligand):
        """Test identification of hydrophobic atoms"""
        hydrophobic_atoms = encoder._find_hydrophobic_atoms(complex_ligand)
        
        assert isinstance(hydrophobic_atoms, list)
        assert len(hydrophobic_atoms) > 0
        
        # Check that returned indices are valid
        for atom_idx in hydrophobic_atoms:
            assert 0 <= atom_idx < complex_ligand.GetNumAtoms()
            atom = complex_ligand.GetAtomWithIdx(atom_idx)
            assert atom.GetSymbol() == 'C'
    
    def test_hydrogen_bond_donor_identification(self, encoder, simple_ligand):
        """Test identification of hydrogen bond donors"""
        hbd_atoms = encoder._find_hb_donors(simple_ligand)
        
        assert isinstance(hbd_atoms, list)
        
        # Ethanol should have one H-bond donor (OH)
        assert len(hbd_atoms) >= 1
        
        for atom_idx in hbd_atoms:
            atom = simple_ligand.GetAtomWithIdx(atom_idx)
            assert atom.GetSymbol() in ['N', 'O']
            assert atom.GetTotalNumHs() > 0
    
    def test_hydrogen_bond_acceptor_identification(self, encoder, simple_ligand):
        """Test identification of hydrogen bond acceptors"""
        hba_atoms = encoder._find_hb_acceptors(simple_ligand)
        
        assert isinstance(hba_atoms, list)
        
        # Ethanol should have H-bond acceptors
        assert len(hba_atoms) >= 1
        
        for atom_idx in hba_atoms:
            atom = simple_ligand.GetAtomWithIdx(atom_idx)
            assert atom.GetSymbol() in ['N', 'O', 'F']
    
    def test_aromatic_ring_identification(self, encoder, complex_ligand):
        """Test identification of aromatic rings"""
        aromatic_rings = encoder._find_aromatic_rings(complex_ligand)
        
        assert isinstance(aromatic_rings, list)
        
        # Ibuprofen has one aromatic ring
        assert len(aromatic_rings) >= 1
        
        for ring_center in aromatic_rings:
            assert isinstance(ring_center, tuple)
            assert len(ring_center) == 3  # x, y, z coordinates
            assert all(isinstance(coord, float) for coord in ring_center)
    
    def test_pharmacophore_extraction_with_protein(self, encoder, simple_ligand, mock_protein):
        """Test pharmacophore extraction with protein context"""
        binding_site_residues = [1, 2]
        
        pharmacophores = encoder.extract_pharmacophores(
            mock_protein, simple_ligand, binding_site_residues
        )
        
        assert isinstance(pharmacophores, list)
        assert len(pharmacophores) > 0
        
        # Should have both ligand and protein pharmacophores
        sources = [p['source'] for p in pharmacophores]
        assert 'ligand' in sources
    
    def test_qubo_matrix_encoding(self, encoder, simple_ligand, mock_protein):
        """Test QUBO matrix generation from molecular problem"""
        pharmacophores = encoder.extract_pharmacophores(
            mock_protein, simple_ligand, None
        )
        
        qubo_matrix, offset = encoder.encode_docking_problem(
            mock_protein, simple_ligand, pharmacophores
        )
        
        # Check QUBO matrix properties
        assert isinstance(qubo_matrix, np.ndarray)
        assert len(qubo_matrix.shape) == 2
        assert qubo_matrix.shape[0] == qubo_matrix.shape[1]  # Square matrix
        assert qubo_matrix.shape[0] > 0  # Non-empty
        
        # Check offset
        assert isinstance(offset, (int, float))
        
        # Check matrix is finite
        assert np.all(np.isfinite(qubo_matrix))
    
    def test_bitstring_decoding(self, encoder, simple_ligand, mock_protein):
        """Test decoding of quantum measurement bitstrings"""
        pharmacophores = encoder.extract_pharmacophores(
            mock_protein, simple_ligand, None
        )
        
        # Create test bitstrings
        test_bitstrings = [
            '000000000000000000000000000000',
            '111111111111111111111111111111',
            '101010101010101010101010101010',
            '010101010101010101010101010101'
        ]
        
        decoded_poses = encoder.decode_quantum_results(test_bitstrings, pharmacophores)
        
        assert isinstance(decoded_poses, list)
        assert len(decoded_poses) == len(test_bitstrings)
        
        for pose in decoded_poses:
            assert 'position' in pose
            assert 'rotation' in pose
            assert 'conformation' in pose
            assert 'bitstring' in pose
            assert 'pharmacophores' in pose
    
    def test_position_decoding(self, encoder):
        """Test position decoding from binary representation"""
        # Test known bit patterns
        test_cases = [
            ([0, 0, 0, 0, 0, 0] * 3, (-10.0, -10.0, -10.0)),  # All zeros -> minimum
            ([1, 1, 1, 1, 1, 1] * 3, (10.0, 10.0, 10.0)),     # All ones -> maximum
        ]
        
        for bits, expected_approx in test_cases:
            position = encoder._decode_position(bits)
            
            assert isinstance(position, tuple)
            assert len(position) == 3
            
            # Check values are in expected range
            for coord, expected_coord in zip(position, expected_approx):
                assert -10.0 <= coord <= 10.0
                if expected_coord == -10.0:
                    assert coord < -5.0
                elif expected_coord == 10.0:
                    assert coord > 5.0
    
    def test_rotation_decoding(self, encoder):
        """Test rotation angle decoding from binary representation"""
        # Test bit patterns for rotation
        test_bits = [0, 0, 0, 0] * 3  # All zeros
        
        rotation = encoder._decode_rotation(test_bits)
        
        assert isinstance(rotation, tuple)
        assert len(rotation) == 3
        
        # Check angles are in valid range [0, 2π]
        for angle in rotation:
            assert 0.0 <= angle <= 2 * np.pi
    
    def test_conformation_decoding(self, encoder):
        """Test conformational angle decoding"""
        test_bits = [1, 0, 1, 0, 1, 0]  # Two bond angles
        
        conformation = encoder._decode_conformation(test_bits)
        
        assert isinstance(conformation, list)
        assert len(conformation) == 2  # Two bonds from 6 bits
        
        # Check angles are in valid range
        for angle in conformation:
            assert 0.0 <= angle <= 2 * np.pi
    
    def test_bits_to_float_conversion(self, encoder):
        """Test binary to float conversion utility"""
        # Test edge cases
        assert encoder._bits_to_float([], 0, 10) == 0  # Empty bits
        assert encoder._bits_to_float([0], 0, 10) == 0  # Single zero bit
        assert encoder._bits_to_float([1], 0, 10) == 10  # Single one bit
        
        # Test multi-bit conversion
        result = encoder._bits_to_float([1, 0, 1], 0, 7)
        expected = 5 / 7 * 7  # (101 binary = 5 decimal) / (2^3 - 1) * range
        assert abs(result - 5.0) < 0.1
    
    def test_atom_position_extraction(self, encoder, simple_ligand):
        """Test atomic position extraction from molecule"""
        atom_idx = 0
        position = encoder._get_atom_position(simple_ligand, atom_idx)
        
        assert isinstance(position, tuple)
        assert len(position) == 3
        assert all(isinstance(coord, float) for coord in position)
    
    def test_rotatable_bond_counting(self, encoder, complex_ligand):
        """Test rotatable bond counting"""
        count = encoder._count_rotatable_bonds(complex_ligand)
        
        assert isinstance(count, int)
        assert count >= 0
        
        # Ibuprofen should have several rotatable bonds
        assert count > 0
    
    def test_pharmacophore_complementarity(self, encoder):
        """Test pharmacophore complementarity checking"""
        # Test complementary pairs
        donor = {'type': 'hydrogen_bond_donor'}
        acceptor = {'type': 'hydrogen_bond_acceptor'}
        
        assert encoder._pharmacophores_complement(donor, acceptor)
        assert encoder._pharmacophores_complement(acceptor, donor)
        
        # Test non-complementary pairs
        hydrophobic1 = {'type': 'hydrophobic'}
        hydrophobic2 = {'type': 'hydrophobic'}
        
        assert encoder._pharmacophores_complement(hydrophobic1, hydrophobic2)
        
        # Test incompatible pairs
        aromatic = {'type': 'aromatic'}
        assert not encoder._pharmacophores_complement(donor, aromatic)
    
    def test_position_interaction_check(self, encoder):
        """Test position interaction checking"""
        # Close positions should interact
        assert encoder._positions_interact(0, 1)
        assert encoder._positions_interact(0, 3)
        
        # Distant positions should not interact
        assert not encoder._positions_interact(0, 10)
    
    def test_ring_center_calculation(self, encoder, complex_ligand):
        """Test aromatic ring center calculation"""
        # Find aromatic atoms to form a ring
        aromatic_atoms = [atom.GetIdx() for atom in complex_ligand.GetAtoms() 
                         if atom.GetIsAromatic()]
        
        if aromatic_atoms:
            # Take first 6 aromatic atoms as a ring
            ring = tuple(aromatic_atoms[:6])
            center = encoder._calculate_ring_center(complex_ligand, ring)
            
            assert isinstance(center, tuple)
            assert len(center) == 3
            assert all(isinstance(coord, float) for coord in center)
    
    def test_molecular_features_integration(self, encoder, complex_ligand):
        """Test integration with molecular feature extraction"""
        # Test that pharmacophore extraction works with various molecular features
        features = encoder._extract_ligand_pharmacophores(complex_ligand)
        
        # Should extract multiple types of features
        feature_types = set(f['type'] for f in features)
        assert len(feature_types) >= 2
        
        # All features should have required properties
        for feature in features:
            assert 'type' in feature
            assert 'position' in feature
            assert 'source' in feature
            
            if 'atom_idx' in feature:
                assert 0 <= feature['atom_idx'] < complex_ligand.GetNumAtoms()
    
    def test_edge_cases(self, encoder):
        """Test edge cases and error handling"""
        # Test with empty molecule (should not crash)
        empty_mol = Chem.MolFromSmiles('')
        if empty_mol is not None:
            try:
                features = encoder._extract_ligand_pharmacophores(empty_mol)
                assert isinstance(features, list)
            except:
                pass  # Expected to fail gracefully
        
        # Test with minimal molecule
        minimal_mol = Chem.MolFromSmiles('C')
        if minimal_mol is not None:
            minimal_mol = Chem.AddHs(minimal_mol)
            AllChem.EmbedMolecule(minimal_mol)
            
            features = encoder._extract_ligand_pharmacophores(minimal_mol)
            assert isinstance(features, list)
    
    def test_pharmacophore_type_coverage(self, encoder):
        """Test that all pharmacophore types are handled"""
        # Verify all expected pharmacophore types are defined
        expected_types = set(PHARMACOPHORE_TYPES.keys())
        encoder_types = set(encoder.pharmacophore_types.keys())
        
        assert expected_types == encoder_types
    
    @pytest.mark.parametrize("num_qubits", [10, 20, 30, 50])
    def test_qubo_scaling(self, encoder, simple_ligand, mock_protein, num_qubits):
        """Test QUBO encoding with different problem sizes"""
        pharmacophores = encoder.extract_pharmacophores(
            mock_protein, simple_ligand, None
        )
        
        # Add dummy pharmacophores to test scaling
        while len(pharmacophores) < num_qubits // 5:
            pharmacophores.append({
                'type': 'hydrophobic',
                'position': (0, 0, 0),
                'source': 'ligand'
            })
        
        qubo_matrix, offset = encoder.encode_docking_problem(
            mock_protein, simple_ligand, pharmacophores
        )
        
        # Matrix should scale appropriately
        assert qubo_matrix.shape[0] >= num_qubits // 10  # Rough scaling check
        assert np.all(np.isfinite(qubo_matrix))

if __name__ == "__main__":
    # Run tests if executed directly
    pytest.main([__file__, "-v"])
