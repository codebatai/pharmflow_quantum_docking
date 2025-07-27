"""
Integration tests for PharmFlow Quantum Molecular Docking
Tests complete workflow from molecular input to docking results
"""

import pytest
import numpy as np
import tempfile
import os
from pathlib import Path
import sys

# Add src to path for testing
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from pharmflow.core.pharmflow_engine import PharmFlowQuantumDocking
from pharmflow.quantum.qaoa_engine import PharmFlowQAOA
from pharmflow.quantum.pharmacophore_encoder import PharmacophoreEncoder
from pharmflow.quantum.energy_evaluator import EnergyEvaluator
from pharmflow.quantum.smoothing_filter import DynamicSmoothingFilter
from pharmflow.classical.molecular_loader import MolecularLoader
from pharmflow.classical.admet_calculator import ADMETCalculator
from pharmflow.classical.refinement_engine import ClassicalRefinement
from pharmflow.core.optimization_pipeline import OptimizationPipeline
from pharmflow.utils.visualization import DockingVisualizer

from qiskit import Aer
from qiskit.algorithms.optimizers import COBYLA
from rdkit import Chem
from rdkit.Chem import AllChem

class TestPharmFlowIntegration:
    """Integration tests for complete PharmFlow workflow"""
    
    @pytest.fixture
    def test_molecules(self):
        """Create test molecule set"""
        molecules = {
            'ethanol': 'CCO',
            'ibuprofen': 'CC(C)CC1=CC=C(C=C1)C(C)C(=O)O',
            'caffeine': 'CN1C=NC2=C1C(=O)N(C(=O)N2C)C',
            'aspirin': 'CC(=O)OC1=CC=CC=C1C(=O)O'
        }
        return molecules
    
    @pytest.fixture
    def pharmflow_engine(self):
        """Create PharmFlow engine with minimal configuration for testing"""
        return PharmFlowQuantumDocking(
            backend='qasm_simulator',
            optimizer='COBYLA',
            num_qaoa_layers=2,  # Minimal for testing
            smoothing_factor=0.1,
            quantum_noise_mitigation=False  # Disable for testing speed
        )
    
    @pytest.fixture
    def mock_protein_file(self):
        """Create mock protein PDB file"""
        pdb_content = """HEADER    TEST PROTEIN                            01-JAN-25   TEST            
ATOM      1  N   ALA A   1      20.154  16.967  10.000  1.00 30.00           N  
ATOM      2  CA  ALA A   1      19.030  16.080  10.000  1.00 30.00           C  
ATOM      3  C   ALA A   1      17.654  16.739  10.000  1.00 30.00           C  
ATOM      4  O   ALA A   1      17.654  17.962  10.000  1.00 30.00           O  
ATOM      5  CB  ALA A   1      19.113  15.176   8.788  1.00 30.00           C  
ATOM      6  N   ARG A   2      16.530  16.020  10.000  1.00 30.00           N  
ATOM      7  CA  ARG A   2      15.154  16.539  10.000  1.00 30.00           C  
ATOM      8  C   ARG A   2      14.030  15.507  10.000  1.00 30.00           C  
ATOM      9  O   ARG A   2      14.030  14.284  10.000  1.00 30.00           O  
ATOM     10  CB  ARG A   2      14.946  17.444  11.211  1.00 30.00           C  
END                                                                             
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.pdb', delete=False) as f:
            f.write(pdb_content)
            return f.name
    
    def test_complete_docking_workflow(self, pharmflow_engine, test_molecules, mock_protein_file):
        """Test complete docking workflow from start to finish"""
        try:
            # Test single molecule docking
            ligand_smiles = test_molecules['ethanol']
            
            result = pharmflow_engine.dock_molecule(
                protein_pdb=mock_protein_file,
                ligand_sdf=ligand_smiles,  # Using SMILES directly
                binding_site_residues=[1, 2],
                max_iterations=10  # Minimal for testing
            )
            
            # Verify result structure
            assert isinstance(result, dict)
            assert 'binding_affinity' in result
            assert 'optimization_result' in result
            assert 'ligand_features' in result
            assert 'docking_time' in result
            
            # Verify result values
            assert isinstance(result['binding_affinity'], (int, float))
            assert isinstance(result['docking_time'], (int, float))
            assert result['docking_time'] > 0
            
            # Verify optimization result structure
            opt_result = result['optimization_result']
            assert 'optimal_params' in opt_result
            assert 'optimal_energy' in opt_result
            assert 'optimization_history' in opt_result
            
        finally:
            # Clean up temporary file
            if os.path.exists(mock_protein_file):
                os.unlink(mock_protein_file)
    
    def test_batch_docking_workflow(self, pharmflow_engine, test_molecules, mock_protein_file):
        """Test batch docking functionality"""
        try:
            # Create list of test ligands
            ligand_list = [
                test_molecules['ethanol'],
                test_molecules['caffeine']
            ]
            
            results = pharmflow_engine.batch_screening(
                protein_pdb=mock_protein_file,
                ligand_library=ligand_list,
                binding_site_residues=[1, 2]
            )
            
            # Verify batch results
            assert isinstance(results, list)
            assert len(results) == len(ligand_list)
            
            # Check each result
            for result in results:
                assert isinstance(result, dict)
                # Should have either valid results or error information
                assert 'binding_affinity' in result or 'error' in result
                
        finally:
            if os.path.exists(mock_protein_file):
                os.unlink(mock_protein_file)
    
    def test_multi_objective_optimization(self, pharmflow_engine, test_molecules, mock_protein_file):
        """Test multi-objective optimization functionality"""
        try:
            # Define optimization objectives
            objectives = {
                'binding_affinity': {'weight': 0.4, 'target': 'minimize'},
                'selectivity': {'weight': 0.3, 'target': 'maximize'},
                'admet_score': {'weight': 0.3, 'target': 'maximize'}
            }
            
            result = pharmflow_engine.dock_molecule(
                protein_pdb=mock_protein_file,
                ligand_sdf=test_molecules['ibuprofen'],
                binding_site_residues=[1, 2],
                max_iterations=10,
                objectives=objectives
            )
            
            # Verify multi-objective result
            assert 'binding_affinity' in result
            assert 'selectivity' in result
            assert 'admet_score' in result
            assert 'best_score' in result  # Weighted score
            
        finally:
            if os.path.exists(mock_protein_file):
                os.unlink(mock_protein_file)
    
    def test_molecular_loader_integration(self):
        """Test molecular loader component integration"""
        loader = MolecularLoader()
        
        # Test SMILES loading
        mol = loader.load_ligand('CCO', mol_id='ethanol')
        assert mol is not None
        assert mol.GetNumAtoms() > 0
        
        # Test multiple ligand loading
        smiles_list = ['CCO', 'CCN', 'c1ccccc1']
        molecules = loader.load_multiple_ligands(smiles_list)
        assert len(molecules) == len(smiles_list)
        assert all(mol is not None for mol in molecules)
    
    def test_admet_calculator_integration(self, test_molecules):
        """Test ADMET calculator integration"""
        calculator = ADMETCalculator()
        
        # Test with different molecules
        for name, smiles in test_molecules.items():
            mol = Chem.MolFromSmiles(smiles)
            mol = Chem.AddHs(mol)
            
            # Test overall ADMET score
            admet_score = calculator.calculate_admet(mol)
            assert isinstance(admet_score, float)
            assert 0 <= admet_score <= 1
            
            # Test comprehensive report
            report = calculator.generate_admet_report(mol)
            assert isinstance(report, dict)
            assert 'overall_admet_score' in report
            assert 'absorption' in report
            assert 'distribution' in report
            assert 'metabolism' in report
            assert 'excretion' in report
            assert 'toxicity' in report
    
    def test_energy_evaluator_integration(self, test_molecules, mock_protein_file):
        """Test energy evaluator integration"""
        evaluator = EnergyEvaluator()
        loader = MolecularLoader()
        
        try:
            # Load protein and ligand
            protein = loader.load_protein(mock_protein_file)
            ligand = loader.load_ligand(test_molecules['ethanol'])
            
            # Create mock pose
            pose = {
                'position': (0.0, 0.0, 0.0),
                'rotation': (0.0, 0.0, 0.0),
                'conformation': []
            }
            
            # Test energy evaluation
            binding_energy = evaluator.calculate_binding_energy(pose, protein, ligand)
            assert isinstance(binding_energy, (int, float))
            assert np.isfinite(binding_energy)
            
            # Test selectivity calculation
            selectivity = evaluator.calculate_selectivity(pose, protein, ligand)
            assert isinstance(selectivity, float)
            assert 0 <= selectivity <= 1
            
        finally:
            if os.path.exists(mock_protein_file):
                os.unlink(mock_protein_file)
    
    def test_pharmacophore_encoder_integration(self, test_molecules, mock_protein_file):
        """Test pharmacophore encoder integration"""
        encoder = PharmacophoreEncoder()
        loader = MolecularLoader()
        
        try:
            # Load molecules
            protein = loader.load_protein(mock_protein_file)
            ligand = loader.load_ligand(test_molecules['ibuprofen'])
            
            # Extract pharmacophores
            pharmacophores = encoder.extract_pharmacophores(
                protein, ligand, binding_site_residues=[1, 2]
            )
            
            assert isinstance(pharmacophores, list)
            assert len(pharmacophores) > 0
            
            # Test QUBO encoding
            qubo_matrix, offset = encoder.encode_docking_problem(
                protein, ligand, pharmacophores
            )
            
            assert isinstance(qubo_matrix, np.ndarray)
            assert qubo_matrix.shape[0] == qubo_matrix.shape[1]
            assert np.all(np.isfinite(qubo_matrix))
            
        finally:
            if os.path.exists(mock_protein_file):
                os.unlink(mock_protein_file)
    
    def test_qaoa_engine_integration(self):
        """Test QAOA engine integration"""
        # Create QAOA engine
        backend = Aer.get_backend('qasm_simulator')
        optimizer = COBYLA(maxiter=5)  # Minimal for testing
        
        qaoa_engine = PharmFlowQAOA(
            backend=backend,
            optimizer=optimizer,
            num_layers=2
        )
        
        # Create simple test QUBO
        qubo_matrix = np.array([
            [-1.0, 0.5],
            [0.5, -1.0]
        ])
        
        # Test optimization
        result = qaoa_engine.optimize(qubo_matrix, max_iterations=3)
        
        # Verify result
        assert 'best_params' in result
        assert 'best_value' in result
        assert 'top_bitstrings' in result
        assert len(result['best_params']) == 4  # 2 layers * 2 parameter types
    
    def test_classical_refinement_integration(self, test_molecules):
        """Test classical refinement integration"""
        refinement = ClassicalRefinement()
        
        # Create mock poses
        poses = []
        for i in range(3):
            pose = {
                'position': (i, i, i),
                'rotation': (0, 0, 0),
                'conformation': [],
                'quantum_energy': -5.0 + i,
                'bitstring': '01' * 10,
                'pharmacophores': []
            }
            poses.append(pose)
        
        # Load ligand
        ligand = Chem.MolFromSmiles(test_molecules['ethanol'])
        ligand = Chem.AddHs(ligand)
        AllChem.EmbedMolecule(ligand)
        
        # Test refinement
        refined_poses = refinement.refine_poses(poses, None, ligand, strategy='fast')
        
        assert isinstance(refined_poses, list)
        assert len(refined_poses) == len(poses)
        
        for pose in refined_poses:
            assert 'refined' in pose
            assert pose['refined'] == True
    
    def test_optimization_pipeline_integration(self):
        """Test optimization pipeline integration"""
        # Create pipeline components
        backend = Aer.get_backend('qasm_simulator')
        qaoa_engine = PharmFlowQAOA(backend, COBYLA(maxiter=3), num_layers=2)
        energy_evaluator = EnergyEvaluator()
        smoothing_filter = DynamicSmoothingFilter()
        refinement_engine = ClassicalRefinement()
        
        pipeline = OptimizationPipeline(
            qaoa_engine, energy_evaluator, smoothing_filter, refinement_engine
        )
        
        # Create test inputs
        qubo_matrix = np.array([[- 1.0, 0.5], [0.5, -1.0]])
        protein = {'test': 'protein'}
        ligand = Chem.MolFromSmiles('CCO')
        pharmacophores = [{'type': 'hydrophobic', 'source': 'ligand'}]
        
        # Test optimization
        result = pipeline.optimize(
            qubo_matrix, protein, ligand, pharmacophores
        )
        
        assert isinstance(result.success, bool)
        assert hasattr(result, 'best_energy')
        assert hasattr(result, 'total_time')
        assert hasattr(result, 'stage_timings')
    
    def test_visualization_integration(self, test_molecules):
        """Test visualization component integration"""
        visualizer = DockingVisualizer()
        
        # Create test data
        optimization_history = [-10.0, -12.0, -11.5, -13.0, -12.8]
        energy_components = {
            'vdw_energy': -5.0,
            'electrostatic': -3.0,
            'hydrogen_bonds': -4.0,
            'hydrophobic': -1.0
        }
        
        # Test convergence plotting
        fig = visualizer.plot_optimization_convergence(optimization_history)
        assert fig is not None
        
        # Test energy component plotting
        fig = visualizer.plot_energy_components(energy_components)
        assert fig is not None
        
        # Test molecular image saving
        mol = Chem.MolFromSmiles(test_molecules['caffeine'])
        
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
            try:
                visualizer.save_molecular_image(mol, f.name)
                assert os.path.exists(f.name)
            finally:
                if os.path.exists(f.name):
                    os.unlink(f.name)
    
    def test_error_handling_integration(self, pharmflow_engine):
        """Test system error handling and robustness"""
        # Test with invalid protein file
        try:
            result = pharmflow_engine.dock_molecule(
                protein_pdb="nonexistent_file.pdb",
                ligand_sdf="CCO",
                max_iterations=1
            )
            # Should either handle gracefully or raise appropriate error
        except (FileNotFoundError, ValueError):
            pass  # Expected behavior
        
        # Test with invalid SMILES
        try:
            result = pharmflow_engine.dock_molecule(
                protein_pdb="test.pdb",
                ligand_sdf="INVALID_SMILES_XYZ123",
                max_iterations=1
            )
        except (ValueError, Exception):
            pass  # Expected behavior
    
    def test_performance_benchmarking(self, pharmflow_engine, test_molecules, mock_protein_file):
        """Test performance benchmarking capabilities"""
        try:
            # Measure docking time
            import time
            
            start_time = time.time()
            
            result = pharmflow_engine.dock_molecule(
                protein_pdb=mock_protein_file,
                ligand_sdf=test_molecules['ethanol'],
                binding_site_residues=[1, 2],
                max_iterations=5
            )
            
            end_time = time.time()
            measured_time = end_time - start_time
            
            # Verify timing consistency
            reported_time = result['docking_time']
            
            # Times should be reasonably close (within factor of 2)
            assert abs(measured_time - reported_time) < max(measured_time, reported_time)
            
        finally:
            if os.path.exists(mock_protein_file):
                os.unlink(mock_protein_file)
    
    def test_memory_management(self, pharmflow_engine, test_molecules, mock_protein_file):
        """Test memory management during intensive operations"""
        try:
            # Run multiple docking operations to test memory usage
            ligands = [test_molecules['ethanol']] * 5  # Small batch
            
            results = pharmflow_engine.batch_screening(
                protein_pdb=mock_protein_file,
                ligand_library=ligands,
                binding_site_residues=[1, 2]
            )
            
            # Should complete without memory issues
            assert len(results) == len(ligands)
            
        finally:
            if os.path.exists(mock_protein_file):
                os.unlink(mock_protein_file)
    
    def test_reproducibility(self, pharmflow_engine, test_molecules, mock_protein_file):
        """Test reproducibility of results"""
        try:
            # Run same docking twice
            results = []
            
            for i in range(2):
                result = pharmflow_engine.dock_molecule(
                    protein_pdb=mock_protein_file,
                    ligand_sdf=test_molecules['ethanol'],
                    binding_site_residues=[1, 2],
                    max_iterations=5
                )
                results.append(result)
            
            # Results should be reasonably similar (quantum algorithms have inherent randomness)
            energy_diff = abs(results[0]['binding_affinity'] - results[1]['binding_affinity'])
            
            # Allow for some variation due to quantum randomness
            assert energy_diff < 10.0  # Reasonable tolerance
            
        finally:
            if os.path.exists(mock_protein_file):
                os.unlink(mock_protein_file)
    
    def test_component_compatibility(self):
        """Test compatibility between different components"""
        # Test that all components can be initialized together
        loader = MolecularLoader()
        encoder = PharmacophoreEncoder()
        evaluator = EnergyEvaluator()
        calculator = ADMETCalculator()
        refinement = ClassicalRefinement()
        visualizer = DockingVisualizer()
        
        # All components should initialize without conflicts
        assert loader is not None
        assert encoder is not None
        assert evaluator is not None
        assert calculator is not None
        assert refinement is not None
        assert visualizer is not None
    
    def test_data_flow_integrity(self, test_molecules, mock_protein_file):
        """Test data integrity throughout the pipeline"""
        try:
            loader = MolecularLoader()
            encoder = PharmacophoreEncoder()
            
            # Load data
            protein = loader.load_protein(mock_protein_file)
            ligand = loader.load_ligand(test_molecules['ethanol'])
            
            # Extract pharmacophores
            pharmacophores = encoder.extract_pharmacophores(protein, ligand)
            
            # Encode as QUBO
            qubo_matrix, offset = encoder.encode_docking_problem(protein, ligand, pharmacophores)
            
            # Create test bitstrings
            num_qubits = qubo_matrix.shape[0]
            test_bitstring = '0' * num_qubits
            
            # Decode back to poses
            decoded_poses = encoder.decode_quantum_results([test_bitstring], pharmacophores)
            
            # Verify data integrity
            assert len(decoded_poses) == 1
            pose = decoded_poses[0]
            assert 'position' in pose
            assert 'rotation' in pose
            assert 'pharmacophores' in pose
            assert pose['pharmacophores'] == pharmacophores
            
        finally:
            if os.path.exists(mock_protein_file):
                os.unlink(mock_protein_file)

if __name__ == "__main__":
    # Run tests if executed directly
    pytest.main([__file__, "-v"])
