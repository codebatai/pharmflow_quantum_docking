"""
Test suite for PharmFlow QAOA Engine
Comprehensive testing of quantum optimization functionality
"""

import pytest
import numpy as np
from qiskit import Aer
from qiskit.algorithms.optimizers import COBYLA
import sys
import os

# Add src to path for testing
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from pharmflow.quantum.qaoa_engine import PharmFlowQAOA
from pharmflow.utils.constants import DEFAULT_QAOA_LAYERS, MEASUREMENT_SHOTS

class TestPharmFlowQAOA:
    """Test cases for PharmFlow QAOA engine"""
    
    @pytest.fixture
    def qaoa_engine(self):
        """Create QAOA engine for testing"""
        backend = Aer.get_backend('qasm_simulator')
        optimizer = COBYLA(maxiter=10)  # Small number for fast testing
        return PharmFlowQAOA(
            backend=backend,
            optimizer=optimizer,
            num_layers=2,  # Small for testing
            mixer_strategy='X_mixer'
        )
    
    @pytest.fixture
    def simple_qubo_matrix(self):
        """Create simple QUBO matrix for testing"""
        # Simple 4x4 QUBO matrix
        qubo = np.array([
            [-1.0,  0.5,  0.0,  0.0],
            [ 0.5, -1.0,  0.5,  0.0],
            [ 0.0,  0.5, -1.0,  0.5],
            [ 0.0,  0.0,  0.5, -1.0]
        ])
        return qubo
    
    @pytest.fixture
    def complex_qubo_matrix(self):
        """Create more complex QUBO matrix"""
        size = 6
        np.random.seed(42)  # For reproducible tests
        
        # Generate symmetric matrix
        A = np.random.randn(size, size)
        qubo = (A + A.T) / 2
        
        # Make diagonal negative (typical for optimization problems)
        np.fill_diagonal(qubo, -np.abs(np.diag(qubo)))
        
        return qubo
    
    def test_qaoa_initialization(self, qaoa_engine):
        """Test QAOA engine initialization"""
        assert qaoa_engine.num_layers == 2
        assert qaoa_engine.mixer_strategy == 'X_mixer'
        assert qaoa_engine.backend is not None
        assert qaoa_engine.optimizer is not None
    
    def test_qubo_to_pauli_operator_conversion(self, qaoa_engine, simple_qubo_matrix):
        """Test QUBO to Pauli operator conversion"""
        pauli_op = qaoa_engine._qubo_to_pauli_operator(simple_qubo_matrix)
        
        # Check that operator is created
        assert pauli_op is not None
        
        # Check that the operator has the right number of qubits
        assert len(pauli_op.primitive.paulis[0]) == simple_qubo_matrix.shape[0]
    
    def test_qaoa_circuit_construction(self, qaoa_engine, simple_qubo_matrix):
        """Test QAOA circuit construction"""
        # Create parameters
        num_qubits = simple_qubo_matrix.shape[0]
        theta = np.random.uniform(0, 2*np.pi, qaoa_engine.num_layers)
        gamma = np.random.uniform(0, 2*np.pi, qaoa_engine.num_layers)
        
        # Convert QUBO to Pauli operator
        cost_operator = qaoa_engine._qubo_to_pauli_operator(simple_qubo_matrix)
        
        # Build circuit
        qc = qaoa_engine.build_pharmflow_qaoa_circuit(theta, gamma, cost_operator, num_qubits)
        
        # Check circuit properties
        assert qc.num_qubits == num_qubits
        assert qc.num_clbits == num_qubits
        
        # Check that circuit has gates (not empty)
        assert len(qc.data) > 0
        
        # Check that circuit starts with Hadamard gates (superposition)
        h_gates = [op for op in qc.data if op[0].name == 'h']
        assert len(h_gates) == num_qubits
    
    def test_mixer_strategies(self, qaoa_engine, simple_qubo_matrix):
        """Test different mixer strategies"""
        strategies = ['X_mixer', 'XY_mixer', 'weighted_pharmacophore']
        num_qubits = simple_qubo_matrix.shape[0]
        
        for strategy in strategies:
            qaoa_engine.mixer_strategy = strategy
            
            # Test circuit construction with this strategy
            theta = np.random.uniform(0, np.pi, qaoa_engine.num_layers)
            gamma = np.random.uniform(0, np.pi, qaoa_engine.num_layers)
            cost_operator = qaoa_engine._qubo_to_pauli_operator(simple_qubo_matrix)
            
            qc = qaoa_engine.build_pharmflow_qaoa_circuit(theta, gamma, cost_operator, num_qubits)
            
            # Circuit should be valid
            assert qc.num_qubits == num_qubits
            assert len(qc.data) > 0
    
    def test_optimization_execution(self, qaoa_engine, simple_qubo_matrix):
        """Test complete optimization execution"""
        # Run optimization with minimal iterations for testing
        result = qaoa_engine.optimize(
            simple_qubo_matrix,
            max_iterations=5,  # Small number for testing
            initial_params=None
        )
        
        # Check result structure
        assert 'best_params' in result
        assert 'best_value' in result
        assert 'optimization_history' in result
        assert 'top_bitstrings' in result
        assert 'num_qubits' in result
        
        # Check parameter dimensions
        expected_param_length = 2 * qaoa_engine.num_layers
        assert len(result['best_params']) == expected_param_length
        
        # Check that we got some bitstrings
        assert len(result['top_bitstrings']) > 0
        
        # Check bitstring format
        for bitstring in result['top_bitstrings']:
            assert isinstance(bitstring, str)
            assert len(bitstring) == simple_qubo_matrix.shape[0]
            assert all(bit in '01' for bit in bitstring)
    
    def test_expectation_value_calculation(self, qaoa_engine, simple_qubo_matrix):
        """Test expectation value calculation from measurement counts"""
        # Create mock measurement counts
        num_qubits = simple_qubo_matrix.shape[0]
        counts = {
            '0000': 1000,
            '0001': 800,
            '1010': 600,
            '1111': 400
        }
        
        cost_operator = qaoa_engine._qubo_to_pauli_operator(simple_qubo_matrix)
        expectation = qaoa_engine._calculate_expectation_value(counts, cost_operator, num_qubits)
        
        # Expectation value should be a finite number
        assert np.isfinite(expectation)
        assert isinstance(expectation, (int, float))
    
    def test_configuration_energy_evaluation(self, qaoa_engine, simple_qubo_matrix):
        """Test energy evaluation for specific configurations"""
        cost_operator = qaoa_engine._qubo_to_pauli_operator(simple_qubo_matrix)
        
        # Test different configurations
        configurations = [
            [0, 0, 0, 0],
            [1, 1, 1, 1],
            [1, 0, 1, 0],
            [0, 1, 0, 1]
        ]
        
        for config in configurations:
            energy = qaoa_engine._evaluate_configuration_energy(config, cost_operator)
            assert np.isfinite(energy)
            assert isinstance(energy, (int, float))
    
    def test_pauli_evolution_application(self, qaoa_engine):
        """Test Pauli evolution gate application"""
        from qiskit import QuantumCircuit
        
        # Create test circuit
        qc = QuantumCircuit(4, 4)
        qc.h(range(4))  # Initialize in superposition
        
        # Test single Z evolution
        qaoa_engine._apply_pauli_evolution(qc, 0.5, 'ZIIII', 4)
        
        # Test ZZ evolution
        qaoa_engine._apply_pauli_evolution(qc, 0.3, 'ZZIIII', 4)
        
        # Circuit should have additional gates
        assert len(qc.data) > 4  # More than just the H gates
    
    def test_multi_z_evolution(self, qaoa_engine):
        """Test multi-qubit Z evolution decomposition"""
        from qiskit import QuantumCircuit
        
        qc = QuantumCircuit(4)
        z_qubits = [0, 1, 3]
        
        qaoa_engine._apply_multi_z_evolution(qc, 0.25, z_qubits)
        
        # Should have CNOT and RZ gates
        cnot_count = sum(1 for op in qc.data if op[0].name == 'cx')
        rz_count = sum(1 for op in qc.data if op[0].name == 'rz')
        
        assert cnot_count > 0
        assert rz_count > 0
    
    def test_optimization_with_initial_parameters(self, qaoa_engine, simple_qubo_matrix):
        """Test optimization with provided initial parameters"""
        # Provide specific initial parameters
        num_params = 2 * qaoa_engine.num_layers
        initial_params = np.ones(num_params) * 0.5
        
        result = qaoa_engine.optimize(
            simple_qubo_matrix,
            max_iterations=3,
            initial_params=initial_params
        )
        
        # Should complete successfully
        assert 'best_params' in result
        assert len(result['best_params']) == num_params
    
    def test_complex_qubo_optimization(self, qaoa_engine, complex_qubo_matrix):
        """Test optimization on more complex QUBO problems"""
        result = qaoa_engine.optimize(
            complex_qubo_matrix,
            max_iterations=5
        )
        
        # Should handle larger problems
        assert result['num_qubits'] == complex_qubo_matrix.shape[0]
        assert len(result['top_bitstrings']) > 0
        
        # Check bitstring lengths
        for bitstring in result['top_bitstrings']:
            assert len(bitstring) == complex_qubo_matrix.shape[0]
    
    def test_pharmacophore_weights(self, qaoa_engine):
        """Test pharmacophore weight calculation"""
        num_qubits = 6
        weights = qaoa_engine._get_pharmacophore_weights(num_qubits)
        
        assert len(weights) == num_qubits
        assert all(isinstance(w, (int, float)) for w in weights)
        assert all(w > 0 for w in weights)
    
    def test_empty_qubo_matrix(self, qaoa_engine):
        """Test handling of edge cases"""
        # Test with minimal QUBO matrix
        minimal_qubo = np.array([[-1.0]])
        
        result = qaoa_engine.optimize(minimal_qubo, max_iterations=2)
        
        assert result['num_qubits'] == 1
        assert len(result['best_params']) == 2 * qaoa_engine.num_layers
    
    def test_symmetric_qubo_matrix(self, qaoa_engine):
        """Test that QUBO matrix symmetry is handled correctly"""
        # Create asymmetric matrix
        asymmetric = np.array([
            [-1.0, 0.5],
            [0.3, -1.0]
        ])
        
        # Convert to Pauli operator (should work despite asymmetry)
        pauli_op = qaoa_engine._qubo_to_pauli_operator(asymmetric)
        assert pauli_op is not None
    
    def test_optimization_convergence_tracking(self, qaoa_engine, simple_qubo_matrix):
        """Test that optimization properly tracks convergence"""
        result = qaoa_engine.optimize(
            simple_qubo_matrix,
            max_iterations=10
        )
        
        history = result['optimization_history']
        
        # Should have some history
        assert len(history) > 0
        
        # History should contain finite values
        assert all(np.isfinite(val) for val in history)
    
    @pytest.mark.parametrize("num_layers", [1, 2, 3])
    def test_different_qaoa_layers(self, num_layers):
        """Test QAOA with different numbers of layers"""
        backend = Aer.get_backend('qasm_simulator')
        optimizer = COBYLA(maxiter=5)
        
        qaoa_engine = PharmFlowQAOA(
            backend=backend,
            optimizer=optimizer,
            num_layers=num_layers
        )
        
        # Simple 2x2 QUBO for quick testing
        qubo = np.array([[-1.0, 0.5], [0.5, -1.0]])
        
        result = qaoa_engine.optimize(qubo, max_iterations=3)
        
        # Should work with any reasonable number of layers
        assert len(result['best_params']) == 2 * num_layers
        assert result['num_qubits'] == 2

    def test_error_handling(self, qaoa_engine):
        """Test error handling for invalid inputs"""
        # Test with invalid QUBO matrix
        with pytest.raises((ValueError, np.linalg.LinAlgError)):
            invalid_qubo = np.array([[np.inf, np.nan], [np.nan, np.inf]])
            qaoa_engine.optimize(invalid_qubo)
        
        # Test with mismatched dimensions
        with pytest.raises((ValueError, IndexError)):
            qaoa_engine.optimize(np.array([]))

if __name__ == "__main__":
    # Run tests if executed directly
    pytest.main([__file__, "-v"])
