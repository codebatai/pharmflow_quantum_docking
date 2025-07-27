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
PharmFlow Quantum Molecular Docking Engine
Main orchestrator for QAOA-based quantum molecular docking
"""

import numpy as np
from typing import Dict, List, Any, Optional, Union, Tuple
import logging
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

from qiskit import Aer, IBMQ
from qiskit.algorithms.optimizers import COBYLA, SPSA, Adam
from rdkit import Chem

from ..quantum.qaoa_engine import PharmFlowQAOA
from ..quantum.pharmacophore_encoder import PharmacophoreEncoder
from ..quantum.energy_evaluator import EnergyEvaluator
from ..quantum.smoothing_filter import DynamicSmoothingFilter
from ..classical.molecular_loader import MolecularLoader
from ..classical.admet_calculator import ADMETCalculator
from ..classical.refinement_engine import ClassicalRefinement
from ..core.optimization_pipeline import OptimizationPipeline
from ..utils.constants import STANDARD_ENERGY_WEIGHTS, DEFAULT_QAOA_LAYERS

logger = logging.getLogger(__name__)

class PharmFlowQuantumDocking:
    """
    Main PharmFlow quantum molecular docking engine
    Integrates QAOA optimization with pharmacophore-guided molecular docking
    """
    
    def __init__(self,
                 backend: Union[str, Any] = 'qasm_simulator',
                 optimizer: Union[str, Any] = 'COBYLA',
                 num_qaoa_layers: int = DEFAULT_QAOA_LAYERS,
                 smoothing_factor: float = 0.1,
                 quantum_noise_mitigation: bool = True,
                 parallel_execution: bool = True):
        """
        Initialize PharmFlow quantum docking engine
        
        Args:
            backend: Quantum backend ('qasm_simulator', 'ibmq_qasm_simulator', etc.)
            optimizer: Classical optimizer ('COBYLA', 'SPSA', 'Adam')
            num_qaoa_layers: Number of QAOA layers
            smoothing_factor: Energy landscape smoothing factor
            quantum_noise_mitigation: Enable quantum noise mitigation
            parallel_execution: Enable parallel batch processing
        """
        self.logger = logging.getLogger(__name__)
        
        # Initialize quantum backend
        self.backend = self._setup_quantum_backend(backend)
        
        # Initialize classical optimizer
        self.optimizer = self._setup_classical_optimizer(optimizer)
        
        # Initialize core components
        self.molecular_loader = MolecularLoader()
        self.pharmacophore_encoder = PharmacophoreEncoder()
        self.energy_evaluator = EnergyEvaluator()
        self.smoothing_filter = DynamicSmoothingFilter(smoothing_factor)
        self.admet_calculator = ADMETCalculator()
        self.classical_refinement = ClassicalRefinement()
        
        # Initialize QAOA engine
        self.qaoa_engine = PharmFlowQAOA(
            backend=self.backend,
            optimizer=self.optimizer,
            num_layers=num_qaoa_layers,
            noise_mitigation=quantum_noise_mitigation
        )
        
        # Initialize optimization pipeline
        self.optimization_pipeline = OptimizationPipeline(
            self.qaoa_engine,
            self.energy_evaluator,
            self.smoothing_filter,
            self.classical_refinement
        )
        
        # Configuration
        self.parallel_execution = parallel_execution
        self.max_workers = 4
        
        # Results storage
        self.docking_results = []
        self.performance_metrics = {}
        
        self.logger.info(f"PharmFlow engine initialized with {backend} backend")
    
    def dock_molecule(self,
                     protein_pdb: str,
                     ligand_sdf: str,
                     binding_site_residues: Optional[List[int]] = None,
                     max_iterations: int = 200,
                     objectives: Optional[Dict[str, Dict]] = None,
                     refinement_strategy: str = 'comprehensive') -> Dict[str, Any]:
        """
        Perform quantum molecular docking for single molecule
        
        Args:
            protein_pdb: Path to protein PDB file
            ligand_sdf: Path to ligand SDF file or SMILES string
            binding_site_residues: List of binding site residue IDs
            max_iterations: Maximum QAOA iterations
            objectives: Multi-objective optimization weights
            refinement_strategy: Classical refinement strategy
            
        Returns:
            Comprehensive docking results
        """
        start_time = time.time()
        
        try:
            self.logger.info(f"Starting molecular docking: {protein_pdb} + {ligand_sdf}")
            
            # Load molecular structures
            protein = self._load_protein_structure(protein_pdb)
            ligand = self._load_ligand_structure(ligand_sdf)
            
            # Extract pharmacophores
            pharmacophores = self.pharmacophore_encoder.extract_pharmacophores(
                protein, ligand, binding_site_residues
            )
            
            # Encode docking problem as QUBO
            qubo_matrix, offset = self.pharmacophore_encoder.encode_docking_problem(
                protein, ligand, pharmacophores
            )
            
            # Apply dynamic smoothing
            smoothed_qubo = self.smoothing_filter.apply_smoothing_filter(qubo_matrix)
            
            # Execute quantum optimization
            optimization_result = self.optimization_pipeline.optimize(
                smoothed_qubo, protein, ligand, pharmacophores,
                optimization_config={'quantum_iterations': max_iterations}
            )
            
            # Calculate molecular properties
            ligand_features = self._extract_ligand_features(ligand)
            admet_score = self.admet_calculator.calculate_admet(ligand)
            
            # Multi-objective evaluation
            if objectives:
                multi_objective_score = self._evaluate_multi_objectives(
                    optimization_result, admet_score, objectives
                )
            else:
                multi_objective_score = optimization_result.best_energy
            
            # Compile comprehensive results
            docking_result = {
                'binding_affinity': optimization_result.best_energy,
                'selectivity': optimization_result.quantum_metrics.get('selectivity', 0.5),
                'admet_score': admet_score,
                'best_score': multi_objective_score,
                'ligand_features': ligand_features,
                'optimization_result': {
                    'optimal_params': optimization_result.best_parameters,
                    'optimal_energy': optimization_result.best_energy,
                    'optimization_history': optimization_result.optimization_history,
                    'num_iterations': len(optimization_result.optimization_history),
                    'convergence_metrics': optimization_result.convergence_metrics
                },
                'pharmacophores': pharmacophores,
                'qubo_properties': {
                    'matrix_size': qubo_matrix.shape[0],
                    'num_pharmacophores': len(pharmacophores),
                    'encoding_offset': offset
                },
                'docking_time': time.time() - start_time,
                'protein_path': protein_pdb,
                'ligand_path': ligand_sdf,
                'success': optimization_result.success
            }
            
            # Store result
            self.docking_results.append(docking_result)
            
            self.logger.info(f"Docking completed: binding_affinity={docking_result['binding_affinity']:.3f}")
            return docking_result
            
        except Exception as e:
            self.logger.error(f"Molecular docking failed: {e}")
            return {
                'error': str(e),
                'binding_affinity': float('inf'),
                'docking_time': time.time() - start_time,
                'success': False
            }
    
    def batch_screening(self,
                       protein_pdb: str,
                       ligand_library: List[str],
                       binding_site_residues: Optional[List[int]] = None,
                       max_iterations: int = 100,
                       top_n: int = 10) -> List[Dict[str, Any]]:
        """
        Perform batch screening of ligand library
        
        Args:
            protein_pdb: Path to protein PDB file
            ligand_library: List of ligand file paths or SMILES strings
            binding_site_residues: Binding site residue IDs
            max_iterations: Maximum iterations per ligand
            top_n: Number of top results to return
            
        Returns:
            Sorted list of docking results
        """
        self.logger.info(f"Starting batch screening: {len(ligand_library)} ligands")
        
        batch_results = []
        
        if self.parallel_execution:
            # Parallel processing
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                # Submit all docking jobs
                future_to_ligand = {
                    executor.submit(
                        self.dock_molecule,
                        protein_pdb, ligand, binding_site_residues, max_iterations
                    ): ligand for ligand in ligand_library
                }
                
                # Collect results as they complete
                for future in as_completed(future_to_ligand):
                    ligand = future_to_ligand[future]
                    try:
                        result = future.result()
                        result['ligand_id'] = ligand
                        batch_results.append(result)
                    except Exception as e:
                        self.logger.warning(f"Ligand {ligand} failed: {e}")
                        batch_results.append({
                            'ligand_id': ligand,
                            'error': str(e),
                            'binding_affinity': float('inf'),
                            'success': False
                        })
        else:
            # Sequential processing
            for i, ligand in enumerate(ligand_library):
                self.logger.info(f"Processing ligand {i+1}/{len(ligand_library)}: {ligand}")
                
                try:
                    result = self.dock_molecule(
                        protein_pdb, ligand, binding_site_residues, max_iterations
                    )
                    result['ligand_id'] = ligand
                    batch_results.append(result)
                    
                except Exception as e:
                    self.logger.warning(f"Ligand {ligand} failed: {e}")
                    batch_results.append({
                        'ligand_id': ligand,
                        'error': str(e),
                        'binding_affinity': float('inf'),
                        'success': False
                    })
        
        # Sort by binding affinity and return top results
        successful_results = [r for r in batch_results if r.get('success', False)]
        successful_results.sort(key=lambda x: x['binding_affinity'])
        
        top_results = successful_results[:top_n]
        
        self.logger.info(f"Batch screening completed: {len(successful_results)} successful, "
                        f"returning top {len(top_results)}")
        
        return top_results
    
    def virtual_screening_campaign(self,
                                  protein_pdb: str,
                                  compound_databases: List[str],
                                  binding_site_residues: Optional[List[int]] = None,
                                  filtering_criteria: Optional[Dict] = None,
                                  max_compounds: int = 1000) -> Dict[str, Any]:
        """
        Execute comprehensive virtual screening campaign
        
        Args:
            protein_pdb: Target protein structure
            compound_databases: List of compound database paths
            binding_site_residues: Binding site definition
            filtering_criteria: ADMET and drug-likeness filters
            max_compounds: Maximum compounds to screen
            
        Returns:
            Campaign results with hit identification
        """
        campaign_start = time.time()
        
        self.logger.info("Starting virtual screening campaign")
        
        # Load and filter compound library
        compound_library = self._load_compound_databases(
            compound_databases, filtering_criteria, max_compounds
        )
        
        # Perform batch screening
        screening_results = self.batch_screening(
            protein_pdb, compound_library, binding_site_residues
        )
        
        # Analyze results and identify hits
        hit_analysis = self._analyze_screening_hits(screening_results)
        
        # Generate campaign report
        campaign_results = {
            'total_compounds_screened': len(compound_library),
            'successful_dockings': len([r for r in screening_results if r.get('success')]),
            'identified_hits': hit_analysis['hits'],
            'hit_rate': hit_analysis['hit_rate'],
            'best_compounds': screening_results[:10],  # Top 10
            'campaign_time': time.time() - campaign_start,
            'performance_metrics': self._calculate_campaign_metrics(screening_results)
        }
        
        self.logger.info(f"Virtual screening campaign completed: "
                        f"{hit_analysis['num_hits']} hits identified")
        
        return campaign_results
    
    def optimize_lead_compound(self,
                              protein_pdb: str,
                              lead_compound: str,
                              optimization_objectives: Dict[str, float],
                              binding_site_residues: Optional[List[int]] = None) -> Dict[str, Any]:
        """
        Optimize lead compound for multiple objectives
        
        Args:
            protein_pdb: Target protein structure
            lead_compound: Lead compound SMILES or file path
            optimization_objectives: Objective weights
            binding_site_residues: Binding site definition
            
        Returns:
            Lead optimization results
        """
        self.logger.info(f"Starting lead optimization: {lead_compound}")
        
        # Perform detailed docking analysis
        detailed_result = self.dock_molecule(
            protein_pdb, lead_compound, binding_site_residues,
            max_iterations=500,  # More thorough optimization
            objectives=optimization_objectives,
            refinement_strategy='thorough'
        )
        
        # Generate optimization recommendations
        recommendations = self._generate_optimization_recommendations(
            detailed_result, optimization_objectives
        )
        
        lead_optimization_result = {
            'original_compound': lead_compound,
            'docking_result': detailed_result,
            'optimization_recommendations': recommendations,
            'structure_activity_analysis': self._analyze_structure_activity(detailed_result),
            'synthetic_accessibility': self._assess_synthetic_accessibility(lead_compound)
        }
        
        return lead_optimization_result
    
    # Private helper methods
    
    def _setup_quantum_backend(self, backend: Union[str, Any]) -> Any:
        """Setup quantum computing backend"""
        if isinstance(backend, str):
            if backend == 'qasm_simulator':
                return Aer.get_backend('qasm_simulator')
            elif backend.startswith('ibmq_'):
                try:
                    # Load IBMQ account if available
                    IBMQ.load_account()
                    provider = IBMQ.get_provider()
                    return provider.get_backend(backend)
                except:
                    self.logger.warning(f"IBMQ backend {backend} not available, using simulator")
                    return Aer.get_backend('qasm_simulator')
            else:
                return Aer.get_backend(backend)
        else:
            return backend
    
    def _setup_classical_optimizer(self, optimizer: Union[str, Any]) -> Any:
        """Setup classical optimizer"""
        if isinstance(optimizer, str):
            if optimizer == 'COBYLA':
                return COBYLA(maxiter=200)
            elif optimizer == 'SPSA':
                return SPSA(maxiter=200)
            elif optimizer == 'Adam':
                return Adam(maxiter=200)
            else:
                return COBYLA(maxiter=200)
        else:
            return optimizer
    
    def _load_protein_structure(self, protein_path: str) -> Dict[str, Any]:
        """Load and validate protein structure"""
        try:
            protein = self.molecular_loader.load_protein(protein_path)
            
            # Validate protein structure
            if not protein['atoms']:
                raise ValueError("Protein structure contains no atoms")
            
            return protein
            
        except Exception as e:
            self.logger.error(f"Failed to load protein {protein_path}: {e}")
            raise ValueError(f"Protein loading error: {e}")
    
    def _load_ligand_structure(self, ligand_path: str) -> Chem.Mol:
        """Load and validate ligand structure"""
        try:
            ligand = self.molecular_loader.load_ligand(ligand_path)
            
            # Validate ligand structure
            if ligand.GetNumAtoms() == 0:
                raise ValueError("Ligand contains no atoms")
            
            return ligand
            
        except Exception as e:
            self.logger.error(f"Failed to load ligand {ligand_path}: {e}")
            raise ValueError(f"Ligand loading error: {e}")
    
    def _extract_ligand_features(self, ligand: Chem.Mol) -> Dict[str, Any]:
        """Extract comprehensive ligand molecular features"""
        from rdkit.Chem import Descriptors, rdMolDescriptors
        
        try:
            features = {
                'molecular_weight': Descriptors.MolWt(ligand),
                'logp': Descriptors.MolLogP(ligand),
                'tpsa': Descriptors.TPSA(ligand),
                'num_atoms': ligand.GetNumAtoms(),
                'num_bonds': ligand.GetNumBonds(),
                'num_rings': rdMolDescriptors.CalcNumRings(ligand),
                'num_aromatic_rings': rdMolDescriptors.CalcNumAromaticRings(ligand),
                'num_rotatable_bonds': Descriptors.NumRotatableBonds(ligand),
                'num_hbd': Descriptors.NumHBD(ligand),
                'num_hba': Descriptors.NumHBA(ligand),
                'formal_charge': Chem.rdmolops.GetFormalCharge(ligand)
            }
            
            return features
            
        except Exception as e:
            self.logger.warning(f"Feature extraction failed: {e}")
            return {}
    
    def _evaluate_multi_objectives(self,
                                  optimization_result: Any,
                                  admet_score: float,
                                  objectives: Dict[str, Dict]) -> float:
        """Evaluate multi-objective optimization score"""
        total_score = 0.0
        total_weight = 0.0
        
        for objective, config in objectives.items():
            weight = config.get('weight', 1.0)
            target = config.get('target', 'minimize')
            
            if objective == 'binding_affinity':
                value = optimization_result.best_energy
                if target == 'minimize':
                    score = -value  # Convert to maximization
                else:
                    score = value
            elif objective == 'admet_score':
                score = admet_score
            elif objective == 'selectivity':
                score = optimization_result.quantum_metrics.get('selectivity', 0.5)
            else:
                continue
            
            total_score += weight * score
            total_weight += weight
        
        return total_score / total_weight if total_weight > 0 else 0.0
    
    def _load_compound_databases(self,
                                database_paths: List[str],
                                filtering_criteria: Optional[Dict],
                                max_compounds: int) -> List[str]:
        """Load and filter compound databases"""
        compounds = []
        
        for db_path in database_paths:
            try:
                if db_path.endswith('.sdf'):
                    # Load SDF file
                    suppl = Chem.SDMolSupplier(db_path)
                    for mol in suppl:
                        if mol is not None:
                            smiles = Chem.MolToSmiles(mol)
                            if self._passes_filters(mol, filtering_criteria):
                                compounds.append(smiles)
                                if len(compounds) >= max_compounds:
                                    break
                elif db_path.endswith('.smiles'):
                    # Load SMILES file
                    with open(db_path, 'r') as f:
                        for line in f:
                            smiles = line.strip()
                            mol = Chem.MolFromSmiles(smiles)
                            if mol is not None and self._passes_filters(mol, filtering_criteria):
                                compounds.append(smiles)
                                if len(compounds) >= max_compounds:
                                    break
                                
            except Exception as e:
                self.logger.warning(f"Failed to load database {db_path}: {e}")
        
        self.logger.info(f"Loaded {len(compounds)} compounds from databases")
        return compounds
    
    def _passes_filters(self, mol: Chem.Mol, filters: Optional[Dict]) -> bool:
        """Check if molecule passes filtering criteria"""
        if filters is None:
            return True
        
        try:
            from rdkit.Chem import Descriptors
            
            # Lipinski's Rule of Five
            if filters.get('lipinski', True):
                mw = Descriptors.MolWt(mol)
                logp = Descriptors.MolLogP(mol)
                hbd = Descriptors.NumHBD(mol)
                hba = Descriptors.NumHBA(mol)
                
                violations = 0
                if mw > 500: violations += 1
                if logp > 5: violations += 1
                if hbd > 5: violations += 1
                if hba > 10: violations += 1
                
                if violations > 1:  # Allow 1 violation
                    return False
            
            # ADMET pre-filtering
            if filters.get('admet_prefilter', False):
                admet_score = self.admet_calculator.calculate_admet(mol)
                if admet_score < filters.get('min_admet_score', 0.3):
                    return False
            
            return True
            
        except Exception:
            return False
    
    def _analyze_screening_hits(self, results: List[Dict]) -> Dict[str, Any]:
        """Analyze screening results to identify hits"""
        successful_results = [r for r in results if r.get('success', False)]
        
        if not successful_results:
            return {'hits': [], 'hit_rate': 0.0, 'num_hits': 0}
        
        # Define hit criteria
        binding_threshold = -5.0  # kcal/mol
        admet_threshold = 0.5
        
        hits = []
        for result in successful_results:
            if (result['binding_affinity'] < binding_threshold and
                result.get('admet_score', 0) > admet_threshold):
                hits.append(result)
        
        hit_rate = len(hits) / len(successful_results)
        
        return {
            'hits': hits,
            'hit_rate': hit_rate,
            'num_hits': len(hits),
            'binding_threshold': binding_threshold,
            'admet_threshold': admet_threshold
        }
    
    def _calculate_campaign_metrics(self, results: List[Dict]) -> Dict[str, float]:
        """Calculate performance metrics for screening campaign"""
        successful_results = [r for r in results if r.get('success', False)]
        
        if not successful_results:
            return {}
        
        binding_affinities = [r['binding_affinity'] for r in successful_results]
        docking_times = [r['docking_time'] for r in successful_results]
        
        return {
            'success_rate': len(successful_results) / len(results),
            'mean_binding_affinity': np.mean(binding_affinities),
            'std_binding_affinity': np.std(binding_affinities),
            'best_binding_affinity': np.min(binding_affinities),
            'mean_docking_time': np.mean(docking_times),
            'total_screening_time': np.sum(docking_times),
            'throughput': len(results) / np.sum(docking_times) * 3600  # compounds/hour
        }
    
    def _generate_optimization_recommendations(self,
                                             result: Dict,
                                             objectives: Dict) -> List[str]:
        """Generate lead optimization recommendations"""
        recommendations = []
        
        # Binding affinity recommendations
        if result['binding_affinity'] > -7.0:
            recommendations.append("Consider adding hydrophobic substituents to improve binding")
        
        # ADMET recommendations
        if result['admet_score'] < 0.6:
            admet_report = self.admet_calculator.generate_admet_report(
                self.molecular_loader.load_ligand(result['ligand_path'])
            )
            
            if admet_report['absorption']['score'] < 0.5:
                recommendations.append("Improve absorption by optimizing LogP and TPSA")
            
            if admet_report['toxicity']['score'] < 0.5:
                recommendations.append("Address potential toxicity concerns")
        
        return recommendations
    
    def _analyze_structure_activity(self, result: Dict) -> Dict[str, Any]:
        """Analyze structure-activity relationships"""
        return {
            'key_interactions': self._identify_key_interactions(result),
            'pharmacophore_contributions': self._analyze_pharmacophore_contributions(result),
            'binding_mode_analysis': self._analyze_binding_mode(result)
        }
    
    def _identify_key_interactions(self, result: Dict) -> List[str]:
        """Identify key protein-ligand interactions"""
        # Simplified interaction analysis
        interactions = []
        
        for pharmacophore in result.get('pharmacophores', []):
            if pharmacophore.get('source') == 'ligand':
                interactions.append(f"{pharmacophore['type']} interaction")
        
        return interactions
    
    def _analyze_pharmacophore_contributions(self, result: Dict) -> Dict[str, float]:
        """Analyze individual pharmacophore contributions to binding"""
        contributions = {}
        
        for pharmacophore in result.get('pharmacophores', []):
            ptype = pharmacophore['type']
            contributions[ptype] = contributions.get(ptype, 0) + 1
        
        # Normalize by total
        total = sum(contributions.values())
        if total > 0:
            contributions = {k: v/total for k, v in contributions.items()}
        
        return contributions
    
    def _analyze_binding_mode(self, result: Dict) -> Dict[str, Any]:
        """Analyze binding mode characteristics"""
        return {
            'binding_energy_components': result.get('optimization_result', {}).get('energy_components', {}),
            'pose_stability': self._assess_pose_stability(result),
            'interaction_network': self._map_interaction_network(result)
        }
    
    def _assess_pose_stability(self, result: Dict) -> float:
        """Assess binding pose stability"""
        # Simplified stability assessment
        convergence_metrics = result.get('optimization_result', {}).get('convergence_metrics', {})
        return convergence_metrics.get('convergence_rate', 0.5)
    
    def _map_interaction_network(self, result: Dict) -> Dict[str, Any]:
        """Map protein-ligand interaction network"""
        return {
            'num_interactions': len(result.get('pharmacophores', [])),
            'interaction_types': list(set(p['type'] for p in result.get('pharmacophores', []))),
            'network_density': 0.5  # Simplified calculation
        }
    
    def _assess_synthetic_accessibility(self, compound: str) -> Dict[str, Any]:
        """Assess synthetic accessibility of compound"""
        try:
            mol = Chem.MolFromSmiles(compound)
            if mol is None:
                return {'accessible': False, 'score': 0.0}
            
            # Simplified synthetic accessibility assessment
            from rdkit.Chem import Descriptors
            
            complexity_score = (
                Descriptors.NumRotatableBonds(mol) * 0.1 +
                mol.GetNumAtoms() * 0.01 +
                len(Chem.GetMolFrags(mol)) * 0.2
            )
            
            accessibility_score = max(0, 1 - complexity_score / 10)
            
            return {
                'accessible': accessibility_score > 0.3,
                'score': accessibility_score,
                'complexity_factors': {
                    'rotatable_bonds': Descriptors.NumRotatableBonds(mol),
                    'heavy_atoms': mol.GetNumAtoms(),
                    'fragments': len(Chem.GetMolFrags(mol))
                }
            }
            
        except Exception as e:
            self.logger.warning(f"Synthetic accessibility assessment failed: {e}")
            return {'accessible': True, 'score': 0.5}
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get engine performance summary"""
        if not self.docking_results:
            return {}
        
        successful_results = [r for r in self.docking_results if r.get('success', True)]
        
        if not successful_results:
            return {'success_rate': 0.0}
        
        binding_affinities = [r['binding_affinity'] for r in successful_results]
        docking_times = [r['docking_time'] for r in successful_results]
        
        return {
            'total_dockings': len(self.docking_results),
            'successful_dockings': len(successful_results),
            'success_rate': len(successful_results) / len(self.docking_results),
            'mean_binding_affinity': np.mean(binding_affinities),
            'best_binding_affinity': np.min(binding_affinities),
            'mean_docking_time': np.mean(docking_times),
            'total_computation_time': np.sum(docking_times),
            'throughput': len(self.docking_results) / np.sum(docking_times) * 3600
        }
