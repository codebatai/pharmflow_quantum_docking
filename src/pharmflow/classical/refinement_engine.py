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
Classical refinement engine for quantum molecular docking results
Post-processing optimization using molecular dynamics and energy minimization
"""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Callable
import logging
from scipy.optimize import minimize, differential_evolution
from rdkit import Chem
from rdkit.Chem import AllChem, rdMolAlign, rdForceFieldHelpers
from rdkit.Chem.rdForceFieldHelpers import UFFOptimizeMolecule, MMFFOptimizeMolecule
import copy

from ..utils.constants import (
    CLASSICAL_MAX_ITERATIONS, CLASSICAL_CONVERGENCE_TOLERANCE,
    CLASSICAL_STEP_SIZE, MD_TEMPERATURE, MD_TIME_STEP
)

logger = logging.getLogger(__name__)

class ClassicalRefinement:
    """
    Advanced classical refinement engine for quantum docking results
    Implements multiple optimization strategies for pose improvement
    """
    
    def __init__(self):
        """Initialize classical refinement engine"""
        self.logger = logging.getLogger(__name__)
        
        # Optimization parameters
        self.max_iterations = CLASSICAL_MAX_ITERATIONS
        self.convergence_tolerance = CLASSICAL_CONVERGENCE_TOLERANCE
        self.step_size = CLASSICAL_STEP_SIZE
        
        # Force field preferences (in order of preference)
        self.force_fields = ['MMFF94', 'UFF']
        
        # Refinement strategies
        self.refinement_strategies = [
            'local_minimization',
            'conformational_search',
            'molecular_dynamics',
            'monte_carlo'
        ]
        
        self.logger.info("Classical refinement engine initialized")
    
    def refine_poses(self, 
                    poses: List[Dict], 
                    protein: Any, 
                    ligand: Chem.Mol,
                    strategy: str = 'comprehensive') -> List[Dict]:
        """
        Refine quantum-generated poses using classical methods
        
        Args:
            poses: List of quantum-generated poses
            protein: Protein structure
            ligand: Ligand molecule
            strategy: Refinement strategy ('fast', 'comprehensive', 'thorough')
            
        Returns:
            List of refined poses
        """
        if not poses:
            self.logger.warning("No poses provided for refinement")
            return []
        
        try:
            refined_poses = []
            
            for i, pose in enumerate(poses):
                self.logger.debug(f"Refining pose {i+1}/{len(poses)}")
                
                try:
                    refined_pose = self._refine_single_pose(
                        pose, protein, ligand, strategy
                    )
                    refined_poses.append(refined_pose)
                    
                except Exception as e:
                    self.logger.warning(f"Failed to refine pose {i}: {e}")
                    # Keep original pose if refinement fails
                    refined_poses.append(pose)
            
            # Sort by refined energy
            refined_poses = self._sort_poses_by_energy(refined_poses)
            
            self.logger.info(f"Successfully refined {len(refined_poses)} poses")
            return refined_poses
            
        except Exception as e:
            self.logger.error(f"Pose refinement failed: {e}")
            return poses  # Return original poses if refinement fails
    
    def _refine_single_pose(self, 
                           pose: Dict, 
                           protein: Any, 
                           ligand: Chem.Mol,
                           strategy: str) -> Dict:
        """
        Refine a single molecular pose
        
        Args:
            pose: Single pose dictionary
            protein: Protein structure
            ligand: Ligand molecule
            strategy: Refinement strategy
            
        Returns:
            Refined pose dictionary
        """
        # Apply pose to ligand molecule
        positioned_ligand = self._apply_pose_to_ligand(ligand, pose)
        
        # Select refinement methods based on strategy
        methods = self._get_refinement_methods(strategy)
        
        # Initialize refinement result
        best_ligand = positioned_ligand
        best_energy = float('inf')
        refinement_history = []
        
        # Apply refinement methods sequentially
        for method in methods:
            try:
                refined_ligand, energy = self._apply_refinement_method(
                    best_ligand, protein, method
                )
                
                if energy < best_energy:
                    best_ligand = refined_ligand
                    best_energy = energy
                
                refinement_history.append({
                    'method': method,
                    'energy': energy,
                    'improvement': best_energy - energy if refinement_history else 0
                })
                
            except Exception as e:
                self.logger.warning(f"Refinement method {method} failed: {e}")
        
        # Extract refined pose parameters
        refined_pose = self._extract_pose_from_ligand(best_ligand, pose)
        refined_pose.update({
            'refined_energy': best_energy,
            'refinement_history': refinement_history,
            'refinement_improvement': pose.get('quantum_energy', 0) - best_energy,
            'refined': True
        })
        
        return refined_pose
    
    def _get_refinement_methods(self, strategy: str) -> List[str]:
        """Get refinement methods based on strategy"""
        strategies = {
            'fast': ['local_minimization'],
            'comprehensive': ['local_minimization', 'conformational_search'],
            'thorough': ['local_minimization', 'conformational_search', 'molecular_dynamics']
        }
        
        return strategies.get(strategy, strategies['comprehensive'])
    
    def _apply_refinement_method(self, 
                                ligand: Chem.Mol, 
                                protein: Any, 
                                method: str) -> Tuple[Chem.Mol, float]:
        """
        Apply specific refinement method
        
        Args:
            ligand: Positioned ligand molecule
            protein: Protein structure
            method: Refinement method name
            
        Returns:
            Refined ligand and energy
        """
        if method == 'local_minimization':
            return self._local_energy_minimization(ligand, protein)
        elif method == 'conformational_search':
            return self._conformational_search(ligand, protein)
        elif method == 'molecular_dynamics':
            return self._molecular_dynamics_refinement(ligand, protein)
        elif method == 'monte_carlo':
            return self._monte_carlo_refinement(ligand, protein)
        else:
            raise ValueError(f"Unknown refinement method: {method}")
    
    def _local_energy_minimization(self, 
                                  ligand: Chem.Mol, 
                                  protein: Any) -> Tuple[Chem.Mol, float]:
        """
        Perform local energy minimization using force fields
        
        Args:
            ligand: Ligand molecule
            protein: Protein structure
            
        Returns:
            Minimized ligand and energy
        """
        try:
            # Create working copy
            mol = Chem.Mol(ligand)
            
            # Try different force fields in order of preference
            for ff_name in self.force_fields:
                try:
                    if ff_name == 'MMFF94':
                        ff = self._setup_mmff94(mol)
                    elif ff_name == 'UFF':
                        ff = self._setup_uff(mol)
                    else:
                        continue
                    
                    if ff is not None:
                        # Perform minimization
                        initial_energy = ff.CalcEnergy()
                        converged = ff.Minimize(maxIts=self.max_iterations)
                        final_energy = ff.CalcEnergy()
                        
                        if converged == 0:  # Successful convergence
                            self.logger.debug(f"{ff_name} minimization converged: "
                                            f"{initial_energy:.3f} → {final_energy:.3f}")
                            return mol, final_energy
                        
                except Exception as e:
                    self.logger.debug(f"{ff_name} minimization failed: {e}")
                    continue
            
            # If all force fields fail, return original molecule with high energy
            self.logger.warning("All force field minimizations failed")
            return mol, 1000.0
            
        except Exception as e:
            self.logger.error(f"Local minimization failed: {e}")
            return ligand, float('inf')
    
    def _conformational_search(self, 
                              ligand: Chem.Mol, 
                              protein: Any) -> Tuple[Chem.Mol, float]:
        """
        Perform systematic conformational search
        
        Args:
            ligand: Ligand molecule
            protein: Protein structure
            
        Returns:
            Best conformer and energy
        """
        try:
            mol = Chem.Mol(ligand)
            
            # Generate multiple conformers
            num_conformers = min(10, max(3, mol.GetNumBonds() // 3))
            
            # Remove existing conformers
            mol.RemoveAllConformers()
            
            # Generate new conformers
            conf_ids = AllChem.EmbedMultipleConfs(
                mol, 
                numConfs=num_conformers,
                randomSeed=42,
                useExpTorsionAnglePrefs=True,
                useBasicKnowledge=True,
                enforceChirality=True
            )
            
            if not conf_ids:
                self.logger.warning("Failed to generate conformers")
                return mol, float('inf')
            
            # Optimize each conformer and find the best one
            best_conf_id = None
            best_energy = float('inf')
            
            for conf_id in conf_ids:
                try:
                    # Set up force field for this conformer
                    ff = self._setup_force_field(mol, conf_id)
                    
                    if ff is not None:
                        # Minimize conformer
                        ff.Minimize(maxIts=100)
                        energy = ff.CalcEnergy()
                        
                        if energy < best_energy:
                            best_energy = energy
                            best_conf_id = conf_id
                            
                except Exception as e:
                    self.logger.debug(f"Conformer {conf_id} optimization failed: {e}")
            
            # Keep only the best conformer
            if best_conf_id is not None:
                # Remove all conformers except the best one
                conf_to_keep = mol.GetConformer(best_conf_id)
                mol.RemoveAllConformers()
                mol.AddConformer(conf_to_keep, assignId=True)
                
                return mol, best_energy
            else:
                return mol, float('inf')
                
        except Exception as e:
            self.logger.error(f"Conformational search failed: {e}")
            return ligand, float('inf')
    
    def _molecular_dynamics_refinement(self, 
                                     ligand: Chem.Mol, 
                                     protein: Any) -> Tuple[Chem.Mol, float]:
        """
        Perform simplified molecular dynamics refinement
        
        Args:
            ligand: Ligand molecule
            protein: Protein structure
            
        Returns:
            MD-refined ligand and energy
        """
        try:
            mol = Chem.Mol(ligand)
            
            # Set up force field
            ff = self._setup_force_field(mol)
            
            if ff is None:
                return mol, float('inf')
            
            # Simplified MD simulation using repeated minimization with perturbations
            best_mol = mol
            best_energy = ff.CalcEnergy()
            
            num_md_steps = 50
            temperature_factor = MD_TEMPERATURE / 300.0  # Scale relative to room temperature
            
            for step in range(num_md_steps):
                # Apply small random perturbations (simulating thermal motion)
                perturbed_mol = self._apply_random_perturbation(mol, temperature_factor)
                
                # Minimize perturbed structure
                perturbed_ff = self._setup_force_field(perturbed_mol)
                
                if perturbed_ff is not None:
                    perturbed_ff.Minimize(maxIts=20)
                    energy = perturbed_ff.CalcEnergy()
                    
                    # Accept if energy is lower (simplified Metropolis criterion)
                    if energy < best_energy:
                        best_mol = perturbed_mol
                        best_energy = energy
            
            return best_mol, best_energy
            
        except Exception as e:
            self.logger.error(f"MD refinement failed: {e}")
            return ligand, float('inf')
    
    def _monte_carlo_refinement(self, 
                               ligand: Chem.Mol, 
                               protein: Any) -> Tuple[Chem.Mol, float]:
        """
        Perform Monte Carlo refinement
        
        Args:
            ligand: Ligand molecule
            protein: Protein structure
            
        Returns:
            MC-refined ligand and energy
        """
        try:
            mol = Chem.Mol(ligand)
            
            # Set up force field
            ff = self._setup_force_field(mol)
            
            if ff is None:
                return mol, float('inf')
            
            current_energy = ff.CalcEnergy()
            best_mol = mol
            best_energy = current_energy
            
            num_mc_steps = 100
            temperature = 300.0  # K
            kb = 0.001987  # kcal/(mol·K)
            
            accepted_moves = 0
            
            for step in range(num_mc_steps):
                # Generate trial move
                trial_mol = self._generate_trial_move(mol)
                
                # Calculate trial energy
                trial_ff = self._setup_force_field(trial_mol)
                
                if trial_ff is not None:
                    trial_ff.Minimize(maxIts=10)
                    trial_energy = trial_ff.CalcEnergy()
                    
                    # Metropolis acceptance criterion
                    delta_e = trial_energy - current_energy
                    
                    if delta_e < 0 or np.random.random() < np.exp(-delta_e / (kb * temperature)):
                        # Accept move
                        mol = trial_mol
                        current_energy = trial_energy
                        accepted_moves += 1
                        
                        # Update best if this is the best so far
                        if trial_energy < best_energy:
                            best_mol = trial_mol
                            best_energy = trial_energy
            
            acceptance_rate = accepted_moves / num_mc_steps
            self.logger.debug(f"MC refinement acceptance rate: {acceptance_rate:.2f}")
            
            return best_mol, best_energy
            
        except Exception as e:
            self.logger.error(f"Monte Carlo refinement failed: {e}")
            return ligand, float('inf')
    
    def _setup_mmff94(self, mol: Chem.Mol, conf_id: int = 0) -> Optional[Any]:
        """Setup MMFF94 force field"""
        try:
            props = AllChem.MMFFGetMoleculeProperties(mol)
            if props is None:
                return None
            
            ff = AllChem.MMFFGetMoleculeForceField(mol, props, confId=conf_id)
            return ff
            
        except Exception:
            return None
    
    def _setup_uff(self, mol: Chem.Mol, conf_id: int = 0) -> Optional[Any]:
        """Setup UFF force field"""
        try:
            ff = AllChem.UFFGetMoleculeForceField(mol, confId=conf_id)
            return ff
            
        except Exception:
            return None
    
    def _setup_force_field(self, mol: Chem.Mol, conf_id: int = 0) -> Optional[Any]:
        """Setup best available force field"""
        # Try MMFF94 first
        ff = self._setup_mmff94(mol, conf_id)
        if ff is not None:
            return ff
        
        # Fall back to UFF
        return self._setup_uff(mol, conf_id)
    
    def _apply_pose_to_ligand(self, ligand: Chem.Mol, pose: Dict) -> Chem.Mol:
        """
        Apply pose transformation to ligand molecule
        
        Args:
            ligand: Original ligand molecule
            pose: Pose dictionary with position and rotation
            
        Returns:
            Positioned ligand molecule
        """
        try:
            mol = Chem.Mol(ligand)
            
            # Ensure molecule has a conformer
            if mol.GetNumConformers() == 0:
                AllChem.EmbedMolecule(mol, randomSeed=42)
            
            conf = mol.GetConformer()
            
            # Apply translation
            if 'position' in pose:
                translation = np.array(pose['position'])
                
                # Calculate current center
                current_center = self._calculate_molecule_center(mol)
                
                # Apply translation
                for i in range(mol.GetNumAtoms()):
                    pos = conf.GetAtomPosition(i)
                    current_pos = np.array([pos.x, pos.y, pos.z])
                    new_pos = current_pos - current_center + translation
                    conf.SetAtomPosition(i, new_pos.tolist())
            
            # Apply rotation
            if 'rotation' in pose:
                alpha, beta, gamma = pose['rotation']
                rotation_matrix = self._euler_to_rotation_matrix(alpha, beta, gamma)
                
                # Get updated center after translation
                center = self._calculate_molecule_center(mol)
                
                # Apply rotation around center
                for i in range(mol.GetNumAtoms()):
                    pos = conf.GetAtomPosition(i)
                    centered_pos = np.array([pos.x, pos.y, pos.z]) - center
                    rotated_pos = rotation_matrix @ centered_pos + center
                    conf.SetAtomPosition(i, rotated_pos.tolist())
            
            # Apply conformational changes
            if 'conformation' in pose and pose['conformation']:
                self._apply_conformational_changes(mol, pose['conformation'])
            
            return mol
            
        except Exception as e:
            self.logger.error(f"Failed to apply pose to ligand: {e}")
            return ligand
    
    def _apply_conformational_changes(self, mol: Chem.Mol, conformation: List[float]):
        """Apply conformational angle changes to rotatable bonds"""
        try:
            # Get rotatable bonds
            rotatable_bonds = self._get_rotatable_bonds(mol)
            
            # Apply angles to rotatable bonds
            for i, (bond_idx, angle) in enumerate(zip(rotatable_bonds, conformation)):
                if i >= len(conformation):
                    break
                
                bond = mol.GetBondWithIdx(bond_idx)
                self._set_dihedral_angle(mol, bond, angle)
                
        except Exception as e:
            self.logger.warning(f"Failed to apply conformational changes: {e}")
    
    def _get_rotatable_bonds(self, mol: Chem.Mol) -> List[int]:
        """Get indices of rotatable bonds"""
        rotatable_bonds = []
        
        for bond in mol.GetBonds():
            if (not bond.GetIsAromatic() and 
                not bond.IsInRing() and 
                bond.GetBondType() == Chem.BondType.SINGLE):
                
                atom1 = bond.GetBeginAtom()
                atom2 = bond.GetEndAtom()
                
                # Check if both atoms have other neighbors (not terminal)
                if (len(atom1.GetNeighbors()) > 1 and 
                    len(atom2.GetNeighbors()) > 1):
                    rotatable_bonds.append(bond.GetIdx())
        
        return rotatable_bonds
    
    def _set_dihedral_angle(self, mol: Chem.Mol, bond: Chem.Bond, angle: float):
        """Set dihedral angle around a bond"""
        try:
            # This is a simplified implementation
            # In practice, would need more sophisticated dihedral manipulation
            pass
        except Exception:
            pass
    
    def _extract_pose_from_ligand(self, ligand: Chem.Mol, original_pose: Dict) -> Dict:
        """
        Extract pose parameters from positioned ligand
        
        Args:
            ligand: Positioned ligand molecule
            original_pose: Original pose for reference
            
        Returns:
            Updated pose dictionary
        """
        try:
            pose = original_pose.copy()
            
            # Calculate new center of mass
            center = self._calculate_molecule_center(ligand)
            pose['position'] = tuple(center.tolist())
            
            # For rotation, we'll keep the original rotation
            # (extracting rotation from coordinates is complex)
            
            return pose
            
        except Exception as e:
            self.logger.error(f"Failed to extract pose from ligand: {e}")
            return original_pose
    
    def _calculate_molecule_center(self, mol: Chem.Mol) -> np.ndarray:
        """Calculate geometric center of molecule"""
        try:
            conf = mol.GetConformer()
            coords = []
            
            for i in range(mol.GetNumAtoms()):
                pos = conf.GetAtomPosition(i)
                coords.append([pos.x, pos.y, pos.z])
            
            return np.mean(coords, axis=0)
            
        except Exception:
            return np.zeros(3)
    
    def _euler_to_rotation_matrix(self, alpha: float, beta: float, gamma: float) -> np.ndarray:
        """Convert Euler angles to rotation matrix"""
        ca, sa = np.cos(alpha), np.sin(alpha)
        cb, sb = np.cos(beta), np.sin(beta)
        cg, sg = np.cos(gamma), np.sin(gamma)
        
        R = np.array([
            [ca*cb*cg - sa*sg, -ca*cb*sg - sa*cg, ca*sb],
            [sa*cb*cg + ca*sg, -sa*cb*sg + ca*cg, sa*sb],
            [-sb*cg, sb*sg, cb]
        ])
        
        return R
    
    def _apply_random_perturbation(self, mol: Chem.Mol, temperature_factor: float) -> Chem.Mol:
        """Apply random perturbation to molecule coordinates"""
        try:
            perturbed_mol = Chem.Mol(mol)
            conf = perturbed_mol.GetConformer()
            
            # Apply small random displacements
            displacement_scale = 0.1 * temperature_factor  # Angstroms
            
            for i in range(perturbed_mol.GetNumAtoms()):
                pos = conf.GetAtomPosition(i)
                
                # Generate random displacement
                displacement = np.random.normal(0, displacement_scale, 3)
                
                new_pos = np.array([pos.x, pos.y, pos.z]) + displacement
                conf.SetAtomPosition(i, new_pos.tolist())
            
            return perturbed_mol
            
        except Exception:
            return mol
    
    def _generate_trial_move(self, mol: Chem.Mol) -> Chem.Mol:
        """Generate trial move for Monte Carlo"""
        # Apply random perturbation
        return self._apply_random_perturbation(mol, 1.0)
    
    def _sort_poses_by_energy(self, poses: List[Dict]) -> List[Dict]:
        """Sort poses by refined energy"""
        try:
            return sorted(poses, key=lambda x: x.get('refined_energy', float('inf')))
        except Exception:
            return poses
    
    def calculate_refinement_metrics(self, 
                                   original_poses: List[Dict], 
                                   refined_poses: List[Dict]) -> Dict[str, float]:
        """
        Calculate refinement quality metrics
        
        Args:
            original_poses: Original quantum poses
            refined_poses: Refined poses
            
        Returns:
            Refinement metrics dictionary
        """
        if not original_poses or not refined_poses:
            return {}
        
        try:
            # Energy improvements
            original_energies = [p.get('quantum_energy', 0) for p in original_poses]
            refined_energies = [p.get('refined_energy', 0) for p in refined_poses]
            
            valid_pairs = [(o, r) for o, r in zip(original_energies, refined_energies) 
                          if o != 0 and r != float('inf')]
            
            if not valid_pairs:
                return {}
            
            improvements = [o - r for o, r in valid_pairs]
            
            metrics = {
                'mean_improvement': np.mean(improvements),
                'median_improvement': np.median(improvements),
                'max_improvement': np.max(improvements),
                'std_improvement': np.std(improvements),
                'improvement_rate': len([i for i in improvements if i > 0]) / len(improvements),
                'mean_original_energy': np.mean([p[0] for p in valid_pairs]),
                'mean_refined_energy': np.mean([p[1] for p in valid_pairs])
            }
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Failed to calculate refinement metrics: {e}")
            return {}
