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
ADMET (Absorption, Distribution, Metabolism, Excretion, Toxicity) Calculator
Comprehensive drug-likeness and pharmacokinetic property prediction
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import logging
from rdkit import Chem
from rdkit.Chem import Descriptors, Crippen, rdMolDescriptors, QED
from rdkit.Chem.FilterCatalog import FilterCatalog, FilterCatalogParams
from rdkit.Chem.Scaffolds import MurckoScaffold
import warnings

from ..utils.constants import (
    LIPINSKI_MW_MAX, LIPINSKI_LOGP_MAX, LIPINSKI_HBD_MAX, LIPINSKI_HBA_MAX,
    VEBER_TPSA_MAX, VEBER_ROTBONDS_MAX, EGAN_LOGP_RANGE, EGAN_TPSA_RANGE,
    SOLUBILITY_THRESHOLD, PERMEABILITY_THRESHOLD, BBB_THRESHOLD
)

logger = logging.getLogger(__name__)

class ADMETCalculator:
    """
    Comprehensive ADMET property calculator for drug-like molecules
    """
    
    def __init__(self):
        """Initialize ADMET calculator with validated models"""
        self.logger = logging.getLogger(__name__)
        
        # Initialize filter catalogs for toxicity screening
        self._setup_filter_catalogs()
        
        # Molecular descriptor cache
        self._descriptor_cache = {}
        
        # ADMET model parameters (based on literature)
        self._setup_admet_models()
        
        self.logger.info("ADMET calculator initialized")
    
    def calculate_admet(self, molecule: Chem.Mol) -> float:
        """
        Calculate comprehensive ADMET score (0-1 scale, higher is better)
        
        Args:
            molecule: RDKit molecule object
            
        Returns:
            Overall ADMET score
        """
        try:
            # Calculate individual ADMET components
            absorption_score = self.calculate_absorption_properties(molecule)['score']
            distribution_score = self.calculate_distribution_properties(molecule)['score']
            metabolism_score = self.calculate_metabolism_properties(molecule)['score']
            excretion_score = self.calculate_excretion_properties(molecule)['score']
            toxicity_score = self.calculate_toxicity_properties(molecule)['score']
            
            # Weighted average (based on pharmaceutical importance)
            weights = {
                'absorption': 0.25,
                'distribution': 0.20,
                'metabolism': 0.20,
                'excretion': 0.15,
                'toxicity': 0.20
            }
            
            admet_score = (
                weights['absorption'] * absorption_score +
                weights['distribution'] * distribution_score +
                weights['metabolism'] * metabolism_score +
                weights['excretion'] * excretion_score +
                weights['toxicity'] * toxicity_score
            )
            
            self.logger.debug(f"ADMET score calculated: {admet_score:.3f}")
            return admet_score
            
        except Exception as e:
            self.logger.error(f"ADMET calculation failed: {e}")
            return 0.0
    
    def calculate_absorption_properties(self, molecule: Chem.Mol) -> Dict[str, Any]:
        """
        Calculate absorption-related properties
        
        Args:
            molecule: RDKit molecule object
            
        Returns:
            Dictionary of absorption properties
        """
        try:
            # Basic molecular descriptors
            mw = Descriptors.MolWt(molecule)
            logp = Descriptors.MolLogP(molecule)
            tpsa = Descriptors.TPSA(molecule)
            hba = Descriptors.NumHBA(molecule)
            hbd = Descriptors.NumHBD(molecule)
            rotbonds = Descriptors.NumRotatableBonds(molecule)
            
            # Lipinski's Rule of Five compliance
            lipinski_violations = 0
            if mw > LIPINSKI_MW_MAX:
                lipinski_violations += 1
            if logp > LIPINSKI_LOGP_MAX:
                lipinski_violations += 1
            if hbd > LIPINSKI_HBD_MAX:
                lipinski_violations += 1
            if hba > LIPINSKI_HBA_MAX:
                lipinski_violations += 1
            
            lipinski_score = max(0, 1 - lipinski_violations / 4)
            
            # Veber's criteria for oral bioavailability
            veber_compliant = (tpsa <= VEBER_TPSA_MAX and rotbonds <= VEBER_ROTBONDS_MAX)
            veber_score = 1.0 if veber_compliant else 0.5
            
            # Egan's criteria
            egan_compliant = (
                EGAN_LOGP_RANGE[0] <= logp <= EGAN_LOGP_RANGE[1] and
                EGAN_TPSA_RANGE[0] <= tpsa <= EGAN_TPSA_RANGE[1]
            )
            egan_score = 1.0 if egan_compliant else 0.5
            
            # Solubility prediction (simplified LogS)
            logs_predicted = self._predict_solubility(molecule)
            solubility_score = self._sigmoid_transform(logs_predicted, SOLUBILITY_THRESHOLD, 2.0)
            
            # Permeability prediction (Caco-2)
            caco2_predicted = self._predict_caco2_permeability(molecule)
            permeability_score = self._sigmoid_transform(caco2_predicted, PERMEABILITY_THRESHOLD, 1.0)
            
            # Combine absorption scores
            absorption_score = (
                0.3 * lipinski_score +
                0.2 * veber_score +
                0.2 * egan_score +
                0.15 * solubility_score +
                0.15 * permeability_score
            )
            
            return {
                'score': absorption_score,
                'molecular_weight': mw,
                'logp': logp,
                'tpsa': tpsa,
                'hba': hba,
                'hbd': hbd,
                'rotatable_bonds': rotbonds,
                'lipinski_violations': lipinski_violations,
                'veber_compliant': veber_compliant,
                'egan_compliant': egan_compliant,
                'predicted_logs': logs_predicted,
                'predicted_caco2': caco2_predicted
            }
            
        except Exception as e:
            self.logger.error(f"Absorption calculation failed: {e}")
            return {'score': 0.0}
    
    def calculate_distribution_properties(self, molecule: Chem.Mol) -> Dict[str, Any]:
        """
        Calculate distribution-related properties
        
        Args:
            molecule: RDKit molecule object
            
        Returns:
            Dictionary of distribution properties
        """
        try:
            # Volume of distribution predictors
            logp = Descriptors.MolLogP(molecule)
            mw = Descriptors.MolWt(molecule)
            tpsa = Descriptors.TPSA(molecule)
            
            # Plasma protein binding prediction
            ppb_predicted = self._predict_plasma_protein_binding(molecule)
            
            # Blood-brain barrier penetration
            bbb_predicted = self._predict_bbb_penetration(molecule)
            bbb_score = self._sigmoid_transform(bbb_predicted, BBB_THRESHOLD, 0.5)
            
            # Central nervous system penetration
            cns_predicted = self._predict_cns_penetration(molecule)
            
            # Tissue distribution score (based on LogP and TPSA)
            tissue_score = self._calculate_tissue_distribution_score(logp, tpsa, mw)
            
            # Combine distribution scores
            distribution_score = (
                0.3 * (1 - ppb_predicted / 100) +  # Lower PPB is better for distribution
                0.25 * bbb_score +
                0.25 * cns_predicted +
                0.2 * tissue_score
            )
            
            return {
                'score': distribution_score,
                'predicted_ppb': ppb_predicted,
                'predicted_bbb': bbb_predicted,
                'predicted_cns': cns_predicted,
                'tissue_distribution_score': tissue_score
            }
            
        except Exception as e:
            self.logger.error(f"Distribution calculation failed: {e}")
            return {'score': 0.0}
    
    def calculate_metabolism_properties(self, molecule: Chem.Mol) -> Dict[str, Any]:
        """
        Calculate metabolism-related properties
        
        Args:
            molecule: RDKit molecule object
            
        Returns:
            Dictionary of metabolism properties
        """
        try:
            # CYP inhibition predictions
            cyp_inhibitions = {}
            cyp_isoforms = ['1A2', '2C9', '2C19', '2D6', '3A4']
            
            for isoform in cyp_isoforms:
                inhibition_prob = self._predict_cyp_inhibition(molecule, isoform)
                cyp_inhibitions[f'CYP{isoform}'] = inhibition_prob
            
            # Average CYP inhibition risk
            avg_cyp_risk = np.mean(list(cyp_inhibitions.values()))
            cyp_score = 1 - avg_cyp_risk  # Lower inhibition risk is better
            
            # Metabolic stability prediction
            stability_predicted = self._predict_metabolic_stability(molecule)
            
            # Phase I metabolism sites
            phase1_sites = self._identify_phase1_sites(molecule)
            
            # Phase II conjugation potential
            phase2_potential = self._assess_phase2_conjugation(molecule)
            
            # Combine metabolism scores
            metabolism_score = (
                0.4 * cyp_score +
                0.3 * stability_predicted +
                0.2 * (1 - len(phase1_sites) / max(molecule.GetNumAtoms(), 1)) +
                0.1 * phase2_potential
            )
            
            return {
                'score': metabolism_score,
                'cyp_inhibitions': cyp_inhibitions,
                'predicted_stability': stability_predicted,
                'phase1_sites': phase1_sites,
                'phase2_potential': phase2_potential
            }
            
        except Exception as e:
            self.logger.error(f"Metabolism calculation failed: {e}")
            return {'score': 0.0}
    
    def calculate_excretion_properties(self, molecule: Chem.Mol) -> Dict[str, Any]:
        """
        Calculate excretion-related properties
        
        Args:
            molecule: RDKit molecule object
            
        Returns:
            Dictionary of excretion properties
        """
        try:
            # Renal clearance prediction
            renal_clearance = self._predict_renal_clearance(molecule)
            
            # Half-life prediction
            half_life = self._predict_half_life(molecule)
            
            # Biliary excretion potential
            biliary_excretion = self._predict_biliary_excretion(molecule)
            
            # Clearance score (moderate clearance is optimal)
            clearance_score = self._optimal_range_score(renal_clearance, 1.0, 10.0)
            
            # Half-life score (moderate half-life is optimal)
            half_life_score = self._optimal_range_score(half_life, 2.0, 12.0)
            
            # Combine excretion scores
            excretion_score = (
                0.4 * clearance_score +
                0.4 * half_life_score +
                0.2 * biliary_excretion
            )
            
            return {
                'score': excretion_score,
                'predicted_renal_clearance': renal_clearance,
                'predicted_half_life': half_life,
                'predicted_biliary_excretion': biliary_excretion
            }
            
        except Exception as e:
            self.logger.error(f"Excretion calculation failed: {e}")
            return {'score': 0.0}
    
    def calculate_toxicity_properties(self, molecule: Chem.Mol) -> Dict[str, Any]:
        """
        Calculate toxicity-related properties
        
        Args:
            molecule: RDKit molecule object
            
        Returns:
            Dictionary of toxicity properties
        """
        try:
            # PAINS (Pan Assay Interference Compounds) filtering
            pains_alerts = self._check_pains_alerts(molecule)
            
            # Structural alerts for toxicity
            tox_alerts = self._check_toxicity_alerts(molecule)
            
            # Mutagenicity prediction (Ames test)
            mutagenicity = self._predict_mutagenicity(molecule)
            
            # hERG inhibition risk
            herg_risk = self._predict_herg_inhibition(molecule)
            
            # Hepatotoxicity prediction
            hepatotoxicity = self._predict_hepatotoxicity(molecule)
            
            # Calculate toxicity score (absence of toxicity is better)
            pains_score = 1.0 if len(pains_alerts) == 0 else max(0, 1 - len(pains_alerts) / 10)
            tox_alerts_score = 1.0 if len(tox_alerts) == 0 else max(0, 1 - len(tox_alerts) / 5)
            mutagenicity_score = 1 - mutagenicity
            herg_score = 1 - herg_risk
            hepatotox_score = 1 - hepatotoxicity
            
            toxicity_score = (
                0.2 * pains_score +
                0.2 * tox_alerts_score +
                0.25 * mutagenicity_score +
                0.2 * herg_score +
                0.15 * hepatotox_score
            )
            
            return {
                'score': toxicity_score,
                'pains_alerts': pains_alerts,
                'toxicity_alerts': tox_alerts,
                'predicted_mutagenicity': mutagenicity,
                'predicted_herg_risk': herg_risk,
                'predicted_hepatotoxicity': hepatotoxicity
            }
            
        except Exception as e:
            self.logger.error(f"Toxicity calculation failed: {e}")
            return {'score': 0.0}
    
    def calculate_qed_score(self, molecule: Chem.Mol) -> float:
        """
        Calculate Quantitative Estimate of Drug-likeness (QED)
        
        Args:
            molecule: RDKit molecule object
            
        Returns:
            QED score (0-1)
        """
        try:
            qed_score = QED.qed(molecule)
            return qed_score
        except Exception as e:
            self.logger.error(f"QED calculation failed: {e}")
            return 0.0
    
    def generate_admet_report(self, molecule: Chem.Mol) -> Dict[str, Any]:
        """
        Generate comprehensive ADMET report
        
        Args:
            molecule: RDKit molecule object
            
        Returns:
            Complete ADMET analysis report
        """
        report = {
            'overall_admet_score': self.calculate_admet(molecule),
            'qed_score': self.calculate_qed_score(molecule),
            'absorption': self.calculate_absorption_properties(molecule),
            'distribution': self.calculate_distribution_properties(molecule),
            'metabolism': self.calculate_metabolism_properties(molecule),
            'excretion': self.calculate_excretion_properties(molecule),
            'toxicity': self.calculate_toxicity_properties(molecule)
        }
        
        # Add interpretation
        report['interpretation'] = self._interpret_admet_scores(report)
        
        return report
    
    # Private methods for specific predictions
    
    def _setup_filter_catalogs(self):
        """Setup filter catalogs for structural alerts"""
        try:
            # PAINS filters
            pains_params = FilterCatalogParams()
            pains_params.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS)
            self.pains_catalog = FilterCatalog(pains_params)
            
            # Toxicity filters
            tox_params = FilterCatalogParams()
            tox_params.AddCatalog(FilterCatalogParams.FilterCatalogs.BRENK)
            self.tox_catalog = FilterCatalog(tox_params)
            
        except Exception as e:
            self.logger.warning(f"Filter catalog setup failed: {e}")
            self.pains_catalog = None
            self.tox_catalog = None
    
    def _setup_admet_models(self):
        """Setup ADMET prediction model parameters"""
        # Model coefficients based on literature (simplified)
        self.solubility_coeffs = {
            'logp': -0.72,
            'mw': -0.0067,
            'tpsa': 0.0085,
            'rotbonds': -0.24,
            'intercept': -0.77
        }
        
        self.caco2_coeffs = {
            'logp': 0.31,
            'tpsa': -0.01,
            'hbd': -0.4,
            'intercept': -4.5
        }
        
        self.bbb_coeffs = {
            'logp': 0.15,
            'tpsa': -0.006,
            'mw': -0.001,
            'intercept': -0.1
        }
    
    def _predict_solubility(self, molecule: Chem.Mol) -> float:
        """Predict aqueous solubility (LogS)"""
        try:
            logp = Descriptors.MolLogP(molecule)
            mw = Descriptors.MolWt(molecule)
            tpsa = Descriptors.TPSA(molecule)
            rotbonds = Descriptors.NumRotatableBonds(molecule)
            
            logs = (
                self.solubility_coeffs['logp'] * logp +
                self.solubility_coeffs['mw'] * mw +
                self.solubility_coeffs['tpsa'] * tpsa +
                self.solubility_coeffs['rotbonds'] * rotbonds +
                self.solubility_coeffs['intercept']
            )
            
            return logs
            
        except Exception:
            return -3.0  # Default poor solubility
    
    def _predict_caco2_permeability(self, molecule: Chem.Mol) -> float:
        """Predict Caco-2 permeability"""
        try:
            logp = Descriptors.MolLogP(molecule)
            tpsa = Descriptors.TPSA(molecule)
            hbd = Descriptors.NumHBD(molecule)
            
            caco2 = (
                self.caco2_coeffs['logp'] * logp +
                self.caco2_coeffs['tpsa'] * tpsa +
                self.caco2_coeffs['hbd'] * hbd +
                self.caco2_coeffs['intercept']
            )
            
            return caco2
            
        except Exception:
            return -6.0  # Default poor permeability
    
    def _predict_bbb_penetration(self, molecule: Chem.Mol) -> float:
        """Predict blood-brain barrier penetration"""
        try:
            logp = Descriptors.MolLogP(molecule)
            tpsa = Descriptors.TPSA(molecule)
            mw = Descriptors.MolWt(molecule)
            
            bbb = (
                self.bbb_coeffs['logp'] * logp +
                self.bbb_coeffs['tpsa'] * tpsa +
                self.bbb_coeffs['mw'] * mw +
                self.bbb_coeffs['intercept']
            )
            
            return self._sigmoid(bbb)
            
        except Exception:
            return 0.1  # Default low BBB penetration
    
    def _predict_plasma_protein_binding(self, molecule: Chem.Mol) -> float:
        """Predict plasma protein binding percentage"""
        try:
            logp = Descriptors.MolLogP(molecule)
            
            # Simplified model based on lipophilicity
            ppb = 85 + 10 * self._sigmoid(logp - 2)
            return min(99, max(10, ppb))
            
        except Exception:
            return 90.0  # Default high binding
    
    def _predict_cns_penetration(self, molecule: Chem.Mol) -> float:
        """Predict CNS penetration score"""
        try:
            # Based on CNS MPO (Multi-Parameter Optimization)
            logp = Descriptors.MolLogP(molecule)
            logd = logp  # Simplified, should be pH-adjusted
            mw = Descriptors.MolWt(molecule)
            tpsa = Descriptors.TPSA(molecule)
            hbd = Descriptors.NumHBD(molecule)
            pka = 8.0  # Simplified, should be calculated
            
            # CNS MPO scoring
            logp_score = self._cns_score_transform(logp, 1, 3)
            logd_score = self._cns_score_transform(logd, 1, 3)
            mw_score = self._cns_score_transform(mw, 360, 500, reverse=True)
            tpsa_score = self._cns_score_transform(tpsa, 40, 90, reverse=True)
            hbd_score = self._cns_score_transform(hbd, 0.5, 3.5, reverse=True)
            pka_score = self._cns_score_transform(pka, 8, 10)
            
            cns_score = (logp_score + logd_score + mw_score + tpsa_score + hbd_score + pka_score) / 6
            return cns_score
            
        except Exception:
            return 0.3  # Default moderate CNS penetration
    
    def _calculate_tissue_distribution_score(self, logp: float, tpsa: float, mw: float) -> float:
        """Calculate tissue distribution score"""
        # Optimal ranges for tissue distribution
        logp_optimal = self._optimal_range_score(logp, 1.0, 3.0)
        tpsa_optimal = self._optimal_range_score(tpsa, 40, 120)
        mw_optimal = self._optimal_range_score(mw, 150, 500)
        
        return (logp_optimal + tpsa_optimal + mw_optimal) / 3
    
    def _predict_cyp_inhibition(self, molecule: Chem.Mol, isoform: str) -> float:
        """Predict CYP inhibition probability"""
        try:
            # Simplified model based on molecular properties
            logp = Descriptors.MolLogP(molecule)
            mw = Descriptors.MolWt(molecule)
            
            # Different isoforms have different susceptibilities
            if isoform == '3A4':
                prob = self._sigmoid((logp - 2) + (mw - 400) / 200)
            elif isoform == '2D6':
                prob = self._sigmoid((logp - 1.5) + (mw - 300) / 150)
            else:
                prob = self._sigmoid((logp - 2.5) + (mw - 350) / 175)
            
            return prob
            
        except Exception:
            return 0.3  # Default moderate risk
    
    def _predict_metabolic_stability(self, molecule: Chem.Mol) -> float:
        """Predict metabolic stability"""
        try:
            # Count potential metabolism sites
            aromatic_atoms = sum(1 for atom in molecule.GetAtoms() if atom.GetIsAromatic())
            aliphatic_carbons = sum(1 for atom in molecule.GetAtoms() 
                                  if atom.GetSymbol() == 'C' and not atom.GetIsAromatic())
            
            total_atoms = molecule.GetNumAtoms()
            
            # Fewer labile sites = higher stability
            labile_ratio = (aliphatic_carbons * 0.1 + aromatic_atoms * 0.05) / total_atoms
            stability = 1 - min(1, labile_ratio)
            
            return stability
            
        except Exception:
            return 0.5  # Default moderate stability
    
    def _identify_phase1_sites(self, molecule: Chem.Mol) -> List[int]:
        """Identify potential Phase I metabolism sites"""
        sites = []
        
        try:
            for atom in molecule.GetAtoms():
                idx = atom.GetIdx()
                
                # Aliphatic carbons (hydroxylation)
                if (atom.GetSymbol() == 'C' and 
                    not atom.GetIsAromatic() and 
                    atom.GetTotalNumHs() > 0):
                    sites.append(idx)
                
                # Aromatic carbons (hydroxylation)
                elif (atom.GetSymbol() == 'C' and 
                      atom.GetIsAromatic() and 
                      atom.GetTotalNumHs() > 0):
                    sites.append(idx)
                
                # Nitrogen dealkylation sites
                elif (atom.GetSymbol() == 'N' and 
                      len([n for n in atom.GetNeighbors() if n.GetSymbol() == 'C']) > 1):
                    sites.append(idx)
        
        except Exception:
            pass
        
        return sites
    
    def _assess_phase2_conjugation(self, molecule: Chem.Mol) -> float:
        """Assess Phase II conjugation potential"""
        try:
            conjugation_score = 0.0
            
            # Look for conjugation sites
            for atom in molecule.GetAtoms():
                # Hydroxyl groups (glucuronidation, sulfation)
                if atom.GetSymbol() == 'O' and atom.GetTotalNumHs() > 0:
                    conjugation_score += 0.2
                
                # Amino groups (acetylation, methylation)
                if atom.GetSymbol() == 'N' and atom.GetTotalNumHs() > 0:
                    conjugation_score += 0.15
                
                # Carboxyl groups (amino acid conjugation)
                if atom.GetSymbol() == 'O' and any(
                    n.GetSymbol() == 'C' and n.GetFormalCharge() == 0 
                    for n in atom.GetNeighbors()
                ):
                    conjugation_score += 0.1
            
            return min(1.0, conjugation_score)
            
        except Exception:
            return 0.3  # Default moderate conjugation potential
    
    def _predict_renal_clearance(self, molecule: Chem.Mol) -> float:
        """Predict renal clearance"""
        try:
            mw = Descriptors.MolWt(molecule)
            logp = Descriptors.MolLogP(molecule)
            
            # Smaller, more polar molecules have higher renal clearance
            clearance = 10 * np.exp(-mw / 200) * np.exp(-abs(logp) / 2)
            return clearance
            
        except Exception:
            return 5.0  # Default moderate clearance
    
    def _predict_half_life(self, molecule: Chem.Mol) -> float:
        """Predict elimination half-life"""
        try:
            mw = Descriptors.MolWt(molecule)
            logp = Descriptors.MolLogP(molecule)
            
            # Larger, more lipophilic molecules tend to have longer half-lives
            half_life = 2 + (mw / 100) * (1 + abs(logp) / 3)
            return min(24, half_life)  # Cap at 24 hours
            
        except Exception:
            return 6.0  # Default 6-hour half-life
    
    def _predict_biliary_excretion(self, molecule: Chem.Mol) -> float:
        """Predict biliary excretion potential"""
        try:
            mw = Descriptors.MolWt(molecule)
            
            # Molecular weight threshold for biliary excretion (~400 Da)
            if mw > 400:
                return self._sigmoid((mw - 400) / 100)
            else:
                return 0.1
                
        except Exception:
            return 0.2  # Default low biliary excretion
    
    def _check_pains_alerts(self, molecule: Chem.Mol) -> List[str]:
        """Check for PAINS (Pan Assay Interference Compounds)"""
        alerts = []
        
        try:
            if self.pains_catalog:
                matches = self.pains_catalog.GetMatches(molecule)
                alerts = [match.GetDescription() for match in matches]
        except Exception:
            pass
        
        return alerts
    
    def _check_toxicity_alerts(self, molecule: Chem.Mol) -> List[str]:
        """Check for structural toxicity alerts"""
        alerts = []
        
        try:
            if self.tox_catalog:
                matches = self.tox_catalog.GetMatches(molecule)
                alerts = [match.GetDescription() for match in matches]
        except Exception:
            pass
        
        return alerts
    
    def _predict_mutagenicity(self, molecule: Chem.Mol) -> float:
        """Predict mutagenicity (Ames test)"""
        try:
            # Simplified structural alert-based prediction
            mutagenic_score = 0.0
            
            # Check for known mutagenic substructures
            aromatic_amines = len(molecule.GetSubstructMatches(Chem.MolFromSmarts('[cH0:1][NH2]')))
            nitro_aromatics = len(molecule.GetSubstructMatches(Chem.MolFromSmarts('[c][N+](=O)[O-]')))
            
            mutagenic_score += aromatic_amines * 0.3
            mutagenic_score += nitro_aromatics * 0.4
            
            return min(1.0, mutagenic_score)
            
        except Exception:
            return 0.1  # Default low mutagenicity
    
    def _predict_herg_inhibition(self, molecule: Chem.Mol) -> float:
        """Predict hERG channel inhibition risk"""
        try:
            logp = Descriptors.MolLogP(molecule)
            mw = Descriptors.MolWt(molecule)
            
            # Large, lipophilic molecules have higher hERG risk
            herg_risk = self._sigmoid((logp - 3) + (mw - 300) / 200)
            return herg_risk
            
        except Exception:
            return 0.2  # Default low risk
    
    def _predict_hepatotoxicity(self, molecule: Chem.Mol) -> float:
        """Predict hepatotoxicity risk"""
        try:
            # Check for hepatotoxic functional groups
            hepatotox_score = 0.0
            
            # Halogenated aromatics
            halogen_aromatics = len(molecule.GetSubstructMatches(Chem.MolFromSmarts('[c][F,Cl,Br,I]')))
            hepatotox_score += halogen_aromatics * 0.1
            
            # Nitro groups
            nitro_groups = len(molecule.GetSubstructMatches(Chem.MolFromSmarts('[N+](=O)[O-]')))
            hepatotox_score += nitro_groups * 0.2
            
            return min(1.0, hepatotox_score)
            
        except Exception:
            return 0.1  # Default low hepatotoxicity
    
    # Utility functions
    
    def _sigmoid(self, x: float) -> float:
        """Sigmoid function"""
        return 1 / (1 + np.exp(-x))
    
    def _sigmoid_transform(self, value: float, threshold: float, slope: float) -> float:
        """Transform value using sigmoid around threshold"""
        return self._sigmoid((value - threshold) * slope)
    
    def _optimal_range_score(self, value: float, min_opt: float, max_opt: float) -> float:
        """Score based on optimal range (1.0 in range, decreasing outside)"""
        if min_opt <= value <= max_opt:
            return 1.0
        elif value < min_opt:
            return max(0, 1 - (min_opt - value) / min_opt)
        else:
            return max(0, 1 - (value - max_opt) / max_opt)
    
    def _cns_score_transform(self, value: float, min_val: float, max_val: float, reverse: bool = False) -> float:
        """CNS MPO scoring transformation"""
        if reverse:
            if value <= min_val:
                return 1.0
            elif value >= max_val:
                return 0.0
            else:
                return 1 - (value - min_val) / (max_val - min_val)
        else:
            if value <= min_val:
                return 0.0
            elif value >= max_val:
                return 1.0
            else:
                return (value - min_val) / (max_val - min_val)
    
    def _interpret_admet_scores(self, report: Dict[str, Any]) -> Dict[str, str]:
        """Interpret ADMET scores and provide recommendations"""
        overall_score = report['overall_admet_score']
        
        if overall_score >= 0.8:
            overall_interpretation = "Excellent drug-like properties"
        elif overall_score >= 0.6:
            overall_interpretation = "Good drug-like properties with minor concerns"
        elif overall_score >= 0.4:
            overall_interpretation = "Moderate drug-like properties, optimization recommended"
        else:
            overall_interpretation = "Poor drug-like properties, significant optimization needed"
        
        return {
            'overall': overall_interpretation,
            'absorption': self._interpret_score(report['absorption']['score'], 'absorption'),
            'distribution': self._interpret_score(report['distribution']['score'], 'distribution'),
            'metabolism': self._interpret_score(report['metabolism']['score'], 'metabolism'),
            'excretion': self._interpret_score(report['excretion']['score'], 'excretion'),
            'toxicity': self._interpret_score(report['toxicity']['score'], 'toxicity')
        }
    
    def _interpret_score(self, score: float, category: str) -> str:
        """Interpret individual ADMET category score"""
        if score >= 0.8:
            return f"Excellent {category} properties"
        elif score >= 0.6:
            return f"Good {category} properties"
        elif score >= 0.4:
            return f"Moderate {category} properties"
        else:
            return f"Poor {category} properties"
