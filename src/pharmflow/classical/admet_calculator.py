"""ADMET property calculation"""

import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors
import logging

class ADMETCalculator:
    """Calculate ADMET properties"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def calculate_admet(self, molecule: Chem.Mol) -> float:
        """Calculate overall ADMET score"""
        
        # Calculate molecular descriptors
        mw = Descriptors.MolWt(molecule)
        logp = Descriptors.MolLogP(molecule)
        hbd = Descriptors.NumHBD(molecule)
        hba = Descriptors.NumHBA(molecule)
        tpsa = Descriptors.TPSA(molecule)
        
        # Lipinski's Rule of Five scoring
        lipinski_score = 0
        lipinski_score += 1 if mw <= 500 else 0
        lipinski_score += 1 if logp <= 5 else 0
        lipinski_score += 1 if hbd <= 5 else 0
        lipinski_score += 1 if hba <= 10 else 0
        
        # TPSA scoring
        tpsa_score = 1 if tpsa <= 140 else 0
        
        # Combined ADMET score (0-1 scale)
        admet_score = (lipinski_score + tpsa_score) / 5.0
        
        self.logger.debug(f"ADMET score: {admet_score:.2f}")
        return admet_score
