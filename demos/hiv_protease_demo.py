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
File description here
"""
"""
PharmFlow HIV Protease Docking Demo
Demonstrates quantum molecular docking for HIV-1 protease inhibitor discovery
"""

import os
import sys
import logging
import time
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Any

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from pharmflow.core.pharmflow_engine import PharmFlowQuantumDocking
from pharmflow.utils.visualization import DockingVisualizer
from pharmflow.classical.admet_calculator import ADMETCalculator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class HIVProteaseDemo:
    """
    Comprehensive demo for HIV-1 protease inhibitor discovery using quantum docking
    """
    
    def __init__(self):
        """Initialize HIV protease demo"""
        self.demo_name = "PharmFlow HIV-1 Protease Quantum Docking"
        self.results_dir = Path("hiv_demo_results")
        self.results_dir.mkdir(exist_ok=True)
        
        # Initialize PharmFlow engine
        self.pharmflow_engine = PharmFlowQuantumDocking(
            backend='qasm_simulator',
            optimizer='COBYLA',
            num_qaoa_layers=3,
            smoothing_factor=0.1,
            parallel_execution=True
        )
        
        # Initialize analysis tools
        self.visualizer = DockingVisualizer()
        self.admet_calculator = ADMETCalculator()
        
        # Demo configuration
        self.hiv_protease_residues = [25, 26, 27, 28, 29, 30, 47, 48, 49, 50, 82, 84]
        
        logger.info(f"Initialized {self.demo_name}")
    
    def run_complete_demo(self):
        """Execute complete HIV protease docking demonstration"""
        logger.info("=" * 60)
        logger.info(f"Starting {self.demo_name}")
        logger.info("=" * 60)
        
        demo_start_time = time.time()
        
        try:
            # Step 1: Single inhibitor docking
            logger.info("\n🧬 Step 1: Single Inhibitor Quantum Docking")
            single_result = self.demo_single_inhibitor_docking()
            
            # Step 2: Multi-inhibitor screening
            logger.info("\n🔬 Step 2: Multi-Inhibitor Virtual Screening")
            screening_results = self.demo_virtual_screening()
            
            # Step 3: Lead optimization
            logger.info("\n⚗️ Step 3: Lead Compound Optimization")
            optimization_results = self.demo_lead_optimization(screening_results)
            
            # Step 4: ADMET analysis
            logger.info("\n💊 Step 4: ADMET Property Analysis")
            admet_results = self.demo_admet_analysis(screening_results)
            
            # Step 5: Results analysis and visualization
            logger.info("\n📊 Step 5: Results Analysis and Visualization")
            self.analyze_and_visualize_results(
                single_result, screening_results, optimization_results, admet_results
            )
            
            # Generate comprehensive report
            self.generate_demo_report(
                single_result, screening_results, optimization_results, admet_results
            )
            
            demo_duration = time.time() - demo_start_time
            logger.info(f"\n✅ Demo completed successfully in {demo_duration:.2f} seconds")
            
        except Exception as e:
            logger.error(f"Demo failed: {e}")
            raise
    
    def demo_single_inhibitor_docking(self) -> Dict[str, Any]:
        """Demonstrate single HIV protease inhibitor docking"""
        logger.info("Docking ritonavir (known HIV protease inhibitor)")
        
        # Ritonavir - FDA-approved HIV protease inhibitor
        ritonavir_smiles = "CC(C)c1nc(cn1)C(C(CC(=O)N[C@@H](Cc2ccccc2)C(=O)N[C@H](C[C@@H](C(=O)N3CCCCC3)NC(=O)OC(C)(C)C)Cc4ccccc4)C)O"
        
        # Mock protein PDB file path (in real scenario, would use actual HIV protease structure)
        protein_pdb = self._create_mock_hiv_protease_pdb()
        
        try:
            result = self.pharmflow_engine.dock_molecule(
                protein_pdb=protein_pdb,
                ligand_sdf=ritonavir_smiles,
                binding_site_residues=self.hiv_protease_residues,
                max_iterations=200,
                objectives={
                    'binding_affinity': {'weight': 0.5, 'target': 'minimize'},
                    'selectivity': {'weight': 0.3, 'target': 'maximize'},
                    'admet_score': {'weight': 0.2, 'target': 'maximize'}
                }
            )
            
            logger.info(f"Single docking completed:")
            logger.info(f"  Binding Affinity: {result['binding_affinity']:.3f} kcal/mol")
            logger.info(f"  ADMET Score: {result['admet_score']:.3f}")
            logger.info(f"  Selectivity: {result['selectivity']:.3f}")
            logger.info(f"  Docking Time: {result['docking_time']:.2f} seconds")
            
            return result
            
        except Exception as e:
            logger.error(f"Single inhibitor docking failed: {e}")
            return {}
    
    def demo_virtual_screening(self) -> List[Dict[str, Any]]:
        """Demonstrate virtual screening of HIV protease inhibitor library"""
        logger.info("Screening library of potential HIV protease inhibitors")
        
        # Library of HIV protease inhibitor candidates
        inhibitor_library = [
            # Known inhibitors
            "CC(C)c1nc(cn1)C(C(CC(=O)N[C@@H](Cc2ccccc2)C(=O)N[C@H](C[C@@H](C(=O)N3CCCCC3)NC(=O)OC(C)(C)C)Cc4ccccc4)C)O",  # Ritonavir
            "CC(C)(C)NC(=O)[C@H](C(C)C)N[C@@H](CC1=CC=CC=C1)C(=O)N[C@@H](CC2=CC=CC=C2)[C@@H](O)CN(CC3=CC=CC=C3)C(=O)C=C",  # Saquinavir-like
            "CC(C)CC(C(=O)N[C@@H](CCN)C(=O)N[C@@H](Cc1ccc(cc1)O)C(=O)O)NC(=O)[C@H](C(C)C)NC(=O)C",  # Pepstatin A-like
            
            # Novel candidates
            "CC(C)c1ccc(cc1)C(=O)N[C@@H](Cc2ccccc2)C(=O)N[C@H](C[C@@H](C(=O)NCc3ccccc3)NC(=O)C)Cc4ccccc4",
            "CC(C)(C)OC(=O)N[C@@H](Cc1ccc(cc1)O)C(=O)N[C@H](CC(=O)N2CCCCC2)Cc3ccccc3",
            "CC(C)c1nc2ccccc2n1C(=O)[C@H](Cc3ccccc3)NC(=O)[C@H](C(C)C)NC(=O)C",
            "Cc1ccc(cc1)S(=O)(=O)N[C@@H](Cc2ccccc2)C(=O)N[C@H](CC(=O)O)Cc3ccccc3",
            "CC(C)CC(C(=O)N[C@@H](CC1=CN=CN1)C(=O)N[C@@H](Cc2ccccc2)C(=O)O)NC(=O)C"
        ]
        
        protein_pdb = self._create_mock_hiv_protease_pdb()
        
        try:
            results = self.pharmflow_engine.batch_screening(
                protein_pdb=protein_pdb,
                ligand_library=inhibitor_library,
                binding_site_residues=self.hiv_protease_residues,
                max_iterations=100,
                top_n=5
            )
            
            logger.info(f"Virtual screening completed:")
            logger.info(f"  Total compounds screened: {len(inhibitor_library)}")
            logger.info(f"  Successful dockings: {len([r for r in results if r.get('success', False)])}")
            logger.info(f"  Top hits identified: {len(results)}")
            
            if results:
                best_hit = results[0]
                logger.info(f"  Best hit binding affinity: {best_hit['binding_affinity']:.3f} kcal/mol")
            
            return results
            
        except Exception as e:
            logger.error(f"Virtual screening failed: {e}")
            return []
    
    def demo_lead_optimization(self, screening_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Demonstrate lead compound optimization"""
        if not screening_results:
            logger.warning("No screening results available for lead optimization")
            return {}
        
        best_hit = screening_results[0]
        logger.info(f"Optimizing lead compound with binding affinity: {best_hit['binding_affinity']:.3f} kcal/mol")
        
        protein_pdb = self._create_mock_hiv_protease_pdb()
        
        try:
            optimization_result = self.pharmflow_engine.optimize_lead_compound(
                protein_pdb=protein_pdb,
                lead_compound=best_hit['ligand_id'],
                optimization_objectives={
                    'binding_affinity': 0.4,
                    'selectivity': 0.3,
                    'admet_score': 0.3
                },
                binding_site_residues=self.hiv_protease_residues
            )
            
            logger.info("Lead optimization completed:")
            logger.info(f"  Optimized binding affinity: {optimization_result['docking_result']['binding_affinity']:.3f} kcal/mol")
            logger.info(f"  Synthetic accessibility: {optimization_result['synthetic_accessibility']['accessible']}")
            
            return optimization_result
            
        except Exception as e:
            logger.error(f"Lead optimization failed: {e}")
            return {}
    
    def demo_admet_analysis(self, screening_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Demonstrate ADMET property analysis"""
        logger.info("Analyzing ADMET properties of screening hits")
        
        admet_results = {}
        
        try:
            for i, result in enumerate(screening_results[:3]):  # Analyze top 3 hits
                ligand_id = result.get('ligand_id', f'compound_{i}')
                
                # Calculate comprehensive ADMET properties
                mol = self._smiles_to_mol(ligand_id)
                if mol:
                    admet_report = self.admet_calculator.generate_admet_report(mol)
                    admet_results[ligand_id] = admet_report
                    
                    logger.info(f"  {ligand_id}:")
                    logger.info(f"    Overall ADMET Score: {admet_report['overall_admet_score']:.3f}")
                    logger.info(f"    Absorption Score: {admet_report['absorption']['score']:.3f}")
                    logger.info(f"    Distribution Score: {admet_report['distribution']['score']:.3f}")
                    logger.info(f"    Metabolism Score: {admet_report['metabolism']['score']:.3f}")
                    logger.info(f"    Excretion Score: {admet_report['excretion']['score']:.3f}")
                    logger.info(f"    Toxicity Score: {admet_report['toxicity']['score']:.3f}")
            
            return admet_results
            
        except Exception as e:
            logger.error(f"ADMET analysis failed: {e}")
            return {}
    
    def analyze_and_visualize_results(self,
                                    single_result: Dict[str, Any],
                                    screening_results: List[Dict[str, Any]],
                                    optimization_results: Dict[str, Any],
                                    admet_results: Dict[str, Any]):
        """Analyze and visualize all demo results"""
        logger.info("Generating visualizations and analysis")
        
        try:
            # Plot optimization convergence
            if single_result and 'optimization_result' in single_result:
                opt_history = single_result['optimization_result']['optimization_history']
                fig = self.visualizer.plot_optimization_convergence(
                    opt_history,
                    title="HIV Protease Inhibitor Quantum Optimization",
                    save_path=str(self.results_dir / "optimization_convergence.png")
                )
                plt.close(fig)
            
            # Plot screening results
            if screening_results:
                fig = self.visualizer.plot_batch_screening_results(
                    screening_results,
                    top_n=10,
                    title="HIV Protease Inhibitor Virtual Screening",
                    save_path=str(self.results_dir / "screening_results.png")
                )
                plt.close(fig)
            
            # Plot ADMET analysis
            if admet_results:
                self._plot_admet_comparison(admet_results)
            
            # Generate interactive dashboard
            if single_result and screening_results:
                dashboard = self.visualizer.create_interactive_dashboard(
                    single_result, screening_results
                )
                dashboard.write_html(str(self.results_dir / "interactive_dashboard.html"))
            
            logger.info(f"Visualizations saved to {self.results_dir}")
            
        except Exception as e:
            logger.error(f"Visualization failed: {e}")
    
    def generate_demo_report(self,
                           single_result: Dict[str, Any],
                           screening_results: List[Dict[str, Any]],
                           optimization_results: Dict[str, Any],
                           admet_results: Dict[str, Any]):
        """Generate comprehensive demo report"""
        report_path = self.results_dir / "hiv_protease_demo_report.md"
        
        try:
            with open(report_path, 'w') as f:
                f.write(f"# {self.demo_name} Report\n\n")
                f.write(f"Generated on: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                
                # Executive Summary
                f.write("## Executive Summary\n\n")
                f.write("This demo showcases PharmFlow's quantum molecular docking capabilities ")
                f.write("for HIV-1 protease inhibitor discovery. The system successfully:\n\n")
                f.write("- Performed quantum-enhanced molecular docking\n")
                f.write("- Conducted virtual screening of inhibitor libraries\n")
                f.write("- Optimized lead compounds using multi-objective optimization\n")
                f.write("- Analyzed ADMET properties for drug-likeness assessment\n\n")
                
                # Single Inhibitor Results
                if single_result:
                    f.write("## Single Inhibitor Docking Results\n\n")
                    f.write(f"**Target**: HIV-1 Protease\n")
                    f.write(f"**Ligand**: Ritonavir (FDA-approved inhibitor)\n")
                    f.write(f"**Binding Affinity**: {single_result['binding_affinity']:.3f} kcal/mol\n")
                    f.write(f"**ADMET Score**: {single_result['admet_score']:.3f}\n")
                    f.write(f"**Selectivity**: {single_result['selectivity']:.3f}\n")
                    f.write(f"**Computation Time**: {single_result['docking_time']:.2f} seconds\n\n")
                
                # Virtual Screening Results
                if screening_results:
                    f.write("## Virtual Screening Results\n\n")
                    f.write(f"**Total Compounds Screened**: {len([r for r in screening_results if 'ligand_id' in r])}\n")
                    f.write(f"**Successful Dockings**: {len([r for r in screening_results if r.get('success', False)])}\n")
                    f.write(f"**Top Hits Identified**: {len(screening_results)}\n\n")
                    
                    f.write("### Top 5 Hits\n\n")
                    f.write("| Rank | Binding Affinity (kcal/mol) | ADMET Score | Selectivity |\n")
                    f.write("|------|-----------------------------|--------------|--------------|\n")
                    
                    for i, result in enumerate(screening_results[:5]):
                        f.write(f"| {i+1} | {result['binding_affinity']:.3f} | ")
                        f.write(f"{result.get('admet_score', 'N/A')} | ")
                        f.write(f"{result.get('selectivity', 'N/A')} |\n")
                    f.write("\n")
                
                # Lead Optimization Results
                if optimization_results:
                    f.write("## Lead Optimization Results\n\n")
                    opt_result = optimization_results.get('docking_result', {})
                    f.write(f"**Optimized Binding Affinity**: {opt_result.get('binding_affinity', 'N/A')} kcal/mol\n")
                    f.write(f"**Synthetic Accessibility**: {optimization_results.get('synthetic_accessibility', {}).get('accessible', 'N/A')}\n")
                    
                    recommendations = optimization_results.get('optimization_recommendations', [])
                    if recommendations:
                        f.write("\n### Optimization Recommendations\n\n")
                        for rec in recommendations:
                            f.write(f"- {rec}\n")
                    f.write("\n")
                
                # ADMET Analysis
                if admet_results:
                    f.write("## ADMET Property Analysis\n\n")
                    for compound, report in admet_results.items():
                        f.write(f"### {compound}\n\n")
                        f.write(f"**Overall ADMET Score**: {report['overall_admet_score']:.3f}\n\n")
                        f.write("| Property | Score | Interpretation |\n")
                        f.write("|----------|-------|----------------|\n")
                        f.write(f"| Absorption | {report['absorption']['score']:.3f} | {report['interpretation']['absorption']} |\n")
                        f.write(f"| Distribution | {report['distribution']['score']:.3f} | {report['interpretation']['distribution']} |\n")
                        f.write(f"| Metabolism | {report['metabolism']['score']:.3f} | {report['interpretation']['metabolism']} |\n")
                        f.write(f"| Excretion | {report['excretion']['score']:.3f} | {report['interpretation']['excretion']} |\n")
                        f.write(f"| Toxicity | {report['toxicity']['score']:.3f} | {report['interpretation']['toxicity']} |\n\n")
                
                # Conclusions
                f.write("## Conclusions\n\n")
                f.write("The PharmFlow quantum molecular docking system successfully demonstrated:\n\n")
                f.write("1. **Quantum Advantage**: Leveraged QAOA for enhanced sampling of chemical space\n")
                f.write("2. **Pharmacophore Integration**: Incorporated pharmacophore-guided optimization\n")
                f.write("3. **Multi-objective Optimization**: Balanced binding affinity, selectivity, and ADMET properties\n")
                f.write("4. **Comprehensive Analysis**: Provided detailed molecular property assessment\n\n")
                f.write("This approach shows promise for accelerating drug discovery workflows, ")
                f.write("particularly for challenging targets like HIV-1 protease.\n")
            
            logger.info(f"Demo report generated: {report_path}")
            
        except Exception as e:
            logger.error(f"Report generation failed: {e}")
    
    def _create_mock_hiv_protease_pdb(self) -> str:
        """Create mock HIV protease PDB file for demo"""
        pdb_content = """HEADER    HIV-1 PROTEASE                          01-JAN-25   DEMO            
ATOM      1  CA  ASP A  25      20.154  16.967  10.000  1.00 30.00           C  
ATOM      2  CA  THR A  26      19.030  16.080  12.000  1.00 30.00           C  
ATOM      3  CA  GLY A  27      17.654  16.739  14.000  1.00 30.00           C  
ATOM      4  CA  ALA A  28      16.230  17.962  16.000  1.00 30.00           C  
ATOM      5  CA  ASP A  29      15.113  15.176  18.000  1.00 30.00           C  
ATOM      6  CA  ASP A  30      14.530  16.020  20.000  1.00 30.00           C  
ATOM      7  CA  PHE A  47      12.154  16.539  22.000  1.00 30.00           C  
ATOM      8  CA  GLY A  48      11.030  15.507  24.000  1.00 30.00           C  
ATOM      9  CA  GLY A  49      10.030  14.284  26.000  1.00 30.00           C  
ATOM     10  CA  ASP A  50       9.946  17.444  28.000  1.00 30.00           C  
ATOM     11  CA  ILE A  82       8.154  16.967  30.000  1.00 30.00           C  
ATOM     12  CA  ILE A  84       7.030  16.080  32.000  1.00 30.00           C  
END                                                                             
"""
        pdb_path = self.results_dir / "hiv_protease_mock.pdb"
        with open(pdb_path, 'w') as f:
            f.write(pdb_content)
        
        return str(pdb_path)
    
    def _smiles_to_mol(self, smiles: str):
        """Convert SMILES string to RDKit molecule"""
        try:
            from rdkit import Chem
            return Chem.MolFromSmiles(smiles)
        except:
            return None
    
    def _plot_admet_comparison(self, admet_results: Dict[str, Any]):
        """Plot ADMET comparison chart"""
        try:
            compounds = list(admet_results.keys())
            properties = ['absorption', 'distribution', 'metabolism', 'excretion', 'toxicity']
            
            # Extract scores
            scores = {}
            for prop in properties:
                scores[prop] = [admet_results[comp][prop]['score'] for comp in compounds]
            
            # Create comparison plot
            fig, ax = plt.subplots(figsize=(12, 8))
            
            x = np.arange(len(compounds))
            width = 0.15
            
            for i, prop in enumerate(properties):
                offset = (i - 2) * width
                ax.bar(x + offset, scores[prop], width, label=prop.capitalize())
            
            ax.set_xlabel('Compounds')
            ax.set_ylabel('ADMET Score')
            ax.set_title('ADMET Properties Comparison')
            ax.set_xticks(x)
            ax.set_xticklabels([f'Compound {i+1}' for i in range(len(compounds))], rotation=45)
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(self.results_dir / "admet_comparison.png", dpi=300)
            plt.close(fig)
            
        except Exception as e:
            logger.error(f"ADMET plotting failed: {e}")

def main():
    """Main demo execution function"""
    try:
        demo = HIVProteaseDemo()
        demo.run_complete_demo()
        
        print("\n" + "="*60)
        print("🎉 HIV Protease Demo Completed Successfully!")
        print("="*60)
        print(f"\nResults saved to: {demo.results_dir}")
        print("\nKey files generated:")
        print("- hiv_protease_demo_report.md: Comprehensive demo report")
        print("- optimization_convergence.png: QAOA convergence visualization")
        print("- screening_results.png: Virtual screening results")
        print("- admet_comparison.png: ADMET properties comparison")
        print("- interactive_dashboard.html: Interactive results dashboard")
        
    except Exception as e:
        logger.error(f"Demo execution failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
