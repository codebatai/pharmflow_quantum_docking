"""
Visualization tools for PharmFlow Quantum Molecular Docking
Provides 3D molecular visualization, energy landscape plotting, and analysis charts
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
import logging
from rdkit import Chem
from rdkit.Chem import Draw, AllChem
from rdkit.Chem.Draw import rdMolDraw2D
import py3Dmol

from .constants import PHARMACOPHORE_TYPES, STANDARD_ENERGY_WEIGHTS

logger = logging.getLogger(__name__)

class DockingVisualizer:
    """
    Comprehensive visualization tools for quantum molecular docking results
    """
    
    def __init__(self, style: str = 'default'):
        """
        Initialize visualization engine
        
        Args:
            style: Visual style ('default', 'publication', 'presentation')
        """
        self.style = style
        self.logger = logging.getLogger(__name__)
        
        # Set up matplotlib and seaborn styles
        self._setup_style()
        
        # Color schemes for different visualization types
        self.color_schemes = {
            'energy': ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'],
            'pharmacophore': self._get_pharmacophore_colors(),
            'stages': ['#e74c3c', '#f39c12', '#f1c40f', '#27ae60', '#3498db'],
            'quality': ['#ff4444', '#ffaa44', '#44ff44']
        }
        
        self.logger.info(f"Visualization engine initialized with {style} style")
    
    def visualize_molecular_complex(self,
                                   protein: Any,
                                   ligand: Chem.Mol,
                                   pose: Dict,
                                   pharmacophores: List[Dict],
                                   width: int = 800,
                                   height: int = 600) -> py3Dmol.view:
        """
        Create 3D visualization of protein-ligand complex
        
        Args:
            protein: Protein structure
            ligand: Ligand molecule
            pose: Docking pose
            pharmacophores: Pharmacophore features
            width: Visualization width
            height: Visualization height
            
        Returns:
            3D molecular viewer
        """
        try:
            # Create 3D viewer
            viewer = py3Dmol.view(width=width, height=height)
            
            # Add protein structure (simplified)
            if hasattr(protein, 'pdb_path'):
                with open(protein['pdb_path'], 'r') as f:
                    pdb_data = f.read()
                viewer.addModel(pdb_data, 'pdb')
                viewer.setStyle({'model': 0}, {'cartoon': {'color': 'lightgray'}})
            
            # Add ligand with applied pose
            positioned_ligand = self._apply_pose_to_ligand(ligand, pose)
            ligand_sdf = Chem.MolToMolBlock(positioned_ligand)
            
            viewer.addModel(ligand_sdf, 'sdf')
            viewer.setStyle({'model': 1}, {
                'stick': {'colorscheme': 'default', 'radius': 0.15},
                'sphere': {'colorscheme': 'default', 'radius': 0.25}
            })
            
            # Add pharmacophore features
            self._add_pharmacophores_to_viewer(viewer, pharmacophores)
            
            # Set optimal view
            viewer.zoomTo()
            viewer.spin(True)
            
            return viewer
            
        except Exception as e:
            self.logger.error(f"3D visualization failed: {e}")
            return None
    
    def plot_optimization_convergence(self,
                                    optimization_history: List[float],
                                    title: str = "QAOA Optimization Convergence",
                                    save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot optimization convergence history
        
        Args:
            optimization_history: List of energy values during optimization
            title: Plot title
            save_path: Optional save path
            
        Returns:
            Matplotlib figure
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
        
        iterations = range(len(optimization_history))
        
        # Plot 1: Energy vs iteration
        ax1.plot(iterations, optimization_history, 'b-', linewidth=2, alpha=0.8)
        ax1.fill_between(iterations, optimization_history, alpha=0.3)
        ax1.set_xlabel('Iteration')
        ax1.set_ylabel('Energy (kcal/mol)')
        ax1.set_title(f'{title} - Energy Trace')
        ax1.grid(True, alpha=0.3)
        
        # Add best energy annotation
        best_idx = np.argmin(optimization_history)
        best_energy = optimization_history[best_idx]
        ax1.annotate(f'Best: {best_energy:.3f}', 
                    xy=(best_idx, best_energy),
                    xytext=(best_idx + len(iterations)*0.1, best_energy),
                    arrowprops=dict(arrowstyle='->', color='red'))
        
        # Plot 2: Energy improvement
        if len(optimization_history) > 1:
            improvements = np.diff(optimization_history)
            ax2.plot(iterations[1:], improvements, 'r-', linewidth=2, alpha=0.8)
            ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5)
            ax2.set_xlabel('Iteration')
            ax2.set_ylabel('Energy Change (kcal/mol)')
            ax2.set_title('Energy Improvement per Iteration')
            ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_energy_components(self,
                             energy_components: Dict[str, float],
                             title: str = "Energy Component Analysis",
                             save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot energy component breakdown
        
        Args:
            energy_components: Dictionary of energy components
            title: Plot title
            save_path: Optional save path
            
        Returns:
            Matplotlib figure
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        components = list(energy_components.keys())
        values = list(energy_components.values())
        
        # Plot 1: Bar chart
        colors = plt.cm.Set3(np.linspace(0, 1, len(components)))
        bars = ax1.bar(components, values, color=colors, alpha=0.8)
        ax1.set_ylabel('Energy (kcal/mol)')
        ax1.set_title(f'{title} - Component Breakdown')
        ax1.tick_params(axis='x', rotation=45)
        ax1.grid(True, axis='y', alpha=0.3)
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax1.annotate(f'{value:.2f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=10)
        
        # Plot 2: Pie chart (only for significant components)
        significant_components = {k: abs(v) for k, v in energy_components.items() if abs(v) > 0.1}
        
        if significant_components:
            labels = list(significant_components.keys())
            sizes = list(significant_components.values())
            colors_pie = plt.cm.Set3(np.linspace(0, 1, len(labels)))
            
            wedges, texts, autotexts = ax2.pie(sizes, labels=labels, colors=colors_pie,
                                             autopct='%1.1f%%', startangle=90)
            ax2.set_title('Relative Energy Contributions')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_energy_landscape_2d(self,
                                energy_landscape: np.ndarray,
                                title: str = "Quantum Energy Landscape",
                                save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot 2D energy landscape heatmap
        
        Args:
            energy_landscape: 2D energy landscape matrix
            title: Plot title
            save_path: Optional save path
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Create heatmap
        im = ax.imshow(energy_landscape, cmap='viridis', aspect='auto', 
                      interpolation='bilinear', origin='lower')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Energy (kcal/mol)', rotation=270, labelpad=20)
        
        # Find and mark global minimum
        min_idx = np.unravel_index(np.argmin(energy_landscape), energy_landscape.shape)
        ax.plot(min_idx[1], min_idx[0], 'r*', markersize=15, label='Global Minimum')
        
        # Find and mark local minima
        local_minima = self._find_local_minima_2d(energy_landscape)
        if local_minima:
            y_coords, x_coords = zip(*local_minima)
            ax.plot(x_coords, y_coords, 'wo', markersize=8, label='Local Minima')
        
        ax.set_xlabel('Parameter 1')
        ax.set_ylabel('Parameter 2')
        ax.set_title(title)
        ax.legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_batch_screening_results(self,
                                   batch_results: List[Dict],
                                   top_n: int = 20,
                                   title: str = "Batch Screening Results",
                                   save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot batch screening results
        
        Args:
            batch_results: List of docking results
            top_n: Number of top compounds to show
            title: Plot title
            save_path: Optional save path
            
        Returns:
            Matplotlib figure
        """
        # Extract data
        compound_names = []
        binding_affinities = []
        admet_scores = []
        
        for result in batch_results:
            if 'error' not in result:
                name = result.get('ligand_path', 'Unknown')
                name = name.split('/')[-1].replace('.sdf', '')  # Extract filename
                compound_names.append(name)
                binding_affinities.append(result.get('binding_affinity', 0))
                admet_scores.append(result.get('admet_score', 0))
        
        # Sort by binding affinity
        sorted_data = sorted(zip(compound_names, binding_affinities, admet_scores),
                           key=lambda x: x[1])  # Lower energy is better
        
        # Take top N
        top_data = sorted_data[:top_n]
        top_names, top_affinities, top_admet = zip(*top_data) if top_data else ([], [], [])
        
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 14))
        
        # Plot 1: Binding affinities
        y_pos = np.arange(len(top_names))
        colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(top_names)))
        
        bars1 = ax1.barh(y_pos, top_affinities, color=colors, alpha=0.8)
        ax1.set_yticks(y_pos)
        ax1.set_yticklabels(top_names, fontsize=10)
        ax1.set_xlabel('Binding Affinity (kcal/mol)')
        ax1.set_title(f'{title} - Top {len(top_names)} Compounds by Binding Affinity')
        ax1.grid(True, axis='x', alpha=0.3)
        
        # Add value labels
        for i, (bar, value) in enumerate(zip(bars1, top_affinities)):
            ax1.text(value + max(top_affinities) * 0.01, bar.get_y() + bar.get_height()/2,
                    f'{value:.2f}', ha='left', va='center', fontsize=9)
        
        # Plot 2: ADMET scores
        bars2 = ax2.barh(y_pos, top_admet, color='lightblue', alpha=0.8)
        ax2.set_yticks(y_pos)
        ax2.set_yticklabels(top_names, fontsize=10)
        ax2.set_xlabel('ADMET Score')
        ax2.set_title('ADMET Scores for Top Compounds')
        ax2.grid(True, axis='x', alpha=0.3)
        
        # Plot 3: Scatter plot of affinity vs ADMET
        scatter = ax3.scatter(top_affinities, top_admet, c=range(len(top_names)),
                            cmap='viridis', s=100, alpha=0.7)
        ax3.set_xlabel('Binding Affinity (kcal/mol)')
        ax3.set_ylabel('ADMET Score')
        ax3.set_title('Binding Affinity vs ADMET Score')
        ax3.grid(True, alpha=0.3)
        
        # Add compound labels to scatter plot
        for i, name in enumerate(top_names):
            ax3.annotate(name, (top_affinities[i], top_admet[i]),
                        xytext=(5, 5), textcoords='offset points',
                        fontsize=8, alpha=0.8)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def create_interactive_dashboard(self,
                                   optimization_results: Dict,
                                   batch_results: List[Dict]) -> go.Figure:
        """
        Create interactive dashboard with multiple plots
        
        Args:
            optimization_results: Optimization results dictionary
            batch_results: Batch screening results
            
        Returns:
            Plotly figure with interactive dashboard
        """
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Optimization Convergence', 'Energy Components',
                          'Batch Screening Results', 'Performance Metrics'),
            specs=[[{"secondary_y": False}, {"type": "domain"}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Plot 1: Optimization convergence
        history = optimization_results.get('optimization_history', [])
        if history:
            fig.add_trace(
                go.Scatter(x=list(range(len(history))), y=history,
                          mode='lines+markers', name='Energy',
                          line=dict(color='blue', width=2)),
                row=1, col=1
            )
        
        # Plot 2: Energy components pie chart
        energy_components = optimization_results.get('energy_components', {})
        if energy_components:
            labels = list(energy_components.keys())
            values = [abs(v) for v in energy_components.values()]
            
            fig.add_trace(
                go.Pie(labels=labels, values=values, name="Energy Components"),
                row=1, col=2
            )
        
        # Plot 3: Batch screening results
        if batch_results:
            valid_results = [r for r in batch_results if 'error' not in r]
            if valid_results:
                affinities = [r.get('binding_affinity', 0) for r in valid_results]
                compound_names = [r.get('ligand_path', 'Unknown').split('/')[-1] 
                                for r in valid_results]
                
                fig.add_trace(
                    go.Scatter(x=list(range(len(affinities))), y=affinities,
                              mode='markers', name='Compounds',
                              text=compound_names, textposition="top center",
                              marker=dict(size=8, color=affinities, colorscale='viridis')),
                    row=2, col=1
                )
        
        # Plot 4: Performance metrics
        metrics = optimization_results.get('convergence_metrics', {})
        if metrics:
            metric_names = list(metrics.keys())
            metric_values = list(metrics.values())
            
            fig.add_trace(
                go.Bar(x=metric_names, y=metric_values, name='Metrics',
                      marker_color='lightgreen'),
                row=2, col=2
            )
        
        # Update layout
        fig.update_layout(
            title_text="PharmFlow Quantum Docking Dashboard",
            showlegend=True,
            height=800
        )
        
        return fig
    
    def plot_pharmacophore_analysis(self,
                                  pharmacophores: List[Dict],
                                  title: str = "Pharmacophore Analysis",
                                  save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot pharmacophore feature analysis
        
        Args:
            pharmacophores: List of pharmacophore features
            title: Plot title
            save_path: Optional save path
            
        Returns:
            Matplotlib figure
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Count pharmacophore types
        type_counts = {}
        source_counts = {'ligand': 0, 'protein': 0}
        
        for pharm in pharmacophores:
            pharm_type = pharm.get('type', 'unknown')
            source = pharm.get('source', 'unknown')
            
            type_counts[pharm_type] = type_counts.get(pharm_type, 0) + 1
            if source in source_counts:
                source_counts[source] += 1
        
        # Plot 1: Pharmacophore type distribution
        types = list(type_counts.keys())
        counts = list(type_counts.values())
        colors = [PHARMACOPHORE_TYPES.get(t, {}).get('color', 'gray') for t in types]
        
        ax1.pie(counts, labels=types, colors=colors, autopct='%1.1f%%', startangle=90)
        ax1.set_title('Pharmacophore Type Distribution')
        
        # Plot 2: Source distribution
        sources = list(source_counts.keys())
        source_values = list(source_counts.values())
        
        ax2.bar(sources, source_values, color=['lightblue', 'lightcoral'], alpha=0.8)
        ax2.set_ylabel('Count')
        ax2.set_title('Pharmacophore Source Distribution')
        ax2.grid(True, axis='y', alpha=0.3)
        
        plt.suptitle(title, fontsize=16)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def save_molecular_image(self,
                           molecule: Chem.Mol,
                           filename: str,
                           size: Tuple[int, int] = (300, 300),
                           highlight_atoms: Optional[List[int]] = None) -> None:
        """
        Save 2D molecular structure image
        
        Args:
            molecule: RDKit molecule
            filename: Output filename
            size: Image size (width, height)
            highlight_atoms: List of atom indices to highlight
        """
        try:
            # Generate 2D coordinates if needed
            if molecule.GetNumConformers() == 0:
                AllChem.Compute2DCoords(molecule)
            
            # Create drawing options
            drawer = rdMolDraw2D.MolDraw2DCairo(size[0], size[1])
            
            if highlight_atoms:
                drawer.DrawMolecule(molecule, highlightAtoms=highlight_atoms)
            else:
                drawer.DrawMolecule(molecule)
            
            drawer.FinishDrawing()
            
            # Save image
            with open(filename, 'wb') as f:
                f.write(drawer.GetDrawingText())
            
            self.logger.info(f"Molecular image saved to {filename}")
            
        except Exception as e:
            self.logger.error(f"Failed to save molecular image: {e}")
    
    # Helper methods
    
    def _setup_style(self):
        """Setup matplotlib and seaborn styles"""
        if self.style == 'publication':
            plt.style.use('seaborn-v0_8-paper')
            sns.set_palette("husl")
        elif self.style == 'presentation':
            plt.style.use('seaborn-v0_8-talk')
            sns.set_palette("bright")
        else:
            plt.style.use('seaborn-v0_8')
            sns.set_palette("deep")
        
        # Common settings
        plt.rcParams['figure.figsize'] = (10, 6)
        plt.rcParams['font.size'] = 12
        plt.rcParams['axes.linewidth'] = 1.5
        plt.rcParams['grid.alpha'] = 0.3
    
    def _get_pharmacophore_colors(self) -> Dict[str, str]:
        """Get pharmacophore type colors"""
        return {ptype: props['color'] for ptype, props in PHARMACOPHORE_TYPES.items()}
    
    def _apply_pose_to_ligand(self, ligand: Chem.Mol, pose: Dict) -> Chem.Mol:
        """Apply pose transformation to ligand"""
        # Simplified pose application
        # In real implementation, would apply full 3D transformation
        mol = Chem.Mol(ligand)
        if mol.GetNumConformers() == 0:
            AllChem.EmbedMolecule(mol, randomSeed=42)
        return mol
    
    def _add_pharmacophores_to_viewer(self, viewer: py3Dmol.view, pharmacophores: List[Dict]):
        """Add pharmacophore features to 3D viewer"""
        for pharm in pharmacophores:
            if 'position' in pharm:
                pos = pharm['position']
                pharm_type = pharm.get('type', 'unknown')
                color = PHARMACOPHORE_TYPES.get(pharm_type, {}).get('color', 'gray')
                radius = PHARMACOPHORE_TYPES.get(pharm_type, {}).get('radius', 1.0)
                
                viewer.addSphere({
                    'center': {'x': pos[0], 'y': pos[1], 'z': pos[2]},
                    'radius': radius,
                    'color': color,
                    'alpha': 0.5
                })
    
    def _find_local_minima_2d(self, landscape: np.ndarray) -> List[Tuple[int, int]]:
        """Find local minima in 2D energy landscape"""
        from scipy.ndimage import minimum_filter
        
        # Use minimum filter to find local minima
        local_minima_mask = (landscape == minimum_filter(landscape, size=3))
        
        # Get coordinates
        minima_coords = np.where(local_minima_mask)
        return list(zip(minima_coords[0], minima_coords[1]))
    
    def export_results_to_html(self,
                             results: Dict,
                             output_file: str = "pharmflow_results.html"):
        """
        Export comprehensive results to HTML report
        
        Args:
            results: Combined results dictionary
            output_file: Output HTML filename
        """
        try:
            # Create HTML content
            html_content = self._generate_html_report(results)
            
            with open(output_file, 'w') as f:
                f.write(html_content)
            
            self.logger.info(f"Results exported to {output_file}")
            
        except Exception as e:
            self.logger.error(f"HTML export failed: {e}")
    
    def _generate_html_report(self, results: Dict) -> str:
        """Generate HTML report content"""
        html_template = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>PharmFlow Quantum Docking Results</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; }
                .header { background-color: #f0f0f0; padding: 20px; border-radius: 5px; }
                .section { margin: 20px 0; padding: 15px; border: 1px solid #ddd; border-radius: 5px; }
                .metric { display: inline-block; margin: 10px; padding: 10px; background-color: #e8f4fd; border-radius: 3px; }
                table { border-collapse: collapse; width: 100%; }
                th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
                th { background-color: #f2f2f2; }
            </style>
        </head>
        <body>
            <div class="header">
                <h1>PharmFlow Quantum Molecular Docking Results</h1>
                <p>Generated on: {timestamp}</p>
            </div>
            
            <div class="section">
                <h2>Summary</h2>
                <div class="metric">Best Energy: {best_energy:.3f} kcal/mol</div>
                <div class="metric">Total Time: {total_time:.2f} seconds</div>
                <div class="metric">Success Rate: {success_rate:.1%}</div>
            </div>
            
            <div class="section">
                <h2>Optimization Details</h2>
                <p>Quantum optimization completed with {iterations} iterations.</p>
                <p>Convergence achieved: {converged}</p>
            </div>
        </body>
        </html>
        """
        
        import datetime
        
        # Extract values with defaults
        best_energy = results.get('best_energy', 0.0)
        total_time = results.get('total_time', 0.0)
        success_rate = results.get('success_rate', 0.0)
        iterations = len(results.get('optimization_history', []))
        converged = results.get('convergence_metrics', {}).get('converged', False)
        timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        return html_template.format(
            timestamp=timestamp,
            best_energy=best_energy,
            total_time=total_time,
            success_rate=success_rate,
            iterations=iterations,
            converged=converged
        )
