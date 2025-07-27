"""
PharmFlow Visualization Utilities
Comprehensive visualization tools for molecular docking results and analysis
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.figure_factory as ff
from typing import Dict, List, Any, Optional, Tuple, Union
import logging
from pathlib import Path

from .constants import (
    COLOR_PALETTES, FIGURE_SIZE, DPI, FONT_SIZE, TITLE_FONT_SIZE, AXIS_FONT_SIZE,
    PHARMACOPHORE_TYPES, BINDING_AFFINITY_RANGE
)

logger = logging.getLogger(__name__)

class DockingVisualizer:
    """
    Comprehensive visualization suite for quantum molecular docking results
    """
    
    def __init__(self, style: str = 'seaborn-v0_8', color_palette: str = 'Set2'):
        """
        Initialize visualization system
        
        Args:
            style: Matplotlib style
            color_palette: Default color palette
        """
        self.logger = logging.getLogger(__name__)
        
        # Set matplotlib style
        try:
            plt.style.use(style)
        except OSError:
            plt.style.use('default')
            self.logger.warning(f"Style '{style}' not found, using default")
        
        # Configure seaborn
        sns.set_palette(color_palette)
        
        # Set global parameters
        plt.rcParams['figure.figsize'] = FIGURE_SIZE
        plt.rcParams['figure.dpi'] = DPI
        plt.rcParams['font.size'] = FONT_SIZE
        plt.rcParams['axes.titlesize'] = TITLE_FONT_SIZE
        plt.rcParams['axes.labelsize'] = AXIS_FONT_SIZE
        plt.rcParams['xtick.labelsize'] = AXIS_FONT_SIZE
        plt.rcParams['ytick.labelsize'] = AXIS_FONT_SIZE
        plt.rcParams['legend.fontsize'] = FONT_SIZE
        
        self.logger.info("DockingVisualizer initialized")
    
    def plot_optimization_convergence(self, 
                                    optimization_history: List[float],
                                    title: str = "QAOA Optimization Convergence",
                                    save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot optimization convergence curve
        
        Args:
            optimization_history: List of energy values during optimization
            title: Plot title
            save_path: Path to save figure
            
        Returns:
            Matplotlib figure object
        """
        try:
            fig, ax = plt.subplots(figsize=FIGURE_SIZE)
            
            iterations = range(1, len(optimization_history) + 1)
            
            # Main convergence line
            ax.plot(iterations, optimization_history, 'b-', linewidth=2, alpha=0.8, label='Energy')
            
            # Add moving average for smoothing
            if len(optimization_history) > 5:
                window_size = min(10, len(optimization_history) // 5)
                moving_avg = pd.Series(optimization_history).rolling(window=window_size).mean()
                ax.plot(iterations, moving_avg, 'r--', linewidth=2, alpha=0.7, label=f'Moving Average ({window_size})')
            
            # Highlight best value
            best_idx = np.argmin(optimization_history)
            best_value = optimization_history[best_idx]
            ax.scatter(best_idx + 1, best_value, color='red', s=100, zorder=5, label=f'Best: {best_value:.4f}')
            
            # Formatting
            ax.set_xlabel('Iteration')
            ax.set_ylabel('Energy (kcal/mol)')
            ax.set_title(title)
            ax.grid(True, alpha=0.3)
            ax.legend()
            
            # Add convergence statistics
            if len(optimization_history) > 1:
                improvement = optimization_history[0] - optimization_history[-1]
                convergence_rate = improvement / len(optimization_history)
                ax.text(0.02, 0.98, f'Total Improvement: {improvement:.4f}\nConvergence Rate: {convergence_rate:.6f}',
                       transform=ax.transAxes, verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=DPI, bbox_inches='tight')
            
            return fig
            
        except Exception as e:
            self.logger.error(f"Optimization convergence plotting failed: {e}")
            return plt.figure()
    
    def save_all_figures(self, figures: Dict[str, plt.Figure], output_dir: str):
        """
        Save all generated figures to specified directory
        
        Args:
            figures: Dictionary of figure names and objects
            output_dir: Output directory path
        """
        try:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            for name, fig in figures.items():
                if fig is not None:
                    save_path = output_path / f"{name}.png"
                    fig.savefig(save_path, dpi=DPI, bbox_inches='tight')
                    self.logger.info(f"Saved figure: {save_path}")
            
            self.logger.info(f"All figures saved to {output_dir}")
            
        except Exception as e:
            self.logger.error(f"Figure saving failed: {e}")
    
    def create_3d_molecular_plot(self, 
                               pharmacophores: List[Dict[str, Any]],
                               title: str = "3D Pharmacophore Visualization") -> go.Figure:
        """
        Create 3D molecular visualization using Plotly
        
        Args:
            pharmacophores: List of pharmacophore dictionaries
            title: Plot title
            
        Returns:
            Plotly 3D figure
        """
        try:
            fig = go.Figure()
            
            # Group pharmacophores by type
            pharm_by_type = {}
            for pharm in pharmacophores:
                ptype = pharm.get('type', 'unknown')
                if ptype not in pharm_by_type:
                    pharm_by_type[ptype] = []
                pharm_by_type[ptype].append(pharm)
            
            # Add traces for each pharmacophore type
            for ptype, pharms in pharm_by_type.items():
                positions = [p.get('position') for p in pharms if p.get('position')]
                if not positions:
                    continue
                
                positions = np.array(positions)
                if positions.shape[1] < 3:
                    continue
                
                # Get pharmacophore properties
                pharm_props = PHARMACOPHORE_TYPES.get(ptype, {})
                color = pharm_props.get('color', '#808080')
                radius = pharm_props.get('radius', 1.0)
                
                fig.add_trace(go.Scatter3d(
                    x=positions[:, 0],
                    y=positions[:, 1], 
                    z=positions[:, 2],
                    mode='markers',
                    marker=dict(
                        size=radius * 10,
                        color=color,
                        opacity=0.8
                    ),
                    name=ptype.replace('_', ' ').title(),
                    text=[f"{ptype}: {p.get('source', 'unknown')}" for p in pharms],
                    hovertemplate='<b>%{text}</b><br>X: %{x:.2f}<br>Y: %{y:.2f}<br>Z: %{z:.2f}<extra></extra>'
                ))
            
            # Update layout
            fig.update_layout(
                title=title,
                scene=dict(
                    xaxis_title='X (Å)',
                    yaxis_title='Y (Å)',
                    zaxis_title='Z (Å)',
                    aspectmode='cube'
                ),
                showlegend=True
            )
            
            return fig
            
        except Exception as e:
            self.logger.error(f"3D molecular plotting failed: {e}")
            return go.Figure()
