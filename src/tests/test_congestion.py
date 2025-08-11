"""
Complete Congestion Tracking and Traffic Flow Visualization Example

This example demonstrates:
1. Spatial density estimation σ(x)
2. Traffic flow computation w_Q with vector directions
3. Traffic intensity i_Q visualization
4. Step-by-step congestion tracking
5. Real-time traffic flow vector field visualization
6. Sobolev regularization integration
7. Complete theoretical framework implementation
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse
import logging

from weight_perturbation import (
    _THEORETICAL_COMPONENTS_AVAILABLE,
    check_theoretical_support,
    compute_spatial_density,
    compute_traffic_flow,
    set_seed,
    compute_device,
    sample_real_data,
    pretrain_wgan_gp,
    sample_evidence_domains,
    Generator,
    SobolevConstrainedCritic,
    CTWeightPerturberTargetNotGiven,
)

# Set up plotting style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Set up logging
logging.basicConfig(level=logging.INFO)

class CongestedTransportVisualizer:
    """
    Comprehensive visualizer for congested transport theory.

    Args:
        figsize (tuple): Figure size for plots.
        save_dir (str): Directory to save visualizations.
    """

    def __init__(self, num_domains=3, figsize=(18, 12), save_dir="test_results/plots/congestion_analysis"):
        self.num_domains = num_domains
        self.figsize = figsize
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True)
        self.epoch_data = []
        self.domain_colors = sns.color_palette("husl", n_colors=num_domains)

    def visualize_congested_transport_step(self, epoch, generator, critics, evidence_list,
                                           virtual_samples, noise_samples, multi_congestion_info=None, save=True):
        """
        Visualize multi-marginal traffic flow for a single epoch.

        Args:
            epoch (int): Current epoch number.
            generator (torch.nn.Module): Generator model.
            critics (list): List of critic models for each domain.
            evidence_list (list): List of evidence samples for each domain.
            virtual_samples (torch.Tensor): Virtual target samples.
            noise_samples (torch.Tensor): Noise samples for generation.
            multi_congestion_info (dict, optional): Multi-domain congestion information.
            save (bool): Whether to save the plot.

        Returns:
            dict: Epoch data including samples and flow information.
        """

        device = next(generator.parameters()).device

        # Generate samples from current generator
        with torch.no_grad():
            gen_samples = generator(noise_samples)

        # Compute flow information for each domain
        domain_flows = []
        domain_densities = []
        for i, (evidence, critic) in enumerate(zip(evidence_list, critics)):
            if critic is None:
                continue

            # Compute spatial density for this domain
            all_samples = torch.cat([gen_samples, evidence], dim=0)
            density_info = compute_spatial_density(all_samples, bandwidth=0.15)
            sigma_gen = density_info['density_at_samples'][:gen_samples.shape[0]]

            # Compute traffic flow for this domain
            flow_info = compute_traffic_flow(
                critic, generator, noise_samples, sigma_gen, lambda_param=100
            )

            domain_flows.append({
                'domain_id': i,
                'flow': flow_info['traffic_flow'].detach().cpu().numpy(),
                'intensity': flow_info['traffic_intensity'].detach().cpu().numpy(),
                'density': sigma_gen.detach().cpu().numpy(),
                'gradient_norm': flow_info['gradient_norm'].detach().cpu().numpy(),
                'evidence': evidence.detach().cpu().numpy()
            })
            domain_densities.append(sigma_gen.detach().cpu().numpy())

        # Store epoch data
        epoch_data = {
            'epoch': epoch,
            'gen_samples': gen_samples.detach().cpu().numpy(),
            'virtual_samples': virtual_samples.detach().cpu().numpy(),
            'evidence_list': [ev.detach().cpu().numpy() for ev in evidence_list],
            'domain_flows': domain_flows,
            'multi_congestion_info': multi_congestion_info
        }
        self.epoch_data.append(epoch_data)

        # Create comprehensive visualization
        fig = plt.figure(figsize=self.figsize)
        gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)
        fig.suptitle(f'Multi-Marginal Traffic Flow Analysis - Epoch {epoch}', fontsize=18, fontweight='bold')

        # Plot 1: Overview with all domains and virtual target
        ax_overview = fig.add_subplot(gs[0, :2])

        # Plot generated samples
        ax_overview.scatter(gen_samples[:, 0].cpu(), gen_samples[:, 1].cpu(),
                    c='blue', alpha=0.6, s=30, label='Generated', zorder=3)

        # Plot evidence domains with different colors
        for i, evidence in enumerate(evidence_list):
            ax_overview.scatter(evidence[:, 0].cpu(), evidence[:, 1].cpu(),
                        c=[self.domain_colors[i]], alpha=0.7, s=40,
                        label=f'Evidence {i+1}', marker='s', zorder=2)

        # Plot virtual target
        ax_overview.scatter(virtual_samples[:, 0].cpu(), virtual_samples[:, 1].cpu(),
                    c='red', alpha=0.5, s=25, label='Virtual Target', marker='^', zorder=1)

        ax_overview.set_title('Multi-Domain Overview')
        ax_overview.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax_overview.grid(True, alpha=0.3)
        ax_overview.set_xlabel('X₁')
        ax_overview.set_ylabel('X₂')

        # Plot 2: Combined traffic flow vector field
        ax_combined_flow = fig.add_subplot(gs[0, 2:])

        if domain_flows:
            # Combine flows from all domains
            all_flows = np.stack([df['flow'] for df in domain_flows])
            combined_flow = np.mean(all_flows, axis=0)  # Average across domains
            all_intensities = np.stack([df['intensity'] for df in domain_flows])
            combined_intensity = np.mean(all_intensities, axis=0)

            # Subsample for cleaner visualization
            gen_samples_np = gen_samples.detach().cpu().numpy()
            subsample_idx = np.random.choice(len(gen_samples_np), min(80, len(gen_samples_np)), replace=False)

            quiv = self._unit_quiver(
                ax_combined_flow,
                gen_samples_np[subsample_idx, 0], gen_samples_np[subsample_idx, 1],
                combined_flow[subsample_idx, 0], combined_flow[subsample_idx, 1],
                color_by=combined_intensity[subsample_idx],
                arrow_frac=0.1,
                width=0.004, cmap='viridis', alpha=0.85
            )
            cbar = plt.colorbar(quiv, ax=ax_combined_flow, shrink=0.7)
            cbar.set_label('Combined Traffic Intensity')

        ax_combined_flow.set_title('Combined Multi-Marginal Flow Field')
        ax_combined_flow.grid(True, alpha=0.3)
        ax_combined_flow.set_xlabel('X₁')
        ax_combined_flow.set_ylabel('X₂')

        # Plots 3-5: Domain-specific traffic flows
        for i, domain_flow in enumerate(domain_flows[:3]):  # Show up to 3 domains
            ax_domain_flow = fig.add_subplot(gs[1, i])
            flow = domain_flow['flow']
            intensity = domain_flow['intensity']
            evidence = domain_flow['evidence']

            # Plot evidence domain
            ax_domain_flow.scatter(evidence[:, 0], evidence[:, 1],
                       c=[self.domain_colors[i]], alpha=0.7, s=50,
                       label=f'Evidence {i+1}', marker='s')

            # Plot flow vectors for this domain
            gen_samples_np = gen_samples.detach().cpu().numpy()
            subsample_idx = np.random.choice(len(gen_samples_np), min(40, len(gen_samples_np)), replace=False)

            quiv = self._unit_quiver(
                ax_domain_flow,
                gen_samples_np[subsample_idx, 0], gen_samples_np[subsample_idx, 1],
                flow[subsample_idx, 0], flow[subsample_idx, 1],
                color_by=intensity[subsample_idx],
                arrow_frac=0.1, width=0.005, cmap='plasma', alpha=0.85
            )
            plt.colorbar(quiv, ax=ax_domain_flow, shrink=0.75)

            ax_domain_flow.set_title(f'Domain {i+1} Flow Field')
            ax_domain_flow.grid(True, alpha=0.3)
            ax_domain_flow.set_xlabel('X₁')
            ax_domain_flow.set_ylabel('X₂')

            # Add domain statistics
            stats_text = f"""
Domain {i+1} Stats:
Mean Intensity: {intensity.mean():.4f}
Max Intensity: {intensity.max():.4f}
Flow Magnitude: {np.linalg.norm(flow, axis=1).mean():.4f}
"""
            ax_domain_flow.text(0.02, 0.98, stats_text, transform=ax_domain_flow.transAxes,
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7),
                    fontsize=8)

        # Plot 6: Multi-marginal intensity comparison
        if len(domain_flows) > 1:
            ax_intensity_comparison = fig.add_subplot(gs[1, 3])
            domain_labels = [f'Domain {df["domain_id"]+1}' for df in domain_flows]
            mean_intensities = [df['intensity'].mean() for df in domain_flows]
            max_intensities = [df['intensity'].max() for df in domain_flows]

            x_pos = np.arange(len(domain_labels))
            width = 0.35

            bars1 = ax_intensity_comparison.bar(x_pos - width/2, mean_intensities, width,
                            label='Mean Intensity', alpha=0.7, color='skyblue')
            bars2 = ax_intensity_comparison.bar(x_pos + width/2, max_intensities, width,
                            label='Max Intensity', alpha=0.7, color='lightcoral')

            ax_intensity_comparison.set_title('Domain Intensity Comparison')
            ax_intensity_comparison.set_xlabel('Evidence Domains')
            ax_intensity_comparison.set_ylabel('Traffic Intensity')
            ax_intensity_comparison.set_xticks(x_pos)
            ax_intensity_comparison.set_xticklabels(domain_labels)
            ax_intensity_comparison.legend()
            ax_intensity_comparison.grid(True, alpha=0.3)

        # Plot 7: Spatial density heatmap
        ax_density_heatmap = fig.add_subplot(gs[2, :2])
        if domain_flows:
            # Combine densities from all domains
            all_densities = np.stack([df['density'] for df in domain_flows])
            combined_density = np.mean(all_densities, axis=0)

            gen_samples_np = gen_samples.detach().cpu().numpy()
            scatter = ax_density_heatmap.scatter(gen_samples_np[:, 0], gen_samples_np[:, 1],
                                  c=combined_density, cmap='coolwarm', s=50, alpha=0.7)

            cbar7 = plt.colorbar(scatter, ax=ax_density_heatmap, shrink=0.7)
            cbar7.set_label('Combined Spatial Density σ(x)')

        ax_density_heatmap.set_title('Multi-Domain Spatial Density')
        ax_density_heatmap.grid(True, alpha=0.3)
        ax_density_heatmap.set_xlabel('X₁')
        ax_density_heatmap.set_ylabel('X₂')
        x_min, x_max = np.nanmin(gen_samples_np[:,0]), np.nanmax(gen_samples_np[:,0])
        y_min, y_max = np.nanmin(gen_samples_np[:,1]), np.nanmax(gen_samples_np[:,1])
        xr = max(x_max - x_min, 1e-6); yr = max(y_max - y_min, 1e-6)
        ax_density_heatmap.set_xlim(x_min - 0.05*xr, x_max + 0.05*xr)
        ax_density_heatmap.set_ylim(y_min - 0.05*yr, y_max + 0.05*yr)

        # Plot 8: Congestion cost evolution
        ax_congestion_cost = fig.add_subplot(gs[2, 2:])
        if multi_congestion_info and 'domains' in multi_congestion_info:
            domain_ids = []
            domain_costs = []
            for domain_info in multi_congestion_info['domains']:
                domain_ids.append(f"Domain {domain_info['domain_id']+1}")
                domain_costs.append(domain_info['congestion_cost'].item())

            bars = ax_congestion_cost.bar(domain_ids, domain_costs, alpha=0.7,
                           color=[self.domain_colors[i] for i in range(len(domain_costs))])

            ax_congestion_cost.set_title('Domain-Specific Congestion Costs')
            ax_congestion_cost.set_xlabel('Evidence Domains')
            ax_congestion_cost.set_ylabel('Congestion Cost H(x,i)')
            ax_congestion_cost.grid(True, alpha=0.3)

            # Add cost values on bars
            for bar, cost in zip(bars, domain_costs):
                height = bar.get_height()
                ax_congestion_cost.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                         f'{cost:.4f}', ha='center', va='bottom', fontsize=9)

        plt.tight_layout()

        if save:
            save_path = self.save_dir / f"multimarginal_flow_epoch_{epoch:03d}.png"
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved multi-marginal flow visualization: {save_path}")

        # if epoch % 5 == 0:
        #     plt.show()
        # else:
        #     plt.close()  # Only show every 5th epoch
        plt.close()

        return epoch_data

    def _unit_quiver(self, ax, X, Y, U, V, color_by=None, arrow_frac=0.1, width=0.004, cmap='viridis', alpha=0.85):
        """
        Draws a quiver plot that shows only directions using arrows of equal length.
        - arrow_frac: Arrow length as a fraction of the axis range (e.g., 0.1 means 10% of the x-range)
        - color_by: Values to encode color (e.g., traffic_intensity); if None, use a single color
        """
        X = np.asarray(X); Y = np.asarray(Y); U = np.asarray(U); V = np.asarray(V)
        # Direction unit vectors
        mag = np.sqrt(U**2 + V**2) + 1e-12
        Uu = U / mag
        Vu = V / mag

        # Axis range extents
        x_min = np.nanmin(X); x_max = np.nanmax(X)
        y_min = np.nanmin(Y); y_max = np.nanmax(Y)
        x_range = max(x_max - x_min, 1e-6)
        y_range = max(y_max - y_min, 1e-6)

        # Determine on-screen uniform arrow length considering both x and y ranges
        # Quiver interprets U,V in data coordinates; compensate for x/y scale differences
        # Reference length: includes compensation for diagonal components
        target_len_x = arrow_frac * x_range
        target_len_y = arrow_frac * y_range
        U_plot = Uu * target_len_x
        V_plot = Vu * target_len_y

        quiv = ax.quiver(
            X, Y, U_plot, V_plot,
            color_by if color_by is not None else None,
            cmap=cmap, angles='xy', scale_units='xy', scale=1.0,
            width=width, alpha=alpha
        )
        # Set axis limits with a small margin for better visuals
        ax.set_xlim(x_min - 0.05*x_range, x_max + 0.05*x_range)
        ax.set_ylim(y_min - 0.05*y_range, y_max + 0.05*y_range)
        return quiv


    def create_multimarginal_summary(self, save=True):
        """
        Create a comprehensive summary of multi-marginal traffic flow evolution.

        Args:
            save (bool): Whether to save the summary plot.
        """

        if not self.epoch_data:
            print("No epoch data available for summary.")
            return

        fig, axes = plt.subplots(3, 3, figsize=(18, 15))
        fig.suptitle('Multi-Marginal Traffic Flow Evolution Summary', fontsize=16, fontweight='bold')

        epochs = [data['epoch'] for data in self.epoch_data]

        # Extract domain-specific metrics over time
        domain_metrics = {i: {'intensities': [], 'densities': [], 'flow_mags': []}
                          for i in range(self.num_domains)}
        combined_intensities = []
        combined_densities = []
        combined_flow_mags = []

        for data in self.epoch_data:
            domain_flows = data['domain_flows']
            if domain_flows:
                # Combined metrics
                all_intensities = [df['intensity'] for df in domain_flows]
                all_densities = [df['density'] for df in domain_flows]
                all_flows = [df['flow'] for df in domain_flows]
    
                combined_intensities.append(np.nanmean([np.mean(intensity) for intensity in all_intensities]) if all_intensities else np.nan)
                combined_densities.append(np.nanmean([np.mean(density) for density in all_densities]) if all_densities else np.nan)
                combined_flow_mags.append(np.nanmean([np.linalg.norm(flow, axis=1).mean() for flow in all_flows]) if all_flows else np.nan)

                # Domain-specific metrics
                for df in domain_flows:
                    domain_id = df['domain_id']
                    domain_metrics[domain_id]['intensities'].append(df['intensity'].mean() if df['intensity'].size else np.nan)
                    domain_metrics[domain_id]['densities'].append(df['density'].mean() if df['density'].size else np.nan)
                    domain_metrics[domain_id]['flow_mags'].append(np.linalg.norm(df['flow'], axis=1).mean() if df['flow'].size else np.nan)

        # Plot 1: Combined intensity evolution
        ax_combined_intensity = axes[0, 0]
        ax_combined_intensity.plot(epochs, combined_intensities, 'b-', linewidth=2, label='Combined Intensity')
        ax_combined_intensity.set_title('Combined Traffic Intensity Evolution')
        ax_combined_intensity.set_xlabel('Epoch')
        ax_combined_intensity.set_ylabel('Mean Traffic Intensity')
        ax_combined_intensity.legend()
        ax_combined_intensity.grid(True, alpha=0.3)

        # Plot 2: Domain-specific intensity evolution
        ax_domain_intensity = axes[0, 1]
        for domain_id in range(self.num_domains):
            if domain_metrics[domain_id]['intensities']:
                ax_domain_intensity.plot(epochs[:len(domain_metrics[domain_id]['intensities'])],
                         domain_metrics[domain_id]['intensities'],
                         linewidth=2, label=f'Domain {domain_id+1}',
                         color=self.domain_colors[domain_id])
        ax_domain_intensity.set_title('Domain-Specific Intensity Evolution')
        ax_domain_intensity.set_xlabel('Epoch')
        ax_domain_intensity.set_ylabel('Mean Traffic Intensity')
        ax_domain_intensity.legend()
        ax_domain_intensity.grid(True, alpha=0.3)

        # Plot 3: Combined density evolution
        ax_combined_density = axes[0, 2]
        ax_combined_density.plot(epochs, combined_densities, 'g-', linewidth=2, label='Combined Density')
        ax_combined_density.set_title('Combined Spatial Density Evolution')
        ax_combined_density.set_xlabel('Epoch')
        ax_combined_density.set_ylabel('Mean Spatial Density')
        ax_combined_density.legend()
        ax_combined_density.grid(True, alpha=0.3)

        # Plot 4: Flow magnitude evolution
        ax_flow_magnitude = axes[1, 0]
        ax_flow_magnitude.plot(epochs, combined_flow_mags, 'm-', linewidth=2, label='Combined Flow Magnitude')
        ax_flow_magnitude.set_title('Combined Flow Magnitude Evolution')
        ax_flow_magnitude.set_xlabel('Epoch')
        ax_flow_magnitude.set_ylabel('Mean Flow Magnitude')
        ax_flow_magnitude.legend()
        ax_flow_magnitude.grid(True, alpha=0.3)

        # Plot 5: Domain comparison at final epoch
        ax_final_domain = axes[1, 1]
        if self.epoch_data:
            final_data = self.epoch_data[-1]
            domain_flows = final_data['domain_flows']
            if domain_flows:
                domain_names = [f'D{df["domain_id"]+1}' for df in domain_flows]
                final_intensities = [df['intensity'].mean() for df in domain_flows]
                bars = ax_final_domain.bar(domain_names, final_intensities,
                               color=[self.domain_colors[df['domain_id']] for df in domain_flows],
                               alpha=0.7)
                ax_final_domain.set_title('Final Domain Intensities')
                ax_final_domain.set_xlabel('Domains')
                ax_final_domain.set_ylabel('Mean Intensity')
                ax_final_domain.grid(True, alpha=0.3)

        # Plot 6: Flow direction coherence
        ax_flow_coherence = axes[1, 2]
        # Compute flow coherence across domains
        flow_coherences = []
        for data in self.epoch_data:
            domain_flows = data['domain_flows']
            if len(domain_flows) > 1:
                # Compute pairwise flow correlations
                flows = [df['flow'] for df in domain_flows]
                correlations = []
                for i in range(len(flows)):
                    for j in range(i+1, len(flows)):
                        # Compute correlation between flow directions
                        flow_i_norm = flows[i] / (np.linalg.norm(flows[i], axis=1, keepdims=True) + 1e-8)
                        flow_j_norm = flows[j] / (np.linalg.norm(flows[j], axis=1, keepdims=True) + 1e-8)
                        corr = np.mean(np.sum(flow_i_norm * flow_j_norm, axis=1))
                        correlations.append(corr)
                flow_coherences.append(np.mean(correlations))
            else:
                flow_coherences.append(1.0)

        ax_flow_coherence.plot(epochs, flow_coherences, 'orange', linewidth=2, marker='o')
        ax_flow_coherence.set_title('Inter-Domain Flow Coherence')
        ax_flow_coherence.set_xlabel('Epoch')
        ax_flow_coherence.set_ylabel('Flow Direction Correlation')
        ax_flow_coherence.grid(True, alpha=0.3)

        # Plot 7: Virtual target convergence
        ax_virtual_convergence = axes[2, 0]
        # Compute distance to virtual target over time
        virtual_distances = []
        for data in self.epoch_data:
            gen_samples = data['gen_samples']
            virtual_samples = data['virtual_samples']
            # Compute mean distance to virtual target
            distances = []
            for gen_point in gen_samples:
                min_dist = np.min(np.linalg.norm(virtual_samples - gen_point, axis=1))
                distances.append(min_dist)
            virtual_distances.append(np.mean(distances))

        ax_virtual_convergence.plot(epochs, virtual_distances, 'purple', linewidth=2)
        ax_virtual_convergence.set_title('Convergence to Virtual Target')
        ax_virtual_convergence.set_xlabel('Epoch')
        ax_virtual_convergence.set_ylabel('Mean Distance to Virtual Target')
        ax_virtual_convergence.grid(True, alpha=0.3)

        # Plot 8: Evidence domain coverage
        ax_evidence_coverage = axes[2, 1]
        # Compute how well generated samples cover evidence domains
        evidence_coverages = []
        for data in self.epoch_data:
            gen_samples = data['gen_samples']
            evidence_list = data['evidence_list']
            domain_coverages = []
            for evidence in evidence_list:
                # Compute minimum distance from each evidence point to generated samples
                min_distances = []
                for ev_point in evidence:
                    min_dist = np.min(np.linalg.norm(gen_samples - ev_point, axis=1))
                    min_distances.append(min_dist)
                domain_coverages.append(np.mean(min_distances))
            evidence_coverages.append(np.mean(domain_coverages))

        ax_evidence_coverage.plot(epochs, evidence_coverages, 'brown', linewidth=2)
        ax_evidence_coverage.set_title('Evidence Domain Coverage')
        ax_evidence_coverage.set_xlabel('Epoch')
        ax_evidence_coverage.set_ylabel('Mean Distance to Evidence')
        ax_evidence_coverage.grid(True, alpha=0.3)

        # Plot 9: Congestion cost evolution
        ax_congestion_evolution = axes[2, 2]
        # Extract congestion costs over time
        total_congestion_costs = []
        domain_congestion_costs = {i: [] for i in range(self.num_domains)}
        for data in self.epoch_data:
            multi_congestion_info = data['multi_congestion_info']
            if multi_congestion_info and 'domains' in multi_congestion_info:
                total_cost = 0
                for domain_info in multi_congestion_info['domains']:
                    cost = domain_info['congestion_cost'].item()
                    domain_id = domain_info['domain_id']
                    domain_congestion_costs[domain_id].append(cost)
                    total_cost += cost
                total_congestion_costs.append(total_cost)
            else:
                total_congestion_costs.append(0)

        ax_congestion_evolution.plot(epochs, total_congestion_costs, 'red', linewidth=2, label='Total Cost')
        for domain_id in range(self.num_domains):
            if domain_congestion_costs[domain_id]:
                ax_congestion_evolution.plot(epochs[:len(domain_congestion_costs[domain_id])],
                         domain_congestion_costs[domain_id],
                         linewidth=1, alpha=0.7, label=f'Domain {domain_id+1}',
                         color=self.domain_colors[domain_id])
        ax_congestion_evolution.set_title('Congestion Cost Evolution')
        ax_congestion_evolution.set_xlabel('Epoch')
        ax_congestion_evolution.set_ylabel('Congestion Cost')
        ax_congestion_evolution.legend()
        ax_congestion_evolution.grid(True, alpha=0.3)

        plt.tight_layout()

        if save:
            save_path = self.save_dir / "multimarginal_flow_summary.png"
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved multi-marginal flow summary: {save_path}")

        # plt.show()

        return {
            'epochs': epochs,
            'combined_intensities': combined_intensities,
            'combined_densities': combined_densities,
            'combined_flow_mags': combined_flow_mags,
            'domain_metrics': domain_metrics,
            'flow_coherences': flow_coherences,
            'virtual_distances': virtual_distances,
            'evidence_coverages': evidence_coverages,
            'total_congestion_costs': total_congestion_costs
        }

def parse_args():
    parser = argparse.ArgumentParser(description="Section 3: Evidence-Based Perturbation with Multi-Marginal Flow Visualization")
    parser.add_argument("--seed", type=int, default=2025, help="Random seed")
    parser.add_argument("--device", type=str, default=None, help="Device to use")
    parser.add_argument("--pretrain_epochs", type=int, default=150, help="Pretraining epochs")
    parser.add_argument("--perturb_epochs", type=int, default=30, help="Perturbation epochs")
    parser.add_argument("--batch_size", type=int, default=96, help="Training batch size")
    parser.add_argument("--eval_batch_size", type=int, default=600, help="Evaluation batch size")
    parser.add_argument("--num_evidence_domains", type=int, default=3, help="Number of evidence domains")
    parser.add_argument("--samples_per_domain", type=int, default=40, help="Samples per evidence domain")
    parser.add_argument("--eta_init", type=float, default=0.045, help="Initial learning rate")
    parser.add_argument("--enable_congestion", action="store_true", default=True, help="Enable congestion tracking")
    parser.add_argument("--use_sobolev_critic", action="store_true", default=True, help="Use Sobolev-constrained critics")
    parser.add_argument("--visualize_every", type=int, default=5, help="Visualize traffic flow every N epochs")
    parser.add_argument("--save_plots", action="store_true", default=True, help="Save plots")
    parser.add_argument("--verbose", action="store_true", default=True, help="Verbose output")
    return parser.parse_args()

def run_section3_example():
    """Run Section 3 example with comprehensive multi-marginal traffic flow visualization."""
    args = parse_args()

    # Set seed and device
    set_seed(args.seed)
    device = torch.device(args.device) if args.device else compute_device()

    print("="*80)
    print("SECTION 3: EVIDENCE-BASED PERTURBATION WITH MULTI-MARGINAL TRAFFIC FLOW")
    print("="*80)
    print(f"Device: {device}")
    print(f"Theoretical components available: {_THEORETICAL_COMPONENTS_AVAILABLE}")
    print(f"Evidence domains: {args.num_evidence_domains}")

    if not _THEORETICAL_COMPONENTS_AVAILABLE:
        print("This example requires theoretical components. Exiting.")
        return

    # Check theoretical support
    support_ok = check_theoretical_support()
    if not support_ok:
        print("Warning: Some theoretical components may not work correctly.")

    # Create models
    print("\nInitializing models...")
    generator = Generator(noise_dim=2, data_dim=2, hidden_dim=256).to(device)

    # Create multiple critics for evidence domains
    critics = []
    for i in range(args.num_evidence_domains):
        if args.use_sobolev_critic:
            critic = SobolevConstrainedCritic(
                data_dim=2, hidden_dim=256,
                use_spectral_norm=True,
                lambda_sobolev=0.1,
                sobolev_bound=1.0
            ).to(device)
        else:
            from weight_perturbation import Critic
            critic = Critic(data_dim=2, hidden_dim=256).to(device)
        critics.append(critic)
    print(f"Created {len(critics)} {'Sobolev-constrained' if args.use_sobolev_critic else 'standard'} critics")

    # Pretrain generator with first critic
    print(f"\nPretraining generator for {args.pretrain_epochs} epochs...")
    real_sampler = lambda bs: sample_real_data(
        bs, means=[torch.tensor([0.0, 0.0], device=device)], std=0.5, device=device
    )

    pretrained_gen, _ = pretrain_wgan_gp(
        generator, critics[0], real_sampler,
        epochs=args.pretrain_epochs,
        batch_size=args.batch_size,
        lr=2e-4,
        gp_lambda=0.5,
        device=device,
        verbose=args.verbose
    )

    # Create evidence domains
    print(f"\nCreating {args.num_evidence_domains} evidence domains...")
    evidence_list, centers = sample_evidence_domains(
        num_domains=args.num_evidence_domains,
        samples_per_domain=args.samples_per_domain,
        random_shift=3.0,  # Spread out domains
        std=0.5,
        device=device
    )

    print("Evidence domain centers:")
    for i, center in enumerate(centers):
        print(f" Domain {i+1}: {center}")

    # Initialize multi-marginal traffic flow visualizer
    visualizer = CongestedTransportVisualizer(
        num_domains=args.num_evidence_domains,
        figsize=(20, 12),
        save_dir=f"test_results/plots/congestion_analysis/"
    )

    # Create perturber with multi-marginal congestion tracking
    print(f"\nInitializing multi-marginal perturber...")
    perturber = CTWeightPerturberTargetNotGiven(
        pretrained_gen, evidence_list, centers,
        critics=critics,
        enable_congestion_tracking=args.enable_congestion
    )

    print(f"Multi-marginal congestion tracking: {args.enable_congestion}")
    if args.enable_congestion:
        print(f"Initialized {len(perturber.multi_congestion_trackers)} domain-specific congestion trackers")

    # Custom perturbation loop with multi-marginal visualization
    print(f"\nStarting multi-marginal perturbation with flow visualization...")
    print(f"Will visualize every {args.visualize_every} epochs")

    try:
        data_dim = evidence_list[0].shape[1]
        pert_gen = perturber._create_generator_copy(data_dim)

        # Initialize perturbation state
        from weight_perturbation.utils import parameters_to_vector, vector_to_parameters
        theta_prev = parameters_to_vector(pert_gen.parameters()).clone()
        delta_theta_prev = torch.zeros_like(theta_prev)
        eta = args.eta_init
        best_vec = None
        best_ot = float('inf')

        # Main perturbation loop with multi-marginal visualization
        for epoch in range(args.perturb_epochs):
            # Estimate virtual target with congestion awareness
            virtual_samples = perturber._estimate_virtual_target_with_congestion(
                evidence_list, epoch
            )

            # Generate noise for this epoch
            noise_samples = torch.randn(args.eval_batch_size, 2, device=device)

            # Compute multi-marginal congestion if enabled
            multi_congestion_info = None
            if args.enable_congestion and critics:
                multi_congestion_info = perturber._compute_multi_marginal_congestion(pert_gen, noise_samples)

            # Visualize multi-marginal traffic flow at specified intervals
            if epoch % args.visualize_every == 0:
                print(f"\n--- Visualizing Multi-Marginal Traffic Flow at Epoch {epoch} ---")
                epoch_data = visualizer.visualize_congested_transport_step(
                    epoch, pert_gen, critics, evidence_list, virtual_samples,
                    noise_samples, multi_congestion_info, save=args.save_plots
                )

                # Print epoch statistics
                if epoch_data['domain_flows']:
                    print(f"Epoch {epoch} Multi-Marginal Statistics:")
                    for df in epoch_data['domain_flows']:
                        domain_id = df['domain_id']
                        print(f" Domain {domain_id+1}:")
                        print(f" Mean intensity: {df['intensity'].mean():.6f}")
                        print(f" Max intensity: {df['intensity'].max():.6f}")
                        print(f" Flow magnitude: {np.linalg.norm(df['flow'], axis=1).mean():.6f}")

                    if multi_congestion_info and 'domains' in multi_congestion_info:
                        total_congestion = sum(d['congestion_cost'].item() for d in multi_congestion_info['domains'])
                        print(f" Total congestion cost: {total_congestion:.6f}")

            # Compute loss and gradients
            loss, grads = perturber._compute_loss_and_grad(
                pert_gen, virtual_samples, 0.012, 0.8, 1.0
            )

            # Compute delta_theta with multi-marginal congestion awareness
            if multi_congestion_info and multi_congestion_info['domains']:
                avg_congestion_info = perturber._average_multi_marginal_congestion(multi_congestion_info)
                delta_theta = perturber._compute_delta_theta_with_congestion(
                    grads, eta, perturber.config.get('clip_norm', 0.4),
                    perturber.config.get('momentum', 0.975),
                    delta_theta_prev, avg_congestion_info
                )
            else:
                delta_theta = perturber._compute_delta_theta(
                    grads, eta, perturber.config.get('clip_norm', 0.4),
                    perturber.config.get('momentum', 0.975),
                    delta_theta_prev
                )

            # Apply update
            theta_prev = perturber._apply_parameter_update(pert_gen, theta_prev, delta_theta)
            delta_theta_prev = delta_theta.clone()

            # Validate and adapt
            ot_pert, improvement = perturber._validate_and_adapt(
                pert_gen, virtual_samples, eta, [], perturber.config.get('patience', 6), args.verbose, epoch
            )

            # Update best state
            best_ot, best_vec = perturber._update_best_state(ot_pert, pert_gen, best_ot, best_vec)

            # Adapt learning rate
            eta, no_improvement_count = perturber._adapt_learning_rate(
                eta, improvement, epoch, 0, []
            )

            # Check rollback
            if perturber._check_rollback_condition_with_congestion([], 0):
                perturber._restore_best_state(pert_gen, best_vec)
                eta *= 0.6
                delta_theta_prev = torch.zeros_like(delta_theta_prev)

            if args.verbose:
                print(f"[{epoch:2d}] OT(Pert, Evidence)={ot_pert:.4f} Improvement={improvement:.4f} eta={eta:.4f}")

        # Final restore
        perturber._restore_best_state(pert_gen, best_vec)

        # Create summary visualization
        visualizer.create_multimarginal_summary(save=args.save_plots)

        print("\nPerturbation completed successfully!")

    except Exception as e:
        print(f"Error in perturbation process: {e}")

if __name__ == "__main__":
    run_section3_example()
