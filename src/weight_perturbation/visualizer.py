import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging

try:
    from scipy.ndimage import gaussian_filter
    from scipy.interpolate import griddata, RBFInterpolator
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    print("Warning: scipy not available, using enhanced fallback methods")


from weight_perturbation import (
    compute_spatial_density,
    compute_traffic_flow,
    congestion_cost_function,
)

# Set up plotting style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Set up logging
logging.basicConfig(level=logging.INFO)

class CongestedTransportVisualizer:
    """Comprehensive visualization for congested transport analysis."""

    def __init__(self, save_dir="test_results/plots/congestion_analysis"):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True)
        self.step_data = []

    def visualize_step(self, step, generator, critic, target_samples,
                       noise_samples, lambda_congestion=1.0):
        """Visualize a single perturbation step with full congestion analysis."""
        device = next(generator.parameters()).device
        
        # Generate samples
        with torch.no_grad():
            gen_samples = generator(noise_samples)
        
        # Compute congestion metrics
        density_info = compute_spatial_density(gen_samples, bandwidth=0.12)
        sigma = density_info['density_at_samples']
        flow_info = compute_traffic_flow(
            critic, generator, noise_samples, sigma, lambda_congestion
        )
        congestion_cost = congestion_cost_function(
            flow_info['traffic_intensity'], sigma, lambda_congestion
        )
        
        # Store step data
        step_data = {
            'step': step,
            'gen_samples': gen_samples.detach().cpu().numpy(),
            'target_samples': target_samples.detach().cpu().numpy(),
            'traffic_flow': flow_info['traffic_flow'].detach().cpu().numpy(),
            'traffic_intensity': flow_info['traffic_intensity'].detach().cpu().numpy(),
            'spatial_density': sigma.detach().cpu().numpy(),
            'congestion_cost': congestion_cost.mean().detach().item()
        }
        self.step_data.append(step_data)
        
        # Create visualization
        self._create_step_plot(step_data)
        return step_data

    def _contour_quiver_plot(self, ax, x_grid, y_grid, U, V, intensity,
                             scale=1.0, title="Flow Field", cmap='plasma', contour_levels=10):
        """
        Curved quiver + contour plot for flow field.

        Args:
            ...
            intensity: for contour (colored background)
            U,V: normalized direction, scaled by intensity
        """
        # Countour intensity map
        contour = ax.contourf(x_grid, y_grid, intensity, levels=contour_levels, cmap='plasma', alpha=0.45)
        # Length of quiver arrow ~ intensity
        mag = np.maximum(np.sqrt(U ** 2 + V ** 2), 1e-8) * np.max(intensity)
        U_scaled = U / mag * intensity
        V_scaled = V / mag * intensity

        quiv = ax.quiver(x_grid, y_grid, U_scaled, V_scaled, cmap=cmap,
                         angles='xy', scale_units='xy', scale=scale, width=0.003,
                         headwidth=4, headlength=5, headaxislength=4, alpha=0.9,
                         )
        # plt.colorbar(contour, ax=ax, shrink=0.8, aspect=24, label='Traffic Intensity')
        ax.set_title(title)
        ax.set_xlabel('X₁'); ax.set_ylabel('X₂')
        ax.grid(True, alpha=0.22)
        return quiv

    def _unit_quiver(self, ax, X, Y, U, V, color_by=None, arrow_frac=0.1, width=0.004, cmap='viridis', alpha=0.85):
        """
        Draws a quiver plot that shows only directions using arrows of equal length.
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

        # Determine on-screen uniform arrow length
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
    
    def _advanced_interpolation(self, points, flows, grid_x, grid_y, method='rbf'):
        """
        Advanced interpolation methods for smooth curved flow fields.
        """
        grid_points = np.column_stack([grid_x.ravel(), grid_y.ravel()])
        if SCIPY_AVAILABLE and method == 'rbf' and len(points) > 3:
            try:
                rbf_x = RBFInterpolator(points, flows[:, 0], kernel='thin_plate_spline', smoothing=0.1)
                rbf_y = RBFInterpolator(points, flows[:, 1], kernel='thin_plate_spline', smoothing=0.1)
                flow_x_interp = rbf_x(grid_points).reshape(grid_x.shape)
                flow_y_interp = rbf_y(grid_points).reshape(grid_y.shape)
                return flow_x_interp, flow_y_interp
            except Exception:
                pass
        if SCIPY_AVAILABLE and len(points) > 3:
            try:
                flow_x_interp = griddata(points, flows[:, 0], grid_points, method='cubic', fill_value=0).reshape(grid_x.shape)
                flow_y_interp = griddata(points, flows[:, 1], grid_points, method='cubic', fill_value=0).reshape(grid_y.shape)
                return flow_x_interp, flow_y_interp
            except Exception:
                pass
        # Enhanced fallback if scipy fails or unavailable
        flow_x_interp = np.zeros_like(grid_x)
        flow_y_interp = np.zeros_like(grid_y)
        for i, grid_point in enumerate(grid_points):
            distances = np.sqrt(np.sum((points - grid_point) ** 2, axis=1)) + 1e-10
            weights = 1.0 / (distances ** 4)
            weights /= np.sum(weights)
            flow_x_interp.flat[i] = np.sum(weights * flows[:, 0])
            flow_y_interp.flat[i] = np.sum(weights * flows[:, 1])
        return flow_x_interp, flow_y_interp

    def _advanced_smoothing(self, U, V, method='gaussian', passes=3):
        """
        Advanced smoothing for ultra-smooth flow fields.
        Applies multi-scale gaussian smoothing.
        """
        if SCIPY_AVAILABLE and method == 'gaussian':
            U_smooth = U.copy()
            V_smooth = V.copy()
            for sigma in [0.8, 1.5, 2.0]:
                U_smooth = 0.7 * U_smooth + 0.3 * gaussian_filter(U_smooth, sigma=sigma)
                V_smooth = 0.7 * V_smooth + 0.3 * gaussian_filter(V_smooth, sigma=sigma)
            return U_smooth, V_smooth
        # Fallback: simple averaging
        U_smooth = U.copy()
        V_smooth = V.copy()
        for _ in range(passes):
            U_smooth = (U_smooth + np.roll(U_smooth, 1, axis=0) + np.roll(U_smooth, -1, axis=0)) / 3
            V_smooth = (V_smooth + np.roll(V_smooth, 1, axis=1) + np.roll(V_smooth, -1, axis=1)) / 3
        return U_smooth, V_smooth

    def _create_step_plot(self, step_data):
        """Create comprehensive plot for a single step."""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(f'Congested Transport Analysis - Step {step_data["step"]}',
                     fontsize=16, fontweight='bold')
        gen_samples = step_data['gen_samples']
        target_samples = step_data['target_samples']
        traffic_flow = step_data['traffic_flow']
        traffic_intensity = step_data['traffic_intensity']
        spatial_density = step_data['spatial_density']
        
        # Plot 1: Samples comparison
        axes[0, 0].scatter(gen_samples[:, 0], gen_samples[:, 1],
                           c='blue', alpha=0.6, s=30, label='Generated')
        axes[0, 0].scatter(target_samples[:, 0], target_samples[:, 1],
                           c='red', alpha=0.6, s=30, label='Target')
        axes[0, 0].set_title('Generated vs Target Samples')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 3: Traffic intensity heatmap
        scatter = axes[0, 2].scatter(gen_samples[:, 0], gen_samples[:, 1],
                                     c=traffic_intensity, cmap='plasma', s=50, alpha=0.7)
        plt.colorbar(scatter, ax=axes[0, 2], shrink=0.8, label='Traffic Intensity')
        axes[0, 2].set_title('Traffic Intensity Distribution')
        axes[0, 2].grid(True, alpha=0.3)

        # Plot 2: Traffic flow vector field
        x_min, x_max = axes[0, 2].get_xlim()
        y_min, y_max = axes[0, 2].get_ylim()
        x_grid = np.linspace(x_min, x_max, 15)
        y_grid = np.linspace(y_min, y_max, 15)
        X_grid, Y_grid = np.meshgrid(x_grid, y_grid)

        U_interp, V_interp = self._advanced_interpolation(gen_samples, traffic_flow, X_grid, Y_grid, method='rbf')
        U_smooth, V_smooth = self._advanced_smoothing(U_interp, V_interp)
        magnitude = np.sqrt(U_smooth ** 2 + V_smooth ** 2)
        quiver = self._contour_quiver_plot(axes[0, 1], x_grid, y_grid, U_smooth, V_smooth, magnitude, scale=1.2, title=f"Combined Multi-Marginal Flow Field", cmap='viridis')

        # n_arrows = min(50, len(gen_samples))
        # indices = np.random.choice(len(gen_samples), n_arrows, replace=False)
        # quiver = self._unit_quiver(
        #     axes[0,1],
        #     gen_samples[indices, 0], gen_samples[indices, 1],
        #     traffic_flow[indices, 0], traffic_flow[indices, 1],
        #     color_by=traffic_intensity[indices],
        #     arrow_frac=0.1, width=0.004, cmap='viridis', alpha=0.85
        # )
        # plt.colorbar(quiver, ax=axes[0, 1], shrink=0.8, label='Traffic Intensity')
        axes[0, 1].set_title('Traffic Flow Vector Field w_Q(x)')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 4: Spatial density
        density_scatter = axes[1, 0].scatter(gen_samples[:, 0], gen_samples[:, 1],
                                             c=spatial_density, cmap='coolwarm', s=50, alpha=0.7)
        plt.colorbar(density_scatter, ax=axes[1, 0], shrink=0.8, label='Spatial Density σ(x)')
        axes[1, 0].set_title('Spatial Density Distribution')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 5: Intensity histogram
        axes[1, 1].hist(traffic_intensity, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        axes[1, 1].axvline(traffic_intensity.mean(), color='red', linestyle='--',
                           label=f'Mean: {traffic_intensity.mean():.4f}')
        axes[1, 1].set_title('Traffic Intensity Distribution')
        axes[1, 1].set_xlabel('Traffic Intensity')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        # Plot 6: Statistics summary
        axes[1, 2].axis('off')
        stats_text = f"""Step {step_data['step']} Statistics:
Traffic Flow:
• Mean Intensity: {traffic_intensity.mean():.6f}
• Max Intensity: {traffic_intensity.max():.6f}
• Flow Magnitude: {np.linalg.norm(traffic_flow, axis=1).mean():.6f}
Spatial Properties:
• Mean Density: {spatial_density.mean():.6f}
• Density Variance: {spatial_density.var():.6f}
Congestion:
• Total Cost: {step_data['congestion_cost']:.6f}
"""
        axes[1, 2].text(0.05, 0.95, stats_text, transform=axes[1, 2].transAxes,
                        verticalalignment='top', fontsize=10, fontfamily='monospace',
                        bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
        plt.tight_layout()
        
        # Save plot
        save_path = self.save_dir / f"step_{step_data['step']:03d}.png"
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        logging.info(f"Saved visualization: {save_path}")

    def create_evolution_summary(self):
        """Create summary of evolution across all steps."""
        if not self.step_data:
            logging.warning("No data available for summary.")
            return
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Congested Transport Evolution Summary', fontsize=16, fontweight='bold')
        steps = [data['step'] for data in self.step_data]
        costs = [data['congestion_cost'] for data in self.step_data]
        mean_intensities = [data['traffic_intensity'].mean() for data in self.step_data]
        mean_densities = [data['spatial_density'].mean() for data in self.step_data]
        
        # Congestion cost evolution
        axes[0, 0].plot(steps, costs, 'b-o', linewidth=2)
        axes[0, 0].set_title('Congestion Cost Evolution')
        axes[0, 0].set_xlabel('Step')
        axes[0, 0].set_ylabel('H(x, i_Q)')
        axes[0, 0].grid(True)
        
        # Traffic intensity evolution
        axes[0, 1].plot(steps, mean_intensities, 'g-o', linewidth=2)
        axes[0, 1].set_title('Mean Traffic Intensity Evolution')
        axes[0, 1].set_xlabel('Step')
        axes[0, 1].set_ylabel('Mean |w_Q|')
        axes[0, 1].grid(True)
        
        # Spatial density evolution
        axes[1, 0].plot(steps, mean_densities, 'm-o', linewidth=2)
        axes[1, 0].set_title('Mean Spatial Density Evolution')
        axes[1, 0].set_xlabel('Step')
        axes[1, 0].set_ylabel('Mean σ(x)')
        axes[1, 0].grid(True)
        
        # Final comparison
        if len(self.step_data) >= 2:
            initial_data = self.step_data[0]
            final_data = self.step_data[-1]
            initial_samples = initial_data['gen_samples']
            final_samples = final_data['gen_samples']
            axes[1, 1].scatter(initial_samples[:, 0], initial_samples[:, 1],
                               c='lightblue', alpha=0.5, s=20, label='Initial')
            axes[1, 1].scatter(final_samples[:, 0], final_samples[:, 1],
                               c='darkblue', alpha=0.7, s=20, label='Final')
            axes[1, 1].scatter(final_data['target_samples'][:, 0],
                               final_data['target_samples'][:, 1],
                               c='red', alpha=0.5, s=20, label='Target')
            axes[1, 1].set_title('Initial vs Final Samples')
            axes[1, 1].legend()
            axes[1, 1].grid(True)
        plt.tight_layout()
        save_path = self.save_dir / "evolution_summary.png"
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        logging.info(f"Saved evolution summary: {save_path}")

class MultiMarginalCongestedTransportVisualizer:
    """
    Comprehensive visualizer for congested transport theory with given multi-marginal distribution.

    Args:
        figsize (tuple): Figure size for plots.
        save_dir (str): Directory to save visualizations.
    """

    def __init__(self, num_domains=3, figsize=(20, 14), save_dir="test_results/plots/congestion_analysis"):
        self.num_domains = num_domains
        self.figsize = figsize
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.epoch_data = []
        self.domain_colors = sns.color_palette("husl", n_colors=num_domains)

    def _advanced_interpolation(self, points, flows, grid_x, grid_y, method='rbf'):
        """
        Advanced interpolation methods for smooth curved flow fields.
        """
        grid_points = np.column_stack([grid_x.ravel(), grid_y.ravel()])
        if SCIPY_AVAILABLE and method == 'rbf' and len(points) > 3:
            try:
                rbf_x = RBFInterpolator(points, flows[:, 0], kernel='thin_plate_spline', smoothing=0.1)
                rbf_y = RBFInterpolator(points, flows[:, 1], kernel='thin_plate_spline', smoothing=0.1)
                flow_x_interp = rbf_x(grid_points).reshape(grid_x.shape)
                flow_y_interp = rbf_y(grid_points).reshape(grid_y.shape)
                return flow_x_interp, flow_y_interp
            except Exception:
                pass
        if SCIPY_AVAILABLE and len(points) > 3:
            try:
                flow_x_interp = griddata(points, flows[:, 0], grid_points, method='cubic', fill_value=0).reshape(grid_x.shape)
                flow_y_interp = griddata(points, flows[:, 1], grid_points, method='cubic', fill_value=0).reshape(grid_y.shape)
                return flow_x_interp, flow_y_interp
            except Exception:
                pass
        # Enhanced fallback if scipy fails or unavailable
        flow_x_interp = np.zeros_like(grid_x)
        flow_y_interp = np.zeros_like(grid_y)
        for i, grid_point in enumerate(grid_points):
            distances = np.sqrt(np.sum((points - grid_point) ** 2, axis=1)) + 1e-10
            weights = 1.0 / (distances ** 4)
            weights /= np.sum(weights)
            flow_x_interp.flat[i] = np.sum(weights * flows[:, 0])
            flow_y_interp.flat[i] = np.sum(weights * flows[:, 1])
        return flow_x_interp, flow_y_interp

    def _advanced_smoothing(self, U, V, method='gaussian', passes=3):
        """
        Advanced smoothing for ultra-smooth flow fields.
        Applies multi-scale gaussian smoothing.
        """
        if SCIPY_AVAILABLE and method == 'gaussian':
            U_smooth = U.copy()
            V_smooth = V.copy()
            for sigma in [0.8, 1.5, 2.0]:
                U_smooth = 0.7 * U_smooth + 0.3 * gaussian_filter(U_smooth, sigma=sigma)
                V_smooth = 0.7 * V_smooth + 0.3 * gaussian_filter(V_smooth, sigma=sigma)
            return U_smooth, V_smooth
        # Fallback: simple averaging
        U_smooth = U.copy()
        V_smooth = V.copy()
        for _ in range(passes):
            U_smooth = (U_smooth + np.roll(U_smooth, 1, axis=0) + np.roll(U_smooth, -1, axis=0)) / 3
            V_smooth = (V_smooth + np.roll(V_smooth, 1, axis=1) + np.roll(V_smooth, -1, axis=1)) / 3
        return U_smooth, V_smooth

    def _create_curved_streamplot(self, ax, x_grid, y_grid, U, V, magnitude, title="Flow Field", cmap='viridis'):
        """
        Create curved streamplot visualization. Falls back to quiver if an error occurs.
        """
        try:
            strm = ax.streamplot(
                x_grid, y_grid, U, V, color=magnitude, cmap=cmap,
                density=[1.8, 1.5], arrowsize=1.5, arrowstyle='->', linewidth=2.0,
                integration_direction='both', maxlength=10.0
            )
            cbar = plt.colorbar(strm.lines, ax=ax, shrink=0.8, aspect=30)
            cbar.set_label('Traffic Intensity')
            ax.set_title(title)
            return strm
        except Exception:
            # Fallback to quiver visualization
            mag_norm = np.sqrt(U ** 2 + V ** 2) + 1e-8
            U_norm = U / mag_norm
            V_norm = V / mag_norm
            scale_factor = 0.4
            max_mag = np.max(mag_norm) if np.max(mag_norm) > 0 else 1
            relative_scale = 1 + 2 * (mag_norm / max_mag)
            U_scaled = U_norm * scale_factor * relative_scale
            V_scaled = V_norm * scale_factor * relative_scale
            quiv = ax.quiver(
                x_grid, y_grid, U_scaled, V_scaled, mag_norm,
                cmap=cmap, alpha=0.85, scale=6, scale_units='xy', angles='xy',
                width=0.004, headwidth=4, headlength=6, headaxislength=5,
                edgecolors='black', linewidth=0.3
            )
            cbar = plt.colorbar(quiv, ax=ax, shrink=0.8, aspect=30)
            cbar.set_label('Traffic Intensity')
            ax.set_title(title)
            return quiv
    
    def _contour_quiver_plot(self, ax, x_grid, y_grid, U, V, intensity,
                             scale=1.0, title="Flow Field", cmap='plasma', contour_levels=10):
        """
        Curved quiver + contour plot for flow field.

        Args:
            ...
            intensity: for contour (colored background)
            U,V: normalized direction, scaled by intensity
        """
        # Countour intensity map
        contour = ax.contourf(x_grid, y_grid, intensity, levels=contour_levels, cmap='plasma', alpha=0.45)
        # Length of quiver arrow ~ intensity
        mag = np.maximum(np.sqrt(U ** 2 + V ** 2), 1e-8) * np.max(intensity)
        U_scaled = U / mag * intensity
        V_scaled = V / mag * intensity

        quiv = ax.quiver(x_grid, y_grid, U_scaled, V_scaled, cmap=cmap,
                         angles='xy', scale_units='xy', scale=scale, width=0.003,
                         headwidth=4, headlength=5, headaxislength=4, alpha=0.9,
                         )
        plt.colorbar(contour, ax=ax, shrink=0.8, aspect=24, label='Traffic Intensity')
        ax.set_title(title)
        ax.set_xlabel('X₁'); ax.set_ylabel('X₂')
        ax.grid(True, alpha=0.22)
        return quiv
    
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

        # Compute flow information for each domain with stability
        domain_flows = []
        for i, (evidence, critic) in enumerate(zip(evidence_list, critics)):
            if critic is None:
                continue

            try:
                # Compute spatial density for this domain with increased bandwidth for stability
                all_samples = torch.cat([gen_samples, evidence], dim=0)
                density_info = compute_spatial_density(all_samples, bandwidth=0.2)
                sigma_gen = density_info['density_at_samples'][:gen_samples.shape[0]]

                # Compute traffic flow for this domain with reduced lambda for stability
                flow_info = compute_traffic_flow(
                    critic, generator, noise_samples, sigma_gen, lambda_param=50  # Reduced from 100
                )

                domain_flows.append({
                    'domain_id': i,
                    'flow': flow_info['traffic_flow'].detach().cpu().numpy(),
                    'intensity': flow_info['traffic_intensity'].detach().cpu().numpy(),
                    'density': sigma_gen.detach().cpu().numpy(),
                    'gradient_norm': flow_info['gradient_norm'].detach().cpu().numpy(),
                    'evidence': evidence.detach().cpu().numpy()
                })
            
            except Exception as e:
                print(f"Warning: Flow computation failed for domain {i}: {e}")
                # Create dummy flow data
                gen_samples_np = gen_samples.detach().cpu().numpy()
                domain_flows.append({
                    'domain_id': i,
                    'flow': np.zeros_like(gen_samples_np),
                    'intensity': np.zeros(gen_samples_np.shape[0]),
                    'density': np.ones(gen_samples_np.shape[0]) * 0.01,
                    'gradient_norm': np.zeros(gen_samples_np.shape[0]),
                    'evidence': evidence.detach().cpu().numpy()
                })

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
        gs = fig.add_gridspec(3, 4, hspace=0.35, wspace=0.35)
        fig.suptitle(f'Multi-Marginal Traffic Flow Analysis - Epoch {epoch}', fontsize=20, fontweight='bold')

        # Plot 1: Overview with all domains and virtual target
        ax_overview = fig.add_subplot(gs[0, :2])

        # Plot generated samples
        gen_samples_np = gen_samples.detach().cpu().numpy()
        ax_overview.scatter(gen_samples_np[:, 0], gen_samples_np[:, 1],
                    c='blue', alpha=0.6, s=35, label='Generated', zorder=3)

        # Plot evidence domains with different colors
        for i, evidence in enumerate(evidence_list):
            ax_overview.scatter(evidence[:, 0].cpu(), evidence[:, 1].cpu(),
                        c=[self.domain_colors[i]], alpha=0.8, s=50,
                        label=f'Evidence {i+1}', marker='s', zorder=2)

        # Plot virtual target
        ax_overview.scatter(virtual_samples[:, 0].cpu(), virtual_samples[:, 1].cpu(),
                    c='red', alpha=0.6, s=30, label='Virtual Target', marker='^', zorder=1)

        ax_overview.set_title('Multi-Domain Overview', fontsize=14, fontweight='bold')
        ax_overview.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax_overview.grid(True, alpha=0.3)
        ax_overview.set_xlabel('X₁', fontsize=12)
        ax_overview.set_ylabel('X₂', fontsize=12)

        # Plot 2: Combined traffic flow vector field with visualization
        ax_combined_flow = fig.add_subplot(gs[0, 2:])

        if domain_flows and any(df['intensity'].max() > 1e-6 for df in domain_flows):
            # Combine flows from all domains more carefully
            all_flows = np.array([df['flow'] for df in domain_flows])
            all_intensities = np.array([df['intensity'] for df in domain_flows])
            
            # Weight flows by intensity to emphasize more active regions
            max_intensities = np.max(all_intensities, axis=1, keepdims=True)
            intensity_weights = all_intensities / (max_intensities + 1e-8)
            weighted_flows = all_flows * intensity_weights[:, :, np.newaxis]
            combined_flow = np.mean(weighted_flows, axis=0)
            combined_intensity = np.mean(all_intensities, axis=0)

            x_min, x_max = ax_overview.get_xlim()
            y_min, y_max = ax_overview.get_ylim()
            # ax_overview
            # Create a denser grid for smoother flow field visualization
            # x_range = gen_samples_np[:, 0].max() - gen_samples_np[:, 0].min()
            # y_range = gen_samples_np[:, 1].max() - gen_samples_np[:, 1].min()
            # margin = max(x_range, y_range) * 0.3
            # x_min = gen_samples_np[:, 0].min() - margin
            # x_max = gen_samples_np[:, 0].max() + margin
            # y_min = gen_samples_np[:, 1].min() - margin
            # y_max = gen_samples_np[:, 1].max() + margin
            grid_resolution = 40
            x_grid = np.linspace(x_min, x_max, grid_resolution)
            y_grid = np.linspace(y_min, y_max, grid_resolution)
            X_grid, Y_grid = np.meshgrid(x_grid, y_grid)
            try:
                U_interp, V_interp = self._advanced_interpolation(gen_samples_np, combined_flow, X_grid, Y_grid, method='rbf')
                U_smooth, V_smooth = self._advanced_smoothing(U_interp, V_interp)
                magnitude = np.sqrt(U_smooth ** 2 + V_smooth ** 2)
                # self._create_curved_streamplot(ax_combined_flow, x_grid, y_grid, U_smooth, V_smooth, magnitude, title="Combined Multi-Marginal Flow Field", cmap='viridis')
                self._contour_quiver_plot(ax_combined_flow, x_grid, y_grid, U_smooth, V_smooth, magnitude, scale=0.75, title=f"Combined Multi-Marginal Flow Field", cmap='viridis')
            except Exception as e:
                print(f"Advanced flow visualization failed: {e}")
                # self._create_curved_streamplot(ax_combined_flow, X_grid, Y_grid, np.zeros_like(X_grid), np.zeros_like(Y_grid), np.zeros_like(X_grid), cmap='viridis')
                self._contour_quiver_plot(ax_combined_flow, X_grid, Y_grid, np.zeros_like(X_grid), np.zeros_like(Y_grid), np.zeros_like(X_grid), scale=0.5, cmap='viridis')
        else:
            ax_combined_flow.set_title('Combined Multi-Marginal Flow Field', fontsize=14, fontweight='bold')


        # Plots 3-5: Domain-specific traffic flows with visualization
        for i, domain_flow in enumerate(domain_flows[:3]):  # Show up to 3 domains
            ax_domain_flow = fig.add_subplot(gs[1, i])
            flow = domain_flow['flow']
            intensity = domain_flow['intensity']
            evidence = domain_flow['evidence']

            # Plot evidence domain
            ax_domain_flow.scatter(evidence[:, 0], evidence[:, 1],
                       c=[self.domain_colors[i]], alpha=0.8, s=60,
                       label=f'Evidence {i+1}', marker='s')

            # Plot flow vectors for this domain with improved visualization
            gen_samples_np = gen_samples.detach().cpu().numpy()
            
            # print(f"domain-{i}:", intensity)

            # Filter flows by intensity for better visualization
            if np.max(intensity) > 1e-4:
                # Keep flows above certain intensity threshold
                # intensity_threshold = np.percentile(intensity[intensity > 0], 3) if np.any(intensity > 0) else 0
                # flow_mask = intensity > intensity_threshold
                flow_mask = intensity > (0.02 * np.max(intensity))

                if np.sum(flow_mask) >= 2:
                    masked_samples = gen_samples_np[flow_mask]
                    masked_flows = flow[flow_mask]
                    evidence_center = np.mean(evidence, axis=0)
                    domain_range = 2.0
                    grid_res = 14
                    x_local = np.linspace(evidence_center[0] - domain_range, evidence_center[0] + domain_range, grid_res)
                    y_local = np.linspace(evidence_center[1] - domain_range, evidence_center[1] + domain_range, grid_res)
                    X_local, Y_local = np.meshgrid(x_local, y_local)
                    try:
                        U_local, V_local = self._advanced_interpolation(masked_samples, masked_flows, X_local, Y_local, method='rbf')
                        U_local_smooth, V_local_smooth = self._advanced_smoothing(U_local, V_local, passes=2)
                        magnitude_local = np.sqrt(U_local_smooth ** 2 + V_local_smooth ** 2)
                        mag_min, mag_max = np.min(magnitude_local), np.max(magnitude_local)
                        contour_levels = np.linspace(mag_min, mag_max if mag_max > mag_min else mag_min+1, 10)

                        # self._create_curved_streamplot(ax_domain_flow, x_local, y_local, U_local_smooth, V_local_smooth, magnitude_local, title=f"Domain {i+1} Flow Field", cmap='plasma')
                        self._contour_quiver_plot(ax_domain_flow, x_local, y_local, U_local_smooth, V_local_smooth, magnitude_local, scale=3, title=f"Domain {i+1} Flow Field", cmap='plasma', contour_levels=contour_levels)
                        stats_text = f"""Domain {i+1} Stats:\nMean Intensity: {intensity.mean():.4f}\nMax Intensity: {intensity.max():.4f}\nFlow Magnitude: {np.linalg.norm(flow, axis=1).mean():.4f}\nPoints: {len(flow)}"""
                        ax_domain_flow.text(0.02, 0.98, stats_text, transform=ax_domain_flow.transAxes, verticalalignment='top', bbox=dict(boxstyle='round,pad=0.3', facecolor='lightblue', alpha=0.8), fontsize=9, fontweight='bold')
                    
                    except Exception as e:
                        print(f"Domain {i} advanced visualization failed: {e}")
            ax_domain_flow.set_title(f'Domain {i+1} Flow Field', fontsize=12, fontweight='bold')
            ax_domain_flow.grid(True, alpha=0.3)
            ax_domain_flow.set_xlabel('X₁', fontsize=10)
            ax_domain_flow.set_ylabel('X₂', fontsize=10)

            # Add domain statistics
            stats_text = f"""
Domain {i+1} Stats:
Mean Intensity: {intensity.mean():.4f}
Max Intensity: {intensity.max():.4f}
Flow Magnitude: {np.linalg.norm(flow, axis=1).mean():.4f}
"""
            ax_domain_flow.text(0.02, 0.98, stats_text, transform=ax_domain_flow.transAxes,
                    verticalalignment='top', bbox=dict(boxstyle='round,pad=0.3', facecolor='lightblue', alpha=0.8),
                    fontsize=9, fontweight='bold')

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

        plt.close()
        return epoch_data

    def _flow_field_quiver(self, ax, X, Y, U, V, color_by=None, arrow_scale=0.3, width=0.004, cmap='viridis', alpha=0.85):
        """
        Quiver plot with improved arrow visualization for curved flow fields.
        """
        X = np.asarray(X); Y = np.asarray(Y); U = np.asarray(U); V = np.asarray(V)
        
        # Compute magnitude and normalize
        mag = np.sqrt(U**2 + V**2)
        non_zero_mask = mag > 1e-8
        
        if not np.any(non_zero_mask):
            return None
            
        # Filter out zero magnitude vectors
        X_filt = X[non_zero_mask]
        Y_filt = Y[non_zero_mask]
        U_filt = U[non_zero_mask]
        V_filt = V[non_zero_mask]
        mag_filt = mag[non_zero_mask]
        
        if color_by is not None:
            color_by_filt = color_by[non_zero_mask]
        else:
            color_by_filt = mag_filt
        
        # Normalize direction vectors
        U_norm = U_filt / mag_filt
        V_norm = V_filt / mag_filt
        
        # Scale arrows based on data range
        x_range = np.ptp(X_filt) if len(X_filt) > 1 else 1.0
        y_range = np.ptp(Y_filt) if len(Y_filt) > 1 else 1.0
        scale_factor = arrow_scale * min(x_range, y_range) / len(X_filt)**0.5
        
        # Create arrows with adaptive scaling
        U_scaled = U_norm * scale_factor * (1 + 0.5 * mag_filt / np.max(mag_filt))
        V_scaled = V_norm * scale_factor * (1 + 0.5 * mag_filt / np.max(mag_filt))
        
        quiv = ax.quiver(
            X_filt, Y_filt, U_scaled, V_scaled,
            color_by_filt,
            cmap=cmap, alpha=alpha,
            angles='xy', scale_units='xy', scale=1.0,
            width=width, headwidth=4, headlength=5, headaxislength=4
        )
        
        return quiv

    def _unit_quiver(self, ax, X, Y, U, V, color_by=None, arrow_scale=0.1, width=0.004, cmap='viridis', alpha=0.85):
        """
        Draws a quiver plot that shows only directions using arrows of equal length.
        - arrow_scale: Arrow length as a fraction of the axis range (e.g., 0.1 means 10% of the x-range)
        - color_by: Values to encode color (e.g., traffic_intensity); if None, use a single color
        """
        return self._flow_field_quiver(ax, X, Y, U, V, color_by, arrow_scale, width, cmap, alpha)

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
    
