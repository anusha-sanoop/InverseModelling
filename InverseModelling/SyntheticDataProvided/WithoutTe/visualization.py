"""
Visualization tools for elastic thickness inversion results
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


# Create custom terrain colormap (white for low elevations)
terrain_cmap = plt.get_cmap('terrain')
terrain_colors = terrain_cmap(np.linspace(0, 1, 256))
terrain_colors[0:64] = [1.0, 1.0, 1.0, 1.0]
custom_terrain_cmap = ListedColormap(terrain_colors)


def plot_inversion_results(topography, moho_obs, result, X, Y, figsize=(15, 10)):
    """
    Plot inversion results showing topography, observed/predicted Moho, and residuals
    
    Parameters:
    -----------
    topography : 2D array
        Topography data (m)
    moho_obs : 2D array
        Observed Moho depth (m)
    result : dict
        Inversion results from invert_elastic_thickness
    X, Y : 2D arrays
        Coordinate grids
    figsize : tuple
        Figure size
    """
    moho_pred = result['moho_pred']
    Te_best = result['Te_best']
    rms_best = result['rms_best']
    
    # Calculate extent
    x_min, x_max = np.min(X)/1000, np.max(X)/1000
    y_min, y_max = np.min(Y)/1000, np.max(Y)/1000
    extent = [x_min, x_max, y_min, y_max]
    
    fig, axes = plt.subplots(2, 3, figsize=figsize)
    
    # Topography
    im1 = axes[0, 0].imshow(topography, cmap=custom_terrain_cmap, 
                            aspect='equal', origin='lower', extent=extent)
    axes[0, 0].set_title('Topographic Load (m)')
    axes[0, 0].set_xlabel('X (km)')
    axes[0, 0].set_ylabel('Y (km)')
    plt.colorbar(im1, ax=axes[0, 0], label='Elevation (m)')
    
    # Observed Moho
    im2 = axes[0, 1].imshow(moho_obs/1000, cmap='viridis_r', 
                           aspect='equal', origin='lower', extent=extent)
    axes[0, 1].set_title('Observed Moho (km)')
    axes[0, 1].set_xlabel('X (km)')
    axes[0, 1].set_ylabel('Y (km)')
    plt.colorbar(im2, ax=axes[0, 1], label='Depth (km)')
    
    # Predicted Moho
    im3 = axes[0, 2].imshow(moho_pred/1000, cmap='viridis_r', 
                           aspect='equal', origin='lower', extent=extent,
                           vmin=np.min(moho_obs)/1000, vmax=np.max(moho_obs)/1000)
    axes[0, 2].set_title(f'Predicted Moho (km)\nTe = {Te_best/1000:.1f} km')
    axes[0, 2].set_xlabel('X (km)')
    axes[0, 2].set_ylabel('Y (km)')
    plt.colorbar(im3, ax=axes[0, 2], label='Depth (km)')
    
    # Residual
    residual = moho_obs - moho_pred
    res_max = np.max(np.abs(residual))
    im4 = axes[1, 0].imshow(residual/1000, cmap='RdBu_r', 
                            aspect='equal', origin='lower', extent=extent,
                            vmin=-res_max/1000, vmax=res_max/1000)
    axes[1, 0].set_title(f'Residual (km)\nRMS = {rms_best/1000:.2f} km')
    axes[1, 0].set_xlabel('X (km)')
    axes[1, 0].set_ylabel('Y (km)')
    plt.colorbar(im4, ax=axes[1, 0], label='Residual (km)')
    
    # Cross-section comparison
    mid_row = topography.shape[0] // 2
    x_km = X[mid_row, :] / 1000
    
    axes[1, 1].plot(x_km, moho_obs[mid_row, :]/1000, 'b-', 
                   label='Observed Moho', linewidth=2)
    axes[1, 1].plot(x_km, moho_pred[mid_row, :]/1000, 'r--', 
                   label='Predicted Moho', linewidth=2)
    axes[1, 1].set_xlabel('X (km)')
    axes[1, 1].set_ylabel('Moho Depth (km)')
    axes[1, 1].set_title('Cross-section Comparison')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].invert_yaxis()  # Plot depth increasing downwards
    
    # Histogram of residuals
    axes[1, 2].hist(residual.flatten()/1000, bins=50, alpha=0.7, 
                   edgecolor='black', color='blue')
    axes[1, 2].set_xlabel('Residual (km)')
    axes[1, 2].set_ylabel('Frequency')
    axes[1, 2].set_title('Residual Distribution')
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def plot_te_map(results, X, Y, figsize=(12, 10)):
    """
    Plot Te map from moving window analysis
    
    Parameters:
    -----------
    results : dict
        Results from moving window analysis
    X, Y : 2D arrays
        Coordinate grids
    figsize : tuple
        Figure size
    """
    Te_map = results['Te_map']
    rms_map = results['rms_map']
    x_centers = results['x_centers']
    y_centers = results['y_centers']
    
    # Create coordinate grids for plotting
    x_centers_km = x_centers * np.mean(np.diff(X[0, :])) / 1000
    y_centers_km = y_centers * np.mean(np.diff(Y[:, 0])) / 1000
    
    extent_te = [x_centers_km.min(), x_centers_km.max(),
                 y_centers_km.min(), y_centers_km.max()]
    
    # Create colormap that handles NaN values
    viridis_with_bad = plt.cm.get_cmap('viridis').copy()
    viridis_with_bad.set_bad('lightgray')
    
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Te map
    im1 = axes[0].imshow(Te_map/1000, extent=extent_te,
                         cmap=viridis_with_bad, origin='lower', aspect='equal')
    axes[0].set_title('Elastic Thickness (Te) Map')
    axes[0].set_xlabel('X (km)')
    axes[0].set_ylabel('Y (km)')
    plt.colorbar(im1, ax=axes[0], label='Te (km)')
    
    # RMS map
    im2 = axes[1].imshow(rms_map/1000, extent=extent_te,
                         cmap=viridis_with_bad, origin='lower', aspect='equal')
    axes[1].set_title('RMS Misfit Map')
    axes[1].set_xlabel('X (km)')
    axes[1].set_ylabel('Y (km)')
    plt.colorbar(im2, ax=axes[1], label='RMS (km)')
    
    plt.tight_layout()
    return fig


def plot_sensitivity(Te_values, rms_values, figsize=(10, 6)):
    """
    Plot sensitivity analysis results
    
    Parameters:
    -----------
    Te_values : array
        Te values (m)
    rms_values : array
        RMS misfit values (in meters)
    figsize : tuple
        Figure size
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    Te_km = Te_values / 1000
    ax.plot(Te_km, rms_values/1000, 'b-', linewidth=2)
    ax.set_xlabel('Elastic Thickness (km)')
    ax.set_ylabel('RMS Misfit (km)')
    ax.set_title('Sensitivity Analysis: RMS vs Elastic Thickness')
    ax.grid(True, alpha=0.3)
    
    # Mark minimum
    min_idx = np.argmin(rms_values)
    ax.plot(Te_km[min_idx], rms_values[min_idx]/1000, 'ro', markersize=8,
            label=f'Minimum: Te = {Te_km[min_idx]:.1f} km')
    ax.legend()
    
    plt.tight_layout()
    return fig

