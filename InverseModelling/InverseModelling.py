"""
Inverse modeling of elastic thickness by convolution method
Based on Braitenberg's work on the eastern Alps

[CHANGED]
This implementation uses the convolution method to estimate the effective
elastic thickness (Te) of the lithosphere from topography and MoHo/CMI data,
as described in Braitenberg et al. (2002).
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import fft
from scipy.optimize import minimize_scalar
import warnings
from matplotlib.colors import ListedColormap

# [NEW] Create custom colormap
# Get the original 'terrain' colormap
terrain_cmap = plt.get_cmap('terrain')
# Get all 256 colors from it
terrain_colors = terrain_cmap(np.linspace(0, 1, 256))
# 'terrain' uses blue/cyan for the first 25% (64 entries)
# We set these to white (R=1, G=1, B=1, Alpha=1)
terrain_colors[0:64] = [1.0, 1.0, 1.0, 1.0]
# Create a new colormap from our modified list of colors
custom_terrain_cmap = ListedColormap(terrain_colors)


class ElasticThicknessInversion:
    """
    Class for inverse modeling of elastic thickness using convolution method
    """

    def __init__(self, dx=1000, dy=1000):
        """
        Initialize the inversion class

        Parameters:
        -----------
        dx, dy : float
            Grid spacing in meters (default: 1000m)
        """
        self.dx = dx
        self.dy = dy

        # Physical constants
        self.G = 6.67430e-11  # Gravitational constant (m³/kg/s²)
        self.g = 9.81         # Gravitational acceleration (m/s²)

        # Densities based on Braitenberg et al. (2002), Table 1
        # Using values from the "Eastern Alps" column as a realistic example
        self.rho_load = 2670    # Load density (kg/m³) (rho_1)
        self.rho_m = 3300       # Mantle density (kg/m³)
        self.rho_infill = 2800  # Infill density (kg/m³) (e.g., mean crustal density)

        # Using E and nu from the paper
        self.E = 1.0e11         # Young's modulus (Pa)
        self.nu = 0.25          # Poisson's ratio

        # Flexural rigidity factor (D = D_factor * Te**3)
        self.D_factor = self.E / (12 * (1 - self.nu**2)) #

    def calculate_flexure_filter(self, k, Te):
        """
        Calculate flexure filter function F(k) in wavenumber domain
        Based on Braitenberg et al. (2002), Eq. 3

        W(k) = F(k) * H(k)
        where:
        W(k) = FT of Moho flexure w(r)
        H(k) = FT of topographic load h(r)
        F(k) = rho_load / ( (rho_m - rho_infill) + (D/g) * k**4 )

        Parameters:
        -----------
        k : array
            Wavenumber magnitude (rad/m)
        Te : float
            Elastic thickness (m)

        Returns:
        --------
        F_k : array
            Flexure filter function
        """
        if Te < 1e-3: # Treat Te=0 as pure Airy isostasy
            k = 0.0

        D = self.D_factor * Te**3  # Flexural rigidity

        # Denominator of Eq. 3
        # (rho_m - rho_infill) is the buoyancy term
        # (D/g) * k**4 is the flexural term
        denominator = (self.rho_m - self.rho_infill) + (D / self.g) * k**4

        # Avoid division by zero if (rho_m == rho_infill) and k=0
        denominator = np.maximum(denominator, 1e-10)

        # Flexure filter F(k)
        # We use rho_load for the numerator (rho_l in the paper)
        F_k = self.rho_load / denominator

        return F_k

    def predict_moho_flexure(self, topography_load, Te):
        """
        Forward model: calculate predicted Moho flexure from load and Te
        This follows the convolution method

        Parameters:
        -----------
        topography_load : 2D array
            Topographic load (m).
            NOTE: This assumes H(k) is FT of topography height.
        Te : float
            Elastic thickness (m)

        Returns:
        --------
        moho_pred : 2D array
            Predicted Moho undulation (m)
        """
        ny, nx = topography_load.shape

        # Create wavenumber grids
        kx = 2 * np.pi * fft.fftfreq(nx, self.dx)
        ky = 2 * np.pi * fft.fftfreq(ny, self.dy)
        KX, KY = np.meshgrid(kx, ky)
        k = np.sqrt(KX**2 + KY**2)

        # Handle k=0 case
        # At k=0, the filter returns the Airy ratio: rho_load / (rho_m - rho_infill)
        # No need to set k[0,0]=1e-10 if denominator is handled in filter func

        # FFT of topographic load
        load_fft = fft.fft2(topography_load)

        # Calculate flexure filter
        flexure_filter = self.calculate_flexure_filter(k, Te)

        # Predicted Moho in frequency domain W(k) = F(k) * H(k)
        moho_fft = load_fft * flexure_filter

        # Convert back to spatial domain w(r)
        moho_pred = np.real(fft.ifft2(moho_fft))

        return moho_pred

    def misfit_function(self, Te, topography_load, moho_obs, mask=None):
        """
        Calculate misfit between observed and predicted Moho undulations
        This follows the paper's method of minimizing the RMS error
        between the model CMI and the observed CMI.

        Parameters:
        -----------
        Te : float
            Elastic thickness (m)
        topography_load : 2D array
            Topography data (m)
        moho_obs : 2D array
            Observed Moho undulations (m)
        mask : 2D array, optional
            Mask for valid data points

        Returns:
        --------
        rms : float
            RMS misfit (in meters of Moho)
        """
        moho_pred = self.predict_moho_flexure(topography_load, Te)

        if mask is not None:
            residual = (moho_obs - moho_pred)[mask]
        else:
            residual = moho_obs - moho_pred

        # Demean the residual to focus on undulations, not absolute depth
        # This is important as the flexure model (FFT) is relative
        residual = residual - np.mean(residual)

        rms = np.sqrt(np.mean(residual**2))
        return rms

    def invert_elastic_thickness(self, topography_load, moho_obs, Te_range=(1000, 50000),
                                mask=None, method='bounded'):
        """
        Invert for elastic thickness by minimizing Moho misfit

        Parameters:
        -----------
        topography_load : 2D array
            Topography data (m)
        moho_obs : 2D array
            Observed Moho undulations (m)
        Te_range : tuple
            Range of Te values to search (m)
        mask : 2D array, optional
            Mask for valid data points
        method : str
            Optimization method ('bounded' or 'grid_search')

        Returns:
        --------
        result : dict
            Inversion results containing Te_best, rms_best (in meters),
            and predicted moho.
        """
        if method == 'bounded':
            # Bounded optimization
            result = minimize_scalar(
                self.misfit_function,
                bounds=Te_range,
                args=(topography_load, moho_obs, mask),
                method='bounded'
            )

            Te_best = result.x
            rms_best = result.fun

        elif method == 'grid_search':
            # Grid search approach
            Te_values = np.linspace(Te_range[0], Te_range[1], 50)
            rms_values = []

            for Te in Te_values:
                rms = self.misfit_function(Te, topography_load, moho_obs, mask)
                rms_values.append(rms)

            rms_values = np.array(rms_values)
            best_idx = np.argmin(rms_values)
            Te_best = Te_values[best_idx]
            rms_best = rms_values[best_idx]

        else:
            raise ValueError("Method must be 'bounded' or 'grid_search'")

        # Calculate final predicted Moho
        moho_pred = self.predict_moho_flexure(topography_load, Te_best)

        return {
            'Te_best': Te_best,
            'rms_best': rms_best,
            'moho_pred': moho_pred,
            'method': method,
            'Te_range': Te_range
        }

    def plot_results(self, topography, moho_obs, result, figsize=(15, 10)):
        """
        Plot inversion results (Moho-based)

        Parameters:
        -----------
        topography : 2D array
            Topography data (m)
        moho_obs : 2D array
            Observed Moho undulations (m)
        result : dict
            Inversion results from invert_elastic_thickness
        figsize : tuple
            Figure size
        """
        moho_pred = result['moho_pred']
        Te_best = result['Te_best']
        rms_best = result['rms_best']

        fig, axes = plt.subplots(2, 3, figsize=figsize)

        # Topography
        # [CHANGED] cmap=custom_terrain_cmap
        im1 = axes[0, 0].imshow(topography, cmap=custom_terrain_cmap, aspect='equal',
                               origin='lower')
        axes[0, 0].set_title('Topographic Load (m)')
        plt.colorbar(im1, ax=axes[0, 0], label='Elevation (m)')

        # Observed Moho
        im2 = axes[0, 1].imshow(moho_obs, cmap='viridis_r', aspect='equal',
                               origin='lower')
        axes[0, 1].set_title('Observed Moho (m)')
        plt.colorbar(im2, ax=axes[0, 1], label='Depth (m)')

        # Predicted Moho
        im3 = axes[0, 2].imshow(moho_pred, cmap='viridis_r', aspect='equal',
                               origin='lower', vmin=np.min(moho_obs), vmax=np.max(moho_obs))
        axes[0, 2].set_title(f'Predicted Moho (m)\nTe = {Te_best/1000:.1f} km')
        plt.colorbar(im3, ax=axes[0, 2], label='Depth (m)')

        # Residual
        residual = moho_obs - moho_pred
        res_max = np.max(np.abs(residual))
        im4 = axes[1, 0].imshow(residual, cmap='RdBu_r', aspect='equal',
                               origin='lower', vmin=-res_max, vmax=res_max)
        axes[1, 0].set_title(f'Residual (m)\nRMS = {rms_best:.2f} m')
        plt.colorbar(im4, ax=axes[1, 0], label='Residual (m)')

        # Cross-section comparison
        mid_row = topography.shape[0] // 2
        x_km = np.arange(topography.shape[1]) * self.dx / 1000

        axes[1, 1].plot(x_km, moho_obs[mid_row, :], 'b-', label='Observed Moho', linewidth=2)
        axes[1, 1].plot(x_km, moho_pred[mid_row, :], 'r--', label='Predicted Moho', linewidth=2)
        axes[1, 1].set_xlabel('Distance (km)')
        axes[1, 1].set_ylabel('Moho Depth (m)')
        axes[1, 1].set_title('Cross-section Comparison')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].invert_yaxis() # Plot depth increasing downwards

        # Histogram of residuals
        axes[1, 2].hist(residual.flatten(), bins=50, alpha=0.7, edgecolor='black', color='blue')
        axes[1, 2].set_xlabel('Residual (m)')
        axes[1, 2].set_ylabel('Frequency')
        axes[1, 2].set_title('Residual Distribution')
        axes[1, 2].grid(True, alpha=0.3)

        plt.tight_layout()
        return fig

    def sensitivity_analysis(self, topography_load, moho_obs, Te_range=(1000, 50000),
                           n_points=50, mask=None):
        """
        Perform sensitivity analysis by testing different Te values

        Parameters:
        -----------
        topography_load : 2D array
            Topography data (m)
        moho_obs : 2D array
            Observed Moho undulations (m)
        Te_range : tuple
            Range of Te values to test (m)
        n_points : int
            Number of Te values to test
        mask : 2D array, optional
            Mask for valid data points

        Returns:
        --------
        Te_values : array
            Tested Te values (m)
        rms_values : array
            Corresponding RMS misfits (in meters)
        """
        Te_values = np.linspace(Te_range[0], Te_range[1], n_points)
        rms_values = []

        print(f"Testing {n_points} Te values from {Te_range[0]/1000:.1f} to {Te_range[1]/1000:.1f} km...")

        for i, Te in enumerate(Te_values):
            if i % 10 == 0:
                print(f"Progress: {i+1}/{n_points}")

            rms = self.misfit_function(Te, topography_load, moho_obs, mask)
            rms_values.append(rms)

        return np.array(Te_values), np.array(rms_values)

    def plot_sensitivity(self, Te_values, rms_values, figsize=(10, 6)):
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
        ax.plot(Te_km, rms_values, 'b-', linewidth=2)
        ax.set_xlabel('Elastic Thickness (km)')
        ax.set_ylabel('RMS Misfit (m)') # Changed from mGal
        ax.set_title('Sensitivity Analysis: RMS vs Elastic Thickness')
        ax.grid(True, alpha=0.3)

        # Mark minimum
        min_idx = np.argmin(rms_values)
        ax.plot(Te_km[min_idx], rms_values[min_idx], 'ro', markersize=8,
                label=f'Minimum: Te = {Te_km[min_idx]:.1f} km')
        ax.legend()

        plt.tight_layout()
        return fig


    # End of InverseModelling.py


import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
# from elastic_thickness_inversion import ElasticThicknessInversion
import time
from matplotlib.colors import ListedColormap
import os
from datetime import datetime

# [NEW] Create custom colormap
# Get the original 'terrain' colormap
terrain_cmap = plt.get_cmap('terrain')
# Get all 256 colors from it

terrain_colors = terrain_cmap(np.linspace(0, 1, 256))
# 'terrain' uses blue/cyan for the first 25% (64 entries)
# We set these to white (R=1, G=1, B=1, Alpha=1)

# Create a new colormap from our modified list of colors
custom_terrain_cmap = ListedColormap(terrain_colors)


class MohoAnalysis:
    """
    Class for Moho analysis with moving window technique
    """

    def __init__(self):
        self.G = 6.67430e-11  # Gravitational constant
        self.rho_c = 2670     # Crustal density (kg/m³)
        self.rho_m = 3300     # Mantle density (kg/m³)
        self.g = 9.81         # Gravitational acceleration

    def get_user_parameters(self):
        """
        Get comprehensive user input for the analysis
        """
        print("="*80)
        print("INTERACTIVE MOHO ANALYSIS WITH MOVING WINDOW TECHNIQUE")
        print("="*80)

        params = {}

        # Domain parameters
        print("\n DOMAIN/AREA:")
        try:
            domain_size = float(input("Total domain size (km, default 2000): ") or "2000") * 1000  # Convert to meters
            grid_spacing = float(input("Grid spacing (km, default 5): ") or "5") * 1000  # Convert to meters
        except ValueError:
            domain_size, grid_spacing = 2000000, 5000

        params['domain_size'] = domain_size
        params['grid_spacing'] = grid_spacing
        params['nx'] = params['ny'] = int(domain_size / grid_spacing)

        print(f"Domain: {domain_size/1000:.0f} x {domain_size/1000:.0f} km")
        print(f"Grid: {params['nx']} x {params['ny']} points")
        # print(f"Resolution: {grid_spacing/1000:.1f} km")

        # Mountain parameters
        print("\n MOUNTAIN :")

        # Large mountain
        print("Large Mountain:")
        try:
            large_height = float(input("  Height (km, default 5): ") or "5") * 1000
            large_x = float(input("  X position (km from left, default 600): ") or "600") * 1000
            large_y = float(input("  Y position (km from bottom, default 800): ") or "800") * 1000
            large_width = float(input("  Width (km, default 150): ") or "150") * 1000
            large_length = float(input("  Length (km, default 200): ") or "200") * 1000
        except ValueError:
            large_height, large_x, large_y, large_width, large_length = 5000, 600000, 800000, 150000, 200000

        # Small mountain
        print("Small Mountain:")
        try:
            small_height = float(input("  Height (km, default 2): ") or "2") * 1000
            small_x = float(input("  X position (km from left, default 1200): ") or "1200") * 1000
            small_y = float(input("  Y position (km from bottom, default 1200): ") or "1200") * 1000
            small_width = float(input("  Width (km, default 80): ") or "80") * 1000
            small_length = float(input("  Length (km, default 100): ") or "100") * 1000
        except ValueError:
            small_height, small_x, small_y, small_width, small_length = 2000, 1200000, 1200000, 80000, 100000

        params['mountains'] = {
            'large': {'x': large_x, 'y': large_y, 'width': large_width, 'length': large_length, 'height': large_height},
            'small': {'x': small_x, 'y': small_y, 'width': small_width, 'length': small_length, 'height': small_height}
        }

        # Moho parameters
        print("\n MOHO CONFIGURATION:")
        print("Choose Moho type:")
        print("1. Flat Moho")
        print("2. Variable Moho (deflections)")

        try:
            moho_type = int(input("Enter choice (default 2): ") or "2")
            if moho_type == 1:
                flat_depth = float(input("Flat Moho depth (km, default 35): ") or "35") * 1000
                params['moho'] = {'type': 'flat', 'depth': flat_depth}
            else:
                print("Variable Moho:")
                background_depth = float(input("  Background Moho depth (km, default 30): ") or "30") * 1000
                large_deflection = float(input("  Moho deflection under large mountain (km, default 20): ") or "20") * 1000
                small_deflection = float(input("  Moho deflection under small mountain (km, default 5): ") or "5") * 1000

                # Calculate actual depths
                large_depth = background_depth + large_deflection
                small_depth = background_depth + small_deflection

                params['moho'] = {
                    'type': 'variable',
                    'background': background_depth,
                    'large_depth': large_depth,
                    'small_depth': small_depth,
                    'large_deflection': large_deflection,
                    'small_deflection': small_deflection
                }

                print(f"  → Background: {background_depth/1000:.0f} km")
                print(f"  → Under large mountain: {large_depth/1000:.0f} km ({large_deflection/1000:.0f} km deflection)")
                print(f"  → Under small mountain: {small_depth/1000:.0f} km ({small_deflection/1000:.0f} km deflection)")

        except ValueError:
            # Default values
            params['moho'] = {
                'type': 'variable',
                'background': 30000,
                'large_depth': 50000,
                'small_depth': 35000,
                'large_deflection': 20000,
                'small_deflection': 5000
            }

        # Moving window parameters
        print("\n MOVING WINDOW:")
        try:
            window_size = float(input("Window size (km, default 200): ") or "200") * 1000
            shift_min = float(input("Minimum shift distance (km, default 20): ") or "20") * 1000
            shift_max = float(input("Maximum shift distance (km, default 50): ") or "50") * 1000
            shift_step = float(input("Shift step size (km, default 10): ") or "10") * 1000
        except ValueError:
            window_size, shift_min, shift_max, shift_step = 200000, 20000, 50000, 10000

        params['window'] = {
            'size': window_size,
            'shift_min': shift_min,
            'shift_max': shift_max,
            'shift_step': shift_step
        }

        # Analysis parameters
        print("\n Te & GRAVITY NOISE LEVEL:")
        try:
            Te_min = float(input("Minimum Te search (km, default 5): ") or "5") * 1000
            Te_max = float(input("Maximum Te search (km, default 80): ") or "80") * 1000
            noise_level = float(input("Gravity noise level (mGal, default 1.0): ") or "1.0")
        except ValueError:
            Te_min, Te_max, noise_level = 5000, 80000, 1.0

        params['analysis'] = {
            'Te_range': (Te_min, Te_max),
            'noise_level': noise_level
        }

        return params

    def create_synthetic_topography(self, params):
        """
        Create synthetic topography with specified mountains
        """
        print("\nCreating synthetic topography...")

        nx, ny = params['nx'], params['ny']
        dx = dy = params['grid_spacing']

        # Create coordinate grids
        x = np.arange(nx) * dx
        y = np.arange(ny) * dy
        X, Y = np.meshgrid(x, y)

        # Initialize topography
        topography = np.zeros((ny, nx))

        # Add large mountain
        large = params['mountains']['large']
        large_dist = np.sqrt(((X - large['x']) / large['length'])**2 +
                           ((Y - large['y']) / large['width'])**2)
        large_mountain = large['height'] * np.exp(-large_dist**2)
        topography += large_mountain

        print(f"Large mountain: {large['height']/1000:.0f}km at ({large['x']/1000:.0f}, {large['y']/1000:.0f}) km")
        #print(f"  Dimensions: {large['length']/1000:.0f} x {large['width']/1000:.0f} km")

        # Add small mountain
        small = params['mountains']['small']
        small_dist = np.sqrt(((X - small['x']) / (small['length']))**2 +
                           ((Y - small['y']) / (small['width']))**2)
        small_mountain = small['height'] * np.exp(-small_dist**2)
        topography += small_mountain

        print(f"Small mountain: {small['height']/1000:.0f}km at ({small['x']/1000:.0f}, {small['y']/1000:.0f}) km")
        #print(f"  Dimensions: {small['length']/1000:.0f} x {small['width']/1000:.0f} km")

        # Add some background topography variation
        background = 200 * np.sin(2*np.pi*X/500000) * np.cos(2*np.pi*Y/300000)
        topography += background

        # Add noise
        noise = 50 * np.random.randn(ny, nx)
        topography += noise

        # Ensure no negative elevations
        topography = np.maximum(topography, 0)

        #print(f"Final topography range: {topography.min():.0f} to {topography.max():.0f} m")

        return X, Y, topography

    def create_moho_depth(self, params, X, Y):
        """
        Create Moho depth variations
        """
        print(f"\nCreating Moho depth ({params['moho']['type']})...")

        ny, nx = X.shape

        if params['moho']['type'] == 'flat':
            moho_depth = np.full((ny, nx), params['moho']['depth'], dtype=float)
            print(f"Flat Moho at {params['moho']['depth']/1000:.0f} km depth")

        else:  # Variable Moho
            # Start with background depth
            moho_depth = np.full((ny, nx), params['moho']['background'], dtype=float)

            # Add deflection under large mountain
            large = params['mountains']['large']
            large_dist = np.sqrt(((X - large['x']) / (large['length']*1.5))**2 +
                               ((Y - large['y']) / (large['width']*1.5))**2)
            large_deflection = (params['moho']['large_depth'] - params['moho']['background']) * np.exp(-large_dist**2)
            moho_depth += large_deflection

            # Add deflection under small mountain
            small = params['mountains']['small']
            small_dist = np.sqrt(((X - small['x']) / (small['length']*1.5))**2 +
                               ((Y - small['y']) / (small['width']*1.5))**2)
            small_deflection = (params['moho']['small_depth'] - params['moho']['background']) * np.exp(-small_dist**2)
            moho_depth += small_deflection

            print(f"Variable Moho: {params['moho']['background']/1000:.0f} km background")
            print(f"  {params['moho']['large_depth']/1000:.0f} km under large mountain")
            print(f"  {params['moho']['small_depth']/1000:.0f} km under small mountain")

        return moho_depth

    def calculate_gravity_from_moho(self, topography, moho_depth, params):
        """
        Calculate gravity anomaly from topography and Moho variations
        (This is for plotting/comparison purposes; not used in inversion)
        """
        print("\nCalculating gravity from topography and Moho...")

        dx = dy = params['grid_spacing']

        # This is a basic Bouguer correction
        topo_gravity = 2 * np.pi * self.G * self.rho_c * topography * 1e5  # Convert to mGal

        # Calculate Moho effect
        # Moho variations create density contrasts
        reference_moho = np.mean(moho_depth)
        moho_anomaly = moho_depth - reference_moho

        # Simple approximation for Moho gravity effect
        moho_gravity = -2 * np.pi * self.G * (self.rho_m - self.rho_c) * moho_anomaly * 1e5  # mGal

        # Total gravity
        total_gravity = topo_gravity + moho_gravity

        # Add noise
        if params['analysis']['noise_level'] > 0:
            noise = params['analysis']['noise_level'] * np.random.randn(*total_gravity.shape)
            total_gravity += noise

        print(f"Gravity range: {total_gravity.min():.1f} to {total_gravity.max():.1f} mGal")

        return total_gravity

    def moving_window_analysis(self, topography, moho_depth, params, X, Y):
        """
        Perform moving window analysis with variable shift distances
        [CHANGED] Takes moho_depth as input, not gravity
        """
        print("\nPerforming moving window analysis...")

        window_size = params['window']['size']
        shift_distances = np.arange(params['window']['shift_min'],
                                  params['window']['shift_max'] + params['window']['shift_step'],
                                  params['window']['shift_step'])

        dx = dy = params['grid_spacing']
        domain_size = params['domain_size']

        results = {}

        for shift_dist in shift_distances:
            print(f"\nAnalyzing with shift distance: {shift_dist/1000:.0f} km")

            # Calculate window positions
            window_pixels = int(window_size / dx)
            shift_pixels = int(shift_dist / dx)

            # Starting from top-left corner
            x_positions = np.arange(0, params['nx'] - window_pixels, shift_pixels)
            y_positions = np.arange(0, params['ny'] - window_pixels, shift_pixels)

            n_windows = len(x_positions) * len(y_positions)
            print(f"  Number of windows: {n_windows} ({len(x_positions)} x {len(y_positions)})")

            # Initialize result arrays
            Te_map = np.full((len(y_positions), len(x_positions)), np.nan)
            rms_map = np.full((len(y_positions), len(x_positions)), np.nan)
            x_centers = np.zeros(len(x_positions))
            y_centers = np.zeros(len(y_positions))

            # Initialize inverter
            # Assumes ElasticThicknessInversion class is defined (from Cell 1)
            inverter = ElasticThicknessInversion(dx=dx, dy=dy)

            # Process each window
            window_count = 0
            start_time = time.time()

            for i, y_start in enumerate(y_positions):
                for j, x_start in enumerate(x_positions):
                    window_count += 1

                    # Extract window data
                    y_end = y_start + window_pixels
                    x_end = x_start + window_pixels

                    topo_window = topography[y_start:y_end, x_start:x_end]
                    # [CHANGED] Use moho_depth grid
                    moho_window = moho_depth[y_start:y_end, x_start:x_end]

                    # [CHANGED] Check if window has sufficient data variation
                    # Using arbitrary but reasonable thresholds for std dev in meters
                    if (np.std(topo_window) > 100 and np.std(moho_window) > 100):
                        try:
                            # [CHANGED] Perform inversion using Moho
                            result = inverter.invert_elastic_thickness(
                                topo_window, moho_window, # Pass moho_window
                                Te_range=params['analysis']['Te_range'],
                                method='bounded'
                            )

                            Te_map[i, j] = result['Te_best']
                            rms_map[i, j] = result['rms_best'] # This is now in meters

                        except Exception as e:
                            print(f"    Warning: Window {window_count} failed: {e}")
                            continue

                    # Store window center coordinates
                    x_centers[j] = X[0, x_start + window_pixels//2]
                    y_centers[i] = Y[y_start + window_pixels//2, 0]

                    # Progress update
                    if window_count % max(1, n_windows//10) == 0:
                        elapsed = time.time() - start_time
                        progress = window_count / n_windows * 100
                        print(f"    Progress: {progress:.1f}% ({window_count}/{n_windows}) - {elapsed:.1f}s")

            # Store results for this shift distance
            results[shift_dist] = {
                'Te_map': Te_map,
                'rms_map': rms_map,
                'x_centers': x_centers,
                'y_centers': y_centers,
                'n_windows': n_windows,
                'window_size': window_size,
                'shift_distance': shift_dist
            }

            # Calculate statistics
            valid_Te = Te_map[~np.isnan(Te_map)]
            if len(valid_Te) > 0:
                print(f"  Te results: {len(valid_Te)} valid windows")
                print(f"    Range: {valid_Te.min()/1000:.1f} - {valid_Te.max()/1000:.1f} km")
                print(f"    Mean: {valid_Te.mean()/1000:.1f} ± {valid_Te.std()/1000:.1f} km")

        return results

    def analyze_distortions(self, results, params):
        """
        Analyze distortions and variations in results
        """
        print("\nAnalyzing spatial distortions and variations...")

        distortion_analysis = {}

        for shift_dist, result in results.items():
            Te_map = result['Te_map']
            valid_mask = ~np.isnan(Te_map)

            if np.sum(valid_mask) < 4:
                continue

            # Calculate spatial gradients
            Te_valid = Te_map.copy()
            Te_valid[~valid_mask] = np.nanmean(Te_map)  # Fill NaN for gradient calculation

            grad_y, grad_x = np.gradient(Te_valid)
            gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)

            # Calculate statistics
            stats = {
                'shift_distance': shift_dist,
                'n_valid_windows': np.sum(valid_mask),
                'Te_mean': np.nanmean(Te_map),
                'Te_std': np.nanstd(Te_map),
                'Te_range': np.nanmax(Te_map) - np.nanmin(Te_map),
                'gradient_mean': np.nanmean(gradient_magnitude[valid_mask]),
                'gradient_max': np.nanmax(gradient_magnitude[valid_mask]),
                'coverage': np.sum(valid_mask) / Te_map.size * 100
            }

            distortion_analysis[shift_dist] = stats

            print(f"Shift {shift_dist/1000:.0f} km:")
            print(f"  Valid windows: {stats['n_valid_windows']} ({stats['coverage']:.1f}% coverage)")
            print(f"  Te: {stats['Te_mean']/1000:.1f} ± {stats['Te_std']/1000:.1f} km")
            print(f"  Range: {stats['Te_range']/1000:.1f} km")
            print(f"  Gradient: {stats['gradient_mean']:.1f} (max: {stats['gradient_max']:.1f})")

        return distortion_analysis

    def plot_comprehensive_results(self, topography, gravity, moho_depth, results,
                                 distortion_analysis, params, X, Y):
        """
        Create comprehensive plots of all results
        """
        print("\nGenerating comprehensive plots...")

        # Figure 1: Input data
        fig1, axes = plt.subplots(2, 2, figsize=(16, 12))

        # Topography
        im1 = axes[0,0].imshow(topography, extent=[0, params['domain_size']/1000,
                                                 0, params['domain_size']/1000],
                              cmap='viridis', origin='lower') # Changed to viridis
        axes[0,0].set_title('Synthetic Topography (m)')
        axes[0,0].set_xlabel('X (km)')
        axes[0,0].set_ylabel('Y (km)')
        plt.colorbar(im1, ax=axes[0,0])

        # Add mountain markers
        large = params['mountains']['large']
        small = params['mountains']['small']
        axes[0,0].plot(large['x']/1000, large['y']/1000, 'r*', markersize=15,
                      label=f'Large Mt ({large["height"]}m)')
        axes[0,0].plot(small['x']/1000, small['y']/1000, 'k*', markersize=12,
                      label=f'Small Mt ({small["height"]}m)')
        axes[0,0].legend()

        # Gravity
        im2 = axes[0,1].imshow(gravity, extent=[0, params['domain_size']/1000,
                                              0, params['domain_size']/1000],
                              cmap='viridis', origin='lower') # Changed to viridis
        axes[0,1].set_title('Gravity Anomaly (mGal) - For Reference')
        axes[0,1].set_xlabel('X (km)')
        axes[0,1].set_ylabel('Y (km)')
        plt.colorbar(im2, ax=axes[0,1])

        # Moho depth
        im3 = axes[1,0].imshow(moho_depth/1000, extent=[0, params['domain_size']/1000,
                                                       0, params['domain_size']/1000],
                              cmap='viridis', origin='lower') # Changed to viridis
        axes[1,0].set_title('Observed Moho Depth (km)')
        axes[1,0].set_xlabel('X (km)')
        axes[1,0].set_ylabel('Y (km)')
        plt.colorbar(im3, ax=axes[1,0])

        # Cross-sections
        mid_row = topography.shape[0] // 2
        x_km = np.arange(topography.shape[1]) * params['grid_spacing'] / 1000

        axes[1,1].plot(x_km, topography[mid_row, :], 'g-', linewidth=2, label='Topography')
        ax2 = axes[1,1].twinx()
        ax2.plot(x_km, gravity[mid_row, :], 'r-', linewidth=2, label='Gravity (Ref)')
        ax3 = axes[1,1].twinx()
        ax3.spines['right'].set_position(('outward', 60))
        ax3.plot(x_km, moho_depth[mid_row, :]/1000, 'b-', linewidth=2, label='Moho')

        axes[1,1].set_xlabel('X (km)')
        axes[1,1].set_ylabel('Elevation (m)', color='g')
        ax2.set_ylabel('Gravity (mGal)', color='r')
        ax3.set_ylabel('Moho Depth (km)', color='b')
        axes[1,1].set_title('Cross-section at Y = 1000 km')
        # Add combined legend
        lines1, labels1 = axes[1,1].get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        lines3, labels3 = ax3.get_legend_handles_labels()
        ax3.legend(lines1 + lines2 + lines3, labels1 + labels2 + labels3, loc='upper center')


        plt.tight_layout()

        # Figure 2: Moving window results for different shift distances
        n_shifts = len(results)
        fig2, axes = plt.subplots(2, min(3, n_shifts), figsize=(5*min(3, n_shifts), 10), squeeze=False)

        shift_keys = list(results.keys())[:3]  # Show first 3 shift distances

        # Create a viridis colormap and set 'bad' values to the lowest color
        viridis_with_bad = plt.cm.get_cmap('viridis').copy()
        viridis_with_bad.set_bad(viridis_with_bad(0.0))

        for i, shift_dist in enumerate(shift_keys):
            result = results[shift_dist]
            Te_map = result['Te_map']
            rms_map = result['rms_map']

            # Create coordinate grids for plotting
            x_centers_km = result['x_centers'] / 1000
            y_centers_km = result['y_centers'] / 1000

            extent_te = [x_centers_km.min(), x_centers_km.max(),
                         y_centers_km.min(), y_centers_km.max()]

            # Te map
            if i < axes.shape[1]:
                im1 = axes[0,i].imshow(Te_map/1000, extent=extent_te,
                                      cmap=viridis_with_bad, origin='lower', aspect='equal')
                axes[0,i].set_title(f'Te Map - Shift {shift_dist/1000:.0f} km')
                axes[0,i].set_xlabel('X (km)')
                axes[0,i].set_ylabel('Y (km)')
                plt.colorbar(im1, ax=axes[0,i], label='Te (km)')

                # RMS map
                im2 = axes[1,i].imshow(rms_map, extent=extent_te,
                                      cmap=viridis_with_bad, origin='lower', aspect='equal')
                axes[1,i].set_title(f'RMS Map - Shift {shift_dist/1000:.0f} km')
                axes[1,i].set_xlabel('X (km)')
                axes[1,i].set_ylabel('Y (km)')
                plt.colorbar(im2, ax=axes[1,i], label='RMS (m of Moho)')

        plt.tight_layout()


        """
        # Figure 3: Distortion analysis
        fig3, axes = plt.subplots(2, 2, figsize=(14, 10))

        shift_distances = [d/1000 for d in distortion_analysis.keys()]

        # Te statistics vs shift distance
        Te_means = [stats['Te_mean']/1000 for stats in distortion_analysis.values()]
        Te_stds = [stats['Te_std']/1000 for stats in distortion_analysis.values()]

        axes[0,0].errorbar(shift_distances, Te_means, yerr=Te_stds, fmt='bo-', capsize=5)
        axes[0,0].set_xlabel('Shift Distance (km)')
        axes[0,0].set_ylabel('Mean Te (km)')
        axes[0,0].set_title('Te Statistics vs Shift Distance')
        axes[0,0].grid(True, alpha=0.3)

        # Coverage vs shift distance
        coverage = [stats['coverage'] for stats in distortion_analysis.values()]
        n_windows = [stats['n_valid_windows'] for stats in distortion_analysis.values()]

        axes[0,1].plot(shift_distances, coverage, 'go-', label='Coverage (%)')
        ax2 = axes[0,1].twinx()
        ax2.plot(shift_distances, n_windows, 'ro-', label='N Windows')
        axes[0,1].set_xlabel('Shift Distance (km)')
        axes[0,1].set_ylabel('Coverage (%)', color='g')
        ax2.set_ylabel('Number of Windows', color='r')
        axes[0,1].set_title('Analysis Coverage vs Shift Distance')
        axes[0,1].grid(True, alpha=0.3)
        # Add combined legend
        lines1, labels1 = axes[0,1].get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

        # Gradient analysis
        grad_means = [stats['gradient_mean'] for stats in distortion_analysis.values()]
        grad_maxs = [stats['gradient_max'] for stats in distortion_analysis.values()]

        axes[1,0].plot(shift_distances, grad_means, 'mo-', label='Mean Gradient')
        axes[1,0].plot(shift_distances, grad_maxs, 'co-', label='Max Gradient')
        axes[1,0].set_xlabel('Shift Distance (km)')
        axes[1,0].set_ylabel('Gradient Magnitude')
        axes[1,0].set_title('Spatial Gradients vs Shift Distance')
        axes[1,0].legend()
        axes[1,0].grid(True, alpha=0.3)

        # Te range vs shift distance
        Te_ranges = [stats['Te_range']/1000 for stats in distortion_analysis.values()]

        axes[1,1].plot(shift_distances, Te_ranges, 'ko-')
        axes[1,1].set_xlabel('Shift Distance (km)')
        axes[1,1].set_ylabel('Te Range (km)')
        axes[1,1].set_title('Te Variability vs Shift Distance')
        axes[1,1].grid(True, alpha=0.3)

        plt.tight_layout()
        """

        return fig1, fig2 #, fig3

    def generate_summary_report(self, params, results, distortion_analysis):
        """
        Generate comprehensive summary report
        """
        print("\n" + "="*80)
        print("COMPREHENSIVE ANALYSIS SUMMARY REPORT")
        print("="*80)

        print(f"\nDOMAIN CONFIGURATION:")
        print(f"  Size: {params['domain_size']/1000:.0f} x {params['domain_size']/1000:.0f} km")
        print(f"  Resolution: {params['grid_spacing']/1000:.1f} km ({params['nx']} x {params['ny']} points)")

        print(f"\nMOUNTAIN CONFIGURATION:")
        large = params['mountains']['large']
        small = params['mountains']['small']
        print(f"  Large Mountain: {large['height']}m at ({large['x']/1000:.0f}, {large['y']/1000:.0f}) km")
        print(f"    Dimensions: {large['length']/1000:.0f} x {large['width']/1000:.0f} km")
        print(f"  Small Mountain: {small['height']}m at ({small['x']/1000:.0f}, {small['y']/1000:.0f}) km")
        print(f"    Dimensions: {small['length']/1000:.0f} x {small['width']/1000:.0f} km")

        print(f"\nMOHO CONFIGURATION:")
        if params['moho']['type'] == 'flat':
            print(f"  Flat Moho at {params['moho']['depth']/1000:.0f} km depth")
        else:
            print(f"  Variable Moho:")
            print(f"    Background: {params['moho']['background']/1000:.0f} km")
            print(f"    Under large mountain: {params['moho']['large_depth']/1000:.0f} km")
            print(f"    Under small mountain: {params['moho']['small_depth']/1000:.0f} km")

        print(f"\nMOVING WINDOW ANALYSIS:")
        print(f"  Window size: {params['window']['size']/1000:.0f} x {params['window']['size']/1000:.0f} km")
        print(f"  Shift distances: {params['window']['shift_min']/1000:.0f} - {params['window']['shift_max']/1000:.0f} km")
        print(f"  Number of shift configurations: {len(results)}")

        print(f"\nRESULTS SUMMARY:")
        for shift_dist, stats in distortion_analysis.items():
            print(f"  Shift {shift_dist/1000:.0f} km:")
            print(f"    Valid windows: {stats['n_valid_windows']} ({stats['coverage']:.1f}% coverage)")
            print(f"    Te: {stats['Te_mean']/1000:.1f} ± {stats['Te_std']/1000:.1f} km")
            print(f"    Range: {stats['Te_range']/1000:.1f} km")
            print(f"    Spatial gradient: {stats['gradient_mean']:.2f} (max: {stats['gradient_max']:.2f})")

        print(f"\nDISTORTION ANALYSIS:")
        shift_distances = list(distortion_analysis.keys())
        if len(shift_distances) > 1:
            # Compare first and last shift distances
            first_stats = distortion_analysis[shift_distances[0]]
            last_stats = distortion_analysis[shift_distances[-1]]

            coverage_change = last_stats['coverage'] - first_stats['coverage']
            Te_change = (last_stats['Te_mean'] - first_stats['Te_mean']) / 1000
            gradient_change = last_stats['gradient_mean'] - first_stats['gradient_mean']

            print(f"  Coverage change: {coverage_change:+.1f}%")
            print(f"  Mean Te change: {Te_change:+.1f} km")
            print(f"  Gradient change: {gradient_change:+.2f}")

            if abs(coverage_change) > 10:
                print(f"  → Significant coverage variation detected")
            if abs(Te_change) > 2:
                print(f"  → Significant Te bias with shift distance")
            if abs(gradient_change) > 0.5:
                print(f"  → Significant gradient variation detected")

        print(f"\nRECOMMENDATIONS:")
        # Find optimal shift distance
        best_shift = min(distortion_analysis.keys(),
                        key=lambda x: distortion_analysis[x]['Te_std'])
        best_stats = distortion_analysis[best_shift]

        print(f"  Optimal shift distance: {best_shift/1000:.0f} km")
        print(f"    Provides lowest Te variability ({best_stats['Te_std']/1000:.1f} km)")
        print(f"    Coverage: {best_stats['coverage']:.1f}%")
        print(f"    Valid windows: {best_stats['n_valid_windows']}")

def create_output_folder():
    """
    Creates a unique output folder named 'Output_YYYYMMDD_HHMMSS'.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    folder_name = f"Output_{timestamp}"
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    print(f"\nCreated output folder: {folder_name}")
    return folder_name

def main():
    """
    Main function to run the interactive Moho analysis
    """
    analyzer = MohoAnalysis()

    # Get user parameters
    params = analyzer.get_user_parameters()

    print(f"\n{'-'*80}")
   # print("STARTING ANALYSIS")
    #print(f"{'='*80}")

    # Create synthetic data
    X, Y, topography = analyzer.create_synthetic_topography(params)
    moho_depth = analyzer.create_moho_depth(params, X, Y)
    # [CHANGED] Gravity is now just for reference plotting
    gravity = analyzer.calculate_gravity_from_moho(topography, moho_depth, params)

    # [CHANGED] Perform moving window analysis using moho_depth
    results = analyzer.moving_window_analysis(topography, moho_depth, params, X, Y)

    # Analyze distortions
    distortion_analysis = analyzer.analyze_distortions(results, params)

    # Generate plots
    fig1, fig2 = analyzer.plot_comprehensive_results(
        topography, gravity, moho_depth, results, distortion_analysis, params, X, Y)
    """
fig1, fig2, fig3 = analyzer.plot_comprehensive_results(
        topography, gravity, moho_depth, results, distortion_analysis, params, X, Y)
        """
    # Generate summary report
    analyzer.generate_summary_report(params, results, distortion_analysis)

    # Show plots
    plt.show()

    # Create output folder
    output_folder = create_output_folder()

    # Ask to save results and figures
    save_data = input(f"\nSave numerical results to '{output_folder}/moho_analysis_results.npz'? (y/n): ").lower()
    if save_data == 'y':
        # Save data
        np.savez(os.path.join(output_folder, 'moho_analysis_results.npz'),
                topography=topography,
                gravity=gravity,
                moho_depth=moho_depth,
                X=X, Y=Y,
                **{f'Te_map_shift_{int(k/1000)}km': v['Te_map'] for k, v in results.items()})

        print(f"Numerical results saved to '{output_folder}/moho_analysis_results.npz")

    save_figures = input(f"Save figures as PNG files to '{output_folder}/'? (y/n): ").lower()
    if save_figures == 'y':
        try:
            fig1.savefig(os.path.join(output_folder, 'moho_analysis_inputs.png'), dpi=300)
            print(f"Figure 1 (Input Data) saved as '{output_folder}/moho_analysis_inputs.png'")
        except Exception as e:
            print(f"Error saving Figure 1: {e}")
        try:
            fig2.savefig(os.path.join(output_folder, 'moho_analysis_Te_maps.png'), dpi=300)
            print(f"Figure 2 (Te Maps) saved as '{output_folder}/moho_analysis_Te_maps.png'")
        except Exception as e:
            print(f"Error saving Figure 2: {e}")



    return {
        'params': params,
        'topography': topography,
        'gravity': gravity,
        'moho_depth': moho_depth,
        'results': results,
        'distortion_analysis': distortion_analysis,
        'coordinates': (X, Y)
    }

if __name__ == "__main__":
    # This assumes the ElasticThicknessInversion class from Cell 1
    # is available in the same scope (e.g., run in the same notebook/script)
    analysis_results = main()