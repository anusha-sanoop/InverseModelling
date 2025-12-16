"""
Elastic Thickness Inversion using Braitenberg Convolution Method

Based on:
Braitenberg, C., Ebbing, J., & Götze, H. J. (2002). 
Inverse modelling of elastic thickness by convolution method—the eastern Alps as a case example.
Earth and Planetary Science Letters, 202(2), 387-404.

This module implements the convolution method for estimating effective elastic thickness (Te)
of the lithosphere from topography and Moho depth data.
"""

import numpy as np
from scipy import fft
from scipy.optimize import minimize_scalar
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


class ElasticThicknessInversion:
    """
    Class for inverse modeling of elastic thickness using convolution method
    Based on Braitenberg et al. (2002)
    """

    def __init__(self, dx=1000, dy=1000, rho_load=2900, rho_m=3500, 
                 rho_infill=2900, E=1.0e11, nu=0.25, g=3.72):
        """
        Initialize the inversion class

        Parameters:
        -----------
        dx, dy : float
            Grid spacing in meters (default: 1000m)
        rho_load : float
            Load density (kg/m³) - default 2900 (crustal density for Mars)
        rho_m : float
            Mantle density (kg/m³) - default 3500 (Mars mantle)
        rho_infill : float
            Infill/crustal density (kg/m³) - default 2900 (Mars crust)
        E : float
            Young's modulus (Pa) - default 1.0e11
        nu : float
            Poisson's ratio - default 0.25
        g : float
            Gravitational acceleration (m/s²) - default 3.72 (Mars)
        """
        self.dx = dx
        self.dy = dy
        self.rho_load = rho_load
        self.rho_m = rho_m
        self.rho_infill = rho_infill
        self.E = E
        self.nu = nu
        self.g = g
        
        # Flexural rigidity factor (D = D_factor * Te**3)
        self.D_factor = self.E / (12 * (1 - self.nu**2))

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
        if Te < 1e-3:  # Treat Te=0 as pure Airy isostasy
            k = np.zeros_like(k)

        D = self.D_factor * Te**3  # Flexural rigidity

        # Denominator of Eq. 3
        # (rho_m - rho_infill) is the buoyancy term
        # (D/g) * k**4 is the flexural term
        denominator = (self.rho_m - self.rho_infill) + (D / self.g) * k**4

        # Avoid division by zero if (rho_m == rho_infill) and k=0
        denominator = np.maximum(denominator, 1e-10)

        # Flexure filter F(k)
        F_k = self.rho_load / denominator

        return F_k

    def predict_moho_flexure(self, topography_load, Te):
        """
        Forward model: calculate predicted Moho flexure from load and Te
        This follows the convolution method

        Parameters:
        -----------
        topography_load : 2D array
            Topographic load (m)
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

    def invert_elastic_thickness(self, topography_load, moho_obs, 
                                Te_range=(1000, 50000), mask=None, method='bounded'):
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

    def sensitivity_analysis(self, topography_load, moho_obs, 
                             Te_range=(1000, 50000), n_points=50, mask=None):
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

