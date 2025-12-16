"""
Moving Window Analysis for Elastic Thickness Estimation

This module implements the moving window technique to estimate spatial
variations in elastic thickness across the study area.
"""

import numpy as np
import time
from elastic_thickness_inversion import ElasticThicknessInversion


class MovingWindowAnalysis:
    """
    Class for performing moving window analysis of elastic thickness
    """
    
    def __init__(self, dx, dy):
        """
        Initialize moving window analysis
        
        Parameters:
        -----------
        dx, dy : float
            Grid spacing in meters
        """
        self.dx = dx
        self.dy = dy
    
    def analyze(self, topography, moho_depth, window_size=1000000, 
                shift_distance=20000, Te_range=(5000, 80000), 
                min_std_topo=100, min_std_moho=100):
        """
        Perform moving window analysis
        
        Parameters:
        -----------
        topography : 2D array
            Topography data (m)
        moho_depth : 2D array
            Moho depth data (m)
        window_size : float
            Size of moving window in meters (default: 1000 km)
        shift_distance : float
            Distance to shift window in meters (default: 20 km)
        Te_range : tuple
            Range of Te values to search (m)
        min_std_topo : float
            Minimum standard deviation in topography for valid window
        min_std_moho : float
            Minimum standard deviation in Moho for valid window
            
        Returns:
        --------
        results : dict
            Dictionary containing Te_map, rms_map, and coordinates
        """
        ny, nx = topography.shape
        
        # Calculate window positions
        window_pixels = int(window_size / self.dx)
        shift_pixels = int(shift_distance / self.dx)
        
        # Starting positions
        x_positions = np.arange(0, nx - window_pixels, shift_pixels)
        y_positions = np.arange(0, ny - window_pixels, shift_pixels)
        
        n_windows = len(x_positions) * len(y_positions)
        print(f"\nMoving Window Analysis:")
        print(f"  Window size: {window_size/1000:.0f} km ({window_pixels} pixels)")
        print(f"  Shift distance: {shift_distance/1000:.0f} km ({shift_pixels} pixels)")
        print(f"  Number of windows: {n_windows} ({len(x_positions)} x {len(y_positions)})")
        
        # Initialize result arrays
        Te_map = np.full((len(y_positions), len(x_positions)), np.nan)
        rms_map = np.full((len(y_positions), len(x_positions)), np.nan)
        x_centers = np.zeros(len(x_positions))
        y_centers = np.zeros(len(y_positions))
        
        # Initialize inverter with Mars parameters
        inverter = ElasticThicknessInversion(dx=self.dx, dy=self.dy,
                                            rho_load=2900, rho_m=3500, 
                                            rho_infill=2900, g=3.72)
        
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
                moho_window = moho_depth[y_start:y_end, x_start:x_end]
                
                # Check if window has sufficient data variation
                if (np.std(topo_window) > min_std_topo and 
                    np.std(moho_window) > min_std_moho):
                    try:
                        # Perform inversion
                        result = inverter.invert_elastic_thickness(
                            topo_window, moho_window,
                            Te_range=Te_range,
                            method='bounded'
                        )
                        
                        Te_map[i, j] = result['Te_best']
                        rms_map[i, j] = result['rms_best']
                        
                    except Exception as e:
                        if window_count % 100 == 0:  # Only print occasionally
                            print(f"    Warning: Window {window_count} failed: {e}")
                        continue
                
                # Store window center coordinates
                x_centers[j] = x_start + window_pixels//2
                y_centers[i] = y_start + window_pixels//2
                
                # Progress update
                if window_count % max(1, n_windows//10) == 0:
                    elapsed = time.time() - start_time
                    progress = window_count / n_windows * 100
                    print(f"    Progress: {progress:.1f}% ({window_count}/{n_windows}) - {elapsed:.1f}s")
        
        # Calculate statistics
        valid_Te = Te_map[~np.isnan(Te_map)]
        if len(valid_Te) > 0:
            print(f"\n  Results: {len(valid_Te)} valid windows")
            print(f"    Te range: {valid_Te.min()/1000:.1f} - {valid_Te.max()/1000:.1f} km")
            print(f"    Te mean: {valid_Te.mean()/1000:.1f} ± {valid_Te.std()/1000:.1f} km")
        else:
            print(f"\n  Warning: No valid windows found!")
        
        return {
            'Te_map': Te_map,
            'rms_map': rms_map,
            'x_centers': x_centers,
            'y_centers': y_centers,
            'n_windows': n_windows,
            'window_size': window_size,
            'shift_distance': shift_distance
        }
    
    def analyze_multiple_shifts(self, topography, moho_depth, window_size=1000000,
                                shift_min=20000, shift_max=80000, shift_step=20000,
                                Te_range=(5000, 80000), min_std_topo=100, min_std_moho=100):
        """
        Perform moving window analysis with multiple shift distances
        
        Parameters:
        -----------
        topography : 2D array
            Topography data (m)
        moho_depth : 2D array
            Moho depth data (m)
        window_size : float
            Size of moving window in meters (default: 1000 km)
        shift_min : float
            Minimum shift distance in meters (default: 20 km)
        shift_max : float
            Maximum shift distance in meters (default: 80 km)
        shift_step : float
            Step size for shift distance in meters (default: 20 km)
        Te_range : tuple
            Range of Te values to search (m)
        min_std_topo : float
            Minimum standard deviation in topography for valid window
        min_std_moho : float
            Minimum standard deviation in Moho for valid window
            
        Returns:
        --------
        results : dict
            Dictionary with results for each shift distance
        """
        shift_distances = np.arange(shift_min, shift_max + shift_step, shift_step)
        all_results = {}
        
        for shift_dist in shift_distances:
            print(f"\n{'='*80}")
            print(f"Analyzing with shift distance: {shift_dist/1000:.0f} km")
            print(f"{'='*80}")
            result = self.analyze(topography, moho_depth, 
                                 window_size=window_size,
                                 shift_distance=shift_dist,
                                 Te_range=Te_range,
                                 min_std_topo=min_std_topo,
                                 min_std_moho=min_std_moho)
            all_results[shift_dist] = result
        
        return all_results

