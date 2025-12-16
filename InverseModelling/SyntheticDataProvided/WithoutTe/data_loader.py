"""
Data Loader for Surfer Grid Files (.grd)

This module provides functions to read Surfer ASCII grid files (DSAA format)
containing topography and Moho depth data.
"""

import numpy as np


def read_surfer_grd(filepath):
    """
    Read Surfer ASCII grid file (.grd) in DSAA format
    
    Parameters:
    -----------
    filepath : str
        Path to the .grd file
        
    Returns:
    --------
    X : 2D array
        X coordinate grid (meters)
    Y : 2D array
        Y coordinate grid (meters)
    data : 2D array
        Data values (same units as in file)
    dx : float
        Grid spacing in X direction (meters)
    dy : float
        Grid spacing in Y direction (meters)
    nx : int
        Number of columns
    ny : int
        Number of rows
    xmin, xmax : float
        X coordinate range
    ymin, ymax : float
        Y coordinate range
    """
    with open(filepath, 'r') as f:
        # Read header
        header = f.readline().strip()
        if header != 'DSAA':
            raise ValueError(f"File {filepath} is not in DSAA format. Found: {header}")
        
        # Read grid dimensions
        nx, ny = map(int, f.readline().split())
        
        # Read coordinate ranges
        xmin, xmax = map(float, f.readline().split())
        ymin, ymax = map(float, f.readline().split())
        zmin, zmax = map(float, f.readline().split())
        
        # Read all data values
        data_values = []
        for line in f:
            line = line.strip()
            if line:  # Skip empty lines
                data_values.extend(map(float, line.split()))
        
        # Reshape data array (Surfer stores data row by row, starting from top)
        data = np.array(data_values).reshape(ny, nx)
        
        # Create coordinate grids
        x = np.linspace(xmin, xmax, nx)
        y = np.linspace(ymin, ymax, ny)
        X, Y = np.meshgrid(x, y)
        
        # Calculate grid spacing
        dx = (xmax - xmin) / (nx - 1) if nx > 1 else 0
        dy = (ymax - ymin) / (ny - 1) if ny > 1 else 0
        
        print(f"Loaded {filepath}:")
        print(f"  Grid size: {nx} x {ny}")
        print(f"  X range: {xmin/1000:.1f} to {xmax/1000:.1f} km")
        print(f"  Y range: {ymin/1000:.1f} to {ymax/1000:.1f} km")
        print(f"  Grid spacing: {dx/1000:.2f} x {dy/1000:.2f} km")
        print(f"  Data range: {np.min(data):.2f} to {np.max(data):.2f}")
        
        return X, Y, data, dx, dy, nx, ny, xmin, xmax, ymin, ymax


def check_grid_compatibility(X1, Y1, X2, Y2):
    """
    Check if two grids are compatible (same dimensions and coordinates)
    
    Parameters:
    -----------
    X1, Y1 : 2D arrays
        First grid coordinates
    X2, Y2 : 2D arrays
        Second grid coordinates
        
    Returns:
    --------
    compatible : bool
        True if grids are compatible
    message : str
        Description of compatibility status
    """
    if X1.shape != X2.shape or Y1.shape != Y2.shape:
        return False, f"Grid dimensions don't match: {X1.shape} vs {X2.shape}"
    
    if not np.allclose(X1, X2, rtol=1e-5):
        return False, "X coordinates don't match"
    
    if not np.allclose(Y1, Y2, rtol=1e-5):
        return False, "Y coordinates don't match"
    
    return True, "Grids are compatible"

