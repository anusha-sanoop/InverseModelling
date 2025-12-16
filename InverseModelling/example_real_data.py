"""
Example script for estimating elastic thickness from real Surfer grid data

This script demonstrates how to use your own topography and Moho depth data
to estimate elastic thickness using the Braitenberg convolution method.

Requirements:
- Topography .grd file (elevation in meters)
- Moho depth .grd file (depth in meters, positive downward)

Both files must be in Surfer ASCII grid format (DSAA format).
"""

from SyntheticDataModelling import main

# ============================================================================
# EXAMPLE 1: Using your own data files
# ============================================================================

# Replace these paths with your actual file paths
topography_file = "Test data_DRK/Real data/Topo_proj.grd"
moho_file = "Test data_DRK/Real data/Moho_Tc+Topo.grd"

# Run the analysis
print("="*80)
print("ELASTIC THICKNESS ESTIMATION FROM REAL DATA")
print("="*80)
print("\nThis will:")
print("1. Load your topography and Moho depth data")
print("2. Perform moving window analysis to estimate Te")
print("3. Generate plots and save results")
print("\n" + "="*80 + "\n")

# Uncomment the line below to run with your data:
# results = main(topography_file=topography_file, moho_file=moho_file)

# ============================================================================
# EXAMPLE 2: Quick single-window inversion (without moving window)
# ============================================================================

from SyntheticDataModelling import ElasticThicknessInversion
import numpy as np
import matplotlib.pyplot as plt

def quick_te_estimation(topography_file, moho_file, Te_range=(5000, 80000)):
    """
    Quick estimation of elastic thickness for the entire domain
    
    Parameters:
    -----------
    topography_file : str
        Path to topography .grd file
    moho_file : str
        Path to Moho depth .grd file
    Te_range : tuple
        Range of Te values to search (meters)
    
    Returns:
    --------
    result : dict
        Inversion results
    """
    from SyntheticDataModelling import MohoAnalysis
    
    analyzer = MohoAnalysis()
    
    # Load data
    print("Loading data files...")
    X, Y, topography, dx, dy, nx, ny = analyzer.read_surfer_grd(topography_file)
    _, _, moho_depth, _, _, _, _ = analyzer.read_surfer_grd(moho_file)
    
    # Initialize inverter
    inverter = ElasticThicknessInversion(dx=dx, dy=dy)
    
    # Perform inversion
    print(f"\nPerforming inversion for Te in range {Te_range[0]/1000:.1f} - {Te_range[1]/1000:.1f} km...")
    result = inverter.invert_elastic_thickness(
        topography, moho_depth,
        Te_range=Te_range,
        method='bounded'
    )
    
    print(f"\n" + "="*60)
    print("RESULTS:")
    print("="*60)
    print(f"Best-fit Elastic Thickness (Te): {result['Te_best']/1000:.2f} km")
    print(f"RMS Misfit: {result['rms_best']:.2f} m")
    print("="*60)
    
    # Plot results
    fig = inverter.plot_results(topography, moho_depth, result)
    plt.show()
    
    return result

# Uncomment to run quick estimation:
# quick_result = quick_te_estimation(topography_file, moho_file)

# ============================================================================
# NOTES:
# ============================================================================
"""
IMPORTANT INFORMATION:

1. DATA REQUIREMENTS:
   - You need BOTH topography AND Moho depth data
   - Both files must be Surfer ASCII grid (.grd) format (DSAA)
   - Topography: elevation in meters (positive = above sea level)
   - Moho depth: depth in meters (positive = below surface)
   - Both grids should have the same dimensions and coordinates

2. WHAT IS ELASTIC THICKNESS (Te)?
   - Te represents the effective elastic thickness of the lithosphere
   - It controls how the lithosphere responds to topographic loads
   - Higher Te = stiffer lithosphere = less flexure
   - Lower Te = weaker lithosphere = more flexure

3. THE METHOD:
   - Uses convolution method (Braitenberg et al., 2002)
   - Compares predicted Moho flexure (from topography) with observed Moho
   - Finds Te that minimizes the misfit
   - Moving window approach allows spatial variation in Te

4. IF YOU ONLY HAVE TOPOGRAPHY:
   - You need Moho depth data from seismic studies or gravity inversion
   - The method requires both datasets to work
   - Consider using gravity data if Moho depth is unavailable
     (requires modification of the code)

5. OUTPUT:
   - Te maps showing spatial variation
   - RMS misfit maps showing fit quality
   - Cross-sections and residual plots
   - Summary statistics

For more information, see:
Braitenberg, C., Ebbing, J., & Götze, H. J. (2002).
Inverse modelling of elastic thickness by convolution method—
the eastern Alps as a case example.
Earth and Planetary Science Letters, 202(2), 387-404.
"""

