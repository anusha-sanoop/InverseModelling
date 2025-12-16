# Elastic Thickness Estimation from Surfer Grid Data

## Overview

This code implements the **Braitenberg convolution method** for estimating elastic thickness (Te) of the lithosphere from topography and Moho depth data. The method is based on:

**Braitenberg, C., Ebbing, J., & Götze, H. J. (2002). Inverse modelling of elastic thickness by convolution method—the eastern Alps as a case example. *Earth and Planetary Science Letters*, 202(2), 387-404.**

## What You Need

### Required Data Files

1. **Topography Grid File** (.grd)
   - Surfer ASCII grid format (DSAA format)
   - Elevation values in **meters**
   - Positive values = above sea level

2. **Moho Depth Grid File** (.grd)
   - Surfer ASCII grid format (DSAA format)
   - Moho depth values in **meters**
   - Positive values = depth below surface
   - Should have same grid dimensions and coordinates as topography

### File Format

Both files must be in **Surfer ASCII grid format (DSAA)**:
```
DSAA
nx ny
xmin xmax
ymin ymax
zmin zmax
[data values row by row]
```

## Quick Start

### Option 1: Using Your Real Data Files

```python
from SyntheticDataModelling import main

# Specify your file paths
topography_file = "path/to/your/topography.grd"
moho_file = "path/to/your/moho_depth.grd"

# Run the analysis
results = main(topography_file=topography_file, moho_file=moho_file)
```

### Option 2: Quick Single-Window Estimation

For a quick estimate without moving window analysis:

```python
from SyntheticDataModelling import ElasticThicknessInversion, MohoAnalysis

analyzer = MohoAnalysis()

# Load your data
X, Y, topography, dx, dy, nx, ny = analyzer.read_surfer_grd("topography.grd")
_, _, moho_depth, _, _, _, _ = analyzer.read_surfer_grd("moho_depth.grd")

# Initialize inverter
inverter = ElasticThicknessInversion(dx=dx, dy=dy)

# Perform inversion
result = inverter.invert_elastic_thickness(
    topography, moho_depth,
    Te_range=(5000, 80000),  # Search range in meters (5-80 km)
    method='bounded'
)

print(f"Best-fit Te: {result['Te_best']/1000:.2f} km")
print(f"RMS Misfit: {result['rms_best']:.2f} m")
```

### Option 3: Using Synthetic Data (for testing)

```python
from SyntheticDataModelling import main

# Run with synthetic data (interactive prompts)
results = main()
```

## Understanding the Method

### What is Elastic Thickness (Te)?

- **Te** represents the effective elastic thickness of the lithosphere
- It controls how the lithosphere flexes under topographic loads
- **Higher Te** = stiffer lithosphere = less flexure
- **Lower Te** = weaker lithosphere = more flexure

### How It Works

1. **Forward Model**: Predicts Moho flexure from topography using flexural theory
   - Uses convolution in frequency domain (FFT)
   - Relates topography load to Moho deflection via flexure filter

2. **Inversion**: Finds Te that minimizes misfit
   - Compares predicted vs. observed Moho depth
   - Minimizes RMS error between model and observations

3. **Moving Window**: Analyzes spatial variation
   - Divides domain into overlapping windows
   - Estimates Te for each window
   - Creates spatial Te map

### The Flexure Filter

The method uses the relationship:

```
W(k) = F(k) × H(k)
```

Where:
- `W(k)` = Fourier transform of Moho flexure
- `H(k)` = Fourier transform of topographic load
- `F(k)` = Flexure filter = ρ_load / [(ρ_m - ρ_infill) + (D/g) × k⁴]

The flexural rigidity `D` depends on Te:
```
D = E × Te³ / [12 × (1 - ν²)]
```

## Outputs

The analysis produces:

1. **Te Maps**: Spatial distribution of elastic thickness
2. **RMS Misfit Maps**: Quality of fit for each window
3. **Cross-sections**: Comparison of observed vs. predicted Moho
4. **Residual Plots**: Difference between model and observations
5. **Summary Statistics**: Mean, std dev, range of Te values

## Parameters You Can Adjust

### Physical Constants (in `ElasticThicknessInversion.__init__`)

- `rho_load` = 2670 kg/m³ (load density)
- `rho_m` = 3300 kg/m³ (mantle density)
- `rho_infill` = 2800 kg/m³ (crustal/infill density)
- `E` = 1.0×10¹¹ Pa (Young's modulus)
- `nu` = 0.25 (Poisson's ratio)

### Analysis Parameters

- **Te_range**: Search range for Te (e.g., (5000, 80000) meters = 5-80 km)
- **Window size**: Size of moving window (typically 100-300 km)
- **Shift distance**: Step size for moving window (typically 20-50 km)

## Troubleshooting

### Problem: "Grid dimensions don't match"

**Solution**: Ensure both grids have the same dimensions and coordinates. You may need to:
- Regrid one dataset to match the other
- Use the same coordinate system
- Check that both files cover the same area

### Problem: "Insufficient data variation"

**Solution**: The method needs sufficient topography and Moho variation. Try:
- Using larger windows
- Checking data quality
- Ensuring data covers areas with significant topography

### Problem: "Te values seem unrealistic"

**Solution**: Check:
- Units are correct (meters, not kilometers)
- Density values are appropriate for your region
- Data quality and coverage

### Problem: "Only have topography, no Moho data"

**Solution**: You need Moho depth data. Options:
- Obtain from seismic studies
- Use gravity data to invert for Moho (requires additional code)
- Use published Moho depth models

## Example Workflow

1. **Prepare your data**:
   - Ensure both .grd files are in DSAA format
   - Check units (meters)
   - Verify grids match

2. **Run quick test**:
   ```python
   # Quick single-window test
   result = quick_te_estimation("topo.grd", "moho.grd")
   ```

3. **Full analysis**:
   ```python
   # Full moving window analysis
   results = main(topography_file="topo.grd", moho_file="moho.grd")
   ```

4. **Review results**:
   - Check Te maps for spatial patterns
   - Review RMS misfit to assess quality
   - Examine cross-sections and residuals

5. **Save outputs**:
   - Figures saved as PNG files
   - Numerical results saved as .npz files

## References

Braitenberg, C., Ebbing, J., & Götze, H. J. (2002). Inverse modelling of elastic thickness by convolution method—the eastern Alps as a case example. *Earth and Planetary Science Letters*, 202(2), 387-404.

## Files

- `SyntheticDataModelling.py`: Main code with all classes and functions
- `example_real_data.py`: Example scripts for using your data
- `README_ElasticThickness.md`: This documentation

## Questions?

If you encounter issues or need help:
1. Check that your .grd files are in correct format
2. Verify units are in meters
3. Ensure both grids have matching dimensions
4. Review the example scripts for usage patterns

