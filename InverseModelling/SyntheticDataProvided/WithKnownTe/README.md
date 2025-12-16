# Elastic Thickness Inversion Project

## Overview

This project implements the **Braitenberg convolution method** for estimating the effective elastic thickness (Te) of the lithosphere from topography and Moho depth data.

**Reference:**
Braitenberg, C., Ebbing, J., & Götze, H. J. (2002). Inverse modelling of elastic thickness by convolution method—the eastern Alps as a case example. *Earth and Planetary Science Letters*, 202(2), 387-404.

## Project Structure

```
ElasticThicknessProject/
├── main.py                          # Main script to run the analysis
├── elastic_thickness_inversion.py  # Core inversion class
├── data_loader.py                  # Functions to read .grd files
├── moving_window_analysis.py       # Moving window analysis class
├── visualization.py               # Plotting functions
└── README.md                       # This file
```

## Requirements

- Python 3.7+
- numpy
- scipy
- matplotlib

Install dependencies:
```bash
pip install numpy scipy matplotlib
```

## Data Files

This project expects two Surfer ASCII grid files (.grd) in DSAA format:

1. **Topography file**: `Model3TwoMountainsFar_1Topo_S20km.grd`
   - Elevation values in meters
   - Positive values = above sea level

2. **Moho depth file**: `Mohod_depth_add30km_final_S20km.grd`
   - Moho depth values in meters
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

## Usage

### Quick Start

1. Place your .grd files in the `ElasticThicknessProject` folder

2. Update file paths in `main.py` if needed:
   ```python
   topography_file = "Model3TwoMountainsFar_1Topo_S20km.grd"
   moho_file = "Mohod_depth_add30km_final_S20km.grd"
   ```

3. Run the analysis:
   ```bash
   python main.py
   ```

### What the Script Does

1. **Loads data** from .grd files
2. **Single window analysis**: Estimates Te for the entire domain
3. **Moving window analysis** (optional): Creates spatial Te map
4. **Sensitivity analysis** (optional): Tests different Te values
5. **Saves results**: Numerical data (.npz) and figures (.png)

## Output

The script creates an output folder `Output_YYYYMMDD_HHMMSS/` containing:

- `inversion_results.npz`: Numerical results (topography, Moho, Te, etc.)
- `inversion_results.png`: Visualization of inversion results
- `te_map.png`: Spatial Te map (if moving window analysis performed)

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

## Parameters

### Physical Constants (Mars - configured in `ElasticThicknessInversion`)

- `rho_load` = 2900 kg/m³ (crustal/load density)
- `rho_m` = 3500 kg/m³ (mantle density)
- `rho_infill` = 2900 kg/m³ (crustal/infill density)
- `E` = 1.0×10¹¹ Pa (Young's modulus)
- `nu` = 0.25 (Poisson's ratio)
- `g` = 3.72 m/s² (Mars gravity)

### Analysis Parameters

- **Te_range**: Search range for Te (5000-80000 meters = 5-80 km)
- **Window size**: 1000 km (moving window size)
- **Grid spacing**: 20 km (from data files)
- **Shift distance range**: 20-80 km (with 20 km step)
- **Computational domain**: 2000 km

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

## Example Workflow

1. **Prepare your data**:
   - Ensure both .grd files are in DSAA format
   - Check units (meters)
   - Verify grids match

2. **Run the analysis**:
   ```bash
   python main.py
   ```

3. **Review results**:
   - Check Te maps for spatial patterns
   - Review RMS misfit to assess quality
   - Examine cross-sections and residuals

4. **Save outputs**:
   - Figures saved as PNG files
   - Numerical results saved as .npz files

## Customization

### Using Different Physical Parameters

Modify the `ElasticThicknessInversion` initialization in `main.py`:

```python
inverter = ElasticThicknessInversion(
    dx=dx, dy=dy,
    rho_load=2670,    # Adjust for your region
    rho_m=3300,       # Adjust for your region
    rho_infill=2800   # Adjust for your region
)
```

### Programmatic Usage

You can also use the classes directly in your own scripts:

```python
from data_loader import read_surfer_grd
from elastic_thickness_inversion import ElasticThicknessInversion

# Load data
X, Y, topography, dx, dy, nx, ny, _, _, _, _ = read_surfer_grd("topo.grd")
_, _, moho_depth, _, _, _, _, _, _, _, _ = read_surfer_grd("moho.grd")

# Initialize inverter
inverter = ElasticThicknessInversion(dx=dx, dy=dy)

# Perform inversion
result = inverter.invert_elastic_thickness(
    topography, moho_depth,
    Te_range=(5000, 80000)
)

print(f"Best-fit Te: {result['Te_best']/1000:.2f} km")
```

## License

This project is provided as-is for research and educational purposes.

## Contact

For questions or issues, please refer to the original paper:
Braitenberg, C., Ebbing, J., & Götze, H. J. (2002). Inverse modelling of elastic thickness by convolution method—the eastern Alps as a case example. *Earth and Planetary Science Letters*, 202(2), 387-404.

