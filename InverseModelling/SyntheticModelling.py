import xarray as xr

class MohoAnalysis(MohoAnalysis):
    def _read_grd_file(self, filepath):
        ds = xr.open_dataset(filepath)
        # Assuming the data variable is 'z' and coordinates are 'x' and 'y'
        # Adjust if variable names are different in the actual GRD files
        data_array = ds['z'].values

        # Extract grid spacing and dimensions
        x_coords = ds['x'].values
        y_coords = ds['y'].values
        dx = np.abs(x_coords[1] - x_coords[0])
        dy = np.abs(y_coords[1] - y_coords[0])
        nx, ny = len(x_coords), len(y_coords)

        X, Y = np.meshgrid(x_coords, y_coords)

        return X, Y, data_array, dx, dy, nx, ny

    def get_user_parameters(self, use_synthetic_data=True):
        """
        Get comprehensive user input for the analysis
        """
        print("="*80)
        print("INTERACTIVE MOHO ANALYSIS WITH MOVING WINDOW TECHNIQUE")
        print("="*80)

        params = {}

        # Domain parameters
        print("\n DOMAIN/AREA:")
        if use_synthetic_data:
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
        else:
            print("Domain size and grid spacing derived from input files.")

        # Mountain parameters (only for synthetic data)
        if use_synthetic_data:
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

            # Moho parameters (only for synthetic data)
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
        else:
            params['mountains'] = {}
            params['moho'] = {'type': 'from_file'}


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

def main(topography_file=None, moho_file=None):
    """
    Main function to run the interactive Moho analysis
    """
    analyzer = MohoAnalysis()

    use_synthetic_data = (topography_file is None or moho_file is None)

    # Get user parameters
    params = analyzer.get_user_parameters(use_synthetic_data=use_synthetic_data)

    print(f"\n{'-'*80}")

    if use_synthetic_data:
        # Create synthetic data
        X, Y, topography = analyzer.create_synthetic_topography(params)
        moho_depth = analyzer.create_moho_depth(params, X, Y)
    else:
        print(f"Loading topography from {topography_file}")
        X, Y, topography, dx, dy, nx, ny = analyzer._read_grd_file(topography_file)
        print(f"Loading Moho from {moho_file}")
        _, _, moho_depth, _, _, _, _ = analyzer._read_grd_file(moho_file)

        params['grid_spacing'] = dx
        params['nx'] = nx
        params['ny'] = ny
        params['domain_size'] = nx * dx # Assuming square domain for simplicity

        print(f"Domain: {params['domain_size']/1000:.0f} x {params['domain_size']/1000:.0f} km")
        print(f"Grid: {params['nx']} x {params['ny']} points")

    # Gravity is now just for reference plotting
    gravity = analyzer.calculate_gravity_from_moho(topography, moho_depth, params)

    # Perform moving window analysis using moho_depth
    results = analyzer.moving_window_analysis(topography, moho_depth, params, X, Y)

    # Analyze distortions
    distortion_analysis = analyzer.analyze_distortions(results, params)

    # Generate plots
    fig1, fig2 = analyzer.plot_comprehensive_results(
        topography, gravity, moho_depth, results, distortion_analysis, params, X, Y)

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

# Call main with the provided file paths
analysis_results = main(topography_file='/content/MarsModel3TwoMountainsFar_1Topo_S20km.grd', moho_file='/content/Mohod_depth_add30km_final_S20km.grd')
