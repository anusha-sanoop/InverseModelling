"""
Main script for Elastic Thickness Inversion

This script loads topography and Moho depth data from .grd files and
performs elastic thickness inversion using the convolution method.

"""

import numpy as np
import os
from datetime import datetime
from data_loader import read_surfer_grd, check_grid_compatibility
from elastic_thickness_inversion import ElasticThicknessInversion
from moving_window_analysis import MovingWindowAnalysis
from visualization import plot_inversion_results, plot_te_map, plot_sensitivity
import matplotlib.pyplot as plt


def create_output_folder():
    """Create a unique output folder with timestamp inside the current directory"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    folder_name = f"Output_{timestamp}"
    output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), folder_name)
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    print(f"\nCreated output folder: {output_path}")
    return output_path


def main():
    
    option = int(input(f"Press \t1. Synthetic data  \t2. Real data: \t"))
    if option == 1:
        print("\n")
        print("Synthetic Data")
        print("."*15)
      #  topography_file = "D:\\Marine\\Project\\InverseModelling\\Test data_DRK\\Large and small mount\\MarsModel3TwoMountainsFar_1Topo_S20km.grd"
      #  moho_file = "D:\\Marine\\Project\\InverseModelling\\Test data_DRK\\Large and small mount\\Mohod_depth_add30km_final_S20km.grd"
        topography_file = "D:\\PhD\\Marine\\Projects\\SyntheticData\\InverseModelling\\Test data_DRK\\Large and small mount\\MarsModel3TwoMountainsFar_1Topo_S20km.grd"
        moho_file = "D:\\PhD\\Marine\\Projects\\SyntheticData\\InverseModelling\\Test data_DRK\\Large and small mount\\Mohod_depth_add30km_final_S20km.grd"

    elif option == 2:
        print("\n")
        print("Real Data")
        print("."*10)
        # topography_file = "D:\\Marine\\Project\\InverseModelling\\Test data_DRK\\Real data\\Topo_proj.grd"
        # moho_file = "D:\\Marine\\Project\\InverseModelling\\Test data_DRK\\Real data\\Moho_Tc+Topo.grd"
        topography_file = "D:\\PhD\\Marine\\Projects\\SyntheticData\\InverseModelling\\Test data_DRK\\Real data\\Topo_proj.grd"
        moho_file = "D:\\PhD\\Marine\\Projects\\SyntheticData\\InverseModelling\\Test data_DRK\\Real data\\Moho_Tc+Topo.grd"
    else:
        print("Invalid option selected. Exiting.")
        return

    # Check if files exist
    if not os.path.exists(topography_file):
        print(f"\nERROR: Topography file not found: {topography_file}")
        print("Please update the 'topography_file' variable in main.py")
        return
    
    if not os.path.exists(moho_file):
        print(f"\nERROR: Moho file not found: {moho_file}")
        print("Please update the 'moho_file' variable in main.py")
        return
    
    # Load data
    print("\n")
    print("LOADING DATA","."*50)
    
    X_topo, Y_topo, topography, dx_topo, dy_topo, nx_topo, ny_topo, \
        xmin_topo, xmax_topo, ymin_topo, ymax_topo = read_surfer_grd(topography_file)
    
    X_moho, Y_moho, moho_depth, dx_moho, dy_moho, nx_moho, ny_moho, \
        xmin_moho, xmax_moho, ymin_moho, ymax_moho = read_surfer_grd(moho_file)
    
    # Check grid compatibility
    compatible, message = check_grid_compatibility(X_topo, Y_topo, X_moho, Y_moho)
    if not compatible:
        print(f"\nWARNING: {message}")
        print("Attempting to use topography grid as reference...")
    
    # Use topography grid spacing
    dx = dx_topo
    dy = dy_topo
    
    print(f"\nUsing grid spacing: {dx/1000:.2f} x {dy/1000:.2f} km")
    
    print("Physical Parameters:")
    print("  Crustal density: 2900 kg/m³")
    print("  Mantle density: 3500 kg/m³")
    print("  Gravity: 3.72 m/s²")
    
    print("\n")
    print("MOVING WINDOW ANALYSIS")
    print("." * 23)

    # Set parameters
    if option == 1:  # Synthetic data
        window_size_list = [1000000]  # 1000 km
        shift_list = list(range(20000, 80001, 20000))  # 20,40,60,80 km
        Te_min = 5000
        Te_max = 80000
    elif option == 2:  # Real data
        window_size_list = [700000, 1000000]  # 700 km and 1000 km
        shift_list = [60000, 80000, 100000, 120000]  # 60,80,100,120 km
        Te_min = 5000
        Te_max = 8000
    mw_analyzer = MovingWindowAnalysis(dx=dx, dy=dy)
    output_folder = create_output_folder()

    # Loop over window sizes
    for window_size in window_size_list:
        for shift_dist in shift_list:
            print(f"\nRunning analysis with window size {window_size/1000:.0f} km and shift {shift_dist/1000:.0f} km")
        
            # Run moving window analysis
            mw_results_dict = mw_analyzer.analyze_multiple_shifts(
                topography, moho_depth,
                window_size=window_size,
                shift_min=shift_dist,
                shift_max=shift_dist,
                shift_step=shift_dist,
                Te_range=(Te_min, Te_max)
            )
            # Plot results
            shift_keys = list(mw_results_dict.keys())
            n_shifts = len(shift_keys)

            fig, axes = plt.subplots(2, n_shifts, figsize=(5 * n_shifts, 10), squeeze=False)
            cmap = plt.get_cmap('viridis')
            cmap.set_bad('lightgray')

            for idx, shift_key in enumerate(shift_keys):
                result = mw_results_dict[shift_key]
                x_centers_km = result['x_centers'] * dx / 1000
                y_centers_km = result['y_centers'] * dy / 1000
                extent = [x_centers_km.min(), x_centers_km.max(),
                        y_centers_km.min(), y_centers_km.max()]

                # Te map
                im1 = axes[0, idx].imshow(result['Te_map']/1000, extent=extent,
                                      cmap=cmap, origin='lower', aspect='equal')
                axes[0, idx].set_title(f'Te (Shift {shift_key/1000:.0f} km)')
                axes[0, idx].set_xlabel('X (km)')
                axes[0, idx].set_ylabel('Y (km)')
                plt.colorbar(im1, ax=axes[0, idx], label='Te (km)')

                # RMS map
                im2 = axes[1, idx].imshow(result['rms_map']/1000, extent=extent,
                                      cmap=cmap, origin='lower', aspect='equal')
                axes[1, idx].set_title('RMS')
                axes[1, idx].set_xlabel('X (km)')
                axes[1, idx].set_ylabel('Y (km)')
                plt.colorbar(im2, ax=axes[1, idx], label='RMS (km)')

            plt.tight_layout()

            # Save figure
            fig_filename = f"moving_window_{int(window_size/1000)}km_shift_{int(shift_dist/1000)}km.png"
            fig.savefig(os.path.join(output_folder, fig_filename), dpi=300, bbox_inches='tight')
            print(f"Figure saved: {fig_filename}")
            plt.close(fig)

        # Sensitivity analysis
    

    inverter = ElasticThicknessInversion(dx=dx, dy=dy,
                                    rho_load=2900, rho_m=3500, 
                                    rho_infill=2900, g=3.72)

    use_sensitivity = input("\nPerform sensitivity analysis? (y/n, default: n): ").lower()
    if use_sensitivity == 'y':
        print("SENSITIVITY ANALYSIS")
        print("."*20)
        Te_values, rms_values = inverter.sensitivity_analysis(
            topography, moho_depth,
            Te_range=(5000, 80000),
            n_points=50
        )

        fig3 = plot_sensitivity(Te_values, rms_values)
        plt.show(block=False)

   
    # Keep plots open
    plt.show()


if __name__ == "__main__":
    main()

