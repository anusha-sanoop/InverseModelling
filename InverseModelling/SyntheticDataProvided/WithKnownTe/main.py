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
    
    topography_file = "D:\Marine\Project\InverseModelling\Test data_DRK\Large and small mount\MarsModel3TwoMountainsFar_1Topo_S20km.grd"
    moho_file = "D:\Marine\Project\InverseModelling\Test data_DRK\Large and small mount\Mohod_depth_add30km_final_S20km.grd"
    
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
    print("\n" + "="*80)
    print("LOADING DATA")
    print("="*80)
    
    X_topo, Y_topo, topography, dx_topo, dy_topo, nx_topo, ny_topo, \
        xmin_topo, xmax_topo, ymin_topo, ymax_topo = read_surfer_grd(topography_file)
    
    X_moho, Y_moho, moho_depth, dx_moho, dy_moho, nx_moho, ny_moho, \
        xmin_moho, xmax_moho, ymin_moho, ymax_moho = read_surfer_grd(moho_file)
    
    # Check grid compatibility
    compatible, message = check_grid_compatibility(X_topo, Y_topo, X_moho, Y_moho)
    if not compatible:
        print(f"\nWARNING: {message}")
        print("Attempting to use topography grid as reference...")
        # For now, we'll proceed with topography grid
        # In a full implementation, you might want to interpolate
    
    # Use topography grid spacing
    dx = dx_topo
    dy = dy_topo
    
    print(f"\nUsing grid spacing: {dx/1000:.2f} x {dy/1000:.2f} km")
    
    # Analysis parameters - Mars parameters
    print("\n" + "="*80)
    print("ANALYSIS PARAMETERS")
    print("="*80)
    print("Physical Parameters (Mars):")
    print("  Crustal density: 2900 kg/m³")
    print("  Mantle density: 3500 kg/m³")
    print("  Gravity: 3.72 m/s²")
    print("\nAnalysis Parameters:")
    print("  Window size: 1000 km")
    print("  Grid spacing: 20 km")
    print("  Shift distance range: 20-80 km")
    print("  Computational domain: 2000 km")
    
    # Single window analysis (entire domain)
    print("\n" + "="*80)
    print("1. SINGLE WINDOW ANALYSIS (ENTIRE DOMAIN)")
    print("="*80)
    
    # Initialize inverter with Mars parameters
    inverter = ElasticThicknessInversion(dx=dx, dy=dy,
                                        rho_load=2900, rho_m=3500, 
                                        rho_infill=2900, g=3.72)
    
    # Perform inversion on entire domain
    result = inverter.invert_elastic_thickness(
        topography, moho_depth,
        Te_range=(5000, 80000),  # Search range: 5-80 km
        method='bounded'
    )
    
    print(f"\nResults:")
    print(f"  Best-fit Te: {result['Te_best']/1000:.2f} km")
    print(f"  RMS Misfit: {result['rms_best']/1000:.2f} km")
    
    # Plot results
    import matplotlib.pyplot as plt
    fig1 = plot_inversion_results(topography, moho_depth, result, X_topo, Y_topo)
    plt.show(block=False)
    
    # Moving window analysis
    print("\n" + "="*80)
    print("MOVING WINDOW ANALYSIS")
    print("="*80)
    
    use_moving_window = input("\nPerform moving window analysis? (y/n, default: y): ").lower()
    if use_moving_window != 'n':
        # Use specified parameters
        window_size = 1000000  # 1000 km
        shift_min = 20000      # 20 km
        shift_max = 80000      # 80 km
        shift_step = 20000     # 20 km
        Te_min = 5000         # 5 km
        Te_max = 80000        # 80 km
        
        print(f"\nUsing parameters:")
        print(f"  Window size: {window_size/1000:.0f} km")
        print(f"  Shift distance range: {shift_min/1000:.0f}-{shift_max/1000:.0f} km (step: {shift_step/1000:.0f} km)")
        print(f"  Te search range: {Te_min/1000:.0f}-{Te_max/1000:.0f} km")
        
        # Ask if multiple shifts or single shift
        use_multiple_shifts = input("\nAnalyze multiple shift distances? (y/n, default: y): ").lower()
        
        mw_analyzer = MovingWindowAnalysis(dx=dx, dy=dy)
        
        if use_multiple_shifts != 'n':
            # Perform analysis with multiple shift distances
            mw_results_dict = mw_analyzer.analyze_multiple_shifts(
                topography, moho_depth,
                window_size=window_size,
                shift_min=shift_min,
                shift_max=shift_max,
                shift_step=shift_step,
                Te_range=(Te_min, Te_max)
            )
            
            # Use the first shift distance result for plotting
            first_shift = list(mw_results_dict.keys())[0]
            mw_results = mw_results_dict[first_shift]
            
            # Plot Te maps for different shift distances
            n_shifts = len(mw_results_dict)
            fig2, axes = plt.subplots(2, min(3, n_shifts), figsize=(5*min(3, n_shifts), 10), squeeze=False)
            
            shift_keys = list(mw_results_dict.keys())[:3]  # Show first 3
            
            for idx, shift_dist in enumerate(shift_keys):
                if idx < axes.shape[1]:
                    result = mw_results_dict[shift_dist]
                    # Plot Te map
                    from visualization import plot_te_map
                    # We'll create a simple plot here
                    viridis_with_bad = plt.cm.get_cmap('viridis').copy()
                    viridis_with_bad.set_bad('lightgray')
                    
                    x_centers_km = result['x_centers'] * dx / 1000
                    y_centers_km = result['y_centers'] * dy / 1000
                    extent_te = [x_centers_km.min(), x_centers_km.max(),
                                 y_centers_km.min(), y_centers_km.max()]
                    
                    im1 = axes[0, idx].imshow(result['Te_map']/1000, extent=extent_te,
                                             cmap=viridis_with_bad, origin='lower', aspect='equal')
                    axes[0, idx].set_title(f'Te Map - Shift {shift_dist/1000:.0f} km')
                    axes[0, idx].set_xlabel('X (km)')
                    axes[0, idx].set_ylabel('Y (km)')
                    plt.colorbar(im1, ax=axes[0, idx], label='Te (km)')
                    
                    im2 = axes[1, idx].imshow(result['rms_map']/1000, extent=extent_te,
                                             cmap=viridis_with_bad, origin='lower', aspect='equal')
                    axes[1, idx].set_title(f'RMS Map - Shift {shift_dist/1000:.0f} km')
                    axes[1, idx].set_xlabel('X (km)')
                    axes[1, idx].set_ylabel('Y (km)')
                    plt.colorbar(im2, ax=axes[1, idx], label='RMS (km)')
            
            plt.tight_layout()
            plt.show(block=False)
        else:
            # Single shift distance analysis
            shift_distance = float(input(f"Shift distance (km, default: {shift_min/1000:.0f}): ") or f"{shift_min/1000:.0f}") * 1000
            mw_results = mw_analyzer.analyze(
                topography, moho_depth,
                window_size=window_size,
                shift_distance=shift_distance,
                Te_range=(Te_min, Te_max)
            )
            
        # Plot Te map
        fig2 = plot_te_map(mw_results, X_topo, Y_topo)
        plt.show(block=False)
        mw_results_dict = None
    else:
        mw_results = None
        mw_results_dict = None
        mw_results_dict = None
    
    # Sensitivity analysis
    print("\n" + "="*80)
    print("SENSITIVITY ANALYSIS")
    print("="*80)
    
    use_sensitivity = input("\nPerform sensitivity analysis? (y/n, default: n): ").lower()
    if use_sensitivity == 'y':
        Te_values, rms_values = inverter.sensitivity_analysis(
            topography, moho_depth,
            Te_range=(5000, 80000),
            n_points=50
        )
        
        fig3 = plot_sensitivity(Te_values, rms_values)
        plt.show(block=False)
    
    # Save results
    print("\n" + "="*80)
    print("SAVING RESULTS")
    print("="*80)
    
    output_folder = create_output_folder()
    
    save_data = input(f"\nSave numerical results? (y/n, default: y): ").lower()
    if save_data != 'n':
        save_dict = {
            'topography': topography,
            'moho_depth': moho_depth,
            'X': X_topo,
            'Y': Y_topo,
            'Te_best': result['Te_best'],
            'rms_best': result['rms_best'],
            'moho_predicted': result['moho_pred']
        }
        
        # Save multiple shift results if available
        if mw_results_dict is not None:
            for shift_dist, result in mw_results_dict.items():
                save_dict[f'Te_map_shift_{int(shift_dist/1000)}km'] = result['Te_map']
                save_dict[f'rms_map_shift_{int(shift_dist/1000)}km'] = result['rms_map']
        elif mw_results is not None:
            save_dict['Te_map'] = mw_results['Te_map']
            save_dict['rms_map'] = mw_results['rms_map']
        
        np.savez(os.path.join(output_folder, 'inversion_results.npz'), **save_dict)
        print(f"Results saved to: {output_folder}/inversion_results.npz")
    
    save_figures = input(f"\nSave figures? (y/n, default: y): ").lower()
    if save_figures != 'n':
        try:
            fig1.savefig(os.path.join(output_folder, 'inversion_results.png'), dpi=300)
            print(f"Figure saved: {output_folder}/inversion_results.png")
        except Exception as e:
            print(f"Error saving figure: {e}")
        
        if mw_results is not None:
            try:
                fig2.savefig(os.path.join(output_folder, 'te_map.png'), dpi=300)
                print(f"Figure saved: {output_folder}/te_map.png")
            except Exception as e:
                print(f"Error saving figure: {e}")
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    print(f"\nResults saved in: {output_folder}/")
    
    # Keep plots open
    plt.show()


if __name__ == "__main__":
    main()

