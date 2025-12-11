import numpy as np 
from matplotlib import pyplot as plt
import plotly.express as px

from typing import List
from typing import Dict, Any
import csv
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import interp2d
from scipy.interpolate import RegularGridInterpolator
import os


from indentation.indentationset import IndentationSet



def overlay_hertzian_fit(*indentation_sets: 'IndentationSet', R=5, nu=0.5, n=3.0/2.0):

    for idx, indentation_set in enumerate(indentation_sets):

            for curve in indentation_set.data:

                param = 'contact_point'
                contact_disp = curve[param][1]
                contact_F = curve[param][2]

                param = 'youngs_modulus'
                E = curve[param][0] / 1.0e3

                z = np.linspace(0, 0.5, 1000)
                
                hertz_F = 4.0 / 3.0 * E * np.sqrt(R) / (1 - nu ** 2) * np.sign(z) * np.power(np.abs(z), n)
                
                print(curve['processed']['keep'])
                print(curve['metadata']['file'])
                
                plt.figure()
                plt.plot(curve["raw"]["z"], curve["raw"]["force"], 'k-')
                plt.plot(z + contact_disp, hertz_F + contact_F, 'r-', linewidth=2)
                plt.axvline(x = curve["contact_point"][1], color='b')
                plt.xlabel("Tip-sample separation [um]")
                plt.ylabel("Force [uN]")
                plt.ylim([min(curve["raw"]["force"]), max(curve["raw"]["force"])])
                #plt.title(f"{curve['keep']}")
                plt.show()






def plot_hertzian_fit(force, disp, hertz_fit, z, E_mod, r_2):
    # fig = px.scatter(x=disp, y=force)
    # fig.add_scatter(x=z, y=hertz_fit, name="E = " + str(round(E_mod, 2)) + " kPa")
    # fig.update_layout(xaxis_title="Displacement [um]", yaxis_title=r'Force [uN]', title="R^2 = " + str(round(r_2, 2)), font=dict(size=20))
    # fig.show()

    plt.figure()
    plt.plot(disp, force, 'b*')
    plt.plot(z, hertz_fit, 'r-')
    plt.legend(['Data', f"E = {round(E_mod, 4)} kPa"])
    plt.xlabel('Displacement [um]')
    plt.ylabel('Force [uN]')
    plt.title(f"R-squared = {str(round(r_2, 2))}")
    plt.show()


def plot_mean_force_curves(*indentation_sets: 'IndentationSet',
                          labels: List[str] = None,
                          **kwargs) -> None:
    """
    Create line plots of mean force curves with standard deviation bands.
    
    Args:
        *indentation_sets: Variable number of IndentationSet instances
        labels: Labels for each IndentationSet. If None, uses default naming
        **kwargs: Additional keyword arguments:
            figsize: Figure size as (width, height), default (10, 6)
            colors: List of colors for each IndentationSet. If None, uses default colors
            title: Plot title, default "Force Curves Comparison"
            xlabel: X-axis label, default "Z Position (nm)"
            ylabel: Y-axis label, default "Force (nN)"
            alpha_band: Alpha value for std bands, default 0.2
    """
    if not indentation_sets:
        raise ValueError("At least one IndentationSet instance is required")
    
    # Get kwargs with defaults
    figsize = kwargs.get('figsize', (8, 6))
    colors = kwargs.get('colors', None)
    title = kwargs.get('title', "Force Curves Comparison")
    xlabel = kwargs.get('xlabel', r'displacement [$\mu$m]')
    ylabel = kwargs.get('ylabel', r'force [$\mu$N]')
    alpha_band = kwargs.get('alpha_band', 0.2)
    
    # If no labels provided, create default ones
    if labels is None:
        labels = [f"Set {i+1}" for i in range(len(indentation_sets))]
    
    # If no colors provided, use default color cycle
    if colors is None:
        colors = plt.cm.tab10(np.linspace(0, 1, len(indentation_sets)))
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot each indentation set
    for idx, indentation_set in enumerate(indentation_sets):
        # Extract mean data
        z = indentation_set.mean["z"]
        force = indentation_set.mean["force"]
        force_std = indentation_set.mean["force_std"]
        
        # Plot mean line
        line = ax.plot(z, force, 
                      color=colors[idx],
                      label=labels[idx],
                      linewidth=2,
                      zorder=2)
        
        # Plot standard deviation band
        ax.fill_between(z,
                       force - force_std,
                       force + force_std,
                       color=colors[idx],
                       alpha=alpha_band,
                       zorder=1)
    
    # Customize plot
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Make sure zero is included in the y-axis
    ylim = ax.get_ylim()
    ax.set_ylim(min(ylim[0], 0), ylim[1])
    
    # Adjust layout
    plt.tight_layout()
    
    # Show plot
    plt.show()



def create_height_map(*indentation_sets: 'IndentationSet', r_2_thresh=0,
                    parameter_names: List[str] = None,
                    labels: List[str] = None):

    param = 'contact_point'

    for idx, indentation_set in enumerate(indentation_sets):
        # for curve in indentation_set.data:
        #     print("!index!")
        #     print(curve["metadata"]["index"])

        values = [curve[param] for curve in indentation_set.data
                  if param in curve]

        values = [item[1] for item in values]
        #r_2 = [item[1] for item in values]
        #values = [item[0] for item in values if (item[1] >= r_2_thresh and item[0] > 0)]
        #values = [item[0] if (item[1] >= r_2_thresh and item[0] > 0) else None for item in values]
        #values = [item[0] for item in values]

        if not values:
            continue

        size = int(np.sqrt(len(values)))  # Ensure square grid
        if size * size != len(values):
            raise ValueError("Number of measurements must be a perfect square.")

        # Generate the order pattern dynamically
        order = np.zeros((size, size), dtype=int)
        index = 0
        for i in range(size - 1, -1, -1):  # Start from the bottom row
            if (size - 1 - i) % 2 == 0:  # Left to right
                order[i, :] = range(index, index + size)
            else:  # Right to left
                order[i, :] = range(index + size - 1, index - 1, -1)
            index += size

        # Fill heatmap data using generated order
        heatmap_data = np.zeros((size, size))
        for i in range(size):
            for j in range(size):
                heatmap_data[i, j] = values[order[i, j]]

        leveled_heatmap = remove_slope(heatmap_data)

        plot_3Dheatmaps(heatmap_data, leveled_heatmap)
        
        cmap = plt.cm.coolwarm.copy()
        cmap.set_bad(color='black')

        rms = np.sqrt(np.mean(heatmap_data**2))
        print("RMS")
        print(rms)

        rms_leveled = np.sqrt(np.mean(leveled_heatmap**2))
        print("RMS_leveled")
        print(rms_leveled)
        
        # Plot heatmap
        plt.figure(figsize=(5, 5))
        plt.imshow(heatmap_data, cmap=cmap, interpolation='none')
        #plt.imshow(heatmap_data, cmap=cmap, interpolation='bilinear')

        # Add color bar
        plt.colorbar(label="h [um]")

        # Add labels
        for i in range(size):
            for j in range(size):
                plt.text(j, i, f"{heatmap_data[i, j]:.1f}", ha='center', va='center', color='black')

        # Set axis labels and title
        plt.xticks([])
        plt.yticks([])
        plt.title(f"{size}x{size} Heightmap of Measurements")

        # Show plot
        plt.show()



def plot_3Dheatmaps(original_data, leveled_data):
    """
    Plot original and slope-removed 2D surfaces with proper spacing and interpolation.

    Parameters:
        original_data (np.ndarray): 2D array of original heights.
        leveled_data (np.ndarray): 2D array with slope removed.
        spacing (float): Distance between adjacent points in units (default 10).
        interp_factor (int): Factor to interpolate between points for smooth surfaces.
    """
    spacing = 10
    interp_factor = 5
    
    rows, cols = original_data.shape
    
    # Original coordinate grids
    x = np.arange(cols) * spacing
    y = np.arange(rows) * spacing
    
    # Interpolators
    interp_orig = RegularGridInterpolator((y, x), original_data, method='linear')
    interp_leveled = RegularGridInterpolator((y, x), leveled_data, method='linear')
    
    # Fine grid for smooth plotting
    x_fine = np.linspace(x.min(), x.max(), cols*interp_factor)
    y_fine = np.linspace(y.min(), y.max(), rows*interp_factor)
    X_fine, Y_fine = np.meshgrid(x_fine, y_fine)
    
    points_fine = np.stack([Y_fine.ravel(), X_fine.ravel()], axis=-1)
    Z_orig_fine = interp_orig(points_fine).reshape(Y_fine.shape)
    Z_leveled_fine = interp_leveled(points_fine).reshape(Y_fine.shape)
    
    # --- 2D Heatmaps ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    im0 = axes[0].imshow(original_data, origin='lower', cmap='viridis')
    axes[0].set_title("Original Heatmap")
    plt.colorbar(im0, ax=axes[0])

    im1 = axes[1].imshow(leveled_data, origin='lower', cmap='viridis')
    axes[1].set_title("Slope-Removed Heatmap")
    plt.colorbar(im1, ax=axes[1])
    plt.show()
    
    # --- 3D Surface Plots ---
    fig = plt.figure(figsize=(14, 6))

    # Original surface
    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    ax1.plot_surface(X_fine, Y_fine, Z_orig_fine, cmap='viridis', edgecolor='k')
    ax1.set_title("Original Surface")
    ax1.set_xlabel("X (units)")
    ax1.set_ylabel("Y (units)")
    ax1.set_zlabel("Height")

    # Slope-removed surface
    ax2 = fig.add_subplot(1, 2, 2, projection='3d')
    ax2.plot_surface(X_fine, Y_fine, Z_leveled_fine, cmap='viridis', edgecolor='k')
    ax2.set_title("Slope-Removed Surface")
    ax2.set_xlabel("X (units)")
    ax2.set_ylabel("Y (units)")
    ax2.set_zlabel("Height")

    plt.show()




def remove_slope(heatmap_data):
    """
    Remove the overall slope from a 2D heatmap by fitting a plane and subtracting it.

    Parameters:
        heatmap_data (np.ndarray): 2D array of heights.

    Returns:
        np.ndarray: 2D array of the same shape with slope removed.
    """
    rows, cols = heatmap_data.shape

    # Create coordinate grids
    X, Y = np.meshgrid(np.arange(cols), np.arange(rows))

    # Flatten the arrays for linear regression
    X_flat = X.flatten()
    Y_flat = Y.flatten()
    Z_flat = heatmap_data.flatten()

    # Stack coordinates with a constant term for intercept
    A = np.c_[X_flat, Y_flat, np.ones_like(X_flat)]

    # Solve for plane coefficients: Z = a*X + b*Y + c
    coeff, _, _, _ = np.linalg.lstsq(A, Z_flat, rcond=None)
    a, b, c = coeff

    # Create the fitted plane
    plane = a*X + b*Y + c

    # Subtract the plane to remove slope
    leveled_data = heatmap_data - plane

    return leveled_data




def create_heat_map(*indentation_sets: 'IndentationSet', r_2_thresh=0,
                    parameter_names: List[str] = None,
                    labels: List[str] = None):

    param = 'youngs_modulus'

    for idx, indentation_set in enumerate(indentation_sets):
        # for curve in indentation_set.data:
        #     print("!index!")
        #     print(curve["metadata"]["index"])

        filepath = indentation_set.data[0]["metadata"]["file"]
        name = indentation_set.data[0]["metadata"]["name"]

        values = [curve[param] for curve in indentation_set.data
                  if param in curve]

        param_values = [item[0] for item in values]
        r_2 = [item[1] for item in values]
        #values = [item[0] for item in values if (item[1] >= r_2_thresh and item[0] > 0)]
        values = [item[0] if (item[1] >= r_2_thresh and item[0] > 0) else None for item in values]
        #values = [item[0] for item in values]

        if not values:
            continue

        size = int(np.sqrt(len(values)))  # Ensure square grid
        if size * size != len(values):
            raise ValueError("Number of measurements must be a perfect square.")

        # Generate the order pattern dynamically
        order = np.zeros((size, size), dtype=int)
        index = 0
        for i in range(size - 1, -1, -1):  # Start from the bottom row
            if (size - 1 - i) % 2 == 0:  # Left to right
                order[i, :] = range(index, index + size)
            else:  # Right to left
                order[i, :] = range(index + size - 1, index - 1, -1)
            index += size

        # Fill heatmap data using generated order
        heatmap_data = np.zeros((size, size))
        for i in range(size):
            for j in range(size):
                heatmap_data[i, j] = values[order[i, j]]

        cmap = plt.cm.coolwarm.copy()
        cmap.set_bad(color='black')

        # Plot heatmap
        plt.figure(figsize=(5, 5))
        plt.imshow(heatmap_data, cmap=cmap, interpolation='none')
        #plt.imshow(heatmap_data, cmap=cmap, interpolation='bilinear')

        # Add color bar
        plt.colorbar(label="E [kPa]")

        # Add labels
        for i in range(size):
            for j in range(size):
                plt.text(j, i, f"{heatmap_data[i, j]:.1f}", ha='center', va='center', color='black')

        # Set axis labels and title
        plt.xticks([])
        plt.yticks([])
        plt.title(f"{size}x{size} Heatmap of Measurements")
        plt.tight_layout()

        folder_name = os.path.dirname(filepath) + "\\"
        folder_name = str(folder_name) + "\\output\\"
        
        os.makedirs(folder_name, exist_ok=True)
        
        plt.savefig(str(folder_name) + "Heatmap_" + str(name) + ".jpg")
        print(str(folder_name) + "Heatmap_" + str(name) + ".jpg")

        # Show plot
        plt.show()



def plot_lengths(*indentation_sets: 'IndentationSet'):

    for idx, indentation_set in enumerate(indentation_sets):

            for curve in indentation_set.data:
                plt.figure()
                plt.plot(curve["raw"]["z"], curve["raw"]["force"], 'k-')
                plt.axvline(x = curve["contact_point"][1], color='r')
                plt.axvline(x = curve["release_point"][1], color='b')
                plt.legend(["Data", "Contact Point", "Release Point"])
                plt.xlabel("Tip-sample separation [um]")
                plt.ylabel("Force [uN]")
                plt.show()


def export_fluidfm(filename_str, *indentation_sets: 'IndentationSet'):

    for idx, indentation_set in enumerate(indentation_sets):
        filename = str(indentation_set.data[0]["metadata"]["name"]) + ".csv"

        print(str(indentation_set.data[0]["metadata"]["file"]))
        print(filename)

        filename_str = filename_str + ".csv"
        with open(filename_str, mode='w', newline='') as file:
            writer = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

            writer.writerow(['Name', 'Index', 'Max_Force', 'Retraction_Length', 'Max_Indentation_Force', 'Indentation_Depth', 'Force_Change', 'Disp_Change', 'Initial_Retraction_Slope', 'Final_Retraction_Slope', 'Initial_Indentation_Slope', 'Final_Indentation_Slope', 'Initial_Unindent_Slope', 'Max_Retract_Slope', 'Max_Overall_Retract_Slope'])
            for curve in indentation_set.data:
                max_force = curve['max_retraction_force']
                retraction_length = curve['retraction_length']
                max_indentation_force = curve['max_indentation_force']
                indentation_depth = curve['indentation_depth']
                force_change = curve['force_change'][0]
                disp_change = curve['force_change'][1]
                index = curve["metadata"]["index"]
                name = curve["metadata"]["name"]
                initial_retract_slope = curve['linear_slope'][2]
                final_retract_slope = curve['linear_slope'][3]
                initial_indent_slope = curve['linear_indent_slope'][2]
                final_indent_slope = curve['linear_indent_slope'][3]
                initial_unindent_slope = curve['linear_unindent_slope'][2]
                max_retract_slope = curve['linear_slope'][4]
                max_overall_retract_slope = curve['linear_unindent_slope'][4]
                print(f"initial slope: {initial_retract_slope}")
                writer.writerow([name, index, max_force, retraction_length, max_indentation_force, indentation_depth, force_change, disp_change, initial_retract_slope, final_retract_slope, initial_indent_slope, final_indent_slope, initial_unindent_slope, max_retract_slope, max_overall_retract_slope])


def export_fluidfm_indentation(*indentation_sets: 'IndentationSet'):

    for idx, indentation_set in enumerate(indentation_sets):
        filename = str(indentation_set.data[0]["metadata"]["name"]) + ".csv"

        print(str(indentation_set.data[0]["metadata"]["file"]))
        print(filename)
        
        with open(filename, mode='w') as file:
            writer = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

            writer.writerow(['Name', 'Index', 'Max_Indentation_Force', 'Indentation_Depth'])
            for curve in indentation_set.data:
                max_indentation_force = curve['max_indentation_force']
                indentation_depth = curve['indentation_depth']
                index = curve["metadata"]["index"]
                name = curve["metadata"]["name"]
                writer.writerow([name, index, max_indentation_force, indentation_depth])


def create_histogram(*indentation_sets: 'IndentationSet', r_2_thresh=0,
                    parameter_names: List[str] = None,
                    labels: List[str] = None):

    param = 'youngs_modulus'

    for idx, indentation_set in enumerate(indentation_sets):
        filename = str(indentation_set.data[0]["metadata"]["name"]) + ".csv"
        filepath = str(indentation_set.data[0]["metadata"]["file"])
        name = str(indentation_set.data[0]["metadata"]["name"])

        with open(filename, mode='w') as file:
            writer = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

            writer.writerow(['Name', 'Index', 'E', 'R_2'])
            for curve in indentation_set.data:
                E = curve[param][0]
                r_2 = curve[param][1]
                index = curve["metadata"]["index"]
                name = curve["metadata"]["name"]
                writer.writerow([name, index, E, r_2])

            

        values = [curve[param] for curve in indentation_set.data
                  if param in curve]

        for E, r_2, keep in values:
            print(f"{E:<5} {r_2:<10}")

        
        param_values = [item[0] for item in values]
        r_2 = [item[1] for item in values]
        #values = [item[0] for item in values if (item[1] >= r_2_thresh and item[0] > 0)]
        values = [item[0] if (item[1] >= r_2_thresh and item[0] > 0) else None for item in values]
        #values = [item[0] for item in values]

        if not values:
            continue

        hist_values = [x for x in values if x is not None]
        print(hist_values)

        plt.figure()
        plt.hist(hist_values)
        plt.xlabel("Apparent Young's modulus [kPa]")
        plt.tight_layout()


        folder_name = os.path.dirname(filepath) + "\\"
        folder_name = str(folder_name) + "\\output\\"
        
        os.makedirs(folder_name, exist_ok=True)
        
        plt.savefig(str(folder_name) + "Histogram_" + str(name) + ".jpg")
        print(str(folder_name) + "Histogram_" + str(name) + ".jpg")

        
        plt.show()





def plot_curve_parameters_bar(*indentation_sets: 'IndentationSet',
                            r_2_thresh=0,
                            parameter_names: List[str] = None,
                            labels: List[str] = None,
                            **kwargs) -> None:
    """
    Create bar plots with error bars for parameters across multiple IndentationSet instances.
    Includes mean value labels inside each bar.
    
    Args:
        *indentation_sets: Variable number of IndentationSet instances
        parameter_names: List of parameter names to plot. If None, plots all parameters
        labels: Labels for each IndentationSet. If None, uses default naming
        **kwargs: Additional keyword arguments:
            figsize: Figure size as (width, height), default (10, 6)
            colors: List of colors for each IndentationSet. If None, uses default colors
            title: Plot title, default "Parameter Comparison"
            ylabel: Y-axis label, default "Value"
            show_individual_points: Whether to overlay individual data points, default True
    """
    if not indentation_sets:
        raise ValueError("At least one IndentationSet instance is required")
        
    # Get kwargs with defaults
    figsize = kwargs.get('figsize', (10, 6))
    colors = kwargs.get('colors', None)
    title = kwargs.get('title', "Parameter Comparison")
    ylabel = kwargs.get('ylabel', "Value")
    show_individual_points = kwargs.get('show_individual_points', True)
    
    # If no labels provided, create default ones
    if labels is None:
        labels = [f"Set {i+1}" for i in range(len(indentation_sets))]
    
    # If no colors provided, use default color cycle
    if colors is None:
        colors = plt.cm.tab10(np.linspace(0, 1, len(indentation_sets)))
    
    # Get all available parameters if none specified
    if parameter_names is None:
        # Get all parameter names from the first curve of the first set
        # excluding 'raw' and 'processed' keys
        parameter_names = [key for key in indentation_sets[0].data[0].keys() 
                         if key not in ['raw', 'processed']]
    
    # Create figure
    fig, axes = plt.subplots(len(parameter_names), 1, figsize=figsize)
    if len(parameter_names) == 1:
        axes = [axes]
    
    # Create a list to store bar containers for the legend
    bar_containers = []
    
    # Plot each parameter
    for ax_idx, param in enumerate(parameter_names):
        ax = axes[ax_idx]
        x_positions = np.arange(len(indentation_sets))
        width = 0.8
        
        bars = []  # Store bars for this parameter
        filepath = ""
        name = ""
        for idx, indentation_set in enumerate(indentation_sets):
            values = [curve[param] for curve in indentation_set.data 
                      if param in curve]

            print(indentation_set.data[0]["metadata"])
            filepath = indentation_set.data[0]["metadata"]["file"]
            name = indentation_set.data[0]["metadata"]["name"]

            param_values = [item[0] for item in values]
            r_2 = [item[1] for item in values]
            values = [item[0] for item in values if (item[1] >= r_2_thresh and item[0] > 0)]
            
            if not values:
                continue
                
            # Calculate statistics
            mean_val = np.mean(values)
            median_val = np.median(values)
            std_val = np.std(values)
            print("Mean and std:", mean_val, std_val)
            print(f"Median: {median_val}")
            print(f"Number included: {len(values)}")
            
            # Plot bar with error
            bar = ax.bar(x_positions[idx], mean_val, width, 
                        yerr=std_val, 
                        color=colors[idx], 
                        alpha=0.6,
                        capsize=5)
            bars.append(bar)
            
            # Add mean value text inside the bar
            ax.text(x_positions[idx], mean_val * 0.1,  # Position at 10% of bar height
                   f'{mean_val:.1f}',  # Round to 1 decimal
                   ha='center',  # Horizontal alignment
                   va='bottom',  # Vertical alignment
                   color='black',
                   fontweight='bold')
            
            # Optionally plot individual points
            if show_individual_points:
                ax.scatter(np.full_like(values, x_positions[idx]), 
                          values,
                          color='black',
                          alpha=0.4,
                          s=20)
        
        # Store the bars for legend (only for first parameter)
        if ax_idx == 0:
            bar_containers = bars
        
        # Customize subplot
        ax.set_title(f"{param}")
        ax.set_xticks(x_positions)
        ax.set_xticklabels(labels)
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
    
    # Adjust layout to prevent overlap
    plt.tight_layout()

    # Save as jpg file
    folder_name = os.path.dirname(filepath) + "\\"
    folder_name = str(folder_name) + "\\output\\"
        
    os.makedirs(folder_name, exist_ok=True)
        
    plt.savefig(str(folder_name) + "BarChart_" + str(name) + ".jpg")
    print(str(folder_name) + "BarChart_" + str(name) + ".jpg")
    

    # Show plot
    plt.show()











def plot_instance_parameters_bar(indentation_set: 'IndentationSet',
                                 r_2_thresh=0,
                               parameter_names: List[str] = None,
                               **kwargs) -> None:
    """
    Create bar plot comparing different parameters within a single IndentationSet instance.
    Includes mean value labels inside each bar.
    
    Args:
        indentation_set: An IndentationSet instance
        parameter_names: List of parameter names to plot. If None, plots all parameters
        **kwargs: Additional keyword arguments:
            figsize: Figure size as (width, height), default (10, 6)
            colors: List of colors for parameters. If None, uses default colors
            title: Plot title, default "Parameter Comparison"
            ylabel: Y-axis label, default "Value"
            show_individual_points: Whether to overlay individual points, default True
    """
    # Get kwargs with defaults
    figsize = kwargs.get('figsize', (10, 6))
    colors = kwargs.get('colors', None)
    title = kwargs.get('title', "Parameter Comparison")
    ylabel = kwargs.get('ylabel', "Value")
    y_text_loc = kwargs.get('y_text_loc', 1)
    show_individual_points = kwargs.get('show_individual_points', True)
    
    # Get all available parameters if none specified
    if parameter_names is None:
        parameter_names = [key for key in indentation_set.data[0].keys() 
                         if key not in ['raw', 'processed']]
    
    # If no colors provided, use default color cycle
    if colors is None:
        colors = plt.cm.tab10(np.linspace(0, 1, len(parameter_names)))
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    width = 0.8
    
    # For each parameter, calculate statistics and plot
    bars = []
    x_positions = np.arange(len(parameter_names))
    
    for idx, param in enumerate(parameter_names):
        # Extract values for this parameter
        values = [curve[param] for curve in indentation_set.data 
                 if param in curve]

        values = [item[0] for item in values if (item[1] >= r_2_thresh and item[0] > 0)]

        if not values:
            continue
            
        # Calculate statistics
        mean_val = np.mean(values)
        std_val = np.std(values)
        
        # Plot bar with error
        bar = ax.bar(x_positions[idx], mean_val, width,
                    yerr=std_val,
                    color=colors[idx],
                    alpha=0.6,
                    capsize=5)
        bars.append(bar)
        
        # Add mean value text inside the bar
        ax.text(x_positions[idx], y_text_loc,  # Position at 10% of bar height
               f'{mean_val:.2f}',  # Round to 1 decimal
               ha='center',  # Horizontal alignment
               va='bottom',  # Vertical alignment
               color='black',
               fontweight='bold')
        
        # Optionally plot individual points
        if show_individual_points:
            ax.scatter(np.full_like(values, x_positions[idx]),
                      values,
                      color='black',
                      alpha=0.4,
                      s=20)
    
    # Customize plot
    ax.set_title(title)
    ax.set_xticks(x_positions)
    ax.set_xticklabels(parameter_names, rotation=45, ha='right')
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    
    # Adjust layout to prevent overlap
    plt.tight_layout()
    
    # Show plot
    plt.show()
