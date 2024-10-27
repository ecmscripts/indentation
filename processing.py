import numpy as np 
import pandas as pd
from scipy.optimize import fmin
from scipy.signal import convolve
from scipy.signal.windows import gaussian
from scipy import interpolate
from tqdm import tqdm
import os, sys, glob

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from neuralnets import *
from auxiliary import *

# prefixes
# findContact: finds contact point and returns cropped and shifted data 
# processing: changeds the curves in "processed"
# parameter: extracts any 1D parameter (like youngs modulus or a structural stiffness) from each curve
# plot: visualization

def do_nothing(data):
    return data

def findContact_minimum(data):
    ix_cut = np.argmin(data["force"])
    for key in ["time", "z", "force"]:
        data[key] = data[key][ix_cut:]
        data[key] -= data[key][0]
    return data
    

def findContact_blackMagic_CNN(data, net, N):

    force_full_sliced = data["force"]
    displ_full_sliced = -data["z"]
    length = len(force_full_sliced)

    force = normalize_signal(force_full_sliced, N)
    img = create_scaleSpace(force, N)
    
    output = net(torch.tensor(img, dtype=torch.float).view(1, 1 ,N, N).to("cpu"))
    
    loc = np.argmax(output.detach().cpu().numpy())
    ix_cut = int(loc/N*length)

    if ix_cut/length < 0.9:
        for key in ["time", "z", "force"]:
            data[key] = data[key][ix_cut:]
            data[key] -= data[key][0]
        return data
    else:
        print("Discarded curve.")


def findContact_blackMagic(data, window_size, padding_fraction):
    """
    Find the contact point in force-displacement data.
    """
    force = data["force"].copy()
    displacement = -data["z"].copy()
    padding = int(padding_fraction * window_size)
    
    # Initialize list for storing filtered signals
    filtered_signals = []
    
    # Generate filtered signals with different window sizes
    for filter_length in range(padding, window_size - padding):
        # Generate Gaussian filter of correct length
        x_filter = np.linspace(-2, 2, filter_length)
        gaussian_filter = 1 / (np.sqrt(2 * np.pi)) * np.exp(-x_filter ** 2 / 2)
        gaussian_filter /= np.sum(gaussian_filter)  # Normalize filter
        
        # Compute second derivative of the Gaussian filter analytically
        second_derivative_filter = (x_filter ** 2 - 1) * gaussian_filter
        
        # Convolve force with the second derivative filter
        filtered_force = np.convolve(force, second_derivative_filter, mode='same')
        filtered_signals.append(filtered_force)
    
    # Copy filtered signals for marking
    filtered_signals_marked = [signal.copy() for signal in filtered_signals]
    signal_length = len(filtered_signals[0])
    zline = np.zeros(signal_length)
    
    # Step through the signal at regular intervals
    step_size = max(int(window_size / 30), 1)  # Ensure step size is at least 1
    for global_index in range(10, int(3 * window_size / 4), step_size):
        index = global_index
        # Track maximum through all filtered versions
        for i_rev, line in enumerate(reversed(filtered_signals)):
            # Ensure index doesn't exceed array bounds
            index = min(index, signal_length - 2)
            indices = [index, index + 1, index - 1]
            neighbor_values = [line[idx] for idx in indices]
            max_index_in_indices = np.argmax(neighbor_values)
            index = indices[max_index_in_indices]
            # Mark the tracked position
            filtered_signals_marked[len(filtered_signals) - 1 - i_rev][index] = -0.3
                
        zline[index] += 1  # Count occurrences of each position
            
    # Find the contact point index
    contact_index = np.argmax(zline)
    
    # Create new dictionary with cropped data
    result_data = {}
    for key in ["time", "z", "force"]:
        # Crop array from contact point and subtract initial value
        cropped_array = data[key][contact_index:].copy()
        result_data[key] = cropped_array - cropped_array[0]
    
    return result_data


# Example of a processing function
def processing_shift_to_zero(data):
    """Subtract the initial force value to correct baseline."""
    f0 = data['force'][0]
    z0 = data['z'][0]
    data['force'] = data['force'] - f0
    data['z'] = data['z'] - z0
    return data


def processing_smooth_data(data, window_size=5):
    """Apply a simple moving average to smooth the data."""
    data['force'] = np.convolve(data['force'], np.ones(window_size)/window_size, mode='valid')
    data['z'] = data['z'][:len(data['force'])]  # Adjust z to match the length
    return data


def parameter_youngs_modulus(data, radius, nu, cutoff, x0=[0.005, 0], show_plot=False, keyname="youngs_modulus"):
    '''
    Enter consistent units, R in um, cutoff in percent of the radius (~1-100).
    Adds the keyword "youngs_modulus" to the data.
    '''

    def force_function(x, displ):
        alpha, beta = x
        return alpha * displ**(3.0/2.0) + beta

    def target_function(x, force, displ):
        phi = np.dot(force_function(x, displ) - force,
                     force_function(x, displ) - force)
        return phi

    fsize = 14
    radius = radius/1000

    displ = -data["z"]
    force = data["force"]
    ix_end = np.argmin(np.abs(displ-1000*(cutoff/100)*radius))
    xOpt = fmin(target_function, x0, args=(force[:ix_end], displ[:ix_end]), disp=False)
    alpha, beta = xOpt
    Emod = np.round(1000*3.0/4.0*(1-nu**2)/radius**0.5*alpha * (1000**(3.0/2.0)/1000000), 2)

    if show_plot:
        f_fit = force_function(xOpt, displ)
        plt.figure(figsize=(8, 6))
        plt.plot(displ, force, lw=4, label="data")
        plt.plot(displ, f_fit, 'r', label="fitted")
        plt.title(r'Hertzian fit with R=' + str(int(R*1000)) + r"$\mu$m, and depth=" + str(cutoff) + r"$\mu$m", fontsize=fsize, y=1.01)
        plt.xlabel(r'displacement [$\mu$m]', fontsize=fsize, labelpad=10)
        plt.ylabel(r'force [$\mu$N]', fontsize=fsize, labelpad=10)
        # plt.xticks([], [])
        # plt.yticks([], [])
        plt.legend(loc=2, fontsize=fsize, fancybox=True, framealpha=0.0)
        plt.tick_params(labelsize=fsize, labelcolor='k', pad=10)
        plt.tight_layout()
        
        print("\nYoungs Modulus: ", Emod, 'kPa\n')

    return Emod, keyname



# VIZ ========================================================================================

from typing import List
from typing import Dict, Any
from matplotlib import pyplot as plt

def plot_curve_parameters_bar(*indentation_sets: 'IndentationSet', 
                            parameter_names: List[str] = None,
                            labels: List[str] = None,
                            **kwargs) -> None:
    """
    Create bar plots with error bars for parameters across multiple IndentationSet instances.
    
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
        for idx, indentation_set in enumerate(indentation_sets):
            # Extract parameter values from all curves
            values = [curve[param] for curve in indentation_set.data 
                     if param in curve]
            
            if not values:
                continue
                
            # Calculate statistics
            mean_val = np.mean(values)
            std_val = np.std(values)
            print("Mean and std:", mean_val, std_val)
            
            # Plot bar with error
            bar = ax.bar(x_positions[idx], mean_val, width, 
                        yerr=std_val, 
                        color=colors[idx], 
                        alpha=0.6,
                        capsize=5)
            bars.append(bar)
            
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
    
    # Set overall title
    # fig.suptitle(title, fontsize=14, y=1.02)
    
    # Adjust layout to prevent overlap
    plt.tight_layout()
    
    # Show plot
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
        line = ax.plot(-z, force, 
                      color=colors[idx],
                      label=labels[idx],
                      linewidth=2,
                      zorder=2)
        
        # Plot standard deviation band
        ax.fill_between(-z,
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


def plot_instance_parameters_bar(indentation_set: 'IndentationSet', 
                               parameter_names: List[str] = None,
                               **kwargs) -> None:
    """
    Create bar plot comparing different parameters within a single IndentationSet instance.
    
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
                    label=param,
                    capsize=5)
        bars.append(bar)
        
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
    
    # Adjust layout to prevent label cutoff
    plt.tight_layout()
    
    # Show plot
    plt.show()
