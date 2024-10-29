import numpy as np 
from matplotlib import pyplot as plt

from typing import List
from typing import Dict, Any

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


def plot_curve_parameters_bar(*indentation_sets: 'IndentationSet', 
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
    
    # Show plot
    plt.show()


def plot_instance_parameters_bar(indentation_set: 'IndentationSet', 
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
