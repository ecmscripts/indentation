from dataclasses import dataclass, field
from typing import List, Dict, Union, Callable, Literal
from pathlib import Path
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt

# Configure Matplotlib for LaTeX rendering
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.size": 14,
    "text.latex.preamble": r"\usepackage{amsmath}"
})

@dataclass
class IndentationSet:
    """Collection of indentation curves from one or multiple files."""
    data: List[Dict] = field(default_factory=list)
    
    def __init__(self, file_paths: Union[str, Path, List[Union[str, Path]]], exp_type):
        """Initialize from one or multiple data files.
        exp_type: "afm" or "ft"
        """
        self.data = []
        self.exp_type = exp_type
        self.append(file_paths)

    def _load_file_afm_calib(self, path: Path) -> List[Dict]:
        """Internal method to load data from a single file."""

        def parse_metadata(file_path):
            """Helper method for metatdata."""
            metadata = {}
            
            with open(file_path, 'r') as file:
                for line in file:
                    # Skip empty lines
                    if not line.strip():
                        continue
                        
                    # Stop when we hit a non-metadata line
                    if not line.startswith('#'):
                        break
                        
                    key, value = line[1:].strip().split('=')
                    
                    # Handle different cases based on key
                    if key in ['Spring-Constant', 'Deflection-Sensitivity']:
                        # Extract number before unit
                        value = float(''.join(c for c in value if c.isdigit() or c in '.-e'))
                        
                    elif key in ['SpecMap-CurIndex', 'SpecMap-PhaseCount']:
                        value = int(value)
                        
                    elif key in ['SpecMap-Dim', 'SpecMap-Size']:
                        # Convert semicolon-separated values to numpy array
                        value = np.array([float(x) if '.' in x or 'e' in x else int(x) 
                                        for x in value.split(';')])
                        
                    metadata[key] = value
            
            return metadata

        metadata = parse_metadata(path)
        _, voltage, z1 = np.loadtxt(path, skiprows=18, delimiter=";").T


        curves = []
        curve_dict = {
            "raw": {
                "force": voltage, # FIX THIS LATER
                "z": -z1,
                "time": np.zeros(len(voltage)),
            },
            "metadata": {
                "file": str(path)
            }
        }
    
        curves.append(curve_dict)
            
        return curves

    def _load_file_afm(self, path: Path) -> List[Dict]:
        """Internal method to load data from a single file."""

        def parse_metadata(file_path):
            """Helper method for metatdata."""
            metadata = {}
            
            with open(file_path, 'r') as file:
                for line in file:
                    # Skip empty lines
                    if not line.strip():
                        continue
                        
                    # Stop when we hit a non-metadata line
                    if not line.startswith('#'):
                        break
                        
                    key, value = line[1:].strip().split('=')
                    
                    # Handle different cases based on key
                    if key in ['Spring-Constant', 'Deflection-Sensitivity']:
                        # Extract number before unit
                        value = float(''.join(c for c in value if c.isdigit() or c in '.-e'))
                        
                    elif key in ['SpecMap-CurIndex', 'SpecMap-PhaseCount']:
                        value = int(value)
                        
                    elif key in ['SpecMap-Dim', 'SpecMap-Size']:
                        # Convert semicolon-separated values to numpy array
                        value = np.array([float(x) if '.' in x or 'e' in x else int(x) 
                                        for x in value.split(';')])
                        
                    metadata[key] = value
            
            return metadata

        metadata = parse_metadata(path)
        z1, voltage, _ = np.loadtxt(path, skiprows=18, delimiter=";").T


        # convert to um
        z1 = z1 * 1e6

        # convert to uN: first volt to deflection in m, then  
        d_load = metadata["Deflection-Sensitivity"] * voltage 
        force = 1e6 * metadata["Spring-Constant"] * d_load 
        w = z1 - d_load
        
        curves = []
        curve_dict = {
            "raw": {
                "force": force,
                "z": -w,
                "time": np.zeros(len(force)),
            },
            "metadata": {
                "file": str(path)
            }
        }
    
        curves.append(curve_dict)
            
        return curves

    def _load_file_ft(self, path: Path) -> List[Dict]:
        """Internal method to load data from a single file."""
        # Read the file using pandas
        imported = pd.read_csv(
            path,
            skiprows=6,
            names=["ix", "t", "displ", "x", "y", "z", "f", "fb"],
            sep=r"\s+"
        )
        
        curves = []
        # Get unique curve indices in this file
        unique_indices = imported['ix'].unique()
        
        # Process each curve
        for _ in unique_indices:
            # Get data for current curve
            curve_data = imported[imported['ix'] == _].copy()
            
            # Create dictionary for current curve
            curve_dict = {
                "raw": {
                    "force": curve_data['f'].values,
                    "z": curve_data['z'].values,
                    "time": curve_data['t'].values,
                },
                "metadata": {
                    "file": str(path)
                }
            }
            curves.append(curve_dict)
            
        return curves
    
    def append(self, file_paths: Union[str, Path, List[Union[str, Path]]]) -> None:
        """Append data from additional files to the existing measurement set."""
        # Convert input to list of Path objects
        if isinstance(file_paths, (str, Path)):
            file_paths = [file_paths]
        
        # Convert all paths to Path objects and resolve them
        paths = [Path(p).resolve() for p in file_paths]
        
        # Validate paths
        for path in paths:
            if not path.is_file():
                raise FileNotFoundError(f"File not found: {path}")
        
        # Process each file 
        if self.exp_type == "ft":
            for path in paths:
                new_curves = self._load_file_ft(path)
                self.data.extend(new_curves)
        elif self.exp_type == "afm":
            for path in paths:
                new_curves = self._load_file_afm(path)
                self.data.extend(new_curves)
        elif self.exp_type == "afmcalib":
            for path in paths:
                new_curves = self._load_file_afm_calib(path)
                self.data.extend(new_curves)
        else:
            print("Experiment type does not exist. :(")

    
    def __len__(self) -> int:
        """Returns the number of curves."""
        return len(self.data)
    
    def get_curve(self, index: int) -> Dict:
        """Get a specific curve by index."""
        if index >= len(self):
            raise IndexError(f"Curve index {index} out of range (0-{len(self)-1})")
        return self.data[index]
    
    def process_raw(self, processing_pipeline: List[Callable]):
        """Process all curves using a sequence of functions."""
        for curve in self.data:
            processed_data = {
                            "force": np.copy(curve["raw"]["force"]),
                            "z": np.copy(curve["raw"]["z"]),
                            "time": np.copy(curve["raw"]["time"]) if "time" in curve["raw"] else None}
            for func in processing_pipeline:
                processed_data = func(processed_data)
            curve["processed"] = processed_data
    
    def calculate_curve_parameter(self, function: Callable, **kwargs):
        """Process all curves using a sequence of functions."""
        if "processed" in self.data[0].keys():
            for curve in self.data:
                processed_data = curve["processed"].copy()
                parameter, parameter_name = function(processed_data, **kwargs)
                curve[parameter_name] = parameter
        else:
            print("Error: Process raw data first.")

    def calculate_mean(self, alpha: float = 0.75) -> None:
        """
        Calculate mean force curve from processed data, considering only curves that reach
        a certain depth threshold defined by alpha * maximum_depth.
        
        Args:
            alpha: Threshold factor (0-1) for including curves based on their maximum depth
        """
        # Check if processed data exists
        if "processed" not in self.data[0]:
            print("Error: Process data first.")
            return
        
        # Find maximum depth across all curves (using absolute values)
        max_depths = [np.max(np.abs(curve["processed"]["z"])) for curve in self.data]
        overall_max_depth = max(max_depths)
        depth_threshold = alpha * overall_max_depth
        
        # Filter curves that meet the depth threshold
        valid_curves = []
        for curve in self.data:
            if np.max(np.abs(curve["processed"]["z"])) >= depth_threshold:
                valid_curves.append(curve["processed"])
        
        if not valid_curves:
            print("Error: No curves meet the depth threshold criteria.")
            return
        
        # Find the curve with the smallest maximum depth among valid curves
        min_max_depths = [np.max(np.abs(curve["z"])) for curve in valid_curves]
        reference_idx = np.argmin(min_max_depths)
        reference_curve = valid_curves[reference_idx]
        
        # Get z values from reference curve
        z_reference = np.sort(reference_curve["z"])  # Ensure z is monotonic
        
        # Initialize array to store interpolated forces
        interpolated_forces = np.zeros((len(valid_curves), len(z_reference)))
        
        # Interpolate all curves to match the reference z values
        for i, curve in enumerate(valid_curves):
            # Sort z and force arrays together to ensure monotonic z
            sort_idx = np.argsort(curve["z"])
            z_sorted = curve["z"][sort_idx]
            f_sorted = curve["force"][sort_idx]
            
            # Interpolate to reference z values
            interpolated_forces[i] = np.interp(z_reference, z_sorted, f_sorted)
        
        # Calculate mean force
        mean_force = np.mean(interpolated_forces, axis=0)
        std_force = np.std(interpolated_forces, axis=0)
        
        # Store the mean curve as a property of the class instance
        self.mean = {
            "z": z_reference,
            "force": mean_force,
            "force_std": std_force
        }

    def plot(self, 
             indices: Union[int, List[int], Literal["all"]], 
             use_processed: bool = True,
             figsize=(12, 6), 
             show_title=True,
             show_legend=True,
             show=True,
             colors=None,
             ax=None):  # Add ax as an optional parameter
        """Plot force vs. z-position for one or multiple curves."""
        
        # Use the provided axis or create a new one if none is provided
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = ax.get_figure()  # Ensure fig is defined when an existing ax is used
        
        # Convert input to list of indices to plot
        if isinstance(indices, int):
            indices_to_plot = [indices]
        elif indices == "all":
            indices_to_plot = list(range(len(self)))
        else:
            indices_to_plot = indices
            
        # Validate indices
        for idx in indices_to_plot:
            if idx >= len(self):
                raise IndexError(f"Curve index {idx} out of range (0-{len(self)-1})")
        
        # Set up colors
        if colors is None:
            colors = plt.cm.viridis(np.linspace(0, 1, len(indices_to_plot)))
        
        # Plot each curve
        for i, idx in enumerate(indices_to_plot):
            curve = self.get_curve(idx)
            data = curve.get("processed" if use_processed and "processed" in curve else "raw")
            metadata = curve.get("metadata")
            
            ax.plot(
                -data["z"], 
                data["force"], 
                '-', 
                color=colors[i] if isinstance(colors, np.ndarray) else None,
                linewidth=2,
                label=os.path.basename(metadata["file"]).split(".")[0] + "_" + f'{idx+1}'
            )
        
        # Add labels and title
        ax.set_xlabel(r'displacement [$\mu$m]')
        ax.set_ylabel(r'force [$\mu$N]')
        if show_title:
            ax.set_title(f'Force vs. Z Position - Multiple Curves\n{len(indices_to_plot)} curves shown')
        
        # Add grid
        ax.grid(True, linestyle='--', alpha=0.2)
        
        # Add legend if multiple curves
        if show_legend:
            if len(indices_to_plot) > 1:
                ax.legend(loc="center left", framealpha=0, bbox_to_anchor=(1, 0.5))
        
        # Adjust layout to prevent label cutoff
        plt.tight_layout()
        
        # Show the plot if requested
        if show:
            plt.show()
        
        return fig, ax
    
    def plot_all(self, **kwargs):
        """Convenience method to plot all curves."""
        return self.plot("all", **kwargs)

    def plot_all_raw(self, **kwargs):
        """Convenience method to plot all curves."""
        return self.plot("all", use_processed=False, **kwargs)


    def plot_mean(self, ax=None, figsize=(8,6)):
        # Create an axis if one is not provided
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = ax.get_figure()  # Ensure fig is defined when an existing ax is used

        # Calculate -z for plotting
        z_neg = -np.array(self.mean["z"])
        force_mean = np.array(self.mean["force"])
        force_std = np.array(self.mean["force_std"])

        # Plot the mean line
        ax.plot(z_neg, force_mean, color="blue", label="mean force")

        # Plot the error fan (mean ± std)
        ax.fill_between(z_neg, force_mean - force_std, force_mean + force_std, color="lightgray", alpha=0.5)

        # Labels and styling
        ax.set_xlabel(r'displacement [$\mu$m]')
        ax.set_ylabel(r'force [$\mu$N]')
        ax.legend(framealpha=0)
        ax.grid(True, linestyle='--', alpha=0.2)

        return fig, ax
