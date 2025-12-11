from dataclasses import dataclass, field
from typing import List, Dict, Union, Callable, Literal
from pathlib import Path
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

# Configure Matplotlib for LaTeX rendering
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.size": 14,
    "text.latex.preamble": r"\usepackage{amsmath}"
})


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


def parse_file(file_path):
    data_blocks = []
    labels = []
    current_block = []
    comment_block = []
    last_label = None
    data_mode = False

    with open(file_path, 'r') as file:
        for line in file:
            line = line.strip()

            if not line:  # ignore blank lines
                continue

            if line.startswith("#"):  # comment block
                data_mode = False
                comment_block.append(line[1:].strip())
            else:
                if not data_mode and comment_block:
                    extracted_label = extract_description(comment_block, last_label)
                    if extracted_label:
                        labels.append(extracted_label)
                        last_label = extracted_label

                        if current_block:
                            data_blocks.append(process_data_block(current_block))
                        current_block = []
                    comment_block = []

                data_mode = True  # reading data
                current_block.append(line)

    if current_block:
        data_blocks.append(process_data_block(current_block))

    data_blocks, labels = fix_backward_blocks(data_blocks, labels)

    combined_data = combine_data_with_labels(data_blocks, labels)

    return labels, data_blocks, combined_data


def extract_description(comment_block, last_label):
    for line in comment_block:
        if "Spec forward" in line:
            return "forward"
        elif "Spec backward" in line:
            return "backward"
        elif "Spec fwd pause" in line:
            if last_label == "forward":
                return "forward pause"
            elif last_label == "backward":
                return "backward pause"
            else:
                return "pause"

    return None


def process_data_block(block):
    processed_block = [list(map(float, row.split(";"))) for row in block]
    return np.array(processed_block)

def fix_backward_blocks(data_blocks, labels):
    for i, label in enumerate(labels):
        if label in ["backward", "backward pause"]:
            data_blocks[i] = data_blocks[i][::-1]

    return data_blocks, labels

def combine_data_with_labels(data_blocks, labels):
    label_map = {
        "forward": "f",
        "forward pause": "fp",
        "backward": "b",
        "backward pause": "bp"
    }

    labeled_data = []

    for block, label in zip(data_blocks, labels):
        section_marker = label_map[label]
        section_column = np.array([[section_marker]] * block.shape[0], dtype=object)
        labeled_block = np.hstack((block.astype(np.float64), section_column))
        labeled_data.append(labeled_block)

    return np.vstack(labeled_data) if labeled_data else np.empty((0, data_blocks.shape[1] + 1), dtype=object)




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
        self.deleted = []

    def _load_file_afm_calib(self, path: Path) -> List[Dict]:
        """Internal method to load data from a single file."""

        metadata = parse_metadata(path)
        labels, data_sections, combined_data = parse_file(path)

        z1 = combined_data[:, 0]
        voltage = combined_data[:, 1]
        labels = combined_data[:, -1]

        name = "Image_" + str(path).split('Image_')[-1].split("_")[0]
        print(name)

        curves = []
        curve_dict = {
            "raw": {
                "force": voltage,
                "deflection": voltage,
                "z": z1,
                "time": np.zeros(len(voltage)),
                "labels": labels
            },
            "metadata": {
                "file": str(path),
                "name": name,
                "index": int(str(path).split("_")[-1].split(".")[0])
            }
        }
    
        curves.append(curve_dict)
            
        return curves


    def _load_file_afm(self, path: Path) -> List[Dict]:
        """Internal method to load data from a single file."""

        metadata = parse_metadata(path)
        labels, data_sections, combined_data = parse_file(path)

        z1 = combined_data[:, 2] # 0 or 2
        voltage = combined_data[:, 1]

        defl_sens = metadata["Deflection-Sensitivity"]
        k = metadata["Spring-Constant"]
        
        d_load = metadata["Deflection-Sensitivity"] * voltage   # deflection of the cantilever in m
        force = metadata["Spring-Constant"] * d_load            # deflection of the cantilever in N
        w = z1 - d_load

        #print(f"deflection sensitivity: {defl_sens}")
        #print(f"spring constant: {k}")

        #name = "Image" + str(path).split('Image')[-1].split(".txt")[0]
        name2 = "Image_" + str(path).split('Image_')[-1].split("_")[0]

        name = "Image_" + str(path).split("\\")[-1].split('.txt')[0]
        
        curves = []
        curve_dict = {
            "raw": {
                "z_piezo": z1 * 1e6,
                "deflection": d_load * 1e6,
                "force": force * 1e6,
                "z": w * 1e6,
                "time": np.zeros(len(force)),
                "labels": combined_data[:, -1],
                "keep": True
            },
            "metadata": {
                "file": str(path),
                "name": name,
                "index": int(str(path).split("_")[-1].split(".")[0])
            }
        }

        curves.append(curve_dict)

        return curves
        
    
    def _load_file_fluidfm(self, path: Path) -> List[Dict]:
        """Internal method to load data from a single file."""

        metadata = parse_metadata(path)
        labels, data_sections, combined_data = parse_file(path)

        z1 = combined_data[:, 0]
        voltage = combined_data[:, 1]

        #forward, backward = parse_file2(path)

        #z1 = forward[:, 0]
        #voltage = forward[:, 1]
  
        d_load = metadata["Deflection-Sensitivity"] * voltage 
        force = metadata["Spring-Constant"] * d_load 
        w = z1 - d_load

        # retraction curve
        #z1_retract = backward[:, 0]
        #voltage_retract = backward[:, 1]

        #d_load_retract = metadata["Deflection-Sensitivity"] * voltage_retract
        #force_retract = metadata["Spring-Constant"] * d_load_retract
        #w_retract = z1_retract - d_load_retract

        name = "Image_" + str(path).split("\\")[-1].split("_")[0]
        
        curves = []
        curve_dict = {
            "raw": {
                "z_piezo": z1 * 1e6,
                "force": force * 1e6,
                "z": w * 1e6,
                "time": np.zeros(len(force)),
                "deflection": d_load * 1e6,
                "labels": combined_data[:, -1]
            },
            "metadata": {
                "file": str(path),
                "name": name,
                "index": int(str(path).split("_")[-1].split(".")[0])
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

            name = "Grid_" + str(path).split("\\")[-1].split(".")[0]
            print(name)
            print(_)
            
            # Create dictionary for current curve
            curve_dict = {
                "raw": {
                    "force": curve_data['f'].values,
                    "deflection": np.zeros(len(curve_data['f'])),
                    "z": -curve_data['z'].values,
                    "time": curve_data['t'].values,
                },
                "metadata": {
                    "file": str(path),
                    "name": name,
                    "index": _
                }
            }
            curves.append(curve_dict)
            
        return curves


    def _load_file_csv(self, path: Path) -> List[Dict]:
        """Internal method to load data from a single file."""

        imported = pd.read_csv(
            path,
            skiprows=1,
            names=["U", "F"]
        )

        #,sep=r"\s+"

        print(imported)

        z1 = imported["U"]
        force = imported["F"]

        curves = []
        curve_dict = {
            "raw": {
                "force": -force,
                "z": -z1,
                "deflection": -z1,
                "time": np.zeros(len(force))
            },
            "metadata": {
                "file": str(path), 
                "name": str(path)
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
        elif self.exp_type == "fluidfm":
            for path in paths:
                new_curves = self._load_file_fluidfm(path)
                self.data.extend(new_curves)
        elif self.exp_type == "csv":
            for path in paths:
                new_curves = self._load_file_csv(path)
                self.data.extend(new_curves)
        else:
            print("Experiment type does not exist. :(")

    
    def __len__(self) -> int:
        """Returns the number of curves."""
        return len(self.data)

    def delete_curves(self, indices):
        """Delete a specific curve by index."""

        for index in indices:
            if index >= len(self):
                raise IndexError(f"Curve index {index} out of range (0-{len(self)-1})")

        paths = []
        for i, item in enumerate(self.data):
            path = item["metadata"]["file"]
            if i in indices:
                paths.append(path)

        for i in indices[::-1]:
            self.data.pop(i)

        if not self.deleted:
            deletedSet = self.deleted_set(paths)
        else:
            self.deleted.append(paths)


    def deleted_set(self, paths):
        deleted = IndentationSet(paths, exp_type = self.exp_type)
        self.deleted = deleted

        return deleted

    def restore_all(self):
        if self.exp_type == "ft":
            self.data = self.deleted.data
            self.deleted = []
        else:
            self.data.extend(self.deleted.data)
            self.deleted = []

    def get_curve(self, index: int) -> Dict:
        """Get a specific curve by index."""
        if index >= len(self):
            raise IndexError(f"Curve index {index} out of range (0-{len(self)-1})")
        return self.data[index]
    
    def process_raw(self, processing_pipeline: List[Callable]):
        """Process all curves using a sequence of functions."""
        for curve in tqdm(self.data):
            processed_data = {
                            "deflection": np.copy(curve["raw"]["deflection"]) if "deflection" in curve["raw"] else None,
                            "z_piezo": np.copy(curve["raw"]["z_piezo"]) if "z_piezo" in curve["raw"] else None,
                            "force": np.copy(curve["raw"]["force"]),
                            "z": np.copy(curve["raw"]["z"]),
                            "time": np.copy(curve["raw"]["time"]) if "time" in curve["raw"] else None,
                            "labels": np.copy(curve["raw"]["labels"]) if "labels" in curve["raw"] else None,
                            "name": np.copy(curve["metadata"]["name"]) if "name" in curve["metadata"] else None,
                            "index": np.copy(curve["metadata"]["index"]) if "index" in curve["metadata"] else None,
                            "file": np.copy(curve["metadata"]["file"]) if "file" in curve["metadata"] else None,
                            "keep": np.copy(curve["raw"]["keep"]) if "keep" in curve["raw"] else None,
                            "contact_point": np.copy(curve["contact_point"]) if "contact_point" in curve else None,
                            "release_point": np.copy(curve["release_point"]) if "release_point" in curve else None,
                            "max_retraction_force": np.copy(curve["max_retraction_force"]) if "max_retraction_force" in curve else None,
                            "max_indentation_force": np.copy(curve["max_indentation_force"]) if "max_indentation_force" in curve else None,
                            "max_indentation_point": np.copy(curve["max_indentation_point"]) if "max_indentation_point" in curve else None,
                            "indentation_depth": np.copy(curve["indentation_depth"]) if "indentation_depth" in curve else None,
                            "youngs_modulus": np.copy(curve["youngs_modulus"]) if "youngs_modulus" in curve else None,
                            "force_change": np.copy(curve["force_change"]) if "force_change" in curve else None,
                            "linear_slope": np.copy(curve["linear_slope"]) if "linear_slope" in curve else None,
                            "linear_indent_slope": np.copy(curve["linear_indent_slope"]) if "linear_indent_slope" in curve else None}
            for func in processing_pipeline:
                processed_data = func(processed_data)
            curve["processed"] = processed_data

    def process_processed(self, processing_pipeline: List[Callable]):
        """Process all curves using a sequence of functions."""
        for curve in tqdm(self.data):
            processed_data = {
                            "deflection": np.copy(curve["processed"]["deflection"]) if "deflection" in curve["processed"] else None,
                            "z_piezo": np.copy(curve["processed"]["z_piezo"]) if "z_piezo" in curve["processed"] else None,
                            "force": np.copy(curve["processed"]["force"]),
                            "z": np.copy(curve["processed"]["z"]),
                            "time": np.copy(curve["processed"]["time"]) if "time" in curve["processed"] else None,
                            "labels": np.copy(curve["processed"]["labels"]) if "labels" in curve["processed"] else None,
                            "index": np.copy(curve["metadata"]["index"]) if "index" in curve["metadata"] else None, 
                            "name": np.copy(curve["metadata"]["name"]) if "name" in curve["metadata"] else None,
                            "file": np.copy(curve["metadata"]["file"]) if "file" in curve["metadata"] else None,
                            "keep": np.copy(curve["processed"]["keep"]) if "keep" in curve["processed"] else None,
                            "contact_point": np.copy(curve["contact_point"]) if "contact_point" in curve else None,
                            "max_retraction_force": np.copy(curve["max_retraction_force"]) if "max_retraction_force" in curve else None,
                            "release_point": np.copy(curve["release_point"]) if "release_point" in curve else None,
                            "max_indentation_force": np.copy(curve["max_indentation_force"]) if "max_indentation_force" in curve else None,
                            "max_indentation_point": np.copy(curve["max_indentation_point"]) if "max_indentation_point" in curve else None,
                            "indentation_depth": np.copy(curve["indentation_depth"]) if "indentation_depth" in curve else None,
                            "youngs_modulus": np.copy(curve["youngs_modulus"]) if "youngs_modulus" in curve else None,
                            "force_change": np.copy(curve["force_change"]) if "force_change" in curve else None,
                            "linear_slope": np.copy(curve["linear_slope"]) if "linear_slope" in curve else None,
                            "linear_indent_slope": np.copy(curve["linear_indent_slope"]) if "linear_indent_slope" in curve else None}
            for func in processing_pipeline:
                processed_data = func(processed_data)
            curve["processed"] = processed_data

    def calculate_curve_parameter(self, function: Callable, **kwargs):
        """Process all curves using a sequence of functions."""
        if "processed" in self.data[0].keys():
            for curve in tqdm(self.data):
                #processed_data = curve["processed"].copy()
                processed_data = {
                            "deflection": np.copy(curve["processed"]["deflection"]) if "deflection" in curve["processed"] else None,
                            "z_piezo": np.copy(curve["processed"]["z_piezo"]) if "z_piezo" in curve["processed"] else None,
                            "force": np.copy(curve["processed"]["force"]),
                            "z": np.copy(curve["processed"]["z"]),
                            "time": np.copy(curve["processed"]["time"]) if "time" in curve["processed"] else None,
                            "labels": np.copy(curve["processed"]["labels"]) if "labels" in curve["processed"] else None,
                            "index": np.copy(curve["metadata"]["index"]) if "index" in curve["metadata"] else None, 
                            "name": np.copy(curve["metadata"]["name"]) if "name" in curve["metadata"] else None,
                            "file": np.copy(curve["metadata"]["file"]) if "file" in curve["metadata"] else None,
                            "keep": np.copy(curve["processed"]["keep"]) if "keep" in curve["processed"] else None,
                            "contact_point": np.copy(curve["contact_point"]) if "contact_point" in curve else None,
                            "max_retraction_force": np.copy(curve["max_retraction_force"]) if "max_retraction_force" in curve else None,
                            "release_point": np.copy(curve["release_point"]) if "release_point" in curve else None,
                            "max_indentation_force": np.copy(curve["max_indentation_force"]) if "max_indentation_force" in curve else None,
                            "max_indentation_point": np.copy(curve["max_indentation_point"]) if "max_indentation_point" in curve else None,
                            "indentation_depth": np.copy(curve["indentation_depth"]) if "indentation_depth" in curve else None,
                            "youngs_modulus": np.copy(curve["youngs_modulus"]) if "youngs_modulus" in curve else None,
                            "force_change": np.copy(curve["force_change"]) if "force_change" in curve else None,
                            "linear_slope": np.copy(curve["linear_slope"]) if "linear_slope" in curve else None,
                            "linear_indent_slope": np.copy(curve["linear_indent_slope"]) if "linear_indent_slope" in curve else None}
                
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
        z_reference = np.sort(reference_curve["z"].astype(float))  # Ensure z is monotonic
        
        # Initialize array to store interpolated forces
        interpolated_forces = np.zeros((len(valid_curves), len(z_reference)))
        
        # Interpolate all curves to match the reference z values
        for i, curve in enumerate(valid_curves):
            # Sort z and force arrays together to ensure monotonic z
            sort_idx = np.argsort(curve["z"])
            z_sorted = curve["z"][sort_idx].astype(float)
            f_sorted = curve["force"][sort_idx].astype(float)
            
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
             only_good: bool = False,
             figsize=(12, 6), 
             linestyle="-",
             marker=".",
             show_title=True,
             show_legend=True,
             name="test",
             show=True,
             colors=None,
             ax=None,
             units="micro",
             **kwargs):  # Add ax as an optional parameter
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

        filename = ""
        # Plot each curve
        for i, idx in enumerate(indices_to_plot):
            curve = self.get_curve(idx)
            data = curve.get("processed" if use_processed and "processed" in curve else "raw")
            metadata = curve.get("metadata")
            filename = metadata["file"]
            
            if only_good and data["keep"]: 
                if units == "micro":
                    ax.plot(
                        data["z"],
                        data["force"],
                        color=colors[i] if isinstance(colors, np.ndarray) else None,
                        linewidth=2,
                        linestyle=linestyle,
                        marker=marker,
                        label=os.path.basename(metadata["file"]).split(".")[0] + "_" + f'{idx+1}',
                        **kwargs
                    )
                elif units == "nano":
                    ax.plot(
                        data["z"] * 1e3,
                        data["force"] * 1e3,
                        color=colors[i] if isinstance(colors, np.ndarray) else None,
                        linewidth=2,
                        linestyle=linestyle,
                        marker=marker,
                        label=os.path.basename(metadata["file"]).split(".")[0] + "_" + f'{idx + 1}',
                        **kwargs
                    )
            elif not only_good: 
                if units == "micro":
                    ax.plot(
                        data["z"],
                        data["force"],
                        color=colors[i] if isinstance(colors, np.ndarray) else None,
                        linewidth=2,
                        linestyle=linestyle,
                        marker=marker,
                        label=os.path.basename(metadata["file"]).split(".")[0] + "_" + f'{idx+1}',
                        **kwargs
                    )
                elif units == "nano":
                    ax.plot(
                        data["z"] * 1e3,
                        data["force"] * 1e3,
                        color=colors[i] if isinstance(colors, np.ndarray) else None,
                        linewidth=2,
                        linestyle=linestyle,
                        marker=marker,
                        label=os.path.basename(metadata["file"]).split(".")[0] + "_" + f'{idx + 1}',
                        **kwargs
                    )

        
        # Add labels and title
        if units == "micro":
            ax.set_xlabel(r'tip-sample separation [$\mu$m]')
            ax.set_ylabel(r'force [$\mu$N]')
        elif units == "nano":
            ax.set_xlabel(r'tip-sample separation [nm]')
            ax.set_ylabel(r'force [nN]')
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


        folder_name = os.path.dirname(filename) + "\\"
        print(folder_name) 
        print(str(folder_name) + "output\\test.jpg")
        folder_name = str(folder_name) + "\\output\\"
        
        os.makedirs(folder_name, exist_ok=True)
        
        fig.savefig(str(folder_name) + str(name) + ".jpg")
        print(str(folder_name) + str(name) + ".jpg")

        
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
        z_neg = np.array(self.mean["z"])
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
