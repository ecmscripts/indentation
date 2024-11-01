import numpy as np
from scipy.optimize import fmin

# def parameter_youngs_modulus(data, radius, nu, cutoff, x0=[0.005, 0], show_plot=False, keyname="youngs_modulus"):
#     '''
#     Calculate Young's modulus from force-displacement data.
#
#     Parameters:
#     -----------
#     data : Dict
#         Must contain 'z' and 'force' keys with numpy array values
#     radius : float
#         Radius in micrometers (must be positive)
#     nu : float
#         Poisson's ratio
#     cutoff : float
#         Cutoff in percent of the radius (~1-100)
#     x0 : list, optional
#         Initial guess for optimization [alpha, beta]
#     show_plot : bool, optional
#         Whether to show debug plots
#     keyname : str, optional
#         Key name for the output
#
#     Returns:
#     --------
#     tuple : (float, str)
#         (Young's modulus, keyname)
#
#     Raises:
#     -------
#     ValueError
#         If inputs are invalid
#     '''
#     # Input validation
#     if not isinstance(data, dict):
#         raise ValueError("data must be a dict")
#
#     if 'z' not in data or 'force' not in data:
#         raise ValueError("data must contain 'z' and 'force' keys")
#
#     # Check if values are numpy arrays
#     if not isinstance(data['z'], np.ndarray) or not isinstance(data['force'], np.ndarray):
#         raise ValueError("'z' and 'force' values must be numpy arrays")
#
#     # Check if arrays are empty
#     if len(data['z']) == 0 or len(data['force']) == 0:
#         raise ValueError("data arrays are empty")
#
#     # Check if arrays have the same length
#     if len(data['z']) != len(data['force']):
#         raise ValueError("'z' and 'force' arrays must have the same length")
#
#     if radius <= 0:
#         raise ValueError("radius must be positive")
#
#     if not (0 <= nu <= 0.5):
#         raise ValueError("Poisson's ratio must be between 0 and 0.5")
#
#     if not (0 < cutoff <= 100):
#         raise ValueError("cutoff must be between 0 and 100")
#
#     def force_function(x, displ):
#         alpha, beta = x
#         return alpha * displ**(3.0/2.0) + beta
#
#     def target_function(x, force, displ):
#         phi = np.dot(force_function(x, displ) - force,
#                      force_function(x, displ) - force)
#         return phi
#
#
#     # Create numpy arrays from the input data
#     displ = -data['z'].copy()  # Displacement remains in µm
#     force = data['force'].copy()  # Force remains in µN
#
#     ix_end = np.argmin(np.abs(displ - (cutoff / 100) * radius))
#
#     # Validate that we have enough data points after cutoff
#     if ix_end < 3:
#         raise ValueError("Not enough data points before cutoff")
#
#     xOpt = fmin(target_function, x0, args=(force[:ix_end], displ[:ix_end]), disp=False)
#     alpha, beta = xOpt
#
#     # Calculate Emod in MPa
#     Emod = np.round((3.0 / 4.0) * (1 - nu**2) * alpha / radius**0.5, 2)
#
#     return Emod, keyname

#
def parameter_youngs_modulus(data, radius, nu, cutoff, x0=[0.005, 0], show_plot=False, keyname="youngs_modulus"):
    '''
    Enter consistent units:
    - radius in µm (micrometers)
    - data["force"] in µN (micronewtons)
    - data["z"] in µm (micrometers)
    - cutoff in percent of the radius (e.g., 1-100)

    Returns:
    - Emod: Young's Modulus in MPa
    - keyname: The key under which Emod is stored in the data dictionary
    '''

    def force_function(x, displ):
        alpha, beta = x
        return alpha * displ ** (1.5) + beta

    def target_function(x, force, displ):
        residuals = force_function(x, displ) - force
        return np.sum(residuals ** 2)

    # Convert radius from µm to mm as in the original code
    radius_mm = radius / 1000.0  # radius in mm

    # Prepare data
    displ_um = -data["z"].copy()   # Displacement in µm (make positive)
    force_uN = data["force"].copy()  # Force in µN

    # Calculate cutoff displacement (in µm)
    cutoff_displ_um = (cutoff / 100.0) * radius_mm * 1000.0  # Convert radius back to µm

    # Find index up to cutoff
    ix_end = np.argmin(np.abs(displ_um - cutoff_displ_um))

    # Ensure there are enough data points
    if ix_end < 3:
        raise ValueError("Not enough data points before cutoff for reliable fitting.")

    # Fit the force-displacement data up to the cutoff point
    xOpt = fmin(target_function, x0, args=(force_uN[:ix_end], displ_um[:ix_end]), disp=False)
    alpha, beta = xOpt

    # Calculate Emod using the simplified formula
    Emod = np.round(23.7170825 * (1 - nu ** 2) * alpha / radius_mm ** 0.5, 2)

    # Optionally, store Emod in the data dictionary under the specified key
    data[keyname] = Emod

    return Emod, keyname
