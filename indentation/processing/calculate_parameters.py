import numpy as np
from scipy.optimize import fmin

import numpy as np
from scipy.optimize import fmin

def parameter_defelection_sensitivity(data, keyname="d_sens"):
    voltage = data["force"]
    displ   = -data["z"]
    displ   = displ - displ[0]
    displ   = 1e9*displ

    d_sens = displ[-1]/voltage[-1]
    print(d_sens)

    return d_sens, keyname
    

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

    # Prepare data
    displ_um = -data["z"].copy()   # Displacement in µm (make positive)
    force_uN = data["force"].copy()  # Force in µN

    # Calculate cutoff displacement (in µm)
    cutoff_displ_um = (cutoff / 100.0) * radius  # Convert radius back to µm

    # Find index up to cutoff
    ix_end = np.argmin(np.abs(displ_um - cutoff_displ_um))

    # Ensure there are enough data points
    if ix_end < 3:
        raise ValueError("Not enough data points before cutoff for reliable fitting.")

    # Fit the force-displacement data up to the cutoff point
    xOpt = fmin(target_function, x0, args=(force_uN[:ix_end], displ_um[:ix_end]), disp=False)
    alpha, beta = xOpt

    # Calculate Emod using the corrected formula
    # Units of alpha: μN / μm^{1.5}
    # Units of sqrt(R): μm^{0.5}
    # Emod (MPa) = [3 * alpha * (1 - nu^2)] / [4 * sqrt(R)]
    # Since alpha / sqrt(R) is in μN / μm^2 = MPa, no unit conversion is needed

    Emod = 1000*(3 * alpha * (1 - nu**2)) / (4 * np.sqrt(radius))  # Emod in MPa

    # Optionally, store Emod in the data dictionary under the specified key
    data[keyname] = Emod

    return Emod, keyname
