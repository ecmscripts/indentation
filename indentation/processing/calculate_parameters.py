import numpy as np
from scipy.optimize import fmin
from indentation.processing import plotting

def parameter_defelection_sensitivity(data, keyname="d_sens"):
    voltage = data["force"]
    displ   = -data["z"]
    displ   = displ - displ[0]
    displ   = 1e9*displ

    d_sens = displ[-1]/voltage[-1]
    print(d_sens)

    return d_sens, keyname
    

def parameter_youngs_modulus(data, radius, nu, cutoff, x0=[50000], show_plot=False, keyname="youngs_modulus"):
    def hertzian_force(E, R, nu, z):
        hertz_F = 4.0 / 3.0 * E * np.sqrt(R) / (1 - nu ** 2) * np.power(z, 3.0 / 2.0)

        return hertz_F

    def sse(param, R, nu, F, z, cutoff):
        E = param

        hertz_F = hertzian_force(E, R, nu, z)
        sse_value = np.sum((force - hertz_F)**2.0)

        return sse_value

    F = data["force"].copy()  # Force in µN (make positive)
    disp = -data["z"].copy()  # Displacement in µm

    ix = np.where(disp > (cutoff / 100) * radius)[0]

    if len(ix) == 0:
        ix = len(F)
    else:
        ix = ix[0]

    force = F[:ix]
    z = disp[:ix]

    result = fmin(sse, x0, args=(radius, nu, force, z, cutoff), disp=False)

    E_mod = result[0]*1000

    hertz_F = hertzian_force(result[0], radius, nu, z)

    if show_plot:
        plotting.plot_hertzian_fit(F, disp, hertz_F, z, E_mod)

    data[keyname] = E_mod


    data[keyname] = {"value"}


    return E_mod, keyname


def parameter_youngs_modulus_2(data, radius, nu, cutoff, x0=[0.005, 0], show_plot=False, keyname="youngs_modulus"):
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
    print(Emod)

    data[keyname] = {"value"}

    return Emod, keyname
