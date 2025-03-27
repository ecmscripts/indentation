import numpy as np
from scipy.optimize import fmin
from indentation.processing import plotting
from matplotlib import pyplot as plt

def parameter_defelection_sensitivity(data, keyname="d_sens"):
    voltage = data["force"]
    displ   = data["z"]
    displ   = displ - displ[0]
    #displ   = 1e9*displ

    ix_end = int(len(voltage) / 2.0)

    d_sens = displ[ix_end]/voltage[ix_end]
    print(d_sens)

    r_2 = calculate_r_squared(displ[:ix_end], voltage[:ix_end] * d_sens)

    print(f"r_square: {r_2}")
    print(f"d_sens: {d_sens}")

    return [d_sens, r_2], keyname


def hertzian_force(E, R, nu, z):
    hertz_F = 4.0 / 3.0 * E * np.sqrt(R) / (1 - nu ** 2) * np.sign(z) * np.power(np.abs(z), 3.0 / 2.0)
    
    return hertz_F


def parameter_youngs_modulus_lstsq(data, radius, nu, cutoff, x0=[500000], show_plot=False, keyname="youngs_modulus"):
    def sse(param, R, nu, F, z, cutoff):
        E = param

        hertz_F = hertzian_force(E, R, nu, z)
        sse_value = np.sum((F - hertz_F)**2.0)
        
        return sse_value
    
    F = data["force"].copy()  # Force in µN 
    disp = data["z"].copy()   # Displacement in µm
    
    ix = np.where(disp > (cutoff / 100) * radius)[0]
    
    if len(ix) == 0:
        ix = len(F)
    else:
        ix = ix[0]

    force = F[:ix]
    z = disp[:ix]
    
    A = np.zeros((len(z), 2))
    A[:,0] = np.sign(z) * np.power(np.abs(z), 3.0 / 2.0)
    A[:,1] = 1.0
    b = np.array(force, dtype=float)
    
    b_norm = np.linalg.norm(b)  # Compute norm of b
    b_normalized = b / b_norm  # Normalize b

    A_norms = np.linalg.norm(A, axis=0)  # Save norms before normalizing
    A_normalized = A / A_norms
    
    result_normalized = np.linalg.lstsq(A_normalized, b_normalized, rcond=None)[0]
    result = (result_normalized / A_norms) * b_norm # Un-normalize

    print(result)

    x = np.linspace(0,0.5,100)
    y = result[0] * x + result[1]
    alpha = result[0]

    # plt.figure()
    # plt.plot(np.sign(z) * np.power(np.abs(z), 3.0 / 2.0), force, 'c-')
    # plt.plot(x, y, 'm-')
    # plt.show()
    
    E_app = 3.0 / 4.0 * alpha * (1 - nu * nu) / np.sqrt(radius)
    print(f"least squares E: {E_app*1e3}")

    hertz_F_lstsq = hertzian_force(E_app, radius, nu, z) + result[1]
    force_2 = force + result[1]
    
    result = fmin(sse, x0, args=(radius, nu, force, z, cutoff), disp=True)
    
    E_mod = result[0]*1e3 #*1e6 #*1000
    print(f"fmin E: {E_mod}")
    
    hertz_F = hertzian_force(result[0], radius, nu, z)

    r_2 = calculate_r_squared(force, hertz_F)
    r_2_lstsq = calculate_r_squared(force, hertz_F_lstsq)
    
    plt.figure()
    plt.plot(z, force, 'k-')
    plt.plot(z, hertz_F_lstsq, 'm-')
    plt.plot(z, hertz_F, 'c-')
    plt.xlabel("Indentation depth [um]")
    plt.ylabel("Force [uN]")
    plt.legend(["Data", f"lstsq: E = {round(E_app*1e3, 2)} kPa, r_2 = {round(r_2_lstsq, 2)}", f"fmin: E = {round(E_mod, 2)} kPa, r_2 = {round(r_2, 2)}"])
    plt.show()

    if show_plot:
        plotting.plot_hertzian_fit(F, disp, hertz_F, z, E_mod, r_2)

    data[keyname] = [E_mod, r_2]

    return [E_mod, r_2], keyname



def parameter_youngs_modulus(data, radius, nu, cutoff, x0=[500], show_plot=False, keyname="youngs_modulus"):
    def sse(param, R, nu, F, z, cutoff):
        E = param

        hertz_F = hertzian_force(E, R, nu, z)
        sse_value = np.sum((F - hertz_F)**2.0)
        
        return sse_value
    
    F = data["force"].copy() * 1e3  # Force in µN (make positive)
    disp = data["z"].copy() * 1e3 # Displacement in µm

    #print(F)
    #print(disp)
    
    ix = np.where(disp > (cutoff / 100) * radius)[0]

    #print(ix)
    
    if len(ix) == 0:
        ix = len(F)
    else:
        ix = ix[0]

    force = F[:ix]
    z = disp[:ix]

    result = fmin(sse, x0, args=(radius, nu, force, z, cutoff), disp=True)
    print(result)
    
    E_mod = result[0]*1e6 #*1e6 #*1000

    hertz_F = hertzian_force(result[0], radius, nu, z)
    #print(hertz_F)

    r_2 = calculate_r_squared(F[:ix], hertz_F)

    if show_plot:
        plotting.plot_hertzian_fit(F, disp, hertz_F, z, E_mod, r_2)

    data[keyname] = [E_mod, r_2]
    #data[keyname] = {"value"}

    return [E_mod, r_2], keyname


def parameter_r_squared(data, radius, nu, cutoff, x0=[5000], keyname="r_squared"):
    def sse(param, R, nu, F, z, cutoff):
        E = param

        hertz_F = hertzian_force(E, R, nu, z)
        sse_value = np.sum((force - hertz_F)**2.0)
        
        return sse_value
    
    F = data["force"].copy()  # Force in µN (make positive)
    disp = data["z"].copy()  # Displacement in µm

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

    r_2 = calculate_r_squared(F[:ix], hertz_F)

    data["r_squared"] = r_2
    data["r_squared"] = {"value"}
    
    return r_2, keyname


def calculate_r_squared(y_actual, y_fitted):
    """
    Calculate the coefficient of determination (R^2) between actual and fitted data.

    Parameters:
        y_actual (array-like): The actual data points.
        y_fitted (array-like): The fitted (predicted) data points.

    Returns:
        float: The R^2 value.
    """
    y_actual = np.array(y_actual)
    y_fitted = np.array(y_fitted)

    # Calculate the mean of actual values
    y_mean = np.mean(y_actual)

    # Calculate SS_res and SS_tot
    ss_res = np.sum((y_actual - y_fitted) ** 2)
    ss_tot = np.sum((y_actual - y_mean) ** 2)

    if ss_tot == 0:
        return 0

        # Calculate R^2
    r_squared = 1 - (ss_res / ss_tot)

    # print("percent deviation")
    # percent_dev = []
    # for i, val in enumerate(y_actual):
    #     if val != 0:
    #         percent_dev.append(np.abs((y_actual[i] - y_fitted[i])) / y_actual[i] * 100.0)
    #
    # mean_percent_dev = np.mean(percent_dev)
    # print(mean_percent_dev)
    #
    # print("RMSE")
    # RMSE = np.sqrt(np.mean((y_actual - y_fitted)**2.0))
    # print(RMSE*1e3)
    #
    # print("R_squared")
    # print(r_squared)

    return r_squared


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


def parameter_youngs_modulus_log(data, radius, nu, cutoff, x0=[500000], show_plot=False, keyname="youngs_modulus"):
    
    def sse(param, R, nu, F, z, cutoff):
        E = param

        hertz_F = hertzian_force(E, R, nu, z)
        sse_value = np.sum((F - hertz_F)**2.0)
        
        return sse_value
    
    F = data["force"].copy()  # Force in µN (make positive)
    disp = data["z"].copy()   # Displacement in µm

    F_orig = F.copy()
    disp_orig = disp.copy()

    near_zero = F[0:50]
    mean_near_zero = np.mean(near_zero)
    F_new = F - mean_near_zero
    F_new[0] = 0

    sub_zero = np.where(F_new <= 0.0)

    sub_zero_i = 0
    if len(sub_zero) > 0:
      sub_zero_i = sub_zero[0][-1] + 1

    F_new_2 = F_new[sub_zero_i:]
    F_new_2[0] = 0.0
    disp_2 = disp[sub_zero_i:]
    disp_2[0] = 0.0

    plt.figure()
    plt.plot(disp, F, 'k-')
    plt.plot(disp, F_new, 'r-')
    plt.plot(disp_2, F_new_2, 'm-')
    plt.show()

    F = np.array(F_new_2, dtype='float')
    disp = np.array(disp_2, dtype='float')
    
    ix = np.where(disp > (cutoff / 100.0) * radius)[0]
    
    if len(ix) == 0:
        ix = len(F)
    else:
        ix = ix[0]

    force = F[:ix]
    z = disp[:ix]

    ix_orig = np.where(disp_orig > (cutoff / 100.0) * radius)[0]

    if len(ix_orig) == 0:
        ix_orig = len(F_orig)
    else:
        ix_orig = ix_orig[0]

    print(ix_orig)
    force_orig = F_orig[:ix_orig]
    z_orig = disp_orig[:ix_orig]

    # set to 0 if wanting slope to be a parameter
    fixed_slope = 1
    
    if fixed_slope:
        # fix slope at 3/2
        A = np.zeros((len(z), 1))
        A[:,0] = 1.0
        b = np.log(force[1:]) - 3.0 / 2.0 * np.log(z[1:])
    else:
        # let slope be solved for
        A = np.zeros((len(z), 2))
        A[:,0] = np.log(z)
        A[:,1] = 1.0
        b = np.log(force[1:])


    nan_indices = np.where(np.isnan(b))[0]  # Find indices of NaNs
    clean_arr = b[~np.isnan(b)]
    b= clean_arr
    b_norm = np.linalg.norm(b)  # Compute norm of b
    b_normalized = b / b_norm  # Normalize b

    A = A[1:]
    A = np.delete(A, nan_indices, axis=0)

    A_norms = np.linalg.norm(A, axis=0)  # Save norms before normalizing
    A_normalized = A / A_norms
    
    # print("Any NaN in A:", np.isnan(A).any())
    # print("Any Inf in A:", np.isinf(A).any())
    # print("Any NaN in b:", np.isnan(b).any())
    # print("Any Inf in b:", np.isinf(b).any())

    #print(b)
    
    result_normalized = np.linalg.lstsq(A_normalized, b_normalized, rcond=None)[0]
    result = (result_normalized / A_norms) * b_norm # Un-normalize

    #print(result_normalized)
    print(result)

    x = np.linspace(-10,10,100)
    
    if fixed_slope:
        # fix slope at 3/2
        y = 3.0 / 2.0 * x + result[0]
        alpha = np.exp(result[0])
    else:
        #let slope be solved for
        y = result[0] * x + result[1]
        alpha = np.exp(result[1])


    plt.figure()
    plt.plot(np.log(z), np.log(force), 'c-')
    plt.plot(x, y, 'm-')
    plt.show()

    print(result)
    
    E_app = 3.0 / 4.0 * alpha * (1 - nu * nu) / np.sqrt(radius)
    print(f" least squares E_app: {E_app*1e3}")

    z_plot = np.linspace(0.0, z[-1], 1000)
    
    hertz_F_lstsq = hertzian_force(E_app, radius, nu, z_plot)
    

    result = fmin(sse, x0, args=(radius, nu, force, z, cutoff), disp=False)
    
    result_orig = fmin(sse, x0, args=(radius, nu, force_orig, z_orig, cutoff), disp=False)

    E_mod_orig = result_orig[0] * 1e3
    
    E_mod = result[0]*1e3 #*1e6 #*1000
    print(f"fmin Emod: {E_mod}")
    print(f"fmin Emod orig: {E_mod_orig}")
    
    hertz_F = hertzian_force(result[0], radius, nu, z)
    hertz_F_orig = hertzian_force(result_orig[0], radius, nu, z_orig)

    plt.figure()
    plt.plot(z, force, 'k-')
    plt.plot(z_orig, force_orig, 'k--')
    plt.plot(z_plot, hertz_F_lstsq, 'm-')
    plt.plot(z, hertz_F, 'c-')
    plt.plot(z_orig, hertz_F_orig, 'c--')
    plt.xlabel("Indentation depth [um]")
    plt.ylabel("Force [uN]")
    if fixed_slope:
        plt.legend(["Data", "Original Data", "Lstsq method - ln (fixed slope)", "Fmin method", "Fmin method - original"])
    else:
        plt.legend(["Data", "Original Data", "Lstsq method - ln (free slope)", "Fmin method", "Fmin method - original"])
    plt.show()
    
    r_2 = calculate_r_squared(F[:ix], hertz_F)

    if show_plot:
        plotting.plot_hertzian_fit(F, disp, hertz_F, z, E_mod, r_2)

    data[keyname] = [E_mod, r_2]

    return [E_mod, r_2], keyname
