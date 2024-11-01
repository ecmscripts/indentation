import numpy as np
from scipy.optimize import fmin

def parameter_youngs_modulus(data, radius, nu, cutoff, x0=[0.005, 0], show_plot=False, keyname="youngs_modulus"):
    '''
    Calculate Young's modulus from force-displacement data.
    
    Parameters:
    -----------
    data : Dict
        Must contain 'z' and 'force' keys with numpy array values
    radius : float
        Radius in micrometers (must be positive)
    nu : float
        Poisson's ratio
    cutoff : float
        Cutoff in percent of the radius (~1-100)
    x0 : list, optional
        Initial guess for optimization [alpha, beta]
    show_plot : bool, optional
        Whether to show debug plots
    keyname : str, optional
        Key name for the output
        
    Returns:
    --------
    tuple : (float, str)
        (Young's modulus, keyname)
        
    Raises:
    -------
    ValueError
        If inputs are invalid
    '''
    # Input validation
    if not isinstance(data, dict):
        raise ValueError("data must be a dict")
    
    if 'z' not in data or 'force' not in data:
        raise ValueError("data must contain 'z' and 'force' keys")
    
    # Check if values are numpy arrays
    if not isinstance(data['z'], np.ndarray) or not isinstance(data['force'], np.ndarray):
        raise ValueError("'z' and 'force' values must be numpy arrays")
    
    # Check if arrays are empty
    if len(data['z']) == 0 or len(data['force']) == 0:
        raise ValueError("data arrays are empty")
    
    # Check if arrays have the same length
    if len(data['z']) != len(data['force']):
        raise ValueError("'z' and 'force' arrays must have the same length")
        
    if radius <= 0:
        raise ValueError("radius must be positive")
        
    if not (0 <= nu <= 0.5):
        raise ValueError("Poisson's ratio must be between 0 and 0.5")
        
    if not (0 < cutoff <= 100):
        raise ValueError("cutoff must be between 0 and 100")
    
    def force_function(x, displ):
        alpha, beta = x
        return alpha * displ**(3.0/2.0) + beta
    
    def target_function(x, force, displ):
        phi = np.dot(force_function(x, displ) - force,
                     force_function(x, displ) - force)
        return phi
    
    fsize = 14
    radius = radius/1000
    # Create numpy arrays from the input data
    displ = -data['z'].copy()  # Make a copy of the numpy array
    force = data['force'].copy()  # Make a copy of the numpy array
    ix_end = np.argmin(np.abs(displ-1000*(cutoff/100)*radius))
    
    # Validate that we have enough data points after cutoff
    if ix_end < 3:
        raise ValueError("Not enough data points before cutoff")
    
    xOpt = fmin(target_function, x0, args=(force[:ix_end], displ[:ix_end]), disp=False)
    alpha, beta = xOpt
    Emod = np.round(1000*3.0/4.0*(1-nu**2)/radius**0.5*alpha * (1000**(3.0/2.0)/1000000), 2)
    
    return Emod, keyname

#
# def parameter_youngs_modulus(data, radius, nu, cutoff, x0=[0.005, 0], show_plot=False, keyname="youngs_modulus"):
#     '''
#     Enter consistent units, R in um, cutoff in percent of the radius (~1-100).
#     Adds the keyword "youngs_modulus" to the data.
#     '''
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
#     fsize = 14
#     radius = radius/1000
#
#     data_copy = data.copy()
#
#     displ = -data_copy["z"]
#     force = data_copy["force"]
#     ix_end = np.argmin(np.abs(displ-1000*(cutoff/100)*radius))
#     xOpt = fmin(target_function, x0, args=(force[:ix_end], displ[:ix_end]), disp=False)
#     alpha, beta = xOpt
#     Emod = np.round(1000*3.0/4.0*(1-nu**2)/radius**0.5*alpha * (1000**(3.0/2.0)/1000000), 2)
#
#     if show_plot:
#         f_fit = force_function(xOpt, displ)
#         plt.figure(figsize=(8, 6))
#         plt.plot(displ, force, lw=4, label="data")
#         plt.plot(displ, f_fit, 'r', label="fitted")
#         plt.title(r'Hertzian fit with R=' + str(int(R*1000)) + r"$\mu$m, and depth=" + str(cutoff) + r"$\mu$m", fontsize=fsize, y=1.01)
#         plt.xlabel(r'displacement [$\mu$m]', fontsize=fsize, labelpad=10)
#         plt.ylabel(r'force [$\mu$N]', fontsize=fsize, labelpad=10)
#         # plt.xticks([], [])
#         # plt.yticks([], [])
#         plt.legend(loc=2, fontsize=fsize, fancybox=True, framealpha=0.0)
#         plt.tick_params(labelsize=fsize, labelcolor='k', pad=10)
#         plt.tight_layout()
#
#         print("\nYoungs Modulus: ", Emod, 'kPa\n')
#
#     return Emod, keyname
