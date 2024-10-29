import numpy as np
from scipy.optimize import fmin

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

    data_copy = data.copy()

    displ = -data_copy["z"]
    force = data_copy["force"]
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
