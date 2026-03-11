import numpy as np
from scipy.optimize import fmin
from indentation.processing import plotting
from matplotlib import pyplot as plt
import os
from pathlib import Path

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


def hertzian_force(E, R, nu, z, n=3.0/2.0):
    hertz_F = 4.0 / 3.0 * E * np.sqrt(R) / (1 - nu ** 2) * np.sign(z) * np.power(np.abs(z), n)
    
    return hertz_F

def parameter_approach_line(data):
    F = data["force"].copy()
    disp = data["z"].copy()

    A = np.zeros((len(disp), 2))
    A[:, 0] = disp
    A[:, 1] = 1.0
    b = np.array(F, dtype=float)
    return 0


def hertzian_force_hollow_cylinder(E, thickness, nu, z):
    hertz_F = 2.0 * E / (1 - nu ** 2) * z * thickness
    
    return hertz_F


def parameter_youngs_modulus_cylinder_lstsq(data, thickness, nu, cutoff, x0=[500000], show_plot=False, keyname="youngs_modulus"):
    def sse(param, thickness, nu, F, z, cutoff):
        E = param

        hertz_F = hertzian_force_hollow_cylinder(E, thickness, nu, z)
        sse_value = np.sum((F - hertz_F)**2.0)
        
        return sse_value
    
    F = data["force"].copy()  # Force in µN 
    disp = data["z"].copy()   # Displacement in µm
    
    ix = np.where(disp > (cutoff / 100) * thickness)[0]
    
    if len(ix) == 0:
        ix = 0 #####len(F)
        E_lstsq = 0.0
        r_2_lstsq = 0.0
        print("not enough indentation depth")
    else:
        ix = ix[0]

        force = F[:ix]
        z = disp[:ix]
        
        A = np.zeros((len(z), 2))
        A[:,0] = z
        A[:,1] = 1.0
        b = np.array(force, dtype=float)
        
        b_norm = np.linalg.norm(b)  # Compute norm of b
        b_normalized = b / b_norm  # Normalize b
    
        A_norms = np.linalg.norm(A, axis=0)  # Save norms before normalizing
        A_normalized = A / A_norms
        
        result_normalized = np.linalg.lstsq(A_normalized, b_normalized, rcond=None)[0]
        result = (result_normalized / A_norms) * b_norm # Un-normalize
    
        x = np.linspace(0,0.5,100)
        y = result[0] * x + result[1]
        alpha = result[0]
        
        E_lstsq = 1.0 / 2.0 * alpha * (1 - nu * nu) / thickness
        #print(f"least squares E: {E_lstsq*1e3}")
    
        hertz_F_lstsq = hertzian_force_hollow_cylinder(E_lstsq, thickness, nu, z) + result[1]
        force_2 = force + result[1]
        
        result = fmin(sse, x0, args=(thickness, nu, force, z, cutoff), disp=False)
        
        E_mod = result[0]*1e3 #*1e6 #*1000
        #print(f"fmin E: {E_mod}")
        
        hertz_F = hertzian_force_hollow_cylinder(result[0], thickness, nu, z)
    
        r_2 = calculate_r_squared(force, hertz_F)
        r_2_lstsq = calculate_r_squared(force, hertz_F_lstsq)
    
        if show_plot:
            plotting.plot_hertzian_fit(F, disp, hertz_F_lstsq, z, E_lstsq*1e3, r_2_lstsq)
    
            print(f"fmin E: {E_mod}")
            print(f"least squares E: {E_lstsq*1e3}")
            
            plt.figure()
            plt.plot(z, force, 'k-')
            plt.plot(z, hertz_F_lstsq, 'm-')
            plt.plot(z, hertz_F, 'c-')
            plt.xlabel("Indentation depth [um]")
            plt.ylabel("Force [uN]")
            plt.legend(["Data", f"lstsq: E = {round(E_lstsq*1e3, 2)} kPa, r_2 = {round(r_2_lstsq, 2)}", f"fmin: E = {round(E_mod, 2)} kPa, r_2 = {round(r_2, 2)}"])
            plt.show()

    data[keyname] = [E_lstsq*1e3, r_2_lstsq]

    print(E_lstsq*1e3)

    return [E_lstsq*1e3, r_2_lstsq], keyname


def get_balanced_split_indices(array_length, N):
    base_size = array_length // N
    remainder = array_length % N

    indices = []
    start = 0

    for i in range(N):
        size = base_size + 1 if i < remainder else base_size
        end = start + size - 1
        indices.append([start, end])
        start = end + 1

    return np.array(indices)


def get_slope_parameters(x, y):
    A = np.zeros((len(x), 2))
    A[:,0] = x
    A[:,1] = 1.0
    b = np.array(y, dtype=float)
    
    b_norm = np.linalg.norm(b)  # Compute norm of b
    b_normalized = b / b_norm  # Normalize b

    A_norms = np.linalg.norm(A, axis=0)  # Save norms before normalizing
    A_normalized = A / A_norms
    
    result_normalized = np.linalg.lstsq(A_normalized, b_normalized, rcond=None)[0]
    result = (result_normalized / A_norms) * b_norm # Un-normalize
    
    m_lstsq = result[0]
    b_lstsq = result[1]

    return m_lstsq, b_lstsq

    

def parameter_approach_slope_lstsq(data, keyname="linear_slope", show_plot=False):    
    force = data["force"].copy()  # Force in µN 
    disp = data["z"].copy()   # Displacement in µm

    seg_length = 1000 #--> change back to this for hydrogels
    #seg_length = 100 # for PDMS this is better
    n = len(force)
    print(f"N: {max(1, int(n/seg_length))}")
    N = max(1, int(n/seg_length))

    print(f"n: {n}")
    if n <= 100:
        print("********")
        print(n)
        print("APPROACH is too short")
        print("********")
    
    split_indices = get_balanced_split_indices(n, N)
    # print(split_indices)
    # print(split_indices[0])

    slope_param = []
    plt.figure()
    plt.plot(disp, force, 'k-')
    plt.xlabel("Tip-sample separation [um]")
    plt.ylabel("Force [uN]")

    max_slope = 0.0
    stdev_arr = []
    slope_arr = []
    for i in range(0, N):
        ixs = split_indices[i]
        ix_start = ixs[0]
        ix_end = ixs[1]

        if ix_end <= ix_start:
            print("*********")
            print("ZERO LENGTH INTERVAL")
            print("*********")
            slope_arr = [0]
        else:
            m, b = get_slope_parameters(disp[ix_start:ix_end], force[ix_start:ix_end])
            slope_arr.append(m)
    
            if np.abs(m) > max_slope:
                max_slope = m
            
            force_line = m * disp[ix_start:ix_end] + b
    
            r_2 = calculate_r_squared(force[ix_start:ix_end], force_line)
            rmse = np.sqrt(np.mean((force[ix_start:ix_end] - force_line)**2))
    
            stdev_arr.append(np.std(force[ix_start:ix_end])*1e6)
            slope_param.append([m, b, r_2, rmse])
            print(f"slope: {m*1e6} pN/um")
            print(f"intercept: {b*1e6} pN")
            print(f"R^2: {r_2}")
            print(f"RMSE: {rmse*1e6}") # pN
            print(f"stdev: {np.std(force[ix_start:ix_end])*1e6} pN")
            print()

            if i % 2 == 0:
                plt.plot(disp[ix_start:ix_end], force_line, 'm-')
            else:
                plt.plot(disp[ix_start:ix_end], force_line, 'c-')

    #print(stdev_arr)
    stdev_arr = np.array(stdev_arr)
    ratios = stdev_arr[1:] / stdev_arr[:-1]
    #print(f"ratios: {ratios}")
    
    print(f"maximum slope: {max_slope*1e6} pN/um")
    if np.abs(max_slope*1e6) > 100.0:
        print("DISCARD CURVE")
        print()
    m_lstsq, b_lstsq = get_slope_parameters(disp, force)

    force_line = m_lstsq * disp + b_lstsq
    print(f"least squares slope: {m_lstsq}")
    print(f"least squares intercept: {b_lstsq}")

    plt.plot(disp, force_line, 'g-')
    # print(slope_param)
    plt.show()

    plt.figure()
    plt.plot(ratios)
    plt.show()
    
    r_2_lstsq = calculate_r_squared(force, force_line)

    if show_plot:  
        plt.figure()
        plt.plot(disp, force, 'k-')
        plt.plot(disp, force_line, 'm-')
        plt.xlabel("Tip-sample separation [um]")
        plt.ylabel("Force [uN]")
        plt.legend(["Data", f"lstsq: m = {round(m_lstsq*1e3, 3)} nN/um, r_2 = {round(r_2_lstsq, 2)}"])
        plt.show()

    data[keyname] = [m_lstsq, r_2_lstsq]

    return [m_lstsq, r_2_lstsq, slope_arr[0], slope_arr[-1], max_slope], keyname


def parameter_youngs_modulus_lstsq(data, radius, nu, cutoff, n=3.0/2.0, x0=[500000], show_plot=False, keyname="youngs_modulus"):
    def sse(param, R, nu, F, z, cutoff):
        E = param

        hertz_F = hertzian_force(E, R, nu, z)
        sse_value = np.sum((F - hertz_F)**2.0)
        
        return sse_value

    if data["name"] is not None:
        name = data["name"].copy()
        print()
        print(f"NAME: {name}")

        if data["index"] is not None:
            index = data["index"].copy()
            print(f"Index: {index}")
        print()
        
    F = data["force"].copy()  # Force in µN 
    disp = data["z"].copy()   # Displacement in µm
    
    ix = np.where(disp > (cutoff / 100) * radius)[0]
    
    if len(ix) == -1: ## change back to 0 after
        E_lstsq = 0.0
        r_2_lstsq = 0.0
        keep = False
        print("not enough indentation depth")
    else:
        if len(ix) == 0:
            ix = len(F)
        else:
            ix = ix[0]

        #ix = ix[0]
        
        force = F[:ix]
        z = disp[:ix]

        print(f"length of signal!: {len(force)}")
        print(len(force) <= 10)
        if len(force) <= 10:
            print("not enough indentation depth -- 10 points or fewer")
            return [0.0, 0.0, False], keyname

        
        A = np.zeros((len(z), 2))
        A[:,0] = np.sign(z) * np.power(np.abs(z), n)
        A[:,1] = 1.0
        b = np.array(force, dtype=float)
        
        b_norm = np.linalg.norm(b)  # Compute norm of b
        b_normalized = b / b_norm  # Normalize b
    
        A_norms = np.linalg.norm(A, axis=0)  # Save norms before normalizing
        A_normalized = A / A_norms
        
        result_normalized = np.linalg.lstsq(A_normalized, b_normalized, rcond=None)[0]
        # print(f"Normalized results: {result_normalized}")
        
        result = (result_normalized / A_norms) * b_norm # Un-normalize
    
        # print(f"Result: {result}")
        
        x = np.linspace(0,0.5,100)
        y = result[0] * x + result[1]
        alpha = result[0]
    
        # plt.figure()
        # plt.plot(np.sign(z) * np.power(np.abs(z), 3.0 / 2.0), force, 'c-')
        # plt.plot(x, y, 'm-')
        # plt.show()
        
        E_lstsq = 3.0 / 4.0 * alpha * (1 - nu * nu) / np.sqrt(radius)
        #print(f"least squares E: {E_lstsq*1e3}")
    
        hertz_F_lstsq = hertzian_force(E_lstsq, radius, nu, z, n) + result[1]
        force_2 = force + result[1]

        rmse = np.sqrt(np.mean((force-hertz_F_lstsq)**2.0))
        # print(f"RMSE: {rmse*1e6} pN")
        # print(np.max(hertz_F_lstsq)*1e6)
        # print(rmse / np.max(hertz_F_lstsq))
        # print(f"maximum force: {np.max(hertz_F_lstsq)} uN")
        # print(f"factor: {round(rmse / np.max(hertz_F_lstsq), 2) * 100} %")
        
        result = fmin(sse, x0, args=(radius, nu, force, z, cutoff), disp=False)
        
        E_mod = result[0]*1e3 #*1e6 #*1000
        #print(f"fmin E: {E_mod}")
        
        hertz_F = hertzian_force(result[0], radius, nu, z, n)
        
        r_2 = calculate_r_squared(force, hertz_F)
        r_2_lstsq = calculate_r_squared(force, hertz_F_lstsq)
    
        if show_plot:
            plotting.plot_hertzian_fit(F, disp, hertz_F_lstsq, z, E_lstsq*1e3, r_2_lstsq)
    
            print(f"fmin E: {E_mod}")
            print(f"least squares E: {E_lstsq*1e3}")
            
            plt.figure()
            plt.plot(z, force, 'k-')
            plt.plot(z, hertz_F_lstsq, 'm-')
            plt.xlabel("Indentation depth [um]")
            plt.ylabel("Force [uN]")
            plt.legend(["Data", f"lstsq: E = {round(E_lstsq*1e3, 2)} kPa, r_2 = {round(r_2_lstsq, 2)}"])
            plt.tight_layout()
            
            filename = str(data["file"])
            folder_name = os.path.dirname(filename)
            new_folder = Path(folder_name) / "output" / "Hertz_fit"  
            
            name = str(data["name"])
            index = str(data["index"])

            folder_name = new_folder
            os.makedirs(folder_name, exist_ok=True)

            image_filename = folder_name / f"HertzFit_{name}_{index}.jpg"
            
            plt.savefig(image_filename)
            print(image_filename)
            
            plt.show()

            # plt.figure()
            # plt.plot(z, force, 'k-')
            # plt.plot(z, hertz_F_lstsq, 'm-')
            # plt.plot(z, hertz_F, 'c-')
            # plt.xlabel("Indentation depth [um]")
            # plt.ylabel("Force [uN]")
            # plt.legend(["Data", f"lstsq: E = {round(E_lstsq*1e3, 2)} kPa, r_2 = {round(r_2_lstsq, 2)}", f"fmin: E = {round(E_mod, 2)} kPa, r_2 = {round(r_2, 2)}"])
            # plt.show()

            # plt.figure()
            # plt.plot(z / np.max(z), force / np.max(force), 'k-')
            # plt.plot(z / np.max(z), hertz_F_lstsq / np.max(force), 'm-')
            # plt.plot(z / np.max(z), hertz_F / np.max(force), 'c-')
            # plt.xlabel("Indentation depth [um]")
            # plt.ylabel("Force [uN]")
            # plt.legend(["Data", f"lstsq: E = {round(E_lstsq*1e3, 2)} kPa, r_2 = {round(r_2_lstsq, 2)}", f"fmin: E = {round(E_mod, 2)} kPa, r_2 = {round(r_2, 2)}"])
            # plt.show()

    if r_2_lstsq >= 0.9:
        keep = True
    else:
        keep = False
    
    data[keyname] = [E_lstsq*1e3, r_2_lstsq, keep]

    print(E_lstsq*1e3)

    return [E_lstsq*1e3, r_2_lstsq, keep], keyname




def JKR_force(E, gamma, z, R):
        c_0 = np.pow(z, 2.0)
        c_1 = -4.0 * np.pi * gamma * np.pow(R, 2.0)
        c_2 = - 2.0 * z * R
        c_3 = 0.0
        c_4 = 1.0

        P = -np.pow(c_2, 2.0) / 12.0 - c_0
        Q = -np.pow(c_2, 3.0) / 108.0 + c_2 * c_0 / 3.0 - np.pow(c_1, 2.0) / 8.0
        U = np.pow((-Q/2.0 + np.pow((np.pow(Q, 2.0) / 4.0 + np.pow(P, 3.0) / 27.0), 0.5)), 1.0/3.0)
        s = -5.0 / 6.0 * c_2 + U - P / (3.0 * U)
        w = np.pow((c_2 + 2.0 * s), 0.5)
        l = c_1 / (2.0 * 2)

        a_1 = 0.5 * (w + np.pow((np.pow(w, 2.0) + 4.0 * (c_2 + s + l)), 0.5))
        a_1 = np.abs(a_1)
        
        JKR_F = np.abs(4 * E * np.pow(a_1, 3.0) / (3.0 * R)) - np.abs(np.pow((16.0 * np.pi * gamma * E * np.pow(a_1, 3.0)), 0.5))

        return JKR_F




def parameter_youngs_modulus_JKR(data, radius, nu, cutoff, n=3.0/2.0, x0=[500000], show_plot=False, keyname="youngs_modulus"):
    def sse(param, R, nu, F, z, cutoff):
        E = param[0]
        gamma = param[1]

        gamma = np.abs(np.min(F)) / (3.0 * np.pi * R)

        JKR_F = JKR_force(E, gamma, z, R)
        
        sse_value = np.sum((F - JKR_F)**2.0)
        
        return sse_value

    if data["name"] is not None:
        name = data["name"].copy()
        print()
        print(f"NAME: {name}")

        if data["index"] is not None:
            index = data["index"].copy()
            print(f"Index: {index}")
        print()
        
    F = data["force"].copy()  # Force in µN 
    disp = data["z"].copy()   # Displacement in µm
    
    ix = np.where(disp > (cutoff / 100) * radius)[0]
    
    if len(ix) == -1: ## change back to 0 after
        E_lstsq = 0.0
        r_2_lstsq = 0.0
        print("not enough indentation depth")
    else:
        if len(ix) == 0:
            ix = len(F)
        else:
            ix = ix[0]

        #ix = ix[0]
        
        force = F[1:ix]
        z = disp[1:ix]

        #print(f"length of signal!: {len(force)}")
        #print(len(force) <= 10)
        if len(force) <= 10:
            print("not enough indentation depth -- 10 points or fewer")
            return [0.0, 0.0], keyname

        
        A = np.zeros((len(z), 2))
        A[:,0] = np.sign(z) * np.power(np.abs(z), n)
        A[:,1] = 1.0
        b = np.array(force, dtype=float)
        
        b_norm = np.linalg.norm(b)  # Compute norm of b
        b_normalized = b / b_norm  # Normalize b
    
        A_norms = np.linalg.norm(A, axis=0)  # Save norms before normalizing
        A_normalized = A / A_norms
        
        result_normalized = np.linalg.lstsq(A_normalized, b_normalized, rcond=None)[0]
        print(f"Normalized results: {result_normalized}")
        
        result = (result_normalized / A_norms) * b_norm # Un-normalize
    
        # print(f"Result: {result}")
        
        x = np.linspace(0,0.5,100)
        y = result[0] * x + result[1]
        alpha = result[0]
    
        # plt.figure()
        # plt.plot(np.sign(z) * np.power(np.abs(z), 3.0 / 2.0), force, 'c-')
        # plt.plot(x, y, 'm-')
        # plt.show()
        
        E_lstsq = 3.0 / 4.0 * alpha * (1 - nu * nu) / np.sqrt(radius)
        print(f"least squares E: {E_lstsq*1e3}")
    
        hertz_F_lstsq = hertzian_force(E_lstsq, radius, nu, z, n) + result[1]
        force_2 = force + result[1]

        rmse = np.sqrt(np.mean((force-hertz_F_lstsq)**2.0))
        # print(f"RMSE: {rmse*1e6} pN")
        # print(np.max(hertz_F_lstsq)*1e6)
        # print(rmse / np.max(hertz_F_lstsq))
        # print(f"maximum force: {np.max(hertz_F_lstsq)} uN")
        # print(f"factor: {round(rmse / np.max(hertz_F_lstsq), 2) * 100} %")

        force_arr = np.array(force, dtype=float)
        force_norm = np.linalg.norm(force_arr)
        force_normalized = force_arr / force_norm
        z_arr = np.array(z, dtype=float)
        z_norm = np.linalg.norm(z_arr)
        z_normalized = z_arr / z_norm
        
        result = fmin(sse, x0=[1000.0, 0.000005], args=(radius, nu, force, z, cutoff), disp=False)
        print(result)


        E_mod = result[0]*1e3 #*1e6 #*1000
        gamma = result[1]
        gamma = np.abs(np.min(F)) / (3.0 * np.pi * radius)
        
        JKR_F = JKR_force(result[0], gamma, z, radius)
        
        print(f"fmin E: {E_mod}")
        print(f"fmin gamma: {gamma}")
        
        hertz_F = hertzian_force(result[0], radius, nu, z, n)

        
        r_2 = calculate_r_squared(force, hertz_F)
        r_2_lstsq = calculate_r_squared(force, hertz_F_lstsq)
    
        if show_plot:
            plotting.plot_hertzian_fit(F, disp, hertz_F_lstsq, z, E_lstsq*1e3, r_2_lstsq)
    
            print(f"fmin E: {E_mod}")
            print(f"least squares E: {E_lstsq*1e3}")
            
            plt.figure()
            plt.plot(z, force, 'k-')
            plt.plot(z, hertz_F_lstsq, 'm-')
            plt.plot(z, JKR_F, 'c-')
            plt.xlabel("Indentation depth [um]")
            plt.ylabel("Force [uN]")
            plt.legend(["Data", f"lstsq: E = {round(E_lstsq*1e3, 2)} kPa, r_2 = {round(r_2_lstsq, 2)}", f"fmin: E = {round(E_mod, 2)} kPa, r_2 = {round(r_2, 2)}"])
            plt.show()

            # plt.figure()
            # plt.plot(z / np.max(z), force / np.max(force), 'k-')
            # plt.plot(z / np.max(z), hertz_F_lstsq / np.max(force), 'm-')
            # plt.plot(z / np.max(z), hertz_F / np.max(force), 'c-')
            # plt.xlabel("Indentation depth [um]")
            # plt.ylabel("Force [uN]")
            # plt.legend(["Data", f"lstsq: E = {round(E_lstsq*1e3, 2)} kPa, r_2 = {round(r_2_lstsq, 2)}", f"fmin: E = {round(E_mod, 2)} kPa, r_2 = {round(r_2, 2)}"])
            # plt.show()

    data[keyname] = [E_lstsq*1e3, r_2_lstsq]

    print(E_lstsq*1e3)

    return [E_lstsq*1e3, r_2_lstsq], keyname








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



def parameter_max_indentation_force(data, keyname="max_indentation_force"):  
    labels = data['labels'].copy()
    indices = np.where(labels == 'f')[0]
    last_index = indices[-1] if indices.size > 0 else -1
    F = data["force"].copy()  # Force in µN (make positive)
    disp = data["z"].copy()  # Displacement in µm

    print("maximum indentation force")
    print(F[last_index - 1])

    max_force = F[last_index - 1]
    
    return max_force, keyname

def parameter_max_indentation_point(data, keyname="max_indentation_point"):  
    labels = data['labels'].copy()
    indices = np.where(labels == 'f')[0]
    last_index = indices[-1] if indices.size > 0 else -1
    F = data["force"].copy()  # Force in µN (make positive)
    disp = data["z"].copy()  # Displacement in µm

    print("maximum indentation point")
    print(disp[last_index - 1])

    max_indentation_point = [last_index - 1, disp[last_index - 1]]
    
    return max_indentation_point, keyname



def parameter_max_retract_force(data, keyname="max_retraction_force"):  
    F = data["force"].copy()  # Force in µN (make positive)
    disp = data["z"].copy()  # Displacement in µm

    print("minimum force")
    print(np.min(F))

    min_force = np.min(F)
    
    return min_force, keyname



def parameter_force_change(data, keyname="force_change"):  
    print(data["name"])
    force = data["force"].copy()
    displ = data["z"].copy()
    contact_point = data["contact_point"].copy()

    labels = data['labels'].copy()
    indices = np.where(labels == 'b')[0]

    print(indices)
    print('contact_point')
    print(contact_point)
    disp_contact = contact_point[1]
    
    indices_contact = np.where(displ <= disp_contact)
    indices_contact = np.where(force <= 0.0)
    #indices_release = np.where(displ >= data[param2][1])

    print(indices_contact)
    ix_max = np.argmin(force)
    print(ix_max)
    
    common = np.intersect1d(indices, indices_contact)
    # common = np.intersect1d(common, indices_release)
    common = common[common <= ix_max]

    print(common)
    print(len(common))

    if len(common) == 0:
        force_change = [0, 0]
        return force_change, keyname
    
    force1 = force[common[0]]
    force2 = force[common[-1]]
    force_change = force1 - force2

    ix_start = common[0]
    ix_end = common[-1]

    plt.figure()
    plt.plot(displ, force, 'r-')
    plt.axhline(y=force[ix_max])
    plt.axhline(y=force[ix_start])
    plt.axhline(y=force[ix_end])
    plt.axvline(x=displ[ix_max])
    plt.axvline(x=displ[ix_start])
    plt.show()
    
    print(f"force 1: {force[ix_start]}")
    print(ix_start)
    print(f"force 2: {force[ix_end]}")
    print(ix_end)
    print(f"force change: {force_change*1e3} nN")

    disp_change = np.abs(displ[ix_start] - displ[ix_end])
    print(f"disp change: {disp_change*1e3} nm")
    stiffness = force_change/disp_change
    print(f"stiffness: {stiffness} uN per um")
        
    force_change = [force_change, disp_change]
    
    return force_change, keyname


def parameter_retraction_length(data, keyname="retraction_length"):  
    force = data["force"].copy()  # Force in µN (make positive)
    disp = data["z"].copy()  # Displacement in µm
    contact_point = data["contact_point"].copy()
    release_point = data["release_point"].copy()

    retraction_length = contact_point[1] - release_point[1]

    print("retraction length")
    print(retraction_length)
    
    return retraction_length, keyname


def parameter_indentation_depth(data, keyname="indentation_depth"):  
    contact_point = data["contact_point"].copy()
    max_indentation_point = data["max_indentation_point"].copy()

    indentation_depth = max_indentation_point[1] - contact_point[1] 

    print("indentation depth")
    print(indentation_depth)
    
    return indentation_depth, keyname




def parameter_release_point(data, keyname="release_point"):  
    force = data["force"].copy()  # Force in µN (make positive)
    disp = data["z"].copy()  # Displacement in µm

    max_deriv_ix = np.argmax(np.diff(force,1))
    rel_point = disp[max_deriv_ix]

    print("release point")
    print(rel_point)

    release_point = [max_deriv_ix, rel_point]
    
    return release_point, keyname



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

    var_signal = np.var(y_fitted)
    var_noise = np.var(y_actual - y_fitted)
    max_r_2 = var_signal / (var_signal + var_noise)
    r_2_alt = 1 - var_noise / var_signal
    
    # print("percent deviation")
    # percent_dev = []
    # for i, val in enumerate(y_actual):
    #     if val != 0:
    #         percent_dev.append(np.abs((y_actual[i] - y_fitted[i])) / y_actual[i] * 100.0)
    #
    # mean_percent_dev = np.mean(percent_dev)
    # print(mean_percent_dev)
    #
    print("RMSE")
    RMSE = np.sqrt(np.mean((y_actual - y_fitted)**2.0))
    print(RMSE*1e3)

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
