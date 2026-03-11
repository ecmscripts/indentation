import numpy as np 
from scipy.optimize import fmin
from scipy.signal import convolve
from scipy.signal.windows import gaussian
import matplotlib.pyplot as plt
import os

import torch

from ..utils.math import numdiff 
from ..utils.signal import create_scaleSpace, normalize, normalize_signal
from ..ml.models import ConvClassifier_1, ConvClassifier_2


def findContact_minimum(data):
    data_copy = data.copy()
    ix_cut = np.argmin(data_copy["force"])
    for key in ["time", "z", "force"]:
        data_copy[key] = data_copy[key][ix_cut:]
        data_copy[key] -= data_copy[key][0]
    return data_copy


def findContact_deflection(data):
    data_copy = data.copy()

    indices = np.where(data_copy["deflection"] > 0.05)[0]
    if indices.size:
        ix_cut = indices[0]
    else:
        ix_cut = -1
    #ix_cut = np.argmin(data_copy["force"])
    for key in ["time", "z", "force"]:
        data_copy[key] = data_copy[key][ix_cut:]
        data_copy[key] -= data_copy[key][0]
    return data_copy
    

def findContact_blackMagic_CNN(data, net, N):

    data_copy = data.copy()

    force_full_sliced = data_copy["force"]
    displ_full_sliced = -data_copy["z"]
    length = len(force_full_sliced)

    force = normalize_signal(force_full_sliced, N)
    img = create_scaleSpace(force, N)
    
    output = net(torch.tensor(img, dtype=torch.float).view(1, 1 ,N, N).to("cpu"))
    
    loc = np.argmax(output.detach().cpu().numpy())
    ix_cut = int(loc/N*length)

    if ix_cut/length < 0.9:
        for key in ["time", "z", "force"]:
            data_copy[key] = data_copy[key][ix_cut:]
            data_copy[key] -= data_copy[key][0]
        return data_copy
    else:
        print("Discarded curve.")


def findContact_blackMagic(data, N_int=2000, padding_fraction=0.02, show_plots=False): #0.02
    # Extract and copy data
    force_r = np.array(data["force"].copy(), dtype=np.float64)
    displ_r = np.array(data["z"].copy(), dtype=np.float64)

    # test
    dx = displ_r[1] - displ_r[0] 
    dy = np.gradient(force_r, dx)     # First derivative
    d2y = np.gradient(dy, dx)
    d3y = np.gradient(d2y, dx)

    plt.figure()
    plt.plot(data["z"], d3y, 'r-')
    plt.axvline(x=0.5)
    plt.show()
    
    plt.figure()
    plt.plot(data["z"], d2y, 'r-')
    plt.axvline(x=0.5)
    plt.show()

    plt.figure()
    plt.plot(data["z"], dy, 'r-')
    plt.axvline(x=0.5)
    plt.show()
    #test

    
    # Interpolate data and normalize
    displ = np.linspace(displ_r[0], displ_r[-1], N_int)
    force = np.interp(displ, displ_r, force_r)
    displ = displ/displ[-1]
    force = force/force[-1]
    
    # Initialize image array
    img = []

    # Process data with gaussian filter
    for n in range(int(padding_fraction * N_int), int(N_int - padding_fraction * N_int)):
        # Set up correct length for selected sigma
        xf = np.linspace(-2, 2, n)
        # Gaussian filter
        g = 1/(np.sqrt(2*np.pi))*np.exp(-xf**2/(2.0))
        # Normalize filter to interval
        g = g/np.sum(g)
        # Derivatives and corresponding normalization
        dn = 2  #second derviation dn=2
        _, ddg = numdiff(xf, g, dn)
        # Convolve with mask
        yf = np.convolve(force, ddg, mode='same')
        img.append(yf)
        
    # Convert to numpy array
    img = np.array(img)
    
    # Initialize tracking arrays
    list_ix = []
    imgc = img.copy()
    zline = np.zeros((len(img.T)))
    
    # Track maximum points
    for ix_glob in range(50, N_int-2, int(N_int/30)):
        ix = ix_glob
        for i, line in enumerate(img[::-1]):
            ixs = [ix, ix+1, ix-1]

            if ix + 1 >= len(line):
                ixs = [ix, ix, ix-1]
            elif ix -1 < 0:
                ix = [ix, ix+1, ix]
            else:
                ixs = [ix, ix+1, ix-1]
            
            ixmax = np.argmax([line[ixs[0]], line[ixs[1]], line[ixs[2]]])
            ix = ixs[ixmax]
            imgc[::-1][i, ix] = -0.3

        zline[ix] += 1
        
        if ix != 0: # maximum curvature cannot be found at index = 0
            list_ix.append(ix) 
    
    # Find cut point
    contact_indices = np.argwhere(zline == np.amax(zline))
    contact_index = np.argmax(zline[1:])
    contact_index = int(contact_index/N_int * len(force_r))
    
    # Create new dictionary with cropped data
    result_data = {}

    for key in["time", "z", "force", "deflection", "z_piezo"]:
        # Crop array from contact point and subtract initial value
        if data[key] is not None:
            cropped_array = data[key][contact_index:].copy()
            result_data[key] = cropped_array - cropped_array[0]

    if show_plots:
        plt.figure()
        plt.imshow(imgc)
        plt.show()
        
        plt.figure()
        plt.plot(data["z"], data["force"], 'r-')
        plt.axvline(x=data["z"][contact_index])
        plt.xlabel("Displacement [um]")
        plt.ylabel("Force [uN]")
        plt.legend(["Data", "Contact Point"])
        plt.tight_layout()

        # plt.figure()
        # plt.plot(data["z"], data["force"], 'r*')
        # plt.show()

        filename = str(data["file"])
        folder_name = os.path.dirname(filename)
        folder_name = str(folder_name) + "\\output\\contact_points\\"
        name = str(data["name"])
        index = str(data["index"])
        
        os.makedirs(folder_name, exist_ok=True)

        image_filename = str(folder_name) + "ContactPoint_" + str(name) + "_" + str(index) + ".jpg"
        
        plt.savefig(image_filename)
        print(image_filename)

        plt.show()

    return result_data

def findContact_blackMagic_parameter(data, N_int=2000, padding_fraction=0.02, show_plots=False, keyname="contact_point"):
    # Extract and copy data
    force_r = np.array(data["force"].copy(), dtype=np.float64)
    displ_r = np.array(data["z"].copy(), dtype=np.float64)

    print(displ_r)
    print(data["name"])

    #N_int = max(int(len(force_r) / 4.0), 1000)
    
    # Interpolate data and normalize
    displ = np.linspace(displ_r[0], displ_r[-1], N_int)
    force = np.interp(displ, displ_r, force_r)
    displ = displ/displ[-1]
    force = force/force[-1]
    
    # Initialize image array
    img = []

    #print("padding fraction * N_int")
    #print(int(padding_fraction * N_int))
    #print(N_int)
    
    # Process data with gaussian filter
    for n in range(int(padding_fraction * N_int), int(N_int - padding_fraction * N_int)):
        # Set up correct length for selected sigma
        xf = np.linspace(-2, 2, n)
        # Gaussian filter
        g = 1/(np.sqrt(2*np.pi))*np.exp(-xf**2/(2.0))
        # Normalize filter to interval
        g = g/np.sum(g)
        # Derivatives and corresponding normalization
        dn = 2 #second derivative dn=2
        _, ddg = numdiff(xf, g, dn)
        # Convolve with mask
        yf = np.convolve(force, ddg, mode='same')
        img.append(yf)
        
    # Convert to numpy array
    img = np.array(img)
    
    # Initialize tracking arrays
    list_ix = []
    imgc = img.copy()
    zline = np.zeros((len(img.T)))
    
    # Track maximum points
    for ix_glob in range(50, N_int-2, int(N_int/30)):
        ix = ix_glob
        for i, line in enumerate(img[::-1]):
            ixs = [ix, ix+1, ix-1]

            if ix + 1 >= len(line):
                ixs = [ix, ix, ix-1]
            elif ix -1 < 0:
                ix = [ix, ix+1, ix]
            else:
                ixs = [ix, ix+1, ix-1]
            
            ixmax = np.argmax([line[ixs[0]], line[ixs[1]], line[ixs[2]]])
            ix = ixs[ixmax]
            imgc[::-1][i, ix] = -0.3

        zline[ix] += 1
        
        if ix != 0: # maximum curvature cannot be found at index = 0
            list_ix.append(ix) 
    
    # Find cut point
    contact_indices = np.argwhere(zline == np.amax(zline))
    contact_index = np.argmax(zline[1:])
    contact_index = int(contact_index/N_int * len(force_r))

    #print(zline)
    #print(f"len(force_r): {len(force_r)}")
    #print(list_ix)
    #print(contact_indices)
    #print(contact_index)
    
    # Create new dictionary with cropped data
    result_data = {}
    for key in ["time", "z", "force"]:
        # Crop array from contact point and subtract initial value
        cropped_array = data[key][contact_index:].copy()
        result_data[key] = cropped_array - cropped_array[0]

    if show_plots:
        plt.figure()
        plt.imshow(imgc)
        plt.show()
        
        plt.figure()
        plt.plot(data["z"], data["force"], 'r-')
        plt.axvline(x=data["z"][contact_index])
        plt.show()

        plt.figure()
        plt.plot(data["z"], data["force"], 'r*')
        plt.show()

    data[keyname] = [contact_index, data["z"][contact_index]]

    print(data[keyname])
    #return result_data
    return [contact_index, data["z"][contact_index], data["force"][contact_index]], keyname