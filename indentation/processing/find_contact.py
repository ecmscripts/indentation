import numpy as np 
from scipy.optimize import fmin
from scipy.signal import convolve
from scipy.signal.windows import gaussian

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


def findContact_blackMagic(data, N_int=1000, padding_fraction=0.02):
        
    # Extract and copy data
    force_r = data["force"].copy()
    displ_r = -data["z"].copy()
    
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
        dn = 2
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
    for ix_glob in range(10, int(3*N_int/4), int(N_int/30)):
        ix = ix_glob
        for i, line in enumerate(img[::-1]):
            ixs = [ix, ix+1, ix-1]
            ixmax = np.argmax([line[ixs[0]], line[ixs[1]], line[ixs[2]]])
            ix = ixs[ixmax]
            imgc[::-1][i, ix] = -0.3
            
        zline[ix] += 1
        list_ix.append(ix)
    
    # Find cut point
    contact_index = np.argmax(zline)
    contact_index = int(contact_index/N_int * len(force_r))
    
    # Create new dictionary with cropped data
    result_data = {}
    for key in ["time", "z", "force"]:
        # Crop array from contact point and subtract initial value
        cropped_array = data[key][contact_index:].copy()
        result_data[key] = cropped_array - cropped_array[0]
    
    return result_data
