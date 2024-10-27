import numpy as np
from scipy.signal import convolve
from scipy.signal.windows import gaussian
from scipy import interpolate
from tqdm import tqdm
import os, sys, glob

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


def indentation_force(displ, displ_max, alpha):
    '''
    Get indentation function with negative displacement.
    '''
    force = np.array(alpha*(np.array(displ))**(3.0/2.0))
    force = np.array(list(np.zeros(len(displ)-1)) + list(force))
    displ = np.array(list(-displ[::-1][:-1]) + list(displ))
    
    return displ, force

def shift(displ, force, displ_max, xs, ys):
    '''
    Shift force and displacement. 
    xs: is the reference to recover the original function starting point.
    '''
    if xs >= displ_max:
        return displ, force
    elif xs <= 0:
        return displ, force
    else:
        return np.array(displ)+xs, np.array(force)+ys
    
    
def add_noise(force, noise_scale1, noise_scale2, noise_var):

    # calculate noises
    random = np.random.rand()
    noise_multiplier = random*(1+noise_var)
    noise1 = noise_multiplier*noise_scale1*np.random.randn(len(force))
    noise2 = noise_multiplier*noise_scale2*np.cumsum(np.random.randn(len(force)))

    # add variation
    return force + noise1 + noise2

def add_baseline_tilt(force, baseline_tilt_max):
    baseline_tilt = baseline_tilt_max * np.random.rand()
    slope = baseline_tilt * np.linspace(0, 1, len(force))
    return force + slope

def clip_data(displ, force):
    # discard anything with negative displacement
    ix_cut = np.argmin(np.abs(displ))
    
    return displ[ix_cut:], force[ix_cut:]

def normalize(displ, force, N):
    
    #normalize and shift
    force = force - np.min(force)
    displ = displ/np.max(displ)
    force = force/np.max(force)
    # interpolate
    dint = np.linspace(0,1,N)
    fint = np.interp(dint, displ, force)
    
    return dint, fint

def normalize_signal(signal, N=100):
    # Convert input to numpy array if it's not already
    signal = np.array(signal)
    
    # 1. Shift the signal so that the starting point is at 0
    shifted_signal = signal - np.min(signal)
    
    # 2. Scale the signal so that the ending point is at 1
    scaled_signal = shifted_signal / np.max(shifted_signal)
    
    # 3. Interpolate the signal to the desired length N
    old_indices = np.linspace(0, 1, num=len(signal))
    new_indices = np.linspace(0, 1, num=N)
    f = interpolate.interp1d(old_indices, scaled_signal)
    interpolated_signal = f(new_indices)
    
    return interpolated_signal

def create_scaleSpace(signal, N=100):
    # vars
    start_width = N
    end_width = 1
    num_convolutions = start_width - end_width + 1

    # normalize signal
    signal = normalize_signal(signal, N=N)
    
    # Initialize the result array
    result = np.zeros((num_convolutions, N))
    
    for i, width in enumerate(range(start_width, end_width - 1, -1)):
        # Create Gaussian kernel
        kernel = gaussian(width, std=width/3)
        
        # Normalize the kernel
        kernel = kernel / np.sum(kernel)
        
        # Perform convolution
        convolved = convolve(signal, kernel, mode='same')
        
        # Store the result
        result[i] = convolved
    
    # Normalize the result to be between 0 and 1
    result = (result - np.min(result)) / (np.max(result) - np.min(result))
    
    return result.T

def numdiff(x, y, n=1):
    
    dydx = np.diff(y, n)/np.diff(x[(n-1):], 1)**n 
    x    = 0.5*(x[n:] + x[:-n])
    return x, dydx

def chop(v, N):
    '''
    Symmetric chop.
    '''
    return v[N:(len(v)-N)]
