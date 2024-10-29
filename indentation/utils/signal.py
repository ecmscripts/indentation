import numpy as np
from scipy.signal import convolve
from scipy.signal.windows import gaussian
from scipy import interpolate


def normalize(displ, force, N):
    
    #normalize and shift
    force = force - np.min(force)
    displ = displ - np.min(displ)
    displ = displ/np.max(displ)
    force = force/np.max(force)
    # interpolate
    dint = np.linspace(0,1,N)
    fint = np.interp(dint, displ, force)
    
    return dint, fint
    

def normalize_signal(signal, N=100):
    # Convert input to numpy array if it's not already
    signal = np.array(np.copy(signal))
    
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
    
    
def chop(v, padding: int):
    '''
    Symmetric chop.
    '''
    return v[padding:(len(v)-padding)]
    

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
