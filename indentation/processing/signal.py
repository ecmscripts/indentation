import numpy as np 
from scipy.fft import fft, fftfreq
from scipy.signal import butter, lfilter, freqz


# all functions should follow the following input/output, where data contains various arrays of the same length, most time, displ and force:
def do_nothing(data):
    return data


def shift_above_y_axis(data):
    fmin = np.min(data['force'])

    data['force'] = data['force'] - fmin

    return data



def organize_curves(data):
    param = 'youngs_modulus'

    print(data["youngs_modulus"])
    data["keep"] = data['youngs_modulus'][2]
    print(data['keep'])

    return data




def shift_contact_to_zero(data):
    param = 'contact_point'
    disp = data["z"].copy()
    
    print(data['contact_point'])
    contact_disp = data[param][1]
    print(contact_disp)
    ix_start = np.where(disp >= contact_disp)
    #print(ix_start)
    ix_start = ix_start[0][0]
    #print(ix_start)
    
    for key in ["time", "z", "force", "deflection", "z_piezo"]:
        # Crop array from contact point and subtract initial value
        data[key] = data[key][ix_start:].copy()
        data[key] = data[key] - data[key][0]

    return data

def shift_contact_to_near_zero(data):
    param = 'contact_point'

    ix_contact = int(data[param][0])
    ix_start = 0
    if ix_contact - 1000 > 0:
        ix_start = ix_contact - 1000
    
    for key in ["time", "z", "force", "deflection", "z_piezo"]:
        # Crop array from contact point and subtract initial value
        data[key] = data[key][ix_start:].copy()
        #data[key] = data[key] - data[key][0]

    return data

def get_approach(data):
    param = 'contact_point'
    disp = data["z"].copy()

    contact_disp = data[param][1]
    print(contact_disp)
    ix_start = np.where(disp >= contact_disp)
    #print(ix_start)
    ix_start = ix_start[0][0]
    #print(ix_start)
    
    for key in ["time", "z", "force", "deflection", "z_piezo"]:
        # Crop array from contact point and subtract initial value
        data[key] = data[key][0:ix_start].copy()

    return data


def get_extension(data):
    param1 = 'contact_point'
    param2 = 'release_point'

    labels = data['labels'].copy()
    indices = np.where(labels == 'b')[0]

    force = data["force"].copy()
    displ = data["z"].copy()

    #indices_contact = np.where(displ <= data[param1][1])
    #indices_release = np.where(displ >= data[param2][1])

    ix_max = np.argmin(force)

    # common = np.intersect1d(indices, indices_contact)
    # common = np.intersect1d(common, indices_release)
    common = indices[indices <= ix_max]

    print(f"contact_point: {data['contact_point'][1]}")
    print(f"release_point: {data['release_point'][1]}")
    
    for key in ["time", "z", "force", "deflection", "z_piezo"]:
        # Crop array from contact point and subtract initial value
        data[key] = data[key][common].copy()

    return data


def get_extension3(data):
    param1 = 'contact_point'
    param2 = 'release_point'

    labels = data['labels'].copy()
    indices = np.where(labels == 'b')[0]

    force = data["force"].copy()
    displ = data["z"].copy()

    print("indices")
    print(indices)
    
    indices_subzeroF = np.where(force <= 0.0)
    #indices_release = np.where(displ >= data[param2][1])

    print("indices_subzeroF")
    print(indices_subzeroF)
    
    ix_max = np.argmin(force)

    common = np.intersect1d(indices, indices_subzeroF)
    # common = np.intersect1d(common, indices_release)
    # common = indices[indices <= ix_max]
    common = common[common <= ix_max]

    print(f"contact_point: {data['contact_point'][1]}")
    print(f"release_point: {data['release_point'][1]}")
    
    for key in ["time", "z", "force", "deflection", "z_piezo"]:
        # Crop array from contact point and subtract initial value
        data[key] = data[key][common].copy()

    return data


def get_extension2(data):
    param1 = 'contact_point'
    param2 = 'release_point'

    labels = data['labels'].copy()
    indices = np.where(labels == 'b')[0]

    force = data["force"].copy()
    displ = data["z"].copy()

    indices_contact = np.where(displ <= data[param1][1])
    indices_release = np.where(displ >= data[param2][1])

    common = np.intersect1d(indices, indices_contact)
    common = np.intersect1d(common, indices_release)

    print(f"contact_point: {data['contact_point'][1]}")
    print(f"release_point: {data['release_point'][1]}")
    
    for key in ["time", "z", "force", "deflection", "z_piezo"]:
        # Crop array from contact point and subtract initial value
        data[key] = data[key][common].copy()

    return data


def processing_shift_to_zero(data):
    """Subtract the initial force value to correct baseline."""
    f0 = data['force'][0]
    z0 = data['z'][0]
    data['force'] = data['force'] - f0
    data['z'] = data['z'] - z0
    return data


def processing_smooth_data(data, window_size=20):
    """Apply a simple moving average to smooth the data."""
    data['force'] = np.convolve(data['force'], np.ones(window_size)/window_size, mode='valid')
    data['z'] = data['z'][:len(data['force'])]  # Adjust z to match the length
    return data


def butter_lowpass(cutoff, fs, order=5):
    return butter(order, cutoff, fs=fs, btype='low', analog=False)

def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y


def filter_out_hf_data(data, cutoff=15):
    force = data['force'].copy()
    disp = data['z'].copy()
    
    # filter
    order = 6
    fs = 1000
    T = 1.0 / fs
    n = len(force)
    time = np.linspace(0, n, n) / fs

    y_filt = butter_lowpass_filter(force, cutoff, fs, order)

    data['force'] = y_filt

    return data


def moving_average_filter(data, window_size=3):
    force = data['force'].copy()
    disp = data['z'].copy()
    
    i = 0
    # Initialize an empty list to store moving averages
    moving_averages = []
    x_midpoints = []
    
    # Loop through the array t o
    #consider every window of size 3
    while i < len(force) - window_size + 1:
    
        # Calculate the average of current window
        window_average = np.sum(force[i:i+window_size]) / window_size
        
        # Store the average of current
        # window in moving average list
        moving_averages.append(window_average)

        mid_index = i + window_size // 2
        x_midpoints.append(disp[mid_index])
        
        # Shift window to right by one position
        i += 1

    data['force'] = np.array(moving_averages, dtype="object")
    data['z'] = np.array(x_midpoints, dtype="object")

    return data



def get_forward_curve(data):
    labels = data['labels'].copy()
    indices = np.where(labels == 'f')[0]
    last_index = indices[-1] if indices.size > 0 else -1

    for key in ["time", "z", "force", "deflection", "z_piezo"]:
        if data[key] is not None:
            data[key] = data[key][0:last_index]

    return data


def add_noise(data):
    force = data['force'].copy()

    amplitude = 1e-4 #1e-6
    freq_H = 1e8 # Hz 1e7
    freq_L = 1e4 # Hz
    fs = 1000 #Hz
    n = len(data["force"])
    i = np.linspace(0, n, n)

    print(i/fs)
    random_noise = np.random.normal(loc=0, scale=amplitude, size=(len(data["force"]),))
    sinusoidal_noise_highF = amplitude * np.sin(2 * np.pi * freq_H * i / fs)
    sinusoidal_noise_lowF = amplitude * np.cos(2 * np.pi * freq_L * i / fs)

    print(random_noise)
    data["force"] = force + sinusoidal_noise_lowF + sinusoidal_noise_highF + random_noise
    return data
    


def remove_curves_below_force_thresh(data):
    print()


def get_retraction_curve(data):
    labels = data['labels'].copy()
    indices = np.where(labels == 'b')[0]

    for key in ["time", "z", "force", "deflection", "z_piezo"]:
        if data[key] is not None:
            data[key] = data[key][indices]

    return data


def get_retraction_curve_return(data):
    force = data["force"].copy()
    displ = data["z"].copy()

    ix_start = np.argmin(force) - 3 

    for key in ["time", "z", "force", "deflection", "z_piezo"]:
        if data[key] is not None:
            data[key] = data[key][ix_start:]

    return data


def get_retraction_curve_extend(data):
    force = data["force"].copy()
    displ = data["z"].copy()

    ix_end = np.argmin(force)

    for key in ["time", "z", "force", "deflection", "z_piezo"]:
        if data[key] is not None:
            data[key] = data[key][:ix_end]

    return data


def shift_extension_to_zero(data):
    force = data["force"].copy()
    displ = data["z"].copy()

    f0 = force[0]
    z0 = displ[0]

    data["z"] = -(displ - z0)
    data["force"] = -force

    return data
    
    

def crop_afm_temp(data):
    force = data["force"].copy()
    displ = data["z"].copy()

    force = force[:-10]
    ix_end = np.argmax(force)
    ix_start = 0
    #ix_start = int(len(force)/2)
 
    ix_end = np.argmax(np.diff(displ)) - 1
    
    for key in ["time", "z", "force", "deflection", "z_piezo"]:
        data[key] = data[key][ix_start:ix_end]
    return data


def crop_ft_temp(data):
    force = data["force"].copy()
    displ = data["z"].copy()

    force = force[:-10]
    ix_end = np.argmax(force)
    ix_start = 0
    
    for key in ["time", "z", "force", "deflection", "z_piezo"]:
        data[key] = data[key][ix_start:ix_end]
    return data


def crop_start(data, ix_start):
    force = data["force"].copy()
    displ = data["z"].copy()

    for key in ["time", "z", "force", "deflection", "z_piezo"]:
        if data[key] is not None and data[key].size > 1:
            data[key] = data[key][ix_start:]
    return data


def crop_end(data, ix_end):
    force = data["force"].copy()
    displ = data["z"].copy()

    for key in ["time", "z", "force", "deflection", "z_piezo"]:
        data[key] = data[key][:ix_end]
    return data
