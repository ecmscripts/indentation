import numpy as np 


# all functions should follow the following input/output, where data contains various arrays of the same length, most time, displ and force:
def do_nothing(data):
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


def get_forward_curve(data):
    labels = data['labels'].copy()
    indices = np.where(labels == 'f')[0]
    last_index = indices[-1] if indices.size > 0 else -1

    for key in data:
        if data[key] is not None and data[key].size > 1:
            data[key] = data[key][0:last_index]

    return data


def add_noise(data):
    force = data['force'].copy()

    amplitude = 1e-5 #1e-6
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

    for key in data:
        if data[key] is not None and data[key].size > 1:
            print(key)
            print(data[key])
            data[key] = data[key][indices]

    return data
    

def crop_afm_temp(data):
    force = data["force"].copy()
    displ = data["z"].copy()

    force = force[:-10]
    ix_end = np.argmax(force)
    ix_start = 0
    #ix_start = int(len(force)/2)
 
    ix_end = np.argmax(np.diff(displ)) - 1
    
    for key in data:
        data[key] = data[key][ix_start:ix_end]
    return data


def crop_ft_temp(data):
    force = data["force"].copy()
    displ = data["z"].copy()

    force = force[:-10]
    ix_end = np.argmax(force)
    ix_start = 0
    
    for key in data:
        data[key] = data[key][ix_start:ix_end]
    return data


def crop_start(data, ix_start):
    force = data["force"].copy()
    displ = data["z"].copy()

    for key in data:
        if data[key] is not None and data[key].size > 1:
            data[key] = data[key][ix_start:]
    return data


def crop_end(data, ix_end):
    force = data["force"].copy()
    displ = data["z"].copy()

    for key in data:
        data[key] = data[key][:ix_end]
    return data
