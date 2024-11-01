import numpy as np 


def do_nothing(data):
    return data


def processing_shift_to_zero(data):
    """Subtract the initial force value to correct baseline."""
    f0 = data['force'][0]
    z0 = data['z'][0]
    data['force'] = data['force'] - f0
    data['z'] = data['z'] - z0
    return data


def processing_smooth_data(data, window_size=5):
    """Apply a simple moving average to smooth the data."""
    data['force'] = np.convolve(data['force'], np.ones(window_size)/window_size, mode='valid')
    data['z'] = data['z'][:len(data['force'])]  # Adjust z to match the length
    return data


def crop(data):
    force = data["force"].copy()
    displ = data["z"].copy()

    force = force[:-10]
    ix_end = np.argmax(force)
    ix_start = int(len(force)/2)

    for key in data:
        data[key] = data[key][ix_start:ix_end]
    return data
