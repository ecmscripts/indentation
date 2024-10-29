import numpy as np

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
