import george
from george import kernels
import numpy as np


def add_correlated_gaussian_noise(data_y_values, data_x_values, data_y_error_bars,
                                  noise_lengthscale, noise_variance):
    """
    Adds correlated noise to data using a gaussian process
    (Matern 3/2 kernel for now)

    Parameters
    ---
    data_y_values (array-like (N,)): y values of your un-scatted generated data
    data_x_values : (array-like (N,)):  x (or input values of the data)
    data_y_error_bars (array-like (N,)): simulated error bars on each data point
        (1 standard deviations or 1 'sigma')
    noise_lengthscale (float): length-scale at which the noise will be
        correlated
    noise_variance (float) : variance of the correlated component of the
        noise (the magnitude of the correlated component of the noise, standard deviation of
        the correlated noise squared)

    Returns
    ----
    simulated_data_with_noise (array-like (N,)): y_values of the data noise with
        correlated noise
    """

    if noise_lengthscale < 0.0:
        raise ValueError(f'Length-scale must be positive, I got {noise_lengthscale}')
    if noise_variance < 0.0:
        raise ValueError(f'Variance must be positive, I got {noise_variance}')

    gp = george.GP(noise_variance * kernels.Matern32Kernel(noise_lengthscale))
    correlated_noise = gp.sample(data_x_values)
    white_noise = data_y_error_bars*np.random.randn(len(data_x_values))
    return data_y_values + correlated_noise + white_noise
