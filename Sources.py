import numpy as np


def create_sinusoidal_source_2D(amplitude, frequency, x0, y0):
    """
    Create a source function for a sinusoidal point source with constant arbitrary amplitude, frequency, and location. 2D version.
    Parameters:
    amplitude: (float)
        Amplitude of the source
    frequency: (float)
        Frequency of the source
    x0, y0: (float, float)
        Source location

    Returns:
    fun: (function(x: float, y: float, t: float) -> float)
        A function representing the point source with arguments (x, y, t).
    """
    def fun(x, y, t):
        # Swap x0 and y0 because grid is indexed as u[y_i, x_i]
        if np.hypot(x - y0, y - x0) < 0.5:
            return amplitude * np.sin(2 * np.pi * frequency * t)
        return 0.0

    return fun


def create_sinusoidal_source_3D(amplitude, frequency, x0, y0, z0):
    """
    Create a source function for a sinusoidal point source with constant arbitrary amplitude, frequency, and location. 3D version.
    Parameters:
    amplitude: (float)
        Amplitude of the source
    frequency: (float)
        Frequency of the source
    x0, y0, z0: (float, float, float)
        Source location

    Returns:
        fun: (function(x: float, y: float, z: float, t: float) -> float)
        A function representing the point source with arguments (x, y, z, t).
    """
    def fun(x, y, z, t):
        # Swap x0 and y0 because grid is indexed as u[y_i, x_i, z_i]
        if np.sqrt((x - y0)**2 + (y - x0)**2 + (z - z0)**2) < 0.5:
            return amplitude * np.sin(2 * np.pi * frequency * t)
        return 0.0

    return fun


def create_impulse_source_2D(amplitude, x0, y0, t0):
    """
    Create a source function for an impulse source with arbitrary amplitude and location. 2D version.
    Parameters:
    amplitude: (float)
        Amplitude of the source
    x0, y0: (float, float)
    t0: (float)
        Time of the impulse

    Returns:
    fun: (function(x: float, y: float, t: float) -> float)
        A function representing the point source with arguments (x, y, t).
    """
    def fun(x, y, t):
        # Width of the Gaussian approximation to the delta function
        width = 1e-2

        # Approximating Dirac delta function with a Gaussian in space and a delta in time
        # Swap x0 and y0 because grid is indexed as u[y_i, x_i]
        spatial_term = np.exp(-((x - y0)**2 + (y - x0)**2) / width**2)
        temporal_term = 1.0 if np.isclose(t, t0) else 0.0

        return amplitude * spatial_term * temporal_term

    return fun


# TODO check if this is correct
def create_impulse_source_3D(amplitude, x0, y0, z0, t0):
    """
    Create a source function for an impulse source with arbitrary amplitude and location. 3D version.
    Parameters:
    amplitude: (float)
        Amplitude of the source
    x0, y0, z0: (float, float, float)
    t0: (float)
        Time of the impulse

    Returns:
    fun: (function(x: float, y: float, z: float, t: float) -> float)
        A function representing the point source with arguments (x, y, z, t).
    """
    def fun(x, y, z, t):
        # Width of the Gaussian approximation to the delta function
        width = 1e-2

        # Approximating Dirac delta function with a Gaussian in space and a delta in time
        # Swap x0 and y0 because grid is indexed as u[y_i, x_i, z_i]
        spatial_term = np.exp(-((x - y0)**2 + (y - x0) **
                              2 + (z - z0)**2) / width**2)
        temporal_term = 1.0 if np.abs(t - t0) < 1e-5 else 0.0

        return amplitude * spatial_term * temporal_term

    return fun
