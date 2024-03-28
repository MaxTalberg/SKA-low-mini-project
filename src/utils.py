import numpy as np
import scipy.io


def compute_k0(freq, c0):
    """
    Computes the wavenumber (k0) given the frequency and the speed of light.

    Parameters
    ----------
    freq : float
        The frequency of operation in MHz (Megahertz).
    c0 : float
        The speed of light in a vacuum, typically 299792458 meters per second.

    Returns
    -------
    float
        The wavenumber corresponding to the given frequency and speed of light,
        calculated as k0 = 2 * pi * freq / c0, where freq is converted from MHz to Hz.
    """
    return 2 * np.pi * freq / c0 * 10**6


def load_antenna_data(filename):
    """
    Loads antenna configuration data from a MATLAB .mat file.

    Parameters
    ----------
    filename : str
        The path to the .mat file containing antenna configuration data.

    Returns
    -------
    tuple:
        max_order : int
            The maximum order of the modes used in the antenna configuration.
        num_mbf : int
            The number of mode basis functions.
        coeffs_polX : np.ndarray
            Coefficients for polarisation X.
        coeffs_polY : np.ndarray
            Coefficients for polarisation Y.
        alpha_te : np.ndarray
            Coefficients for transverse electric (TE) modes.
        alpha_tm : np.ndarray
            Coefficients for transverse magnetic (TM) modes.
        pos_ant : np.ndarray
            Positions of the antennas in the array.
    """
    mat = scipy.io.loadmat(filename)
    max_order = int(mat["max_order"])
    num_mbf = int(mat["num_mbf"])
    coeffs_polX = np.array(mat["coeffs_polX"])
    coeffs_polY = np.array(mat["coeffs_polY"])
    alpha_te = np.array(mat["alpha_te"])
    alpha_tm = np.array(mat["alpha_tm"])
    pos_ant = np.array(mat["pos_ant"])
    return max_order, num_mbf, coeffs_polX, coeffs_polY, alpha_te, alpha_tm, pos_ant


def load_arrays_from_mat(filename):
    """
    Loads specified arrays from a MATLAB .mat file, typically used
    in signal processing or antenna array simulations,
    into a Python dictionary for easy access.

    Parameters
    ----------
    filename : str
        The path to the .mat file from which the data is to be loaded.
        This file is expected to contain specific matrices that represent
        various parameters or results from simulations or experimental setups
        in the context of signal processing or antenna array analysis.

    Returns
    -------
    dict
        A dictionary containing the following key-value pairs:
        - 'R': np.ndarray, the covariance matrix representing signal or
        noise correlations between array elements.
        - 'M_AEP': np.ndarray, the model matrix derived using the
        Average Electric Field Pattern (AEP).
        - 'M_EEPs': np.ndarray, the model matrix derived using all
        Equivalent Electric Field Patterns (EEPs).
        - 'g_sol': np.ndarray, the exact gain solution, typically used
        as a reference or true value.
        - 'g_AEP': np.ndarray, the gain estimation derived from the
        'M_AEP' model matrix.
        - 'g_EEPs': np.ndarray, the gain estimation derived from the
        'M_EEPs' model matrix.
    """
    # Load the .mat file
    mat = scipy.io.loadmat(filename)

    # Extract the desired arrays
    R = np.array(mat["R"])  # Covariance matrix
    M_AEP = np.array(mat["M_AEP"])  # Model matrix using AEP
    M_EEPs = np.array(mat["M_EEPs"])  # Model matrix using all EEPs
    g_sol = np.array(mat["g_sol"])  # Exact gain solution
    g_AEP = np.array(mat["g_AEP"])  # Estimation using M_AEP
    g_EEPs = np.array(mat["g_EEPs"])  # Estimation using M_EEPs

    # Return the arrays in a dictionary for easy access
    return {
        "R": R,
        "M_AEP": M_AEP,
        "M_EEPs": M_EEPs,
        "g_sol": g_sol,
        "g_AEP": g_AEP,
        "g_EEPs": g_EEPs,
    }
