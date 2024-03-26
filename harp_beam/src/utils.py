import numpy as np
import scipy.io

def compute_k0(freq, c0):
    """Compute the wavenumber based on frequency and speed of light."""
    return 2 * np.pi * freq / c0 * 10**6

def load_antenna_data(filename):
    """Load antenna data from a .mat file."""
    mat = scipy.io.loadmat(filename)
    max_order = int(mat['max_order'])
    num_mbf = int(mat['num_mbf'])
    coeffs_polX = np.array(mat['coeffs_polX'])
    coeffs_polY = np.array(mat['coeffs_polY'])
    alpha_te = np.array(mat['alpha_te'])
    alpha_tm = np.array(mat['alpha_tm'])
    pos_ant = np.array(mat['pos_ant'])
    return max_order, num_mbf, coeffs_polX, coeffs_polY, alpha_te, alpha_tm, pos_ant

def load_arrays_from_mat(filename):
    """
    Load specific arrays from a .mat file.

    Parameters:
    - filename: The path to the .mat file.

    Returns:
    - A dictionary containing the loaded arrays.
    """
    # Load the .mat file
    mat = scipy.io.loadmat(filename)

    # Extract the desired arrays
    R = np.array(mat['R'])  # Covariance matrix
    M_AEP = np.array(mat['M_AEP'])  # Model matrix using AEP
    M_EEPs = np.array(mat['M_EEPs'])  # Model matrix using all EEPs
    g_sol = np.array(mat['g_sol'])  # Exact gain solution
    g_AEP = np.array(mat['g_AEP'])  # Estimation using M_AEP
    g_EEPs = np.array(mat['g_EEPs'])  # Estimation using M_EEPs

    # Return the arrays in a dictionary for easy access
    return {
        'R': R,
        'M_AEP': M_AEP,
        'M_EEPs': M_EEPs,
        'g_sol': g_sol,
        'g_AEP': g_AEP,
        'g_EEPs': g_EEPs
    }