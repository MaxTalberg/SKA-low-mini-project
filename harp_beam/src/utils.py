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
