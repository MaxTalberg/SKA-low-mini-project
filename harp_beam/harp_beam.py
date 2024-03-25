## MPHil in data intensive science
# Astronomy in the SKA-era mini project
# SKA-low mini project
# 01.03.2024
# Q. Gueuning (qdg20@cam.ac.uk) and O. O'Hara
# see license file attached

import numpy as np
import scipy.io
from scipy.special import lpmv, factorial

def legendre(deg, x):
    """
    Calculate the associated Legendre function for integer orders and degree at value x.

    Parameters
    ----------
    deg : float
        Degree of the Legendre function.
    x : float
        Position to evaluate function

    Returns
    -------
        return : np.array
        Legendre function for all integer orders from 0 to deg.
    """
    return np.asarray([lpmv(i, deg, x) for i in range(deg + 1)])[:, 0, :]

def legendre3(n, u):
    """
    Calculate all associated Legendre functions up to max order n at value x.

    Parameters
    ----------
    deg : float
        Max degree of the Legendre function.
    x : float
        Position to evaluate function

    Returns
    -------
        return : np.array
        Legendre functions (Pnm,Pnm/costheta,dPnmdsintheta) for all integer orders from 0 to deg.
    """
    pn = legendre(n, u)
    pnd = np.divide(pn, np.ones_like(n + 1) * np.sqrt(1 - u ** 2))

    mv = np.arange(n)

    dpns = np.zeros((n + 1, len(u[0])))
    dpns[:-1, :] = np.multiply(-(mv[:, None]), np.divide(u, 1 - u ** 2)) * pn[mv, :] - pnd[mv + 1, :]
    dpns[n, :] = np.multiply(-n, np.divide(u, 1 - u ** 2)) * pn[n, :];
    dpns *= np.sqrt(1 - u ** 2)
    return pn, pnd, dpns

def smodes_eval(order, alpha_tm, alpha_te, theta, phi):
    """
    Calculate spherical wave modes TE and TM according to definitions in the book J.E. Hansen, Spherical near-field measurements

    Parameters
    ----------
    order : float
        Max order of the Legendre function.
    alpha_tm : np.array, complex double
        coefficients for TM modes, 3d array of size (num_mbf, 2 * max_order + 1, max_order)
    alpha_te : np.array, complex double
        coefficients for TE modes, 3d array of size (num_mbf, 2 * max_order + 1, max_order)
    theta : np.araay, float
        zenith angle
    phi : np.array, float
        azimuth angle
    Returns
    -------
        return : np.array, complex double
        gvv 
        ghh 
    """
    tol = 1e-5
    theta[theta < tol] = tol

    Na = len(alpha_tm[:, 1, 1])

    u = np.cos(theta.T)
    gvv = np.zeros((Na, theta.shape[0]), dtype=complex)
    ghh = np.zeros((Na, theta.shape[0]), dtype=complex)

    EE = np.exp(1j * np.arange(-order, order + 1) * phi).T
    for n in range(1, order + 1):
        mv = np.arange(-n, n + 1)
        pn, pnd, dpns = legendre3(n, u)
        pmn = np.row_stack((np.flipud(pnd[1:]), pnd))
        dpmn = np.row_stack((np.flipud(dpns[1:]), dpns))

        Nv = 2 * np.pi * n * (n + 1) / (2 * n + 1) * factorial(n + np.abs(mv)) / factorial(n - abs(mv))
        Nf = np.sqrt(2 * Nv)
        ee = EE[mv + order]
        qq = -ee * dpmn
        dd = ee * pmn

        mat1 = np.multiply(np.ones((Na, 1)), 1 / Nf)
        mat2 = np.multiply(np.ones((Na, 1)), mv * 1j / Nf)
        an_te_polY = alpha_te[:, n - 1, (mv + n)]
        an_tm_polY = alpha_tm[:, n - 1, (mv + n)]

        gvv += np.matmul(an_tm_polY * mat1, qq) - np.matmul(an_te_polY * mat2, dd)
        ghh += np.matmul(an_tm_polY * mat2, dd) + np.matmul(an_te_polY * mat1, qq)

    return gvv.T, ghh.T

def wrapTo2Pi(phi):
    return phi % (2 * np.pi)

def compute_EEPs(theta, phi):

    ind = theta < 0
    theta[ind] = -theta[ind]
    phi[ind] = wrapTo2Pi(phi[ind] + np.pi)

    freq = 100
    c0 = 299792458  # speed of light
    k0 = 2 * np.pi * freq / c0 * 10**6  # wavenumber
    antenna = 'SKALA41'  # antenna name
    layout = 'random'  # array layout
    data_folder = 'harp_beam'
    filename_eep = f"data_EEPs_{antenna}_{layout}_{freq}MHz.mat"
    mat = scipy.io.loadmat(filename_eep)

    max_order = int(mat['max_order'])
    num_mbf = int(mat['num_mbf'])
    coeffs_polX = np.array(mat['coeffs_polX'])
    coeffs_polY = np.array(mat['coeffs_polY'])
    alpha_te = np.array(mat['alpha_te'])
    alpha_tm = np.array(mat['alpha_tm'])
    pos_ant = np.array(mat['pos_ant'])
    x_pos = pos_ant[:,0]
    y_pos = pos_ant[:,1]

    # reshaping
    alpha_te = np.ndarray.transpose(np.reshape(alpha_te, (num_mbf, 2 * max_order + 1, max_order), order='F'), (0, 2, 1))
    alpha_tm = np.ndarray.transpose(np.reshape(alpha_tm, (num_mbf, 2 * max_order + 1, max_order), order='F'), (0, 2, 1))

    num_dir = len(theta)
    num_ant = len(pos_ant)
    num_beam = len(coeffs_polY[0])
    num_mbf = len(alpha_tm)

    ux = np.sin(theta) * np.cos(phi)
    uy = np.sin(theta) * np.sin(phi)

    v_mbf_theta, v_mbf_phi = smodes_eval(max_order, alpha_tm, alpha_te, theta, phi)

    # Beam assembling
    v_theta_polY = np.zeros((num_dir, num_beam), dtype=np.complex128)
    v_phi_polY = np.zeros((num_dir, num_beam), dtype=np.complex128)
    v_theta_polX = np.zeros((num_dir, num_beam), dtype=np.complex128)
    v_phi_polX = np.zeros((num_dir, num_beam), dtype=np.complex128)
    phase_factor = np.exp(1j * k0 * (ux*x_pos + uy*y_pos))
    for i in range(num_mbf):
        p_thetai = v_mbf_theta[:, i]
        p_phii = v_mbf_phi[:, i]

        c_polY = np.matmul(phase_factor,coeffs_polY[np.arange(num_ant) * num_mbf + i, :])
        c_polX = np.matmul(phase_factor,coeffs_polX[np.arange(num_ant) * num_mbf + i, :])

        v_theta_polY += p_thetai[:,None] * c_polY
        v_phi_polY += p_phii[:,None] * c_polY
        v_theta_polX += p_thetai[:,None] * c_polX
        v_phi_polX += p_phii[:,None] * c_polX

    v_theta_polY *= np.conj(phase_factor)
    v_phi_polY *= np.conj(phase_factor)
    v_theta_polX *= np.conj(phase_factor)
    v_phi_polX *= np.conj(phase_factor)

    return v_theta_polY, v_phi_polY, v_theta_polX, v_phi_polX

# convert to dBV
def to_dBV(magnitude):
    '''
    Convert magnitude to dBV
    -------------------------
    magnitude: float
        Magnitude of EEPs

    Returns
    -------
    float
        Magnitude in dBV
    '''
    return 20*np.log10(magnitude)

# Q2 Power EEPs
def power_EEPs(v_theta_polY, v_phi_polY, v_theta_polX, v_phi_polX):

    # Calculate magnitude of EEPs
    EEPs_theta_polY = np.abs(v_theta_polY)
    EEPs_phi_polY = np.abs(v_phi_polY)
    EEPs_theta_polX = np.abs(v_theta_polX)
    EEPs_phi_polX = np.abs(v_phi_polX)

    # Calculate EEPs
    EEPs_polY = np.sqrt(EEPs_theta_polY**2 + EEPs_phi_polY**2)
    EEPs_polX = np.sqrt(EEPs_theta_polX**2 + EEPs_phi_polX**2)

    # Calculate AEPs
    AEP_polY = np.mean(EEPs_polY, axis=1)
    AEP_polX = np.mean(EEPs_polX, axis=1)

    # Convert to dBV
    EEPs_polY_dBV = to_dBV(EEPs_polY)
    EEPs_polX_dBV = to_dBV(EEPs_polX)
    AEP_polY_dBV = to_dBV(AEP_polY)
    AEP_polX_dBV = to_dBV(AEP_polX)

    return EEPs_polY_dBV, EEPs_polX_dBV, AEP_polY_dBV, AEP_polX_dBV

# Q3/4 StEFCal algorithm
def stefcal(M, R, g_sol, max_iteration=1000, threshold=1e-6):
    abs_error = []
    amp_error = []
    phase_error = []
    # Number of antennas
    # Initial gain matrix G
    G = np.eye(M.shape[0], dtype=complex) # Identity matrix

    # Iterative loop
    for i in range(max_iteration):
        # Last iteration of G for comparison
        G_prev = G.copy()

        for p in range(G.shape[0]):  # Loop over antennas p
            z = np.dot(G_prev, M[:, p])  # Use all rows of M for antenna p
            gp = np.dot(np.conjugate(R[:, p]), z) / np.dot(np.conjugate(z), z)  # Calculate new gain for antenna p
            G[p, p] = gp  # Update the gain for antenna p in the matrix


        # Convergence check even iterations
        if i % 2 == 0:
            
            delta_G = np.linalg.norm(G - G_prev, 'fro') / np.linalg.norm(G, 'fro')
            if delta_G < threshold:
                print(f"Convergence reached after {i+1} iterations.")
                break
            else:
                G = (G + G_prev) / 2
                # Calculate errors
                G_diagonal = G.diagonal().reshape(-1, 1)
                abs_error.append(np.linalg.norm(np.abs(G_diagonal - g_sol), 'fro') / np.linalg.norm(g_sol, 'fro'))
                amp_error.append(np.linalg.norm(np.abs(G_diagonal) - np.abs(g_sol), 'fro') / np.linalg.norm(g_sol, 'fro'))
                phase_error.append(np.linalg.norm(np.angle(G_diagonal) - np.angle(g_sol), 'fro')/ np.linalg.norm(g_sol, 'fro'))

    return G, abs_error, amp_error, phase_error

#Q5 Beamforming
def beamforming(G_diag, EEP, pos_ant, theta, phi, theta0, phi0):
    freq = 100  # Frequency in MHz
    c0 = 299792458  # Speed of light in m/s
    lambda_0 = c0 / (freq * 10**6)  # Wavelength
    k = 2 * np.pi / lambda_0  # Wavenumber

    # Compute the weights for the steering direction
    weights = np.exp(1j * k * (np.sin(theta0) * np.cos(phi0) * pos_ant[:, 0] +
                               np.sin(theta0) * np.sin(phi0) * pos_ant[:, 1]))

    # Check if phi is a scalar or an array
    if np.isscalar(phi):
        # Phi is a scalar, simplify the computation
        pattern = np.zeros_like(theta, dtype=np.complex64)
        for i, theta_val in enumerate(theta):
            phase_factor = np.exp(-1j * k * (np.sin(theta_val) * np.cos(phi) * pos_ant[:, 0] +
                                             np.sin(theta_val) * np.sin(phi) * pos_ant[:, 1]))
            array_factor = np.sum(weights * G_diag * EEP * phase_factor)
            pattern[i] = array_factor
    else:
        # Phi is an array, proceed with the full computation
        pattern = np.zeros((len(theta), len(phi)), dtype=np.complex64)
        for i, theta_val in enumerate(theta):
            for j, phi_val in enumerate(phi):
                phase_factor = np.exp(-1j * k * (np.sin(theta_val) * np.cos(phi_val) * pos_ant[:, 0] +
                                                 np.sin(theta_val) * np.sin(phi_val) * pos_ant[:, 1]))
                array_factor = np.sum(weights * G_diag * EEP * phase_factor)
                pattern[i, j] = array_factor

    return pattern

def compute_beamforming(G, v_theta_polY, v_phi_polY, v_theta_polX, v_phi_polX, pos_ant, theta, phi, theta0, phi0):
    
    AP_theta_polY = np.abs(beamforming(G, v_theta_polY, pos_ant, theta, phi, theta0, phi0))
    AP_phi_polY = np.abs(beamforming(G, v_phi_polY, pos_ant, theta, phi, theta0, phi0))
    AP_polY = to_dBV(AP_theta_polY**2 + AP_phi_polY**2)

    AP_theta_polX = np.abs(beamforming(G, v_theta_polX, pos_ant, theta, phi, theta0, phi0))
    AP_phi_polX = np.abs(beamforming(G, v_phi_polX, pos_ant, theta, phi, theta0, phi0))
    AP_polX = to_dBV(AP_theta_polX**2 + AP_phi_polX**2)

    return AP_polY, AP_polX
