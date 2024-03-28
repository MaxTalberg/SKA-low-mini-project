## MPHil in data intensive science
# Astronomy in the SKA-era mini project
# SKA-low mini project
# 01.03.2024
# Q. Gueuning (qdg20@cam.ac.uk) and O. O'Hara
# see license file attached

import numpy as np
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
        Legendre functions (Pnm,Pnm/costheta,dPnmdsintheta) for all
        integer orders from 0 to deg.
    """
    pn = legendre(n, u)
    pnd = np.divide(pn, np.ones_like(n + 1) * np.sqrt(1 - u**2))

    mv = np.arange(n)

    dpns = np.zeros((n + 1, len(u[0])))
    dpns[:-1, :] = (
        np.multiply(-(mv[:, None]), np.divide(u, 1 - u**2)) * pn[mv, :] - pnd[mv + 1, :]
    )
    dpns[n, :] = np.multiply(-n, np.divide(u, 1 - u**2)) * pn[n, :]
    dpns *= np.sqrt(1 - u**2)
    return pn, pnd, dpns


def smodes_eval(order, alpha_tm, alpha_te, theta, phi):
    """
    Calculate spherical wave modes TE and TM according to definitions in
    the book J.E. Hansen, Spherical near-field measurements

    Parameters
    ----------
    order : float
        Max order of the Legendre function.
    alpha_tm : np.array, complex double
        coefficients for TM modes, 3d array of size
        (num_mbf, 2 * max_order + 1, max_order)
    alpha_te : np.array, complex double
        coefficients for TE modes, 3d array of size
        (num_mbf, 2 * max_order + 1, max_order)
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

        Nv = (
            2
            * np.pi
            * n
            * (n + 1)
            / (2 * n + 1)
            * factorial(n + np.abs(mv))
            / factorial(n - abs(mv))
        )
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
    """
    Wraps the given angle phi to the range [0, 2*pi).

    Parameters
    ----------
    phi : float or np.ndarray
        Angle or array of angles in radians.

    Returns
    -------
    wrapped_phi : float or np.ndarray
        Angle(s) wrapped to the range [0, 2*pi).
        The returned value has the same type as the input `phi`.
    """
    return phi % (2 * np.pi)


def compute_EEPs(
    theta,
    phi,
    alpha_te,
    alpha_tm,
    coeffs_polX,
    coeffs_polY,
    pos_ant,
    num_mbf,
    max_order,
    k0,
):
    """
    Computes the Equivalent Electric Field Patterns (EEPs) for a
    given antenna array configuration, considering both TE and TM modes.

    Parameters
    ----------
    theta : np.ndarray
        Array of elevation angles in radians, where negative values are
        reflected and wrapped accordingly.
    phi : np.ndarray
        Array of azimuth angles in radians, wrapped to the range [0, 2*pi)
        for negative theta.
    alpha_te : np.ndarray
        Coefficients for TE modes, reshaped according to
        'num_mbf', 'max_order', and beam directions.
    alpha_tm : np.ndarray
        Coefficients for TM modes, similarly reshaped.
    coeffs_polX : np.ndarray
        Coefficients for polarisation X, considering antenna positions
        and mode basis functions (MBFs).
    coeffs_polY : np.ndarray
        Coefficients for polarisation Y, with a similar structure to
        'coeffs_polX'.
    pos_ant : np.ndarray
        Positions of the antennas in the array.
    num_mbf : int
        Number of mode basis functions.
    max_order : int
        Maximum order of the modes.
    k0 : float
        Wave number in free space.

    Returns
    -------
    v_theta_polY : np.ndarray
        Vertical component of the electric field pattern for polarisation Y.
    v_phi_polY : np.ndarray
        Horizontal component of the electric field pattern for polarisation Y.
    v_theta_polX : np.ndarray
        Vertical component of the electric field pattern for polarisation X.
    v_phi_polX : np.ndarray
        Horizontal component of the electric field pattern for polarisation X.

    Notes
    -----
    The function computes the EEPs by first correcting the input angles and
    then calculating the field components using the provided mode coefficients,
    antenna positions, and the wave number. The output patterns are
    provided for each polarisation and direction.
    """
    ind = theta < 0
    theta[ind] = -theta[ind]
    phi[ind] = wrapTo2Pi(phi[ind] + np.pi)

    # unpack postions
    x_pos = pos_ant[:, 0]
    y_pos = pos_ant[:, 1]

    # reshaping
    alpha_te = np.ndarray.transpose(
        np.reshape(alpha_te, (num_mbf, 2 * max_order + 1, max_order), order="F"),
        (0, 2, 1),
    )
    alpha_tm = np.ndarray.transpose(
        np.reshape(alpha_tm, (num_mbf, 2 * max_order + 1, max_order), order="F"),
        (0, 2, 1),
    )

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
    phase_factor = np.exp(1j * k0 * (ux * x_pos + uy * y_pos))
    for i in range(num_mbf):
        p_thetai = v_mbf_theta[:, i]
        p_phii = v_mbf_phi[:, i]

        c_polY = np.matmul(
            phase_factor, coeffs_polY[np.arange(num_ant) * num_mbf + i, :]
        )
        c_polX = np.matmul(
            phase_factor, coeffs_polX[np.arange(num_ant) * num_mbf + i, :]
        )

        v_theta_polY += p_thetai[:, None] * c_polY
        v_phi_polY += p_phii[:, None] * c_polY
        v_theta_polX += p_thetai[:, None] * c_polX
        v_phi_polX += p_phii[:, None] * c_polX

    v_theta_polY *= np.conj(phase_factor)
    v_phi_polY *= np.conj(phase_factor)
    v_theta_polX *= np.conj(phase_factor)
    v_phi_polX *= np.conj(phase_factor)

    return v_theta_polY, v_phi_polY, v_theta_polX, v_phi_polX


def to_dBV(magnitude):
    """
    Convert a given magnitude to decibels relative to 1 volt (dBV).

    Parameters
    ----------
    magnitude : float or np.ndarray
        The magnitude (or array of magnitudes) of the signals whose level is to
        be converted to dBV.

    Returns
    -------
    float or np.ndarray
        The level of the input magnitudes expressed in dBV.
    """
    return 20 * np.log10(magnitude)


## Q2
def power_EEPs(v_theta_polY, v_phi_polY, v_theta_polX, v_phi_polX):
    """
    Calculate the power Embedding Element Patterns (EEP) for
    two polarizations (X and Y) and convert these patterns and their
    averages (AEPs) to dBV.

    Parameters
    ----------
    v_theta_polY : np.ndarray
        The vertical component of the electric field pattern for polarisation Y.
    v_phi_polY : np.ndarray
        The horizontal component of the electric field pattern for polarisation Y.
    v_theta_polX : np.ndarray
        The vertical component of the electric field pattern for polarisation X.
    v_phi_polX : np.ndarray
        The horizontal component of the electric field pattern for polarisation X.

    Returns
    -------
    EEPs_polY_dBV : np.ndarray
        EEPs for polarisation Y in dBV.
    EEPs_polX_dBV : np.ndarray
        EEPs for polarisation X in dBV.
    AEP_polY_dBV : np.ndarray
        Average EEP for polarisation Y in dBV.
    AEP_polX_dBV : np.ndarray
        Average EEP for polarisation X in dBV.

    Notes
    -----
    The function calculates the magnitude of the EEPs for each component
    and polarisation, computes the total EEPs by summing the squared
    magnitudes of the vertical and horizontal components and calculates
    the average EEPs (AEPs) across all directions.
    It then converts these values to dBV using the `to_dBV` function.
    """
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


## Q3/4
def stefcal(M, R, g_sol, max_iteration=1000, threshold=1e-5, algorithm2=False):
    """
    Perform the StEFCal (Simplified Tikhonov-based Efficient Calibration)
    algorithm to solve for the complex gain solutions of an antenna array in
    radio interferometry.

    Parameters
    ----------
    M : np.ndarray
        The measured visibility matrix, representing the cross-correlations between
        antennas' signals.
    R : np.ndarray
        The model visibility matrix, derived from a known sky model and representing
        expected correlations.
    g_sol : np.ndarray
        The true gain solutions for the antennas, used to compute errors for
        convergence analysis.
    max_iteration : int, optional
        The maximum number of iterations to run the algorithm for.
        Default is 1000.
    threshold : float, optional
        The convergence threshold, used to determine when the algorithm has
        sufficiently converged. Default is 1e-5.
    algorithm2 : bool, optional
        Flag to determine which version of the algorithm to use. If False, uses G[i-1]
        for calculations; if True, uses G[i]. Default is False.

    Returns
    -------
    tuple:
        G : np.ndarray
            The final estimated gains matrix, where each diagonal element represents the
            gain for one antenna.
        convergence : list
            A list of convergence values, one for every second iteration, measuring the
            relative change in G.
        abs_gain_error : list
            A list of the absolute gain errors, measuring the difference between the
            estimated and true gains.
        abs_amp_error : list
            A list of the absolute amplitude errors, comparing the magnitudes of the
            estimated and true gains.
        abs_phase_error : list
            A list of the absolute phase errors, comparing the phases of the
            estimated and true gains.

    The function iteratively updates the gain solutions by minimising the difference
    between the measured and model visibilities, adjusting the gains to better fit the
    model to the measurements. Convergence is checked every second iteration and various
    errors are calculated to evaluate the accuracy of the
    gain estimates.
    """
    convergence = []
    abs_gain_error = []
    abs_amp_error = []
    abs_phase_error = []

    # Number of antennas
    N = R.shape[0]

    # Initial gain matrix G
    G = np.eye(N, dtype=complex)  # Identity matrix

    # Iterative loop
    for i in range(max_iteration):

        # Last iteration of G for comparison
        G_prev = G.copy()

        for p in range(N):  # Loop over antennas p
            if algorithm2:
                z = np.dot(G, M[:, p])  # Use G[i]
            else:
                z = np.dot(G_prev, M[:, p])  # Use G[i-1]
            gp = np.dot(np.conjugate(R[:, p]), z) / np.dot(
                np.conjugate(z), z
            )  # Calculate new gain for antenna p
            G[p, p] = gp  # Update the gain for antenna p in the matrix

        # Convergence check even iterations
        if i % 2 == 0:

            delta_G = np.linalg.norm(G - G_prev, "fro") / np.linalg.norm(G, "fro")
            convergence.append(delta_G)
            if delta_G < threshold:
                print(f"Convergence reached after {i+1} iterations.")
                break
            else:
                G = (G + G_prev) / 2

                # Calculate errors
                G_diagonal = G.diagonal().reshape(-1, 1)
                abs_gain_error.append(
                    np.linalg.norm(np.abs(G_diagonal - g_sol), "fro")
                    / np.linalg.norm(g_sol, "fro")
                )
                abs_amp_error.append(
                    np.linalg.norm(np.abs(G_diagonal) - np.abs(g_sol), "fro")
                    / np.linalg.norm(g_sol, "fro")
                )
                abs_phase_error.append(
                    np.linalg.norm(np.angle(G_diagonal) - np.angle(g_sol), "fro")
                    / np.linalg.norm(g_sol, "fro")
                )

    return G, convergence, abs_gain_error, abs_amp_error, abs_phase_error


## Q5
def beamforming(G_diag, EEP, pos_ant, k, theta, phi, theta0, phi0):
    """
    Perform beamforming to compute the array pattern for a given steering direction
    (theta0, phi0).

    Parameters
    ----------
    G_diag : np.ndarray
        The diagonal elements of the gains matrix, representing the gain for each
        antenna.
    EEP : np.ndarray
        The Equivalent Electric Field Pattern for the antennas in the array for a
        specific polarization.
    pos_ant : np.ndarray
        The positions of the antennas in the array.
    k : float
        The wave number in free space.
    theta : np.ndarray
        The elevation angles for which the array pattern is computed.
    phi : float or np.ndarray
        The azimuth angle(s) for which the array pattern is computed.
        Can be a scalar or an array.
    theta0 : float
        The elevation angle of the steering direction.
    phi0 : float
        The azimuth angle of the steering direction.

    Returns
    -------
    np.ndarray
        The computed array pattern for the specified theta and phi.
        The array is 1D if phi is a scalar, or 2D if phi is an array,
        corresponding to the meshgrid formed by theta and phi.

    The function calculates the beamforming pattern by applying phase shifts to
    the EEPs based on the antenna positions and the desired steering direction.
    It accounts for both scalar and array inputs for the azimuth angle phi.
    """
    # Compute the weights for the steering direction
    weights = np.exp(
        1j
        * k
        * (
            np.sin(theta0) * np.cos(phi0) * pos_ant[:, 0]
            + np.sin(theta0) * np.sin(phi0) * pos_ant[:, 1]
        )
    ).reshape(-1, 1)

    # Check if phi is a scalar or an array
    if np.isscalar(phi):
        # Phi is a scalar, simplify the computation
        pattern = np.zeros_like(theta, dtype=np.complex64)
        for i, theta_val in enumerate(theta):
            phase_factor = np.exp(
                -1j
                * k
                * (
                    np.sin(theta_val) * np.cos(phi) * pos_ant[:, 0]
                    + np.sin(theta_val) * np.sin(phi) * pos_ant[:, 1]
                )
            ).reshape(-1, 1)
            array_factor = np.sum(weights * G_diag * EEP[i, :] * phase_factor)
            pattern[i] = array_factor
    else:
        # Phi is an array, proceed with the full computation
        pattern = np.zeros((len(theta), len(phi)), dtype=np.complex64)
        for i, theta_val in enumerate(theta):
            for j, phi_val in enumerate(phi):
                phase_factor = np.exp(
                    -1j
                    * k
                    * (
                        np.sin(theta_val) * np.cos(phi_val) * pos_ant[:, 0]
                        + np.sin(theta_val) * np.sin(phi_val) * pos_ant[:, 1]
                    )
                ).reshape(-1, 1)
                array_factor = np.sum(weights * G_diag * EEP[i, :] * phase_factor)
                pattern[i, j] = array_factor

    return pattern


## Q5
def compute_beamforming(
    G,
    v_theta_polY,
    v_phi_polY,
    v_theta_polX,
    v_phi_polX,
    pos_ant,
    k,
    theta,
    phi,
    theta0,
    phi0,
):
    """
    Compute the beamforming patterns for both Y and X polarisations
    and convert the results to dBV.

    Parameters
    ----------
    G : np.ndarray
        The gains matrix for the antennas in the array, with diagonal elements
        representing the gains per antenna.
    v_theta_polY : np.ndarray
        The vertical component of the electric field pattern for polarisation Y.
    v_phi_polY : np.ndarray
        The horizontal component of the electric field pattern for polarisation Y.
    v_theta_polX : np.ndarray
        The vertical component of the electric field pattern for polarisation X.
    v_phi_polX : np.ndarray
        The horizontal component of the electric field pattern for polarisation X.
    pos_ant : np.ndarray
        The positions of the antennas in the array.
    k : float
        The wave number in free space.
    theta : np.ndarray
        The elevation angles for which the array pattern is computed.
    phi : float or np.ndarray
        The azimuth angle(s) for which the array pattern is computed.
        Can be a scalar or an array.
    theta0 : float
        The elevation angle of the steering direction.
    phi0 : float
        The azimuth angle of the steering direction.

    Returns
    -------
    tuple of np.ndarray
        The computed array patterns for polarisations Y and X, converted to dBV.
        Each pattern is 1D if phi is a scalar, or 2D if phi is an array,
        corresponding to the meshgrid formed by theta and phi.

    This function utilises the `beamforming` function to calculate the array patterns
    for both Y and X polarisations based on the provided EEPs and gains.
    It then computes the absolute patterns and converts them to dBV.
    """
    AP_theta_polY = np.abs(
        beamforming(G, v_theta_polY, pos_ant, k, theta, phi, theta0, phi0)
    )
    AP_phi_polY = np.abs(
        beamforming(G, v_phi_polY, pos_ant, k, theta, phi, theta0, phi0)
    )
    AP_polY = to_dBV(np.sqrt(AP_theta_polY**2 + AP_phi_polY**2))

    AP_theta_polX = np.abs(
        beamforming(G, v_theta_polX, pos_ant, k, theta, phi, theta0, phi0)
    )
    AP_phi_polX = np.abs(
        beamforming(G, v_phi_polX, pos_ant, k, theta, phi, theta0, phi0)
    )
    AP_polX = to_dBV(np.sqrt(AP_theta_polX**2 + AP_phi_polX**2))

    return AP_polY, AP_polX
