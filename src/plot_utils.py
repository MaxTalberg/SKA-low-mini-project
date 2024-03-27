import warnings
import numpy as np
import matplotlib.pyplot as plt


def plot_power_EEPs_and_AEP(
    theta, magnitude_EEP_polY, magnitude_EEP_polX, AEP_polY, AEP_polX
):
    """
    Plots the Equivalent Electric Field Patterns (EEPs) and
    Average Electric Field Patterns (AEPs) for Y and X polarizations.

    Parameter
    ---------
    theta : np.ndarray
        The angular positions (in radians) at which the EEPs and AEPs are evaluated,
        ypically covering -pi/2 to pi/2 (or -90 to 90 degrees).
    magnitude_EEP_polY : np.ndarray
        The magnitudes of the EEPs for Y polarisation across the
        specified theta angles for each antenna element.
    magnitude_EEP_polX : np.ndarray
        The magnitudes of the EEPs for X polarisation across the
        specified theta angles for each antenna element.
    AEP_polY : np.ndarray
        The magnitude of the AEP for Y polarisation, averaged over all antenna elements.
    AEP_polX : np.ndarray
        The magnitude of the AEP for X polarisation, averaged over all antenna elements.

    This function generates two plots, one for each polarisation (Y and X), s
    howing the variation of the electric field with the angle theta.
    The EEPs for each antenna element are plotted with partial transparency
    to visualise their distribution, while the AEP is highlighted as a distinct,
    solid line for each polarisation.
    """
    # Plot EEPs for polY
    plt.figure(figsize=(12, 5))

    for i in range(magnitude_EEP_polY.shape[1]):
        if i == 0:
            # Label only the first EEP line for the legend
            plt.plot(
                theta * 180 / np.pi,
                magnitude_EEP_polY[:, i],
                "r-",
                alpha=0.1,
                label="EEPs",
            )
        else:
            # Plot the rest without a label
            plt.plot(theta * 180 / np.pi, magnitude_EEP_polY[:, i], "r-", alpha=0.1)

    plt.plot(theta * 180 / np.pi, AEP_polY, "black", label="AEP")
    plt.title("polyY, Y-plane, freq =100MHz")
    plt.xlabel("Theta (deg)")
    plt.ylabel("E-field (dBV)")
    plt.xlim(-90, 90)
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig('./plots/plot_power_EEPs_and_AEP_polY.png')
    plt.show()

    # Plot EEPs for polX
    plt.figure(figsize=(12, 5))

    for i in range(magnitude_EEP_polX.shape[1]):
        if i == 0:
            plt.plot(
                theta * 180 / np.pi,
                magnitude_EEP_polX[:, i],
                "r-",
                alpha=0.1,
                label="EEPs",
            )
        else:
            plt.plot(theta * 180 / np.pi, magnitude_EEP_polX[:, i], "r-", alpha=0.1)

    plt.plot(theta * 180 / np.pi, AEP_polX, "black", label="AEP")
    plt.title("polyX, X-plane, freq =100MHz")
    plt.xlabel("Theta (deg)")
    plt.ylabel("E-field (dBV)")
    plt.xlim(-90, 90)
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig('./plots/plot_power_EEPs_and_AEP_polX.png')
    plt.show()
    
def plot_stefcal_comparison(algo1_AEP, algo1_EEPs, algo2_AEP, algo2_EEPs):
    """
    Compares and visualises the convergence and error metrics for
    two different calibration algorithms using AEP and EEP data.

    Parameters
    ----------
    algo1_AEP : tuple
        A tuple containing convergence data and error metrics
        (absolute, amplitude, phase) for Algorithm 1 using AEP data.
    algo1_EEPs : tuple
        A tuple containing convergence data and error metrics for
        Algorithm 1 using EEP data.
    algo2_AEP : tuple
        A tuple containing convergence data and error metrics for
        Algorithm 2 using AEP data.
    algo2_EEPs : tuple
        A tuple containing convergence data and error metrics for
        Algorithm 2 using EEP data.

    This function generates a series of plots to compare two calibration algorithms.
    It visualises:
    - The convergence trends for both algorithms using AEP and EEP data.
    - The absolute gain errors, showcasing how closely each algorithm's gain estimations
    match the true gains.
    - The amplitude errors, illustrating the differences in the magnitude of the
    estimated gains compared to the true gains.
    - The phase errors, demonstrating the discrepancies in the phase of the estimated
    gains relative to the true gains.

    The comparison is done across several metrics to provide a comprehensive
    overview of each algorithm's performance and to facilitate the assessment
    of their convergence behavior and accuracy in gain estimation.
    """
    # Unpack the results
    convergence_AEP1, abs_error_AEP1, amp_error_AEP1, phase_error_AEP1 = algo1_AEP
    convergence_EEPs1, abs_error_EEPs1, amp_error_EEPs1, phase_error_EEPs1 = algo1_EEPs
    convergence_AEP2, abs_error_AEP2, amp_error_AEP2, phase_error_AEP2 = algo2_AEP
    convergence_EEPs2, abs_error_EEPs2, amp_error_EEPs2, phase_error_EEPs2 = algo2_EEPs

    # Plot the convergence
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 3, 1)
    plt.plot(convergence_AEP1, label="AEP")
    plt.plot(convergence_EEPs1, label="EEPs")
    plt.xlabel("Iteration")
    plt.ylabel("Convergence")
    plt.title("Algorithm 1, Convergence")
    plt.legend()

    plt.subplot(1, 3, 2)
    plt.plot(convergence_AEP2, label="AEP")
    plt.plot(convergence_EEPs2, label="EEPs")
    plt.xlabel("Iteration")
    plt.title("Algorithm 2, Convergence")
    plt.legend()
    plt.tight_layout()

    plt.subplot(1, 3, 3)
    plt.plot(np.log(convergence_AEP1), label="StEFCal 1")
    plt.plot(np.log(convergence_AEP2), label="StEFCal 2")
    plt.hlines(
        np.log(1e-5),
        xmin=0,
        xmax=len(convergence_AEP1),
        colors="r",
        linestyles="dashed",
        label=r"$\log(\tau) = -5$",
    )
    plt.xlabel("Iteration")
    plt.ylabel("Log Convergence")
    plt.title("AEP, Log Convergence")
    plt.legend()

    plt.tight_layout()
    plt.savefig('./plots/plot_stefcal_convergence_comparison.png')
    plt.show()
    
    # abs error
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(abs_error_AEP1, label="AEP")
    plt.plot(abs_error_EEPs1, label="EEPs")
    plt.xlabel("Iteration")
    plt.ylabel("Error")
    plt.title("Algorithm 1, Absolute Gain Error")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(abs_error_AEP2, label="AEP")
    plt.plot(abs_error_EEPs2, label="EEPs")
    plt.xlabel("Iteration")
    plt.title("Algorithm 2, Absolute Gain Error")
    plt.legend()

    plt.tight_layout()
    plt.savefig('./plots/plot_stefcal_gain_comparison.png')
    plt.show()

    # amp error
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(amp_error_AEP1, label="AEP")
    plt.plot(amp_error_EEPs1, label="EEPs")
    plt.xlabel("Iteration")
    plt.ylabel("Error")
    plt.title("Algorithm 1, Absolute Amplitude Error")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(amp_error_AEP2, label="AEP")
    plt.plot(amp_error_EEPs2, label="EEPs")
    plt.xlabel("Iteration")
    plt.title("Algorithm 2,Absolute Amplitude Error")
    plt.legend()

    plt.tight_layout()
    plt.savefig('./plots/plot_stefcal_amplitude_comparison.png')
    plt.show()

    # phase error
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(phase_error_AEP1, label="AEP")
    plt.plot(phase_error_EEPs1, label="EEPs")
    plt.xlabel("Iteration")
    plt.ylabel("Error")
    plt.title("Algorithm 1, Absolute Phase Error")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(phase_error_AEP2, label="AEP")
    plt.plot(phase_error_EEPs2, label="EEPs")
    plt.xlabel("Iteration")
    plt.title("Algorithm 2, Absolute Phase Error")
    plt.legend()

    plt.tight_layout()
    plt.savefig('./plots/plot_stefcal_phase_comparison.png')
    plt.show()

def plot_beamforming_results(theta_range, polY_data, polX_data, labels, linestyles):
    """
    Plots the beamforming results for both Y and X polarisations over a
    specified range of theta values.

    Parameters
    ----------
    theta_range : np.ndarray
        The range of theta values (in radians) over which the beamforming data is
        plotted. These values typically span from -pi/2 to pi/2 to cover the full
        angular span in the elevation plane.
    polY_data : list of np.ndarray
        A list containing the data arrays for Y polarisation. Each array corresponds
        to a different beamforming result or condition to be plotted.
    polX_data : list of np.ndarray
        A list containing the data arrays for X polarisation, structured similarly to
        `polY_data`.
    labels : list of str
        A list of labels corresponding to each data array in `polY_data` and
        `polX_data`, used for the plot legend.
    linestyles : list of str
        A list of matplotlib linestyles corresponding to each data array,
        used to differentiate the plotted lines.

    The function generates two separate plots, one for Y polarisation and another for
    X polarisation. Each plot displays multiple beamforming results or conditions,
    differentiated by linestyles and annotated with a legend.
    """
    # Check if the input lists are of the same length
    if not (len(polY_data) == len(polX_data) == len(labels)):
        raise ValueError(
            "The length of polY_data, polX_data, and labels must be the same."
        )

    # Plot polY data in its own figure
    plt.figure(figsize=(12, 5))
    for data, label, linestyle in zip(polY_data, labels, linestyles):
        plt.plot(np.rad2deg(theta_range), data, label=label, linestyle=linestyle)
    plt.title("Station Beam, polY, Y feed, freq = 100MHz")
    plt.xlabel("Theta (degrees)")
    plt.ylabel("E-field (dBV)")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig('./plots/plot_beamforming_polY.png')
    plt.show()

    # Plot polX data in its own figure
    plt.figure(figsize=(12, 5))
    for data, label, linestyle in zip(polX_data, labels, linestyles):
        plt.plot(np.rad2deg(theta_range), data, label=label, linestyle=linestyle)
    plt.title("Station Beam, polX, X feed, freq = 100MHz")
    plt.xlabel("Theta (degrees)")
    plt.ylabel("E-field (dBV)")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig('./plots/plot_beamforming_polX.png')
    plt.show()

def plot_station_beam_pattern(
    x,
    y,
    APy,
    APx,
    xlabel=r"$\sin(\theta)\cos(\phi)$",
    ylabel=r"$\sin(\theta)\sin(\phi)$",
    intensity_label="Intensity [dBV]",
    cmap="viridis",
):
    """
    Plots the station beam patterns for Y and X polarisations using color-mapped
    intensity data over a 2D spatial grid.

    Parameters
    ----------
    x : np.ndarray
        2D array of X-coordinates for the meshgrid over which the station beam patterns
        are plotted. Typically represents the sine of the elevation angle multiplied
        by the cosine of the azimuth angle.
    y : np.ndarray
        2D array of Y-coordinates for the meshgrid, corresponding to `x`.
        Typically represents the sine of the elevation
        angle multiplied by the sine of the azimuth angle.
    APy : np.ndarray
        The antenna pattern data for Y polarisation to be visualised,
        mapped onto the meshgrid defined by `x` and `y`.
    APx : np.ndarray
        The antenna pattern data for X polarisation, structured similarly to `APy`.
    xlabel : str, optional
        Label for the x-axis of the plots. Defaults to representing the
        mathematical expression for `x`.
    ylabel : str, optional
        Label for the y-axis of the plots. Defaults to representing the
        mathematical expression for `y`.
    intensity_label : str, optional
        Label for the colorbar indicating the intensity scale.
        Defaults to 'Intensity [dBV]'.
    cmap : str, optional
        Colormap used for the pcolormesh plots. Defaults to 'viridis'.

    This function creates two side-by-side color-mesh plots representing the
    station beam patterns for Y and X polarisations, respectively.
    The intensity of the beam patterns is color-coded according to the specified
    colormap, with a colorbar indicating the intensity scale.
    """
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', UserWarning)
        mesh = plt.pcolormesh(x, y, APy, cmap=cmap)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title("Station Beam, polY, Y feed, freq=100MHz")
    plt.xlim(-1.1, 1.1)
    plt.ylim(-1.1, 1.1)
    plt.grid(True)
    plt.colorbar(mesh)

    plt.subplot(1, 2, 2)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', UserWarning)
        mesh = plt.pcolormesh(x, y, APx, cmap=cmap)
    plt.xlabel(xlabel)
    plt.title("Station Beam, polX, X feed, freq=100MHz")
    plt.xlim(-1.1, 1.1)
    plt.ylim(-1.1, 1.1)
    plt.grid(True)
    plt.colorbar(mesh, label=intensity_label)

    plt.tight_layout()
    plt.savefig('./plots/plot_station_beam_pattern.png')
    plt.show()
