import numpy as np
import matplotlib.pyplot as plt


# function to plot EEPs and AEPs from vector compoenents
def plot2(theta, magnitude_EEP_polY, magnitude_EEP_polX, AEP_polY, AEP_polX):
    
    plt.figure(figsize=(10, 12))

    plt.subplot(2, 1, 1)  # 2 rows, 1 column, first plot
    for i in range(magnitude_EEP_polY.shape[1]):
        if i == 0:
            # Label only the first EEP line for the legend
            plt.plot(theta * 180 / np.pi, magnitude_EEP_polY[:, i], 'r-', alpha=0.1, label='EEPs')
        else:
            # Plot the rest without a label
            plt.plot(theta * 180 / np.pi, magnitude_EEP_polY[:, i], 'r-', alpha=0.1) 
    plt.plot(theta * 180 / np.pi, AEP_polY, 'b-', label='AEP') 
    plt.title('polyY, Y-plane, freq =100MHz')
    plt.xlabel('Theta (deg)')
    plt.ylabel('E-field (dBV)')
    plt.xlim(-90, 90)
    plt.legend()
    plt.grid(True)

    # Plot EEPs for polX
    plt.subplot(2, 1, 2)  # 2 rows, 1 column, second plot
    for i in range(magnitude_EEP_polX.shape[1]):
        if i == 0:
            plt.plot(theta * 180 / np.pi, magnitude_EEP_polX[:, i], 'r-', alpha=0.1, label='EEPs')
        else:
            plt.plot(theta * 180 / np.pi, magnitude_EEP_polX[:, i], 'r-', alpha=0.1) 
    plt.plot(theta * 180 / np.pi, AEP_polX, 'b-', label='AEP') 
    plt.title('polyX, X-plane, freq =100MHz')
    plt.xlabel('Theta (deg)')
    plt.ylabel('E-field (dBV)')
    plt.xlim(-90, 90)
    plt.legend()
    plt.grid(True)

    # Show the plot
    plt.tight_layout() 
    plt.show()

def plot_stefcal(abs_error_AEP, amp_error_AEP, phase_error_AEP, abs_error_EEPs, amp_error_EEPs, phase_error_EEPs):

    # Plot the errors
    plt.figure(figsize=(14, 5))

    plt.subplot(1, 3, 1)
    plt.plot(abs_error_AEP, label='AEP')
    plt.plot(abs_error_EEPs, label='EEPs')
    plt.xlabel('Iteration')
    plt.ylabel('Error')
    plt.title('Absolute Error')
    plt.legend()

    plt.subplot(1, 3, 2)
    plt.plot(amp_error_AEP, label='AEP')
    plt.plot(amp_error_EEPs, label='EEPs')
    plt.xlabel('Iteration')
    plt.ylabel('Error')
    plt.title('Amplitude Error')
    plt.legend()

    plt.subplot(1, 3, 3)
    plt.plot(phase_error_AEP, label='AEP')
    plt.plot(phase_error_EEPs, label='EEPs')
    plt.xlabel('Iteration')
    plt.ylabel('Error')
    plt.title('Phase Error')
    plt.legend()

    plt.tight_layout()
    plt.show()

def plot_beamforming_results(theta_range, polY_data, polX_data, labels):
    """
    Plots beamforming results for polY and polX as subplots.

    Parameters:
    - theta_range: The range of theta values over which the data is plotted.
    - polY_data: A list of arrays containing the data for polY.
    - polX_data: A list of arrays containing the data for polX.
    - labels: A list of labels for the plotted data.
    """
    # Check if the input lists are of the same length
    if not (len(polY_data) == len(polX_data) == len(labels)):
        raise ValueError("The length of polY_data, polX_data, and labels must be the same.")

    # Create a figure with two subplots
    fig, axs = plt.subplots(2, 1, figsize=(10, 12))

    # Plot polY data
    for data, label in zip(polY_data, labels):
        axs[0].plot(np.rad2deg(theta_range), data, label=label)
    axs[0].set_title('Polarization Y')
    axs[0].set_xlabel('Theta (degrees)')
    axs[0].set_ylabel('Pattern Magnitude (dBV)')
    axs[0].legend()
    axs[0].grid(True)

    # Plot polX data
    for data, label in zip(polX_data, labels):
        axs[1].plot(np.rad2deg(theta_range), data, label=label)
    axs[1].set_title('Polarization X')
    axs[1].set_xlabel('Theta (degrees)')
    axs[1].set_ylabel('Pattern Magnitude (dBV)')
    axs[1].legend()
    axs[1].grid(True)

    # Adjust layout
    plt.tight_layout()
    plt.show()

def plot_station_beam_pattern(x, y, AP, title="Calibrated Station Beam Pattern", xlabel=r'$\sin(\theta)\cos(\phi)$', ylabel=r'$\sin(\theta)\sin(\phi)$', intensity_label='Intensity [dBV]', cmap='viridis'):
    """
    Plots the station beam pattern using pcolormesh.

    Parameters:
    - x: X-coordinates of the mesh (2D array).
    - y: Y-coordinates of the mesh (2D array).
    - AP: Antenna Pattern data to be plotted (2D array).
    - title: (Optional) Title of the plot.
    - xlabel: (Optional) Label for the x-axis.
    - ylabel: (Optional) Label for the y-axis.
    - intensity_label: (Optional) Label for the colorbar.
    - cmap: (Optional) Colormap for the pcolormesh.
    """
    plt.figure(figsize=(10, 8))
    mesh = plt.pcolormesh(x, y, AP, shading='auto', cmap=cmap)
    plt.colorbar(mesh, label=intensity_label)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.xlim(-1, 1)
    plt.ylim(-1, 1)
    plt.grid(True)
    plt.show()