import numpy as np
import matplotlib.pyplot as plt


# function to plot EEPs and AEPs from vector compoenents
def plot2(theta, magnitude_EEP_polY, magnitude_EEP_polX, AEP_polY, AEP_polX):
    
    #plt.figure(figsize=(12, 10))

    plt.figure(figsize=(12, 5))
    #plt.subplot(2, 1, 1)  # 2 rows, 1 column, first plot
    for i in range(magnitude_EEP_polY.shape[1]):
        if i == 0:
            # Label only the first EEP line for the legend
            plt.plot(theta * 180 / np.pi, magnitude_EEP_polY[:, i], 'r-', alpha=0.1, label='EEPs')
        else:
            # Plot the rest without a label
            plt.plot(theta * 180 / np.pi, magnitude_EEP_polY[:, i], 'r-', alpha=0.1)
            
    plt.plot(theta * 180 / np.pi, AEP_polY, 'black', label='AEP') 
    plt.title('polyY, Y-plane, freq =100MHz')
    plt.xlabel('Theta (deg)')
    plt.ylabel('E-field (dBV)')
    plt.xlim(-90, 90)
    plt.legend()
    plt.grid(True)
    plt.show()

    # Plot EEPs for polX
    #plt.subplot(2, 1, 2)  # 2 rows, 1 column, second plot
    plt.figure(figsize=(12, 5))

    for i in range(magnitude_EEP_polX.shape[1]):
        if i == 0:
            plt.plot(theta * 180 / np.pi, magnitude_EEP_polX[:, i], 'r-', alpha=0.1, label='EEPs')
        else:
            plt.plot(theta * 180 / np.pi, magnitude_EEP_polX[:, i], 'r-', alpha=0.1)

    plt.plot(theta * 180 / np.pi, AEP_polX, 'black', label='AEP') 
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

def plot_stefcal_comparison(algo1_AEP, algo1_EEPs, algo2_AEP, algo2_EEPs):

    # Unpack the results
    convergence_AEP1, abs_error_AEP1, amp_error_AEP1, phase_error_AEP1 = algo1_AEP
    convergence_EEPs1, abs_error_EEPs1, amp_error_EEPs1, phase_error_EEPs1 = algo1_EEPs
    convergence_AEP2, abs_error_AEP2, amp_error_AEP2, phase_error_AEP2 = algo2_AEP
    convergence_EEPs2, abs_error_EEPs2, amp_error_EEPs2, phase_error_EEPs2 = algo2_EEPs

    # Plot the convergence
    plt.figure(figsize=(12, 5))

    plt.subplot(1,3,1)
    plt.plot(convergence_AEP1, label='AEP')
    plt.plot(convergence_EEPs1, label='EEPs')
    plt.xlabel('Iteration')
    plt.ylabel('Convergence')
    plt.title('Algorithm 1, Convergence')
    plt.legend()

    plt.subplot(1,3,2)
    plt.plot(convergence_AEP2, label='AEP')
    plt.plot(convergence_EEPs2, label='EEPs')
    plt.xlabel('Iteration')
    plt.title('Algorithm 2, Convergence')
    plt.legend()
    plt.tight_layout()

    plt.subplot(1,3,3)
    plt.plot(np.log(convergence_AEP1), label='StEFCal 1')
    plt.plot(np.log(convergence_AEP2), label='StEFCal 2')
    plt.hlines(np.log(1e-5), xmin=0, xmax=len(convergence_AEP1), colors='r', linestyles='dashed', label=r'$\log(\tau) = -5$')
    plt.xlabel('Iteration')
    plt.ylabel('Log Convergence')
    plt.title('AEP, Log Convergence')
    plt.legend()
    plt.tight_layout()
    plt.show()

    # abs error
    plt.figure(figsize=(12, 5))

    plt.subplot(1,2,1)
    plt.plot(abs_error_AEP1, label='AEP')
    plt.plot(abs_error_EEPs1, label='EEPs')
    plt.xlabel('Iteration')
    plt.ylabel('Error')    
    plt.title('Algorithm 1, Absolute Gain Error')
    plt.legend()

    plt.subplot(1,2,2)
    plt.plot(abs_error_AEP2, label='AEP')
    plt.plot(abs_error_EEPs2, label='EEPs')
    plt.xlabel('Iteration')
    plt.title('Algorithm 2, Absolute Gain Error')
    plt.legend()
    plt.tight_layout()
    plt.show()

    # amp error
    plt.figure(figsize=(12, 5))

    plt.subplot(1,2,1)
    plt.plot(amp_error_AEP1, label='AEP')
    plt.plot(amp_error_EEPs1, label='EEPs')
    plt.xlabel('Iteration')
    plt.ylabel('Error')
    plt.title('Algorithm 1, Absolute Amplitude Error')
    plt.legend()


    plt.subplot(1,2,2)
    plt.plot(amp_error_AEP2, label='AEP')
    plt.plot(amp_error_EEPs2, label='EEPs')
    plt.xlabel('Iteration')
    plt.title('Algorithm 2,Absolute Amplitude Error')
    plt.legend()
    plt.tight_layout()
    plt.show()


    # phase error
    plt.figure(figsize=(12, 5))

    plt.subplot(1,2,1)
    plt.plot(phase_error_AEP1, label='AEP')
    plt.plot(phase_error_EEPs1, label='EEPs')
    plt.xlabel('Iteration')
    plt.ylabel('Error')
    plt.title('Algorithm 1, Absolute Phase Error')
    plt.legend()
    
    plt.subplot(1,2,2)
    plt.plot(phase_error_AEP2, label='AEP')
    plt.plot(phase_error_EEPs2, label='EEPs')
    plt.xlabel('Iteration')
    plt.title('Algorithm 2, Absolute Phase Error')
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_beamforming_results(theta_range, polY_data, polX_data, labels):
    """
    Plots beamforming results for polY and polX as separate figures.

    Parameters:
    - theta_range: The range of theta values over which the data is plotted.
    - polY_data: A list of arrays containing the data for polY.
    - polX_data: A list of arrays containing the data for polX.
    - labels: A list of labels for the plotted data.
    """
    # Check if the input lists are of the same length
    if not (len(polY_data) == len(polX_data) == len(labels)):
        raise ValueError("The length of polY_data, polX_data, and labels must be the same.")

    # Plot polY data in its own figure
    plt.figure(figsize=(12, 5))
    for data, label in zip(polY_data, labels):
        plt.plot(np.rad2deg(theta_range), data, label=label)
    plt.title('Station Beam, polY, Y-plane, freq = 100MHz')
    plt.xlabel('Theta (degrees)')
    plt.ylabel('Pattern Magnitude (dBV)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Plot polX data in its own figure
    plt.figure(figsize=(12, 5))
    for data, label in zip(polX_data, labels):
        plt.plot(np.rad2deg(theta_range), data, label=label)
    plt.title('Station Beam, polX, X-plane, freq = 100MHz')
    plt.xlabel('Theta (degrees)')
    plt.ylabel('Pattern Magnitude (dBV)')
    plt.legend()
    plt.grid(True)
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