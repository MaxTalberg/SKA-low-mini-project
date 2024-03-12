import numpy as np
import matplotlib.pyplot as plt


# function to plot EEPs and AEPs from vector compoenents
def plot2(theta, v_theta_polY, v_theta_polX, magnitude_EEP_polY, magnitude_EEP_polX, AEP_polY, AEP_polX):
    
    plt.figure(figsize=(10, 12))

    plt.subplot(2, 1, 1)  # 2 rows, 1 column, first plot
    for i in range(v_theta_polY.shape[1]):
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
    plt.legend()
    plt.grid(True)

    # Plot EEPs for polX
    plt.subplot(2, 1, 2)  # 2 rows, 1 column, second plot
    for i in range(v_theta_polX.shape[1]):
        if i == 0:
            plt.plot(theta * 180 / np.pi, magnitude_EEP_polX[:, i], 'r-', alpha=0.1, label='EEPs')
        else:
            plt.plot(theta * 180 / np.pi, magnitude_EEP_polX[:, i], 'r-', alpha=0.1) 
    plt.plot(theta * 180 / np.pi, AEP_polX, 'b-', label='AEP') 
    plt.title('polyX, X-plane, freq =100MHz')
    plt.xlabel('Theta (deg)')
    plt.ylabel('E-field (dBV)')
    plt.legend()
    plt.grid(True)

    # Show the plot
    plt.tight_layout() 
    plt.show()
