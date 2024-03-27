import os
import configparser
import numpy as np
from utils import compute_k0, load_antenna_data, load_arrays_from_mat
from harp_beam import compute_EEPs, power_EEPs, stefcal, compute_beamforming
from plot_utils import (
    plot_power_EEPs_and_AEP,
    plot_stefcal_comparison,
    plot_beamforming_results,
    plot_station_beam_pattern,
)

# initialise the config parser
config = configparser.ConfigParser()

# read the config file
config.read("config.ini")

# accessing variables
num_dir = config.getint("PARAMETERS", "num_dir")
freq = config.getint("PARAMETERS", "freq")
c0 = config.getint("PARAMETERS", "speed_of_light")
max_iteration = config.getint("PARAMETERS", "max_iteration")
threshold = config.getfloat("PARAMETERS", "threshold")
antenna_name = config.get("ANTENNA", "antenna_name")
array_layout = config.get("ANTENNA", "array_layout")

# Data path
data_path = config.get("PATHS", "data_path")

# filenames
filename_eep_pattern = config["PATHS"]["filename_eep"]
filename_eep = os.path.join(
    data_path, filename_eep_pattern % (antenna_name, array_layout, freq)
)
filename_vismat = os.path.join(data_path, config.get("PATHS", "filename_vismat"))

# computed variables
k0 = compute_k0(freq, c0)
max_order, num_mbf, coeffs_polX, coeffs_polY, alpha_te, alpha_tm, pos_ant = (
    load_antenna_data(filename_eep)
)

# accessing data
arrays = load_arrays_from_mat(str(filename_vismat))

R = arrays["R"]
M_AEP = arrays["M_AEP"]
M_EEPs = arrays["M_EEPs"]
g_sol = arrays["g_sol"]
g_AEP = arrays["g_AEP"]
g_EEPs = arrays["g_EEPs"]

# parameters
theta_min = -np.pi / 2
theta_max = np.pi / 2
theta0 = 0
phi0 = 0
phi_min = -np.pi
phi_max = np.pi
theta0_steering = np.radians(40)
phi0_steering = np.radians(80)


def main():

    ## Q2. Compute the EEPs and AEPs for the given antenna data and parameters
    # theta and phi values
    theta = np.linspace(theta_min, theta_max, num_dir)
    phi = np.zeros_like(theta)

    # Compute EEPs
    complex_E_fields = compute_EEPs(
        theta.copy()[:, None],
        phi.copy()[:, None],
        alpha_te,
        alpha_tm,
        coeffs_polX,
        coeffs_polY,
        pos_ant,
        num_mbf,
        max_order,
        k0,
    )

    # Compute power EEPs
    power_E_fields = power_EEPs(*complex_E_fields)

    # plot EEPs and AEPs
    plot_power_EEPs_and_AEP(theta, *power_E_fields)

    ## Q3/4. Implement stefcal and plot the errors
    # Run stefcal Algorithm 1
    print("Running Stefcal Algorithm 1...")
    print("AEP Stefcal Algorithm 1:")
    G_AEP1, *algo1_AEP = stefcal(
        M_AEP, R, g_sol, max_iteration, threshold, algorithm2=False
    )
    print("EEPs Stefcal Algorithm 1:")
    G_EEPs1, *algo1_EEP = stefcal(
        M_EEPs, R, g_sol, max_iteration, threshold, algorithm2=False
    )

    # Run stefcal Algorithm 2
    print("Running Stefcal Algorithm 2...")
    print("AEP Stefcal Algorithm 2:")
    G_AEP2, *algo2_AEP = stefcal(
        M_AEP, R, g_sol, max_iteration, threshold, algorithm2=True
    )
    print("EEPs Stefcal Algorithm 2:")
    G_EEPs2, *algo2_EEP = stefcal(
        M_EEPs, R, g_sol, max_iteration, threshold, algorithm2=True
    )

    # plot stefcal
    plot_stefcal_comparison(algo1_AEP, algo1_EEP, algo2_AEP, algo2_EEP)

    ## Q5. Calibrating EEP using Array Pattern
    # theta values and phi to zenith
    theta = np.linspace(theta_min, theta_max, num_dir)
    phi = 0

    # Use gain values with the lowest error (StEFCal Algorithm 1)
    G_EEPs = G_EEPs1.diagonal().reshape(-1, 1)
    G_AEP = G_AEP1.diagonal().reshape(-1, 1)

    # Array pattern for Gain Solutions
    AP_sol_polY, AP_sol_polX = compute_beamforming(
        g_sol, *complex_E_fields, pos_ant, k0, theta, phi, theta0, phi0
    )

    # Array pattern for EEPs from Algorithm 1
    AP_G_EEPs_polY, AP_G_EEPs_polX = compute_beamforming(
        G_EEPs, *complex_E_fields, pos_ant, k0, theta, phi, theta0, phi0
    )

    # Array pattern for AEP from Algorithm 1
    AP_G_AEP_polY, AP_G_AEP_polX = compute_beamforming(
        G_AEP, *complex_E_fields, pos_ant, k0, theta, phi, theta0, phi0
    )

    # Plot beamforming results
    plot_beamforming_results(
        theta,
        [AP_sol_polY, AP_G_EEPs_polY, AP_G_AEP_polY],
        [AP_sol_polX, AP_G_EEPs_polX, AP_G_AEP_polX],
        ["Gain Solutions", "EEPs", "AEP"],
        ["solid", "dashed", "dashed"],
    )

    ## Q6. Calibrated station beam in sine-cosine space
    # Initialise grid for theta and phi
    theta = np.linspace(theta_min, theta_max, num_dir)
    phi = np.linspace(-np.pi, np.pi, num_dir)
    theta_grid, phi_grid = np.meshgrid(theta, phi, indexing="ij")

    # Use most accurate gain values (EEPs StEFCal Algorithm 1)
    G = G_EEPs1.diagonal().reshape(-1, 1)

    # Compute the beamforming
    AP_ploY, AP_polX = compute_beamforming(
        G, *complex_E_fields, pos_ant, k0, theta, phi, theta0_steering, phi0_steering
    )

    # Map theta and phi to sine-cosine coordinates
    x = np.sin(theta_grid) * np.cos(phi_grid)
    y = np.sin(theta_grid) * np.sin(phi_grid)

    # Create the plot using pcolormesh, which is more appropriate for spherical data
    plot_station_beam_pattern(x, y, AP_ploY, AP_polX)


if __name__ == "__main__":
    main()
