import configparser
import numpy as np
from harp_beam import compute_EEPs, power_EEPs
from utils import compute_k0, load_antenna_data
from plot_utils import plot_power_EEPs_and_AEP

# initialise the config parser
config = configparser.ConfigParser()

# read the config file
config.read('config.ini')
print(config.sections())

# accessing variables
num_dir = config.getint('BEAM', 'num_dir')
freq = config.getint('BEAM', 'freq')
c0 = config.getint('BEAM', 'speed_of_light')
antenna_name = config.get('ANTENNA', 'antenna_name')
array_layout = config.get('ANTENNA', 'array_layout')
filename_eep_pattern = config['PATHS']['filename_eep']
filename_eep = filename_eep_pattern % (antenna_name, array_layout, freq)

# computed variables
k0 = compute_k0(freq, c0)
max_order, num_mbf, coeffs_polX, coeffs_polY, alpha_te, alpha_tm, pos_ant = load_antenna_data(filename_eep)

# parameters
theta_min = -np.pi / 2
theta_max = np.pi / 2
theta0 = 0
phi0 = 0
phi_min = -np.pi
phi_max = np.pi
theta0_steering = np.radians(80)
phi0_steering = np.radians(40)

# computed parameters
theta = np.linspace(theta_min, theta_max, num_dir)
phi = np.zeros_like(theta)


## Q1. Compute the EEPs and AEPs for the given antenna data and parameters
# Compute EEPs
complex_E_fields = compute_EEPs(theta.copy()[:, None], 
                                                                  phi.copy()[:, None], 
                                                                  alpha_te, alpha_tm, 
                                                                  coeffs_polX, 
                                                                  coeffs_polY, 
                                                                  pos_ant, 
                                                                  num_mbf, 
                                                                  max_order,
                                                                  k0)

# Compute power EEPs
power_E_fields = power_EEPs(*complex_E_fields)

# plot EEPs and AEPs
plot_power_EEPs_and_AEP(theta, *power_E_fields)
