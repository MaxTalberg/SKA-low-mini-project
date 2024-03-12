import numpy as np
import pandas as ps
import matplotlib.pyplot as plt
import numpy as np
from harp_beam import compute_EEPs
from functions import to_dBV
from plots import plot2

## Q 2. plot all the 256 EEPs and their average (AEP)
num_dir = 256
theta = np.linspace(-np.pi/2, np.pi/2, num_dir)
phi = np.zeros_like(theta)

#EEPs
v_theta_polY, v_phi_polY, v_theta_polX, v_phi_polX = compute_EEPs(theta.copy()[:, None], phi.copy()[:, None])

# Calculate magnitude of EEPs in dBV
magnitude_EEP_polY = to_dBV(np.abs(v_theta_polY))
magnitude_EEP_polX = to_dBV(np.abs(v_theta_polX))

# Calculate AEPs
AEP_polY = to_dBV(np.sqrt(np.mean(np.abs(v_theta_polY)**2, axis=1)))
AEP_polX = to_dBV(np.sqrt(np.mean(np.abs(v_theta_polX)**2, axis=1)))


# plot EEPs and AEPs
plot2(theta, v_theta_polY, v_theta_polX, magnitude_EEP_polY, magnitude_EEP_polX, AEP_polY, AEP_polX)