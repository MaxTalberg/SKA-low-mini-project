import numpy as np

# conver to dBV
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