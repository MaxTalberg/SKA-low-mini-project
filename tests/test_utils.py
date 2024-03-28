import numpy as np

from src.utils import load_antenna_data
from src.harp_beam import to_dBV


def test_load_antenna_data():
    # Define the filename
    test_filename = 'src/harp_beam_data/data_EEPs_SKALA41_random_100MHz.mat'

    # Call the function with the test file
    result = load_antenna_data(test_filename)

    # Verify the type of each item in the result
    assert isinstance(result[0], int), "max_order should be an integer"
    assert isinstance(result[1], int), "num_mbf should be an integer"
    assert isinstance(result[2], np.ndarray), "coeffs_polX should be a numpy array"
    assert isinstance(result[3], np.ndarray), "coeffs_polY should be a numpy array"
    assert isinstance(result[4], np.ndarray), "alpha_te should be a numpy array"
    assert isinstance(result[5], np.ndarray), "alpha_tm should be a numpy array"
    assert isinstance(result[6], np.ndarray), "pos_ant should be a numpy array"

    # Assert the values of each item in the result
    assert result[0] == 15, "max_order does not match expected value"

def test_to_dBV_single_value():
    # Test with a single float value
    magnitude = 1
    expected_dBV = 0  # since 20 * log10(1) = 0
    assert to_dBV(magnitude) == expected_dBV, "dBV conversion for single value failed"

def test_to_dBV_array():
    # Test with an array of magnitudes
    magnitudes = np.array([1, 10, 0.1])
    expected_dBV_array = np.array([0, 20, -20])  # Corresponding dBV values
    assert np.allclose(to_dBV(magnitudes), expected_dBV_array), "dBV conversion for array failed"

