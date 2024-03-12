import numpy as np

# convert to dBV
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

# StEFCal algorithm 1
def stefcal1(max_iteration=100, threshold=1e-6):
    # Initial gain matrix G
    G = np.eye(len(g_sol)) # Identity matrix

    # Iterative loop
    for i in range(max_iteration):
        # Last iteration of G for comparison
        G_prev = G.copy() 

        for p in range(G.shape[0]):  # Loop over antennas p
            z = np.dot(G_prev, M_AEP[:, p])  # Use all rows of M_AEP for antenna p
            gp = np.dot(np.conjugate(R[:, p]).T, z) / np.dot(np.conjugate(z).T, z)  # Calculate new gain for antenna p
            G[p, p] = gp  # Update the gain for antenna p in the matrix

        # Convergence check even iterations
        if i % 2 == 0:
            delta_G = np.linalg.norm(G - G_prev, 'fro') / np.linalg.norm(G, 'fro')
            if delta_G < threshold:
                print(f"Convergence reached after {i+1} iterations.")
                break
        # Averaging step
        G = (G + G_prev) / 2

    return G