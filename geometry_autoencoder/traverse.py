"""traverse.py"""

import numpy as np

def traverse_latent_space(mu, n_latent, traversal_range=(-3, 3), steps=10):
    latent_vectors = []
    for dim in range(n_latent): 
        for alpha in np.linspace(traversal_range[0], traversal_range[1], steps):
            new_mu = mu.clone()
            new_mu[0, dim] = alpha
            latent_vectors.append(new_mu)
    return latent_vectors