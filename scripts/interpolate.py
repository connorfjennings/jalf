import numpy as np
import jax.numpy as jnp
from jax import random, lax, vmap, jit
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS

def find_bounds(x_values, x_target):
    upper = jnp.clip(jnp.searchsorted(x_values, x_target), 1, len(x_values) - 1)
    lower = upper - 1
    return lower, upper

def generate_hypercube_indices(n_dims):
    """
    Returns all 2^n_dims combinations of 0 and 1, shape: (2^n, n_dims)
    JAX-compatible version of itertools.product([0,1], repeat=n_dims)
    """
    n_corners = 2 ** n_dims
    bits = jnp.arange(n_corners)[:, None]
    bit_shifts = 2 ** jnp.arange(n_dims)[::-1]
    return ((bits // bit_shifts) % 2).astype(jnp.int32)

def interpolate_nd_jax(tgt_params, value_grids, flux_grid, n_dims):
    #actually use this one

    # Find bounds and weights for each dimension
    bounds = []
    weights = []
    for i in range(n_dims):
        axis_vals = value_grids[i]
        idx_lo, idx_hi = find_bounds(axis_vals, tgt_params[i])

        x_lo = axis_vals[idx_lo]
        x_hi = axis_vals[idx_hi]

        # Avoid divide-by-zero if x_lo == x_hi
        w = jnp.where(x_hi != x_lo, (tgt_params[i] - x_lo) / (x_hi - x_lo), 0.0)

        bounds.append((idx_lo, idx_hi))
        weights.append(w)

    bounds = jnp.array(bounds)  # shape (n_dims, 2)
    weights = jnp.array(weights)  # shape (n_dims,)

    # Generate all corner combinations (0 = lo, 1 = hi)
    corner_indices = generate_hypercube_indices(n_dims)  # shape (2^n_dims, n_dims)

    # Map corners to grid indices
    idx_map = bounds.T  # shape (2, n_dims)
    corner_grid_indices = idx_map[corner_indices, jnp.arange(n_dims)]

    # Compute weights for each corner
    axis_weights = jnp.stack([1 - weights, weights], axis=1)  # shape (n_dims, 2)
    corner_weights = jnp.prod(axis_weights[jnp.arange(n_dims)[:, None], corner_indices.T], axis=0)  # shape (2^n_dims,)

    # Fetch fluxes at each corner
    def get_flux(indices):
        return flux_grid[tuple(indices)]

    fluxes = vmap(get_flux)(corner_grid_indices)

    # Weighted sum
    interpolated = jnp.sum(fluxes * corner_weights[:, None], axis=0)

    return interpolated
interpolate_nd_jax = jit(interpolate_nd_jax,static_argnames=['n_dims'])


@jit
def interpolate_spectrum_4d_jax(tgt_params, ssp_value_grid, flux_grid):
    #older version, testing says the general version is just as fast so I'm using that
    #unpack inputs
    t, Z, imf1, imf2 = tgt_params

    t_vals = ssp_value_grid[0][:,0,0,0]
    z_vals = ssp_value_grid[1][0,:,0,0]
    imf1_vals = ssp_value_grid[2][0,0,:,0]
    imf2_vals = ssp_value_grid[3][0,0,0,:]

    # Find bounding indices
    i_t_lo, i_t_hi = find_bounds(t_vals, t)
    i_z_lo, i_z_hi = find_bounds(z_vals, Z)
    i_1_lo, i_1_hi = find_bounds(imf1_vals, imf1)
    i_2_lo, i_2_hi = find_bounds(imf2_vals, imf2)

    # Get the weights for each axis
    wt = (t - t_vals[i_t_lo]) / (t_vals[i_t_hi] - t_vals[i_t_lo])
    wz = (Z - z_vals[i_z_lo]) / (z_vals[i_z_hi] - z_vals[i_z_lo])
    w1 = (imf1 - imf1_vals[i_1_lo]) / (imf1_vals[i_1_hi] - imf1_vals[i_1_lo])
    w2 = (imf2 - imf2_vals[i_2_lo]) / (imf2_vals[i_2_hi] - imf2_vals[i_2_lo])
    weights = jnp.array([wt, wz, w1, w2])

    # Corner index combinations: all 2^4 = 16 combinations of 0 (lo) and 1 (hi)
    corner_indices = generate_hypercube_indices(4)

    # Map 0 → lo index, 1 → hi index
    idx_map = jnp.array([
        [i_t_lo, i_z_lo, i_1_lo, i_2_lo],
        [i_t_hi, i_z_hi, i_1_hi, i_2_hi]
    ])  # shape (2, 4)

    # Convert corner indices into actual grid indices
    corner_grid_indices = idx_map[corner_indices, jnp.arange(4)]  # shape (16, 4)

    # Compute weights for all 16 corners
    axis_weights = jnp.stack([1 - weights, weights], axis=1)  # shape (4, 2)
    corner_weights = jnp.prod(axis_weights[jnp.arange(4)[:,jnp.newaxis], corner_indices.T], axis=0)  # shape (16,)

    # Gather all 16 fluxes (shape: 16, n_lambda)
    def get_flux(indices):
        i_t, i_z, i_1, i_2 = indices
        return flux_grid[i_t, i_z, i_1, i_2]
    fluxes = vmap(get_flux)(corner_grid_indices)

    # Weighted sum
    interpolated = jnp.sum(fluxes * corner_weights[:, jnp.newaxis], axis=0)

    return interpolated