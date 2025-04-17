"""
Experimentation to test which potential implementations may be quicker with JAX.
"""

import numpy as np
import jax.numpy as jnp
from jax import random, jit, grad, vmap, jacfwd, jacrev, lax
from copy import deepcopy
from pytictoc import TicToc
from tqdm import tqdm

"""
Baseline functions for basis creation and likelihood evaluation.
"""

seed = 42
key = random.key(seed)

def kernel_function(centroids: jnp.ndarray, data: jnp.ndarray, kernel_widths: jnp.ndarray) -> jnp.ndarray:
    """
    Compute the kernel function for a given set of centroids, data points, and kernel widths.

    Args:
        centroids (jnp.ndarray): Centroids for the kernel function.
        data (jnp.ndarray): Data points to evaluate the kernel function at.
        kernel_widths (jnp.ndarray): Widths of the kernels.

    Returns:
        jnp.ndarray: Kernel values for each data point and centroid.
    
    """
    diff = data - centroids.T
    return jnp.exp(-0.5 * (diff / kernel_widths.T) ** 2)


def evaluate_likelihood(state: dict) -> jnp.ndarray:
    """
    Evaluate the likelihood of the data given the model parameters.

    Args:
        y (jnp.ndarray): Observed data.
        A (jnp.ndarray): Design matrix.
        beta (jnp.ndarray): Model parameters.
        switch (jnp.ndarray): Switch variable indicating which parameters to use.
        sigma (jnp.ndarray): Noise standard deviation.

    Returns:
        jnp.ndarray: Log-likelihood of the data given the model parameters.
    
    """
    residuals = state["y"] - state["A"] @ (state["beta"] * state["switch"])
    return -0.5 * jnp.sum((residuals / state["sigma"]) ** 2) - jnp.log(state["sigma"]) * state["y"].shape[0] / 2


"""
Functions for birth and death moves.
"""

def birth_move(state: dict):
    """Simple "birth-like" move."""
    proposed_state = deepcopy(state)
    # select element to turn on- the first zero
    first_zero_index = jnp.argwhere(proposed_state["switch"] == 0, size=1)
    proposed_state["switch"] = proposed_state["switch"].at[first_zero_index[0]].set(1)
    # sample location and beta values
    proposed_state["z"] = proposed_state["z"].at[first_zero_index].set(np.random.normal((1, )))
    proposed_state["beta"] = proposed_state["beta"].at[first_zero_index].set(np.random.normal((1, )))
    proposed_state["A"] = kernel_function(
        proposed_state["z"], proposed_state["x"], proposed_state["tau"]
    )
    return proposed_state

def death_move(state: dict):
    """Simple "death-like" move."""
    proposed_state = deepcopy(state)
    deletion_index = np.random.randint(low=0, high=state["switch"].shape[0], size=1)
    proposed_state["switch"] = proposed_state["switch"].at[deletion_index].set(0)
    return proposed_state

birth_move_jit = jit(birth_move)
death_move_jit = jit(death_move)
evaluate_likelihood_jit = jit(evaluate_likelihood)

num_knots = 20
num_data = 10000
domain = jnp.array([-1, 1])
z = domain[0] + jnp.diff(domain) * random.uniform(key, shape=(num_knots, 1))
x = domain[0] + jnp.diff(domain) * random.uniform(key, shape=(num_data, 1))
tau = 0.1 * jnp.ones((num_knots, 1))
beta = random.normal(key, shape=(num_knots, 1))
sigma = 0.01

num_on = 4
switch = jnp.concatenate(
    [jnp.ones((num_on, 1)), jnp.zeros((num_knots - num_on, 1))], axis=0
)

# test evaluation of kernel function
A = kernel_function(z, x, tau)
# generate data
y = A @ beta + sigma * random.normal(key, shape=(num_data, 1))

state = {
    "z": z,
    "x": x,
    "tau": tau,
    "beta": beta,
    "sigma": sigma,
    "y": y,
    "A": A,
    "switch": switch
}

# test evaluation of likelihood function
likelihood = evaluate_likelihood(state)

"""
Run iterations of MCMC-like algorithm.
"""

num_iterations = 2000

t = TicToc()
t.tic()
for it in tqdm(range(num_iterations)):
    # fake reversible jump
    if np.random.uniform() < 0.5:
        # perform birth move
        proposed_state = birth_move(state)
    else:
        # perform death move
        proposed_state = death_move(state)
    # evaluate likelihood
    proposed_likelihood = evaluate_likelihood(proposed_state)
    if proposed_likelihood > likelihood:
        # accept proposed state
        state = proposed_state
        likelihood = proposed_likelihood

    # fake regression
    A_screened = state["A"] @ jnp.diag(state["switch"][:, 0])
    state["beta"] = jnp.linalg.solve(A_screened.T @ A_screened + 0.01 * jnp.eye(num_knots), A_screened.T @ state["y"])
t.toc()

t = TicToc()
t.tic()
for it in tqdm(range(num_iterations)):
    # fake reversible jump
    if np.random.uniform() < 0.5:
        # perform birth move
        proposed_state = birth_move_jit(state)
    else:
        # perform death move
        proposed_state = death_move_jit(state)
    # evaluate likelihood
    proposed_likelihood = evaluate_likelihood_jit(proposed_state)
    if proposed_likelihood > likelihood:
        # accept proposed state
        state = proposed_state
        likelihood = proposed_likelihood

    # fake regression
    A_screened = state["A"] @ jnp.diag(state["switch"][:, 0])
    state["beta"] = jnp.linalg.solve(A_screened.T @ A_screened + 0.01 * jnp.eye(num_knots), A_screened.T @ state["y"])
t.toc()

def run_mcmc(state: dict, n_iter: int = num_iterations):
    likelihood = evaluate_likelihood(state)
    for it in tqdm(range(n_iter)):
        # fake reversible jump
        if np.random.uniform() < 0.5:
            # perform birth move
            proposed_state = birth_move(state)
        else:
            # perform death move
            proposed_state = death_move(state)
        # evaluate likelihood
        proposed_likelihood = evaluate_likelihood(proposed_state)
        def true_fun(state, proposed_state, likelihood, proposed_likelihood):
            return proposed_state, proposed_likelihood
        def false_fun(state, proposed_state, likelihood, proposed_likelihood):
            return state, likelihood
        state, likelihood = lax.cond(
            proposed_likelihood > likelihood, true_fun, false_fun, state, proposed_state, likelihood, proposed_likelihood
        )

        # fake regression
        A_screened = state["A"] @ jnp.diag(state["switch"][:, 0])
        state["beta"] = jnp.linalg.solve(A_screened.T @ A_screened + 0.01 * jnp.eye(num_knots), A_screened.T @ state["y"])

# jit_run_mcmc = jit(run_mcmc, static_argnums=1)

# t = TicToc()
# t.tic()
# jit_run_mcmc(state)
# t.toc()

"""
Notes (15/04/25): this last one works, but is VERY slow. Think this is because JAX had to trace our all of the possible
outcomes of the accept/reject case statement, so compile takes ages.
"""